# -*- coding: utf-8 -*-
"""Experimental full-parameter Vietnamese fine-tuning for Kokoro."""
import argparse
import csv
import json
import math
import os

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lora_trainer import CFG
from train_kokoro_lora import (
    AlignedVieNeuDataset,
    collate_aligned,
    expand_model_vocab,
    train_forward,
)

KOKORO_FRAME_RATE = 40
MODEL_COMPONENTS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def merge_lora_warm_start(model, adapter_payload):
    """Optional one-time conversion from the existing pilot LoRA to full weights."""
    from lora_trainer import LoRALinear
    from train_kokoro_lora import load_experimental_adapter

    model = load_experimental_adapter(model, adapter_payload)
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.merge())
    return model


def main():
    parser = argparse.ArgumentParser(description="Full-parameter Kokoro Vietnamese fine-tuning")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-alignment", default="finetune_data/train_alignments.jsonl")
    parser.add_argument("--val-alignment", default="finetune_data/val_alignments.jsonl")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 uses all aligned samples")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for full-parameter fine-tuning")
    parser.add_argument("--save-every-steps", type=int, default=5000,
                        help="Full checkpoints are large; 0 saves only at epoch boundaries")
    parser.add_argument("--resume", help="Resume full training from kokoro_vi_full_last.pt")
    parser.add_argument("--init-adapter", help="Warm-start once from the existing LoRA pilot")
    args = parser.parse_args()

    if not args.train:
        raise SystemExit("Pass --train to start full-parameter fine-tuning.")
    if args.resume and args.init_adapter:
        raise ValueError("Use either --resume or --init-adapter, not both")
    if args.num_workers < 0 or args.save_every_steps < 0:
        raise ValueError("num-workers and save-every-steps must be >= 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")

    use_bf16 = CFG["device"] == "cuda" and args.precision != "fp32"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --precision fp32 or auto")
    use_bf16 = use_bf16 and torch.cuda.is_bf16_supported()
    print(f"Device: {CFG['device']} | Precision: {'bf16' if use_bf16 else 'fp32'}")

    for path in (args.train_alignment, args.val_alignment):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(f"Missing/empty alignment file: {path}")

    from kokoro import KModel, KPipeline

    config_path = os.path.join(CFG["model_dir"], "config.json")
    base_checkpoint = os.path.join(CFG["model_dir"], "kokoro-v1_0.pth")
    with open(config_path, encoding="utf-8") as source:
        base_config = json.load(source)

    speakers = set()
    symbols = set()
    for metadata_path in (CFG["train_meta"], CFG["val_meta"]):
        with open(metadata_path, encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter="|"):
                speakers.add(row["speaker"])
                symbols.update(row["phoneme"])
    speaker_to_id = {speaker: index for index, speaker in enumerate(sorted(speakers))}

    model = KModel(config=config_path, model=base_checkpoint)
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_payload["speaker_to_id"] != speaker_to_id:
            raise ValueError("Speaker map changed since checkpoint")
        expected = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "lr": args.lr,
        }
        if resume_payload.get("training_args") != expected:
            raise ValueError(f"Resume requires unchanged training arguments: {expected}")
        additions, base_vocab_size = expand_model_vocab(model, resume_payload["added_tokens"].keys())
        if additions != resume_payload["added_tokens"]:
            raise ValueError("Checkpoint token IDs do not match base Kokoro vocabulary")
        model.vocab = resume_payload["vocab"]
        model.load_state_dict(resume_payload["model_state"], strict=True)
    else:
        if args.init_adapter:
            warm_payload = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
            if warm_payload["speaker_to_id"] != speaker_to_id:
                raise ValueError("Speaker map changed since the LoRA adapter")
            model = merge_lora_warm_start(model, warm_payload)
            additions = warm_payload["added_tokens"]
            base_vocab_size = warm_payload["base_vocab_size"]
        else:
            additions, base_vocab_size = expand_model_vocab(model, symbols)

    CFG["active_vocab"] = model.vocab
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    pipeline = KPipeline(lang_code="a", model=False)
    base_voice = pipeline.load_voice("af_heart").reshape(-1, 256).mean(0)
    speaker_deltas = nn.Embedding(len(speaker_to_id), 256)
    nn.init.zeros_(speaker_deltas.weight)
    if resume_payload is not None:
        speaker_deltas.weight.data.copy_(resume_payload["speaker_deltas"])
        base_voice = resume_payload["base_voice"]
    elif args.init_adapter:
        speaker_deltas.weight.data.copy_(warm_payload["speaker_deltas"])
        base_voice = warm_payload["base_voice"]

    model = model.to(CFG["device"])
    speaker_deltas = speaker_deltas.to(CFG["device"])
    base_voice = base_voice.to(CFG["device"])
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Full fine-tune parameters: {trainable_count:,}/{sum(p.numel() for p in model.parameters()):,}")

    train_dataset = AlignedVieNeuDataset(
        CFG["train_meta"], args.train_alignment, CFG["wav_dir"], speaker_to_id
    )
    val_dataset = AlignedVieNeuDataset(
        CFG["val_meta"], args.val_alignment, CFG["wav_dir"], speaker_to_id
    )
    if args.max_samples:
        train_dataset.items = train_dataset.items[:args.max_samples]
        val_dataset.items = val_dataset.items[:max(1, args.max_samples // 5)]
    if not train_dataset:
        raise RuntimeError("No aligned training samples")

    pin_memory = CFG["device"] == "cuda"
    loader_options = {
        "batch_size": args.batch_size,
        "collate_fn": collate_aligned,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options) if val_dataset else None

    parameters = list(model.parameters()) + list(speaker_deltas.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    import torchaudio.transforms as T
    mel_transform = T.MelSpectrogram(
        sample_rate=24000, n_fft=1024, hop_length=300, n_mels=80, f_min=0, f_max=8000
    ).to(CFG["device"])

    def batch_loss(batch):
        token_ids = batch["token_ids"].to(CFG["device"], non_blocking=pin_memory)
        token_lengths = batch["token_lengths"].to(CFG["device"], non_blocking=pin_memory)
        alignment = batch["alignment"].to(CFG["device"], non_blocking=pin_memory)
        duration_targets = batch["duration_targets"].to(CFG["device"], non_blocking=pin_memory)
        speaker_ids = batch["speaker_ids"].to(CFG["device"], non_blocking=pin_memory)
        styles = base_voice.unsqueeze(0) + speaker_deltas(speaker_ids)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            prediction, predicted_durations = train_forward(
                model, token_ids, token_lengths, alignment, styles
            )
        prediction = prediction.float()
        predicted_durations = predicted_durations.float()
        valid = torch.arange(token_ids.shape[1], device=CFG["device"])[None, :] < token_lengths[:, None]
        duration_loss = F.smooth_l1_loss(predicted_durations[valid], duration_targets[valid])
        mel_losses = []
        for index, audio_length in enumerate(batch["audio_lengths"].tolist()):
            target = batch["audio"][index, :audio_length].to(CFG["device"], non_blocking=pin_memory)
            predicted = prediction[index].reshape(-1)
            aligned_length = int(batch["frame_lengths"][index]) * (24000 // KOKORO_FRAME_RATE)
            if min(target.numel(), aligned_length) / max(target.numel(), aligned_length) < 0.8:
                raise RuntimeError("Target/audio duration differs from CTC alignment")
            if min(predicted.numel(), aligned_length) / aligned_length < 0.8:
                raise RuntimeError("Generated audio is shorter than this sample's aligned duration")
            common = min(target.numel(), predicted.numel(), aligned_length)
            if common < 1024:
                continue
            target_mel = torch.log(mel_transform(target[:common]).clamp_min(1e-5))
            prediction_mel = torch.log(mel_transform(predicted[:common]).clamp_min(1e-5))
            mel_frames = min(target_mel.shape[-1], prediction_mel.shape[-1])
            mel_losses.append(F.l1_loss(prediction_mel[..., :mel_frames], target_mel[..., :mel_frames]))
        if not mel_losses:
            raise RuntimeError("Generated audio was too short for mel loss")
        mel_loss = torch.stack(mel_losses).mean()
        return mel_loss + 0.1 * duration_loss, mel_loss.detach(), duration_loss.detach()

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, steps_per_epoch * args.epochs), pct_start=0.1
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        saved_step = resume_payload["step_in_epoch"]
        if saved_step < steps_per_epoch:
            start_epoch = resume_payload["epoch"]
            start_step = saved_step
            order = resume_payload["epoch_order"]
            running_sum = resume_payload["loss_sum"]
            running_count = resume_payload["loss_count"]
        else:
            start_epoch = resume_payload["epoch"] + 1
            start_step = 0
            order = None
            running_sum = 0.0
            running_count = 0
        history = resume_payload["history"]
        print(f"Resume full train epoch={start_epoch}, step={start_step}/{steps_per_epoch}")
    else:
        start_epoch, start_step, order, history = 1, 0, None, []
        running_sum, running_count = 0.0, 0

    output_dir = CFG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    last_path = os.path.join(output_dir, "kokoro_vi_full_last.pt")

    def checkpoint(epoch, step, sample_order, loss_sum, loss_count):
        return {
            "model_state": cpu_state_dict(model),
            "vocab": model.vocab,
            "added_tokens": additions,
            "base_vocab_size": base_vocab_size,
            "speaker_to_id": speaker_to_id,
            "speaker_deltas": speaker_deltas.weight.detach().cpu(),
            "base_voice": base_voice.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step_in_epoch": step,
            "epoch_order": sample_order,
            "loss_sum": loss_sum,
            "loss_count": loss_count,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_args": {"epochs": args.epochs, "batch_size": args.batch_size,
                              "max_samples": args.max_samples, "lr": args.lr},
        }

    def save_full_checkpoint(path, payload):
        temp_path = path + ".tmp"
        torch.save(payload, temp_path)
        os.replace(temp_path, path)

    if resume_payload is None:
        generator = torch.Generator().manual_seed(1000 + start_epoch)
        order = torch.randperm(len(train_dataset), generator=generator).tolist()
        save_full_checkpoint(last_path, checkpoint(start_epoch, 0, order, 0.0, 0))

    restore_rng = resume_payload is not None
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == start_epoch and order is not None:
            sample_order = order
            first_step = start_step
            if first_step == 0:
                running_sum, running_count = 0.0, 0
        else:
            generator = torch.Generator().manual_seed(1000 + epoch)
            sample_order = torch.randperm(len(train_dataset), generator=generator).tolist()
            first_step, running_sum, running_count = 0, 0.0, 0
        epoch_dataset = Subset(train_dataset, sample_order[first_step * args.batch_size:])
        train_loader = DataLoader(epoch_dataset, shuffle=False, **loader_options)
        model.train()
        speaker_deltas.train()
        if restore_rng:
            if resume_payload.get("torch_rng_state") is not None:
                torch.set_rng_state(resume_payload["torch_rng_state"])
            if torch.cuda.is_available() and resume_payload.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
            restore_rng = False
        for step, batch in enumerate(train_loader, start=first_step + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, mel_loss, duration_loss = batch_loss(batch)
            loss.backward()
            if step == first_step + 1:
                grad_norm = sum(
                    p.grad.norm().item() for p in model.parameters() if p.grad is not None
                )
                if not torch.isfinite(loss) or grad_norm <= 0:
                    raise RuntimeError(f"Full-model gradient check failed: loss={loss.item()}, grad={grad_norm}")
                print(f"[Full gradient check] loss={loss.item():.4f} grad={grad_norm:.3e}")
            torch.nn.utils.clip_grad_norm_(parameters, CFG["grad_clip"])
            optimizer.step()
            scheduler.step()
            running_sum += loss.item()
            running_count += 1
            if step % 10 == 0:
                print(f"epoch={epoch} step={step}/{steps_per_epoch} loss={loss.item():.4f} "
                      f"mel={mel_loss.item():.4f} duration={duration_loss.item():.4f}")
            if args.save_every_steps and step % args.save_every_steps == 0:
                save_full_checkpoint(last_path, checkpoint(epoch, step, sample_order, running_sum, running_count))
                print(f"Saved full checkpoint: epoch={epoch} step={step}")

        val_loss = None
        if val_loader:
            model.eval()
            speaker_deltas.eval()
            values = []
            with torch.no_grad():
                for batch in val_loader:
                    values.append(batch_loss(batch)[0].item())
            val_loss = sum(values) / max(1, len(values))
        train_loss = running_sum / max(1, running_count)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss}")
        save_full_checkpoint(last_path, checkpoint(epoch, steps_per_epoch, sample_order, running_sum, running_count))
        start_step, order = 0, None

    final_config = dict(base_config)
    final_config["vocab"] = model.vocab
    final_config["n_token"] = model.bert.config.vocab_size
    config_out = os.path.join(output_dir, "kokoro_vi_full_config.json")
    with open(config_out, "w", encoding="utf-8") as target:
        json.dump(final_config, target, ensure_ascii=False, indent=2)
    weights_out = os.path.join(output_dir, "kokoro_vi_full.pth")
    torch.save({name: cpu_state_dict(getattr(model, name)) for name in MODEL_COMPONENTS}, weights_out)
    speaker_out = os.path.join(output_dir, "kokoro_vi_full_speakers.pt")
    torch.save({"speaker_to_id": speaker_to_id, "speaker_deltas": speaker_deltas.weight.detach().cpu(),
                "base_voice": base_voice.detach().cpu()}, speaker_out)
    print(f"Saved full Kokoro model: {weights_out}")
    print(f"Saved expanded config: {config_out}")
    print(f"Saved speaker vectors: {speaker_out}")


if __name__ == "__main__":
    main()