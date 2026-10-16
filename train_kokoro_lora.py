# -*- coding: utf-8 -*-
"""Experimental Kokoro LoRA training with CTC-derived word timings."""
import argparse
import csv
import json
import math
import os
import unicodedata
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lora_trainer import CFG, LoRALinear

KOKORO_FRAME_RATE = 40


def load_alignment_index(path):
    index = {}
    with open(path, encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            index[record["id"]] = record
    return index


def distribute_word_frames(phone_word, frame_count):
    """Approximate frame allocation; stress/tone/combining marks get less time."""
    weights = []
    for char in phone_word:
        category = unicodedata.category(char)
        if category.startswith("M") or char.isdigit() or char in "\u02c8\u02cc\u02d0":
            weights.append(0.15)
        else:
            weights.append(1.0)
    if not weights:
        return []

    minimum_total = sum(weight >= 1.0 for weight in weights)
    frame_count = max(frame_count, minimum_total)
    exact = [frame_count * weight / sum(weights) for weight in weights]
    allocated = [int(value) for value in exact]
    remainder = frame_count - sum(allocated)
    order = sorted(range(len(exact)), key=lambda i: exact[i] - allocated[i], reverse=True)
    for index in order[:remainder]:
        allocated[index] += 1
    return allocated


class AlignedVieNeuDataset(Dataset):
    def __init__(self, metadata_path, alignment_path, wav_dir, speaker_to_id):
        import soundfile as sf

        self.wav_dir = wav_dir
        self.speaker_to_id = speaker_to_id
        self.items = []
        alignments = load_alignment_index(alignment_path)
        with open(metadata_path, encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter="|"):
                alignment = alignments.get(row["id"])
                if not alignment or not row.get("phoneme") or not row.get("speaker"):
                    continue
                phone_words = row["phoneme"].split()
                timings = alignment.get("phoneme_timings", [])
                if len(phone_words) != len(timings):
                    continue

                token_ids = [0]
                durations = [1]
                valid = True
                for word_index, phone_word in enumerate(phone_words):
                    word_frames = timings[word_index]["duration_frames"]
                    phone_durations = distribute_word_frames(phone_word, word_frames)
                    for char, duration in zip(phone_word, phone_durations):
                        token_id = CFG.get("active_vocab", {}).get(char)
                        if token_id is None:
                            valid = False
                            break
                        token_ids.append(token_id)
                        durations.append(duration)
                    if not valid:
                        break

                    if word_index + 1 < len(phone_words):
                        gap = max(0.0, timings[word_index + 1]["start"] - timings[word_index]["end"])
                        token_id = CFG.get("active_vocab", {}).get(" ")
                        if token_id is None:
                            valid = False
                            break
                        token_ids.append(token_id)
                        durations.append(max(0, round(gap * KOKORO_FRAME_RATE)))

                if not valid:
                    continue
                token_ids.append(0)
                durations.append(1)
                wav_path = os.path.join(wav_dir, f"{row['id']}.wav")
                if not os.path.isfile(wav_path):
                    continue

                audio_info = sf.info(wav_path)
                audio_duration = audio_info.frames / audio_info.samplerate
                start = max(0.0, timings[0]["start"])
                end = min(audio_duration, timings[-1]["end"])
                self.items.append({
                    "id": row["id"],
                    "wav": wav_path,
                    "speaker_id": speaker_to_id[row["speaker"]],
                    "token_ids": token_ids,
                    "durations": durations,
                    "crop_start": start,
                    "crop_end": end,
                    "sample_rate": audio_info.samplerate,
                    "audio_frames": audio_info.frames,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import soundfile as sf
        from scipy.signal import resample_poly

        item = self.items[index]
        start = max(0, int(item["crop_start"] * item["sample_rate"]))
        stop = min(item["audio_frames"], int(item["crop_end"] * item["sample_rate"]))
        audio, sample_rate = sf.read(
            item["wav"], start=start, stop=stop, dtype="float32", always_2d=True
        )
        audio = audio.mean(axis=1)
        if sample_rate != 24000:
            divisor = math.gcd(sample_rate, 24000)
            audio = resample_poly(audio, 24000 // divisor, sample_rate // divisor)
            audio = audio.astype("float32", copy=False)
        return {
            "id": item["id"],
            "audio": torch.from_numpy(audio),
            "speaker_id": item["speaker_id"],
            "token_ids": item["token_ids"],
            "durations": item["durations"],
        }


def collate_aligned(batch):
    batch_size = len(batch)
    max_tokens = max(len(item["token_ids"]) for item in batch)
    max_frames = max(sum(item["durations"]) for item in batch)
    max_audio = max(item["audio"].numel() for item in batch)
    token_ids = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    token_lengths = torch.zeros(batch_size, dtype=torch.long)
    duration_targets = torch.zeros(batch_size, max_tokens)
    alignment = torch.zeros(batch_size, max_tokens, max_frames)
    audio = torch.zeros(batch_size, max_audio)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    frame_lengths = torch.zeros(batch_size, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)

    for row, item in enumerate(batch):
        ids = item["token_ids"]
        durations = torch.as_tensor(item["durations"], dtype=torch.long)
        token_ids[row, :len(ids)] = torch.tensor(ids)
        token_lengths[row] = len(ids)
        duration_targets[row, :len(ids)] = durations
        frame_count = int(durations.sum().item())
        frame_lengths[row] = frame_count
        if frame_count:
            token_positions = torch.repeat_interleave(torch.arange(len(ids)), durations)
            alignment[row, token_positions, torch.arange(frame_count)] = 1.0
        sample = item["audio"]
        audio[row, :sample.numel()] = sample
        audio_lengths[row] = sample.numel()
        speaker_ids[row] = item["speaker_id"]

    return {
        "token_ids": token_ids,
        "token_lengths": token_lengths,
        "duration_targets": duration_targets,
        "alignment": alignment,
        "audio": audio,
        "audio_lengths": audio_lengths,
        "frame_lengths": frame_lengths,
        "speaker_ids": speaker_ids,
    }


def expand_model_vocab(model, symbols):
    vocab = dict(model.vocab)
    next_id = model.bert.config.vocab_size
    additions = {}
    for symbol in sorted(set(symbols) - set(vocab)):
        additions[symbol] = next_id
        vocab[symbol] = next_id
        next_id += 1

    if not additions:
        model.vocab = vocab
        return {}, model.bert.config.vocab_size

    old_size = model.bert.config.vocab_size
    embedding_paths = [model.bert.embeddings.word_embeddings, model.text_encoder.embedding]
    for old_embedding in embedding_paths:
        new_embedding = nn.Embedding(
            next_id,
            old_embedding.embedding_dim,
            padding_idx=old_embedding.padding_idx,
            device=old_embedding.weight.device,
            dtype=old_embedding.weight.dtype,
        )
        with torch.no_grad():
            new_embedding.weight[:old_size].copy_(old_embedding.weight)
            nn.init.normal_(new_embedding.weight[old_size:], mean=0.0, std=0.02)
        old_embedding_parent = model.bert.embeddings if old_embedding is embedding_paths[0] else model.text_encoder
        if old_embedding_parent is model.bert.embeddings:
            old_embedding_parent.word_embeddings = new_embedding
        else:
            old_embedding_parent.embedding = new_embedding

    model.bert.config.vocab_size = next_id
    model.vocab = vocab
    return additions, old_size


def inject_bert_lora(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = {"query", "key", "value", "dense"}
    injected = 0
    for name, module in list(model.bert.named_modules()):
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            continue
        parent = model.bert
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, CFG["lora_rank"], CFG["lora_alpha"], CFG["lora_dropout"]))
        injected += 1
    if not injected:
        raise RuntimeError("No PL-BERT Linear layers matched the LoRA target names")
    print(f"  [LoRA] Injected into {injected} PL-BERT layers")


def load_experimental_adapter(model, payload):
    additions, base_vocab_size = expand_model_vocab(model, payload["added_tokens"].keys())
    if additions != payload["added_tokens"]:
        raise ValueError("Adapter token IDs do not match this base Kokoro vocabulary")
    inject_bert_lora(model)
    with torch.no_grad():
        model.bert.embeddings.word_embeddings.weight[base_vocab_size:].copy_(
            payload["bert_embedding_rows"]
        )
        model.text_encoder.embedding.weight[base_vocab_size:].copy_(
            payload["text_embedding_rows"]
        )
    missing, unexpected = model.load_state_dict(payload["lora"], strict=False)
    if unexpected or any("lora_" in key for key in missing):
        raise RuntimeError(f"Adapter load mismatch: missing={missing}, unexpected={unexpected}")
    return model


def train_forward(model, token_ids, token_lengths, alignment, styles):
    batch_size, max_tokens = token_ids.shape
    positions = torch.arange(max_tokens, device=token_ids.device).expand(batch_size, -1)
    text_mask = positions >= token_lengths.unsqueeze(1)
    bert_states = model.bert(token_ids, attention_mask=(~text_mask).int())
    encoded = model.bert_encoder(bert_states).transpose(-1, -2)
    style_prosody = styles[:, 128:]
    duration_logits, encoded = model.predictor(
        encoded, style_prosody, token_lengths, alignment, text_mask
    )
    pitch, noise = model.predictor.F0Ntrain(encoded, style_prosody)
    text_features = model.text_encoder(token_ids, token_lengths, text_mask)
    aligned_features = text_features @ alignment
    generated = model.decoder(aligned_features, pitch, noise, styles[:, :128])
    if generated.ndim == 3:
        generated = generated.squeeze(1)
    duration_prediction = torch.sigmoid(duration_logits).sum(-1)
    return generated, duration_prediction


def main():
    parser = argparse.ArgumentParser(description="Experimental Kokoro Vietnamese LoRA training")
    parser.add_argument("--train", action="store_true", help="Run the experimental trainer")
    parser.add_argument("--train-alignment", default="finetune_data/train_alignments.jsonl")
    parser.add_argument("--val-alignment", default="finetune_data/val_alignments.jsonl")
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Limit training rows for the first pilot; 0 uses all rows")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="CPU workers for loading audio; use 0 to disable multiprocessing")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto",
                        help="Use BF16 autocast when supported; otherwise auto uses FP32")
    parser.add_argument("--save-every-steps", type=int, default=250,
                        help="Save a resumable checkpoint every N optimizer steps")
    parser.add_argument("--resume", help="Resume from kokoro_vi_experimental_last.pt")
    parser.add_argument("--init-adapter", help="Warm-start model/LoRA from a completed adapter, with a fresh optimizer")
    args = parser.parse_args()
    if not args.train:
        raise SystemExit("Pass --train to start the experimental loop; generate CTC alignments first.")
    if args.resume and args.init_adapter:
        raise ValueError("Use either --resume or --init-adapter, not both")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    use_bf16 = CFG["device"] == "cuda" and args.precision != "fp32"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use --precision fp32 or auto")
    use_bf16 = use_bf16 and torch.cuda.is_bf16_supported()
    print(f"Precision: {'bf16' if use_bf16 else 'fp32'}")
    if args.save_every_steps < 1:
        raise ValueError("--save-every-steps must be >= 1")

    for path in (args.train_alignment, args.val_alignment):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Alignment file not found: {path}. Run align_vieneu.py successfully first."
            )
        if os.path.getsize(path) == 0:
            raise RuntimeError(f"Alignment file is empty: {path}. Alignment must succeed before training.")

    from kokoro import KModel, KPipeline
    config_path = os.path.join(CFG["model_dir"], "config.json")
    checkpoint_path = os.path.join(CFG["model_dir"], "kokoro-v1_0.pth")
    with open(config_path, encoding="utf-8") as source:
        config = json.load(source)

    metadata_paths = [CFG["train_meta"], CFG["val_meta"]]
    speakers = set()
    symbols = set()
    for path in metadata_paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter="|"):
                speakers.add(row["speaker"])
                symbols.update(row["phoneme"])
    speaker_to_id = {speaker: index for index, speaker in enumerate(sorted(speakers))}
    model = KModel(config=config_path, model=checkpoint_path)
    resume_payload = None
    init_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_payload["speaker_to_id"] != speaker_to_id:
            raise ValueError("Speaker map changed since checkpoint; refusing an unsafe resume")
        expected_args = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
        }
        if resume_payload.get("training_args") != expected_args:
            raise ValueError(
                "Resume requires the original --epochs, --batch-size, and --max-samples values: "
                f"checkpoint={resume_payload.get('training_args')}, current={expected_args}"
            )
        model = load_experimental_adapter(model, resume_payload)
        additions = resume_payload["added_tokens"]
        base_vocab_size = resume_payload["base_vocab_size"]
    else:
        if args.init_adapter:
            init_payload = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
            if init_payload["speaker_to_id"] != speaker_to_id:
                raise ValueError("Speaker map changed since adapter; refusing an unsafe warm-start")
            model = load_experimental_adapter(model, init_payload)
            additions = init_payload["added_tokens"]
            base_vocab_size = init_payload["base_vocab_size"]
        else:
            additions, base_vocab_size = expand_model_vocab(model, symbols)
            inject_bert_lora(model)
    CFG["active_vocab"] = model.vocab

    pipeline = KPipeline(lang_code="a", model=False)
    base_voice = pipeline.load_voice("af_heart").reshape(-1, 256).mean(0)
    speaker_deltas = nn.Embedding(len(speaker_to_id), 256).to(CFG["device"])
    nn.init.zeros_(speaker_deltas.weight)
    starting_payload = resume_payload or init_payload
    if starting_payload is not None:
        speaker_deltas.weight.data.copy_(starting_payload["speaker_deltas"])
        base_voice = starting_payload["base_voice"]
    base_voice = base_voice.to(CFG["device"])
    model = model.to(CFG["device"])
    for embedding in (model.bert.embeddings.word_embeddings, model.text_encoder.embedding):
        embedding.weight.requires_grad_(True)
        row_mask = torch.zeros_like(embedding.weight)
        row_mask[base_vocab_size:] = 1
        embedding.weight.register_hook(lambda grad, mask=row_mask: grad * mask)

    train_dataset = AlignedVieNeuDataset(
        CFG["train_meta"], args.train_alignment, CFG["wav_dir"], speaker_to_id
    )
    if args.max_samples:
        train_dataset.items = train_dataset.items[:args.max_samples]
    val_dataset = AlignedVieNeuDataset(
        CFG["val_meta"], args.val_alignment, CFG["wav_dir"], speaker_to_id
    )
    if not len(train_dataset):
        raise RuntimeError("No aligned training samples remain after validation")
    if val_dataset is not None and args.max_samples:
        val_dataset.items = val_dataset.items[:max(1, args.max_samples // 5)]

    pin_memory = CFG["device"] == "cuda"
    loader_options = {
        "batch_size": args.batch_size,
        "collate_fn": collate_aligned,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)

    val_loader = DataLoader(
        val_dataset, shuffle=False, **loader_options
    ) if val_dataset is not None and len(val_dataset) else None

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters += list(speaker_deltas.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=CFG["lr"], weight_decay=0.0)
    import torchaudio.transforms as T
    mel_transform = T.MelSpectrogram(
        sample_rate=24000, n_fft=1024, hop_length=300, n_mels=80, f_min=0, f_max=8000
    ).to(CFG["device"])

    def batch_loss(batch):
        token_ids = batch["token_ids"].to(CFG["device"], non_blocking=pin_memory)
        token_lengths = batch["token_lengths"].to(CFG["device"], non_blocking=pin_memory)
        alignment = batch["alignment"].to(CFG["device"], non_blocking=pin_memory)
        target_durations = batch["duration_targets"].to(CFG["device"], non_blocking=pin_memory)
        speaker_ids = batch["speaker_ids"].to(CFG["device"], non_blocking=pin_memory)
        styles = base_voice.unsqueeze(0) + speaker_deltas(speaker_ids)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
        ):
            prediction, predicted_durations = train_forward(
                model, token_ids, token_lengths, alignment, styles
            )
        prediction = prediction.float()
        predicted_durations = predicted_durations.float()
        valid = torch.arange(token_ids.shape[1], device=CFG["device"])[None, :] < token_lengths[:, None]
        duration_loss = F.smooth_l1_loss(
            predicted_durations[valid], target_durations[valid]
        )
        mel_losses = []
        for index, target_length in enumerate(batch["audio_lengths"].tolist()):
            target = batch["audio"][index, :target_length].to(
                CFG["device"], non_blocking=pin_memory
            )
            predicted = prediction[index].reshape(-1)
            expected_length = int(batch["frame_lengths"][index].item()) * (24000 // KOKORO_FRAME_RATE)
            target_coverage = min(target.numel(), expected_length) / max(target.numel(), expected_length)
            if target_coverage < 0.8:
                raise RuntimeError(
                    f"Target/alignment length mismatch ({target_coverage:.1%} overlap): "
                    f"audio={target.numel()} samples, duration={expected_length} samples"
                )
            predicted_coverage = min(predicted.numel(), expected_length) / expected_length
            if predicted_coverage < 0.8:
                raise RuntimeError(
                    f"Generated audio is shorter than this sample's aligned duration "
                    f"({predicted_coverage:.1%} coverage): "
                    f"generated={predicted.numel()} samples, duration={expected_length} samples"
                )
            common = min(target.numel(), predicted.numel(), expected_length)
            if common < 1024:
                continue
            target_mel = torch.log(mel_transform(target[:common]).clamp_min(1e-5))
            predicted_mel = torch.log(mel_transform(predicted[:common]).clamp_min(1e-5))
            mel_frames = min(target_mel.shape[-1], predicted_mel.shape[-1])
            mel_losses.append(F.l1_loss(
                predicted_mel[..., :mel_frames], target_mel[..., :mel_frames]
            ))
        if not mel_losses:
            raise RuntimeError("Generated audio was too short for the mel loss")
        mel_loss = torch.stack(mel_losses).mean()
        return mel_loss + 0.1 * duration_loss, mel_loss.detach(), duration_loss.detach()

    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG["lr"], total_steps=total_steps, pct_start=0.1
    )
    resume_step = 0
    resume_order = None
    resume_loss_sum = 0.0
    resume_loss_count = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        saved_step = resume_payload.get("step_in_epoch")
        if saved_step is not None and saved_step < steps_per_epoch:
            start_epoch = resume_payload["epoch"]
            resume_step = saved_step
            resume_order = resume_payload.get("epoch_order")
            if resume_order is None or len(resume_order) != len(train_dataset):
                raise ValueError("Checkpoint lacks the sample order required for mid-epoch resume")
            resume_loss_sum = resume_payload.get("epoch_loss_sum", 0.0)
            resume_loss_count = resume_payload.get("epoch_loss_count", 0)
        else:
            start_epoch = max(1, resume_payload["epoch"] + 1)
        print(
            f"Resuming at epoch {start_epoch}/{args.epochs}, "
            f"step {resume_step}/{steps_per_epoch}"
        )
    else:
        start_epoch = 1

    os.makedirs(CFG["output_dir"], exist_ok=True)
    last_checkpoint_path = os.path.join(CFG["output_dir"], "kokoro_vi_experimental_last.pt")

    def make_checkpoint(epoch, step_in_epoch, epoch_order, loss_sum, loss_count):
        lora_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if "lora_A" in key or "lora_B" in key
        }
        return {
            "lora": lora_state,
            "added_tokens": additions,
            "bert_embedding_rows": model.bert.embeddings.word_embeddings.weight[base_vocab_size:].detach().cpu(),
            "text_embedding_rows": model.text_encoder.embedding.weight[base_vocab_size:].detach().cpu(),
            "base_vocab_size": base_vocab_size,
            "speaker_to_id": speaker_to_id,
            "speaker_deltas": speaker_deltas.weight.detach().cpu(),
            "base_voice": base_voice.detach().cpu(),
            "alignment": "CTC word timing with estimated per-phoneme durations",
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "epoch_order": epoch_order,
            "epoch_loss_sum": loss_sum,
            "epoch_loss_count": loss_count,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_args": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
            },
        }

    def save_checkpoint(path, payload):
        temporary_path = path + ".tmp"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)

    if resume_payload is None:
        generator = torch.Generator().manual_seed(42 + start_epoch)
        resume_order = torch.randperm(len(train_dataset), generator=generator).tolist()
        save_checkpoint(
            last_checkpoint_path,
            make_checkpoint(start_epoch, 0, resume_order, 0.0, 0),
        )
        print(f"Saved initial checkpoint: epoch={start_epoch} step=0")

    restore_rng = resume_payload is not None
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == start_epoch and resume_order is not None:
            epoch_order = resume_order
            first_step = resume_step
            running_sum = resume_loss_sum
            running_count = resume_loss_count
        else:
            generator = torch.Generator().manual_seed(42 + epoch)
            epoch_order = torch.randperm(len(train_dataset), generator=generator).tolist()
            first_step = 0
            running_sum = 0.0
            running_count = 0

        remaining_indices = epoch_order[first_step * args.batch_size:]
        epoch_dataset = torch.utils.data.Subset(train_dataset, remaining_indices)
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
                lora_grad = sum(
                    parameter.grad.norm().item()
                    for name, parameter in model.named_parameters()
                    if ("lora_A" in name or "lora_B" in name) and parameter.grad is not None
                )
                speaker_grad = (
                    speaker_deltas.weight.grad.norm().item()
                    if speaker_deltas.weight.grad is not None else 0.0
                )
                if not torch.isfinite(loss) or lora_grad <= 0 or speaker_grad <= 0:
                    raise RuntimeError(
                        f"Gradient check failed: loss={loss.item()}, "
                        f"lora_grad={lora_grad}, speaker_grad={speaker_grad}"
                    )
                print(
                    f"[Gradient check] loss={loss.item():.4f} "
                    f"lora_grad={lora_grad:.3e} speaker_grad={speaker_grad:.3e}"
                )
            torch.nn.utils.clip_grad_norm_(parameters, CFG["grad_clip"])
            optimizer.step()
            scheduler.step()
            running_sum += loss.item()
            running_count += 1
            if step % 10 == 0:
                print(
                    f"epoch={epoch} step={step}/{steps_per_epoch} "
                    f"loss={loss.item():.4f} mel={mel_loss.item():.4f} "
                    f"duration={duration_loss.item():.4f}"
                )
            if step % args.save_every_steps == 0:
                save_checkpoint(
                    last_checkpoint_path,
                    make_checkpoint(epoch, step, epoch_order, running_sum, running_count),
                )
                print(f"Saved mid-epoch checkpoint: epoch={epoch} step={step}")

        validation = None
        if val_loader is not None:
            model.eval()
            speaker_deltas.eval()
            values = []
            with torch.no_grad():
                for batch in val_loader:
                    loss, _, _ = batch_loss(batch)
                    values.append(loss.item())
            validation = sum(values) / max(1, len(values))
        print(
            f"epoch={epoch} train_loss={running_sum / max(1, running_count):.4f} "
            f"val_loss={validation if validation is not None else 'n/a'}"
        )
        save_checkpoint(
            last_checkpoint_path,
            make_checkpoint(epoch, steps_per_epoch, epoch_order, running_sum, running_count),
        )
        print(f"Saved resumable checkpoint: {last_checkpoint_path}")
        resume_order = None
        resume_step = 0
        resume_loss_sum = 0.0
        resume_loss_count = 0

    adapter_path = os.path.join(CFG["output_dir"], "kokoro_vi_experimental.pt")
    if start_epoch <= args.epochs:
        final_payload = make_checkpoint(
            args.epochs, steps_per_epoch, epoch_order, running_sum, running_count
        )
    else:
        final_payload = resume_payload
    save_checkpoint(adapter_path, final_payload)
    print(f"Saved experimental adapter: {adapter_path}")
    print("This checkpoint requires a matching custom loader; it is not merge_lora.py compatible.")


if __name__ == "__main__":
    main()