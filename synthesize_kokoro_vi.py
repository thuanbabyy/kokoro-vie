# -*- coding: utf-8 -*-
"""Synthesize with the experimental Vietnamese Kokoro LoRA checkpoint."""
import argparse
import csv
import os

import torch
import soundfile as sf

from lora_trainer import CFG
from train_kokoro_lora import load_experimental_adapter


def find_sample(sample_id):
    for metadata_path in (CFG["train_meta"], CFG["val_meta"], CFG["metadata"]):
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source, delimiter="|"):
                if row.get("id") == sample_id:
                    return row
    raise ValueError(f"Sample ID not found in metadata: {sample_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", help="VieNeu speaker ID")
    parser.add_argument("--phonemes", help="Vietnamese phoneme string")
    parser.add_argument("--sample-id", help="Use phonemes and speaker from a dataset row")
    parser.add_argument("--adapter", default=os.path.join(CFG["output_dir"], "kokoro_vi_experimental.pt"))
    parser.add_argument("--output", default=os.path.join(CFG["output_dir"], "sample_vi.wav"))
    args = parser.parse_args()

    if args.sample_id:
        sample = find_sample(args.sample_id)
        args.speaker = sample["speaker"]
        args.phonemes = sample["phoneme"]
    if not args.speaker or not args.phonemes:
        parser.error("Provide --speaker and --phonemes, or use --sample-id")

    payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    speaker_to_id = payload["speaker_to_id"]
    if args.speaker not in speaker_to_id:
        raise ValueError(f"Unknown speaker ID: {args.speaker}")

    from kokoro import KModel, KPipeline
    config_path = os.path.join(CFG["model_dir"], "config.json")
    checkpoint_path = os.path.join(CFG["model_dir"], "kokoro-v1_0.pth")
    model = KModel(config=config_path, model=checkpoint_path)
    model = load_experimental_adapter(model, payload).to(CFG["device"]).eval()

    speaker_index = speaker_to_id[args.speaker]
    style = payload["base_voice"] + payload["speaker_deltas"][speaker_index]
    voice_pack = style.reshape(1, 1, 256).expand(512, 1, 256).contiguous()
    pipeline = KPipeline(lang_code="a", model=False)
    results = pipeline.generate_from_tokens(
        args.phonemes, voice=voice_pack, model=model
    )
    result = next(results, None)
    if result is None or result.audio is None:
        raise RuntimeError("Kokoro returned no audio for the supplied phonemes")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, result.audio.detach().cpu().numpy(), 24000)
    print(f"Saved {args.output} for speaker {args.speaker}")


if __name__ == "__main__":
    main()