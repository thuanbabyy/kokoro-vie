# -*- coding: utf-8 -*-
"""Synthesize with the Vietnamese Kokoro Full-parameter checkpoint."""
import argparse
import csv
import os

import torch
import soundfile as sf

from lora_trainer import CFG


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
    # Trỏ đến file config và trọng số full đã train
    parser.add_argument("--config", default=os.path.join(CFG["output_dir"], "kokoro_vi_full_config.json"))
    parser.add_argument("--weights", default=os.path.join(CFG["output_dir"], "kokoro_vi_full.pth"))
    parser.add_argument("--speakers-file", default=os.path.join(CFG["output_dir"], "kokoro_vi_full_speakers.pt"))
    parser.add_argument("--output", default=os.path.join(CFG["output_dir"], "sample_vi.wav"))
    args = parser.parse_args()

    if args.sample_id:
        sample = find_sample(args.sample_id)
        args.speaker = sample["speaker"]
        args.phonemes = sample["phoneme"]
    if not args.speaker or not args.phonemes:
        parser.error("Provide --speaker and --phonemes, or use --sample-id")

    # Load thông tin speaker từ file full_speakers.pt
    speaker_payload = torch.load(args.speakers_file, map_location="cpu", weights_only=False)
    speaker_to_id = speaker_payload["speaker_to_id"]
    if args.speaker not in speaker_to_id:
        raise ValueError(f"Unknown speaker ID: {args.speaker}")

    from kokoro import KModel, KPipeline
    
    # Khởi tạo mô hình trực tiếp từ file weights full và config mở rộng
    model = KModel(config=args.config, model=args.weights)
    model = model.to(CFG["device"]).eval()

    speaker_index = speaker_to_id[args.speaker]
    style = speaker_payload["base_voice"] + speaker_payload["speaker_deltas"][speaker_index]
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