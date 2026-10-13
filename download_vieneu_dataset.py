# -*- coding: utf-8 -*-
"""
Script tai dataset VieNeu-TTS-140h tren GPU Server de finetune Kokoro
- Ho tro tai Full dataset hoac so luong tuy chon
- Tu dong phan chia Train 80% / Test (Val) 20%
- Ho tro Resume (bo qua neu file da ton tai, khong tai lai)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, argparse, random
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Tai dataset VieNeu-TTS-140h va chia tap Train/Val")
    parser.add_argument("--full", action="store_true", help="Tai toan bo dataset (~74.858 mau, ~140 gio)")
    parser.add_argument("--max_samples", type=int, default=None, help="Gioi han so mau can tai neu khong tai full (VD: 5000)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ti le tap test/validation (mac dinh: 0.2 = 20%)")
    parser.add_argument("--speaker", type=str, default=None, help="Loc theo speaker cu the neu muon single-speaker")
    parser.add_argument("--output_dir", type=str, default="finetune_data", help="Thu muc luu data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed chia tap train/val")
    args = parser.parse_args()

    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    
    train_csv = os.path.join(args.output_dir, "train_metadata.csv")
    val_csv   = os.path.join(args.output_dir, "val_metadata.csv")
    all_csv   = os.path.join(args.output_dir, "metadata.csv")

    print("=" * 65)
    print("  TAI DATASET VIENEU-TTS-140H & CHIA TAP TRAIN (80%) / TEST (20%)")
    print("=" * 65)
    print(f"  Che do tai     : {'FULL DATASET (~74.800 cau)' if args.full else f'Gioi han {args.max_samples} cau'}")
    print(f"  Ti le chia     : Train {(1 - args.val_ratio)*100:.0f}% / Val {args.val_ratio*100:.0f}%")
    print(f"  Thu muc luu    : {args.output_dir}")
    print("  Dang ket noi toi Hugging Face Hub...")

    # Load dataset streaming
    ds = load_dataset("pnnbao-ump/VieNeu-TTS-140h", split="train", streaming=True)

    saved_items = []
    skipped_existing = 0
    total_target = None if args.full else args.max_samples

    pbar = tqdm(total=total_target, desc="Downloading Samples")

    for item in ds:
        if total_target and len(saved_items) >= total_target:
            break

        spk = item.get("speaker", "")
        if args.speaker and spk != args.speaker:
            continue

        uid = item.get("_id", f"sample_{len(saved_items)}")
        text = item.get("text", "").strip()
        phoneme = item.get("phonemized_text", "").strip()
        audio_info = item.get("audio", {})
        audio_bytes = audio_info.get("bytes")

        if not text:
            continue

        wav_path = os.path.join(wav_dir, f"{uid}.wav")

        # Resume support: neu file da co san thi khong ghi de
        if not os.path.exists(wav_path):
            if not audio_bytes:
                continue
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
        else:
            skipped_existing += 1

        saved_items.append({"id": uid, "text": text, "phoneme": phoneme, "speaker": spk})
        pbar.update(1)

    pbar.close()

    total_count = len(saved_items)
    print(f"\n[OK] Da thu thap tong cong: {total_count} mau audio.")
    if skipped_existing > 0:
        print(f"     (Trong do da co san {skipped_existing} file, khong can tai lai)")

    if total_count == 0:
        print("[!] Khong co mau nao duoc tai.")
        return

    # Shuffle va chia tap Train 80% / Test 20%
    random.seed(args.seed)
    random.shuffle(saved_items)

    val_size = int(total_count * args.val_ratio)
    train_size = total_count - val_size

    train_data = saved_items[:train_size]
    val_data = saved_items[train_size:]

    # Ghi file train_metadata.csv
    with open(train_csv, "w", encoding="utf-8") as f:
        for x in train_data:
            f.write(f"{x['id']}|{x['text']}\n")

    # Ghi file val_metadata.csv
    with open(val_csv, "w", encoding="utf-8") as f:
        for x in val_data:
            f.write(f"{x['id']}|{x['text']}\n")

    # Ghi file metadata.csv (toan bo)
    with open(all_csv, "w", encoding="utf-8") as f:
        for x in saved_items:
            f.write(f"{x['id']}|{x['text']}\n")

    print("\n" + "=" * 65)
    print("  KET QUA PHAN CHIA DATASET")
    print("=" * 65)
    print(f"  Tong so mau  : {total_count:,}")
    print(f"  Tap TRAIN    : {train_size:,} cau ({train_size/total_count*100:.1f}%) -> {train_csv}")
    print(f"  Tap VAL/TEST : {val_size:,} cau ({val_size/total_count*100:.1f}%) -> {val_csv}")
    print(f"  Folder WAV   : {wav_dir}")
    print("=" * 65)
    print("[SAN SANG] Du lieu da san sang de huan luyen LoRA voi tap danh gia!")

if __name__ == "__main__":
    main()
