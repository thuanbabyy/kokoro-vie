# -*- coding: utf-8 -*-
"""Create experimental word timings for VieNeu using a Vietnamese CTC model."""
import argparse
import csv
import json
import os
import re
import unicodedata

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


ALIGNER_ID = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
SAMPLE_RATE = 16000
KOKORO_FRAME_RATE = 40


def ctc_forced_spans(log_probs, labels, blank_id):
    """Return frame spans using torchaudio's optimized CTC alignment kernel."""
    if not labels:
        raise ValueError("Cannot align an empty transcript")

    import torchaudio.functional as audio_functional

    targets = torch.tensor(labels, dtype=torch.int32, device=log_probs.device).unsqueeze(0)
    aligned_tokens, _ = audio_functional.forced_align(
        log_probs.unsqueeze(0), targets, blank=blank_id
    )
    path = aligned_tokens[0].detach().cpu().tolist()
    spans = []
    active_token = None
    span_start = 0
    target_index = 0

    def close_span(end_frame):
        nonlocal active_token, target_index
        if active_token is None:
            return
        if target_index >= len(labels) or active_token != labels[target_index]:
            raise ValueError("CTC kernel path does not match the target sequence")
        spans.append((span_start, end_frame))
        target_index += 1
        active_token = None

    for frame, token in enumerate(path):
        if token == blank_id:
            close_span(frame)
        elif active_token is None:
            active_token = token
            span_start = frame
        elif token != active_token:
            close_span(frame)
            active_token = token
            span_start = frame
    close_span(len(path))

    if target_index != len(labels):
        raise ValueError(f"CTC path aligned {target_index}/{len(labels)} transcript tokens")
    return spans


def map_word_timings_to_phonemes(words, word_timings, phoneme_words):
    """Expand number words and distribute their CTC span across phone groups."""
    from num2words import num2words

    group_counts = [1] * len(words)
    number_indices = []
    number_values = {}
    for index, word in enumerate(words):
        number = word.strip(".,;:!?()[]{}\"'")
        if not re.fullmatch(r"\d+", number):
            continue
        try:
            spoken_words = num2words(int(number), lang="vi").split()
        except (NotImplementedError, OverflowError, ValueError) as error:
            raise ValueError(f"Cannot normalize numeric token {word!r} to Vietnamese") from error
        group_counts[index] = max(1, len(spoken_words))
        number_indices.append(index)
        number_values[index] = int(number)

    difference = len(phoneme_words) - sum(group_counts)
    if difference and not number_indices:
        raise ValueError(
            f"word/phoneme mismatch ({len(words)} words, {len(phoneme_words)} phoneme groups; no number to expand)"
        )
    if abs(difference) > max(1, len(number_indices)):
        raise ValueError(
            f"number expansion does not explain mismatch ({sum(group_counts)} predicted groups, "
            f"{len(phoneme_words)} actual groups)"
        )

    if difference > 0:
        index = max(number_indices, key=lambda item: len(str(number_values[item])))
        group_counts[index] += difference
    elif difference < 0:
        for index in sorted(number_indices, key=lambda item: group_counts[item], reverse=True):
            reduction = min(group_counts[index] - 1, -difference)
            group_counts[index] -= reduction
            difference += reduction
            if difference == 0:
                break
        if difference:
            raise ValueError("Cannot reconcile number normalization with phoneme groups")

    if len(words) != len(word_timings) or sum(group_counts) != len(phoneme_words):
        raise ValueError("Word timings cannot be mapped to the phoneme groups")

    phone_timings = []
    for word_index, group_count in enumerate(group_counts):
        timing = word_timings[word_index]
        total_frames = timing["duration_frames"]
        if total_frames < group_count:
            raise ValueError("Aligned word is too short to allocate to its phoneme groups")
        base_frames, remainder = divmod(total_frames, group_count)
        elapsed_frames = 0
        span_seconds = timing["end"] - timing["start"]
        for group_index in range(group_count):
            group_frames = base_frames + (1 if group_index < remainder else 0)
            start_fraction = elapsed_frames / total_frames
            elapsed_frames += group_frames
            end_fraction = elapsed_frames / total_frames
            phone_timings.append({
                "start": timing["start"] + span_seconds * start_fraction,
                "end": timing["start"] + span_seconds * end_fraction,
                "duration_frames": group_frames,
            })
    return phone_timings


def align_sample(model, processor, audio_path, text, device):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    normalized = unicodedata.normalize("NFC", text.lower()).strip()
    words = normalized.split()
    if not words:
        raise ValueError("Empty transcript")

    tokenizer = processor.tokenizer
    delimiter_id = tokenizer.convert_tokens_to_ids(tokenizer.word_delimiter_token)
    unknown_id = tokenizer.unk_token_id
    word_ids = []
    unknown_chars = set()
    for word in words:
        ids = []
        for char in word:
            token_id = tokenizer.convert_tokens_to_ids(char)
            if token_id is None or (unknown_id is not None and token_id == unknown_id):
                category = unicodedata.category(char)
                if category[0] in "PS" or category == "Cf":
                    continue
                unknown_chars.add(f"{char!r} (U+{ord(char):04X})")
            else:
                ids.append(token_id)
        word_ids.append(ids)

    if unknown_chars:
        raise ValueError(f"CTC tokenizer does not support characters: {sorted(unknown_chars)}")
    if any(not ids for ids in word_ids):
        raise ValueError("Transcript contains a word with no CTC-supported characters")

    labels = []
    ranges = []
    for word_index, ids in enumerate(word_ids):
        start = len(labels)
        labels.extend(ids)
        ranges.append((start, len(labels)))
        if word_index + 1 < len(word_ids):
            labels.append(delimiter_id)

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
    with torch.inference_mode():
        log_probs = model(inputs).logits[0].log_softmax(-1)

    token_spans = ctc_forced_spans(log_probs, labels, processor.tokenizer.pad_token_id)
    seconds_per_frame = len(audio) / SAMPLE_RATE / log_probs.shape[0]
    timings = []
    for start, end in ranges:
        begin_frame = token_spans[start][0]
        end_frame = token_spans[end - 1][1]
        begin_seconds = begin_frame * seconds_per_frame
        end_seconds = end_frame * seconds_per_frame
        timings.append({
            "start": begin_seconds,
            "end": end_seconds,
            "duration_frames": max(1, round((end_seconds - begin_seconds) * KOKORO_FRAME_RATE)),
        })
    return words, timings


def main():
    parser = argparse.ArgumentParser(description="Experimental CTC word alignment for VieNeu")
    parser.add_argument("--metadata", default="finetune_data/train_metadata.csv")
    parser.add_argument("--wav-dir", default="finetune_data/wavs")
    parser.add_argument("--output", default="finetune_data/train_alignments.jsonl")
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Limit pilot size; use 0 to process all rows")
    parser.add_argument("--model", default=ALIGNER_ID)
    parser.add_argument("--resume", action="store_true",
                        help="Keep completed alignment records and process remaining IDs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Vietnamese CTC aligner on {device}: {args.model}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    existing_ids = set()
    if args.resume and os.path.isfile(args.output):
        with open(args.output, "rb+") as previous:
            contents = previous.read()
            last_newline = contents.rfind(b"\n")
            if last_newline < len(contents) - 1:
                contents = contents[:last_newline + 1]
                previous.seek(len(contents))
                previous.truncate()

            for line_number, line in enumerate(contents.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    phoneme_words = record.get("phoneme_words") or []
                    phoneme_timings = record.get("phoneme_timings") or []
                    if record.get("id") and phoneme_words and len(phoneme_timings) == len(phoneme_words):
                        existing_ids.add(record["id"])
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as error:
                    raise ValueError(
                        f"Cannot resume: invalid record in {args.output} at line {line_number}"
                    ) from error
        print(f"Resuming: {len(existing_ids)} IDs already aligned")

    total = aligned = skipped = 0
    output_mode = "a" if args.resume else "w"
    with open(args.metadata, encoding="utf-8", newline="") as source, \
            open(args.output, output_mode, encoding="utf-8") as target:
        rows = csv.DictReader(source, delimiter="|")
        for row in rows:
            if args.max_samples and total >= args.max_samples:
                break
            total += 1
            if row["id"] in existing_ids:
                continue
            wav_path = os.path.join(args.wav_dir, f"{row['id']}.wav")
            try:
                words, timings = align_sample(model, processor, wav_path, row["text"], device)
                phoneme_words = row["phoneme"].split()
                phoneme_timings = map_word_timings_to_phonemes(words, timings, phoneme_words)
                record = {
                    "id": row["id"],
                    "speaker": row["speaker"],
                    "words": words,
                    "phoneme_words": phoneme_words,
                    "word_timings": timings,
                    "phoneme_timings": phoneme_timings,
                }
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                existing_ids.add(row["id"])
                aligned += 1
            except Exception as error:
                skipped += 1
                print(f"[skip] {row.get('id', '?')}: {error}")

            if total % 16 == 0:
                print(f"Processed {total}: aligned={aligned}, skipped={skipped}")

    print(
        f"Done: aligned_this_run={aligned}, already_aligned={len(existing_ids) - aligned}, "
        f"skipped={skipped}, output={args.output}"
    )
    print("Word timings are approximate; they are not phoneme-level ground truth.")
    if not existing_ids:
        raise RuntimeError("No samples aligned; refusing to continue with an empty alignment file")


if __name__ == "__main__":
    main()