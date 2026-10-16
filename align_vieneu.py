# -*- coding: utf-8 -*-
"""Create experimental word timings for VieNeu using a Vietnamese CTC model."""
import argparse
import csv
import json
import os
import unicodedata

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


ALIGNER_ID = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
SAMPLE_RATE = 16000
KOKORO_FRAME_RATE = 40


def ctc_viterbi(log_probs, labels, blank_id):
    """Return frame spans for each label in a forced CTC alignment."""
    if not labels:
        raise ValueError("Cannot align an empty transcript")

    states = [blank_id]
    for label in labels:
        states.extend((label, blank_id))

    frames, state_count = log_probs.shape[0], len(states)
    if frames < len(labels):
        raise ValueError("Audio is too short for the transcript")

    scores = log_probs.new_full((state_count,), float("-inf"))
    scores[0] = log_probs[0, blank_id]
    scores[1] = log_probs[0, labels[0]]
    backpointers = torch.full((frames, state_count), -1, dtype=torch.int32)

    for frame in range(1, frames):
        next_scores = log_probs.new_full((state_count,), float("-inf"))
        for state, token in enumerate(states):
            candidates = [scores[state]]
            sources = [state]
            if state > 0:
                candidates.append(scores[state - 1])
                sources.append(state - 1)
            if (state > 1 and token != blank_id
                    and token != states[state - 2]):
                candidates.append(scores[state - 2])
                sources.append(state - 2)

            candidate_scores = torch.stack(candidates)
            best = int(candidate_scores.argmax().item())
            next_scores[state] = candidate_scores[best] + log_probs[frame, token]
            backpointers[frame, state] = sources[best]
        scores = next_scores

    state = state_count - 1
    if scores[state - 1] > scores[state]:
        state -= 1
    if not torch.isfinite(scores[state]):
        raise ValueError("No valid CTC path")

    path = [state]
    for frame in range(frames - 1, 0, -1):
        state = int(backpointers[frame, state].item())
        if state < 0:
            raise ValueError("Could not backtrack CTC path")
        path.append(state)
    path.reverse()

    spans = []
    for label_index in range(len(labels)):
        label_state = 2 * label_index + 1
        occupied = [frame for frame, value in enumerate(path) if value == label_state]
        if not occupied:
            raise ValueError(f"Transcript token {label_index} received no frames")
        spans.append((occupied[0], occupied[-1] + 1))
    return spans


def align_sample(model, processor, audio_path, text, device):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) / SAMPLE_RATE > 10.0:
        raise ValueError("Vietnamese CTC checkpoint recommends utterances shorter than 10 seconds")
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
        log_probs = model(inputs).logits[0].log_softmax(-1).cpu()

    token_spans = ctc_viterbi(log_probs, labels, processor.tokenizer.pad_token_id)
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Vietnamese CTC aligner on {device}: {args.model}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    total = aligned = skipped = 0
    with open(args.metadata, encoding="utf-8", newline="") as source, \
            open(args.output, "w", encoding="utf-8") as target:
        rows = csv.DictReader(source, delimiter="|")
        for row in rows:
            if args.max_samples and total >= args.max_samples:
                break
            total += 1
            wav_path = os.path.join(args.wav_dir, f"{row['id']}.wav")
            try:
                words, timings = align_sample(model, processor, wav_path, row["text"], device)
                phoneme_words = row["phoneme"].split()
                if len(words) != len(phoneme_words):
                    raise ValueError(
                        f"word/phoneme count mismatch ({len(words)} vs {len(phoneme_words)})"
                    )
                record = {
                    "id": row["id"],
                    "speaker": row["speaker"],
                    "words": words,
                    "phoneme_words": phoneme_words,
                    "word_timings": timings,
                }
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                aligned += 1
            except Exception as error:
                skipped += 1
                print(f"[skip] {row.get('id', '?')}: {error}")

            if total % 16 == 0:
                print(f"Processed {total}: aligned={aligned}, skipped={skipped}")

    print(f"Done: aligned={aligned}, skipped={skipped}, output={args.output}")
    print("Word timings are approximate; they are not phoneme-level ground truth.")
    if aligned == 0:
        raise RuntimeError("No samples aligned; refusing to continue with an empty alignment file")


if __name__ == "__main__":
    main()