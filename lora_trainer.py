# -*- coding: utf-8 -*-
"""
LoRA Trainer cho Kokoro-82M - Tieng Viet
=========================================
Chay sau khi co du audio trong finetune_data/wavs/
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, torch, torch.nn as nn
import csv
from itertools import chain
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time

# ================================================================
# CONFIG
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    # LoRA
    "lora_rank"    : 8,
    "lora_alpha"   : 16.0,
    "lora_dropout" : 0.05,
    "target_mods"  : ["query", "key", "value", "dense", "fc1", "fc2"],

    # Training
    "lr"           : 1e-4,
    "batch_size"   : 8 if torch.cuda.is_available() else 4,
    "epochs"       : 50,
    "warmup_steps" : 100,
    "grad_clip"    : 1.0,
    "save_every"   : 10,

    # Paths
    "model_dir"    : os.path.join(BASE_DIR, "Kokoro-82M"),
    "train_meta"   : os.path.join(BASE_DIR, "finetune_data", "train_metadata.csv"),
    "val_meta"     : os.path.join(BASE_DIR, "finetune_data", "val_metadata.csv"),
    "metadata"     : os.path.join(BASE_DIR, "finetune_data", "metadata.csv"),
    "wav_dir"      : os.path.join(BASE_DIR, "finetune_data", "wavs"),
    "output_dir"   : os.path.join(BASE_DIR, "lora_output"),
    "sample_rate"  : 24000,
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
}


# ================================================================
# LORA LAYER
# ================================================================
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.original  = linear
        self.rank      = rank
        self.scaling   = alpha / rank
        in_f, out_f    = linear.in_features, linear.out_features

        self.lora_A    = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B    = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze goc
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.dropout(x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self):
        with torch.no_grad():
            self.original.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        return self.original


def inject_lora(model, targets, rank, alpha, dropout, verbose=True):
    count, trainable = 0, 0
    for param in model.parameters():
        param.requires_grad = False

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if not any(t in leaf for t in targets):
            continue
        # Lay parent
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        lora = LoRALinear(mod, rank, alpha, dropout)
        setattr(parent, parts[-1], lora)
        trainable += rank * (mod.in_features + mod.out_features)
        count += 1

    total = sum(p.numel() for p in model.parameters())
    tr    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  [LoRA] Injected {count} layers | Trainable: {tr:,}/{total:,} ({tr/total*100:.2f}%)")
    return model


def save_lora_weights(model, path):
    """Chi luu cac trong so LoRA (nho hon nhieu so voi full model)"""
    lora_state = {k: v for k, v in model.state_dict().items()
                  if "lora_A" in k or "lora_B" in k}
    torch.save(lora_state, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  [Save] {path} ({size_kb:.0f} KB)")


def load_lora_weights(model, path):
    """Load LoRA weights vao model"""
    lora_state = torch.load(path, map_location="cpu")
    model.load_state_dict(lora_state, strict=False)
    print(f"  [Load] LoRA weights tu {path}")
    return model


# ================================================================
# DATASET
# ================================================================
class TTSDataset(Dataset):
    def __init__(self, metadata_path, wav_dir, sr=24000, max_dur=20.0):
        import librosa
        self.data = []
        self.sr   = sr

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            first_row = next(reader, None)
            if first_row == ["id", "text", "phoneme", "speaker"]:
                rows = reader
            else:
                rows = chain([first_row] if first_row else [], reader)

            for row in rows:
                if not row or len(row) < 2:
                    continue
                uid, text = row[0].strip(), row[1].strip()
                phoneme = row[2].strip() if len(row) > 2 else ""
                speaker = row[3].strip() if len(row) > 3 else "unknown"
                wav_path  = os.path.join(wav_dir, f"{uid}.wav")
                if os.path.exists(wav_path):
                    dur = librosa.get_duration(path=wav_path)
                    if dur <= max_dur:
                        self.data.append({
                            "id": uid, "text": text, "phoneme": phoneme,
                            "speaker": speaker, "wav": wav_path,
                        })

        print(f"  [Dataset] Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import librosa
        item  = self.data[idx]
        audio, _ = librosa.load(item["wav"], sr=self.sr, mono=True)
        return {
            "text": item["text"],
            "phoneme": item["phoneme"],
            "speaker": item["speaker"],
            "audio": torch.FloatTensor(audio),
        }


def collate_fn(batch):
    """Pad audio ve cung do dai"""
    max_len = max(x["audio"].shape[0] for x in batch)
    audios  = torch.zeros(len(batch), max_len)
    texts   = []
    phonemes = []
    speakers = []
    for i, x in enumerate(batch):
        n = x["audio"].shape[0]
        audios[i, :n] = x["audio"]
        texts.append(x["text"])
        phonemes.append(x["phoneme"])
        speakers.append(x["speaker"])
    return {
        "text": texts, "phoneme": phonemes, "speaker": speakers,
        "audio": audios,
    }


def report_preflight(model, train_dataset, val_dataset=None):
    print("\n  [Preflight] Dataset and Kokoro compatibility")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Speakers: {len({item['speaker'] for item in train_dataset.data})}")

    if not train_dataset.data or any(not item["phoneme"] for item in train_dataset.data):
        raise ValueError(
            "Metadata must contain phoneme and speaker columns. "
            "Regenerate it with the updated downloader before continuing."
        )

    counts = Counter(item["speaker"] for item in train_dataset.data)
    print(f"  Samples per speaker: min={min(counts.values())}, max={max(counts.values())}")
    if val_dataset is not None:
        print(f"  Validation samples: {len(val_dataset)}")

    symbols = set("".join(item["phoneme"] for item in train_dataset.data))
    missing = sorted(symbols - set(model.vocab))
    print(f"  Phoneme symbols in data: {len(symbols)}")
    print(f"  Symbols missing from Kokoro vocab: {missing!r}")
    if missing:
        raise ValueError(
            "Kokoro silently drops phoneme symbols absent from its vocabulary; "
            "resolve these symbols before training."
        )


# ================================================================
# TRAINING LOOP
# ================================================================
def compute_mel(audio, sr=24000, n_mels=80, n_fft=1024, hop=256):
    """Tinh mel spectrogram don gian"""
    import torchaudio.transforms as T
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, f_min=0, f_max=8000
    )
    mel = mel_transform(audio)
    return mel.clamp(min=1e-5).log()


class LoRATrainer:
    def __init__(self, model, cfg):
        self.model    = model.to(cfg["device"])
        self.cfg      = cfg
        self.device   = cfg["device"]

        # Chi optimize LoRA params
        lora_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(lora_params, lr=cfg["lr"], weight_decay=0.01)

        # LR scheduler voi warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg["lr"],
            total_steps=cfg["epochs"] * 100,  # ~100 steps/epoch
            pct_start=cfg["warmup_steps"] / (cfg["epochs"] * 100)
        )

        os.makedirs(cfg["output_dir"], exist_ok=True)
        self.loss_history = []

    def train_epoch(self, dataloader, epoch):
        raise NotImplementedError(
            "Kokoro's released inference forward is no_grad and this project "
            "does not yet implement a differentiable multi-speaker training loss."
        )

    def validate(self, val_loader):
        raise NotImplementedError(
            "Validation requires the same real Kokoro training objective as training."
        )

    def train(self, train_loader, val_loader=None):
        raise NotImplementedError(
            "Training is intentionally stopped: do not interpret the former "
            "random demo loss or its checkpoints as a trained Kokoro adapter."
        )


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 60)
    print("  KOKORO LORA TRAINER - TIENG VIET")
    print("=" * 60)
    print(f"  Device: {CFG['device']}")
    print(f"  LoRA Rank: {CFG['lora_rank']} | Alpha: {CFG['lora_alpha']}")

    # Kiem tra dataset: uu tien train_metadata.csv, neu khong thi metadata.csv
    train_meta = CFG["train_meta"] if os.path.exists(CFG["train_meta"]) else CFG["metadata"]
    val_meta   = CFG["val_meta"] if os.path.exists(CFG["val_meta"]) else None

    if not os.path.exists(train_meta):
        print(f"\n  [!] Chua co metadata tai: {train_meta}")
        print("  Chay: python download_vieneu_dataset.py truoc")
        return

    wav_files = [x for x in os.listdir(CFG["wav_dir"]) if x.endswith(".wav")] \
                if os.path.exists(CFG["wav_dir"]) else []

    if len(wav_files) == 0:
        print("\n  [!] Chua co file audio trong wavs/")
        print("  Chay: python download_vieneu_dataset.py de tai audio")
        return

    print(f"\n  Tim thay {len(wav_files)} file audio")

    from kokoro import KModel
    config_path = os.path.join(CFG["model_dir"], "config.json")
    checkpoint_path = os.path.join(CFG["model_dir"], "kokoro-v1_0.pth")
    model = KModel(config=config_path, model=checkpoint_path)
    print("  [OK] Da load Kokoro model tu checkpoint local")

    # Inject LoRA
    print("\n  Dang inject LoRA...")
    model = inject_lora(
        model,
        targets  = CFG["target_mods"],
        rank     = CFG["lora_rank"],
        alpha    = CFG["lora_alpha"],
        dropout  = CFG["lora_dropout"],
    )

    print("\n  Dang load metadata...")
    train_ds = TTSDataset(train_meta, CFG["wav_dir"], CFG["sample_rate"])
    val_ds = None
    if val_meta and os.path.exists(val_meta):
        val_ds = TTSDataset(val_meta, CFG["wav_dir"], CFG["sample_rate"])

    report_preflight(model, train_ds, val_ds)
    print("\n  [PREFLIGHT ONLY] Data/model checks passed.")
    print("  No training was started: a differentiable Kokoro training loss is not implemented.")


if __name__ == "__main__":
    main()
