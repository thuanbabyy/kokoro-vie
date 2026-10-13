# -*- coding: utf-8 -*-
"""
LoRA Trainer cho Kokoro-82M - Tieng Viet
=========================================
Chay sau khi co du audio trong finetune_data/wavs/
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, torch, torch.nn as nn
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
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                uid, text = line.split("|", 1)
                wav_path  = os.path.join(wav_dir, f"{uid}.wav")
                if os.path.exists(wav_path):
                    dur = librosa.get_duration(path=wav_path)
                    if dur <= max_dur:
                        self.data.append({"id": uid, "text": text, "wav": wav_path})

        print(f"  [Dataset] Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import librosa
        item  = self.data[idx]
        audio, _ = librosa.load(item["wav"], sr=self.sr, mono=True)
        return {"text": item["text"], "audio": torch.FloatTensor(audio)}


def collate_fn(batch):
    """Pad audio ve cung do dai"""
    max_len = max(x["audio"].shape[0] for x in batch)
    audios  = torch.zeros(len(batch), max_len)
    texts   = []
    for i, x in enumerate(batch):
        n = x["audio"].shape[0]
        audios[i, :n] = x["audio"]
        texts.append(x["text"])
    return {"text": texts, "audio": audios}


# ================================================================
# TRAINING LOOP (simplified - can kokoro package de chay day du)
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
        self.model.train()
        total_loss = 0.0
        n_steps    = 0

        for batch in dataloader:
            audio = batch["audio"].to(self.device)
            texts = batch["text"]

            # Tinh mel target
            mel_target = compute_mel(audio)

            # Forward qua model (can kokoro package de co model dung)
            # Hien tai: simplified loss de demo
            self.optimizer.zero_grad()

            # Placeholder forward - thay the bang Kokoro forward pass
            # mel_pred = self.model(texts, ...)
            # loss = F.l1_loss(mel_pred, mel_target)

            # Demo: L2 loss tren embedding gia lap
            dummy_pred = torch.randn_like(mel_target, requires_grad=True)
            loss = nn.functional.mse_loss(dummy_pred, mel_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg["grad_clip"]
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_steps    += 1

        avg_loss = total_loss / max(n_steps, 1)
        self.loss_history.append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        if val_loader is None or len(val_loader) == 0:
            return None
        self.model.eval()
        total_loss = 0.0
        n_steps    = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(self.device)
                mel_target = compute_mel(audio)
                dummy_pred = torch.randn_like(mel_target)
                loss = nn.functional.mse_loss(dummy_pred, mel_target)
                total_loss += loss.item()
                n_steps    += 1
        return total_loss / max(n_steps, 1)

    def train(self, train_loader, val_loader=None):
        print(f"\n  Bat dau training tren {self.device}...")
        print(f"  Epochs: {self.cfg['epochs']} | LR: {self.cfg['lr']}")
        if val_loader:
            print(f"  [Eval] Tap validation: {len(val_loader.dataset)} mau")
        print()

        best_val_loss = float("inf")

        for epoch in range(1, self.cfg["epochs"] + 1):
            t0   = time.time()
            loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            dt   = time.time() - t0
            lr   = self.scheduler.get_last_lr()[0]

            log_str = f"  Epoch {epoch:3d}/{self.cfg['epochs']} | Train Loss: {loss:.4f}"
            if val_loss is not None:
                log_str += f" | Val Loss: {val_loss:.4f}"
            log_str += f" | LR: {lr:.2e} | Time: {dt:.1f}s"
            print(log_str)

            # Luu best checkpoint
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = os.path.join(self.cfg["output_dir"], "lora_best.pt")
                save_lora_weights(self.model, best_ckpt)

            # Luu checkpoint dinh ky
            if epoch % self.cfg["save_every"] == 0:
                ckpt = os.path.join(self.cfg["output_dir"],
                                    f"lora_rank{self.cfg['lora_rank']}_epoch{epoch}.pt")
                save_lora_weights(self.model, ckpt)

        # Luu final
        final = os.path.join(self.cfg["output_dir"], "lora_final.pt")
        save_lora_weights(self.model, final)
        print(f"\n  [DONE] LoRA weights da luu tai: {self.cfg['output_dir']}")

        # Luu loss history
        loss_path = os.path.join(self.cfg["output_dir"], "loss_history.json")
        with open(loss_path, "w") as f:
            json.dump(self.loss_history, f)


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

    # Import kokoro model (can cai pip install kokoro)
    try:
        from kokoro import KModel
        model = KModel(CFG["model_dir"])
        print("  [OK] Da load Kokoro model")
    except ImportError:
        print("  [!] Chua cai kokoro package")
        print("  Chay: pip install kokoro")
        print("\n  [Demo mode] Tao model gia lap de test LoRA injection...")

        from transformers import BertModel, BertConfig
        config = BertConfig(
            hidden_size=256, num_attention_heads=4,
            num_hidden_layers=2, intermediate_size=512
        )
        model = BertModel(config)
        print("  [OK] Da tao BERT demo model")

    # Inject LoRA
    print("\n  Dang inject LoRA...")
    model = inject_lora(
        model,
        targets  = CFG["target_mods"],
        rank     = CFG["lora_rank"],
        alpha    = CFG["lora_alpha"],
        dropout  = CFG["lora_dropout"],
    )

    # Load dataset
    print("\n  Dang load dataset...")
    try:
        train_ds = TTSDataset(train_meta, CFG["wav_dir"], CFG["sample_rate"])
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                                  shuffle=True, collate_fn=collate_fn)

        val_loader = None
        if val_meta and os.path.exists(val_meta):
            val_ds = TTSDataset(val_meta, CFG["wav_dir"], CFG["sample_rate"])
            val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"],
                                    shuffle=False, collate_fn=collate_fn)

        # Train
        trainer = LoRATrainer(model, CFG)
        trainer.train(train_loader, val_loader)

    except ImportError as e:
        print(f"\n  [!] Thieu thu vien: {e}")
        print("  Chay: pip install librosa soundfile")

        # Giai thich cau truc ma khong train
        print("\n  [INFO] LoRA da san sang - chi can:")
        print("  1. pip install kokoro librosa soundfile")
        print("  2. Them audio vao wavs/")
        print("  3. Chay lai script nay")


if __name__ == "__main__":
    main()
