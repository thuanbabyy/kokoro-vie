# -*- coding: utf-8 -*-
"""
LoRA Finetune Kokoro-82M cho tieng Viet
========================================
Kokoro dung kien truc StyleTTS2:
  - Text Encoder (BERT-based = PL-BERT) <- Co the gan LoRA
  - Style Encoder
  - Decoder (ISTFTNet)
  - Duration Predictor
  - Prosody Model

LoRA se duoc gan vao cac lop Linear cua Text Encoder va Prosody Model
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import torch
import torch.nn as nn
from pathlib import Path

# ================================================================
# KIEM TRA GPU / DEPENDENCIES
# ================================================================
def check_environment():
    print("=" * 60)
    print("  KIEM TRA MOI TRUONG")
    print("=" * 60)

    # Python
    import sys
    print(f"  Python     : {sys.version.split()[0]}")

    # PyTorch
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA co san: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM       : {vram:.1f} GB")
    else:
        print("  [!] Khong co GPU - Se dung CPU (chay cham hon)")

    # Kiem tra cac thu vien can thiet
    missing = []
    deps = {
        "peft": "peft",
        "accelerate": "accelerate",
        "transformers": "transformers",
        "soundfile": "soundfile",
        "librosa": "librosa",
    }
    for name, pkg in deps.items():
        try:
            __import__(name)
            print(f"  {name:15s}: OK")
        except ImportError:
            print(f"  {name:15s}: THIEU - can cai: pip install {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\n  [!] Cai dat cac thu vien con thieu:")
        print(f"      pip install {' '.join(missing)}")
        return False
    return True


# ================================================================
# LOA MODULE - Gan vao cac lop Linear
# ================================================================
class LoRALinear(nn.Module):
    """
    Thay the nn.Linear bang LoRA version
    W_new = W_pretrained + alpha/r * (A @ B)
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices - khoi tao: A~N(0,1), B=0 de ban dau W_new = W_pretrained
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dong bang trong so goc
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Output goc
        base_output = self.original(x)
        # LoRA delta: x @ A.T @ B.T * scaling
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output

    def merge_weights(self):
        """Merge LoRA vao trong so goc (de inference nhanh hon)"""
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            self.original.weight.data += delta
        return self.original


def inject_lora(model, target_modules=None, rank=8, alpha=16.0, verbose=True):
    """
    Quet qua model, thay the cac nn.Linear bang LoRALinear

    target_modules: list ten module can gan LoRA
                    None = gan vao tat ca Linear layer
    """
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense", "fc1", "fc2"]

    lora_count = 0
    total_params = 0
    lora_params = 0

    for name, module in model.named_modules():
        # Kiem tra neu module la nn.Linear va ten khop voi target
        if isinstance(module, nn.Linear):
            module_name = name.split(".")[-1]
            if any(t in module_name for t in target_modules):
                # Lay parent module de thay the
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)

                lora_count += 1
                lora_params += rank * (module.in_features + module.out_features)

    # Dem tong tham so
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"\n  [LoRA] Da inject vao {lora_count} layers")
        print(f"  [LoRA] Tong tham so     : {total_params:,}")
        print(f"  [LoRA] Tham so trainable: {trainable_params:,}")
        print(f"  [LoRA] Ti le trainable  : {trainable_params/total_params*100:.2f}%")

    return model, trainable_params


# ================================================================
# LOAD KOKORO MODEL
# ================================================================
def load_kokoro_model(model_dir):
    """Load Kokoro-82M model tu file .pth"""
    print("\n[...] Dang load Kokoro model...")

    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "kokoro-v1_0.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Khong tim thay model: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

    print(f"  [OK] Da load model tu: {model_path}")
    print(f"  [OK] File size: {os.path.getsize(model_path)/1024**2:.1f} MB")

    # In cac key chinh cua model
    if isinstance(state_dict, dict):
        keys = list(state_dict.keys()) if "net" not in state_dict else list(state_dict["net"].keys())
        print(f"\n  Cac module chinh trong model:")
        modules_seen = set()
        for k in keys[:50]:
            top_module = k.split(".")[0]
            if top_module not in modules_seen:
                modules_seen.add(top_module)
                print(f"    - {top_module}")

    return state_dict, config


# ================================================================
# TAO DATASET
# ================================================================
class VietnameseTTSDataset(torch.utils.data.Dataset):
    """
    Dataset cho finetune TTS tieng Viet
    Can co: text + audio tuong ung
    """
    def __init__(self, metadata_path, wav_dir, sample_rate=24000):
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.data = []

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "|" in line:
                        parts = line.split("|")
                        wav_id = parts[0].strip()
                        text = parts[1].strip()
                        wav_path = os.path.join(wav_dir, f"{wav_id}.wav")
                        if os.path.exists(wav_path):
                            self.data.append({"id": wav_id, "text": text, "wav": wav_path})
        else:
            print(f"  [!] Chua co metadata.csv - can tao truoc khi train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import librosa
        item = self.data[idx]
        audio, sr = librosa.load(item["wav"], sr=self.sample_rate)
        return {
            "id": item["id"],
            "text": item["text"],
            "audio": torch.FloatTensor(audio),
            "duration": len(audio) / sr
        }


# ================================================================
# TRAINING CONFIG
# ================================================================
LORA_CONFIG = {
    # LoRA hyperparams
    "rank": 8,                    # Rank cua LoRA (4, 8, 16, 32) - nho = nhe hon
    "alpha": 16.0,                # LoRA alpha (thuong = 2*rank)
    "target_modules": [           # Cac lop Linear se duoc gan LoRA
        "query", "key", "value",  # Attention layers trong BERT/PL-BERT
        "dense",                  # Feed-forward layers
        "fc1", "fc2",             # MLP layers
    ],
    "dropout": 0.05,              # LoRA dropout

    # Training hyperparams
    "learning_rate": 1e-4,        # LR cao hon khi train LoRA
    "batch_size": 4,              # Batch size (giam neu het VRAM)
    "num_epochs": 50,             # So epoch
    "warmup_steps": 100,
    "gradient_clip": 1.0,
    "save_every": 10,             # Luu checkpoint moi N epoch

    # Paths
    "model_dir": r"c:\Politetech\kokoro\Kokoro-82M",
    "data_dir": r"c:\Politetech\kokoro\finetune_data",
    "wav_dir": r"c:\Politetech\kokoro\finetune_data\wavs",
    "output_dir": r"c:\Politetech\kokoro\lora_output",
    "metadata": r"c:\Politetech\kokoro\finetune_data\metadata.csv",
}


# ================================================================
# MAIN - KIEM TRA VA HIEN THI THONG TIN
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  KOKORO LORA FINETUNE - TIENG VIET")
    print("=" * 60)

    # 1. Kiem tra moi truong
    env_ok = check_environment()

    # 2. Load va inspect model
    try:
        state_dict, config = load_kokoro_model(LORA_CONFIG["model_dir"])

        print("\n" + "=" * 60)
        print("  THONG TIN MODEL")
        print("=" * 60)
        print(f"  Hidden dim  : {config.get('hidden_dim', 'N/A')}")
        print(f"  n_layer     : {config.get('n_layer', 'N/A')}")
        print(f"  style_dim   : {config.get('style_dim', 'N/A')}")
        print(f"  n_token     : {config.get('n_token', 'N/A')}")
        plbert = config.get("plbert", {})
        print(f"\n  PL-BERT (Text Encoder):")
        print(f"    hidden_size         : {plbert.get('hidden_size', 768)}")
        print(f"    num_attention_heads : {plbert.get('num_attention_heads', 12)}")
        print(f"    num_hidden_layers   : {plbert.get('num_hidden_layers', 12)}")
        print(f"    intermediate_size   : {plbert.get('intermediate_size', 2048)}")

    except Exception as e:
        print(f"\n  [!] Loi load model: {e}")

    # 3. Hien thi cau hinh LoRA
    print("\n" + "=" * 60)
    print("  CAU HINH LORA")
    print("=" * 60)
    print(f"  Rank        : {LORA_CONFIG['rank']}")
    print(f"  Alpha       : {LORA_CONFIG['alpha']}")
    print(f"  LR          : {LORA_CONFIG['learning_rate']}")
    print(f"  Batch size  : {LORA_CONFIG['batch_size']}")
    print(f"  Epochs      : {LORA_CONFIG['num_epochs']}")
    print(f"  Target mods : {', '.join(LORA_CONFIG['target_modules'])}")

    # 4. Uoc tinh VRAM
    print("\n" + "=" * 60)
    print("  UOC TINH VRAM")
    print("=" * 60)
    model_size_mb = 327  # kokoro-v1_0.pth
    lora_overhead = model_size_mb * 0.02 * LORA_CONFIG["rank"] / 8
    grad_size = lora_overhead * 2
    activations = LORA_CONFIG["batch_size"] * 50  # ~50MB/sample
    total_vram = model_size_mb + lora_overhead + grad_size + activations
    print(f"  Model weights   : ~{model_size_mb} MB")
    print(f"  LoRA params     : ~{lora_overhead:.0f} MB")
    print(f"  Gradients       : ~{grad_size:.0f} MB")
    print(f"  Activations     : ~{activations} MB")
    print(f"  ─────────────────────────────")
    print(f"  TONG UOC TINH   : ~{total_vram:.0f} MB ({total_vram/1024:.1f} GB)")

    # 5. Cau truc thu muc can thiet
    print("\n" + "=" * 60)
    print("  CAU TRUC DIRECTORY CAN TAO")
    print("=" * 60)
    print("""
  c:\\Politetech\\kokoro\\
    finetune_data\\
      wavs\\           <- Cho audio .wav (22050Hz/24000Hz, mono)
        vi_0000.wav
        vi_0001.wav
        ...
      metadata.csv    <- Format: vi_0000|Noi dung cau viet
      sentences.txt   <- (Da co) ✓

    lora_output\\     <- Checkpoint LoRA se luu o day
      lora_rank8_epoch10.pt
      lora_rank8_epoch20.pt
      ...
      lora_final.pt
  """)

    print("=" * 60)
    print("  CACH CHAY FINETUNE")
    print("=" * 60)
    print("""
  Buoc 1: Cai thu vien can thiet
    pip install peft accelerate transformers librosa soundfile

  Buoc 2: Chuan bi audio
    - Ghi am tung cau trong sentences.txt
    - Luu vao finetune_data\\wavs\\vi_XXXX.wav

  Buoc 3: Tao metadata.csv
    python prepare_metadata.py

  Buoc 4: Chay finetune LoRA
    python lora_trainer.py

  Buoc 5: Merge LoRA va test
    python merge_lora.py
  """)
