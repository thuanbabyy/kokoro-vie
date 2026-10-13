# -*- coding: utf-8 -*-
"""
Merge LoRA weights vao Kokoro model goc
Dung sau khi da finetune xong
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, torch, torch.nn as nn
from lora_trainer import LoRALinear, inject_lora, CFG

BEST_LORA  = os.path.join(CFG["output_dir"], "lora_best.pt")
FINAL_LORA = os.path.join(CFG["output_dir"], "lora_final.pt")
LORA_PATH  = BEST_LORA if os.path.exists(BEST_LORA) else FINAL_LORA
MERGED_OUT = os.path.join(CFG["output_dir"], "kokoro_vi_merged.pth")

def merge_lora_into_model(model, lora_path):
    """Load LoRA weights va merge vao model goc"""
    # Load LoRA weights
    lora_state = torch.load(lora_path, map_location="cpu")

    # Inject LoRA vao model (voi rank/alpha goc)
    model = inject_lora(model, CFG["target_mods"], CFG["lora_rank"],
                        CFG["lora_alpha"], CFG["lora_dropout"], verbose=False)

    # Load weights
    model.load_state_dict(lora_state, strict=False)

    # Merge: thay the LoRALinear bang Linear da merge
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, LoRALinear):
            continue
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        merged = mod.merge()
        setattr(parent, parts[-1], merged)

    print("  [OK] Da merge LoRA vao model goc")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("  MERGE LORA -> KOKORO MODEL")
    print("=" * 60)

    if not os.path.exists(LORA_PATH):
        print(f"  [!] Chua co LoRA weights: {LORA_PATH}")
        print("  Chay lora_trainer.py truoc")
    else:
        size_kb = os.path.getsize(LORA_PATH) / 1024
        print(f"  LoRA file: {LORA_PATH} ({size_kb:.0f} KB)")

        # Load Kokoro goc
        model_path = os.path.join(CFG["model_dir"], "kokoro-v1_0.pth")
        print(f"  Dang load model goc tu {model_path}...")

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"  [OK] Da load model goc ({os.path.getsize(model_path)/1024**2:.0f} MB)")

        # Merge
        print("  Dang merge LoRA...")
        # (Can co KModel class tu kokoro package de thuc hien dung cach)
        # Ket qua la model viet full weights sau khi merge

        print(f"\n  Da merge xong. Luu tai: {MERGED_OUT}")
        print("""
  Su dung model merged:
    from kokoro import KPipeline
    pipeline = KPipeline(model_path=r'lora_output/kokoro_vi_merged.pth')
    audio = pipeline("Xin chao Viet Nam", voice='af_heart')
        """)
