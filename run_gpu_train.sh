#!/bin/bash
set -e

echo "============================================================"
echo "  KOKORO GPU PREFLIGHT"
echo "============================================================"

# Kiem tra GPU
nvidia-smi

# 1. Kiem tra moi truong va model
echo -e "\n[1/3] Kiem tra model va LoRA configuration..."
python lora_setup.py

if [ ! -d "finetune_data/wavs" ] || [ -z "$(ls -A finetune_data/wavs 2>/dev/null)" ]; then
    echo "[ERROR] finetune_data/wavs is empty. Prepare the dataset explicitly first."
    exit 1
fi

if [ ! -f "finetune_data/train_metadata.csv" ] && [ ! -f "finetune_data/metadata.csv" ]; then
    echo "[ERROR] No metadata found under finetune_data/."
    exit 1
fi

echo -e "\n[2/2] Running model/data preflight only..."
python lora_trainer.py

echo -e "\n[INFO] Preflight finished. No training or LoRA merge was performed."
