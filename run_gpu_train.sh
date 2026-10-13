#!/bin/bash
set -e

echo "============================================================"
echo "  KOKORO LORA GPU TRAINING PIPELINE"
echo "============================================================"

# Kiem tra GPU
nvidia-smi

# 1. Kiem tra moi truong va model
echo -e "\n[1/3] Kiem tra model va LoRA configuration..."
python lora_setup.py

# 2. Neu chua co data trong wavs, tai VieNeu-TTS-140h (Train 80% / Val 20%)
if [ ! -d "finetune_data/wavs" ] || [ -z "$(ls -A finetune_data/wavs 2>/dev/null)" ]; then
    echo -e "\n[!] Thu muc finetune_data/wavs dang trong."
    if [ "$TRAIN_MODE" == "subset" ]; then
        echo "[!] Dang tai 5000 mau test (Train 80% / Val 20%)..."
        python download_vieneu_dataset.py --max_samples 5000 --val_ratio 0.2
    else
        echo "[!] Dang tai toan bo dataset VieNeu-TTS-140h (Train 80% / Val 20%)..."
        python download_vieneu_dataset.py --full --val_ratio 0.2
    fi
fi

# 3. Chay LoRA Training
echo -e "\n[2/3] Bat dau LoRA Training tren GPU..."
python lora_trainer.py

# 4. Merge LoRA vao model goc
echo -e "\n[3/3] Merge LoRA vao Kokoro model goc..."
python merge_lora.py

echo -e "\n[THANH CONG] Pipeline hoan tat!"
echo "Model merged da san sang tai: lora_output/kokoro_vi_merged.pth"
