"""
Script tai Kokoro-82M tu Hugging Face
"""
import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Cai huggingface_hub neu chua co
try:
    from huggingface_hub import snapshot_download
    print("[OK] huggingface_hub da duoc cai san")
except ImportError:
    print("[...] Dang cai huggingface_hub...")
    install("huggingface_hub")
    from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE_DIR, "Kokoro-82M")
os.makedirs(LOCAL_DIR, exist_ok=True)

print()
print("[>>>] Bat dau tai Kokoro-82M...")
print(f"[DIR] Thu muc luu: {LOCAL_DIR}")
print()

try:
    snapshot_download(
        repo_id="hexgrad/Kokoro-82M",
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.git", "*.gitattributes"],
    )
    print()
    print(f"[DONE] Tai xong! Model da luu tai: {LOCAL_DIR}")
except Exception as e:
    print(f"\n[ERROR] Loi: {e}")
    print()
    print("[TIP] Thu dung mirror neu mang cham:")
    print("   $env:HF_ENDPOINT='https://hf-mirror.com'")
    print("   python download_model.py")
