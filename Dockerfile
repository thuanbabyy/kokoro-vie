# Base image PyTorch voi CUDA 12.1
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cai dat cac package he thong xu ly am thanh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependencies truoc de tan dung Docker cache
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy toan bo code vao container
COPY . /workspace/

CMD ["python", "lora_setup.py"]
