# GR00T Fine-Tuning Container
#
# Fine-tunes NVIDIA GR00T N1.7-3B VLA model on customer data in LeRobot v2 format.
# Designed to run as a SageMaker Training Job on ml.g5.12xlarge (4x A10G GPUs).
#
# Build:
#   docker build --platform linux/amd64 -t groot-training .
#
# SageMaker mounts:
#   /opt/ml/input/data/training/  <- dataset (from S3)
#   /opt/ml/model/                <- output (uploaded to S3 after training)
#
# NOTE: This Dockerfile installs only the packages needed for GR00T fine-tuning.
# It does NOT include simulation backends (MuJoCo, Gymnasium envs) since training
# runs on pre-collected datasets, not in a live simulator.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.5.* \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# GR00T fine-tuning dependencies (no simulator backends)
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate \
    diffusers \
    safetensors \
    datasets \
    pyarrow \
    pandas \
    numpy \
    pillow \
    opencv-python-headless \
    pyyaml \
    boto3 \
    mlflow \
    huggingface-hub

# LeRobot dataset loading only (skip simulation extras)
RUN pip install --no-cache-dir --no-deps lerobot

# Install remaining lerobot data dependencies without pulling sim backends
RUN pip install --no-cache-dir \
    av \
    imageio \
    imageio-ffmpeg \
    jsonlines

# Isaac GR00T SDK (fine-tuning scripts)
# NOTE: If isaac-groot is not yet on PyPI, this will be replaced with
# a git clone of the SDK into /opt/isaac-groot
RUN pip install --no-cache-dir isaac-groot 2>/dev/null || \
    echo "isaac-groot not available on PyPI — will use transformers fallback"

# Copy training entrypoint
COPY train_entrypoint.py /opt/ml/code/train_entrypoint.py

ENV SAGEMAKER_PROGRAM=train_entrypoint.py
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "train_entrypoint.py"]
