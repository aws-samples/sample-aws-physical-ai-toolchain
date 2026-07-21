# GR00T Fine-Tuning Container
#
# Fine-tunes NVIDIA GR00T N1.6-3B on customer data in LeRobot v2 format.
# Runs as a SageMaker Training Job on ml.g5.12xlarge (4x A10G, 24 GB each).
#
# Builds in AWS CodeBuild → ECR (NEVER locally — the base + flash-attn build is
# large and x86-only). See containers/groot-training/buildspec.yml.
#
# === PROVENANCE ==================================================================
# This mirrors a PROVEN, validated GR00T fine-tuning setup (the lab-cloud-env
# `groot-train` reference, which has run real UR3 fine-tunes on g5.12xlarge):
#   - SageMaker PyTorch DLC base (Python 3.11, CUDA 12.4)
#   - Isaac-GR00T installed from source as editable, with the pyproject
#     `requires-python == 3.10.*` relaxed to `>= 3.10` so it installs under the
#     DLC's Python 3.11 (the proven workaround)
#   - transformers pinned to 4.51.3 (GR00T N1.6 / Eagle backbone compatibility)
#   - flash-attn for the Eagle backbone's flash_attention_2 config
#   - decord for video loading (GR00T's pyav backend is buggy — proven lesson)
# 24 GB-per-GPU fit is handled at RUNTIME by gradient checkpointing + DeepSpeed
# ZeRO-2 + gradient accumulation (see train_entrypoint.py), NOT a bigger GPU.
#
# UNVALIDATED HERE: this toolchain has not yet rebuilt this exact image in its own
# CodeBuild→ECR or run a job from it. The pins/approach are copied from a validated
# source; confirm with one CodeBuild run + one g5 job before asserting "validated".
# =================================================================================

# SageMaker PyTorch 2.5.1 / Py3.11 / CUDA 12.4 Deep Learning Container (us-west-2).
# This is an AWS-account ECR image — CodeBuild logs into it in buildspec.yml.
# Matches the validated reference's base EXACTLY (pytorch-inference DLC): flash-attn
# --no-build-isolation was only ever verified to resolve against this toolchain.
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

ENV DEBIAN_FRONTEND=noninteractive

# git-lfs (GR00T ships LFS assets), ninja for flash-attn build (matches proven ref).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs ninja-build \
    && rm -rf /var/lib/apt/lists/* && git lfs install

# Clone Isaac-GR00T pinned to the N1.6 RELEASE TAG (immutable commit ead52833).
# CRITICAL: pin the tag, do NOT clone `main`. The entrypoint uses the N1.6 API
# (gr00t.experiment.experiment.run + get_default_config + the gr00t_n1d6 module +
# the vendored Eagle-Block2A-2B-v2 backbone). `main` has since moved to N1.7
# (qwen3 backbone, no gr00t_n1d6, no vendored Eagle) — cloning it breaks both the
# entrypoint import and the Eagle canary below. (The reference Dockerfile this is
# based on cloned `main` unpinned and would now break the same way — fixed here.)
ARG GROOT_REF=n1.6-release
RUN git clone --depth 1 --branch ${GROOT_REF} https://github.com/NVIDIA/Isaac-GR00T.git /opt/isaac-gr00t && \
    cd /opt/isaac-gr00t && git lfs pull && \
    python -c "import pathlib; p=pathlib.Path('pyproject.toml'); p.write_text(p.read_text().replace('==3.10.*', '>=3.10'))" && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir transformers==4.51.3

# flash-attn (Eagle backbone's flash_attention_2) + decord (pyav backend is buggy).
# Install a PREBUILT wheel matching this exact env — `pip install -e .` above pulls
# torch==2.7.1 (GR00T's pin), so we need the cu12 / torch2.7 / cp311 / cxx11abiFALSE
# wheel. Two traps this avoids:
#   - unpinned `pip install flash-attn` → wheel built against a different torch ABI
#     → `undefined symbol: _ZN3c10...` at import (observed);
#   - pinning a version with no matching prebuilt wheel → a 40-min source compile
#     that blows the CodeBuild timeout (observed).
# GR00T pins flash-attn 2.7.4.post1, but that release has NO torch2.7 wheel; 2.8.3.post1
# is the earliest with a prebuilt torch2.7 wheel and is API-compatible. (cxx11abiFALSE
# matches PyTorch's standard prebuilt binaries.)
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir "${FLASH_ATTN_WHEEL}" && \
    python -c "import flash_attn; print(f'flash_attn={flash_attn.__version__}')" && \
    pip install --no-cache-dir decord && \
    python -c "import decord; print(f'decord={decord.__version__}')"

# Fast-fail canary (from the proven ref): assert git-lfs fetched the Eagle backbone
# assets that runtime depends on (config.model.model_name = nvidia/Eagle-Block2A-2B-v2),
# and that CUDA imports — catches an LFS regression at build time, not on a paid g5.
RUN echo "=== Verify Eagle config ===" && \
    python -c "import json; c=json.load(open('/opt/isaac-gr00t/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/config.json')); print(f'model_type={c[\"model_type\"]}')" && \
    echo "=== Verify CUDA ===" && \
    python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

# SageMaker glue + data deps for the entrypoint, baselines, and in-container ingest.
RUN pip install --no-cache-dir boto3 matplotlib pandas pyarrow "zarr>=2.16" opencv-python-headless

# Copy the SageMaker entrypoint.
COPY train_entrypoint.py /opt/ml/code/train_entrypoint.py

ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_PROGRAM=train_entrypoint.py
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "/opt/ml/code/train_entrypoint.py"]
