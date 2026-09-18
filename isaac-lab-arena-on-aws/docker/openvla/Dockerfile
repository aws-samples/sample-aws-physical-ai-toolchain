# OpenVLA prebuilt image: bakes the full install sequence at exact pins.
# Build via CodeBuild (scripts/build_images.py) -- not locally.

ARG DLC_REGISTRY=763104351884.dkr.ecr.us-east-1.amazonaws.com
FROM ${DLC_REGISTRY}/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# Pins (must match train_entry.py / eval_entry.py exactly)
ENV OFT_REPO="https://github.com/moojink/openvla-oft.git" \
    OFT_COMMIT="e4287e94541f459edc4feabc4e181f537cd569a8" \
    TFMD_PIN="tensorflow-metadata==1.17.3" \
    FLASH_ATTN_PIN="flash-attn==2.5.5" \
    WORK="/opt/vla"

# PyPI resilience -- these MUST be set before any pip command
ENV PIP_RETRIES=8 \
    PIP_TIMEOUT=300 \
    PIP_DEFAULT_TIMEOUT=300

WORKDIR ${WORK}

# --- APT packages ---
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends git git-lfs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Clone openvla-oft at pinned commit ---
RUN git clone ${OFT_REPO} ${WORK}/openvla-oft && \
    cd ${WORK}/openvla-oft && \
    git checkout ${OFT_COMMIT}

# --- Install in stages (so retries don't restart from scratch) ---
RUN pip install --no-cache-dir -e ${WORK}/openvla-oft

RUN pip install --no-cache-dir ${TFMD_PIN}

RUN pip install --no-cache-dir packaging ninja

# flash-attn is the long compile step -- give it maximum resilience
RUN pip install --no-cache-dir ${FLASH_ATTN_PIN} --no-build-isolation || \
    (echo "flash-attn failed, retrying..." && sleep 30 && \
     pip install --no-cache-dir ${FLASH_ATTN_PIN} --no-build-isolation)

RUN pip install --no-cache-dir --force-reinstall "numpy<2"

RUN pip install --no-cache-dir ${TFMD_PIN}

# --- Marker file for idempotence check in entry scripts ---
RUN echo "openvla-baked:${OFT_COMMIT}" > /opt/vla/.baked_env

# --- Sanity check ---
RUN python -c "import prismatic; import flash_attn; print('OpenVLA env OK')"
