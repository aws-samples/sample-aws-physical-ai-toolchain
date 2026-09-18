# MolmoAct2 prebuilt image: bakes LeRobot + dependencies at pinned commits.
# FROM the same DLC, then run the exact install steps from train_entry.py.
#
# What's baked:  apt packages, uv, LeRobot repo at pinned commit, venv synced
# What's NOT baked: model weights, datasets, FAST tokenizer (downloaded at runtime)

ARG DLC_REGISTRY=763104351884.dkr.ecr.us-east-1.amazonaws.com
FROM ${DLC_REGISTRY}/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# The commit pin matches train_entry.py; the full clone below checks out that
# exact commit, independent of the initial branch name.
ENV LEROBOT_REPO="https://github.com/allenai/lerobot.git" \
    LEROBOT_BRANCH="main" \
    LEROBOT_COMMIT="a4f15bf347dee7eb8a8c5f4a70a37f476b091113" \
    PYTHON_PIN="3.12" \
    WORK="/opt/vla"

# PyPI resilience
ENV PIP_RETRIES=5 \
    PIP_TIMEOUT=120 \
    PIP_DEFAULT_TIMEOUT=120 \
    UV_HTTP_RETRIES=8

WORKDIR ${WORK}

# --- APT packages ---
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git git-lfs build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Install uv ---
RUN pip install --no-cache-dir uv

# --- Clone LeRobot at pinned commit ---
RUN git clone --branch ${LEROBOT_BRANCH} ${LEROBOT_REPO} ${WORK}/lerobot && \
    cd ${WORK}/lerobot && \
    git checkout ${LEROBOT_COMMIT}

# --- Sync the LeRobot venv (uv sync --locked) ---
RUN cd ${WORK}/lerobot && \
    uv sync --locked --python ${PYTHON_PIN} --extra "dataset"

# --- Marker file for idempotence ---
RUN echo "molmoact2-baked:${LEROBOT_COMMIT}" > /opt/vla/.baked_env

# --- Sanity check ---
RUN cd ${WORK}/lerobot && \
    uv run python -c "import lerobot; print('MolmoAct2/LeRobot env OK')"
