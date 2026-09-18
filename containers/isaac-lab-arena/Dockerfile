ARG ARENA_BASE_IMAGE
FROM ${ARENA_BASE_IMAGE}

# This image runs a finite evaluation job, not Isaac Sim's interactive service.
# The inherited AppReady marker check can report unhealthy during valid rollouts.
# Readiness probes, the rollout watchdog and process/validation results below
# determine batch-job success; do not replace them with an always-green check.
HEALTHCHECK NONE

# The base image ships Isaac Sim's system Python with numpy <2.0, torch, and
# transformers intact. Do NOT pip-install into this Python -- transitive deps
# (albumentations -> opencv-python-headless) collide with Isaac Sim's omni.*
# extension namespace, causing omni.pip.compute / omni.isaac.core_archive /
# omni.kit.loop import failures and a segfault during SimulationApp init.
# All GR00T-specific deps (diffusers, accelerate, huggingface-hub) live in
# the baked venvs below (/opt/gr00t-n17/.venv, /opt/gr00t-n16/.venv).

COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/gr00t/_verify_gr00t.py /tmp/_verify_gr00t.py
RUN /isaac-sim/python.sh /tmp/_verify_gr00t.py

COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/_shared/docker_entrypoint_multi.sh /workspace/docker_entrypoint_multi.sh
ENTRYPOINT ["/bin/bash", "/workspace/docker_entrypoint_multi.sh"]
COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/gr00t/eval_entry.py /workspace/eval_entry.py
COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/_shared/digest.py /workspace/digest.py
COPY isaac-lab-arena-on-aws/src/vla_pipeline/common/validator.py /workspace/validator.py
# eval_entry.extract_checkpoint imports capped_reader flatly. _required_zip_members() ships it to
# the build CONTEXT, which is not the same as landing it in the image: without this line the import
# raises ModuleNotFoundError only when a real job reaches extraction -- after the image has pulled
# and the checkpoint has downloaded. tests/test_shared_module_shipping.py parses these COPY lines.
COPY isaac-lab-arena-on-aws/src/vla_pipeline/common/capped_reader.py /workspace/capped_reader.py
COPY isaac-lab-arena-on-aws/src/vla_pipeline/common/source_identity.py /workspace/source_identity.py
COPY isaac-lab-arena-on-aws/src/vla_pipeline/common/checkpoint_compat.py /workspace/checkpoint_compat.py
COPY isaac-lab-arena-on-aws/entrypoints/train/gr00t/defaults.json /workspace/defaults.json
COPY isaac-lab-arena-on-aws/entrypoints/train/gr00t/arena_gr1_data_config.py /workspace/arena_gr1_data_config.py
# Seeding wrapper for the GR00T inference server. eval_entry launches the pinned server
# THROUGH this file so the evaluation seed binds the server process's RNG; without it the
# policy's inference noise is unbound while client-side seeding makes the scene sequence
# look reproducible. The path must match SEEDED_SERVER_ENTRY in eval_entry.py.
COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py /workspace/gr00t_seeded_server.py

COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/gr00t/gr00t_n17_n16_action_adapter.py /tmp/gr00t_n17_n16_action_adapter.py
COPY isaac-lab-arena-on-aws/entrypoints/eval/isaac_arena/gr00t/gr00t_n17_n16_action_adapter.pth /tmp/gr00t_n17_n16_action_adapter.pth
RUN SP=$(/isaac-sim/python.sh -c "import site; print(site.getsitepackages()[0])") && \
    cp /tmp/gr00t_n17_n16_action_adapter.py "$SP/gr00t_n17_n16_action_adapter.py" && \
    cp /tmp/gr00t_n17_n16_action_adapter.pth "$SP/gr00t_n17_n16_action_adapter.pth" && \
    /isaac-sim/python.sh -c "import gr00t_n17_n16_action_adapter"

# ---------------------------------------------------------------------------
# R1: Bake GR00T server venvs at image build time.
#
# The connector image previously installed all GR00T dependencies at runtime
# (clone repo, strip deepspeed, uv lock, uv sync) every time a SageMaker eval
# job started. This made each run non-reproducible (network-dependent resolution,
# varying transitive versions) and added ~10 minutes of setup to every job.
#
# Now both venvs (N1.7 and N1.6) are built at image time with pinned upstream
# pyproject.toml contracts. Runtime eval_entry.py requires them to exist and
# fails immediately if absent. The deepspeed dep is stripped at build time
# (inference-only venv; deepspeed is training-only).
# ---------------------------------------------------------------------------
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git git-lfs curl ca-certificates build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages uv==0.12.1

# --- N1.7 GR00T venv (376ba890, requires-python >=3.12,<3.13) ---
ARG GR00T_N17_COMMIT=376ba890cff8c9de64d71d982772a9c36185fdd7
RUN git lfs install --skip-repo && \
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/Isaac-GR00T.git /opt/gr00t-n17 && \
    git -C /opt/gr00t-n17 checkout ${GR00T_N17_COMMIT} && \
    test "$(git -C /opt/gr00t-n17 rev-parse HEAD)" = "${GR00T_N17_COMMIT}" && \
    git -C /opt/gr00t-n17 lfs pull --include "scripts/deployment/**/wheels/*.whl"

# Allow transient dependency-host failures time to recover. Keep the upstream
# lock and hash checks; an exhausted retry budget still fails the image build.
RUN cd /opt/gr00t-n17 && \
    UV_HTTP_RETRIES=10 UV_HTTP_TIMEOUT=120 \
    uv sync --frozen --no-dev --no-install-package deepspeed --python 3.12 && \
    test -f /opt/gr00t-n17/.venv/bin/python

RUN /opt/gr00t-n17/.venv/bin/python -c \
    "import gr00t; from gr00t.model import Gr00tN1d7Pipeline; print('N1.7 venv OK')"

# --- N1.6 GR00T venv (5dc80c4, requires-python >=3.10,<3.13) ---
ARG GR00T_N16_COMMIT=5dc80c4afd726b34faad1d8f7e007a13b34e4c88
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/Isaac-GR00T.git /opt/gr00t-n16 && \
    git -C /opt/gr00t-n16 checkout ${GR00T_N16_COMMIT} && \
    test "$(git -C /opt/gr00t-n16 rev-parse HEAD)" = "${GR00T_N16_COMMIT}" && \
    git -C /opt/gr00t-n16 lfs pull --include "scripts/deployment/**/wheels/*.whl"

RUN cd /opt/gr00t-n16 && \
    UV_HTTP_RETRIES=10 UV_HTTP_TIMEOUT=120 \
    uv sync --frozen --no-dev --no-install-package deepspeed --python 3.12 && \
    test -f /opt/gr00t-n16/.venv/bin/python

RUN /opt/gr00t-n16/.venv/bin/python -c \
    "import gr00t; from gr00t.data.embodiment_tags import EmbodimentTag; \
     assert hasattr(EmbodimentTag, 'GR1'); \
     import gr00t.eval.run_gr00t_server; \
     print('N1.6 venv OK: EmbodimentTag.GR1 + run_gr00t_server')"
