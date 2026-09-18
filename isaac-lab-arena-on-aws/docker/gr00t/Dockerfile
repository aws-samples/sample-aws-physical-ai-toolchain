# GR00T prebuilt image: bakes the full stack including isolated FFmpeg-7.
# This is the hardest build (ffmpeg-7 from source, torchcodec, uv venv).
#
# What's baked:  apt packages, isolated ffmpeg-7 prefix, uv, GR00T repo at
#                pinned commit, venv synced, torchcodec working
# What's NOT baked: model weights, datasets, HF token (fetched from Secrets Manager)

ARG DLC_REGISTRY=763104351884.dkr.ecr.us-east-1.amazonaws.com
# Ubuntu 22.04 (glibc 2.35) + CUDA 12.4 DLC. REQUIRED for the n16 path: NVIDIA's
# Isaac-GR00T n1.6.1-release pins flash-attn 2.7.4.post1 (prebuilt wheel needs
# glibc >= 2.32) + torch 2.7.1 (CUDA >= 12.4 kernels); the old 20.04/cu121 base
# (glibc 2.31) failed with "GLIBC_2.32 not found" from flash_attn_2_cuda.so.
# The runtime re-clones GR00T at the selected commit + uv-syncs its own 3.12 venv,
# so the DLC's system python/torch are irrelevant -- only glibc/CUDA runtime matter.
# n17 (376ba890) remains compatible (glibc/CUDA are backward-compatible).
FROM ${DLC_REGISTRY}/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

# Pins (must match train_entry.py / defaults.json exactly)
ENV GR00T_REPO="https://github.com/NVIDIA/Isaac-GR00T.git" \
    GR00T_COMMIT="376ba890cff8c9de64d71d982772a9c36185fdd7" \
    PYTHON_PIN="3.12" \
    WORK="/opt/vla" \
    FFMPEG_PREFIX="/opt/vla/ffmpeg7"

# PyPI/UV resilience
ENV PIP_RETRIES=5 \
    PIP_TIMEOUT=120 \
    PIP_DEFAULT_TIMEOUT=120 \
    UV_HTTP_RETRIES=8

WORKDIR ${WORK}

# --- APT packages (including ffmpeg build deps) ---
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git git-lfs build-essential nasm yasm \
        libx264-dev libx265-dev libdav1d-dev pkg-config && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Install uv ---
RUN pip install --no-cache-dir uv

# --- Isolated FFmpeg-7 prefix ---
# Keep the pinned FFmpeg version isolated from the DLC's base libraries.
# LIBERO's LeRobot videos use AV1; importing TorchCodec alone does not exercise
# decoding. The shared recipe also supports a small repair of an existing image.
COPY build_ffmpeg.sh /tmp/build_ffmpeg.sh
RUN bash /tmp/build_ffmpeg.sh && rm /tmp/build_ffmpeg.sh

# Set LD_LIBRARY_PATH for torchcodec to find ffmpeg-7 libs
ENV LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib:${LD_LIBRARY_PATH}"

# --- Clone GR00T at pinned commit ---
# NOTE: GR00T_COMMIT must be set correctly before build (from defaults.json)
RUN git clone ${GR00T_REPO} ${WORK}/gr00t && \
    cd ${WORK}/gr00t && \
    git checkout ${GR00T_COMMIT}

# --- Sync the GR00T venv ---
RUN cd ${WORK}/gr00t && \
    uv sync --locked --python ${PYTHON_PIN}

# --- Verify torchcodec works with our ffmpeg-7 (REQUIRED, not advisory) ---
# This used to end in `|| echo "WARNING: ..."`, so a failed import produced a
# SUCCESSFUL build -- and the baked marker below was then written anyway, certifying an
# environment whose required import had failed. That is not a warning-level condition:
# the whole ffmpeg-7-from-source stage above exists to make this import work, and an
# expensive GPU job would be the thing to discover it was broken.
COPY check_video_decode.py av1-two-frames.mp4 /tmp/video-check/
RUN ${WORK}/gr00t/.venv/bin/python /tmp/video-check/check_video_decode.py && \
    rm -rf /tmp/video-check
LABEL dev.physical-ai.av1-decoding="verified"

# --- LIBERO client environment (REQUIRED for the LIBERO evaluator) ---
# The root `uv sync` above does NOT create this. Upstream's setup_libero.sh creates a
# SEPARATE `libero_uv/.venv` interpreter, and the evaluator requires it unconditionally.
# Without this step a "baked" image had the GR00T server environment but no LIBERO client,
# so the evaluator either reinstalled everything at runtime on a GPU node or -- once it
# started trusting the marker and skipping setup -- failed outright on the missing
# interpreter. Baking it here is what makes the marker's claim true.
#
# CMAKE_POLICY_VERSION_MINIMUM is required, not optional. LIBERO pulls egl-probe==1.0.2, whose
# CMakeLists.txt declares `cmake_minimum_required` below 3.5. CMake 4.x REMOVED compatibility with
# those declarations and errors out rather than warning, so the wheel build fails and takes the whole
# layer with it -- which is why this step had never completed successfully since it was added. The
# variable tells CMake to accept the old declaration; it does not change what is compiled, and it is
# the mechanism CMake provides for exactly this case. Pinning an older CMake instead would mean
# maintaining a second toolchain in the image for one transitive dependency.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN cd ${WORK}/gr00t && \
    bash gr00t/eval/sim/LIBERO/setup_libero.sh

# Verify the client interpreter exists and runs, before anything claims the capability.
RUN test -x ${WORK}/gr00t/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python && \
    ${WORK}/gr00t/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python -c \
      "import libero; print('LIBERO client env OK')"

# --- Marker file for idempotence ---
# Written only AFTER the verifications above succeed, so the marker cannot certify a
# broken environment. It records WHICH CAPABILITIES were verified, not merely which commit
# was cloned: the evaluator checks for the libero-client capability specifically, so an
# image missing it fails with that named cause rather than an opaque missing-interpreter
# error.
# Composed ONCE, into an ARG, then written to BOTH the on-disk marker and an OCI label. The two must
# never be able to disagree: the marker is what the evaluator reads at runtime, inside the container,
# after a GPU node has already been paid for; the label is what a launcher can read from ECR BEFORE
# submitting anything. Two independently-written strings would drift, and the drift would be invisible
# until an image passed the cheap submit-time check and then failed the expensive runtime one.
ARG BAKED_CAPABILITIES="gr00t-baked:${GR00T_COMMIT}:torchcodec-verified:libero-client-verified"
RUN echo "${BAKED_CAPABILITIES}" > /opt/vla/.baked_env
LABEL dev.physical-ai.baked-capabilities="${BAKED_CAPABILITIES}"

# --- Sanity check ---
RUN cd ${WORK}/gr00t && \
    uv run python -c "print('GR00T env OK')"
