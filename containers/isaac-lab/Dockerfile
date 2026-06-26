# Isaac Lab Training Container
#
# Extends the official NVIDIA Isaac Lab container with:
# - rl_games (PPO/SAC implementation)
# - Custom UR3 pick-and-place environment
# - Training and evaluation scripts
# - S3 integration for checkpoints and models
#
# Build:
#   docker build -t isaac-lab-training:latest .
#
# Run (headless training):
#   docker run --gpus all -v /data:/storage isaac-lab-training:latest \
#     python /workspace/training/scripts/train.py --config /workspace/training/configs/ppo_pick_place.yaml

# Base: Official Isaac Lab container from NGC (includes Isaac Sim + Isaac Lab)
FROM nvcr.io/nvidia/isaac-lab:2.1.0

# Install additional RL dependencies
# Isaac Lab's Python is at /isaac-sim/kit/python/bin/python3
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir \
    rl_games==1.6.1 \
    tensorboard>=2.14 \
    boto3>=1.34 \
    pyyaml>=6.0 \
    onnx>=1.15 \
    onnxruntime-gpu>=1.17

# Install TensorRT Python bindings (for export step)
RUN /isaac-sim/kit/python/bin/python3 -m pip install --no-cache-dir \
    tensorrt>=10.0

# Copy training entrypoints
# NOTE: adding batch-train-entrypoint.sh changes the CodeBuild source-asset
# hash (ContainerBuild construct) so deploying Foundation retriggers the build.
COPY containers/isaac-lab/sm-train-entrypoint.sh /opt/ml/code/sm-train-entrypoint.sh
COPY containers/isaac-lab/batch-train-entrypoint.sh /opt/ml/code/batch-train
COPY containers/isaac-lab/train_entrypoint.py /opt/ml/code/train_entrypoint.py
RUN chmod +x /opt/ml/code/sm-train-entrypoint.sh /opt/ml/code/batch-train

# Copy training code
COPY training/ /workspace/training/
COPY workflows/ /workspace/workflows/

# Copy custom UR3 environment. NOTE: not yet gym.register()-ed as an Isaac Lab
# task — see docs/ROADMAP.md (standalone RL feature). The validated RL path
# currently uses Isaac Lab's built-in tasks (e.g. Isaac-Velocity-Flat-Anymal-D-v0).
COPY training/envs/ /workspace/training/envs/

# Set working directory
WORKDIR /opt/ml/code

ENV ACCEPT_EULA=Y
ENV OMNI_ENV_PRIVACY_CONSENT=Y

# Make Isaac Sim's Python available as system python3
# (Isaac Lab container doesn't have system python — only /isaac-sim/kit/python/bin/python3)
RUN ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python3 \
    && ln -sf /isaac-sim/kit/python/bin/python3 /usr/local/bin/python

# SageMaker invokes the container as: docker run <image> train
# So we create /opt/ml/code/train as the script SageMaker expects
RUN ln -sf /opt/ml/code/sm-train-entrypoint.sh /opt/ml/code/train \
    && chmod +x /opt/ml/code/train

ENV PATH="/isaac-sim/kit/python/bin:/opt/ml/code:${PATH}"
ENV SAGEMAKER_PROGRAM=train

ENTRYPOINT []
CMD ["train"]
