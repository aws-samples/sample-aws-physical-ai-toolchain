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
# Isaac Lab's Python is inside the kit environment — use isaaclab.sh wrapper
RUN /opt/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
    rl_games==1.6.1 \
    tensorboard>=2.14 \
    boto3>=1.34 \
    pyyaml>=6.0 \
    onnx>=1.15 \
    onnxruntime-gpu>=1.17

# Install TensorRT Python bindings (for export step)
RUN /opt/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
    tensorrt>=10.0

# Copy training code
COPY training/ /workspace/training/
COPY workflows/ /workspace/workflows/

# Copy custom UR3 environment (registered as Isaac Lab task)
COPY training/envs/ /workspace/training/envs/

# Set working directory
WORKDIR /workspace

# Default: run training
ENTRYPOINT ["python"]
CMD ["/workspace/training/scripts/train.py", "--help"]
