# Isaac Sim — Headless Mode Guide

Run Isaac Lab RL training and policy evaluation in headless mode (no GUI rendering). This is how training runs at scale on AWS Batch and SageMaker.

---

## Overview

Headless mode runs the full physics simulation and policy inference without rendering pixels to a display. This is faster and doesn't require X11/display access.

---

## Prerequisites

On the Isaac Sim workstation:

```bash
# Install AWS CLI (if not already)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip /tmp/awscliv2.zip -d /tmp/
sudo /tmp/aws/install

# Login to ECR and pull the Isaac Lab container
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-2
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker pull $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/physical-ai/isaac-lab:latest
```

---

## Run Headless RL Training

Train a policy from scratch on the workstation (same as Batch, but interactive):

```bash
docker run --shm-size=60g --gpus all --rm -it \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ACCEPT_EULA=Y \
  -w /workspace/isaaclab \
  804152302157.dkr.ecr.us-east-2.amazonaws.com/physical-ai/isaac-lab:latest \
  bash -c "./isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Anymal-D-v0 \
    --num_envs 64 --max_iterations 50 --headless"
```

You'll see training logs:
```
Learning iteration 1/50
Computation: 31367 steps/s (collection: 0.698s, learning 0.086s)
Mean reward: -0.90
Mean episode length: 26.33
```

---

## Run Headless Policy Playback (from checkpoint)

Load a trained checkpoint and run the policy headlessly:

```bash
# Download checkpoint from S3
mkdir -p ~/checkpoints
aws s3 cp s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/isaac-lab/<JOB_NAME>/output/model.tar.gz ~/checkpoints/ --region us-east-2
cd ~/checkpoints && tar -xzf model.tar.gz

# Run the policy headlessly
docker run --shm-size=60g --gpus all --rm -it \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ACCEPT_EULA=Y \
  -v ~/checkpoints:/workspace/checkpoints \
  -w /workspace/isaaclab \
  804152302157.dkr.ecr.us-east-2.amazonaws.com/physical-ai/isaac-lab:latest \
  bash -c "./isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Anymal-D-v0 \
    --checkpoint /workspace/checkpoints/logs/rsl_rl/anymal_d_flat/<TIMESTAMP>/model_99.pt \
    --num_envs 4 --headless"
```

---

## Run Interactive Shell (for debugging)

Enter the container interactively to explore, debug, or run custom scripts:

```bash
docker run --shm-size=60g --gpus all --rm -it \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ACCEPT_EULA=Y \
  -v ~/checkpoints:/workspace/checkpoints \
  -w /workspace/isaaclab \
  804152302157.dkr.ecr.us-east-2.amazonaws.com/physical-ai/isaac-lab:latest \
  bash
```

Inside the container:
```bash
# Train
./isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 64 --max_iterations 50 --headless

# Play a checkpoint
./isaaclab.sh -p /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --checkpoint /workspace/checkpoints/logs/rsl_rl/anymal_d_flat/<TIMESTAMP>/model_99.pt \
  --num_envs 4 --headless

# List available tasks
./isaaclab.sh -p scripts/tools/list_envs.py
```

---

## Important Docker Flags

| Flag | Why |
|------|-----|
| `--shm-size=60g` | Isaac Sim requires large shared memory for rendering/physics |
| `--gpus all` | GPU access for CUDA physics + rendering |
| `-e NVIDIA_DRIVER_CAPABILITIES=all` | Expose all GPU capabilities to the container |
| `-e ACCEPT_EULA=Y` | Accept NVIDIA EULA |
| `-w /workspace/isaaclab` | Set working directory for `isaaclab.sh` |

---

## Available Tasks

| Task | Type | Description |
|------|------|-------------|
| `Isaac-Velocity-Flat-Anymal-D-v0` | Locomotion | Quadruped walking on flat terrain |
| `Isaac-Velocity-Rough-Anymal-D-v0` | Locomotion | Quadruped on rough terrain |
| `Isaac-Reach-Franka-v0` | Manipulation | Franka arm reaching |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Segmentation fault` with `--video` | Known issue with video rendering in Docker. Use `--headless` without `--video` |
| `GLFW initialization failed` | Expected in headless mode — container has no display. Physics still runs. |
| `Warp CUDA error: cuDeviceGetUuid` | Non-fatal warning. Training proceeds normally. |
| `unknown runtime: nvidia` | Remove `--runtime=nvidia`, use `--gpus all` instead |
| Container OOM killed | Increase `--shm-size` (try `60g` or higher) |

---

## Next Steps

- For GUI visualization (requires `nvcr.io/nvidia/isaac-lab:3.0.0-beta2`), see [GUI Mode Guide](gui-mode-guide.md)
- For scale-out training on AWS Batch, see [Isaac Lab Batch Guide](../isaac-lab-on-aws/batch-rl-training-guide.md)
- For SageMaker training, see [Isaac Lab SageMaker Guide](../isaac-lab-on-aws/sagemaker-rl-training-guide.md)
