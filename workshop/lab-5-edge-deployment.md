# Lab 5: Edge Deployment

**Goal:** Deploy the trained policy from Lab 4 to a physical robot via AWS IoT Greengrass
**Time:** 2 hours
**Cost:** ~$5-10 (Greengrass deployment + Jetson inference)

---

## What You're Building

Take the RL-refined model from Lab 4 and deploy it to the edge (NVIDIA Jetson) where it runs inference in real-time on a physical robot.

**The deployment pipeline:**

1. **Export model** — Convert PyTorch checkpoint to TensorRT (optimized for Jetson)
2. **Package as Greengrass component** — Docker container with model + inference runtime
3. **Deploy via IoT Greengrass** — Push to device fleet (one robot or hundreds)
4. **Run inference** — Model receives camera frames + joint states, outputs actions at ~200Hz

---

## Architecture

```
┌────────────────────────────────────────────────────┐
│  AWS Cloud                                         │
│                                                    │
│  S3 (model artifacts)                              │
│       │                                            │
│       ▼                                            │
│  IoT Greengrass ──── Component Registry            │
│       │                                            │
└───────┼────────────────────────────────────────────┘
        │  OTA Deploy
        ▼
┌────────────────────────────────────────────────────┐
│  Edge (NVIDIA Jetson Orin)                         │
│                                                    │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │ Inference Container│ │ ROS2 topics                  │ │
│  │ - TensorRT model  │  │ - /ur3/joint_states (sub)    │ │
│  │ - Camera pipeline │  │ - /ur3/wrist_camera/... (sub)│ │
│  │ - ~200Hz loop     │  │ - /ur3/joint_commands (pub)  │ │
│  └──────────────────┘  │ - /ur3/gripper_command (pub) │ │
│                        └──────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ Robot Hardware (UR3 + Robotiq gripper)        │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Lab 4 completed (trained + RL-refined model checkpoint in S3)
- NVIDIA Jetson Orin device (or Jetson AGX Xavier)
- Jetson flashed with JetPack 6.x
- AWS IoT Greengrass v2 installed on Jetson
- Robot connected to Jetson via ROS2

---

## Steps

> **Status:** the deploy tooling (export, component publish, fleet deploy) is
> implemented and dry-run-validated against AWS, but the **on-robot steps have not
> been run on real hardware** — they need a Jetson Orin + UR3 + camera. Treat the
> hardware steps as the documented procedure, not a validated result. Every script
> supports `--dry-run` so you can preview the exact AWS calls without a robot.

### Step 1: Export Model to TensorRT
```bash
# export.py loads a TorchScript checkpoint and writes ONNX + a TensorRT engine.
python training/scripts/export.py \
  --checkpoint ./model_exported/policy.pt \
  --output-onnx ./model_exported/policy.onnx \
  --output-trt ./model_exported/policy.trt \
  --target-device jetson-orin \
  --fp16
```

> **TorchScript first:** `export.py` uses `torch.jit.load`, so `--checkpoint` must be
> a **TorchScript** model. RL jobs save plain `model_*.pt` checkpoints, so convert first:
> ```bash
> python training/scripts/scriptify_policy.py \
>   --checkpoint model_49.pt --output policy.pt --obs-dim 12308 --action-dim 7
> # (use --inspect first to see the checkpoint's actor layer shapes)
> ```
> Then pass `policy.pt` to `export.py --checkpoint`.

### Step 2: Get the Inference Container

Both inference images are built in CodeBuild and pushed to ECR by the Foundation
stack — **nothing builds on your laptop or on the Jetson:**

- `physical-ai/inference:latest` — x86_64 (GPU PC / workstation testing), built on the standard fleet
- `physical-ai/inference:jetson` — aarch64 for NVIDIA Jetson, built natively on a **Graviton (ARM) CodeBuild fleet**

```bash
INFERENCE_URI=$(aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`InferenceRepoUri`].OutputValue' --output text)

# Both tags should appear (x86 = latest, Jetson = jetson):
aws ecr describe-images --repository-name physical-ai/inference \
  --query 'imageDetails[].imageTags' --output text

# Rebuild after changing the inference node or Dockerfile:
aws codebuild start-build --project-name physical-ai-inference-build         # x86
aws codebuild start-build --project-name physical-ai-inference-jetson-build  # Jetson (aarch64)
```

On the Jetson, you just **pull** the prebuilt aarch64 image from ECR (no on-device build):

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker pull <acct>.dkr.ecr.<region>.amazonaws.com/physical-ai/inference:jetson
```

### Step 3: Publish the Greengrass Component
```bash
# Uploads policy.trt to the models bucket + publishes a Greengrass component
# version. Account/region/bucket/image are resolved from your AWS identity.
# Preview first with --dry-run (prints the exact recipe, no AWS writes):
python edge/create_component.py --model ./model_exported/policy.trt --dry-run

# Then publish for real:
python edge/create_component.py --model ./model_exported/policy.trt --component-version 1.0.0
```

### Step 4: Deploy to the Robot Fleet
```bash
# Deploys the component to the fleet thing group (physical-ai-<env>-robots).
python edge/deploy_to_fleet.py --inference-version 1.0.0 --dry-run   # preview
python edge/deploy_to_fleet.py --inference-version 1.0.0             # deploy
python edge/deploy_to_fleet.py --status <DEPLOYMENT_ID>             # track rollout
```

> Steps 1, 3, 4 can be chained with `edge/deploy.sh --checkpoint <ckpt> [--dry-run]`
> (this is what the OSMO workflow's edge stage calls).

### Step 5: Validate on the Physical Robot
On the Jetson (this is the documented procedure — not yet hardware-validated):
```bash
# The inference node publishes joint + gripper commands and consumes camera + joint state.
ros2 topic hz /ur3/joint_commands     # target ~200 Hz (the node's configured rate)
ros2 topic echo /ur3/gripper_command  # 1.0 = close, 0.0 = open
ros2 topic list | grep /ur3           # /ur3/wrist_camera/image_raw, /ur3/joint_states, ...
```

> **Reality check on observations:** the inference node currently feeds a
> **placeholder (zeros) for object pose** — there's no perception/pose-estimation
> source wired in yet. So even a correct deploy will not grasp reliably until a
> pose source replaces that placeholder (`_build_observation` in
> `edge/ros2-workspace/src/ur3_inference/ur3_inference/ur3_inference_node.py`).

---

## ✅ Lab 5 Checkpoint

- [ ] Model exported to TensorRT format
- [ ] Inference container built and tested locally
- [ ] Greengrass component created and uploaded
- [ ] Deployment succeeded to target device
- [ ] Robot executing policy at ~200Hz from camera input
- [ ] Success rate on physical hardware measured

---

## Key Metrics to Track

| Metric | Sim (Lab 4) | Real (Lab 5) | Target |
|--------|-------------|--------------|--------|
| Success rate | ~95% | ~85-90% | >85% |
| Cycle time | 2.1s | 2.5-3.0s | <3.5s |
| Inference latency | <5ms | <10ms | <20ms |

The gap between sim and real (sim-to-real transfer) should be small if Lab 4's domain randomization was effective.

---

**Previous:** [← Lab 4: RL Refinement](lab-4-rl-refinement.md)
**Next:** [Lab 6: OSMO Orchestration →](lab-6-osmo-orchestration.md)
