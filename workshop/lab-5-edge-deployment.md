# Lab 5: Edge Deployment

**Goal:** Deploy the trained policy from Lab 4 to a physical robot via AWS IoT Greengrass
**Time:** 2 hours
**Cost:** ~$5-10 (Greengrass deployment + Jetson inference)

---

## What You're Building

Take the RL-refined model from Lab 3 and deploy it to the edge (NVIDIA Jetson) where it runs inference in real-time on a physical robot.

**The deployment pipeline:**

1. **Export model** — Convert PyTorch checkpoint to TensorRT (optimized for Jetson)
2. **Package as Greengrass component** — Docker container with model + inference runtime
3. **Deploy via IoT Greengrass** — Push to device fleet (one robot or hundreds)
4. **Run inference** — Model receives camera frames + joint states, outputs actions at 50Hz

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
│  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ Inference Container│  │ ROS2 Bridge           │ │
│  │ - TensorRT model  │  │ - /joint_states (sub) │ │
│  │ - Camera pipeline  │  │ - /cmd_vel (pub)      │ │
│  │ - 50Hz control loop│  │ - /camera/rgb (sub)   │ │
│  └──────────────────┘  └────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ Robot Hardware (UR3 + Robotiq gripper)        │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Lab 3 completed (trained + refined model checkpoint in S3)
- NVIDIA Jetson Orin device (or Jetson AGX Xavier)
- Jetson flashed with JetPack 6.x
- AWS IoT Greengrass v2 installed on Jetson
- Robot connected to Jetson via ROS2

---

## Steps (placeholder — to be detailed)

### Step 1: Export Model to TensorRT
```bash
python training/scripts/export.py \
  --checkpoint s3://$BUCKET/isaac-lab/output/model.tar.gz \
  --output ./model_exported/ \
  --format tensorrt \
  --precision fp16 \
  --target jetson-orin
```

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

### Step 3: Create Greengrass Component
```bash
# Package model + container as Greengrass component
python edge/create_component.py \
  --model ./model_exported/model.trt \
  --container ur3-inference:latest \
  --component-name PhysicalAI_UR3_Inference \
  --version 1.0.0
```

### Step 4: Deploy to Robot Fleet
```bash
# Deploy to a single robot (or group)
aws greengrassv2 create-deployment \
  --target-arn arn:aws:iot:us-east-1:$ACCOUNT:thinggroup/ur3-robots \
  --components '{
    "PhysicalAI_UR3_Inference": {"componentVersion": "1.0.0"}
  }'
```

### Step 5: Validate on Physical Robot
```bash
# Monitor inference on the robot
ssh jetson@<ROBOT_IP>
ros2 topic echo /inference/status
ros2 topic hz /cmd_vel  # Should show ~50Hz
```

---

## ✅ Lab 4 Checkpoint

- [ ] Model exported to TensorRT format
- [ ] Inference container built and tested locally
- [ ] Greengrass component created and uploaded
- [ ] Deployment succeeded to target device
- [ ] Robot executing policy at 50Hz from camera input
- [ ] Success rate on physical hardware measured

---

## Key Metrics to Track

| Metric | Sim (Lab 3) | Real (Lab 4) | Target |
|--------|-------------|--------------|--------|
| Success rate | ~95% | ~85-90% | >85% |
| Cycle time | 2.1s | 2.5-3.0s | <3.5s |
| Inference latency | <5ms | <10ms | <20ms |

The gap between sim and real (sim-to-real transfer) should be small if Lab 3's domain randomization was effective.

---

**Previous:** [← Lab 4: RL Refinement](lab-4-rl-refinement.md)
**Next:** [Lab 6: OSMO Orchestration →](lab-6-osmo-orchestration.md)
