# Physical AI Toolkit

## Build an End-to-End Robot Learning Pipeline on AWS

This workshop takes you from raw teleoperation recordings to a deployed robot policy — running on real hardware — using AWS infrastructure-as-code and the most widely adopted open-source tools in physical AI (GR00T, Isaac Sim, Isaac Lab, Cosmos, LeRobot, Hugging Face, ROS 2, PyTorch).

You'll build a complete **pick-and-place** pipeline: the most common industrial manipulation task and the starting point for most robotics teams. By the end, you'll have a trained, refined, and deployable policy that can pick objects from a bin and place them at a target location.

---

## What You'll Build

```
Teleop Data  →  Imitation Learning  →  World Generation   →  RL Training      →  Export .onnx
(Lab 1)          (GR00T on              (Cosmos V2V)         (Isaac Lab on       (ready for
                  SageMaker)                                  SageMaker)          deployment)
```

A production-grade Physical AI pipeline with:

- **Infrastructure as Code** — everything deploys via `cdk deploy`. Reproducible, versionable, teardown-able.
- **Open-source toolchain** — GR00T (foundation model), Isaac Lab (RL simulation), Cosmos (world generation), LeRobot (data format), ROS 2 (robot middleware), PyTorch (training), Docker (containers) — all open source
- **AWS services** — SageMaker (training), S3 (data), ECR (containers), CodeBuild (CI), EC2 (Cosmos generation)
- **GPU-accelerated** — NVIDIA GPUs for parallel simulation (4096 robots simultaneously) and photorealistic world generation (Cosmos 3 on H100)

---

## What You'll Build and Learn

| Lab | What You Learn | Key Skill | Time | Depends On |
|-----|---------------|-----------|------|-----------|
| [Lab 0](lab-0-prerequisites.md) | Environment setup and infrastructure deployment | CDK, AWS account configuration | 30 min | — |
| [Lab 1](lab-1-train-groot.md) | Train a robot policy from demonstration data | GR00T fine-tuning, SageMaker Pipelines | 2 hrs | Lab 0 |
| [Lab 2](lab-2-isaac-workstation.md) | Visual development and debugging in simulation | Isaac Sim, GPU remote desktop | 30 min | Lab 0 |
| [Lab 3](lab-3-cosmos-world-generation.md) | Generate synthetic demonstrations with Cosmos (Predict) | Cosmos 3 Super, World Foundation Models | 1-2 hrs | Lab 1 |
| [Lab 4](lab-4-cosmos-transfer.md) | Restyle existing training data with Cosmos (Transfer) | Cosmos Transfer 2.5, data augmentation | 1-2 hrs | Lab 1 |
| [Lab 5](lab-5-rl-refinement-with-isaac.md) | Train an RL policy in simulation (standalone) | Isaac Lab, PPO, domain randomization | 3 hrs | Lab 0 |
| [Lab 6](lab-6-osmo-orchestration.md) | Orchestrate the full pipeline for production | NVIDIA OSMO, EKS, Kueue | 2-3 hrs | Labs 1-5 |

---

## The Use Case: Pick and Place

We use pick-and-place because:

- It's the **#1 most common** industrial robot task (bin picking, kitting, palletizing)
- It exercises the **full pipeline** — perception, planning, grasping, placement
- It's **simple enough to learn in a workshop** but hard enough to require real AI
- It has **clear success metrics** — did the object get picked up and placed correctly?
- It **transfers to other tasks** — the same pipeline works for assembly, sorting, inspection

The reference implementation uses a **UR3** arm with a **Robotiq 2F-85** gripper — the most popular collaborative robot arm in research and industry. But the pipeline is robot-agnostic.

---

## No Hardware Required

This entire workshop runs in the cloud. You don't need a physical robot to complete Labs 0-4. The simulation (Isaac Lab) provides a physically accurate UR3 environment where you can develop, train, and validate policies.

**If you have a UR3 arm:** The trained policy can be exported as `.onnx` and deployed to edge hardware manually. Automated edge deployment (Greengrass + Jetson) is planned as a future lab.

**If you have different hardware:** You can adapt the pipeline by:
1. Recording your own teleoperation data (any robot + camera setup)
2. Converting to LeRobot v2 format (scripts provided)
3. Creating an Isaac Lab environment matching your robot's URDF
4. Running the same pipeline (Labs 1-6) with your data

The toolchain handles UR3 out of the box. For other robots, you bring your URDF and teleop data — everything else stays the same.

---

## Bring Your Own Data

The pipeline accepts teleoperation recordings in **LeRobot v2 format** (Parquet + MP4):

```
your-dataset/
├── data/
│   └── chunk-000/
│       └── file-000.parquet    # Joint states, actions, timestamps
├── videos/
│   └── observation.images.wrist/
│       └── chunk-000/
│           └── file-000.mp4    # Camera recordings
└── meta/
    ├── info.json               # Dataset metadata (robot type, fps, features)
    ├── modality.json           # GR00T embodiment config
    └── tasks.parquet           # Task descriptions
```

If your recordings are in a different format (ROS bags, Zarr, CSV), conversion scripts are provided:
```bash
python training/groot/convert_dataset.py --input ./my_rosbags/ --output ./lerobot_dataset/ --robot-type ur3
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  AWS Account                                                             │
│                                                                         │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │   S3    │  │   ECR    │  │SageMaker │  │   EC2    │               │
│  │Datasets │  │Containers│  │Training  │  │ Cosmos   │               │
│  │Models   │  │          │  │Pipelines │  │ GPU Gen  │               │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘               │
│       │            │            │              │                       │
│       └────────────┴────────────┴──────────────┘                       │
│                              │                                          │
│                    CDK (Infrastructure as Code)                          │
│                    One command: `cdk deploy`                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Estimated Costs

Running the full workshop in your own AWS account:

| Component | Cost | Notes |
|-----------|------|-------|
| Lab 1 (GR00T training) | ~$15-30 | ml.g5.12xlarge for 1-2 hours |
| Lab 2 (Workstation) | ~$3.00/hr | Stop when not in use |
| Lab 3 (Cosmos) | ~$300-500 | Capacity Block (p5.48xlarge, ~$37/hr) |
| Lab 4 (RL training) | ~$10-30 | ml.g5.xlarge for 2-4 hours |
| Lab 5 (RL training) | ~$10-30 | ml.g5.xlarge for 2-4 hours |
| Lab 6 (OSMO/EKS) | ~$50-100 | EKS cluster + GPU nodes |
| **Total (Labs 0-4, no EKS)** | **~$50-100** | Most common path |
| **Total (all labs)** | **~$150-250** | Full pipeline with OSMO |

All infrastructure can be torn down with `cdk destroy` when done.

---

## Getting Started

This workshop is based on a public repository that you clone and run self-paced in your own AWS account:

```bash
# Clone the toolchain
git clone [REPO_URL_HERE]
cd aws-physical-ai-toolchain

# Deploy infrastructure (takes ~5 minutes)
cd cdk && npm install
npx cdk deploy --all --context mode=simple

# Start with Lab 0 to verify everything is ready
```

→ **[Begin with Lab 0: Prerequisites](lab-0-prerequisites.md)**

---

## Dependencies

Labs 2, 3, and 4 can run in parallel with Lab 1. Labs 3 and 4 take Lab 1's dataset as input (wrist camera MP4s) but can also use sim renders from Lab 2. **Lab 5 is standalone RL** — it trains a policy in simulation from scratch and needs only the Foundation stack (the `isaac-lab` image in ECR); it does **not** require Lab 1 (GR00T) or Labs 3-4 (Cosmos). RL and imitation learning (Lab 1) are two independent ways to obtain a policy.

---

## Who This Is For

- **Robotics engineers** building manipulation or locomotion policies
- **ML engineers** moving from cloud training to physical deployment
- **Solutions architects** designing Physical AI platforms for customers
- **Anyone curious** about how robots learn from human demonstrations and simulation

**Prerequisites:** Familiarity with AWS (CLI, console), Python, and basic ML concepts. No robotics experience required — the labs explain the domain concepts as you go.


---

## Physical AI Terminology

New to robotics and Physical AI? Here's what the key terms mean.

### The Basics

| Term | Plain English |
|------|--------------|
| **Physical AI** | AI that moves things in the real world — robot arms picking objects, drones navigating, humanoids walking. Unlike chatbots (text in, text out), Physical AI takes camera images in and produces motor commands out. |
| **Policy** | The trained "brain" of the robot. A neural network file (`.pt`) that takes in sensor data (camera image + joint angles) and outputs actions (move arm here, close gripper). Same thing as a "model" — roboticists say "policy" because it makes decisions. |
| **Teleoperation (Teleop)** | A human remotely controlling a robot to demonstrate a task. You move the robot through the task while it records everything — what it saw (camera) and what it did (joint movements). These recordings become training data. |
| **Episode** | One complete task demonstration from start to finish. "Pick up the cube and place it in the bin" = one episode. A training dataset might contain 50-200 episodes. |
| **Embodiment** | The physical robot body. A UR3 arm has 6 joints. A humanoid has 30+. Policies are embodiment-specific — a policy trained for a UR3 won't work on a different arm without retraining. |

### Training Approaches

| Term | Plain English |
|------|--------------|
| **Imitation Learning** | "Learn by watching." You show the robot 50 demonstrations, then a model (GR00T) learns to copy those behaviors. Fast to get working but limited to what you demonstrated. |
| **Reinforcement Learning (RL)** | "Learn by practice." The robot tries the task millions of times in simulation, getting a score each time (+1 for success, -0.5 for dropping). Through trial and error it discovers strategies better than what any human showed it. |
| **Fine-tuning** | Taking a pre-trained model (like GR00T, already trained on millions of robot examples) and training just the last few layers on your specific data. Much cheaper than training from scratch. Like customizing a pre-built app instead of writing from zero. |
| **PPO** | Proximal Policy Optimization. The standard RL algorithm for robotics. You don't need to understand the math — just know it's what Isaac Lab uses to improve the policy through simulated practice. |
| **Domain Randomization** | During training in sim, randomly vary everything: object positions, lighting, colors, camera angles. Forces the policy to work regardless of conditions. Like training a self-driving car in rain, snow, and sun simultaneously. |
| **Sim-to-Real Transfer** | The gap between simulation and reality. Physics in sim is approximate — objects slide differently, lighting looks different. Domain randomization and Cosmos close this gap so sim-trained policies work on real robots. |

### Models

| Term | Plain English |
|------|--------------|
| **GR00T** | NVIDIA's robot foundation model (Generalist Robot 00 Technology). A 3-billion-parameter neural network pre-trained on diverse robot data. You fine-tune it on your specific robot + task with 50-200 demonstrations. Outputs motor commands from camera images + language instructions. |
| **VLA (Vision-Language-Action)** | A single model that sees images, understands language, and outputs motor commands. GR00T is a VLA. You say "pick up the red cube," it sees the camera, it moves the arm. |
| **Foundation Model** | A large pre-trained model you customize for your task. Like GPT is a foundation model for text, GR00T is a foundation model for robot control. You never train these from scratch — you fine-tune. |
| **Action Chunking** | Instead of deciding one movement at a time, the model predicts the next 16 movements as a chunk. Produces smoother, more natural robot motion. |

### Simulation

| Term | Plain English |
|------|--------------|
| **Isaac Sim** | NVIDIA's robot simulation platform. A physics engine that can simulate gravity, friction, collisions, and cameras realistically. Think of it as a video game engine purpose-built for robots. |
| **Isaac Lab** | A training framework that runs on top of Isaac Sim. Provides the RL training loop — creates thousands of parallel robot copies, collects experience, updates the policy. You write your task definition here. |
| **Cosmos** | NVIDIA's World Foundation Model. Takes sim-rendered or real-world video and produces photorealistic variations — applies realistic materials, lighting, and textures while preserving geometry and motion. Used to augment training data so policies transfer to real hardware. |
| **Parallel Environments** | Running 4096 copies of the same robot simultaneously on one GPU. Each practices independently. In one second of real time, the robot accumulates days of practice. This is why RL training takes hours instead of years. |
| **Headless** | Running the simulator without displaying graphics. All physics still work, but no screen rendering. Faster because the GPU focuses on computation instead of pixels. Used during training. |

### Hardware & Deployment

| Term | Plain English |
|------|--------------|
| **UR3** | A Universal Robots 6-joint collaborative arm. 3kg payload, ~500mm reach. The most popular robot arm in research and light manufacturing. Our reference robot. |
| **Jetson** | NVIDIA's edge GPU board (Orin, Xavier). A small computer with a GPU designed to run AI models inside robots. Runs trained policies at 50-200 Hz. Think "GPU for your robot's brain." |
| **TensorRT** | NVIDIA's model compiler. Takes your trained PyTorch model and optimizes it for specific hardware (Jetson), making it run 10-100x faster. Necessary for real-time robot control. |
| **ROS 2** | Robot Operating System 2. Not actually an OS — it's middleware that connects robot components. Camera nodes publish images, your policy node subscribes to images and publishes motor commands. Like message queues but for robots. |
| **Greengrass** | AWS IoT Greengrass. Deploys and manages software on robots. When you have a new policy, Greengrass pushes it to your robot fleet over-the-air. Like deploying a Lambda update, but to physical hardware. |
| **NICE DCV** | AWS remote desktop protocol. Streams GPU-rendered graphics from a cloud instance to your laptop browser. How you interact with Isaac Sim visually without a local GPU. |

### Orchestration

| Term | Plain English |
|------|--------------|
| **OSMO** | NVIDIA's workflow orchestrator for Physical AI. Chains multiple stages (train → simulate → evaluate → deploy) into one automated pipeline. Handles GPU scheduling and retry logic. Like CI/CD but for robot training. |
| **SageMaker Pipeline** | AWS's ML workflow orchestration. We use this for the simpler GR00T training path (Lab 1). OSMO adds value when you need Isaac Lab simulation stages and complex multi-GPU scheduling. |

### Data Formats

| Term | Plain English |
|------|--------------|
| **LeRobot v2** | HuggingFace's standard format for robot training data. Parquet files for numbers (joint angles, actions) + MP4 files for camera video. GR00T reads this format directly. |
| **Zarr** | A chunked array format for storing raw robot recordings before conversion. If you record teleop data, it likely starts as Zarr and gets converted to LeRobot. |
| **URDF** | Universal Robot Description Format. An XML file describing your robot's geometry — how many joints, how long the links, what are the limits. Every simulator needs this to model your robot. |
| **USD** | Universal Scene Description. A 3D scene format (originally from Pixar). Isaac Sim uses USD for all scene content — robots, objects, environments. |

---

For the complete glossary with cloud analogies and deeper explanations, see [docs/glossary.md](../docs/glossary.md).
