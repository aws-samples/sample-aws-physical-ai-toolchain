# Physical AI Toolchain Workshop

## Build an End-to-End Robot Learning Pipeline on AWS

This workshop takes you from raw teleoperation recordings to a deployed robot policy — running on real hardware — using AWS infrastructure-as-code and the NVIDIA Physical AI stack.

You'll build a complete **pick-and-place** pipeline: the most common industrial manipulation task and the starting point for most robotics teams. By the end, you'll have a trained, refined, and deployable policy that can pick objects from a bin and place them at a target location.

---

## What You'll Build

```
Teleop Data  →  Imitation Learning  →  World Generation  →  RL Refinement  →  Edge Deployment
(Lab 1)          (GR00T on              (Cosmos NIM)         (Isaac Lab on       (Greengrass +
                  SageMaker)                                  SageMaker)          Jetson)
```

A production-grade Physical AI pipeline with:

- **Infrastructure as Code** — everything deploys via `cdk deploy`. Reproducible, versionable, teardown-able.
- **NVIDIA toolstack** — GR00T N1 (foundation model), Isaac Lab (physics simulation), Cosmos (world generation), OSMO (orchestration), TensorRT (edge inference)
- **AWS services** — SageMaker (training), S3 (data), ECR (containers), CodeBuild (CI), IoT Greengrass (edge deployment), EKS (orchestration)
- **Open source** — LeRobot (data format), ROS2 (robot middleware), PyTorch (training), Docker (containers)

---

## What You'll Learn

| Lab | What You Learn | Key Skill |
|-----|---------------|-----------|
| Lab 0 | Environment setup and infrastructure deployment | CDK, AWS account configuration |
| Lab 1 | Train a robot policy from demonstration data | GR00T fine-tuning, SageMaker Pipelines |
| Lab 2 | Visual development and debugging in simulation | Isaac Sim, GPU remote desktop |
| Lab 3 | Generate photorealistic training environments | Cosmos NIM API, domain gap |
| Lab 4 | Improve policy robustness via reinforcement learning | Isaac Lab, PPO, domain randomization |
| Lab 5 | Deploy to physical hardware at the edge | TensorRT, Greengrass, Jetson |
| Lab 6 | Orchestrate the full pipeline for production | NVIDIA OSMO, EKS, Kueue |

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

**If you have a UR3 arm:** Lab 5 walks you through deploying to real hardware via Greengrass + Jetson. The sim-trained policy runs directly on the physical robot.

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
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   S3    │  │   ECR    │  │SageMaker │  │   EKS    │  │   IoT    │ │
│  │Datasets │  │Containers│  │Training  │  │  OSMO    │  │Greengrass│ │
│  │Models   │  │          │  │Pipelines │  │          │  │  Edge    │ │
│  └────┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │            │            │              │              │        │
│       └────────────┴────────────┴──────────────┴──────────────┘        │
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
| Lab 2 (Workstation) | ~$4.50/hr | Stop when not in use |
| Lab 3 (Cosmos) | ~$15-30 | NIM API calls |
| Lab 4 (RL training) | ~$10-30 | ml.g5.xlarge for 2-4 hours |
| Lab 5 (Edge) | ~$5 | Greengrass deployment |
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

## Lab Sequence

| # | Lab | Time | Depends On |
|---|-----|------|-----------|
| 0 | [Prerequisites](lab-0-prerequisites.md) | 30 min | — |
| 1 | [Train from Demonstrations](lab-1-train-groot.md) | 2 hrs | Lab 0 |
| 2 | [Isaac Sim Workstation](lab-2-isaac-workstation.md) | 30 min | Lab 0 |
| 3 | [Cosmos World Generation](lab-3-cosmos-world-generation.md) | 1-2 hrs | Lab 2 |
| 4 | [RL Refinement](lab-4-rl-refinement.md) | 3 hrs | Labs 1, 3 |
| 5 | [Edge Deployment](lab-5-edge-deployment.md) | 2 hrs | Lab 4 |
| 6 | [OSMO Orchestration](lab-6-osmo-orchestration.md) | 2-3 hrs | Labs 1-5 |

Labs 2 and 3 can run in parallel with Lab 1. Lab 4 requires both Lab 1 (trained model) and Lab 3 (generated scenes) to be complete.

---

## Who This Is For

- **Robotics engineers** building manipulation or locomotion policies
- **ML engineers** moving from cloud training to physical deployment
- **Solutions architects** designing Physical AI platforms for customers
- **Anyone curious** about how robots learn from human demonstrations and simulation

**Prerequisites:** Familiarity with AWS (CLI, console), Python, and basic ML concepts. No robotics experience required — the labs explain the domain concepts as you go.
