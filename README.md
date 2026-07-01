# AWS Physical AI Toolkit

An end-to-end pipeline for training robot manipulation policies on AWS — from human demonstrations to a deployed physical robot. Built on open-source tools (GR00T, Isaac Sim, Isaac Lab, Cosmos, LeRobot, Hugging Face, ROS 2, PyTorch) running on AWS infrastructure.

You'll build a complete **pick-and-place** pipeline: the most common industrial manipulation task and the starting point for most robotics teams.

---

## Quick Start

```bash
git clone <REPO_URL> && cd aws-physical-ai-toolchain

python3 -m venv .venv && source .venv/bin/activate
pip install -e .                                        # installs the `pai` CLI

pai config set aws.region us-west-2
pai doctor                                              # verify credentials + region
pai deploy foundation                                   # S3 + ECR + roles + CodeBuild (~10 min)

git lfs pull && unzip training/data/ur3_episodes_001_027.zip -d training/data/episodes

pai groot convert                                       # Zarr teleop → LeRobot v2
pai groot upload                                        # sync dataset to S3
pai groot launch --max-steps 100                        # smoke-test fine-tuning (~15 min, ~$2)
```

Prerequisites: AWS CLI v2, Node.js 18+, git-lfs. No Docker needed — containers build in CodeBuild.

> **Why CodeBuild?** The NVIDIA base images (Isaac Lab ~16 GB, Cosmos ~30 GB) are too large
> to pull or build locally, and several are x86-only (won't build on Apple Silicon). CodeBuild
> builds them in the cloud and pushes to your ECR automatically on `cdk deploy`.

---

## What You'll Build

```
Teleop Data  →  Imitation Learning  →  World Generation   →  RL Training      →  Export .onnx
(Lab 1)          (GR00T on              (Cosmos V2V)         (Isaac Lab on       (ready for
                  SageMaker)                                  SageMaker)          deployment)
```

A production-grade Physical AI pipeline with:

- **Infrastructure as Code** — everything deploys via `cdk deploy`. Reproducible, versionable, teardown-able.
- **Open-source toolchain** — GR00T, Isaac Sim, Isaac Lab, Cosmos, LeRobot, Hugging Face, ROS 2, PyTorch
- **AWS services** — SageMaker (training), S3 (data), ECR (containers), CodeBuild (CI), EC2 (Cosmos generation)
- **GPU-accelerated** — NVIDIA GPUs for parallel simulation (4096 robots simultaneously) and photorealistic world generation (Cosmos 3 on H100)

---

## Labs

| # | Lab | What You Build | Time | Cost |
|---|-----|---------------|------|------|
| 0 | [Prerequisites](workshop/lab-0-prerequisites.md) | Deploy AWS infrastructure | 30 min | — |
| 1 | [Train from Demos](workshop/lab-1-train-groot.md) | GR00T fine-tuning on SageMaker | 2 hrs | ~$15-30 |
| 2 | [Isaac Sim Workstation](workshop/lab-2-isaac-workstation.md) | GPU remote desktop for visual dev | 30 min | ~$3/hr |
| 3 | [Cosmos World Generation](workshop/lab-3-cosmos-world-generation.md) | Generate synthetic demos (Cosmos 3 Super, Predict) | 1-2 hrs | ~$300-500 |
| 4 | [Cosmos Transfer](workshop/lab-4-cosmos-transfer.md) | Restyle existing data preserving actions (Transfer 2.5) | 1-2 hrs | ~$37/hr |
| 5 | [RL Policy Training](workshop/lab-5-rl-refinement-with-isaac.md) | Train a policy in simulation with RL | 3 hrs | ~$10-30 |
| 6 | [OSMO Orchestration](workshop/lab-6-osmo-orchestration.md) | Production pipeline on EKS (placeholder) | 2-3 hrs | ~$50-100 |

**No robot hardware required.** Labs 0-5 run entirely in the cloud.

→ **New to Physical AI?** Read the [Workshop Introduction](workshop/README.md) for background on how robots learn, key terminology, and what each lab teaches.

### Dependencies

Labs 2, 3, and 4 can run in parallel with Lab 1. Labs 3 and 4 take Lab 1's dataset as input (wrist camera MP4s). **Lab 5 is standalone RL** — it does not require Lab 1 or Labs 3-4. RL and imitation learning are two independent ways to obtain a policy.

---

## The Use Case: Pick and Place

We use pick-and-place because it's the **#1 most common** industrial robot task, exercises the full pipeline (perception → planning → grasping → placement), and has clear success metrics.

The reference uses a **UR3 arm** with a **Robotiq 2F-85 gripper**. The pipeline is robot-agnostic — bring your own URDF and teleop data.

---

## Architecture

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

## What's In This Repo

```
├── config.json              # Region, instance types, AMI mappings
├── cdk/                     # Infrastructure as Code (CDK stacks)
├── containers/              # Dockerfiles + buildspecs (CodeBuild → ECR)
├── training/                # Training scripts, dataset, RL environments
│   ├── groot/               # GR00T pipeline (convert, upload, launch, deploy)
│   ├── scripts/             # Cosmos, RL, evaluation scripts
│   ├── data/                # Teleop dataset (Git LFS) + Cosmos samples
│   └── envs/                # Isaac Lab RL environments
├── pai/                     # The `pai` CLI tool (commands for all labs)
├── scripts/                 # Automation (cosmos3-launch.sh, mirror scripts)
├── kubernetes/              # EKS manifests (production path)
├── docs/                    # Glossary, runbooks, prompt catalog
├── tests/                   # Unit + integration tests
├── workshop/                # Lab guides (README + 7 lab .md files)
├── edge/                    # Greengrass + ROS2 (future)
└── workflows/               # OSMO workflow YAML (future)
```

---

## Bring Your Own Data

The pipeline accepts teleoperation recordings in **LeRobot v2 format** (Parquet + MP4):

```
your-dataset/
├── data/chunk-000/file-000.parquet    # Joint states, actions, timestamps
├── videos/observation.images.wrist/chunk-000/file-000.mp4
└── meta/info.json                     # Dataset metadata
```

Conversion scripts provided for Zarr and ROS bags:
```bash
pai groot convert --episodes-dir ./my_episodes
```

---

## No Hardware Required

This entire toolkit runs in the cloud. You don't need a physical robot for Labs 0-5.

**If you have a UR3 arm:** The trained policy exports as `.onnx` and can be deployed to edge hardware manually. Automated edge deployment (Greengrass + Jetson) is planned as a future lab.

**If you have different hardware:** Bring your URDF and teleop data — everything else stays the same.

---

## Estimated Costs

| Component | Cost | Notes |
|-----------|------|-------|
| Lab 1 (GR00T training) | ~$15-30 | ml.g5.12xlarge for 1-2 hours |
| Lab 2 (Workstation) | ~$3/hr | Stop when not in use |
| Lab 3 (Cosmos Predict) | ~$300-500 | Capacity Block (p5.48xlarge, ~$37/hr) |
| Lab 4 (Cosmos Transfer) | ~$37/hr | p5.48xlarge Capacity Block |
| Lab 5 (RL training) | ~$10-30 | ml.g5.xlarge for 2-4 hours |
| **Total (Labs 0-5)** | **~$50-150** | Excluding Cosmos GPU blocks |

All resources tear down with `cdk destroy`.

---

## What's Working

- ✅ Foundation infrastructure (S3, ECR, IAM, CodeBuild)
- ✅ GR00T fine-tuning on SageMaker (27 real UR3 teleop episodes)
- ✅ Isaac Lab RL training (4096 parallel envs, 60K steps/s)
- ✅ Isaac Sim workstation (g6e.4xlarge, DCV + Isaac Sim pre-baked)
- ✅ Cosmos 3 Super V2V generation (synthetic demos from prompts, ~5 min/video)
- ✅ Cosmos Transfer 2.5 NIM (geometry-preserving restyle, quality tuning needed)
- ✅ Real UR3 teleop data (27 episodes, LeRobot v2 format)
- ✅ `pai` CLI for all operations (doctor, deploy, groot, rl, workstation)
- 🔲 OSMO orchestration (Lab 6 — placeholder, contributions welcome)

---

## Who This Is For

- **Robotics engineers** building manipulation or locomotion policies
- **ML engineers** moving from cloud training to physical deployment
- **Solutions architects** designing Physical AI platforms for customers
- **Anyone curious** about how robots learn from human demonstrations and simulation

**Prerequisites:** Familiarity with AWS (CLI, console), Python, and basic ML concepts. No robotics experience required — the [workshop introduction](workshop/README.md) explains the domain concepts as you go.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and development notes.

## License

Apache 2.0
