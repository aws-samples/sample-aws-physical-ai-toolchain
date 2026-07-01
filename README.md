# AWS Physical AI Toolkit

An end-to-end pipeline for training robot manipulation policies on AWS — from human demonstrations to a deployed physical robot. Built entirely on open-source tools (GR00T, Isaac Lab, Cosmos, LeRobot, ROS 2, PyTorch) running on AWS infrastructure.

---

## What is Physical AI?

Physical AI is artificial intelligence that interacts with the real world. Unlike chatbots or image generators that produce text and pixels, Physical AI produces **motor commands** — signals that move robot arms, open grippers, and navigate through space.

A Physical AI system takes in camera images and joint sensor readings, reasons about what it sees, and outputs precise movements 50-200 times per second. Teaching a robot to pick up an object from a bin requires solving perception (where is it?), planning (how do I reach it?), and control (what exact motor commands get me there?) — all in real-time.

This toolchain provides the infrastructure and workflow to build these systems using AWS services and the NVIDIA robotics stack.

---

## How Robots Learn

Traditional robot programming is manual: engineers write explicit rules for every movement, every edge case, every variation. This breaks down in unstructured environments where objects can be anywhere and look different every time.

Modern Physical AI uses **learned policies** — neural networks trained from data that can generalize to new situations. There are two complementary approaches:

### 1. Imitation Learning (Lab 1)

A human demonstrates the task using teleoperation (remote control). The robot records what it sees (camera) and what it does (motor commands). A foundation model called **GR00T** (Generalist Robot 00 Technology) is fine-tuned on these demonstrations to predict: *given what I see now, what should I do next?*

This gives you a working policy in hours from as few as 50 demonstrations. But it only works well in situations that look like the demos.

### 2. Reinforcement Learning (Lab 4)

The robot practices in simulation — millions of attempts with a reward signal ("+1 when the object is picked up, -0.1 for dropping it"). Through trial and error across thousands of randomized scenes, it discovers strategies that handle variations the demos never showed.

Through trial and error across thousands of randomized scenes, RL discovers robust strategies — given a good simulator and a clear reward function.

### Two approaches, two pipelines — choose per task

Imitation learning (GR00T) and reinforcement learning (Isaac Lab PPO) are **separate pipelines you choose between**, not stages you chain. You pick based on what you have: high-quality demonstrations, or a strong simulator with a definable reward.

```
Path A — Imitation:   Demos ──▶ GR00T fine-tune (SageMaker) ──┐
                                                              ├──▶ Edge Deployment
Path B — RL:          Sim + reward ──▶ Isaac Lab PPO (GPU EC2)─┘     (TensorRT → Jetson)
```

**Compute placement:** VLA/imitation training (GR00T) runs on **SageMaker**; Isaac Sim + RL jobs run on **GPU EC2**. Distributed RL scales two ways: SageMaker multi-instance (`launch_rl.py --instance-count N`) and an opt-in **AWS Batch** multi-node stack (`cdk deploy PhysicalAi-dev-Batch --context batch=true`, submit with `launch_rl_batch.py`). Multi-node NCCL convergence is wired but **unvalidated on hardware**. This toolchain provides all paths as independent, deployable building blocks.

---

## Key Technologies

| Technology | What It Is | Role |
|-----------|-----------|------|
| **GR00T** | NVIDIA's Vision-Language-Action (VLA) foundation model. A 3B-parameter neural network pre-trained on diverse robot data. You fine-tune it on your specific robot and task. | Lab 1: imitation learning from demonstrations |
| **Isaac Lab** | NVIDIA's RL training framework running on the Isaac Sim physics engine. Simulates thousands of parallel robot environments on a single GPU. | Lab 4: RL policy training at scale |
| **Cosmos** | NVIDIA's World Foundation Model (Cosmos 3 Super). Generates synthetic robot demonstrations from text prompts + reference video using the Predict capability. | Lab 3: synthetic demo generation |
| **TensorRT** | NVIDIA's model compiler. Optimizes trained models for real-time inference on edge hardware (Jetson). | Lab 5: edge deployment |
| **OSMO** | NVIDIA's workflow orchestrator for multi-stage Physical AI pipelines. Manages GPU scheduling, stage sequencing, and quality gates. | Lab 6: production orchestration |
| **LeRobot** | HuggingFace's standard data format for robot learning (Parquet + MP4). Used by GR00T for training data. | Data format throughout |
| **ROS 2** | Industry-standard robot middleware. Pub/sub messaging between sensors, models, and actuators. | Edge inference communication |

---

## What This Repo Contains

A complete, deployable Physical AI pipeline:

```
├── config.json                    # Single source of truth: region + workstation settings
│                                  #   (instance type, Isaac Sim AMI map, EBS, DCV port)
├── cdk/                           # Infrastructure as Code (AWS CDK)
│   ├── lib/foundation-stack.ts    # S3, ECR, IAM, CodeBuild image builds
│   ├── lib/constructs/container-build.ts  # S3-asset → CodeBuild → ECR (auto-trigger)
│   ├── lib/workstation-stack.ts   # GPU dev workstation (DCV + Isaac Sim)
│   ├── lib/eks-cluster-stack.ts   # EKS for OSMO (optional)
│   └── lib/edge-stack.ts          # IoT Greengrass for robot fleet
├── containers/                    # Each has a Dockerfile + buildspec.yml (CodeBuild)
│   ├── groot-training/            # GR00T fine-tuning container
│   ├── isaac-lab/                 # Isaac Lab RL training container
│   ├── isaac-sim/                 # Isaac Sim scene-generation container
│   ├── cosmos/                    # Cosmos Transfer (mirrored from NGC → ECR)
│   └── inference/                 # TensorRT + ROS2 inference container
├── training/
│   ├── groot/                     # Pipeline scripts, dataset tools
│   ├── data/                      # Teleop dataset (Git LFS): 27 UR3 pick-and-place episodes
│   ├── scripts/                   # Train, evaluate, export
│   └── envs/                      # RL environments (UR3 pick-and-place)
├── edge/                          # Greengrass components + ROS2 node
└── workshop/                      # Hands-on lab guides (7 labs)
```

---

## Quick Start

From a fresh clone to a running GR00T training job. Prerequisites: AWS CLI v2
configured, Node.js 18+, and git-lfs (`brew install git-lfs && git lfs install`).
No Docker needed — container images build in AWS CodeBuild, not locally.

```bash
git clone <REPO_URL> && cd aws-physical-ai-toolchain

python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .                                        # installs the `pai` CLI + deps

pai config set aws.region us-west-2                     # single source of truth (pai + CDK read it)
pai doctor                                              # credentials + region — catch problems now
pai deploy foundation                                   # S3 + ECR + roles; kicks off CodeBuild (~10 min)

git lfs pull && unzip training/data/ur3_episodes_001_027.zip -d training/data/episodes

pai groot convert                                       # Zarr teleop episodes -> LeRobot v2 (local)
pai groot upload                                        # sync the dataset to S3 (bucket auto-resolved)
pai groot launch --max-steps 100                        # smoke-test the fine-tuning pipeline (~15 min, ~$2)
pai groot runs                                          # watch the run; registers to the groot-models registry
```

That's [Lab 1](workshop/lab-1-train-groot.md) end-to-end. For the full walk-through
(monitoring, the full training run, serving the model as an endpoint) and the rest
of the workshop, see **[workshop/](workshop/README.md)**. The deeper setup that the
other labs need (CDK bootstrap, NGC API key for the Isaac/Cosmos image builds) is in
**[Lab 0: Prerequisites](workshop/lab-0-prerequisites.md)**.

> **Why CodeBuild?** The NVIDIA base images (Isaac Lab ~16 GB, Cosmos ~30 GB) are
> too large to pull or build on a laptop, and several are x86-only (they can't be
> built on Apple Silicon at all). CodeBuild builds them on a large cloud instance
> and pushes to your ECR. See [Lab 0](workshop/lab-0-prerequisites.md) for the
> one-time NGC API key setup that the NVIDIA-based builds need.

See [workshop/README.md](workshop/README.md) for the full guided experience.

---

## Workshop (Self-Paced)

Seven hands-on labs taking you from zero to a deployed robot policy:

| Lab | What You Build | Time | Cost |
|-----|---------------|------|------|
| [Lab 0: Prerequisites](workshop/lab-0-prerequisites.md) | Deploy AWS infrastructure | 30 min | — |
| [Lab 1: Train from Demos](workshop/lab-1-train-groot.md) | GR00T fine-tuning on SageMaker | 2 hrs | ~$15-30 |
| [Lab 2: Isaac Sim Workstation](workshop/lab-2-isaac-workstation.md) | GPU remote desktop for visual dev | 30 min | ~$3.00/hr |
| [Lab 3: Cosmos World Generation](workshop/lab-3-cosmos-world-generation.md) | Synthetic demo generation (Cosmos 3 Super, Predict mode) | 1-2 hrs | ~$300-500 |
| [Lab 4: RL Policy Training](workshop/lab-5-rl-refinement-with-isaac.md) | Train a policy in simulation with RL | 3 hrs | ~$10-30 |
| [Lab 5: Edge Deployment](workshop/lab-5-edge-deployment.md) | Deploy to Jetson via Greengrass | 2 hrs | ~$5 |
| [Lab 6: OSMO Orchestration](workshop/lab-6-osmo-orchestration.md) | Production pipeline on EKS | 2-3 hrs | ~$50-100 |

**No robot hardware required.** Labs 0-4 run entirely in the cloud. Lab 5 deploys to a physical robot if you have one (UR3 + Jetson).

---

## The Use Case: Pick and Place

We build a pick-and-place policy because it's:

- The **#1 most common** industrial robot task (bin picking, kitting, palletizing)
- **Simple enough** to learn in a workshop but hard enough to need real AI
- **Exercises the full pipeline** — perception, planning, grasping, placement
- **Transferable** — the same pipeline works for assembly, sorting, inspection

The reference uses a **UR3 arm** (most popular collaborative robot in industry) with a **Robotiq 2F-85 gripper**. The pipeline is robot-agnostic — bring your own URDF and teleop data.

---

## What's Working Today

- ✅ Foundation infrastructure (S3, ECR, IAM)
- ✅ Cloud container builds — every image built in CodeBuild and pushed to ECR on `cdk deploy` (no local Docker, works on Apple Silicon)
- ✅ GR00T fine-tuning on SageMaker (full pipeline: train → eval → register)
- ✅ GR00T fine-tuning with real UR3 data (100-step smoke test succeeded with 27 real teleop episodes)
- ✅ Isaac Lab RL training on SageMaker (100 iterations, 60K steps/s, reward -0.36→+8.58)
- ✅ Isaac Sim workstation on the NVIDIA Isaac Sim Marketplace AMI (g6e.4xlarge / L40S, DCV + Isaac Sim pre-baked)
- ✅ Imitation (GR00T/SageMaker) and RL (Isaac Lab/EC2) as separate, independently runnable pipelines
- ✅ Real UR3 teleop data (27 episodes, 3,467 frames, converted to LeRobot v2)
- ✅ Lab docs (0-6) written with full intro + terminology glossary
- ✅ Cosmos 3 Super V2V generation validated (p5.48xlarge, vLLM-Omni, Capacity Block)
- ✅ Synthetic demo generation working (multiple prompts, ~5 min/video on 8x H100)
- 🔲 Edge deployment (CDK stack ready, untested on hardware)

---

## Cost Summary

| Activity | Cost | Notes |
|----------|------|-------|
| Infrastructure (idle) | ~$1/month | S3 storage only |
| GR00T training (smoke test) | ~$2 | 15 min on ml.g5.12xlarge |
| GR00T training (full) | ~$79 | 11 hours |
| Isaac Lab RL (100 iterations) | ~$3 | 15 min on ml.g5.xlarge |
| Isaac Lab RL (full, 2000 iterations) | ~$28 | 4 hrs on ml.g5.12xlarge |
| Workstation (per hour) | ~$3.00 | g6e.4xlarge (L40S). Stop when not using |
| **Total workshop (Labs 0-4)** | **~$50-100** | |

All resources tear down cleanly with `cdk destroy`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and development notes.

## License

Apache 2.0
