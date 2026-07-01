# AWS Physical AI Toolkit

An end-to-end pipeline for training robot manipulation policies on AWS — from human demonstrations to a deployed physical robot. Built on open-source tools (GR00T, Isaac Sim, Isaac Lab, Cosmos, LeRobot, Hugging Face, ROS 2, PyTorch) running on AWS infrastructure.

---

## Quick Start

```bash
git clone <REPO_URL> && cd aws-physical-ai-toolchain

python3 -m venv .venv && source .venv/bin/activate
pip install -e .

pai config set aws.region us-west-2
pai doctor
pai deploy foundation                # S3 + ECR + roles; kicks off CodeBuild (~10 min)

git lfs pull && unzip training/data/ur3_episodes_001_027.zip -d training/data/episodes

pai groot convert                    # Zarr teleop → LeRobot v2
pai groot upload                     # sync dataset to S3
pai groot launch --max-steps 100     # smoke-test fine-tuning (~15 min, ~$2)
```

Prerequisites: AWS CLI v2, Node.js 18+, git-lfs. No Docker needed — containers build in CodeBuild.

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

→ **[Start with the Workshop Introduction](workshop/README.md)** for background on Physical AI, how robots learn, and key terminology before diving into the labs.

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

## What's Working

- ✅ Foundation infrastructure (S3, ECR, IAM, CodeBuild)
- ✅ GR00T fine-tuning on SageMaker (27 real UR3 teleop episodes)
- ✅ Isaac Lab RL training (4096 parallel envs, 60K steps/s)
- ✅ Isaac Sim workstation (g6e.4xlarge, DCV + Isaac Sim pre-baked)
- ✅ Cosmos 3 Super V2V generation (synthetic demos from prompts, ~5 min/video)
- ✅ Cosmos Transfer 2.5 NIM (geometry-preserving restyle, quality tuning needed)
- ✅ Real UR3 teleop data (27 episodes, LeRobot v2 format)
- 🔲 OSMO orchestration (Lab 6 — placeholder, contributions welcome)

---

## Cost Summary

| Activity | Cost |
|----------|------|
| GR00T smoke test | ~$2 |
| GR00T full training | ~$79 |
| Isaac Lab RL (100 iterations) | ~$3 |
| Workstation (per hour) | ~$3 |
| Cosmos Predict (Capacity Block) | ~$37/hr |
| **Total workshop (Labs 0-5)** | **~$50-150** |

All resources tear down with `cdk destroy`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and development notes.

## License

Apache 2.0
