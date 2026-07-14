# AWS NVIDIA Physical AI Toolchain

A framework for training and deploying robot policies on AWS — any robot, any task, any hardware. Provides infrastructure, scripts, and guided labs for the core pipeline (data ingestion, imitation learning, synthetic data generation, RL refinement, deployment) using [NVIDIA](https://developer.nvidia.com/physical-ai) tools ([GR00T](https://developer.nvidia.com/isaac/gr00t), [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), [Isaac Lab](https://developer.nvidia.com/isaac/lab), [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/), [OSMO](https://nvidia.github.io/OSMO/main/user_guide/index.html)) on AWS services.

The toolkit is **robot-agnostic and task-agnostic**. Bring your own URDF, your own teleoperation data, and your own task definition — the pipeline handles the rest. We include a complete **pick-and-place example** (UR3 arm + Robotiq gripper, 27 real teleoperation demonstrations) so you can see the full system working end-to-end before adapting it to your own use case.

---

## What You'll Build

```
Teleop Data      Imitation Learning      World Generation      Data Augmentation     RL Training         Export .onnx
                 (GR00T on               (Cosmos Predict V2V)  (Cosmos Transfer)     (Isaac Lab on       (ready for
                  SageMaker)                                                          SageMaker)          deployment)
```

A Physical AI pipeline with:

- **Infrastructure as Code** — everything deploys via `terraform apply` or `aws cloudformation deploy`. Reproducible, versionable, teardown-able.
- **Open-source toolchain** — GR00T, Isaac Sim, Isaac Lab, Cosmos, LeRobot, Hugging Face, ROS 2, PyTorch
- **AWS services** — SageMaker (training), S3 (data), ECR (containers), CodeBuild (CI), EC2 (Cosmos generation)
- **GPU-accelerated** — NVIDIA GPUs for parallel simulation (4096 robots simultaneously) and photorealistic world generation (Cosmos 3 on H100)

**No robot hardware required.** Labs 0-6 run entirely in the cloud on the bundled demonstrations. Lab 1 also includes an optional bring-your-own-robot track — teams with a physical UR3 can record their own demonstrations and run the trained policy on the arm, completing the full teleop → train → deploy → autonomous-control loop.

→ **New to Physical AI?** Read the [Workshop Introduction](workshop/README.md) for background on how robots learn, key terminology, and what each lab teaches.

---

## Two Ways to Use This Toolkit

### Path A: Deploy Components (Terraform)

You want to deploy specific NVIDIA components to your AWS account. Each is independent — deploy what you need:

```bash
cd foundation/infra && terraform apply                    # Shared base (S3, ECR, IAM)
cd groot-training-on-aws/infra && terraform apply         # GR00T fine-tuning infra
cd cosmos-on-aws/infra && terraform apply                 # Cosmos generation infra
cd isaac-sim-on-aws/infra && terraform apply              # Isaac Sim workstation
cd isaac-lab-on-aws/infra && terraform apply              # Isaac Lab RL training infra
cd osmo-on-aws/001-iac && terraform apply                 # OSMO orchestration platform
```

### Path B: Learn via Workshop (CloudFormation + CLI)

You want a guided, step-by-step learning experience. No Terraform needed:

```bash
# 1. Bootstrap infrastructure (one command)
aws cloudformation deploy --template-file workshop/bootstrap.cfn.yaml \
  --stack-name physical-ai-foundation --capabilities CAPABILITY_NAMED_IAM

# 2. Follow the labs in order
cd workshop/ && cat README.md
```

This repo is both an **accelerator framework** and a **hands-on learning experience**. Each lab includes:

- Step-by-step instructions with exact CLI commands and expected outputs
- Cost and time estimates so you know what you're spending before you run anything
- "Under the hood" sections that explain what each command does and why
- Troubleshooting tables for common issues

While some AWS cloud experience is assumed, no prior robotics experience is required. The [workshop introduction](workshop/README.md) covers foundational concepts — how robots learn from demonstrations vs. simulation, what a policy is, why sim-to-real transfer is hard, and a full terminology glossary.

The workshop format is modular: run all labs in a day as an instructor-led session, work through them self-paced over a week, or jump directly to the lab that matches your immediate need.

---

## Toolchain Components

| Component | Description | Deploy (Terraform) | Learn (Workshop) | Status |
|-----------|-------------|-------------------|-----------------|--------|
| [**GR00T Training**](groot-training-on-aws/) | Fine-tune [NVIDIA GR00T](https://developer.nvidia.com/isaac/gr00t) N1.6 VLA model on SageMaker | `groot-training-on-aws/infra/` | [Lab 1](workshop/lab-1-train-groot.md) | Available |
| [**Cosmos**](cosmos-on-aws/) | [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) world generation (Predict V2V) + data augmentation (Transfer 2.5) | `cosmos-on-aws/infra/` | [Labs 3-4](workshop/lab-3-cosmos-world-generation.md) | Available |
| [**Isaac Sim**](isaac-sim-on-aws/) | [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) GPU workstation for physics simulation | `isaac-sim-on-aws/infra/` | [Lab 2](workshop/lab-2-isaac-workstation.md) | Available |
| [**Isaac Lab**](isaac-lab-on-aws/) | [NVIDIA Isaac Lab](https://developer.nvidia.com/isaac/lab) RL training (4096 parallel envs) on SageMaker + Batch | `isaac-lab-on-aws/infra/` | [Lab 5](workshop/lab-5-rl-refinement-with-isaac.md) | Available |
| [**OSMO**](osmo-on-aws/) | [NVIDIA OSMO](https://nvidia.github.io/OSMO/main/user_guide/index.html) 6.3 orchestration on EKS — control plane, compute, GPU scheduling | `osmo-on-aws/001-iac/` | [Lab 6](workshop/lab-6-osmo-orchestration.md) | Available |
| **Foundation** | Shared S3 buckets, ECR repos, IAM roles, SSM parameters | `foundation/infra/` | [Lab 0](workshop/lab-0-prerequisites.md) | Available |
| *Edge Deployment* | Model packaging to [Jetson](https://developer.nvidia.com/embedded-computing) via EKS Hybrid Nodes + Greengrass | Planned | — | Planned |
| *Agentic Layer* | AI orchestration with Strands Agents SDK + Amazon Bedrock AgentCore | Planned | — | Planned |

---

## Architecture

![AWS Physical AI Toolkit Architecture](arch-diagram.png)

**Pipeline stages:**
1. **Ingest** — Convert teleoperation recordings (Zarr, ROS bags, CSV) to LeRobot v2 format and store in S3
2. **Train (Imitation Learning)** — Fine-tune GR00T on demonstrations via SageMaker
3. **World Generation** — Generate new synthetic demos with Cosmos 3 Predict, or restyle existing video with Cosmos Transfer 2.5
4. **Train (Reinforcement Learning)** — Train a policy from scratch in Isaac Lab (4096 parallel environments on one GPU)
5. **Deploy** — Export to TensorRT, deploy to robot fleet via Greengrass

---

## Physical AI Development Flywheel

Physical AI development follows a continuous improvement cycle. Each stage feeds the next, accelerating model quality with every iteration:

**Data → Train → Validate → Deploy → Feedback → Generate**

| Pillar 1 | Pillar 2 | Pillar 3 | Pillar 4 |
|----------|----------|----------|----------|
| **Synthetic Data Generation** | **Model Training** | **SIL Simulation** | **Sim-to-Real / HIL** |
| Scene composition, domain randomization, curriculum-aware augmentation | Distributed training, RL, hyperparameter search, checkpoint promotion | Physics-accurate validation, adversarial scenarios, regression gating | Domain adaptation, safety monitoring, digital twin sync, deployment scoring |
| *[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) + [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/)* | *[GR00T](https://developer.nvidia.com/isaac/gr00t), [Isaac Lab](https://developer.nvidia.com/isaac/lab)* | *[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)* | *[Jetson](https://developer.nvidia.com/embedded-computing) / RTX* |

**Orchestration** spans the entire cycle — [NVIDIA OSMO](https://nvidia.github.io/OSMO/main/user_guide/index.html) coordinates task scheduling, data flow, dependency resolution, and resource allocation across heterogeneous compute.

---

## How It Maps to AWS

| Flywheel Stage | AWS Compute | Supporting Services |
|----------------|-------------|---------------------|
| **Synthetic Data Generation** | Amazon EC2 (G6e / L40S, P5 / H100) on EKS | Amazon S3, Amazon ECR, [NVIDIA NGC](https://catalog.ngc.nvidia.com/) |
| **Model Training** | Amazon SageMaker, AWS Batch (P5 / P6 with EFA) | Amazon S3, Amazon FSx for Lustre |
| **SIL Simulation** | Amazon EC2 (G6e GPUs) on EKS | Amazon S3, Amazon CloudWatch |
| **HIL / Edge Deployment** | EKS Hybrid Nodes (Jetson, RTX workstations) | AWS Site-to-Site VPN, AWS Direct Connect |
| **Orchestration (OSMO)** | Amazon EKS (system nodes) | Amazon RDS, ElastiCache, S3, Secrets Manager, KMS |

---

## Workshop Labs

| # | Lab | What You Build | Time | Cost |
|---|-----|---------------|------|------|
| 0 | [Prerequisites](workshop/lab-0-prerequisites.md) | Deploy foundation infrastructure | 30 min | Free |
| 1 | [Train from Demos](workshop/lab-1-train-groot.md) | GR00T fine-tuning on SageMaker | 2 hrs | ~$2-79 |
| 2 | [Isaac Sim Workstation](workshop/lab-2-isaac-workstation.md) | GPU remote desktop for visual dev | 30 min | ~$1.86/hr |
| 3 | [Cosmos World Generation](workshop/lab-3-cosmos-world-generation.md) | Generate synthetic demos (Cosmos 3 Predict) | 1-2 hrs | ~$37/hr |
| 4 | [Cosmos Transfer](workshop/lab-4-cosmos-transfer.md) | Restyle data preserving actions (Transfer 2.5) | 1-2 hrs | ~$8/hr |
| 5 | [RL Policy Training](workshop/lab-5-rl-refinement-with-isaac.md) | Isaac Lab RL in simulation (4096 envs) | 3 hrs | ~$10-30 |
| 6 | [OSMO Orchestration](workshop/lab-6-osmo-orchestration.md) | Production pipeline on EKS | 2-3 hrs | ~$5/hr |

**No robot hardware required.** Labs 0-5 run entirely in the cloud.

---

## Modular by Design

This is a **modular framework** — use the pieces you need. Each component is an independent building block: imitation learning (GR00T), simulation (Isaac Sim, Isaac Lab), synthetic data generation (Cosmos), and orchestration (OSMO) can be adopted individually or combined. The foundation provides the shared infrastructure that all other components build on.

---

## Pick and Place Example Use Case Included

The toolkit is generic infrastructure for any robot, any task, any hardware. To demonstrate it working end-to-end, we provide a complete **pick-and-place** example — the most common industrial robot task (bin picking, kitting, palletizing).

The example uses a **UR3 arm** (a popular collaborative robot in the industry) with its standard **Robotiq 2F-85 gripper** and includes 27 real teleoperation episodes. You can swap in any robot by providing your own URDF and teleop data — the pipeline stays the same regardless of embodiment or task.

---

## Estimated Costs

| Component | Cost | Notes |
|-----------|------|-------|
| GR00T training (smoke test) | ~$2 | ml.g5.12xlarge for 15 min |
| GR00T training (full) | ~$79 | ml.g5.12xlarge for 11 hrs |
| Cosmos 3 Predict | ~$37/hr | p5.48xlarge (Capacity Block) |
| Cosmos Transfer 2.5 | ~$8/hr | g6e.12xlarge (Spot) |
| Isaac Sim workstation | ~$1.86/hr | g6e.4xlarge (stop when idle) |
| Isaac Lab RL training | ~$10-30 | ml.g5.xlarge for 2-4 hrs |
| OSMO (full deployment) | ~$5/hr | EKS + RDS + ElastiCache |

All resources tear down with `terraform destroy` or `aws cloudformation delete-stack`.

---

## Prerequisites

- AWS account with GPU quota (SageMaker + EC2)
- AWS CLI v2, Python 3.11+
- NVIDIA NGC API key (for container image pulls) — [generate here](https://ngc.nvidia.com/setup/api-key)
- HuggingFace token (for model weight downloads) — [create here](https://huggingface.co/settings/tokens)
- **Production path:** Terraform >= 1.5
- **Workshop path:** No Terraform needed
- No Docker required locally — containers build in AWS CodeBuild

---

## When to Use OSMO vs. Individual Components

| Scenario | Recommended Approach |
|----------|---------------------|
| Greenfield Physical AI platform | Start with **osmo-on-aws** — orchestration + compute for all stages |
| Existing pipeline, need SDG only | Deploy **cosmos-on-aws** standalone, call from your orchestrator |
| Existing pipeline, need training | Deploy **groot-training-on-aws**, submit jobs via your scheduler |
| Existing pipeline, need RL sim | Deploy **isaac-lab-on-aws** standalone |
| Migrating to managed orchestration | Start with **osmo-on-aws**, then migrate stages incrementally |

---

## What's Working

- GR00T fine-tuning on SageMaker (27 real UR3 teleop episodes)
- Isaac Lab RL training (4096 parallel envs, 60K steps/s)
- Isaac Sim workstation (g6e.4xlarge, DCV + Isaac Sim pre-baked)
- Cosmos 3 Super V2V generation (synthetic demos from prompts)
- Cosmos Transfer 2.5 NIM (geometry-preserving restyle)
- Real UR3 teleop data (27 episodes, LeRobot v2 format)
- OSMO 6.3 deployment on EKS (full Terraform + Helm)
- Per-component Terraform (independently deployable)
- Workshop CFN bootstrap (single-command setup)
- `pai` CLI for all workshop operations

---

## Who This Is For

- **Robotics engineers** building manipulation or locomotion policies
- **ML engineers** moving from cloud training to physical deployment
- **Solutions architects** designing Physical AI platforms for customers
- **Platform teams** deploying NVIDIA tools on AWS infrastructure
- **Anyone curious** about how robots learn from demonstrations and simulation

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and development notes.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Authors

- **Ignacio Salvar** — Solutions Architect, AWS
- **Adam** — Solutions Architect, AWS
- **Abhishek Srivastav** — Principal Solutions Architect, AWS
- **Jathavan Sriram** — Senior Solutions Architect, NVIDIA
