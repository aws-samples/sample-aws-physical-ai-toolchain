# AWS Physical AI Toolchain

A curated collection of reference architectures, Infrastructure as Code, and deployment automation for running the Physical AI stack on Amazon Web Services.

Physical AI systems — humanoid robots, autonomous mobile robots, self-driving vehicles, and smart factories — are moving from research demonstrations to production deployments. Developing these systems requires three classes of compute working in concert:

1. **High-bandwidth GPU clusters** for foundation model pre-training and post-training (fine-tuning, alignment)
2. **Elastic mid-tier GPU capacity** for simulation and software-in-the-loop validation
3. **Edge GPUs** ([NVIDIA Jetson Thor / AGX](https://developer.nvidia.com/embedded-computing)) inside the robot or at the edge for real-time inference

This toolchain provides AWS sample code for each stage, built on AWS managed services and integrated with the NVIDIA Physical AI software ecosystem.

---

## Two Ways to Use This Toolkit

### Path A: Deploy to Production (Terraform)

You have your own infrastructure pipeline and want to deploy specific components. Each is independent — deploy what you need:

```bash
cd foundation/infra && terraform apply         # Shared base (S3, ECR, IAM)
cd groot-training-on-aws/infra && terraform apply   # GR00T fine-tuning infra
cd cosmos-on-aws/infra && terraform apply           # Cosmos generation infra
cd isaac-sim-on-aws/infra && terraform apply        # Isaac Sim workstation
cd isaac-lab-on-aws/infra && terraform apply        # Isaac Lab RL training infra
cd osmo-on-aws/001-iac && terraform apply           # OSMO orchestration platform
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

The CFN template bootstraps everything. Then it's Python scripts and the `pai` CLI for the hands-on exercises.

---

## Toolchain Components

| Component | What it does | Deploy (Terraform) | Learn (Workshop) |
|-----------|-------------|-------------------|-----------------|
| [**GR00T Training**](groot-training-on-aws/) | Fine-tune NVIDIA GR00T VLA model on SageMaker | `groot-training-on-aws/infra/` | [Lab 1](workshop/lab-1-train-groot.md) |
| [**Cosmos**](cosmos-on-aws/) | Synthetic data generation (Predict) + augmentation (Transfer) | `cosmos-on-aws/infra/` | [Labs 3-4](workshop/lab-3-cosmos-world-generation.md) |
| [**Isaac Sim**](isaac-sim-on-aws/) | GPU workstation for physics simulation | `isaac-sim-on-aws/infra/` | [Lab 2](workshop/lab-2-isaac-workstation.md) |
| [**Isaac Lab**](isaac-lab-on-aws/) | RL policy training (4096 parallel envs) | `isaac-lab-on-aws/infra/` | [Lab 5](workshop/lab-5-rl-refinement-with-isaac.md) |
| [**OSMO**](osmo-on-aws/) | Pipeline orchestration across heterogeneous compute | `osmo-on-aws/001-iac/` | [Lab 6](workshop/lab-6-osmo-orchestration.md) |
| **Foundation** | Shared S3 buckets, ECR repos, IAM roles | `foundation/infra/` | [Lab 0](workshop/lab-0-prerequisites.md) |

---

## Physical AI Development Flywheel

Physical AI development follows a continuous improvement cycle. Each stage feeds the next:

**Data → Train → Validate → Deploy → Feedback → Generate**

| Pillar 1 | Pillar 2 | Pillar 3 | Pillar 4 |
|----------|----------|----------|----------|
| **Synthetic Data Generation** | **Model Training** | **SIL Simulation** | **Sim-to-Real / HIL** |
| Scene composition, domain randomization, curriculum-aware augmentation | Distributed training, RL, hyperparameter search, checkpoint promotion | Physics-accurate validation, adversarial scenarios, regression gating | Domain adaptation, safety monitoring, digital twin sync |
| *Isaac Sim + Cosmos* | *GR00T, Isaac Lab* | *Isaac Sim* | *Jetson / RTX* |

**Orchestration** spans the entire cycle — [NVIDIA OSMO](https://nvidia.github.io/OSMO/main/user_guide/index.html) coordinates task scheduling, data flow, and resource allocation across heterogeneous compute.

---

## How It Maps to AWS

| Flywheel Stage | AWS Compute | Supporting Services |
|----------------|-------------|---------------------|
| **Synthetic Data Generation** | Amazon EC2 (G6e / L40S, P5 / H100) | Amazon S3, Amazon ECR, NVIDIA NGC |
| **Model Training** | Amazon SageMaker, AWS Batch | Amazon S3, Amazon FSx for Lustre |
| **SIL Simulation** | Amazon EC2 (G6e GPUs) on EKS | Amazon S3, Amazon CloudWatch |
| **HIL / Edge Deployment** | EKS Hybrid Nodes (Jetson, RTX) | AWS IoT Greengrass |
| **Orchestration (OSMO)** | Amazon EKS (system nodes) | Amazon RDS, ElastiCache, S3, KMS |

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
- NVIDIA NGC API key (for container image pulls)
- HuggingFace token (for model weight downloads)
- **Production path:** Terraform >= 1.5
- **Workshop path:** No Terraform needed — containers build in AWS CodeBuild
- No Docker required locally

---

## When to Use OSMO vs. Individual Components

**Use OSMO (full orchestration)** when you need an end-to-end platform that manages the entire flywheel — scheduling tasks across heterogeneous compute, resolving data dependencies, and routing workloads from cloud GPUs to edge devices.

**Use individual components standalone** when you already have an established orchestration pipeline (Kubeflow, Airflow, Argo, Step Functions) and want to integrate a specific NVIDIA capability into your existing infrastructure.

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
