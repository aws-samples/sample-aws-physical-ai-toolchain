# AWS Physical AI Toolchain

A curated collection of reference architectures, Infrastructure as Code, and deployment automation for running the Physical AI stack on Amazon Web Services.

Physical AI systems — humanoid robots, autonomous mobile robots, self-driving vehicles, and smart factories — are moving from research demonstrations to production deployments. Developing these systems requires three classes of compute working in concert:

1. **High-bandwidth GPU clusters** for foundation model pre-training and post-training (fine-tuning, alignment)
2. **Elastic mid-tier GPU capacity** for simulation and software-in-the-loop validation
3. **Edge GPUs** ([NVIDIA Jetson Thor / AGX](https://developer.nvidia.com/embedded-computing)) inside the robot or at the edge for real-time inference

This toolchain provides AWS sample code for each stage, built on AWS managed services and integrated with the NVIDIA Physical AI software ecosystem.

## Physical AI Development Flywheel

Physical AI development follows a continuous improvement cycle. Each stage feeds the next, accelerating model quality with every iteration:

<p align="center">
  <img src="flywheel.png" alt="Physical AI Development Flywheel" width="700"/>
</p>

The flywheel consists of four pillars with an **Agentic AI Orchestration Layer** at the center:

<table>
<tr>
<th align="left">Pillar 1</th>
<th align="left">Pillar 2</th>
<th align="left">Pillar 3</th>
<th align="left">Pillar 4</th>
</tr>
<tr>
<td align="left"><b>Synthetic Data Generation</b></td>
<td align="left"><b>Model Training</b></td>
<td align="left"><b>SIL Simulation</b></td>
<td align="left"><b>Sim-to-Real / HIL</b></td>
</tr>
<tr>
<td align="left">Scene composition, domain randomization, curriculum-aware augmentation</td>
<td align="left">Distributed training, RL, hyperparameter search, checkpoint promotion</td>
<td align="left">Physics-accurate validation, adversarial scenarios, regression gating</td>
<td align="left">Domain adaptation, safety monitoring, digital twin sync, deployment scoring</td>
</tr>
<tr>
<td align="left"><i><a href="https://docs.isaacsim.omniverse.nvidia.com/latest/index.html">Isaac Sim</a> + <a href="https://www.nvidia.com/en-us/ai/cosmos/">Cosmos</a></i></td>
<td align="left"><i><a href="https://developer.nvidia.com/isaac/gr00t">GR00T</a>, <a href="https://isaac-sim.github.io/IsaacLab/main/index.html">Isaac Lab</a></i></td>
<td align="left"><i><a href="https://docs.isaacsim.omniverse.nvidia.com/latest/index.html">Isaac Sim</a></i></td>
<td align="left"><i><a href="https://developer.nvidia.com/embedded-computing">Jetson</a> / <a href="https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/">RTX</a></i></td>
</tr>
</table>

<p align="center"><b>Data → Train → Validate → Deploy → Feedback → Generate</b> — a closed loop of continuous model improvement</p>

**Orchestration** spans the entire cycle — [NVIDIA OSMO](https://nvidia.github.io/OSMO/main/user_guide/index.html) coordinates task scheduling, data flow, dependency resolution, and resource allocation across heterogeneous compute — training GPUs, simulation hardware, and edge devices — from a single declarative pipeline definition.

## Toolchain Components

<table>
<tr>
<th width="220">Component</th>
<th width="500">Description</th>
<th width="100">Status</th>
</tr>
<tr>
<td><a href="osmo-on-aws/"><b>osmo-on-aws</b></a></td>
<td><a href="https://nvidia.github.io/OSMO/main/user_guide/index.html">NVIDIA OSMO</a> 6.3 on Amazon EKS — orchestration control plane, compute plane, GPU scheduling, and example Physical AI workflows</td>
<td><img src="https://img.shields.io/badge/Available-brightgreen" alt="Available"/></td>
</tr>
<tr>
<td><i>nvidia-cosmos-on-aws</i></td>
<td><a href="https://www.nvidia.com/en-us/ai/cosmos/">NVIDIA Cosmos</a> world foundation model for synthetic data generation and scene understanding on AWS GPU instances</td>
<td><img src="https://img.shields.io/badge/Planned-blue" alt="Planned"/></td>
</tr>
<tr>
<td><i>isaac-sim-on-aws</i></td>
<td><a href="https://docs.isaacsim.omniverse.nvidia.com/latest/index.html">NVIDIA Isaac Sim</a> headless rendering and synthetic data generation pipelines on EC2 GPU instances</td>
<td><img src="https://img.shields.io/badge/Planned-blue" alt="Planned"/></td>
</tr>
<tr>
<td><i>isaac-lab-on-aws</i></td>
<td><a href="https://developer.nvidia.com/isaac/lab">NVIDIA Isaac Lab</a> reinforcement learning and simulation validation environments on Amazon EKS with GPU node groups</td>
<td><img src="https://img.shields.io/badge/Planned-blue" alt="Planned"/></td>
</tr>
<tr>
<td><i>isaaclab-arena-on-aws</i></td>
<td><a href="https://developer.nvidia.com/isaac/lab-arena">NVIDIA Isaac Lab Arena</a> for Physical AI model evaluation, benchmarking, and regression testing on AWS GPU compute</td>
<td><img src="https://img.shields.io/badge/Planned-blue" alt="Planned"/></td>
</tr>
<tr>
<td><i>gr00t-training-on-aws</i></td>
<td><a href="https://developer.nvidia.com/isaac/gr00t">GR00T</a> foundation model fine-tuning with EFA-enabled multi-node distributed training</td>
<td><img src="https://img.shields.io/badge/Planned-blue" alt="Planned"/></td>
</tr>
<tr>
<td><i>jetson-edge-deployment</i></td>
<td>Model packaging and deployment to <a href="https://developer.nvidia.com/embedded-computing">Jetson</a> devices via Amazon EKS Hybrid Nodes</td>
<td><img src="https://img.shields.io/badge/Planned-blue" alt="Planned"/></td>
</tr>
<tr>
<td><i>agentic-layer-on-aws</i></td>
<td>Agentic AI orchestration layer built with Strands Agents SDK and Amazon Bedrock AgentCore for autonomous pipeline coordination</td>
<td><img src="https://img.shields.io/badge/Planned-blue" alt="Planned"/></td>
</tr>
</table>

## When to Use OSMO vs. Individual Tools

This toolchain supports two usage patterns depending on where you are in your Physical AI journey:

**Use OSMO (full orchestration)** when you need an end-to-end platform that manages the entire flywheel — scheduling tasks across heterogeneous compute, resolving data dependencies between stages, and routing workloads from cloud GPUs to edge devices. OSMO is the right choice when:

- You are building a new Physical AI pipeline from scratch
- You need a single control plane to coordinate SDG, training, simulation, and deployment
- You want declarative YAML-driven workflows with automatic dependency resolution
- You need multi-cluster orchestration (cloud + on-premises lab + edge)

**Use individual tools standalone** when you already have an established orchestration pipeline (e.g., Kubeflow, Airflow, Argo Workflows, or a custom CI/CD system) and want to integrate a specific NVIDIA capability into your existing infrastructure. Each tool in this toolchain is self-contained and can be deployed independently:

| Scenario | Recommended Approach |
|----------|---------------------|
| Greenfield Physical AI platform | Start with **osmo-on-aws** — it provides orchestration + compute for all stages |
| Existing pipeline, need SDG only | Deploy **isaac-sim-on-aws** as a standalone service, call it from your orchestrator |
| Existing pipeline, need distributed training | Deploy **groot-training-on-aws**, submit jobs via your scheduler |
| Existing pipeline, need edge deployment | Use **jetson-edge-deployment** to package and push models to devices |
| Migrating from scripts to managed orchestration | Start with **osmo-on-aws**, then migrate stages incrementally |

The tools are designed to be composable — you can start with one standalone component and adopt OSMO later as your orchestration needs grow, or use OSMO from day one and let it manage all stages.

## How It Maps to AWS

Each stage of the flywheel maps to specific AWS services:

| Flywheel Stage | AWS Compute | Supporting Services |
|----------------|-------------|---------------------|
| **Synthetic Data Generation** | Amazon EC2 (G6e / L40S GPUs) on EKS | Amazon S3, Amazon ECR, [NVIDIA NGC](https://catalog.ngc.nvidia.com/) |
| **Model Training** | Amazon EC2 (P5 / P6 with EFA) on EKS | Amazon S3, Amazon FSx for Lustre |
| **SIL Simulation** | Amazon EC2 (G6e GPUs) on EKS | Amazon S3, Amazon CloudWatch |
| **HIL / Edge Deployment** | EKS Hybrid Nodes (Jetson, RTX workstations) | AWS Site-to-Site VPN, AWS Direct Connect |
| **Orchestration (OSMO)** | Amazon EKS (system nodes) | Amazon RDS, ElastiCache, S3, Secrets Manager, KMS |

## Getting Started

Start with the OSMO orchestration layer — it provides the control plane that coordinates all other stages:

```bash
cd osmo-on-aws/
cat README.md          # Full deployment guide
```

The [osmo-on-aws](osmo-on-aws/) component includes:
- Terraform IaC for VPC, EKS, RDS, ElastiCache, S3, KMS, and observability
- Helm-based deployment scripts for OSMO control plane and compute plane
- Example OSMO workflows for Cosmos Transfer SDG, Isaac Sim rendering, and pick-and-place training
- Keycloak identity provider integration
- GPU node groups with NVIDIA GPU Operator and KAI Scheduler

## Prerequisites

- AWS account with permissions to create VPCs, EKS clusters, RDS, and GPU instances
- NVIDIA NGC API key (for pulling OSMO and Isaac Sim container images)
- Terraform >= 1.5, kubectl, Helm 3, AWS CLI v2
- (Optional) Route 53 hosted zone for custom domain names

## License

Apache License 2.0 — see [osmo-on-aws/LICENSE-2.0.txt](osmo-on-aws/LICENSE-2.0.txt).

## Authors

- **Abhishek Srivastav** — Principal Solutions Architect, AWS
- **Jathavan Sriram** — Senior Solutions Architect, NVIDIA
