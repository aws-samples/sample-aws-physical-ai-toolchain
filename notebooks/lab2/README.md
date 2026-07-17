# Lab 2 — Isaac Sim Development Workstation

Deploy a GPU-powered remote desktop for visual development and debugging of RL
robot environments. Use this before Lab 4 (RL training on SageMaker) to validate
that your Isaac Lab environment works correctly.

---

## What this lab does

Lab 2 deploys an EC2 `g6e.4xlarge` instance with Isaac Sim 5.1.0 pre-installed,
accessible via a NICE DCV remote desktop in the browser. It is used to visually
validate and debug Isaac Lab RL environments before running training at scale in
Lab 4, avoiding wasted compute on broken environments.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Lab2_Isaac_Workstation.ipynb` | Deploys the workstation via CDK, monitors bootstrap, provides the DCV URL, and manages the instance lifecycle (start/stop/destroy). |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| AWS account | With EC2, CloudFormation, IAM, S3, and SSM access |
| SageMaker execution role | Must have `CloudFormationFullAccess`, `AmazonEC2FullAccess`, and two CDK-specific inline policies: `CDKBootstrapIAM` (for `cdk-hnb659fds-*` roles) and `CDKBootstrapS3` (for `cdk-hnb659fds-assets-*` bucket). See README for exact JSON. |
| GPU quota | `g6e.4xlarge` (1× L40S 48 GB) — the Marketplace AMI only supports `g6e` instances |
| Node.js ≥ 18 | Required for CDK (`node --version` to check) |
| AWS Marketplace subscription | Subscribe to the NVIDIA Isaac Sim AMI before deploying: https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw |

---

## Dependencies

Lab 2 has no dependency on other labs and can be run independently.
It is recommended to run Lab 2 before Lab 4 to visually validate the UR3
pick-and-place environment before training.

---

## Instance types launched

| Resource | Instance / Type | Cost |
|----------|----------------|------|
| EC2 workstation | `g6e.4xlarge` (1× L40S 48 GB) | ~$2.20/hr while running |
| EBS storage | 512 GB gp3 | ~$41/month (always-on) |

---

## Key notes

- **Stop the instance when not debugging** — your work is preserved on EBS.
- NICE DCV uses port **8443**. Disconnect corporate VPN before connecting.
- Isaac Sim first launch takes **5–10 minutes** to compile shaders. If the
  dialog says "not responding", click **Wait** — it is still loading.
- Isaac Sim is at `~/IsaacSim`. Launch with `./run-isaac-sim-gui.sh`.
- `run-isaac-lab.sh` requires the `physical-ai/isaac-lab` ECR image from Lab 4.
