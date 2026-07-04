# Physical AI Toolchain — Notebook Labs

A set of Jupyter notebooks for fine-tuning NVIDIA GR00T robot foundation models on AWS
SageMaker, running Isaac Lab RL training, generating photorealistic training scenes with
Cosmos, and visually debugging robot environments in Isaac Sim on EC2.

---

## Labs at a glance

| Lab | What it does | Key notebook(s) | Instance type | Cost estimate |
|-----|-------------|-----------------|---------------|--------------|
| **Lab 1** — GR00T Fine-Tuning | Fine-tunes GR00T N1.6-3B on 27 UR3 pick-and-place demos; runs open-loop inference to validate | `Lab0_Build_Container.ipynb`, `Lab1_Training_Inference.ipynb` | `ml.g5.2xlarge` (SageMaker) | ~$1-2 (smoke test, 200 steps) |
| **Lab 2** — Isaac Sim Workstation | Deploys a GPU EC2 instance with Isaac Sim for visual debugging of RL environments | `Lab2_Isaac_Workstation.ipynb` | `g6e.4xlarge` (EC2) | ~$2.20/hr while running |
| **Lab 3** — Cosmos World Generation | Restyling sim video to photorealistic using NVIDIA Cosmos Transfer NIM on Spot p5 | `Lab3_Cosmos_World_Generation.ipynb` | `p5.48xlarge` Spot (EC2) | ~$13.58/hr (~$4-6/session) |
| **Lab 4** — Isaac Lab RL Refinement | Builds Isaac Lab container, runs RL training (Anymal-D / UR10), renders video | `Lab4_0_Build_Container.ipynb`, `Lab4_RL_Training.ipynb` | `ml.g5.xlarge` / `ml.g5.12xlarge` (SageMaker) | ~$1-5 (smoke test to full run) |

---

## Getting started

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| AWS account | With SageMaker, EC2, ECR, CodeBuild, S3, Secrets Manager, CloudWatch, SSM, and IAM access |
| SageMaker execution role | Permissions vary by lab — see each lab's `README.md` for the exact policy list |
| HuggingFace account + token | Required for Labs 1: to download `nvidia/GR00T-N1.6-3B` base weights |
| NGC API key | Required for Labs 3 and 4: to pull NVIDIA container images from `nvcr.io`. Must be an `nvapi-...` Personal Key from https://org.ngc.nvidia.com/setup/personal-keys with NGC Catalog scope |

### Setup

1. Clone this repository onto your SageMaker JupyterLab instance.
2. Open the lab folder for the lab you want to run.
3. Read the lab's `README.md` for prerequisites and IAM requirements.
4. Follow the notebooks in order.

---

## Lab sequence

The labs are largely independent but have the following recommended sequence:

```
Lab 1 (GR00T fine-tuning)
    │
    ▼
Lab 2 (Isaac Sim workstation) ──────────────────────────► Lab 4 (RL training)
                                                                │
Lab 3 (Cosmos scene generation) ──► (optional input to Lab 4)  │
                                                                ▼
                                                         Trained RL policy
```

- **Lab 1 before Lab 4**: The GR00T→MLP bridge connecting Lab 1 output to Lab 4 is
  planned future work. Labs 1 and 4 currently run independently.
- **Lab 2 before Lab 4**: Strongly recommended for visually validating the UR3
  pick-and-place environment in Isaac Sim before training at scale.
- **Lab 3 is optional**: Lab 4 works without Cosmos using Isaac Lab's built-in domain
  randomization. Add Lab 3 only if sim-to-real transfer fails on real hardware due to
  visual domain gap.

---

## Repository layout

```
notebooks/
├── README.md                          ← this file
├── lab1/                              ← GR00T fine-tuning
│   ├── README.md
│   ├── Lab0_Build_Container.ipynb
│   ├── Lab1_Training_Inference.ipynb
│   └── Lab1_Inference.ipynb
├── lab2/                              ← Isaac Sim workstation
│   ├── README.md
│   └── Lab2_Isaac_Workstation.ipynb
├── lab3/                              ← Cosmos world generation
│   ├── README.md
│   ├── Lab3_Cosmos_World_Generation.ipynb
│   └── Lab3b_Cosmos3_World_Generation.ipynb
└── lab4/                              ← Isaac Lab RL training
    ├── README.md
    ├── Lab4_0_Build_Container.ipynb
    ├── Lab4_RL_Training.ipynb
    └── run_pipeline.py
```
