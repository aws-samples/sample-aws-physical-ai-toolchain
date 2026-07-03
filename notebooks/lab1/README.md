# Lab 1 — GR00T Fine-Tuning

Fine-tune NVIDIA GR00T N1.7-3B on 27 UR3 pick-and-place teleoperation demos,
then validate the model with open-loop inference.

---

## Quick Start

**Step 1 — Read the background doc**
Open `BACKGROUND.md`. It explains the dataset, model architecture, training
objectives, and what inference produces. ~15 min read. Essential for newcomers.

**Step 2 — Build the container (one-time)**
Open `Lab0_Build_Container.ipynb` and run top to bottom.
Builds a Docker image with the Isaac-GR00T SDK pre-installed and pushes to ECR.
Takes ~20-30 min (CodeBuild in AWS). Only needs to be repeated if you change
files in `container/`.

**Step 3 — Train and infer**
Open `Lab1_Training_Inference.ipynb` and run top to bottom.
Uploads dataset to S3 → launches SageMaker training job → downloads model →
runs open-loop inference → shows comparison plots.

---

## Folder Structure

```
lab1/
├── README.md                         ← this file
├── BACKGROUND.md                     ← read before running notebooks
│
├── Lab0_Build_Container.ipynb        ← Notebook 1: build ECR image (one-time)
├── Lab1_Training_Inference.ipynb     ← Notebook 2: train + infer (rerunnable)
│
├── container/                        ← all Docker build files
│   ├── Dockerfile                    ← image definition (CUDA 12.4, PyTorch 2.5,
│   │                                    Isaac-GR00T SDK pre-installed)
│   ├── train_entrypoint_v2.py        ← runs inside container during SageMaker job
│   ├── ur3_modality_config.py        ← registers UR3 robot with GR00T SDK
│   └── video_utils_patched.py        ← PyAV fallback for video decoding
│
├── ur3_modality_config.py            ← copy used by Lab0 when packaging container
│
├── isaac-groot/                      ← Isaac-GR00T SDK (cloned from NVIDIA GitHub)
│   └── gr00t/
│       ├── experiment/
│       │   └── launch_finetune.py    ← main training entry point (NVIDIA code)
│       ├── eval/
│       │   └── open_loop_eval.py     ← inference evaluation (NVIDIA code)
│       └── data/
│           └── stats.py              ← dataset statistics computation
│
└── deleted/                          ← old files (safe to purge, kept for reference)
```

---

## What the notebooks use

### Lab0_Build_Container.ipynb
- Reads files from `container/` to build the Docker image
- Uses AWS CodeBuild (not local Docker — JupyterLab blocks outbound network in Docker builds)
- Pushes to ECR repo `physical-ai/groot-training`

### Lab1_Training_Inference.ipynb
- Reads dataset from `training/data/ur3_lerobot_dataset/` (outside lab1)
- Uploads dataset to S3
- Submits SageMaker Training Job using the ECR image built in Lab0
- The job runs `container/train_entrypoint_v2.py` which calls the Isaac-GR00T SDK
- Downloads the trained checkpoint from S3
- Runs inference via `lab1/isaac-groot/gr00t/eval/open_loop_eval.py`

---

## External Dependencies

| Dependency | Location | Notes |
|-----------|----------|-------|
| Training dataset | `training/data/ur3_lerobot_dataset/` | Outside lab1; 27 UR3 episodes |
| Isaac-GR00T SDK | `lab1/isaac-groot/` | Cloned from github.com/NVIDIA/Isaac-GR00T |
| Base model weights | HuggingFace (`nvidia/GR00T-N1.7-3B`) | Downloaded by SageMaker during job |
| AWS (S3, ECR, SageMaker, CodeBuild) | Cloud | Account + GPU quota required |

---

## AWS IAM Requirements

### For Lab1_Training_Inference.ipynb (training jobs)
The SageMaker execution role needs:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `AmazonEC2ContainerRegistryReadOnly`

### For Lab0_Build_Container.ipynb (container build) — additional
- Attached policy: `AWSCodeBuildDeveloperAccess`
- Inline policy: `{ "Action": "iam:PassRole", "Resource": "*", "Effect": "Allow" }`
- Trust relationship must include `codebuild.amazonaws.com`

---

## Key Design Decisions

**Why SageMaker instead of running locally?**
GR00T N1.7-3B has 3B parameters. With the projector + diffusion head trainable
(~1.6B trainable params), the optimizer states alone need ~13GB on top of the
7GB model weights. A single 24GB A10G (local L4 GPU) runs out of memory.
SageMaker `ml.g5.2xlarge` provides a clean 24GB A10G with no environment conflicts.

**Why CodeBuild instead of local Docker?**
SageMaker JupyterLab blocks outbound network inside `docker build`. The NVIDIA CUDA
base image is 5GB+ and the Isaac-GR00T SDK requires a git clone at build time.
CodeBuild runs on a large cloud instance with full network access.

**What is actually trained (smoke test, 200 steps)?**
Only the Projector layer (~201M of 3,144M total params = 6.4%). The visual encoder,
LLM backbone, and diffusion action head are all frozen. This fits in a single A10G.
For full fine-tuning, remove `--no-tune-diffusion-model` from `train_entrypoint_v2.py`
and use `ml.g5.12xlarge` (4× A10G).

**What does inference produce?**
Comparison plots (PNG): predicted joint velocity curves vs ground-truth.
Not a video of the robot moving. For robot video, see Lab 2 (Isaac Sim).
