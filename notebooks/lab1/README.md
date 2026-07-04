# Lab 1 — GR00T Fine-Tuning

Fine-tune NVIDIA GR00T N1.6-3B on 27 UR3 pick-and-place teleoperation demos,
then validate the model with open-loop inference.

---

## What this lab does

Lab 1 fine-tunes NVIDIA's GR00T N1.6-3B robot foundation model on 27 human
teleoperation demonstrations of a UR3 pick-and-place task. After training, it
runs open-loop evaluation that compares predicted joint velocity curves against
ground-truth, confirming the model has learned from the data.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Lab0_Build_Container.ipynb` | Builds a Docker image with the Isaac-GR00T SDK and pushes it to ECR via CodeBuild. Run once before training. |
| `Lab1_Training_Inference.ipynb` | Uploads the dataset to S3, launches a SageMaker training job, downloads the checkpoint, and runs open-loop inference. |
| `Lab1_Inference.ipynb` | Standalone inference notebook — download and evaluate a previously trained checkpoint without re-running training. |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| AWS account | With SageMaker, ECR, S3, and CodeBuild access |
| SageMaker execution role | `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `AWSCodeBuildDeveloperAccess`, `AmazonEC2ContainerRegistryFullAccess`, plus `iam:PassRole` inline policy and CodeBuild trust relationship |
| GPU quota | `ml.g5.2xlarge` (1× A10G 24 GB) for training; `ml.g5.12xlarge` for full fine-tuning |
| HuggingFace token | Required to download base model weights (`nvidia/GR00T-N1.6-3B`). Create `notebooks/lab1/.env` with `HF_TOKEN=hf_your_token`. |

---

## Dependencies

Lab 1 has no dependency on other labs. It is the starting point of the pipeline.

---

## Instance types launched

| Notebook | Instance | Purpose |
|----------|----------|---------|
| `Lab0_Build_Container.ipynb` | CodeBuild `BUILD_GENERAL1_XLARGE` | Docker image build |
| `Lab1_Training_Inference.ipynb` | `ml.g5.2xlarge` (smoke test) | SageMaker training job |

---

## Key notes

- Lab 0 only needs to be rerun if you change files in `container/`.
- The 200-step smoke test trains only the Projector layer (~6% of parameters).
- Inference output is comparison plots (PNG), not a video of the robot moving.
  For robot video, run Lab 4 (Isaac Lab RL) or Lab 2 (Isaac Sim workstation).
