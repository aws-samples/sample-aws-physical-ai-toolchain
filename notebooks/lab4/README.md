# Lab 4 — RL Refinement with Isaac Lab

Improve a GR00T policy using Reinforcement Learning in Isaac Lab simulation.
Domain randomization + PPO trains the policy to handle scene variations it never
saw in the demonstrations.

---

## What this lab does

Lab 4 builds an Isaac Lab Docker container and runs RL training jobs on SageMaker
using G-family GPU instances. It validates the container with a built-in Anymal-D
locomotion task, then runs UR10 arm reaching training. It also renders video of
trained policies. The UR3 pick-and-place environment is written but not yet wired
into the container (future work).

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Lab4_0_Build_Container.ipynb` | Builds the Isaac Lab training container (~60-90 min, one-time) and pushes to ECR via CodeBuild. Handles NGC key setup and IAM policy. |
| `Lab4_RL_Training.ipynb` | Runs Anymal-D smoke test, UR10 arm reaching training, monitors jobs, downloads checkpoints, and renders video of trained policies. |

## Supporting scripts

| Script | Description |
|--------|-------------|
| `run_pipeline.py` | CLI pipeline runner: build container → RL training → render video → download MP4. Supports `--rebuild`, `--skip-training`, `--skip-video` flags. |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| AWS account | With SageMaker, ECR, CodeBuild, S3, Secrets Manager, and CloudWatch access |
| SageMaker execution role | Same as Lab 1 plus `secretsmanager:GetSecretValue` on `ngc-api-key` |
| GPU quota | `ml.g5.xlarge` (1× A10G 24 GB) for smoke test; `ml.g5.12xlarge` (4× A10G) for full training |
| NGC API key | Required to pull the 16 GB Isaac Lab base image. Store in Secrets Manager as `ngc-api-key`. |

---

## Dependencies

- **Lab 1** is not a hard dependency — RL training runs standalone today.
  The GR00T→MLP bridge connecting Lab 1 output to Lab 4 is future work.
- **Lab 2** (Isaac Sim workstation) is recommended before Lab 4 to visually
  validate the UR3 pick-and-place environment before training at scale.

---

## Instance types launched

| Notebook | Instance | Purpose |
|----------|----------|---------|
| `Lab4_0_Build_Container.ipynb` | CodeBuild `BUILD_GENERAL1_2XLARGE` | Docker image build (~60-90 min) |
| `Lab4_RL_Training.ipynb` | `ml.g5.xlarge` | Anymal-D smoke test + video render |
| `Lab4_RL_Training.ipynb` | `ml.g5.12xlarge` | UR10 / full RL training |

---

## Key notes

- **G-family instances only.** P-family (P4, P5) instances lack RT Cores required
  by Isaac Sim and will crash.
- Validated tasks: `Isaac-Velocity-Flat-Anymal-D-v0` (locomotion) and
  `Isaac-Reach-UR10-v0` (arm reaching).
- `PickAndPlaceUR3-v0` is written in `training/envs/pick_and_place_ur3.py` but
  is not yet wired into the container entrypoint (future work).
