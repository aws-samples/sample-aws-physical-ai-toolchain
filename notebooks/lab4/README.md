# Lab 4 — RL Refinement with Isaac Lab

Improve the Lab 1 GR00T policy using Reinforcement Learning in Isaac Lab simulation.
Domain randomisation + PPO trains the policy to handle scene variations it never saw
in the demonstrations.

---

## Quick Start

1. Read `BACKGROUND.md` — explains RL vs imitation, Isaac Lab architecture, current limitations
2. Run `Lab4_0_Build_Container.ipynb` — builds the Isaac Lab ECR image (~60-90 min, one-time)
3. Run `Lab4_RL_Training.ipynb` — smoke test → UR3 RL training → download results

---

## Folder Structure

```
lab4/
├── README.md
├── BACKGROUND.md                      ← read before running
├── Lab4_0_Build_Container.ipynb       ← build Isaac Lab ECR image
├── Lab4_RL_Training.ipynb             ← RL training + evaluation
└── container/
    ├── Dockerfile                     ← extends nvcr.io/nvidia/isaac-lab:2.1.0
    ├── sm-train-entrypoint.sh         ← SageMaker entry point (parses resource config, torchrun)
    └── buildspec.yml                  ← CodeBuild spec (pulls NGC key from Secrets Manager)
```

The UR3 environment and PPO config live outside lab4 (shared with the rest of the toolchain):
```
training/envs/pick_and_place_ur3.py   ← RL environment definition
training/configs/ppo_pick_place.yaml  ← PPO hyperparameters
training/scripts/groot_to_rl_bridge.py ← GR00T → MLP bridge
```

---

## Current Status

| Component | Status |
|-----------|--------|
| Isaac Lab container (ECR) | 🔲 Needs CodeBuild (Lab4_0) |
| Anymal-D smoke test | ✅ Validated |
| UR3 pick-and-place env | ⚠️ Written, needs visual validation (Lab 2) |
| GR00T → MLP bridge | ⚠️ Partial (uses teleop data as proxy) |

---

## GPU Requirement

**G-family instances only.** P-family (P4, P5) instances lack RT Cores required by
Isaac Sim and will crash.

| Instance | GPUs | Use case |
|----------|------|---------|
| `ml.g5.xlarge` | 1× A10G 24GB | Smoke test |
| `ml.g5.12xlarge` | 4× A10G 96GB | Full training |

---

## IAM Requirements

Same as Lab 1 (CodeBuild + ECR + SageMaker).  
Additionally: `secretsmanager:GetSecretValue` on the NGC API key secret.
