# AWS Physical AI Toolchain — Project Plan

**Status:** Active development
**Last updated:** 2026-06-12
**Owner:** devris
**Repo:** aws-physical-ai-toolchain (internal GitLab — setup pending)

---

## Vision

A modular, CDK-deployable reference architecture for Physical AI on AWS. Customers choose their complexity level:

- **Path A (Simple):** GR00T/π0 fine-tuning from existing robot data on SageMaker. No EKS, no OSMO. Deploy in 5 minutes, train in hours.
- **Path B (Full):** Isaac Lab RL training orchestrated by OSMO on EKS. Cosmos scene generation, multi-stage pipelines, GPU autoscaling. Deploy in 20 minutes.

Both paths output the same artifact: a model deployable to edge hardware via Greengrass.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Foundation Stack (always deployed)                     │
│  S3 (datasets, models)  │  ECR (containers)  │  IAM (roles)             │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────────┐ ┌──────▼──────────┐ ┌──────▼──────────┐
│  Path A (V1)    │ │  Path B (V2)    │ │  Path C (V3)    │
│  Imitation      │ │  Simulation     │ │  Scale          │
│  Learning       │ │                 │ │                 │
│                 │ │ Cosmos (scenes) │ │ OSMO on EKS     │
│ SageMaker       │ │ Isaac Lab (RL)  │ │ Multi-node      │
│ Pipeline:       │ │                 │ │ distributed     │
│  Train GR00T    │ │ SageMaker /     │ │ training        │
│  → Eval         │ │ AWS Batch       │ │                 │
│  → Register     │ │ (single GPU)    │ │ (100+ envs)    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                      ┌──────▼──────┐
                      │ Model       │
                      │ Registry    │
                      └──────┬──────┘
                             │
                      ┌──────▼──────┐  (optional)
                      │    Edge     │
                      │ Greengrass  │
                      │ → Jetson    │
                      └─────────────┘
```

### Path A — What's deployed today (us-east-1):
- **S3:** `physical-ai-dev-datasets-802782083985` (datasets + model output)
- **ECR:** `physical-ai/groot-training` (training container, pushed)
- **IAM:** `physical-ai-dev-sagemaker-role` (SM execution role)
- **Pipeline:** `groot-finetune-pipeline` (train → register model)
- **Model Registry:** `groot-models` (versioned model packages)

Deploy command:
```bash
cdk deploy --context mode=simple   # Path A only (Foundation stack)
cdk deploy --context mode=full     # Path A + B (adds Network, EKS, OSMO)
```

---

## What Already Exists

### From this repo (`aws-physical-ai-toolchain`):
- ✅ CDK: Network, Storage, EKS, OSMO, Edge stacks (all synthesize)
- ✅ EKS cluster deployed and tested (v1.31, 3 nodes)
- ✅ RDS + Redis deployed for OSMO
- ✅ Isaac Lab RL environment (`training/envs/pick_and_place_ur3.py`)
- ✅ Training, evaluation, export scripts
- ✅ Cosmos scene generation script
- ✅ Dockerfiles for Isaac Sim, Isaac Lab, Inference containers
- ✅ ROS2 TensorRT inference node (200Hz, Jetson + x86)
- ✅ Greengrass component recipes (CDK inline)
- ✅ OSMO workflow YAML

### From hackathon repo (`physical-ai-toolkit`):
- ✅ GR00T fine-tuning on SageMaker (complete pipeline)
- ✅ CDK: S3, ECR, SageMaker role, CodeBuild projects
- ✅ Zarr → LeRobot v2 conversion
- ✅ CLI with dry-run, cost estimation, prerequisite checks
- ✅ IDE skill pack (10 skills)
- ✅ 5 training paths (real robot, Gazebo, Phantom, Isaac ACT, Isaac RL)

---

## Build Order

### Phase 1: Path A — GR00T on SageMaker (V1 launch) ← CURRENT FOCUS

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Restructure CDK with `mode` context flag | ✅ | `app.ts` routes simple/full |
| 2 | Create `foundation-stack.ts` (S3, ECR, SageMaker role) | ✅ | Deployed to us-east-1 |
| 3 | Create training launch script + upload helpers | ✅ | `launch_training.py`, `upload_dataset.py` |
| 4 | Build GR00T training container + push to ECR | ✅ | 7.1 GB image, pushed |
| 5 | Bundle demo dataset (download script + docs) | ✅ | `lerobot/aloha_sim_insertion_human`, 87 MB |
| 6a | End-to-end smoke test: deploy → train → model to S3 | ✅ | 100-step job completed successfully |
| 6b | Eval report: action prediction error on held-out data | 🚧 | Code done, container pushed, awaiting test run |
| 6c | Eval video: sim rollout with Isaac-GR00T SDK | 🔲 | Needs SDK installed from source in container (see note) |
| 7 | Workshop Lab 1 docs | ✅ | `workshop/lab-1-train-groot.md` |
| 8 | SageMaker Pipeline: train → register model | ✅ | `groot-finetune-pipeline` created, executing |
| 9 | Polish README + getting-started for GitLab review | 🔲 | |

**Note on 6c (eval video):** Clone the [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) repo into the container, install via `uv sync`, then use `standalone_inference_script.py` for open-loop rollouts on dataset trajectories. Not on PyPI — install from source. Good task for a contributor.

### Phase 2: Simulation + Synthetic Data (V2)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 10 | Isaac Lab RL training on SageMaker (no EKS needed) | 🔲 | Existing env + scripts, just needs container on SM/Batch |
| 11 | Cosmos scene generation on SageMaker/Batch | 🔲 | `generate_scenes.py` exists, needs Cosmos API access |
| 12 | Build Isaac Lab container, push to ECR | 🔲 | Dockerfile exists, untested |
| 13 | Build Isaac Sim container (for Cosmos), push to ECR | 🔲 | Dockerfile exists, untested |
| 14 | Pipeline: Cosmos scenes → Isaac train → eval → register | 🔲 | Extend `pipeline.py` or new SM Pipeline |
| 15 | Workshop Lab 2 docs | 🔲 | |

**Key insight:** Isaac Lab and Cosmos both run as single GPU jobs. They don't need EKS or OSMO — a SageMaker Training Job or AWS Batch job works fine. OSMO is only needed if you want multi-node distributed training or complex DAG orchestration across hundreds of jobs. For V2, SageMaker Pipeline is the orchestrator.

### Phase 2b: OSMO on EKS (V3 — only if needed for scale)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 16 | Wire EKS + OSMO stacks behind `mode=full` flag | ✅ | CDK synths all 5 stacks |
| 17 | Fix OSMO Helm deployment | 🔲 | Helm chart not publicly stable |
| 18 | Submit OSMO workflow, validate multi-stage pipeline | 🔲 | `workflows/pick-and-place.yaml` ready |

**When to use OSMO:** 100+ parallel sim environments, distributed RL, or complex multi-stage pipelines with branching logic. Not needed for single-job training or small-scale scene generation.

### Phase 3: Edge Deployment

| # | Task | Status | Notes |
|---|------|--------|-------|
| 19 | Test Edge stack deployment (IoT Core + Greengrass) | 🔲 | CDK exists: `edge-stack.ts` |
| 20 | Build inference container, push to ECR | 🔲 | Dockerfile exists (Jetson + x86) |
| 21 | Test Greengrass component deployment | 🔲 | Needs Jetson or simulated device |
| 22 | Workshop Lab 3 docs | 🔲 | |

### Phase 4: Developer Experience

| # | Task | Status | Notes |
|---|------|--------|-------|
| 23 | Port CLI from hackathon repo | 🔲 | `physical-ai-cli/` has dry-run, cost est |
| 24 | Port IDE skills | 🔲 | 10 skills in hackathon `skills/` |
| 25 | Add WebRTC viz option | 🔲 | Roy Allela's pattern |

---

## Deferred (V2+)

- MCAP ingestion / streaming (Kinesis, IoT Core data path)
- Data management platform (SDMA / Foxglove / Roboto)
- AI-assisted annotation (Ground Truth + Bedrock)
- Dataset catalog and lineage
- Fleet management at scale (100+ robots)
- Provider abstraction (MuJoCo, Gazebo, Genesis)
- Step Functions as alternative to OSMO
- Trainium support

---

## Workshop / Immersion Day Structure

This repo doubles as a hands-on workshop. Each phase maps to a lab:

### Lab 0: Prerequisites (30 min)
- AWS account setup, GPU quota request
- Install CDK, Docker, NGC account
- Clone repo

### Lab 1: Train Your First Robot Policy (2 hrs) — Path A
- Deploy foundation stack (`cdk deploy --context mode=simple`)
- Explore the demo dataset (LeRobot format)
- Build + push training container
- Launch SageMaker training job
- Monitor loss curve in CloudWatch/MLflow
- Download eval video, interpret results
- **Checkpoint:** "I trained a GR00T model on AWS"

### Lab 2: Simulation-Based Training at Scale (3 hrs) — Path B
- Deploy full stack (`cdk deploy --context mode=full`)
- Explore the Isaac Lab RL environment code
- Submit OSMO workflow (scene gen → train → eval → export)
- Compare RL-trained policy vs. imitation-learned policy
- **Checkpoint:** "I ran a multi-stage sim training pipeline with OSMO"

### Lab 3: Deploy to Edge (1.5 hrs)
- Export trained model to TensorRT
- Deploy Greengrass component to a simulated device (or real Jetson)
- Verify ROS2 inference node publishes joint commands
- **Checkpoint:** "My trained model is running on edge hardware"

### Lab 4: Extend the Platform (1 hr) — Self-guided
- Modify the RL environment (change reward, add obstacles)
- Try a different robot (swap URDF)
- Enable Cosmos scene generation
- Connect the CLI / IDE skills

### Key workshop design principles:
- Each lab is **independent** — you can skip Lab 2 if you only want Path A
- Each lab has a clear **checkpoint** (verifiable outcome)
- All labs work **without physical hardware** (sim only)
- Labs are structured as: **Concept (10 min) → Deploy (15 min) → Experiment (rest)**
- Code is annotated with `# WORKSHOP NOTE:` comments at decision points

---

## File Structure (Actual)

```
aws-physical-ai-toolchain/
├── PLAN.md                          # This file
├── README.md                        # Getting started
├── TODO.md                          # Historical build/test notes
├── run-path-a.sh                    # Quick-start script (one-off training)
├── cdk/
│   ├── bin/app.ts                   # Mode switch: simple | full
│   ├── lib/
│   │   ├── foundation-stack.ts      # S3, ECR, IAM (always deployed)
│   │   ├── network-stack.ts         # VPC (mode=full only)
│   │   ├── storage-stack.ts         # Additional S3/ECR for OSMO (mode=full)
│   │   ├── eks-cluster-stack.ts     # EKS + GPU nodes (mode=full)
│   │   ├── osmo-stack.ts           # RDS, Redis, OSMO (mode=full)
│   │   └── edge-stack.ts           # IoT Core + Greengrass (optional)
│   └── config/
│       ├── dev.ts                   # Dev: g5.xlarge, single-AZ
│       └── prod.ts                  # Prod: P5e, multi-AZ
├── containers/
│   ├── groot-training/              # GR00T fine-tuning (Path A) ← pushed to ECR
│   │   ├── Dockerfile
│   │   └── train_entrypoint.py      # Train + eval report
│   ├── isaac-sim/Dockerfile         # Scene generation (Path B, untested)
│   ├── isaac-lab/Dockerfile         # RL training (Path B, untested)
│   └── inference/Dockerfile         # Edge inference (Jetson + x86)
├── training/
│   ├── groot/                       # Path A scripts
│   │   ├── launch_training.py       # One-off SageMaker job
│   │   ├── pipeline.py             # SageMaker Pipeline (train → register)
│   │   ├── download_demo_dataset.py
│   │   └── upload_dataset.py
│   ├── scripts/                     # Path B (Isaac Lab)
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export.py
│   │   └── generate_scenes.py
│   ├── envs/                        # Isaac Lab RL environments
│   │   └── pick_and_place_ur3.py
│   └── configs/
│       └── ppo_pick_place.yaml
├── workflows/
│   └── pick-and-place.yaml          # OSMO workflow (Path B)
├── edge/
│   ├── entrypoint.sh
│   └── ros2-workspace/src/ur3_inference/
├── workshop/
│   ├── lab-0-prerequisites.md
│   └── lab-1-train-groot.md
└── docs/
    └── glossary.md
```

---

## How to Pick Up This Project

If you're a new builder joining this repo:

1. **Read this PLAN.md** — you're here
2. **Check the Build Order table** — find the next `🔲` task
3. **Look at TODO.md** — has historical context on what worked/failed during prior testing
4. **The two source repos** to understand lineage:
   - `physical-ai-toolkit` (hackathon) — the GR00T/SageMaker path + CLI + skills
   - This repo — the Isaac Lab/OSMO/EKS path + Greengrass edge
5. **Key decision:** We're merging both into this repo as Path A and Path B

---

## Open Questions

- [ ] Robot for demo: UR3 (existing code) vs. Franka (better Isaac Lab support) vs. SO-100 (cheapest, LeRobot native)?
  - Current demo uses ALOHA sim (LeRobot dataset). SO-100 would be cheapest for physical hardware demos.
- [x] ~~SageMaker Pipeline vs. simple sequential Training Jobs for Path A?~~ → **Both.** `launch_training.py` for quick one-off runs, `pipeline.py` for production repeatable workflows.
- [ ] Include MLflow tracking in V1 or defer?
- [ ] AWS account for workshop: shared or per-participant?
- [ ] GitLab repo location — which team namespace?

## Decisions Made (2026-06-12)

- **OSMO deferred** — Helm chart not stable, expensive to keep EKS running. Path B code stays in repo as reference.
- **Eval report over eval video for V1** — Action prediction error (MSE on held-out episodes) validates training without needing sim. Real sim rollout video requires Isaac-GR00T SDK from source.
- **boto3 for Pipeline definition** — SageMaker Python SDK v3 restructured the workflow module. Using boto3 `create_pipeline` API directly is more stable and has no package dependency issues.
- **100-step smoke tests** — Full training (5000 steps, 11 hrs, $79) is for final demos. 100-step runs (~15 min, ~$2) validate the pipeline.
