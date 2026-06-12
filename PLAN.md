# AWS Physical AI Toolchain — Project Plan

**Status:** Active development
**Last updated:** 2026-06-12
**Owner:** devris
**Repo:** aws-physical-ai-toolchain

---

## Vision

A modular, CDK-deployable reference architecture for Physical AI on AWS. Customers choose their complexity level:

- **Path A (Simple):** GR00T/π0 fine-tuning from existing robot data on SageMaker. No EKS, no OSMO. Deploy in 5 minutes, train in hours.
- **Path B (Full):** Isaac Lab RL training orchestrated by OSMO on EKS. Cosmos scene generation, multi-stage pipelines, GPU autoscaling. Deploy in 20 minutes.

Both paths output the same artifact: a TensorRT model deployable to edge hardware via Greengrass.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Foundation (always deployed)                   │
│  S3 (datasets, models, checkpoints) │ ECR (containers) │ IAM    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │                               │
    ┌──────▼──────┐                 ┌──────▼──────┐
    │   Path A    │                 │   Path B    │
    │  (simple)   │                 │   (full)    │
    │             │                 │             │
    │ SageMaker   │                 │ EKS + OSMO  │
    │ Training    │                 │ Isaac Lab   │
    │ Jobs        │                 │ Cosmos      │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                    ┌──────▼──────┐
                    │   Output    │
                    │ model.trt   │
                    │ in S3       │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐  (optional)
                    │    Edge     │
                    │ Greengrass  │
                    │ → Jetson    │
                    └─────────────┘
```

Deploy command:
```bash
cdk deploy --context mode=simple   # Path A only
cdk deploy --context mode=full     # Path A + B
```

---

## What Already Exists (from prior work)

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

### Phase 1: Path A — GR00T on SageMaker (V1 launch)

| # | Task | Source | Status |
|---|------|--------|--------|
| 1 | Restructure CDK with `mode` context flag | New | ✅ |
| 2 | Create `foundation-stack.ts` (S3, ECR, SageMaker role) | Port from hackathon `physical-ai-stack.ts` | ✅ |
| 3 | Create training launch script + upload helpers | New | ✅ |
| 4 | Build GR00T training container + push to ECR | Port from hackathon CodeBuild pattern | ✅ |
| 5 | Bundle demo dataset (download script + docs) | New | ✅ |
| 6 | End-to-end test: deploy → train → get eval video | — | 🔲 |
| 7 | Workshop Lab 1 docs | New | ✅ |

### Phase 2: Path B — Isaac Lab + OSMO on EKS

| # | Task | Source | Status |
|---|------|--------|--------|
| 8 | Wire EKS + OSMO stacks behind `mode=full` flag | Existing code, refactor `app.ts` | 🔲 |
| 9 | Fix OSMO Helm deployment (values tuning or raw manifests) | Continue from TODO Phase 4 | 🔲 |
| 10 | Build Isaac Sim + Isaac Lab containers, push to ECR | Existing Dockerfiles | 🔲 |
| 11 | Submit OSMO workflow, validate training converges | Existing `workflows/pick-and-place.yaml` | 🔲 |
| 12 | Workshop Lab 2 docs | New | 🔲 |

### Phase 3: Edge Deployment

| # | Task | Source | Status |
|---|------|--------|--------|
| 13 | Test Edge stack deployment (IoT Core + Greengrass) | Existing `edge-stack.ts` | 🔲 |
| 14 | Build inference container, push to ECR | Existing Dockerfile | 🔲 |
| 15 | Test Greengrass component deployment (sim or real Jetson) | — | 🔲 |
| 16 | Workshop Lab 3 docs | New | 🔲 |

### Phase 4: Developer Experience

| # | Task | Source | Status |
|---|------|--------|--------|
| 17 | Port CLI from hackathon repo | Existing `physical-ai-cli/` | 🔲 |
| 18 | Port IDE skills | Existing `skills/` | 🔲 |
| 19 | Add WebRTC viz option (Optional C) | Roy Allela's pattern | 🔲 |

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

## File Structure (Target)

```
aws-physical-ai-toolchain/
├── PLAN.md                          # This file
├── README.md                        # Getting started (update for dual-path)
├── cdk/
│   ├── bin/app.ts                   # Mode switch: simple | full
│   ├── lib/
│   │   ├── foundation-stack.ts      # NEW: S3, ECR, IAM (shared)
│   │   ├── training-stack.ts        # NEW: SageMaker Pipeline (Path A)
│   │   ├── network-stack.ts         # Existing (Path B only)
│   │   ├── storage-stack.ts         # Existing (refactor: shared portions move to foundation)
│   │   ├── eks-cluster-stack.ts     # Existing (Path B only)
│   │   ├── osmo-stack.ts            # Existing (Path B only)
│   │   └── edge-stack.ts            # Existing (optional, both paths)
│   └── config/
│       ├── dev.ts                   # Dev config (both modes)
│       └── prod.ts                  # Prod config
├── containers/
│   ├── groot-training/              # NEW: GR00T fine-tuning container (Path A)
│   ├── isaac-sim/                   # Existing (Path B)
│   ├── isaac-lab/                   # Existing (Path B)
│   └── inference/                   # Existing (both paths)
├── training/
│   ├── groot/                       # NEW: GR00T fine-tuning scripts (Path A)
│   │   ├── train_groot.py
│   │   ├── convert_episodes.py
│   │   └── configs/
│   ├── isaac-lab/                   # Rename from training/ (Path B)
│   │   ├── scripts/
│   │   ├── envs/
│   │   └── configs/
│   └── export.py                    # Shared: PyTorch → ONNX → TRT
├── workflows/
│   └── pick-and-place.yaml          # OSMO workflow (Path B)
├── edge/                            # Existing (Greengrass + ROS2)
├── workshop/                        # NEW: Lab guides
│   ├── lab-0-prerequisites.md
│   ├── lab-1-train-groot.md
│   ├── lab-2-isaac-lab-osmo.md
│   ├── lab-3-edge-deployment.md
│   └── lab-4-extend.md
└── docs/
    ├── architecture.md
    └── troubleshooting.md
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
- [ ] SageMaker Pipeline vs. simple sequential Training Jobs for Path A? (Pipeline adds visibility but more CDK code)
- [ ] Include MLflow tracking in V1 or defer?
- [ ] AWS account for workshop: shared or per-participant?
