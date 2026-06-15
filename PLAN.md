# AWS Physical AI Toolchain — Project Plan

**Status:** Active development
**Last updated:** 2026-06-12
**Owner:** devris
**Repo:** [gitlab.aws.dev/devris/aws-physical-ai-toolchain](https://gitlab.aws.dev/devris/aws-physical-ai-toolchain)

---

## Vision

A modular, CDK-deployable reference architecture for Physical AI on AWS. The platform implements the industry-standard pipeline for training robust robot policies:

1. **Imitation Learning** — Fine-tune a foundation model (GR00T) on teleoperation demonstrations
2. **RL Refinement** — Improve the policy in simulation (Isaac Lab) to handle variations it hasn't seen
3. **Domain Randomization** — Use Cosmos to generate diverse scenes so the policy transfers to real hardware
4. **Edge Deployment** — Deploy the final model to physical robots via Greengrass

These stages form a **pipeline**, not independent paths. A customer can stop at any stage:
- Stage 1 alone gets you a working policy (~70-80% success) in hours
- Stages 1+2+3 get you a production-grade policy (~95% success) in days
- Stage 4 puts it on hardware

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Foundation Stack (always deployed)                     │
│  S3 (datasets, models)  │  ECR (containers)  │  IAM (roles)             │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SageMaker Pipeline (orchestrator)                     │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Stage 1     │    │  Stage 2     │    │  Stage 3     │               │
│  │  GR00T       │───▶│  Cosmos      │───▶│  Isaac Lab   │               │
│  │  Fine-tune   │    │  Scene Gen   │    │  RL Refine   │               │
│  │  (imitation) │    │  (synthetic  │    │  (improve    │               │
│  │              │    │   data)      │    │   robustness)│               │
│  └──────┬───────┘    └──────────────┘    └──────┬───────┘               │
│         │                                       │                        │
│         ▼ (good enough?)                        ▼                        │
│  ┌──────────────┐                        ┌──────────────┐               │
│  │  Evaluate    │                        │  Evaluate    │               │
│  │  (MSE or     │                        │  (sim rollout│               │
│  │   rollout)   │                        │   success %) │               │
│  └──────┬───────┘                        └──────┬───────┘               │
│         │                                       │                        │
│         └───────────────────┬───────────────────┘                        │
│                             ▼                                            │
│                      ┌──────────────┐                                    │
│                      │  Register    │                                    │
│                      │  Model       │                                    │
│                      └──────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                      ┌──────────────┐  (optional)
                      │    Edge      │
                      │  Greengrass  │
                      │  → Jetson    │
                      └──────────────┘
```

**All stages run on SageMaker** (Training Jobs + Processing Jobs). No EKS needed unless you need 100+ parallel sim environments at scale (OSMO path, deferred to V3).

### What's deployed today (us-east-1):
- **S3:** `physical-ai-dev-datasets-802782083985` (datasets + model output)
- **ECR:** `physical-ai/groot-training` (training container, pushed)
- **IAM:** `physical-ai-dev-sagemaker-role` (SM execution role)
- **Pipeline:** `groot-finetune-pipeline` (Stage 1: train → register)
- **Model Registry:** `groot-models` (versioned model packages)

Deploy command:
```bash
cdk deploy --context mode=simple   # Foundation stack only
cdk deploy --context mode=full     # Foundation + EKS + OSMO (V3 scale path)
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

### Phase 1: Stage 1 — GR00T Imitation Learning (V1) ← CURRENT FOCUS

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Restructure CDK with `mode` context flag | ✅ | `app.ts` routes simple/full |
| 2 | Create `foundation-stack.ts` (S3, ECR, SageMaker role) | ✅ | Deployed to us-east-1 |
| 3 | Create training launch script + upload helpers | ✅ | `launch_training.py`, `upload_dataset.py` |
| 4 | Build GR00T training container + push to ECR | ✅ | 7.1 GB image, pushed |
| 5 | Bundle demo dataset (download script + docs) | ✅ | `lerobot/aloha_sim_insertion_human`, 87 MB |
| 6a | End-to-end smoke test: deploy → train → model to S3 | ✅ | 100-step job completed successfully |
| 6b | Eval report: action prediction error on held-out data | ✅ | Produces `eval_report.json` + `eval_action_error.png` in model artifact |
| 6c | Eval video: sim rollout with Isaac-GR00T SDK | 🔲 | Needs SDK installed from source in container (see note) |
| 7 | Workshop Lab 1 docs | ✅ | `workshop/lab-1-train-groot.md` |
| 8 | SageMaker Pipeline: train → register model | ✅ | `groot-finetune-pipeline` created, executing |
| 9 | Polish README + getting-started for GitLab review | 🔲 | |

**Note on 6c (eval video):** Clone the [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) repo into the container, install via `uv sync`, then use `standalone_inference_script.py` for open-loop rollouts on dataset trajectories. Not on PyPI — install from source. Good task for a contributor.

### Phase 2: Stage 2+3 — Cosmos Scene Gen + Isaac Lab RL Refinement (V2)

The key insight: Isaac Lab RL isn't a separate training path — it **refines** the policy from Stage 1. Cosmos generates scene variations so the refined policy generalizes to real hardware.

| # | Task | Status | Notes |
|---|------|--------|-------|
| 10 | Build Isaac Lab RL container via CodeBuild | 🚧 | CodeBuild project deployed, build triggered. Waiting for completion. |
| 10a | Verify Isaac Lab container in ECR | 🔲 | Check image exists, correct size |
| 10b | Deploy Isaac Sim development workstation (GPU EC2 + DCV) | 🔲 | **Required for debugging.** Developers need to see the sim to iterate on environments. Port from [aws-samples scaffolding kit](https://github.com/aws-samples/sample-physical-ai-scaffolding-kit/tree/main/isaacsim-workstation). |
| 10c | Test Isaac Lab as SageMaker Training Job (dry run) | 🔲 | Submit a short RL training job, verify it starts |
| 10d | Run RL refinement with GR00T checkpoint as init | 🔲 | Full pipeline: load Lab 1 model → RL refine → save |
| 11 | Build Cosmos/scene-gen container via CodeBuild | 🔲 | Uses `nvcr.io/nvidia/isaac-sim:4.5.0` base. Lower priority — procedural randomization works without it. |
| 11a | Test procedural scene generation (no Cosmos API) | 🔲 | `generate_scenes.py --no-cosmos` — just randomized USD scenes |
| 11b | Integrate Cosmos NIM API (V3, optional) | 🔲 | Requires NVIDIA NIM API access. Adds photorealistic textures to procedural scenes. Not needed for RL to work. |
| 12 | Extend SM Pipeline: train → RL refine → eval → register | 🔲 | Full V2 pipeline combining Stage 1 + Stage 3 |
| 13 | Workshop Lab 2 docs | ✅ | `workshop/lab-2-rl-refinement.md` |

**On Cosmos (tasks 11/11b):** Cosmos is optional. Isaac Lab's built-in procedural domain randomization (random object positions, lighting, textures) provides diversity for RL training without needing any external API. Cosmos adds photorealistic enhancement for better sim-to-real transfer — it's a V3 optimization, not a V2 requirement.

**Blocker: NGC API key.** ~~Tasks 10-13 are blocked on getting NGC credentials to pull base images.~~ Resolved — NGC key stored in Secrets Manager, CodeBuild uses it automatically.

**Pipeline flow (V2):**
```
SM Pipeline:
  Step 1: [Processing] Cosmos generates scene variations from base task
  Step 2: [Training]   GR00T fine-tune on teleop demos (existing Stage 1)
  Step 3: [Training]   Isaac Lab RL loads Stage 1 model, refines in sim with Cosmos scenes
  Step 4: [Processing] Evaluate refined policy (sim rollout success rate)
  Step 5: [Condition]  If success_rate > 90%...
  Step 6: [Register]   Register production-ready model
```

### Phase 2b: OSMO on EKS (V3 — scale path, deferred)

Only needed when single-GPU RL refinement isn't enough (100+ parallel environments, distributed training, multi-robot fleet policies).

| # | Task | Status | Notes |
|---|------|--------|-------|
| 16 | EKS + OSMO stacks (mode=full) | ✅ | CDK synths, previously deployed and tested |
| 17 | Fix OSMO Helm deployment | 🔲 | Helm chart not publicly stable |
| 18 | Submit OSMO workflow with distributed Isaac Lab | 🔲 | `workflows/pick-and-place.yaml` ready |

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
| 23 | Port CLI from hackathon repo | 🔲 | `physical-ai-cli/` — a wrapper CLI that simplifies common operations. Example commands: `pai train --dataset ./data --dry-run` (shows config + cost estimate without launching), `pai status` (check running jobs), `pai check` (verify prerequisites: GPU quota, Docker, NGC key, HF token). Saves users from writing raw `aws sagemaker` commands. |
| 24 | Port IDE skills | 🔲 | 10 Kiro skill files from hackathon `skills/` directory. AI-powered dev assistance specific to Physical AI: "convert my dataset to LeRobot format," "explain this reward function," "why did my training job fail," "what instance type should I use." Makes the toolchain accessible to developers who aren't ML experts. |
| 25 | Add WebRTC viz option | 🔲 | Stream Isaac Sim's renderer to a browser via WebRTC — lighter weight than full DCV workstation. User opens a URL, sees sim running in real-time. No GPU desktop client needed. Based on Roy Allela's pattern. Good for quick visual checks without spinning up a full workstation. |

### Stretch Goals (V4+ — for contributors)

| # | Task | Notes |
|---|------|-------|
| S1 | **π0 (Pi-Zero) training path** | Alternative foundation model to GR00T. Physical Intelligence's open-weights VLA. See [aws-samples scaffolding kit π0 sample](https://github.com/aws-samples/sample-physical-ai-scaffolding-kit/tree/main/samples/openpi-sample). Would add a second model option for imitation learning (Stage 1). |
| S2 | **Upgrade to Isaac Lab 3.0 + Newton physics** | Newton is the next-gen physics backend (faster, more accurate cloth/deformable sim). The [scaffolding kit newton-rl sample](https://github.com/aws-samples/sample-physical-ai-scaffolding-kit/tree/main/samples/newton-rl) runs Isaac Lab 3.0-beta1 on HyperPod. We'd adapt for SageMaker. Also uses RSL-RL (newer than our rl_games). |
| S3 | **SageMaker HyperPod support** | For teams that want persistent GPU clusters instead of ephemeral SageMaker Training Jobs. The scaffolding kit has a full HyperPod Slurm setup. |
| S4 | **Multi-robot / fleet training** | Train policies for multiple robot types simultaneously. |

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

This repo doubles as a hands-on workshop. Each lab maps to a pipeline stage:

### Lab 0: Prerequisites (30 min)
- AWS account setup, GPU quota request
- Install CDK, Docker, HuggingFace account
- Clone repo

### Lab 1: Train Your First Robot Policy (2 hrs) — Stage 1
- Deploy foundation stack (`cdk deploy --context mode=simple`)
- Explore the demo dataset (LeRobot format)
- Build + push training container
- Launch SageMaker training job (or trigger pipeline)
- Review eval report (action prediction error)
- **Checkpoint:** "I have a fine-tuned GR00T model that predicts robot actions"

### Lab 2: Refine in Simulation (3 hrs) — Stages 2+3
- Generate scene variations with Cosmos
- Load Stage 1 model into Isaac Lab sim environment
- Run RL refinement (policy improves via reward signal)
- Compare success rate: Stage 1 model vs. refined model
- **Checkpoint:** "My policy went from 70% to 95% success rate in sim"

### Lab 3: Deploy to Edge (1.5 hrs) — Stage 4
- Export refined model to TensorRT
- Deploy Greengrass component to a simulated device (or real Jetson)
- Verify ROS2 inference node publishes joint commands
- **Checkpoint:** "My trained model is running on edge hardware"

### Lab 4: Extend the Platform (1 hr) — Self-guided
- Modify the RL environment (change reward, add obstacles)
- Try a different robot (swap URDF)
- Add more Cosmos scene variations
- Run the full pipeline end-to-end

### Key workshop design principles:
- Labs are **incremental** — each builds on the previous (but Lab 1 works standalone)
- Each lab has a clear **checkpoint** (verifiable outcome)
- All labs work **without physical hardware** (sim only, except Lab 3 with real Jetson)
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
- [ ] **CodeBuild source:** Currently uses PLACEHOLDER GitHub source. When repo goes public, update `foundation-stack.ts` CodeBuild projects to point to the real repo URL so customers can trigger builds without the zip-to-S3 workaround.

## Decisions Made (2026-06-12)

- **Pipeline, not paths** — Isaac Lab RL refines the GR00T imitation model, it doesn't replace it. This is industry standard practice (imitation → RL refinement → domain randomization). The original "Path A vs Path B" framing was wrong.
- **SageMaker for everything (V1/V2)** — Isaac Lab and Cosmos both run as single-GPU SageMaker jobs. No EKS needed. Same orchestrator (SM Pipeline), same IAM, same monitoring. Consistent developer experience.
- **OSMO deferred to V3** — Only needed for 100+ parallel sim environments at scale. The code exists in the repo but isn't needed until customers outgrow single-GPU training.
- **Eval report over eval video for V1** — Action prediction error (MSE on held-out episodes) validates training without needing sim. Real sim rollout video requires Isaac-GR00T SDK from source.
- **boto3 for Pipeline definition** — SageMaker Python SDK v3 restructured the workflow module. Using boto3 `create_pipeline` API directly is more stable.
- **100-step smoke tests** — Full training (5000 steps, 11 hrs, $79) is for final demos. 100-step runs (~15 min, ~$2) validate the pipeline.
- **CodeBuild for NGC containers** — Isaac Lab base image is x86-only, can't build on ARM Mac. CodeBuild with X2_LARGE instance handles it. Same approach customers will use.

## Reference Implementations (external)

- **[aws-samples/sample-physical-ai-scaffolding-kit](https://github.com/aws-samples/sample-physical-ai-scaffolding-kit)** — AWS Japan's Physical AI samples. Key components:
  - `isaacsim-workstation/` — CDK for Isaac Sim EC2 + DCV remote desktop (our debug UI source for task 25)
  - `samples/newton-rl/` — Isaac Lab 3.0-beta1 RL on HyperPod. Uses RSL-RL + Newton physics. Confirms our training approach (headless, 4096 envs, tensorboard). Key differences: they use HyperPod/Slurm (we use SageMaker), Isaac Lab 3.0 (we use 2.1), RSL-RL (we use rl_games).
  - `samples/openpi-sample/` — π0 VLA training (alternative to GR00T, stretch goal S1)
  - `physai/` — Pipeline SDK with data conversion + schema validation
