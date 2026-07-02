# AWS Physical AI Toolkit vs. Microsoft Physical AI Toolchain

**Compared:** Our toolkit (`gitlab.aws.dev:devris/aws-physical-ai-toolchain`) vs. Microsoft's open-source repo ([github.com/microsoft/physical-ai-toolchain](https://github.com/microsoft/physical-ai-toolchain), v0.1.0 released 2026-02-07, actively developed through mid-2026, approaching v1.0.0 with Sigstore-signed releases).

**Last verified:** July 2026 (Microsoft README paste confirmed current state).

**Purpose of this comparison:** Understand where we lead, where Microsoft leads, and where the gaps are — so we can position our blog/workshop appropriately and prioritize roadmap items.

---

## TL;DR

| Dimension | Our Toolkit (AWS) | Microsoft (Azure) | Edge |
|-----------|-------------------|-------------------|------|
| **VLA / GR00T fine-tuning** | Fully implemented (N1.6-3B on SageMaker Pipeline, 4x A10G, gradient checkpointing, model registry) | `.gitkeep` placeholders only — no VLA training code | **AWS** |
| **Synthetic data (Cosmos)** | Cosmos 3 Super (Predict) + Transfer 2.5 both validated on p5.48xlarge with runnable scripts | Comment-only YAML stubs ("Status: Planned", "Container: TBD") | **AWS** |
| **RL training** | Isaac Lab PPO on SageMaker + Batch MNP, 4096 parallel envs, multi-GPU validated | SKRL (PPO/AMP/IPPO/MAPPO) + RSL-RL, fully implemented with AzureML + OSMO | **Microsoft** (algorithm variety) |
| **IaC** | AWS CDK (TypeScript), single monorepo, deploys in one command | Terraform with 35 .tftest.hcl files, Go e2e tests | **Microsoft** (test coverage) |
| **CI/CD & security** | CodeBuild for containers; security scan findings being addressed; no GitHub Actions | 50 GitHub Actions workflows (CodeQL, Gitleaks, Checkov, DAST, SBOM) | **Microsoft** |
| **Workshop / developer experience** | 6 hands-on labs with step-by-step guides, CLI tool (`pai`), cost estimates | Documentation site (VitePress), recipes, but no step-by-step workshop | **AWS** |
| **Model packaging & export** | ONNX + TensorRT export via Isaac Lab play.py; ROS 2 inference container for Jetson | ONNX export (opset 18) + ROS 2 inference node for UR10E at 30 Hz | **Parity** |
| **Orchestration** | SageMaker Pipelines (GR00T); OSMO Lab 6 is placeholder | NVIDIA OSMO on AKS (Helm-deployed, working workflow templates) | **Microsoft** |
| **Edge deployment** | IoT Greengrass + ROS 2 in Dockerfile (not yet automated) | FluxCD GitOps manifests + Arc (bootstrap prints "not yet implemented") | **Parity** (both incomplete) |
| **Cost transparency** | Explicit per-lab cost estimates ($2-$79 per stage) | No cost guidance | **AWS** |
| **Experiment tracking** | SageMaker Model Registry (GR00T pipeline auto-registers) | AzureML + MLflow with lineage tags, correlation IDs | **Microsoft** |
| **Dataset format** | LeRobot v2 (Parquet + MP4), validated conversion from Zarr | LeRobot v2 (same format) | **Parity** |
| **Agentic workflows** | None | Instruction-driven agents that orchestrate full pipeline (opt-in, auditable, with approval gates) | **Microsoft** |
| **Test suite** | No automated tests | pytest across 4 component suites + CI | **Microsoft** |

---

## Where We Lead

### 1. GR00T VLA Fine-Tuning (decisive advantage)

Our toolkit has a **complete, runnable** GR00T N1.6-3B fine-tuning pipeline:
- Custom training container built via CodeBuild (7.1 GB, Isaac-GR00T SDK + PyTorch 2.5)
- SageMaker Pipeline with auto model registry
- Gradient checkpointing + DeepSpeed ZeRO-2 + grad accumulation (fits 24 GB A10G)
- UR3 embodiment config with EEF action space
- 27 real teleop episodes included
- `pai groot launch --max-steps 100` → trained model in 15 min

Microsoft has **zero VLA code** — `training/vla/` contains only `.gitkeep` files and a README saying "planned future capability."

### 2. Cosmos Synthetic Data (decisive advantage)

Both Cosmos capabilities are validated and runnable:
- **Cosmos 3 Super (Predict):** Generates novel demonstrations from prompts (~5 min/video on 8x H100). Automated launch script, prompt catalog, pre-generated samples.
- **Cosmos Transfer 2.5:** Restyles existing videos preserving geometry/actions (~8-10 min/clip). NIM container on p5.48xlarge.

Microsoft's synthetic data directory contains only comment-only YAML workflow stubs with "Container: TBD" — no Python code, no Dockerfiles, no runnable path.

### 3. Workshop / Hands-On Experience

Six detailed labs (0-5) with:
- Step-by-step instructions, CLI commands, expected outputs
- Time and cost estimates for every operation
- Troubleshooting tables
- "Under the hood" expandable sections showing raw commands
- `pai` CLI abstracting infrastructure complexity
- Terminology guide for newcomers

Microsoft has a documentation site (VitePress) with architecture docs and recipe pages, but no equivalent hands-on workshop flow. Their own tiered-architecture proposal acknowledges the barrier: "A newcomer reading the architecture documentation concludes they must stand up Azure Arc, AKS, FluxCD, ACSA, IoT Operations, and a cloud training plane before they can do anything useful."

### 4. Cost Transparency

Every lab includes explicit cost breakdowns:
- Lab 1: ~$2 smoke test, ~$79 full training
- Lab 3/4: ~$37/hr (Capacity Block)
- Lab 5: ~$3 smoke test, ~$28 full

Microsoft's repo has no cost estimation tooling (their own contributing docs note "Cost estimates in this document were captured on 2026-02-03" but these are for infrastructure provisioning, not per-task training costs).

### 5. Single-Command Infrastructure

```bash
pai deploy foundation    # S3 + ECR + IAM + CodeBuild, all containers auto-built
```

No Terraform, no Helm, no AKS. One CDK command deploys everything. Containers build in CodeBuild (no local Docker needed, works on Apple Silicon). Compare to Microsoft's multi-step: Terraform apply → 4 ordered shell scripts for Helm → AKS GPU node pool provisioning.

---

## Where Microsoft Leads

### 1. CI/CD & Security Posture

50 GitHub Actions workflows covering:
- CodeQL (Python, JS, Go)
- Gitleaks secret scanning
- Checkov/TFLint for Terraform
- OWASP ZAP DAST
- Bandit (Python security)
- OpenSSF Scorecard
- Sigstore keyless tag signing
- SBOM generation (Syft)
- 14 Dependabot ecosystem configs

We have: CodeBuild for container builds, recently addressed security scan findings, no CI/CD pipeline beyond that. **This is our biggest gap for enterprise adoption.**

### 2. RL Algorithm Variety

Microsoft offers SKRL with PPO, AMP, IPPO, and MAPPO selectable via `--algorithm` flag, plus RSL-RL as a separate launcher. We use Isaac Lab's stock `train.py` (PPO only via rsl_rl). For most robotics tasks PPO is sufficient, but the algorithm breadth gives Microsoft flexibility for multi-agent scenarios.

### 3. NVIDIA OSMO Orchestration (working)

Microsoft has genuine OSMO integration:
- OIDC federation for 5 OSMO ServiceAccounts
- 4 ordered Helm charts deployed
- Parameterized workflow YAML templates (train, evaluate)
- OSMO CLI integration

Our Lab 6 (OSMO) is a placeholder. We do have SageMaker Pipelines for GR00T (which is working orchestration for that specific stage), but nothing equivalent for multi-stage sequencing.

### 4. Experiment Tracking

AzureML + MLflow with:
- `mlflow_run_context` context manager
- Custom SKRL metric interception
- Model registry auto-registration on completion
- Checkpoint URI tagging
- Dataset lineage tags

We have SageMaker Model Registry for GR00T (auto-registered by the pipeline), but no MLflow-equivalent lineage tracking or TensorBoard/W&B integration wired in.

### 5. IaC Test Coverage

35 `.tftest.hcl` files + Go end-to-end tests. Our CDK has no test suite. This matters for enterprise customers who require infrastructure validation before deployment.

### 6. Edge Data Ingestion

ACSA IngestSubvolume CRD auto-syncs ROS 2 bags from edge devices to Azure Blob Storage with managed identity — zero application code. Their edge data capture now includes ROS 2 demonstration recording on Jetson with chunking, compression, and cloud upload. We have manual `aws s3 sync` or the upload script.

### 7. Agentic Workflows (new since earlier analysis)

Microsoft now has instruction-driven agents that can orchestrate the full pipeline:
- "Collect 50 demonstrations and train an IL policy" → agent decomposes, executes, evaluates
- Composable — use agents for some stages, manual CLI for others
- Approval gates before destructive actions (production deploy, data deletion)
- All agent actions logged and auditable via Azure Monitor + MLflow
- Guardrails and customizable behavior via config files

We have nothing equivalent. This is a compelling developer experience differentiator for teams that want to reduce operational toil.

### 8. Test Suite

Microsoft ships `uv run pytest` across 4 component suites (training, data-management, data-pipeline, fleet-deployment). We have no automated tests.

---

## Where Both Are Incomplete

| Gap | AWS Status | Microsoft Status |
|-----|-----------|-----------------|
| Fleet OTA deployment | Greengrass in Dockerfile, not automated | FluxCD bootstrap prints "not yet implemented" |
| VLA inference serving | SageMaker endpoint (`pai groot deploy`) works | No VLA code at all |
| End-to-end auto pipeline | SageMaker Pipeline for GR00T only | OSMO pipeline script Stage 3 calls a non-existent file |
| Dataset validation tools | Basic conversion script | 5 CLI tools labeled "Planned" with no code |

---

## Positioning Recommendations

### For the blog:

1. **Lead with what's runnable.** Our GR00T + Cosmos + Isaac Lab pipeline is the most complete *end-to-end working* Physical AI toolkit on any cloud. Microsoft's is architecturally broader but has major functional gaps (no VLA, no Cosmos, no workshop).

2. **Emphasize developer experience.** The `pai` CLI, step-by-step labs, cost estimates, and "works on Apple Silicon" story are unique. Microsoft's own docs acknowledge their barrier-to-entry problem.

3. **Don't claim we win on security/CI.** That's verifiably false. Position it as "open-source, actively hardening" and focus on the functional completeness story.

### For the roadmap (to close gaps):

| Priority | Gap | Effort |
|----------|-----|--------|
| P1 | Add GitHub Actions / GitLab CI (lint, test, security scan) | 1-2 days |
| P1 | Add CDK tests (jest snapshots at minimum) | 1 day |
| P1 | Add pytest suite for training scripts | 2-3 days |
| P2 | MLflow or SageMaker Experiments integration for RL | 3-5 days |
| P2 | Automate edge deployment (Greengrass component packaging) | 1 week |
| P2 | Agentic workflow exploration (Bedrock Agents or Strands for pipeline orchestration) | 1-2 weeks |
| P3 | Multi-algorithm RL support (AMP, SAC) | 1 week |
| P3 | OSMO Lab 6 implementation | 2 weeks |

---

## Key Differentiator Summary

**Microsoft built the enterprise platform shell. We built the working robotics pipeline.**

Microsoft has better infrastructure engineering (Terraform tests, CI/CD, security, OSMO orchestration, agentic workflows) but cannot actually train a VLA foundation model, generate synthetic data with Cosmos World Foundation Models, or walk a user through the complete pipeline with cost estimates. We can — in under a day, for under $50.

Their quickstart claims "under 2 hours" to a trained RL policy + MLflow + Jetson deploy. That's competitive with our Lab 5 flow. But they don't have:
- GR00T VLA fine-tuning (the harder, higher-value capability)
- Cosmos 3 Predict (novel demonstration generation)
- Cosmos Transfer 2.5 (action-preserving visual augmentation)
- Real teleoperation data included in the repo

For a practitioner who needs to train a robot policy this week, our toolkit is the starting point. For an enterprise architect evaluating cloud platforms for a 2027 production deployment, Microsoft's security, governance, and agentic posture is stronger.

The blog should position us as: **"The first complete, runnable Physical AI pipeline on any cloud — from 27 real demonstrations to a deployed robot policy, with NVIDIA's full stack (GR00T + Cosmos + Isaac Lab) integrated end-to-end."**
