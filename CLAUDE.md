# CLAUDE.md — project context for the AWS Physical AI Toolchain

Orientation for AI assistants / new contributors. Pairs with `docs/ROADMAP.md`
(feature plan) and `PLAN.md` (build history).

## What this is
An end-to-end reference pipeline for training robot manipulation policies on AWS +
NVIDIA: GR00T imitation learning → Isaac Lab RL refinement → Cosmos photorealistic
enhancement → edge deployment (Jetson/Greengrass). Distributed as open source, so
**no step may require large local downloads or local Docker builds.**

## Container builds — the core architecture decision
**Every container image builds in AWS CodeBuild and is pushed to ECR. Never build
locally** (NVIDIA bases are huge + often x86-only; can't build on Apple Silicon).
- Construct: `cdk/lib/constructs/container-build.ts` — shared S3 source asset →
  CodeBuild project (S3 source) → ECR, auto-triggered on `cdk deploy`. Trigger's
  physical-id hashes source **and** env vars, so re-builds fire on either change.
- 7 builds wired in `cdk/lib/foundation-stack.ts`: groot-training, isaac-lab,
  isaac-sim, inference (x86 `:latest` + jetson `:jetson` on a Graviton/ARM fleet),
  cosmos-transfer (2.5, from source), cosmos3 (cosmos-framework, from source).
- NGC-based builds read `physical-ai/ngc-api-key` from Secrets Manager. **All bases
  are pulled from `nvcr.io`, never anonymous Docker Hub** (rate limits). Per-image
  build context differs — see each `containers/*/buildspec.yml`.
- ECR repos live ONLY in FoundationStack (`physical-ai/*`); EKS pulls from those.
- All 7 images built green in CodeBuild→ECR (us-west-2) as of this work.

## Deploy
- `cd cdk && npx cdk deploy PhysicalAi-dev-Foundation --context mode=simple`
- Region: validated in **us-west-2**, account `149536462911`. Secrets Manager is
  regional — keep the NGC/HF keys in the deploy region.
- `cdk.context.json` is gitignored (it leaked account/VPC ids; breaks Vpc.fromLookup
  for other accounts).
- Git pushes use `ssh.gitlab.aws.dev`; cert expires — refresh with `mwinit -s`.

## Cosmos: Transfer 2.5 vs Cosmos 3 (easily confused)
- **Transfer 2.5** (`nvidia-cosmos/cosmos-transfer2.5`) is the ONLY one doing
  controlled sim→real *transfer* (edge/depth/seg control). Build pin: Python 3.10
  (cp310 flash-attn wheels). Runtime: EC2 Spot p5 + NIM `/v1/infer` on :8000 —
  **NOT SageMaker** (SM GPUs ship driver 470; Cosmos needs 580+). Driven by
  `training/scripts/cosmos_setup.py` (launch/status/generate/terminate) +
  `scripts/cosmos-userdata.sh`. Runbook: `docs/cosmos-deployment-guide.md`.
- **Cosmos 3** (`NVIDIA/cosmos-framework`, image `physical-ai/cosmos3`) does
  *generation* (text2video/video2video), NOT controlled transfer yet. Build pin:
  Python 3.13 (cp313 wheels). Runner: `training/scripts/cosmos3_generate.py`.
- Weights are gated HuggingFace downloads (HF_TOKEN), not NGC.

## Tests
`tests/` (pytest, no AWS/GPU needed; AWS clients stubbed). `pytest tests/ -q`.
Covers the TorchScript scriptify round-trip, env registration, all launcher
`--dry-run`s (assert zero real AWS calls), and repo hygiene. 12 pass / 1 skip.

## Current in-flight work
Branch **`fix/tier0-bugfix-sweep`** — 8 commits ahead of `main`, NOT pushed yet
(needs `mwinit -s`). Contains: Tier-0 bugfixes, all 4 roadmap features
code-complete, and the test suite. See `docs/ROADMAP.md` for the per-feature
status table.

## The honesty rule (important)
This codebase was deliberately made to **not over-claim**. Validation that needs a
Jetson, a G-family GPU, or sustained p5 capacity is **wired correctly and labelled
"unvalidated"**, never asserted as working. Known still-unvalidated-on-hardware:
edge deploy on a real Jetson, UR3 Isaac Lab env *instantiation* (registered but
GPU-untested; the validated RL path is the built-in Anymal task), Cosmos
restyle/generate (needs p5). Do not flip these to "done" without a real run, and
keep the object-pose placeholder in the edge inference node flagged. When unsure of
an external fact (image tags, NVIDIA APIs), verify via `gh`/docs — guessing caused
real bugs earlier (e.g. wrong l4t-tensorrt tag, wrong Cosmos image ref).
