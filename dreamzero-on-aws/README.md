# DreamZero Fine-Tuning on AWS

Fine-tune [NVIDIA DreamZero](https://github.com/dreamzero0/dreamzero) — a 14B-parameter
video-diffusion **World Action Model** for robot manipulation — on your own LeRobot-format
demonstrations, as a SageMaker Training Job with a bring-your-own container, and get
**servable merged weights in S3** with one command.

DreamZero is the World-Action-Model entry in the toolchain's **Model Training** pillar: where a
VLA policy predicts the next action chunk, a World Action Model predicts the next seconds of
video *and* the actions that get the robot there. Its base weights (`GEAR-Dreams/DreamZero-AgiBot`)
are **Apache-2.0**, so the whole fine-tuning path is commercially usable end to end.

> **Where the code lives.** This component is maintained as a standalone AWS sample so that
> its CDK stack and registered solution ID stay in one place:
>
> **→ [aws-samples/sample-dreamzero-finetuning-on-sagemaker](https://github.com/aws-samples/sample-dreamzero-finetuning-on-sagemaker)**
>
> The story behind it — what was measured, and the failure modes that only showed up with
> real money on the line — is on [AWS Builder Center](https://builder.aws.com/content/3Im0ezyNtOji0HQBCTwqwgkQ9VI/fine-tuning-nvidias-dreamzero-a-14b-world-action-model-on-amazon-sagemaker-what-it-actually-takes).

---

## What You'll Build

```
┌──────────────────────────────────────────────────────────────────────┐
│  Infrastructure (one `cdk deploy`, per region)                       │
│  S3 bucket · ECR repo · SageMaker execution role · CodeBuild factory │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Container build (CodeBuild, no local Docker)                        │
│  SageMaker PyTorch 2.8 DLC + pinned DreamZero + 4 upstream patches   │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
  fetch → detect → convert → validate → prep → stage → smoke → train → merge
  └──────────── local (stage uploads to S3) ──────────┘   └─ SageMaker jobs ─┘
```

- **Validation before any compute** — embodiment config checked against the real dataset
  (dimension slices, camera order, fps, file completeness, and a scan for `action == state`
  labels, the data-collection bug that makes a useless model score perfectly).
- **A smoke gate** — the same container, data and hyperparameters for a handful of steps
  (~$10, ~25 min) so you know the full run's cost before you spend it.
- **An automatic merge stage** — LoRA adapters merged onto the correct base as a SageMaker job.
  Serving the raw adapters through the upstream loader composes them onto the wrong base and
  is measured **9.9× worse** than correct; the merge closes that trap and rewrites the
  checkpoint config so a loader cannot walk back into it.
- **Managed spot with safe resume** — `CheckpointConfig` mirrors ~3.6 GB checkpoints; a
  completeness guard in the entrypoint discards a checkpoint torn by a reclaim rather than
  resuming from it. Proven on a job reclaimed three times.

## Prerequisites

- SageMaker training quota for an **80 GB+ per GPU** instance type (`ml.g7e.24xlarge` is the
  shipped default; `ml.p4de.24xlarge` and `ml.p5.48xlarge` are validated). Quotas default to 0 —
  request early. 48 GB GPUs (`ml.g6e`) will OOM.
- AWS CLI v2, Node.js (for the CDK CLI), Python 3.9+. **No local Docker** — the image builds in
  CodeBuild.
- ~250 GB local disk for the one-time base-weight staging (~128 GB into S3).

## Get Started

Three commands on the shipped public demo dataset (`lerobot/aloha_static_screw_driver`, MIT):

```bash
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1   # set both
cd cdk && pip install -r requirements.txt && cdk bootstrap && cdk deploy && cd ..
./setup.sh --stage-assets
python3 pipeline/run_pipeline.py --name aloha-demo
```

Full instructions, the bring-your-own-dataset guide (including LeRobot v3 → v2.1 conversion),
per-stage commands and troubleshooting are in the
[standalone repo's README](https://github.com/aws-samples/sample-dreamzero-finetuning-on-sagemaker#readme).

## Estimated Costs

Measured in us-east-1 on the demo dataset:

| Phase | Wall clock | Cost |
|-------|-----------|------|
| CDK deploy + CodeBuild image build (once per region) | ~10–15 min + queue | ~$3 |
| One-time base-weight staging (~128 GB) | ~1–2 h | ~$3/month S3 |
| Smoke gate (10 steps) | ~25 min | ~$10 |
| 1000-step LoRA fine-tune, `ml.g7e.24xlarge` | 4 h 11 m | ~$93 |
| LoRA → base merge | ~15 min | ~$5 |
| **First servable checkpoint** | **~6–7 h** | **~$110** |

Compute is linear in `max_steps`; storage grows ~3.6 GB per checkpoint plus a ~92 GB merged model.

## Validated Results

A 1000-step LoRA fine-tune on `ml.g7e.24xlarge` reproduced an EC2 reference run of the same
recipe (final loss 0.0957 vs ~0.096; open-loop MSE of the merged checkpoint 0.00106 vs 0.00100),
and the fine-tune beats the base model by ~7× overall MSE on its own data.

## Security and Licensing

- `docs/SECURITY.md` in the standalone repo: cdk-nag on every synth, Bandit (0 High / 0 Medium),
  a threat-model summary and the production gaps deliberately left to you, each with a remedy.
- `docs/DEPENDENCY-INVENTORY.md`: pip-audit of the built image with per-advisory reachability.
- Every model, dataset and image the pipeline touches is listed with its license at the pinned
  upstream revision (DreamZero-AgiBot, Wan2.1, umt5-xxl tokenizer: Apache-2.0; ALOHA demo
  dataset: MIT). The sample's own code is MIT-0.

## Relationship to the Toolchain

The component is self-contained today (its own bucket, ECR repository and execution role via
CDK) rather than built on `foundation/`. Consolidating onto the shared Foundation resources and
the toolchain's Terraform convention is planned as a follow-up once the toolchain's CDK
coverage lands; until then this folder is the toolchain's entry point and the standalone repo
is the source of truth.
