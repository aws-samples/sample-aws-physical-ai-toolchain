# Plan: Distributed RL + Closed-Loop Evaluator

Adds two capabilities to the AWS Physical AI Toolchain, inspired by the hi-space
"Physical AI on AWS" workshop, implemented with THIS repo's conventions (CDK TS,
config.json single source of truth, CodeBuild→ECR container builds — never local
docker, honesty rule: label anything needing real GPU/hardware "unvalidated until
run").

## Features
1. **SageMaker multi-instance RL** (quick, low-risk unlock) — `01-sagemaker-multinode.md`
2. **AWS Batch Multi-Node Parallel RL** (the reference architecture) — `02-batch-mnp.md`
3. **Closed-loop evaluator** (ZMQ policy-server ↔ Isaac Lab, runs on Lab 2 box) — `03-closed-loop-evaluator.md`

## Waves (dependencies)
- **Wave 1 (fully parallel — disjoint files):**
  - Track A: SageMaker multi-instance unlock (`launch_rl.py` + test)
  - Track B: Closed-loop evaluator (`eval_*.py`, `evaluate.py` refactor, tests, requirements)
  - Track C-infra: Batch container entrypoint (`batch-train-entrypoint.sh`, Dockerfile COPY)
  - Track C-cdk: Batch CDK construct + stack + `app.ts` wiring + `config.json`
  - Track C-submitter: `launch_rl_batch.py` + test
- **Wave 2 (after Wave 1):** code-review (3 reviewers: security, completeness, maintainability) → fix.
- **Validation gate (user-run):** `cdk synth`, `pytest tests/ -q`, then real deploys.

## Hard constraints
- Never run `cdk deploy/destroy/bootstrap` (user does). Never local docker build.
- All new GPU/multi-node/Isaac paths ship LABELLED unvalidated.
- `pytest tests/` must stay green (no AWS/GPU; AWS clients stubbed; dry-runs make zero AWS calls).
- Don't break the validated single-instance SageMaker RL path (just ran green).

## Biggest open risk (flag in docs)
Multi-node NCCL *convergence* is unverified on hardware for both SM-multi-instance
and Batch. The torchrun rendezvous wiring is structurally correct, but whether
Isaac Lab's `rsl_rl` honors `RANK`/`WORLD_SIZE` for true data-parallel RL (vs only
`LOCAL_RANK` single-node) is unknown. Label prominently.
