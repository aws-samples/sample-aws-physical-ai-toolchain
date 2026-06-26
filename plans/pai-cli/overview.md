# Plan: in-repo `pai` CLI (click + boto3)

## Goal
Replace "clone the repo → remember which script → remember the flags → set the
region" with a single installable command. After `pip install -e .` the user types
`pai rl launch --instance-count 2` from anywhere. The CLI is a **thin skin over the
existing `training/scripts/*.py`** — it wraps their `launch()` functions, it does not
duplicate logic.

## Why a CLI (decision record)
We evaluated four control-plane patterns against three reference workshops:
- **hi-space (AWS)** — DCV GPU workstation IS the control plane. ❌ our anti-goal.
- **EDH/SOCA** — self-hosted HPC portal (web UI + scheduler). ❌ over-engineered.
- **Microsoft physical-ai-toolchain** — thin config-driven submitter → OSMO. ✅ model.
- Cognito web app — "browser + login only, zero local creds." Nice, but a full
  frontend + API GW + Cognito stack is more than this OSS toolchain needs now.

**Chosen:** in-repo `click`+`boto3` CLI (the `aws-pai` pattern from
`jetson-lerobot/lerobot-so101-aws-sim2real2sim`). Smallest footprint that improves
UX; composes toward Lab 6/OSMO as the production end-state.

## Accepted tradeoff (honesty)
A CLI does **not** deliver the "no local AWS creds" property the web app would. The
user still needs working AWS credentials locally **and** `iam:PassRole` for the
SageMaker role (the same thing that blocked launching from the workstation role
earlier). We accept this: the audience is SAs / technical users who already have AWS
creds. The CLI's `pai doctor` command will detect missing creds / missing PassRole
and print the exact remediation rather than a raw traceback.

## Scope — full lab coverage (decided)
The CLI is the SINGLE tool for the whole workshop: lifecycle + jobs + eval, so the
README Quickstart and Lab 2/4 docs become `pai`-driven. Command surface kept lean —
**8 top-level groups, 1–5 verbs each** — mapped to lab steps:

```
pai doctor                                    # preflight (creds/region/PassRole/image/stacks)
pai config [show | set k v]                   # read/edit config.json
pai deploy   <foundation|workstation|batch>   # wraps cdk deploy + deploy-workstation.sh  (Lab 0/2/4)
pai destroy  <foundation|workstation|batch>   # wraps cdk destroy
pai workstation <start|stop|status|ip|password|connect>   # absorbs raw aws ec2 + ssm     (Lab 2)
pai groot launch [--dry-run]                  # wraps run-path-a.sh GR00T pipeline          (Lab 1)
pai rl launch [--engine sagemaker|batch] …    # launch_rl.py / launch_rl_batch.py           (Lab 4)
pai rl status <job>                           # wraps aws sagemaker/batch describe           (Lab 4)
pai eval [--closed-loop] | pai eval serve     # evaluate.py / eval_*                         (Lab 2/4)
pai export                                    # export.py → ONNX/TensorRT                    (Lab 4/5)
```

**Decisions baked in:**
- **Wrap ALL lifecycle** — `pai deploy/destroy` shell out to `cdk`, `pai workstation`
  to `aws ec2`/`ssm`. The "assistant never runs cdk deploy/destroy" rule still binds
  ME; it does not stop the CLI from offering the command for the USER to run.
- **CLI-first docs, raw as fallback** — every Lab 2/4 + README step shows `pai …`
  primary; the raw `aws/cdk/python` form kept in a collapsed "under the hood" note.
  Nothing removed; also fixes the broken `start-workstation.py` reference (Lab 4
  Step 4b points at a file that does not exist).

**Out (v1):** `pai cosmos …` (Lab 3 — GPU/p5, lazy-import later; slots in cleanly),
resumable setup wizard, custom help renderers, web/Cognito anything.

## Real lifecycle tooling the CLI wraps (verified, not invented)
- Deploy: `cdk/deploy-workstation.sh` (auto-IP + AZ retry) and `npx cdk deploy
  PhysicalAi-dev-{Foundation|Workstation|Batch}` with the right `--context` flags.
- Workstation: raw `aws ec2 start/stop/describe-instances` + `aws ssm` (password) —
  there is NO existing `start-workstation.py`. `pai workstation` replaces these.
- GR00T (Lab 1): `run-path-a.sh` reads Foundation CfnOutputs (bucket/role/ECR) →
  uploads dataset → SageMaker job. `pai groot launch` wraps this logic.

## Key findings from investigation (drive the design)
1. **Most scripts already have a clean `launch(...)`** — `launch_rl.py`,
   `launch_rl_batch.py`, `export.py`, `eval_sim_client.py` need NO refactor.
   `evaluate.py` needs a small extract (`run_open_loop(...)` out of `main()`);
   `eval_policy_server.py` optionally extract `serve(...)`.
2. **Lazy-import the GPU/torch modules** inside subcommand bodies. `train.py` and
   `generate_scenes.py` crash on import off-GPU; `export.py`/`eval_policy_server.py`
   import torch (slow). Keep `pai --help` fast.
3. **`training/scripts/` sibling imports** (`from eval_protocol import ...`) require
   that dir on `sys.path`. CLI prepends it before importing the eval trio — matches
   what `tests/conftest.py` already does. Zero refactor, no `__init__.py` churn.
4. **Resolution helpers are duplicated** across scripts (`_account`, `_role_arn`,
   `_isaac_lab_image`, `_bucket`). Do NOT centralize yet — each `launch()` resolves
   internally, so the CLI just calls `launch()`. Note as known debt.
5. **Region inconsistency:** `cdk/bin/app.ts` falls back to us-east-1, the Python
   scripts default to us-west-2. The validated region is **us-west-2** — the CLI's
   config loader resolves to us-west-2 and reads `aws.region` from config.json.
6. **Tests:** reuse `tests/conftest.py`'s `fake_boto3` + `_FakeSTS/_FakeSSM/_FakeS3`,
   use `click.testing.CliRunner`, assert the `"[dry-run] No AWS calls made."` banner
   and the honesty-rule warning strings. Add `click` to `tests/requirements.txt`.

## Package layout (in-repo)
```
pai/                      # new top-level package
  __init__.py
  __main__.py             # `python -m pai`
  cli.py                  # root click group + command registration
  config.py               # config.json loader (dotted get), region resolution
  helpers.py              # message helpers, boto3 error → friendly remediation, run()/run_capture()
  _scriptpath.py          # prepend training/scripts to sys.path (one place)
  cfn.py                  # read CfnOutputs / stack status (used by groot/rl status/doctor)
  commands/
    doctor.py             # pai doctor — creds/region/PassRole/image/stacks preflight
    config_cmd.py         # pai config show|set
    deploy.py             # pai deploy|destroy <foundation|workstation|batch>
    workstation.py        # pai workstation start|stop|status|ip|password|connect
    groot.py              # pai groot launch  (wraps run-path-a.sh logic)
    rl.py                 # pai rl launch [--engine sagemaker|batch], pai rl status
    eval.py               # pai eval [--closed-loop], pai eval serve
    export.py             # pai export
pyproject.toml            # NEW at repo root: [project.scripts] pai = "pai.cli:main"
```

## Phases & dependencies
- **Phase 1 — packaging skeleton** (`01-packaging.md`): pyproject.toml, `pai/`
  skeleton, config loader, helpers (incl. `run()/run_capture()` for shelling to
  cdk/aws), `cfn.py`, `.gitignore` patterns. No script changes. Independent.
- **Phase 2 — script refactors** (`02-script-refactors.md`): extract
  `evaluate.run_open_loop`, optional `eval_policy_server.serve`. Keep `main()`s
  path-runnable. Independent of Phase 1.
- **Phase 3 — job/eval subcommands** (`03-subcommands.md`): `pai rl`, `pai eval`,
  `pai export`, `pai groot`, `pai doctor`. **Depends on Phase 1 + 2.**
- **Phase 3b — lifecycle subcommands** (`03b-lifecycle.md`): `pai deploy/destroy`,
  `pai workstation`, `pai config`. Shells out to cdk + aws ec2/ssm. **Depends on
  Phase 1.** Can run parallel to Phase 3 (different files).
- **Phase 4 — tests + docs** (`04-tests-and-docs.md`): `tests/test_cli.py` reusing
  fake_boto3; convert README Quickstart + Lab 2/4 to CLI-first (raw as fallback);
  fix the `start-workstation.py` reference. **Depends on Phase 3 + 3b.**

Waves: (1) Phase 1 ∥ Phase 2 → (2) Phase 3 ∥ Phase 3b → (3) Phase 4.

## Honesty-rule constraints the CLI must preserve
- The multi-node NCCL UNVALIDATED note printed by `launch_rl.py`/`launch_rl_batch.py`
  must still surface through the CLI (it will — we call `launch()` which prints it).
- The UR3-not-registered warning likewise.
- `--dry-run` must still make ZERO AWS calls through the CLI path (tested).
- No hardcoded account IDs (tests/test_repo_hygiene.py enforces).

## What the user runs (target UX — full workshop in one tool)
```bash
pip install -e .                       # one-time, in the cloned repo
pai doctor                             # creds + region + PassRole + image preflight
pai config set aws.region us-west-2

# Lab 0 / infra
pai deploy foundation                  # → cdk deploy PhysicalAi-dev-Foundation
# Lab 2 — workstation lifecycle
pai deploy workstation                 # → deploy-workstation.sh (auto-IP, AZ retry)
pai workstation ip                     # fetch current public IP (re-fetch after start)
pai workstation password               # set/get DCV password via SSM
pai workstation stop                   # halt billing when away
# Lab 1 — GR00T
pai groot launch --dry-run
# Lab 4 — RL
pai rl launch --instance-count 2 --dry-run
pai rl launch --instance-count 2       # real SageMaker multi-instance
pai rl launch --engine batch --num-nodes 2
pai rl status <job>                    # describe SageMaker/Batch job
# Lab 2/4 — eval (on the workstation)
pai eval serve --checkpoint ./m.pt     # terminal 1
pai eval --closed-loop --eval-rounds 100   # terminal 2
# Lab 4/5 — export
pai export --checkpoint s3://… --output-trt ./model.trt
```

Every command above keeps its raw `aws/cdk/python` equivalent documented in a
collapsed "under the hood" note in the labs (CLI-first, raw as fallback).
