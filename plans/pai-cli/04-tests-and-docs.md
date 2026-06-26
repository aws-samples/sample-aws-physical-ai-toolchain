# Phase 4 — Tests + docs

Depends on Phase 3.

## 4.1 `tests/test_cli.py` (NEW)
Reuse existing harness — do NOT invent a new stubbing approach.
- Reuse `tests/conftest.py`'s `fake_boto3` fixture + `_FakeSTS/_FakeSSM/_FakeS3`.
- Use `click.testing.CliRunner` to invoke `pai.cli.main` (via the group object).
- Tests (all offline, zero real AWS, no GPU):
  1. `pai --help` and `pai --version` exit 0.
  2. `pai rl launch --instance-count 2 --dry-run` → exit 0, output contains
     `"InstanceCount": 2`, the UNVALIDATED multi-node note, and
     `"[dry-run] No AWS calls made."`. Assert the fake boto3 recorded ZERO
     create_training_job calls.
  3. `pai rl launch --engine batch --num-nodes 2 --dry-run` → output contains the
     default queue (`physical-ai-dev-rl-queue`) + job def (`physical-ai-dev-rl-mnp`)
     and `numNodes: 2`; zero submit_job calls.
  4. `pai rl launch --task PickAndPlaceUR3-v0 --dry-run` → prints the UR3-not-
     registered warning.
  5. `pai doctor` with fake boto3 → runs all checks, exits 0 (or expected code).
  6. region resolution: with `AWS_DEFAULT_REGION` unset and a config.json fixture,
     resolves us-west-2.
  7. `pai groot launch --dry-run` → resolves bucket/role/image from a stubbed
     Foundation stack, prints the request, ZERO AWS create_training_job calls.
  8. `pai rl status <job>` → calls describe_training_job on fake boto3.
  9. **Lifecycle (Phase 3b):** `pai deploy foundation --dry-run` prints the cdk
     command and runs NO subprocess (assert via a patched `helpers.run`).
     `pai workstation ip|start|stop` call the right boto3 methods (fake_boto3 recorder).
     `pai config set aws.region X` round-trips through a temp config.json.
- **Guardrail test:** assert no test path invokes a real `cdk`/`aws` subprocess
  (patch `helpers.run` to raise if called without dry-run in deploy tests).
- Add `click` to `tests/requirements.txt` (and `training/requirements.txt` if the
  CLI is imported there). Confirm `tests/test_repo_hygiene.py` (no hardcoded account)
  still passes against the new `pai/` package.

## 4.2 Docs — CLI-first, raw as collapsed fallback (decided)
Every step shows `pai …` as the PRIMARY command; the raw `aws/cdk/python` form goes
in a collapsed `<details><summary>Under the hood</summary>` block. Nothing removed.

- **README.md — Quickstart becomes pai-driven:** `pip install -e .` →
  `pai doctor` → `pai deploy foundation` → `pai groot launch` (Path A) etc. Keep the
  raw `npx cdk deploy` / `./run-path-a.sh` forms in the under-the-hood notes. Add a
  short "Install the CLI" subsection.
- **workshop/lab-2-isaac-workstation.md — convert the runbook + bodies:**
  - Step 2 deploy → `pai deploy workstation` (raw: `./deploy-workstation.sh`).
  - The start/stop/ip/password section → `pai workstation start|stop|ip|password|
    connect` (raw: the `aws ec2`/`ssm` commands currently there).
  - Closed-loop eval → `pai eval serve` / `pai eval --closed-loop`.
  - Container test step keeps the docker commands (not a CLI concern).
- **workshop/lab-4-rl-refinement.md — convert the runbook + bodies:**
  - Step 1/3/3b launch → `pai rl launch [--engine batch] …`.
  - Step 2 watch → `pai rl status <job>` (raw: `aws sagemaker describe-training-job`).
  - Step 3b Batch deploy → `pai deploy foundation` / `pai deploy batch`.
  - Step 4 → `pai eval serve` / `pai eval --closed-loop` / `pai export`.
  - **FIX the broken reference:** Step 4b currently says
    `python cdk/start-workstation.py …` — that file does NOT exist. Replace with
    `pai workstation start` (raw fallback: `aws ec2 start-instances …`).
  - Preserve EVERY UNVALIDATED label and the UR3-not-registered warnings verbatim.
- **CLAUDE.md**: one line noting the `pai` CLI is the primary entry point and wraps
  the scripts/lifecycle (so future sessions know).
- Replace the "confirm credentials / region / image present" checklists with a
  `pai doctor` mention (it automates them).

## 4.3 Honesty-rule audit (must hold)
- The CLI changes HOW jobs are launched, not WHAT runs. Every UNVALIDATED label
  (multi-node NCCL, UR3 env, closed-loop-on-GPU, Cosmos) stays exactly as-is.
- Do NOT let the docs imply the CLI validates anything. A green `pai rl launch` is
  still just a submitted job.

## Acceptance
- `pytest tests/ -q` → previous 28 passed / 2 skipped PLUS the new CLI tests, all green.
- README/Lab 2/Lab 4 show both invocation styles; no path-based command removed.
- `run-path-a.sh` and all existing `python training/scripts/...` invocations still work.

## Final review (per global workflow)
After implementation, dispatch code-reviewer agents:
1. security — entry-point/packaging, no creds leakage, friendly_boto_error doesn't
   print secrets.
2. correctness — flags map 1:1 to wrapped `launch()` signatures; dry-run truly zero
   AWS calls.
3. honesty/docs — UNVALIDATED labels preserved; path-based commands still documented.
