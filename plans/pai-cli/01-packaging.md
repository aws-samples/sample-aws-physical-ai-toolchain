# Phase 1 — Packaging skeleton + config loader + helpers

No script changes. Net-new files only. Independent of Phase 2.

## 1.1 `pyproject.toml` (NEW, repo root)
First pyproject in the repo — no conflict. Model on the aws-pai reference.
```toml
[project]
name = "aws-physical-ai-toolchain"
version = "0.1.0"
description = "CLI for the AWS Physical AI Toolchain (GR00T / Isaac Lab / Cosmos)"
requires-python = ">=3.10"
dependencies = ["click>=8.1", "boto3>=1.34"]

[project.optional-dependencies]
dev = ["pytest>=7", "numpy>=1.24"]

[project.scripts]
pai = "pai.cli:main"

[tool.setuptools.packages.find]
include = ["pai*"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```
- `pip install -e .` exposes the `pai` command (entry point → `pai.cli:main`).
- Keep deps minimal (click + boto3). Do NOT pull torch/isaacsim — those live in
  `training/requirements.txt` and are only needed on GPU at runtime.
- NOTE: `tool.setuptools.packages.find include=["pai*"]` so we do NOT accidentally
  package `training/`, `cdk/`, `tests/`.

## 1.2 `pai/` skeleton
- `pai/__init__.py` — `__version__`.
- `pai/__main__.py` — `from pai.cli import main; main()`.
- `pai/cli.py` — root `@click.group()` with `@click.version_option`; a `main()`
  callable that is the entry point. Registers subgroups at the bottom:
  `from pai.commands import rl, eval as eval_cmd, export, doctor` then
  `rl.register(cli)`, etc. (decorator-group idiom from the reference's rl.py — pick
  ONE registration style, skip the cloud.py module-level style).

## 1.3 `pai/config.py` — config.json loader (first Python consumer)
- `load() -> dict` reads repo-root `config.json`; tolerate `_comment` keys.
- `get("aws.region", default)` dotted-path getter (copy the reference pattern).
- `resolve_region()` precedence: `AWS_DEFAULT_REGION` env → `config.aws.region` →
  **us-west-2** (the validated region; do NOT fall back to us-east-1 like
  cdk/bin/app.ts does — note this divergence in a comment).
- `batch_defaults()` → reads `config.batch.{instanceType,numNodes,maxvCpus}`.
- Locate repo root robustly (walk up from `__file__` to the dir containing
  `config.json`) so `pai` works from any cwd.

## 1.4 `pai/helpers.py` — UX + error handling + subprocess
- Message helpers: `info/success/warn/error/heading/pass_msg/fail_msg` (click.secho
  with colors), copied/trimmed from the reference.
- `friendly_boto_error(e)` — catch `ClientError`/`BotoCoreError`; special-case
  `AccessDenied*` containing `iam:PassRole` → print the exact required policy
  (PassRole on `arn:aws:iam::<acct>:role/physical-ai-dev-sagemaker-role`) and how to
  attach it. Never dump a raw traceback. Map credential/region errors to
  "run `aws configure` / set region" guidance. Return SystemExit(1).
- `run(cmd, cwd=None, env=None)` and `run_capture(cmd, ...) -> (rc, stdout)` —
  subprocess wrappers (copy the reference's). Needed by Phase 3b to shell out to
  `cdk` / `deploy-workstation.sh` / `aws`. Stream output for long-running deploys.

## 1.4b `pai/cfn.py` — CloudFormation output reader
- `stack_outputs(stack_name, region) -> dict` via boto3 cloudformation.
- `stack_status(stack_name, region) -> str | None` (None if absent).
- Helpers `bucket()/role_arn()/ecr_uri()` that read the Foundation stack's
  `DatasetsBucketName`/`SageMakerRoleArn`/ECR outputs — replaces the `aws
  cloudformation describe-stacks --query …` calls in `run-path-a.sh`. Used by
  `pai groot`, `pai rl status`, `pai doctor`.

## 1.5 `pai/_scriptpath.py` — sibling-import shim
- `ensure_scripts_on_path()` prepends `<repo>/training/scripts` to `sys.path` exactly
  once. Called inside eval/export subcommands BEFORE importing `evaluate` /
  `eval_sim_client` / `eval_policy_server` (they do `from eval_protocol import ...`).
- Mirrors what `tests/conftest.py` already does — keeps zero-refactor on the scripts.

## 1.6 `.gitignore` additions
Add (currently missing): `*.egg-info/`, `build/`, `dist/`, `.eggs/`.
(`__pycache__/`, `.venv/` already present. `pai/` source dir is NOT caught by any
existing ignore — safe.)

## Acceptance
- `pip install -e .` succeeds; `pai --help` and `pai --version` work and are fast
  (no torch/isaacsim import at group load).
- `python -c "from pai.config import resolve_region; print(resolve_region())"` →
  us-west-2 with no env set (or config value if present).
- No subcommands wired yet (Phase 3) — group shows the registered-but-empty tree or
  a stub message.
