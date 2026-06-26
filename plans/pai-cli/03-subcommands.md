# Phase 3 — Subcommands (wire the wrappers onto the group)

Depends on Phase 1 (skeleton/config/helpers) + Phase 2 (refactors).
Each subcommand is a THIN click body that lazy-imports the target script and calls
its function. All flags pass through 1:1. The honesty-rule prints come from the
wrapped `launch()` automatically.

## 3.1 `pai/commands/rl.py` → `pai rl launch`
- `--engine {sagemaker,batch}` (default sagemaker) selects the module.
- **sagemaker:** flags `--task --num-envs --max-iterations --framework
  --instance-type --instance-count --runtime-min --dry-run` →
  `launch_rl.launch(...)`. (`--instance-count > 1` triggers the script's UNVALIDATED
  note — keep it.)
- **batch:** flags `--task --num-envs --max-iterations --framework --num-nodes
  --job-queue --job-definition --dry-run` → `launch_rl_batch.launch(...)`.
  When `--job-queue/--job-definition` omitted, call `_default_queue()/_default_job_def()`
  first (Phase 2.3). Multi-node UNVALIDATED note comes from the script.
- Region: set `AWS_DEFAULT_REGION` from `pai.config.resolve_region()` before calling
  `launch()` (the scripts read it from env). This fixes the us-west-2 vs us-east-1
  drift centrally.
- Wrap the call in `try/except` → `helpers.friendly_boto_error` (PassRole guidance).

## 3.2 `pai/commands/eval.py` → `pai eval`, `pai eval serve`
- `ensure_scripts_on_path()` first (sibling imports).
- `pai eval` (open-loop, default): `--checkpoint --num-episodes --render-video
  --output-dir --env --seed` → `evaluate.run_open_loop(...)`. Print the returned JSON.
- `pai eval --closed-loop`: `--task --endpoint --eval-rounds --max-steps --output-dir`
  → construct `eval_protocol.ZmqTransport(endpoint)` then
  `eval_sim_client.run_eval(...)`. Assumes a server is already running; print the
  UNVALIDATED-on-GPU note (mirror Lab 4 Step 4 / Lab 2 wording).
- `pai eval serve`: `--checkpoint --endpoint --device` →
  `eval_policy_server.serve(...)` (Phase 2.2). Long-running; localhost-only default.

## 3.3 `pai/commands/export.py` → `pai export`, `pai scriptify`
- `pai export`: `--checkpoint --output-onnx --output-trt --target-device --fp16
  --benchmark` → `export.export_to_onnx(...)` then `export.compile_tensorrt(...)`
  (or `export.main()` if simpler). Lazy-import (pulls torch).
- `pai scriptify` (only if Phase 2.4 done): `--checkpoint --output ...` →
  `scriptify_policy.scriptify(...)`. Else omit from v1.

## 3.3b `pai/commands/groot.py` → `pai groot launch` (Lab 1)
- Port `run-path-a.sh` logic to Python (it's bash around raw aws calls): read
  Foundation CfnOutputs via `cfn.py` (bucket/role/ECR), upload the dataset dir to S3,
  launch the SageMaker GR00T job. Flags: `--dataset-dir` (default the bundled UR3
  dataset), `--max-steps`, `--dry-run`.
- `--dry-run` prints the resolved bucket/role/image + the create_training_job request,
  ZERO AWS calls (mirror launch_rl's dry-run banner).
- HF_TOKEN: read from env; warn (don't fail) if missing, since model download needs it.
- Keep `run-path-a.sh` in the repo (docs reference it as the fallback); `pai groot`
  is the CLI-first path.

## 3.3c `pai rl status` (in rl.py)
- `pai rl status <job-name>` → `sagemaker.describe_training_job` (or
  `batch.describe_jobs` if `--engine batch`); print status + key timestamps.
  Replaces the raw `aws sagemaker describe-training-job` / `aws batch describe-jobs`
  in Lab 4. Friendly errors via helpers.

## 3.4 `pai/commands/doctor.py` → `pai doctor`
- Preflight checks, each pass/fail with remediation, overall exit code:
  1. AWS creds resolvable (`sts.get_caller_identity`) — print account.
  2. Region resolves (show resolved region; warn if not us-west-2 where the
     validated image lives).
  3. `iam:PassRole` likely available — best-effort: detect if caller is the minimal
     workstation role (warn: "launch from a context with PassRole, e.g. your laptop
     admin creds or CloudShell"). Do NOT actually call PassRole.
  4. isaac-lab image present in ECR for resolved account/region
     (`ecr.describe-images physical-ai/isaac-lab:latest`). Warn if missing →
     "deploy Foundation / wait for CodeBuild".
  5. Foundation stack exists (`cloudformation.describe-stacks
     PhysicalAi-<env>-Foundation`).
- This command is the CLI's answer to the PassRole/region foot-guns we hit manually.

## Cross-cutting
- Every subcommand: resolve region centrally, friendly errors, never raw traceback.
- `pai --help` must stay fast — all heavy imports inside command bodies.
- Out of v1 (note in `pai --help` epilog as "coming soon" or just omit): cosmos,
  scenes, train wrappers.

## Acceptance
- `pai rl launch --instance-count 2 --dry-run` prints the SageMaker request with
  `InstanceCount: 2` and the UNVALIDATED note, ZERO AWS calls.
- `pai rl launch --engine batch --num-nodes 2 --dry-run` prints the Batch
  submit_job request with the default queue/def names, ZERO AWS calls.
- `pai eval --closed-loop --dry-run`-equivalent path importable without GPU.
- `pai doctor` runs all checks against fake boto3 in tests.
