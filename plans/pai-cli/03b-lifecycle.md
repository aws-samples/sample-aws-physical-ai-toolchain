# Phase 3b — Lifecycle subcommands (deploy / destroy / workstation / config)

Depends on Phase 1. Parallel to Phase 3 (different files). Shells out to `cdk`,
`deploy-workstation.sh`, and `aws ec2`/`ssm` via `helpers.run/run_capture` and boto3.

## IMPORTANT guardrail
The assistant must NEVER run `cdk deploy/destroy/bootstrap` itself. These subcommands
exist for the USER to run. Implementation/tests must use `--dry-run`/`echo`-mode or
mocked subprocess — coder agents must NOT execute a real deploy to "verify."

## 3b.1 `pai/commands/deploy.py` → `pai deploy`, `pai destroy`
- `pai deploy foundation` → `npx cdk deploy PhysicalAi-<env>-Foundation --context mode=simple`
  (cwd `cdk/`). Env from config.
- `pai deploy workstation` → exec `cdk/deploy-workstation.sh` (it already does
  auto-IP + AZ retry). Pass through `INSTANCE_TYPE`/`ALLOWED_CIDR` env if flags given
  (`--instance-type`, `--allowed-cidr`).
- `pai deploy batch` → `npx cdk deploy PhysicalAi-<env>-Batch --context batch=true`.
- `pai destroy <target>` → corresponding `npx cdk destroy …`. Confirm prompt
  (`click.confirm`) unless `--yes`.
- `--dry-run` prints the exact command without executing (so the assistant/tests can
  exercise it safely).
- Pre-check: `pai deploy batch` warns if Foundation stack absent or isaac-lab image
  missing (reuse `cfn.py`/doctor checks).

## 3b.2 `pai/commands/workstation.py` → `pai workstation <verb>`
Resolve the instance via the Workstation stack output (instance id) or by tag.
- `start` → `ec2.start_instances`; then poll until running; print new public IP.
- `stop` → `ec2.stop_instances`; confirm billing halts.
- `status` → instance state + type + IP.
- `ip` → just the current public IP (the thing users re-fetch after every start).
- `password` → run the SSM SetPassword command (the one in the stack's SetPassword
  output) via `ssm.send_command`; or print it for the user to run.
- `connect` → print the DCV URL `https://<ip>:8443` and the SSM start-session command.
- All via boto3 (testable with fake_boto3), NOT raw aws-cli shelling where a boto3
  call is cleaner. This is the clearest UX win — replaces 4–5 raw `aws ec2`/`ssm`
  incantations from Lab 2.

## 3b.3 `pai/commands/config_cmd.py` → `pai config`
- `pai config show` → pretty-print resolved config.json (+ resolved region).
- `pai config set <dotted.key> <value>` → write back to config.json (e.g.
  `aws.region`, `workstation.instanceType`). Preserve `_comment` keys + formatting
  as much as practical.
- Guardrail: refuse to print/log `allowedCidr` value loudly (it's the user's
  personal IP — must not leak into transcripts/commits).

## Acceptance
- `pai deploy foundation --dry-run` prints the exact cdk command, runs nothing.
- `pai workstation ip` with fake_boto3 returns the stubbed IP; `start/stop` call the
  right boto3 methods (asserted via fake_boto3 call recorder).
- `pai config set aws.region eu-west-1` updates config.json and `pai config show`
  reflects it.
- No subcommand here ever executes a real cdk deploy during tests.