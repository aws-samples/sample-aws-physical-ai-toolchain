#!/usr/bin/env bash
# Detach the local pipeline from SSH while retaining logs and terminal status.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/local/launch.sh RUN_ID --development-bucket BUCKET --expected-role ROLE [runner options]" >&2
  exit 2
fi
run_id="$1"
shift
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9-]{0,39}$ ]] || {
  echo "Invalid run ID: use 1–40 letters, digits or hyphens; start with a letter or digit" >&2
  exit 2
}
tooling_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
component="$(cd "$tooling_dir/../.." && pwd)"
python_bin="${VLA_LOCAL_PYTHON:-$component/.venv/bin/python}"
test -x "$python_bin" || { echo "Run bash scripts/local/setup.sh first" >&2; exit 2; }
expected_commit="$(git -C "$component" rev-parse HEAD)"
timeout_seconds=10800
runner_args=("$@")
for ((index=0; index<${#runner_args[@]}; index++)); do
  case "${runner_args[index]}" in
    --timeout-seconds)
      timeout_seconds="${runner_args[index+1]:-}"
      ;;
    --timeout-seconds=*)
      timeout_seconds="${runner_args[index]#*=}"
      ;;
  esac
done
[[ "$timeout_seconds" =~ ^[1-9][0-9]{0,5}$ ]] || {
  echo "--timeout-seconds must be a positive integer below 1000000" >&2; exit 2;
}
runtime="$component/local-dev"
run_dir="$runtime/runs/$run_id"
[[ ! -e "$run_dir" ]] || { echo "Run directory already exists; choose a new run ID" >&2; exit 2; }
sudo install -d -m 700 "$run_dir" "$runtime/docker-config" "$runtime/sdk-cache"
sudo systemd-run \
  --unit="vla-local-$run_id" \
  --working-directory="$component" \
  --property="RuntimeMaxSec=$((timeout_seconds + 120))" \
  --property=TimeoutStopSec=90 \
  --property="StandardOutput=append:$run_dir/run.log" \
  --property="StandardError=append:$run_dir/run.log" \
  --setenv="PATH=$(dirname "$python_bin"):/usr/local/bin:/usr/bin:/bin" \
  --setenv=SAGEMAKER_SUPPRESS_V2_WARNING=1 \
  --setenv="VLA_FOUNDATION_PROJECT=${VLA_FOUNDATION_PROJECT:-physical-ai}" \
  --setenv="DOCKER_CONFIG=$runtime/docker-config" \
  --setenv="VLA_VALIDATION_SDK_CACHE=$runtime/sdk-cache" \
  --setenv="VLA_LOCAL_SYSTEMD_UNIT=vla-local-$run_id" \
  "$python_bin" -u "$tooling_dir/run_local_pipeline.py" \
    "$@" --component "$component" --run-dir "$run_dir" \
    --run-id "$run_id" --expected-commit "$expected_commit"
printf '%s\n' "$run_id" | sudo tee "$runtime/latest-run-id.txt" >/dev/null
printf 'UNIT=vla-local-%s\nRUN_DIR=%s\n' "$run_id" "$run_dir"
printf 'Reconnect: sudo %q %q --run-dir %q\n' \
  "$python_bin" "$tooling_dir/wait_run.py" "$run_dir"
