#!/usr/bin/env bash
# Install launcher dependencies in this clone; images and workloads are separate.
set -euo pipefail
component="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10–3.12 required"'
command -v docker >/dev/null
test -x /usr/bin/pigz || { echo "Install pigz first: sudo apt-get install pigz" >&2; exit 2; }
docker compose version
nvidia-smi --query-gpu=name,memory.total --format=csv
"$python_bin" -m venv "$component/.venv"
"$component/.venv/bin/python" -m pip install --upgrade pip
"$component/.venv/bin/python" -m pip install \
  -e "$component" 'sagemaker[local]==2.257.6' 'docker==7.2.0'
printf 'Environment ready: %s/.venv\nNext: follow README.md, Run locally — EC2 GPU debugging, for IAM, storage and launch.\n' "$component"
