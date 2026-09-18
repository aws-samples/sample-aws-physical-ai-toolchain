#!/usr/bin/env bash
# One-time Ubuntu GPU host prerequisites. Called only by an approved vla deploy.
set -euo pipefail
test "$(id -u)" = 0 || { echo "Host preparation must run as root through SSM"; exit 1; }
exec 9>/var/lock/vla-local-gpu.lock
flock -n 9 || { echo "A VLA worker is active; retry host preparation after it finishes"; exit 1; }
if command -v docker >/dev/null && [ -n "$(docker ps -q)" ]; then
  echo "Running containers found; host preparation refuses to interrupt them"; exit 1
fi
if pgrep -f '[r]un_local_pipeline.py' >/dev/null; then
  echo "A local pipeline is preparing or running; leave its source and dependencies unchanged"; exit 1
fi
test "$(uname -m)" = x86_64 || { echo "The sample requires an x86-64 GPU host"; exit 1; }
source /etc/os-release
test "$ID:$VERSION_ID" = "ubuntu:22.04" || {
  echo "Automated setup supports Ubuntu 22.04; prepare other hosts through the README"; exit 1;
}
command -v nvidia-smi >/dev/null || {
  echo "NVIDIA driver missing. Use the README GPU DLAMI or install its NVIDIA driver before host preparation; bootstrap installs container tools, not GPU drivers." >&2
  exit 1
}
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU compute processes are active; retry after they finish"; exit 1
fi
command -v aws >/dev/null || {
  echo "Install AWS CLI v2 using the README prerequisite before host preparation"; exit 1;
}
python3 - "$VLA_SCRATCH_ROOT" <<'PY'
import pathlib, shutil, sys
p = pathlib.Path(sys.argv[1])
while not p.exists():
    p = p.parent
if p.stat().st_dev == pathlib.Path("/").stat().st_dev:
    raise SystemExit("Scratch is not a separately mounted filesystem; no disk will be formatted")
if shutil.disk_usage("/").free < 10 * 1024**3:
    raise SystemExit("Less than 10 GiB free on root for installing launcher tools")
PY
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git python3-venv pigz ca-certificates curl gnupg util-linux
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  conflicts=()
  for package in docker.io docker-compose docker-compose-v2 docker-buildx docker-doc podman-docker containerd runc; do
    if dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null | grep -qx installed; then
      conflicts+=("$package")
    fi
  done
  if [ "${#conflicts[@]}" -gt 0 ]; then apt-get remove -y "${conflicts[@]}"; fi
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  cat >/etc/apt/sources.list.d/docker.sources <<'EOF'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: jammy
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
fi
systemctl enable --now docker
if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi
install -d -m 0700 "$VLA_SCRATCH_ROOT"
docker compose version
df -hT / "$VLA_SCRATCH_ROOT"
echo "Host system prerequisites ready"
