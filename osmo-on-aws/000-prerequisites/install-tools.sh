#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Install required tools for OSMO on AWS deployment

set -o errexit
set -o nounset
set -o pipefail

TERRAFORM_VERSION="${TERRAFORM_VERSION:-1.7.0}"
KUBECTL_VERSION="${KUBECTL_VERSION:-1.35.0}"
HELM_VERSION="${HELM_VERSION:-3.14.0}"
TFLINT_VERSION="${TFLINT_VERSION:-0.53.0}"

info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*" >&2; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }
fatal() { error "$@"; exit 1; }

check_command() { command -v "$1" &>/dev/null; }

detect_os() {
  case "$(uname -s)" in
    Linux*)  [ -f /etc/os-release ] && . /etc/os-release && echo "${ID:-linux}" || echo "linux" ;;
    Darwin*) echo "darwin" ;;
    *)       echo "unknown" ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64)  echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *)       echo "amd64" ;;
  esac
}

install_awscli() {
  if check_command aws; then
    info "AWS CLI already installed: $(aws --version)"
    return 0
  fi
  info "Installing AWS CLI..."
  local os
  os=$(detect_os)
  case "$os" in
    darwin)
      if check_command brew; then
        brew install awscli
      else
        curl -sL "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o /tmp/AWSCLIV2.pkg
        sudo installer -pkg /tmp/AWSCLIV2.pkg -target /
      fi
      ;;
    ubuntu|debian)
      sudo apt-get update && sudo apt-get install -y unzip curl
      curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o /tmp/awscliv2.zip
      unzip -q /tmp/awscliv2.zip -d /tmp && sudo /tmp/aws/install
      rm -rf /tmp/awscliv2.zip /tmp/aws
      ;;
    *) fatal "Unsupported OS: $os" ;;
  esac
  info "AWS CLI installed: $(aws --version)"
}

install_kubectl() {
  if check_command kubectl; then
    info "kubectl already installed: $(kubectl version --client 2>/dev/null | head -1)"
    return 0
  fi
  info "Installing kubectl v${KUBECTL_VERSION}..."
  local os arch platform
  os=$(detect_os)
  arch=$(detect_arch)
  platform=$([[ "$os" == "darwin" ]] && echo "darwin" || echo "linux")
  curl -sLO "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/${platform}/${arch}/kubectl"
  chmod +x kubectl && sudo mv kubectl /usr/local/bin/
  info "kubectl installed"
}

install_helm() {
  if check_command helm; then
    info "Helm already installed: $(helm version --short)"
    return 0
  fi
  info "Installing Helm..."
  local os
  os=$(detect_os)
  case "$os" in
    darwin) brew install helm ;;
    ubuntu|debian) curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash ;;
    *) fatal "Unsupported OS: $os" ;;
  esac
  info "Helm installed: $(helm version --short)"
}

install_terraform() {
  if check_command terraform; then
    info "Terraform already installed: $(terraform version | head -1)"
    return 0
  fi
  info "Installing Terraform v${TERRAFORM_VERSION}..."
  local os arch platform
  os=$(detect_os)
  arch=$(detect_arch)
  platform=$([[ "$os" == "darwin" ]] && echo "darwin" || echo "linux")
  curl -sLO "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_${platform}_${arch}.zip"
  unzip -q "terraform_${TERRAFORM_VERSION}_${platform}_${arch}.zip"
  chmod +x terraform && sudo mv terraform /usr/local/bin/
  rm "terraform_${TERRAFORM_VERSION}_${platform}_${arch}.zip"
  info "Terraform installed: $(terraform version | head -1)"
}

install_jq() {
  if check_command jq; then
    info "jq already installed: $(jq --version)"
    return 0
  fi
  info "Installing jq..."
  local os
  os=$(detect_os)
  case "$os" in
    darwin) brew install jq ;;
    ubuntu|debian) sudo apt-get update && sudo apt-get install -y jq ;;
    *) fatal "Unsupported OS: $os" ;;
  esac
  info "jq installed: $(jq --version)"
}

install_yq() {
  if check_command yq; then
    info "yq already installed: $(yq --version)"
    return 0
  fi
  info "Installing yq..."
  local os arch platform
  os=$(detect_os)
  arch=$(detect_arch)
  platform=$([[ "$os" == "darwin" ]] && echo "darwin" || echo "linux")
  local version="4.44.1"
  curl -sL "https://github.com/mikefarah/yq/releases/download/v${version}/yq_${platform}_${arch}" -o /tmp/yq
  chmod +x /tmp/yq && sudo mv /tmp/yq /usr/local/bin/yq
  info "yq installed: $(yq --version)"
}

install_openssl() {
  if check_command openssl; then
    info "openssl already installed: $(openssl version)"
    return 0
  fi
  info "Installing openssl..."
  local os
  os=$(detect_os)
  case "$os" in
    darwin) brew install openssl ;;
    ubuntu|debian) sudo apt-get update && sudo apt-get install -y openssl ;;
    *) fatal "Unsupported OS: $os" ;;
  esac
  info "openssl installed: $(openssl version)"
}

install_python_pip() {
  if check_command python3 && check_command pip3; then
    info "Python3 and pip already installed"
    return 0
  fi
  info "Installing Python3 and pip..."
  local os
  os=$(detect_os)
  case "$os" in
    darwin) brew install python3 ;;
    ubuntu|debian) sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv ;;
    *) fatal "Unsupported OS: $os" ;;
  esac
  info "Python3 installed: $(python3 --version)"
}

install_pre_commit() {
  if check_command pre-commit; then
    info "pre-commit already installed: $(pre-commit --version)"
    return 0
  fi
  info "Installing pre-commit..."
  pip3 install --user pre-commit
  # Ensure user bin is in PATH
  export PATH="$HOME/.local/bin:$PATH"
  info "pre-commit installed: $(pre-commit --version)"
}

install_checkov() {
  if check_command checkov; then
    info "Checkov already installed: $(checkov --version 2>/dev/null | head -1)"
    return 0
  fi
  info "Installing Checkov..."
  pip3 install --user checkov
  export PATH="$HOME/.local/bin:$PATH"
  info "Checkov installed: $(checkov --version 2>/dev/null | head -1)"
}

install_tflint() {
  if check_command tflint; then
    info "TFLint already installed: $(tflint --version | head -1)"
    return 0
  fi
  info "Installing TFLint v${TFLINT_VERSION}..."
  local os arch platform
  os=$(detect_os)
  arch=$(detect_arch)
  platform=$([[ "$os" == "darwin" ]] && echo "darwin" || echo "linux")
  curl -sLO "https://github.com/terraform-linters/tflint/releases/download/v${TFLINT_VERSION}/tflint_${platform}_${arch}.zip"
  unzip -q "tflint_${platform}_${arch}.zip"
  chmod +x tflint && sudo mv tflint /usr/local/bin/
  rm "tflint_${platform}_${arch}.zip"
  info "TFLint installed: $(tflint --version | head -1)"
}

install_shellcheck() {
  if check_command shellcheck; then
    info "ShellCheck already installed: $(shellcheck --version | grep version: | head -1)"
    return 0
  fi
  info "Installing ShellCheck..."
  local os
  os=$(detect_os)
  case "$os" in
    darwin) brew install shellcheck ;;
    ubuntu|debian) sudo apt-get update && sudo apt-get install -y shellcheck ;;
    *) fatal "Unsupported OS: $os" ;;
  esac
  info "ShellCheck installed"
}

install_gitleaks() {
  if check_command gitleaks; then
    info "Gitleaks already installed: $(gitleaks version 2>/dev/null)"
    return 0
  fi
  info "Installing Gitleaks..."
  local os
  os=$(detect_os)
  case "$os" in
    darwin) brew install gitleaks ;;
    ubuntu|debian)
      local arch
      arch=$(detect_arch)
      local version="8.18.4"
      curl -sLO "https://github.com/gitleaks/gitleaks/releases/download/v${version}/gitleaks_${version}_linux_${arch/amd64/x64}.tar.gz"
      tar -xzf "gitleaks_${version}_linux_${arch/amd64/x64}.tar.gz" gitleaks
      chmod +x gitleaks && sudo mv gitleaks /usr/local/bin/
      rm "gitleaks_${version}_linux_${arch/amd64/x64}.tar.gz"
      ;;
    *) fatal "Unsupported OS: $os" ;;
  esac
  info "Gitleaks installed"
}

setup_pre_commit_hooks() {
  local repo_root="${1:-$(git rev-parse --show-toplevel 2>/dev/null || echo "")}"
  if [[ -z "$repo_root" ]]; then
    warn "Not in a git repository, skipping pre-commit hook setup"
    return 0
  fi
  if [[ -f "$repo_root/.pre-commit-config.yaml" ]]; then
    info "Setting up pre-commit hooks..."
    cd "$repo_root"
    pre-commit install
    info "Pre-commit hooks installed"
    info "Run 'pre-commit run --all-files' to test"
  else
    warn "No .pre-commit-config.yaml found, skipping hook setup"
  fi
}

main() {
  info "Installing OSMO on AWS prerequisites..."
  info "Detected OS: $(detect_os), Architecture: $(detect_arch)"
  echo

  # Core tools
  info "=== Installing Core Tools ==="
  install_awscli
  install_kubectl
  install_helm
  install_terraform
  install_jq
  install_yq
  install_openssl
  echo

  # Development and security tools
  info "=== Installing Development & Security Tools ==="
  install_python_pip
  install_pre_commit
  install_checkov
  install_tflint
  install_shellcheck
  install_gitleaks
  echo

  # Setup pre-commit hooks if in repo
  info "=== Setting Up Pre-commit Hooks ==="
  setup_pre_commit_hooks "$(dirname "$0")/.."
  echo

  # Add local bin to PATH reminder
  info "=== Setup Complete ==="
  info "All prerequisites installed!"
  echo
  warn "Add the following to your shell profile (~/.bashrc or ~/.zshrc):"
  # shellcheck disable=SC2016
  echo '  export PATH="$HOME/.local/bin:$PATH"'
  echo
  info "Next: Run ./aws-env-init.sh"
}

main "$@"
