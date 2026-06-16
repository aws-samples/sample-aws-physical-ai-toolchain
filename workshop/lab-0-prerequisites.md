# Lab 0: Prerequisites

**Goal:** Verify your environment is ready to deploy the Physical AI Toolchain.
**Time:** 30 minutes
**Cost:** Free (no AWS resources created)

---

## What You Need

| Requirement | Why | How to check |
|-------------|-----|--------------|
| AWS Account with GPU quota | Training runs on `ml.g5.12xlarge` | `aws service-quotas get-service-quota --service-code sagemaker --quota-code L-3B05A85A` |
| AWS CLI v2 configured | Deploy infrastructure | `aws sts get-caller-identity` |
| Node.js 18+ | CDK requires it | `node --version` |
| AWS CDK CLI | Deploy stacks | `npx cdk --version` |
| Docker | Build training containers | `docker --version` |
| NVIDIA NGC account (free) | Pull Isaac Sim base images | Sign up at https://ngc.nvidia.com |
| Hugging Face token | Pull GR00T base model | Create at https://huggingface.co/settings/tokens |

## Step-by-Step Setup

### 1. Clone the repo

```bash
git clone https://gitlab.aws.dev/devris/aws-physical-ai-toolchain.git
cd aws-physical-ai-toolchain
```

### 2. Install CDK dependencies

```bash
cd cdk
npm install
cd ..
```

### 3. Bootstrap CDK (once per account/region)

```bash
npx cdk bootstrap aws://<ACCOUNT_ID>/us-east-1
```

### 4. Set environment variables

```bash
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION=us-east-1
export HF_TOKEN=hf_xxxx  # Your Hugging Face token
```

### 5. Verify

```bash
cd cdk
npx cdk synth --context mode=simple 2>&1 | head -5
# Should print: Successfully synthesized to cdk.out
```

---

## GPU Quota

SageMaker training (Path A) requires quota for `ml.g5.12xlarge` instances. Request it:

1. Go to [Service Quotas → SageMaker](https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas)
2. Search for "ml.g5.12xlarge for training job usage"
3. Request increase to at least 1

**Quota approval typically takes 1-24 hours.** Request before the workshop.

---

## ✅ Ready Check

Run this to verify everything:

```bash
echo "=== Prerequisites Check ==="
echo "AWS CLI:     $(aws --version 2>&1 | head -1)"
echo "Account:     $(aws sts get-caller-identity --query Account --output text)"
echo "Region:      ${CDK_DEFAULT_REGION:-not set}"
echo "Node:        $(node --version)"
echo "CDK:         $(npx cdk --version 2>/dev/null || echo 'not installed')"
echo "Docker:      $(docker --version 2>/dev/null || echo 'not installed')"
echo "HF_TOKEN:    ${HF_TOKEN:+set}${HF_TOKEN:-NOT SET}"
echo "==========================="
```

All items should show versions or "set". If anything says "not installed" or "NOT SET", fix it before proceeding to Lab 1.

---

**Next:** [Lab 1: Train Your First Robot Policy →](lab-1-train-groot.md)
