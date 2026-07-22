# SageMaker RL Training Guide

Train Isaac Lab reinforcement learning policies on Amazon SageMaker. SageMaker provisions GPU instances on demand, runs training, and tears them down automatically — no persistent infrastructure to manage.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Foundation (Terraform)                                                │
│    S3 buckets · ECR repos · IAM roles · SSM parameters                   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Container Build (CodeBuild)                                           │
│    Pull NGC base (16 GB) · Install RL deps · Push to ECR                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Training Job (SageMaker)                                              │
│    ml.g5.xlarge · Isaac Lab PPO · 1024-4096 parallel envs                │
│    Auto-provisions → Trains → Uploads to S3 → Terminates                 │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Checkpoints → S3                                                      │
│    s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/isaac-lab/<job_name>/    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Time:** ~1 hour (30 min setup + training)
**Cost:** ~$3 for smoke test (50 iterations), ~$28 for full training (1500 iterations)

---

## Prerequisites

| Requirement | How to verify |
|-------------|---------------|
| AWS CLI configured | `aws sts get-caller-identity` |
| Terraform >= 1.5 | `terraform --version` |
| Python 3.11+ | `python3 --version` |
| boto3 installed | `python3 -c "import boto3"` |
| NVIDIA NGC API key | Generate at https://ngc.nvidia.com/setup/api-key |
| GPU quota for ml.g5.xlarge | Check SageMaker service quotas in your target region |

---

## Step 1: Deploy Foundation Infrastructure

The foundation creates the IAM role, S3 buckets, and ECR repos that SageMaker needs.

```bash
cd foundation/infra
terraform init
terraform plan -var="aws_region=us-east-2"
terraform apply -var="aws_region=us-east-2"
```

**What SageMaker uses from Foundation:**

| Resource | Purpose |
|----------|---------|
| IAM Role: `physical-ai-dev-sagemaker-role` | Execution role with S3 + ECR access |
| ECR Repo: `physical-ai/isaac-lab` | Training container image |
| S3: `physical-ai-dev-checkpoints-<ACCOUNT_ID>` | Training output (model.tar.gz) |
| SSM Parameters | Auto-discovery (no hardcoded ARNs) |

---

## Step 2: Store NGC API Key

```bash
aws secretsmanager create-secret \
  --name "physical-ai/ngc-api-key" \
  --secret-string "<YOUR_NGC_API_KEY>" \
  --region us-east-2 \
  --description "NVIDIA NGC API key for pulling container images from nvcr.io"
```

---

## Step 3: Deploy Isaac Lab CodeBuild

```bash
cd isaac-lab-on-aws/infra
terraform init
terraform apply -var="aws_region=us-east-2"
```

This creates the CodeBuild project. No VPC or Batch resources needed for the SageMaker path.

---

## Step 4: Build the Training Container

### 4a. Package and upload the source

```bash
# From repo root
zip -r /tmp/source.zip . \
  -x ".git/*" \
  -x "*/node_modules/*" \
  -x "*/.terraform/*" \
  -x "*/terraform.tfstate*"

aws s3 cp /tmp/source.zip \
  s3://physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source.zip \
  --region us-east-2
```

### 4b. Trigger the build

```bash
aws codebuild start-build \
  --project-name "physical-ai-dev-isaac-lab-build" \
  --region us-east-2 \
  --environment-variables-override \
    "name=NGC_SECRET_NAME,value=physical-ai/ngc-api-key,type=PLAINTEXT" \
  --source-type-override S3 \
  --source-location-override \
    "physical-ai-dev-datasets-<ACCOUNT_ID>/codebuild-source/source.zip"
```

**Build time:** 30-60 minutes. Monitor in the [CodeBuild console](https://us-east-2.console.aws.amazon.com/codesuite/codebuild/projects/physical-ai-dev-isaac-lab-build/history?region=us-east-2).

---

## Step 5: Launch Training

### Preview first (no cost)

```bash
AWS_DEFAULT_REGION=us-east-2 python3 training/scripts/launch_rl.py --dry-run
```

This prints the exact SageMaker API call without making any AWS calls.

### Smoke test (~$3, 5-10 min)

```bash
AWS_DEFAULT_REGION=us-east-2 python3 training/scripts/launch_rl.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 1024 \
  --max-iterations 100 \
  --instance-type ml.g5.xlarge
```

### Full training (~$28, 2-4 hours)

```bash
AWS_DEFAULT_REGION=us-east-2 python3 training/scripts/launch_rl.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 \
  --max-iterations 1500 \
  --instance-type ml.g5.12xlarge
```

---

## Step 6: Monitor Training

```bash
aws sagemaker describe-training-job \
  --training-job-name <JOB_NAME> \
  --region us-east-2 \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus}'
```

Job lifecycle: `InProgress` → `Completed`

Or view in the [SageMaker console](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs) under **Training → Training jobs**.

### View training logs

```bash
aws logs get-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name <JOB_NAME>/algo-1-<timestamp> \
  --region us-east-2 \
  --query 'events[*].message' --output text | tail -20
```

---

## Step 7: Retrieve Checkpoints

After training completes, SageMaker packages model artifacts into `model.tar.gz`:

```bash
aws s3 ls s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/isaac-lab/<JOB_NAME>/ \
  --region us-east-2 --recursive
```

Download and extract:

```bash
aws s3 cp \
  s3://physical-ai-dev-checkpoints-<ACCOUNT_ID>/isaac-lab/<JOB_NAME>/output/model.tar.gz .

tar -xzf model.tar.gz
ls logs/rsl_rl/*/*/model_*.pt   # trained policy checkpoint
```

---

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | `Isaac-Velocity-Flat-Anymal-D-v0` | Isaac Lab task name |
| `--num-envs` | `4096` | Parallel simulation environments |
| `--max-iterations` | `50` | PPO training iterations |
| `--framework` | `rsl_rl` | RL framework: `rsl_rl`, `skrl`, `rl_games` |
| `--instance-type` | `ml.g5.xlarge` | GPU instance type |
| `--instance-count` | `1` | Number of instances (multi-node) |
| `--runtime-min` | `60` | Max runtime in minutes |
| `--dry-run` | — | Preview only, no AWS calls |

---

## Multi-GPU / Multi-Node

### Single node, multiple GPUs

```bash
AWS_DEFAULT_REGION=us-east-2 python3 training/scripts/launch_rl.py \
  --instance-type ml.g5.12xlarge \
  --instance-count 1
```

Uses 4 A10G GPUs on one instance. The SageMaker entrypoint launches torchrun with `--nproc_per_node=4`. This is the validated multi-GPU path.

### Multiple nodes (experimental)

```bash
AWS_DEFAULT_REGION=us-east-2 python3 training/scripts/launch_rl.py \
  --instance-type ml.g5.12xlarge \
  --instance-count 2
```

SageMaker handles inter-node networking. The entrypoint reads `resourceconfig.json` and configures torchrun for distributed training.

> ⚠️ Multi-node NCCL across instances is wired but not yet validated on hardware.

---

## How It Works Under the Hood

1. `launch_rl.py` resolves your account's ECR image, IAM role, and S3 bucket from SSM parameters
2. Calls `sagemaker.create_training_job()` with hyperparameters as environment variables
3. SageMaker provisions an `ml.g5.xlarge`, pulls the container from ECR
4. The container runs `sm-train-entrypoint.sh` which:
   - Reads SageMaker's `resourceconfig.json` for cluster topology
   - Launches `torchrun` with the appropriate node/rank configuration
   - Runs Isaac Lab's `rsl_rl/train.py` with your hyperparameters
5. Training output writes to `/opt/ml/model/`
6. SageMaker packages `/opt/ml/model/` → `model.tar.gz` → uploads to S3
7. Instance terminates automatically

---

## Cost Control

- SageMaker **auto-terminates** when training completes — no manual cleanup needed
- Set `--runtime-min` to cap spending (default: 60 min)
- Use `--dry-run` to preview before spending
- Smoke test (50 iterations, ml.g5.xlarge): ~$3
- Full training (1500 iterations, ml.g5.12xlarge): ~$28

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ResourceLimitExceeded` | GPU quota too low | Request `ml.g5.xlarge` quota increase in Service Quotas |
| Job fails immediately | IAM role can't pull from ECR | Verify `physical-ai-dev-sagemaker-role` has ECR permissions |
| `No module named 'isaaclab'` | Container image not built | Run the CodeBuild step (Step 4) |
| Training doesn't start | Image not in ECR | Check `aws ecr describe-images --repository-name physical-ai/isaac-lab --region us-east-2` |
| `CUDA out of memory` | Too many parallel envs | Reduce `--num-envs` (4096 → 2048 → 1024) |
| Job timeout | Training too slow for runtime limit | Increase `--runtime-min` |

---

## Next Steps

- Increase `--max-iterations` to 1500+ for a converged policy
- Export to ONNX for edge deployment (see [Lab 5 Step 4](../workshop/lab-5-rl-refinement-with-isaac.md))
- Try the [AWS Batch path](batch-rl-training-guide.md) for persistent GPU fleet control
