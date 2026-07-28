# Foundation Infrastructure

Shared AWS resources that all other toolchain components depend on. Deploy this first.

---

## What It Creates

| Resource | Name Pattern | Purpose |
|----------|-------------|---------|
| S3 Bucket | `physical-ai-<env>-datasets-<account>` | Training datasets (LeRobot format, USD) |
| S3 Bucket | `physical-ai-<env>-models-<account>` | Trained models (ONNX, TensorRT) |
| S3 Bucket | `physical-ai-<env>-checkpoints-<account>` | Training checkpoints (14-day auto-expiry) |
| ECR Repo | `physical-ai/groot-training` | GR00T fine-tuning container |
| ECR Repo | `physical-ai/groot-inference` | GR00T inference container |
| ECR Repo | `physical-ai/isaac-lab` | Isaac Lab RL training container |
| ECR Repo | `physical-ai/cosmos-transfer` | Cosmos Transfer 2.5 container |
| ECR Repo | `physical-ai/cosmos3` | Cosmos 3 generation container |
| IAM Role | `physical-ai-<env>-sagemaker-role` | SageMaker execution role |
| IAM Role | `physical-ai-<env>-cosmos-role` | Cosmos EC2 instance role |
| VPC | `physical-ai-<env>-vpc` | Shared networking (2 public + 3 private subnets) |
| NAT Gateway | `physical-ai-<env>-nat` | Outbound internet for private subnets |
| SSM Parameters | `/physical-ai/*` | Discovery mechanism for downstream components |

---

## Deploy with Terraform

```bash
cd foundation/infra
terraform init
terraform plan -var="aws_region=us-east-2"
terraform apply -var="aws_region=us-east-2"
```

### Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | `us-east-1` | AWS region to deploy into |
| `environment` | `dev` | Environment name (dev, staging, prod) |
| `project_name` | `physical-ai` | Project name prefix for all resources |

### Outputs

```bash
terraform output
```

| Output | Description |
|--------|-------------|
| `datasets_bucket_name` | S3 bucket for training datasets |
| `models_bucket_name` | S3 bucket for trained models |
| `checkpoints_bucket_name` | S3 bucket for training checkpoints |
| `sagemaker_role_arn` | SageMaker execution role ARN |
| `cosmos_instance_profile_name` | Instance profile for Cosmos EC2 |

---

## Deploy with CDK

```bash
cd cdk
npm install
npx cdk deploy PhysicalAi-dev-Foundation --context mode=simple
```

The CDK Foundation stack creates the same resources as Terraform plus triggers CodeBuild projects that automatically build all container images.

### CDK Modes

| Mode | Command | What it deploys |
|------|---------|-----------------|
| Simple | `--context mode=simple` | Foundation only (S3, ECR, IAM, CodeBuild) |
| Full | `--context mode=full` | Foundation + Network + EKS + OSMO |

---

## How Other Components Find These Resources

All resources are published as SSM Parameters under `/physical-ai/`:

| Parameter | Value |
|-----------|-------|
| `/physical-ai/datasets-bucket` | Datasets bucket name |
| `/physical-ai/models-bucket` | Models bucket name |
| `/physical-ai/checkpoints-bucket` | Checkpoints bucket name |
| `/physical-ai/sagemaker-role-arn` | SageMaker execution role ARN |
| `/physical-ai/cosmos-instance-profile` | Cosmos instance profile name |
| `/physical-ai/ecr/groot-training` | GR00T training ECR URI |
| `/physical-ai/ecr/groot-inference` | GR00T inference ECR URI |
| `/physical-ai/ecr/isaac-lab` | Isaac Lab ECR URI |
| `/physical-ai/ecr/cosmos-transfer` | Cosmos Transfer ECR URI |
| `/physical-ai/ecr/cosmos3` | Cosmos 3 ECR URI |
| `/physical-ai/vpc-id` | Shared VPC ID |
| `/physical-ai/private-subnet-ids` | Private subnet IDs (comma-separated) |

Downstream components (Isaac Lab, GR00T, etc.) read these parameters at deploy time — no hardcoded ARNs or bucket names needed.

---

## Teardown

```bash
cd foundation/infra
terraform destroy -var="aws_region=us-east-2"
```

> **Note:** S3 buckets with data won't be deleted unless emptied first. ECR repos with images require `force_delete = true` or manual image deletion.

---

## Prerequisites

- AWS CLI configured with appropriate permissions
- Terraform >= 1.5 (for Terraform path)
- Node.js 18+ and AWS CDK CLI (for CDK path)

---

## Cost

The foundation resources are essentially free at rest:
- S3 buckets: pay per GB stored
- ECR repos: pay per GB stored
- VPC + NAT Gateway: ~$32/month (NAT gateway hourly charge)
- IAM roles, SSM parameters: free
