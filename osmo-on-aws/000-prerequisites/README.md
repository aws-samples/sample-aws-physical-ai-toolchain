# OSMO on AWS - Prerequisites

This directory contains scripts to set up prerequisites for OSMO deployment on AWS.

## Quick Start

```bash
# 1. Install required tools
./install-tools.sh

# 2. Configure AWS credentials (if not already done)
#    See: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html

# 3. Initialize environment (verifies credentials, exports AWS region)
./aws-env-init.sh --region us-west-2

# 4. Verify everything is ready
./verify-prerequisites.sh
```

## Scripts

### install-tools.sh

Installs required CLI tools:
- AWS CLI v2
- kubectl
- Helm
- Terraform
- jq

Supports Ubuntu/Debian and macOS.

### aws-env-init.sh

Verifies AWS credentials and exports the AWS region for the CLI.

```bash
# Basic usage
./aws-env-init.sh --region us-west-2

# With AWS profile
./aws-env-init.sh --profile my-profile --region us-east-1

# Export to file for later sourcing
./aws-env-init.sh --export
source .env
```

### verify-prerequisites.sh

Verifies all prerequisites are met:
- Required tools installed
- AWS credentials configured
- AWS permissions available

## Required AWS Permissions

The deployment requires IAM permissions for:
- EC2 (VPC, subnets, security groups, NAT gateways)
- EKS (cluster, node groups, addons)
- RDS (PostgreSQL instances)
- ElastiCache (Redis clusters)
- S3 (buckets)
- Secrets Manager (secrets)
- KMS (encryption keys)
- IAM (roles, policies)

For a minimal IAM policy, see the [AWS documentation](https://docs.aws.amazon.com/eks/latest/userguide/security-iam.html).

## Environment Variables

After running `aws-env-init.sh`, the following variables are set:

| Variable | Description |
|----------|-------------|
| `AWS_REGION` | AWS region for the AWS CLI |
| `AWS_DEFAULT_REGION` | AWS region for the AWS CLI |

## Next Steps

After prerequisites are verified:

1. Navigate to the IaC directory: `cd ../001-iac`
2. Copy example tfvars: `cp terraform.tfvars.example terraform.tfvars`
3. Edit configuration: `vim terraform.tfvars`
4. Deploy infrastructure: `terraform init && terraform apply`
