# SageMaker GR00T Training Guide

*(Coming soon)*

Fine-tune GR00T N1.7 on Amazon SageMaker. This guide will cover:

- Deploying foundation infrastructure (Terraform)
- Building the GR00T training container
- Preparing and uploading training data
- Launching fine-tuning jobs with `training/groot/launch_training.py`
- Monitoring training progress
- Retrieving fine-tuned checkpoints

For now, see [Lab 1: Train GR00T](../workshop/lab-1-train-groot.md) in the workshop for the SageMaker-based workflow.

## Quick Start (if infra is already deployed)

```bash
# Dry run — preview the job
python3 training/groot/launch_training.py \
  --s3-bucket physical-ai-dev-datasets-<ACCOUNT_ID> \
  --dataset-prefix groot-data/ur3 \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/physical-ai-dev-sagemaker-role \
  --ecr-image <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/groot-training:latest \
  --base-model nvidia/GR00T-N1.7-3B \
  --max-steps 100 \
  --region us-east-2 \
  --dry-run

# Launch for real (drop --dry-run)
```
