# Lab 1: Train Your First Robot Policy

**Time:** 2 hours (15 min deploy + 10 min prep + training time)
**Mode:** Simple (Path A)
**Goal:** Deploy infrastructure, fine-tune GR00T on a demo dataset, get an evaluation video.

---

## What You'll Do

```
Deploy CDK → Upload dataset → Build container → Launch training → View results
     5 min       2 min           10 min           ~5-11 hrs         5 min
```

You won't wait for training to finish during the lab — you'll launch the job, verify it's running, and review pre-computed results.

---

## Step 1: Deploy the Foundation Stack

```bash
cd cdk
npx cdk deploy PhysicalAi-dev-Foundation --context mode=simple --require-approval never
```

**What this creates:**
- S3 buckets (datasets, models, checkpoints)
- ECR repositories (groot-training, inference)
- SageMaker execution role
- CodeBuild project for container builds

**Verify:** Check the CloudFormation outputs:
```bash
aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' --output table
```

You should see bucket names, role ARN, and ECR URIs.

---

## Step 2: Get a Demo Dataset

For this lab, we use a pre-recorded dataset of a robot arm performing pick-and-place.

**Option A: Download pre-built demo dataset** (recommended for workshop)
```bash
# Download from the workshop S3 bucket (public, read-only)
aws s3 sync s3://physical-ai-workshop-datasets/pick-and-place-50ep/ ./data/demo-dataset/
```

**Option B: Use a HuggingFace LeRobot dataset**
```bash
pip install lerobot
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('lerobot/aloha_sim_insertion_human')
dataset.save_to_disk('./data/demo-dataset/')
"
```

**Verify:**
```bash
ls data/demo-dataset/
# Should show: data/ videos/ meta/
```

---

## Step 3: Upload Dataset to S3

```bash
DATASETS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`DatasetsBucketName`].OutputValue' \
  --output text)

aws s3 sync ./data/demo-dataset/ s3://$DATASETS_BUCKET/groot-data/lab1/dataset/
echo "Uploaded to: s3://$DATASETS_BUCKET/groot-data/lab1/dataset/"
```

---

## Step 4: Build the Training Container

```bash
# Trigger CodeBuild (or build locally)
aws codebuild start-build --project-name physical-ai-groot-training-build

# Monitor build (~5-10 minutes)
aws codebuild batch-get-builds --ids $(aws codebuild list-builds-for-project \
  --project-name physical-ai-groot-training-build \
  --query 'ids[0]' --output text) \
  --query 'builds[0].buildStatus'
```

**Or build locally** (faster if you have Docker):
```bash
ECR_REPO=$(aws cloudformation describe-stacks \
  --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`GrootTrainingRepoUri`].OutputValue' \
  --output text)

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_REPO

cd containers/groot-training
docker build -t groot-training .
docker tag groot-training:latest $ECR_REPO:latest
docker push $ECR_REPO:latest
cd ../..
```

---

## Step 5: Launch the Training Job

```bash
SAGEMAKER_ROLE=$(aws cloudformation describe-stacks \
  --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' \
  --output text)

python training/groot/launch_training.py \
  --s3-bucket $DATASETS_BUCKET \
  --dataset-prefix groot-data/lab1 \
  --role-arn $SAGEMAKER_ROLE \
  --ecr-image $ECR_REPO:latest \
  --max-steps 5000 \
  --instance-type ml.g5.12xlarge
```

**What happens:**
1. SageMaker provisions a `ml.g5.12xlarge` (4× A10G GPUs)
2. Pulls your training container from ECR
3. Downloads dataset from S3
4. Runs GR00T fine-tuning for 5000 steps (~11 hours)
5. Uploads `model.tar.gz` to S3

---

## Step 6: Monitor Training

```bash
# Get the job name (printed by launch script, or find it)
JOB_NAME=$(aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending \
  --query 'TrainingJobSummaries[0].TrainingJobName' --output text)

# Check status
aws sagemaker describe-training-job --training-job-name $JOB_NAME \
  --query '{Status: TrainingJobStatus, Secondary: SecondaryStatus}'

# Stream logs (Ctrl+C to stop)
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix $JOB_NAME
```

**What to look for in logs:**
```
Step 100/5000  loss: 0.45  lr: 1e-4
Step 500/5000  loss: 0.18  lr: 1e-4
Step 1000/5000 loss: 0.08  lr: 1e-4   ← good convergence
```

---

## Step 7: View Results

After training completes (or using pre-computed results for the workshop):

```bash
MODELS_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`ModelsBucketName`].OutputValue' \
  --output text)

# Download the evaluation video
aws s3 cp s3://$MODELS_BUCKET/training-output/lab1/$JOB_NAME/output/eval_video.mp4 ./

# Download metrics
aws s3 cp s3://$MODELS_BUCKET/training-output/lab1/$JOB_NAME/output/eval_metrics.json ./
cat eval_metrics.json
```

**Expected results:**
- `success_rate`: >80% (robot successfully picks objects)
- `avg_cycle_time`: 2-3 seconds
- `eval_video.mp4`: 30-second video of the trained policy in action

---

## ✅ Lab 1 Checkpoint

You've completed Lab 1 if:
- [ ] Foundation stack deployed (S3 + ECR + IAM visible in console)
- [ ] Training job launched (status: InProgress or Completed)
- [ ] You can explain what GR00T is fine-tuning (projector + diffusion model, backbone frozen)
- [ ] You know where the trained model ends up (S3 `models` bucket)

---

## What's Next

- **Lab 2:** Add simulation-based training with Isaac Lab + OSMO (Path B)
- **Lab 3:** Export the model to TensorRT and deploy to edge via Greengrass
- **On your own:** Try with your own dataset (record teleop data, convert to LeRobot format)

---

## Troubleshooting

**Training job fails immediately with "ResourceLimitExceeded"**
→ You don't have GPU quota. See Lab 0 prerequisites.

**Container build fails with "pull access denied"**
→ NGC base image requires auth. Set `NGC_API_KEY` in CodeBuild environment variables.

**Loss doesn't decrease after 500 steps**
→ Dataset quality issue. Check that episodes have consistent camera angles and at least 50 demonstrations.

---

**Previous:** [← Lab 0: Prerequisites](lab-0-prerequisites.md)
**Next:** [Lab 2: Simulation Training with OSMO →](lab-2-isaac-lab-osmo.md)
