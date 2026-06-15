# Lab 1: Train a Robot Policy from Demonstrations

**Time:** 2 hours (30 min hands-on + training runs in background)
**Cost:** ~$2 for smoke test, ~$79 for full training
**Goal:** Fine-tune NVIDIA GR00T on teleoperation data → get a model that predicts robot joint actions from camera images

---

## What You're Building

A robot manipulation policy trained via **imitation learning**. Here's what that means:

1. A human teleoperated a robot arm to perform a task (peg insertion) — 50 demonstrations were recorded
2. Each demo captured: camera images + joint positions at every timestep
3. You fine-tune GR00T (a 3B parameter vision-language-action model) to learn the pattern: "given this image → predict these joint movements"
4. The result is a model that can control the robot autonomously for that task

**What GR00T fine-tuning actually trains:**
- The backbone (vision encoder + language model) stays **frozen** — NVIDIA already trained this on millions of robot videos
- Only the **projector** (maps vision features to action space) and **diffusion action head** (predicts action sequences) get updated
- This is why 50 demos are enough — you're not training from scratch, you're teaching an existing model a new task

**The limitation of imitation learning alone:**
- The model can replicate what it saw in the demos (~70-80% success)
- It struggles with variations it hasn't seen (different object positions, lighting changes)
- Lab 2 fixes this with RL refinement in simulation

---

## The Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Demo Data   │────▶│  SageMaker   │────▶│   Evaluate   │────▶│   Model      │
│  (LeRobot    │     │  Training    │     │   (action    │     │   Registry   │
│   parquet +  │     │  Job         │     │    error)    │     │              │
│   video)     │     │              │     │              │     │  groot-models│
│              │     │  ml.g5.      │     │              │     │  version N   │
│  in S3       │     │  12xlarge    │     │  in S3       │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘

Orchestrated by: SageMaker Pipeline (groot-finetune-pipeline)
```

**Key details:**
- **Instance:** `ml.g5.12xlarge` — 4× NVIDIA A10G GPUs, 48 GB GPU RAM, 192 GB system RAM
- **Container:** `groot-training` (7.1 GB, based on `nvidia/cuda:12.4.1-devel-ubuntu22.04`)
- **Dataset format:** LeRobot v2 — parquet files (actions, states, episode metadata) + MP4 video (camera observations)
- **Training time:** ~8 sec/step. 100 steps = 15 min (smoke test). 5000 steps = 11 hrs (full)
- **Cost:** SageMaker on-demand pricing for ml.g5.12xlarge is ~$7.09/hr. No idle cost — you only pay while training runs.
- **Output:** `model.tar.gz` in S3 containing: model checkpoint + `eval_report.json` + `eval_action_error.png`

---

## Prerequisites

- Foundation stack deployed (`cdk deploy --context mode=simple` — see Lab 0)
- Docker running locally (for container build)
- `HF_TOKEN` environment variable set (HuggingFace token for downloading GR00T base model weights)
  - Get one at https://huggingface.co/settings/tokens
  - Accept the GR00T license at https://huggingface.co/nvidia/GR00T-N1.7-3B
- Python 3.11+ with `boto3` and `huggingface-hub` installed

---

## Step 1: Verify Infrastructure

```bash
# Check the Foundation stack is deployed
aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' --output table
```

You should see:
| Key | Value |
|-----|-------|
| DatasetsBucketName | `physical-ai-dev-datasets-<ACCOUNT_ID>` |
| ModelsBucketName | `physical-ai-dev-models-<ACCOUNT_ID>` |
| SageMakerRoleArn | `arn:aws:iam::<ACCOUNT_ID>:role/physical-ai-dev-sagemaker-role` |
| GrootTrainingRepoUri | `<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/physical-ai/groot-training` |

Save these values — you'll use them throughout the lab:
```bash
export BUCKET=$(aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`DatasetsBucketName`].OutputValue' --output text)
export ROLE_ARN=$(aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' --output text)
export ECR_URI=$(aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`GrootTrainingRepoUri`].OutputValue' --output text)
```

---

## Step 2: Download the Demo Dataset

```bash
python training/groot/download_demo_dataset.py --output ./data/demo-dataset
```

**What this downloads:** `lerobot/aloha_sim_insertion_human` from HuggingFace — 50 episodes of a bimanual robot (ALOHA) performing peg insertion, teleoperated by a human in simulation.

**Dataset structure:**
```
data/demo-dataset/
├── data/chunk-000/          # Parquet files: actions (14 joints), states, timestamps
│   ├── file-000.parquet     # Each row = one timestep (50Hz)
│   ├── file-001.parquet     # Columns: action (14-dim array), observation.state, episode_index
│   └── ...
├── meta/
│   ├── info.json            # Dataset metadata (robot type, fps, action dimensions)
│   ├── episodes/            # Episode boundaries
│   └── stats.json           # Action/state statistics (mean, std)
└── videos/                  # Camera observations (top-down view)
    └── observation.images.top/chunk-000/file-000.mp4
```

**Size:** ~87 MB (50 episodes × ~500 frames each = 25,000 total frames)

---

## Step 3: Upload Dataset to S3

```bash
aws s3 sync ./data/demo-dataset/ "s3://$BUCKET/groot-data/demo/dataset/"
```

---

## Step 4: Build and Push the Training Container

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URI

# Build (takes ~5 min first time, uses Docker layer cache after)
cd containers/groot-training
docker build --platform linux/amd64 -t groot-training .

# Tag and push
docker tag groot-training:latest $ECR_URI:latest
docker push $ECR_URI:latest
cd ../..
```

**What's in the container:**
- CUDA 12.4 + PyTorch 2.5 (GPU compute)
- HuggingFace transformers + diffusers (model loading)
- LeRobot (dataset loading)
- pandas + matplotlib (eval report generation)
- `train_entrypoint.py` — the script SageMaker runs

---

## Step 5: Launch Training (Smoke Test)

Start with 100 steps to verify everything works (~15 min, ~$2):

```bash
export HF_TOKEN="your-huggingface-token-here"

python training/groot/launch_training.py \
  --s3-bucket $BUCKET \
  --dataset-prefix groot-data/demo \
  --role-arn $ROLE_ARN \
  --ecr-image $ECR_URI:latest \
  --max-steps 100 \
  --instance-type ml.g5.12xlarge \
  --region us-east-1
```

**What SageMaker does:**
1. Provisions an ml.g5.12xlarge instance (~3-5 min)
2. Pulls your container from ECR (~2 min, 7.1 GB)
3. Downloads dataset from S3 to `/opt/ml/input/data/training/` (~30 sec)
4. Runs `train_entrypoint.py` which:
   - Downloads GR00T N1.7-3B base model from HuggingFace (~5 min first time)
   - Fine-tunes projector + action head for 100 steps
   - Generates eval report (MSE baselines on held-out 20% of data)
   - Saves everything to `/opt/ml/model/`
5. Uploads `model.tar.gz` to S3 automatically
6. Terminates the instance (no idle charges)

---

## Step 6: Monitor

```bash
JOB_NAME=$(aws sagemaker list-training-jobs --sort-by CreationTime \
  --sort-order Descending --max-results 1 \
  --query 'TrainingJobSummaries[0].TrainingJobName' --output text)

# Check status
aws sagemaker describe-training-job --training-job-name $JOB_NAME \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus}'
```

Status progression: `Pending` → `Downloading` → `Training` → `Uploading` → `Completed`

---

## Step 7: Review Results

```bash
# Download the model artifact
aws s3 cp "s3://$BUCKET/groot-data/demo/output/$JOB_NAME/output/model.tar.gz" /tmp/
tar -xzf /tmp/model.tar.gz -C /tmp/model-output/

# View eval report
cat /tmp/model-output/eval_report.json
```

**What the eval report shows:**
- `baseline_mse_overall` — error if the model always predicted the mean action (worst reasonable)
- `naive_prediction_mse_overall` — error if the model just repeated the previous timestep
- A well-trained model should achieve MSE well below the naive baseline
- `eval_action_error.png` — bar chart showing per-joint error for both baselines

---

## Step 5b (Optional): Use the SageMaker Pipeline

Instead of a one-off job, use the pipeline for a production workflow:

```bash
# Create the pipeline (one-time setup)
python training/groot/pipeline.py --create \
  --s3-bucket $BUCKET \
  --role-arn $ROLE_ARN \
  --ecr-image $ECR_URI:latest \
  --region us-east-1

# Execute (train + register model to Model Registry)
python training/groot/pipeline.py --execute --max-steps 100 --region us-east-1

# Check runs
python training/groot/pipeline.py --list-runs --region us-east-1
```

**What the pipeline adds over a raw training job:**
- Automatic model registration to `groot-models` Model Registry group
- Versioned model packages (v1, v2, v3...) with approval workflow
- Visible in SageMaker Studio UI as a DAG
- Re-executable with different parameters without editing code

---

## Step 6b (Optional): Full Training Run

Once the smoke test passes, kick off the real training:

```bash
python training/groot/launch_training.py \
  --s3-bucket $BUCKET \
  --dataset-prefix groot-data/demo \
  --role-arn $ROLE_ARN \
  --ecr-image $ECR_URI:latest \
  --max-steps 5000 \
  --instance-type ml.g5.12xlarge \
  --region us-east-1
```

This runs overnight (~11 hrs, ~$79). The model will be significantly better — loss should drop from ~0.4 to <0.05.

---

## ✅ Lab 1 Checkpoint

You've completed Lab 1 if you can answer:
- [ ] What dataset did we train on? (ALOHA sim peg insertion, 50 human teleop demos, LeRobot v2 format)
- [ ] What parts of GR00T get trained? (projector + diffusion action head; backbone is frozen)
- [ ] What instance type did we use? (ml.g5.12xlarge — 4× A10G GPUs)
- [ ] What does the eval report tell us? (MSE baselines — trained model should beat naive prediction)
- [ ] Where is the trained model? (S3 bucket + Model Registry `groot-models`)
- [ ] What's the limitation? (imitation only — fails on unseen variations → Lab 2 fixes this)

---

## What's Next: Lab 2

Lab 1 gave you a trained policy. Next, you'll set up a visual development workstation to see what's happening inside the Isaac Lab simulation — essential for debugging RL environments before running headless training at scale.

**Lab 2 deploys an Isaac Sim workstation** where you can:
- Visually watch the robot attempt tasks in simulation
- Iterate on reward functions and environment design with instant feedback
- Test training containers locally before sending to SageMaker
- Debug physics issues you can't diagnose from logs alone

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ResourceLimitExceeded` on ml.g5.12xlarge | Request GPU quota increase in Service Quotas console |
| Container pull fails | Ensure ECR URI is correct and in same region as training job |
| `HF_TOKEN` error during training | Set `HF_TOKEN` env var; accept GR00T license on HuggingFace |
| Training job stays in `Pending` for >20 min | GPU capacity shortage — try a different AZ or instance type |
| Eval report missing from model.tar.gz | Check CloudWatch logs for errors in the eval step |

---

**Previous:** [← Lab 0: Prerequisites](lab-0-prerequisites.md)
**Next:** [Lab 2: Isaac Sim Workstation →](lab-2-isaac-workstation.md)
