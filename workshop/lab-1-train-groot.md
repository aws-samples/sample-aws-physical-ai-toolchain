# Lab 1: Train a Robot Policy from Demonstrations

**Goal:** Fine-tune NVIDIA GR00T on teleoperation data → get a model that predicts robot joint actions from camera images
**Time:** 2 hours (30 min hands-on + training runs in background)
**Cost:** ~$2 for smoke test, ~$79 for full training

---

## What You're Building

> **TODO:** Add images or video of teleoperation showing operator using Xbox controller with UR3 arm

A robot manipulation policy trained via **imitation learning**. Here's what that means:

1. A human teleoperated a robot arm to perform a task (pick and place) — 27 demonstrations were recorded. Teleoperation is typically done using a game controller (we used Xbox), a VR headset (Apple Vision Pro is increasingly popular), or a 3D Space Mouse. The operator controls the robot's end-effector while camera and joint data are recorded automatically.
2. Each demo captured: wrist camera images + joint positions + Cartesian velocity commands at every timestep
3. You fine-tune GR00T (a 3B parameter vision-language-action model) to learn the pattern: "given this camera image → predict these motor commands"
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

> TODO: Insert pipeline architecture diagram here (showing data flow from S3 → SageMaker → Evaluation → Model Registry)

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

- Foundation stack deployed (`cdk deploy --context mode=simple` — see Lab 0). This
  already triggered the CodeBuild job that builds the training container in the cloud.
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

## Step 2: Prepare the Training Data

The training data is 27 episodes of UR3 pick-and-place, recorded via Xbox controller teleoperation. The raw data is in Zarr format (how our recording tools capture it) and needs to be converted to LeRobot v2 format (what GR00T reads).

```bash
# Convert Zarr episodes → LeRobot v2 format
python training/groot/convert_zarr_to_lerobot.py \
  --episodes-dir training/data/ur3_episodes/episodes \
  --output-dir training/data/ur3_lerobot_dataset
```

**What the conversion does:**
- Reads each Zarr episode (wrist camera frames + joint states + velocity commands)
- Aligns camera frames to telemetry timestamps (camera runs at 5Hz, telemetry at 10Hz)
- Extracts action vectors from URScript `speedl` commands (6D Cartesian velocity + gripper)
- Encodes camera frames to MP4 video
- Writes LeRobot v2 parquet + metadata files

**Output structure:**
```
training/data/ur3_lerobot_dataset/
├── data/chunk-000/          # Parquet files: 7D actions (6 velocity + gripper), 7D states (6 joints + gripper)
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ... (27 episodes)
├── meta/
│   ├── info.json            # Dataset metadata (robot_type: ur3, fps: 5)
│   ├── modality.json        # GR00T embodiment config (arm start/end, gripper start/end)
│   ├── episodes.jsonl       # Episode lengths and task descriptions
│   └── tasks.jsonl          # Task vocabulary
└── videos/chunk-000/
    └── observation.images.wrist/  # Wrist camera MP4s (one per episode)
```

**Bringing your own data:** If you have Zarr episodes from a different robot, this same script works — just point `--episodes-dir` at your recordings. The Zarr schema expects `observations/joints`, `observations/gripper_position`, `images/wrist`, and `commands.json`. See `convert_zarr_to_lerobot.py` for the full format specification.

---

## Step 3: Upload Dataset to S3

```bash
aws s3 sync training/data/ur3_lerobot_dataset/ "s3://$BUCKET/groot-data/ur3/dataset/"
```

---

## Step 4: Get the Training Container

**You don't build this locally.** When you deployed the Foundation stack (Lab 0),
CDK kicked off an AWS CodeBuild job that builds the `groot-training` image in the
cloud and pushes it to ECR. This means **no multi-GB `docker build` on your
laptop** — and it works even on Apple Silicon, where the image can't be built
locally at all.

Check that the image is ready:

```bash
# Was the build triggered? (it runs automatically on cdk deploy)
aws codebuild list-builds-for-project --project-name physical-ai-groot-training-build \
  --query 'ids[0]' --output text

# Is the image in ECR yet? (build takes ~10 min)
aws ecr describe-images --repository-name physical-ai/groot-training \
  --query 'imageDetails[?contains(imageTags, `latest`)].imagePushedAt' --output text
```

If the image isn't there yet, watch the build in the [CodeBuild console](https://console.aws.amazon.com/codesuite/codebuild/projects)
(the `BuildConsole` Foundation stack output links straight to it). Re-run a build
any time with:

```bash
aws codebuild start-build --project-name physical-ai-groot-training-build
```

**What's in the container:**
- CUDA 12.4 + PyTorch 2.5 (GPU compute)
- HuggingFace transformers + diffusers (model loading)
- LeRobot (dataset loading)
- pandas + matplotlib (eval report generation)
- `train_entrypoint.py` — the script SageMaker runs

<details>
<summary><strong>Optional:</strong> build locally instead (only if you're iterating on the Dockerfile)</summary>

The `groot-training` image builds from a public CUDA base, so you *can* build it
on an x86 machine with Docker if you want a faster edit/rebuild loop:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR_URI
cd containers/groot-training
docker build --platform linux/amd64 -t groot-training .
docker tag groot-training:latest $ECR_URI:latest
docker push $ECR_URI:latest
cd ../..
```

For everyone else, the CodeBuild image above is all you need.
</details>

---

## Step 5: Launch Training (Smoke Test)

Start with 100 steps to verify everything works (~15 min, ~$2):

```bash
# One command to run the full pipeline (train + register to Model Registry)
python training/groot/pipeline.py --execute \
  --max-steps 100 \
  --dataset-prefix groot-data/ur3
```

That's it. The pipeline handles:
1. Provisioning an ml.g5.12xlarge instance (4× A10G GPUs)
2. Pulling your container from ECR
3. Downloading the UR3 dataset from S3
4. Downloading GR00T N1.7-3B base model from HuggingFace
5. Fine-tuning projector + action head for 100 steps
6. Generating an eval report (action prediction error)
7. Registering the trained model to the `groot-models` Model Registry
8. Terminating the instance (no idle charges)

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
python training/groot/pipeline.py --execute --max-steps 5000 --dataset-prefix groot-data/ur3
```

This runs overnight (~11 hrs, ~$79). The model will be significantly better — loss should drop from ~0.4 to <0.05.

---

## ✅ Lab 1 Checkpoint

You've completed Lab 1 if you can answer:
- [ ] What dataset did we train on? (27 real UR3 pick-and-place episodes, recorded via Xbox controller teleop)
- [ ] What format does GR00T expect? (LeRobot v2 — parquet + MP4, with modality.json for embodiment config)
- [ ] What parts of GR00T get trained? (projector + diffusion action head; backbone is frozen)
- [ ] What instance type did we use? (ml.g5.12xlarge — 4× A10G GPUs)
- [ ] Where is the trained model? (S3 bucket + Model Registry `groot-models`)
- [ ] What's the limitation? (imitation only — fails on unseen variations → Lab 4 fixes this with RL)

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
