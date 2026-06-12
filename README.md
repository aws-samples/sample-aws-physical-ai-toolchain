# AWS Physical AI Toolchain

Train robot manipulation policies on AWS and deploy to edge hardware. From cloud to robot in one `cdk deploy`.

## Two Paths, One Platform

| | Path A: Simple | Path B: Full |
|---|---|---|
| **What** | GR00T fine-tuning from existing data | Isaac Lab RL in simulation + OSMO orchestration |
| **Compute** | SageMaker Training Jobs | EKS + GPU autoscaling |
| **Deploy time** | 5 minutes | 20 minutes |
| **When to use** | Have robot data, want a trained model fast | Need simulation, synthetic data, multi-stage pipelines |
| **Command** | `cdk deploy --context mode=simple` | `cdk deploy --context mode=full` |

Both paths output the same thing: a TensorRT model deployable to Jetson/GPU PC via Greengrass.

## Quick Start (Path A — recommended first)

```bash
# 1. Deploy infrastructure
cd cdk && npm install
npx cdk deploy --context mode=simple --require-approval never

# 2. Upload your dataset (LeRobot format) or use the demo
aws s3 sync ./data/demo-dataset/ s3://<DATASETS_BUCKET>/groot-data/my-task/dataset/

# 3. Build training container
aws codebuild start-build --project-name physical-ai-groot-training-build

# 4. Launch training (~11 hours for 5000 steps)
python training/groot/launch_training.py \
  --s3-bucket <DATASETS_BUCKET> \
  --dataset-prefix groot-data/my-task \
  --role-arn <SAGEMAKER_ROLE_ARN> \
  --ecr-image <GROOT_TRAINING_ECR>:latest \
  --max-steps 5000

# 5. Get results
aws s3 cp s3://<MODELS_BUCKET>/training-output/.../eval_video.mp4 ./
```

## Quick Start (Path B — simulation + OSMO)

```bash
# 1. Deploy full infrastructure (adds EKS, OSMO, GPU nodes)
npx cdk deploy --context mode=full --require-approval never

# 2. Configure kubectl
aws eks update-kubeconfig --name physical-ai-dev

# 3. Submit training pipeline via OSMO
osmo workflow submit -f workflows/pick-and-place.yaml

# 4. Monitor
osmo workflow status <workflow-id>
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Foundation (always)              │
│  S3 · ECR · IAM · CodeBuild             │
└────────────────┬────────────────────────┘
        ┌────────┴────────┐
   Path A (SageMaker)  Path B (EKS+OSMO)
        └────────┬────────┘
          model.trt in S3
                │
         Edge (Greengrass)
          → Jetson / GPU PC
```

## Project Structure

```
├── PLAN.md                    # Project plan, build order, open questions
├── cdk/                       # Infrastructure as Code (TypeScript)
│   ├── lib/
│   │   ├── foundation-stack   # S3, ECR, IAM, CodeBuild (both paths)
│   │   ├── network-stack      # VPC (Path B only)
│   │   ├── eks-cluster-stack  # EKS + GPU nodes (Path B only)
│   │   ├── osmo-stack         # OSMO + RDS + Redis (Path B only)
│   │   └── edge-stack         # Greengrass (optional, both paths)
│   └── config/                # Dev/prod environment configs
├── containers/
│   ├── groot-training/        # GR00T fine-tuning (Path A)
│   ├── isaac-sim/             # Scene generation (Path B)
│   ├── isaac-lab/             # RL training (Path B)
│   └── inference/             # TensorRT + ROS2 (edge, both paths)
├── training/
│   ├── groot/                 # GR00T scripts (Path A)
│   └── isaac-lab/             # Isaac Lab RL (Path B: envs, configs, scripts)
├── workflows/                 # OSMO pipeline definitions (Path B)
├── edge/                      # Greengrass components + ROS2 inference node
└── workshop/                  # Hands-on lab guides (immersion day)
```

## Workshop Labs

This repo doubles as a hands-on workshop:

| Lab | Duration | What you learn |
|-----|----------|---------------|
| [Lab 0: Prerequisites](workshop/lab-0-prerequisites.md) | 30 min | Environment setup |
| [Lab 1: Train GR00T](workshop/lab-1-train-groot.md) | 2 hrs | Path A end-to-end |
| [Lab 2: Isaac Lab + OSMO](workshop/lab-2-isaac-lab-osmo.md) | 3 hrs | Path B end-to-end |
| [Lab 3: Edge Deployment](workshop/lab-3-edge-deployment.md) | 1.5 hrs | TensorRT + Greengrass |
| [Lab 4: Extend](workshop/lab-4-extend.md) | 1 hr | Customize environments |

## Cost Estimate

| Mode | Running (training active) | Idle | Teardown |
|------|--------------------------|------|----------|
| Simple (Path A) | ~$7/hr (ml.g5.12xlarge) | $0 (no persistent infra) | `cdk destroy` |
| Full (Path B) | ~$10-20/hr (EKS + GPU nodes) | ~$300/mo (EKS + RDS + Redis) | `cdk destroy` |

Path A has zero idle cost — SageMaker only bills during training. Path B has persistent EKS infrastructure but GPU nodes scale to zero when not training.

## Contributing

See [PLAN.md](PLAN.md) for build order and open questions. Pick the next `🔲` task.

## License

Apache 2.0
