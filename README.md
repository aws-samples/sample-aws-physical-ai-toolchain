# AWS Physical AI Toolchain

Train robot manipulation policies on AWS and deploy to edge hardware.

## What This Does

Implements the standard Physical AI training pipeline:

1. **Imitation Learning** — Fine-tune GR00T on teleoperation demos → working policy in hours
2. **RL Refinement** — Improve the policy in Isaac Lab simulation → production-grade robustness
3. **Domain Randomization** — Cosmos generates scene variations → sim-to-real transfer
4. **Edge Deployment** — Deploy to Jetson/GPU hardware via Greengrass

You can stop at any stage. Stage 1 alone gives you a usable policy.

## Quick Start (15 min to first training job)

```bash
# Prerequisites: AWS CLI configured, CDK installed, Docker running

# 1. Deploy infrastructure
cd cdk && npm install
npx cdk deploy --context mode=simple

# 2. Download demo dataset
python training/groot/download_demo_dataset.py --output ./data/demo-dataset

# 3. Build + push training container
cd containers/groot-training
docker build --platform linux/amd64 -t groot-training .
# Tag and push to ECR (URI from CDK outputs)

# 4. Run the pipeline (100 steps = smoke test, 5000 = full training)
./run-path-a.sh --max-steps=100
```

Or use the SageMaker Pipeline for a production workflow:
```bash
python training/groot/pipeline.py --create \
  --s3-bucket <DATASETS_BUCKET> \
  --role-arn <SAGEMAKER_ROLE_ARN> \
  --ecr-image <ECR_URI>:latest

python training/groot/pipeline.py --execute --max-steps 5000
```

## What's Working Today

- ✅ Foundation stack (S3, ECR, IAM) deployed
- ✅ GR00T training container built and pushed to ECR
- ✅ SageMaker training jobs completing successfully
- ✅ SageMaker Pipeline (train → register to Model Registry)
- ✅ Eval report (action prediction error on held-out data)
- 🔲 Isaac Lab RL refinement (Stage 2 — containers exist, untested on SM)
- 🔲 Cosmos scene generation (Stage 3 — script exists, needs API access)
- 🔲 Edge deployment (CDK stack exists, untested)

## Architecture

```
Foundation Stack (S3, ECR, IAM)
         │
         ▼
SageMaker Pipeline
  ├── Stage 1: GR00T fine-tune (imitation)     ← working
  ├── Stage 2: Cosmos scene generation          ← next
  ├── Stage 3: Isaac Lab RL refinement          ← next
  ├── Evaluate (sim rollout success rate)
  └── Register to Model Registry               ← working
         │
         ▼
Edge (Greengrass → Jetson)                      ← later
```

All stages run on SageMaker (Training Jobs + Processing Jobs). No EKS needed unless you need 100+ parallel sim environments.

## Project Structure

```
├── PLAN.md                        # Detailed plan, decisions, build order
├── run-path-a.sh                  # One-command quick start
├── cdk/                           # Infrastructure (TypeScript CDK)
│   ├── lib/foundation-stack.ts    # S3, ECR, IAM (always deployed)
│   ├── lib/eks-cluster-stack.ts   # EKS (scale path, optional)
│   └── lib/osmo-stack.ts          # OSMO orchestrator (scale path)
├── containers/
│   ├── groot-training/            # Stage 1: fine-tuning container
│   ├── isaac-sim/                 # Stage 2: Cosmos scene gen
│   ├── isaac-lab/                 # Stage 3: RL refinement
│   └── inference/                 # Edge: TensorRT + ROS2
├── training/
│   ├── groot/                     # Launch scripts, pipeline, dataset tools
│   ├── scripts/                   # Isaac Lab training/eval/export
│   └── envs/                      # RL environments (pick-and-place)
├── edge/                          # Greengrass + ROS2 inference node
└── workshop/                      # Hands-on lab guides
```

## Cost

| What | Cost | Notes |
|------|------|-------|
| Smoke test (100 steps) | ~$2 | 15 min on ml.g5.12xlarge |
| Full training (5000 steps) | ~$79 | 11 hrs |
| Idle (no training running) | $0 | SageMaker has no idle cost |
| Foundation stack | ~$1/mo | S3 storage only |

## Contributing

Read [PLAN.md](PLAN.md) for:
- Build order with task status
- Architecture decisions and rationale
- Open questions that need input
- How to pick up the next task

Getting a robotics practitioner to review the RL refinement architecture would be especially valuable.

## License

Apache 2.0
