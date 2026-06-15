# Lab 2: RL Refinement in Simulation

**Time:** 3 hours (30 min hands-on + training runs in background)
**Cost:** ~$10-30 depending on training duration
**Goal:** Take the imitation-learned model from Lab 1 and make it robust via reinforcement learning in Isaac Lab simulation

---

## What You're Building

In Lab 1, you trained a policy by copying human demonstrations. It works ~70-80% of the time — but fails when the environment looks different from the training demos. This lab fixes that.

**The RL refinement process:**

1. **Load the Lab 1 model** into a simulated robot (Isaac Lab running on a GPU instance)
2. **Procedural domain randomization** generates thousands of scene variations:
   - Random object positions in the bin (not just where the human placed them)
   - Random lighting (dim warehouse, bright factory, harsh spotlight)
   - Random textures/colors on objects
   - Random camera noise and slight position offsets
3. **Run the policy in sim** — the robot attempts the task in each variation
4. **Reward signal tells it how to improve:**
   - +1.0 for successful pick-and-place
   - +0.3 for approaching the object correctly
   - -0.1 for dropping the object
   - -0.5 for collision with the bin walls
5. **PPO (Proximal Policy Optimization)** updates the model weights to maximize reward
6. **After thousands of episodes** → policy succeeds ~95% across all variations

**Why this works:** The model already knows "roughly what to do" from Lab 1 (imitation). RL just needs to refine the edges — handle variations, recover from errors, improve precision. Starting from a good initial policy makes RL converge in hours instead of weeks.

**Key distinction from Lab 1:**
- Lab 1: "Copy what the human did" (supervised learning from demonstrations)
- Lab 2: "Figure out how to succeed through trial-and-error" (reinforcement learning from reward)
- Combined: The industry-standard approach for production robot policies

---

## The Pipeline (V2 — what we're building toward)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Lab 1 Model │────▶│  Isaac Lab   │────▶│  Evaluate    │────▶│   Model      │
│  (GR00T      │     │  RL Refine   │     │  (sim        │     │   Registry   │
│   checkpoint │     │              │     │   rollout    │     │              │
│   from S3)   │     │  Procedural  │     │   success %) │     │  groot-models│
│              │     │  domain rand │     │              │     │  version N+1 │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘

Instance: ml.g5.12xlarge or ml.p4d.24xlarge (for faster sim)
Container: isaac-lab-training (Isaac Lab + rl_games + custom UR3 env)
Training: PPO, 4096 parallel environments on one GPU, ~4-8 hours
```

**What "procedural domain randomization" means in practice:**

Isaac Lab runs 4096 copies of the robot environment simultaneously on one GPU. In each copy, the scene is slightly different:
- Object positions sampled from a distribution (±5cm from nominal)
- Lighting intensity: 1000-5000 lux, random direction
- Object colors: sampled from RGB space
- Camera position: ±2cm and ±3° from nominal
- Physics noise: slight random forces applied to objects

The policy must succeed across ALL these variations to get high reward. This forces it to be robust.

---

## Prerequisites

- Lab 1 completed (trained GR00T model in S3 / Model Registry)
- **NVIDIA NGC API key** — required to pull the Isaac Lab base container image
  - Sign up at https://ngc.nvidia.com
  - Generate an API key at https://ngc.nvidia.com/setup/api-key
  - The base image is: `nvcr.io/nvidia/isaac-lab:4.5.0` (~15 GB)
- Docker with NVIDIA Container Toolkit (for local testing)
- GPU quota for ml.g5.12xlarge (same as Lab 1)

---

## Architecture: What's in the Isaac Lab Container

```
Isaac Lab container (~20 GB):
├── Isaac Sim runtime (physics engine, headless renderer)
├── Isaac Lab framework (RL environment interface)
├── rl_games (PPO/SAC implementation)
├── Our custom environment: training/envs/pick_and_place_ur3.py
│   ├── Observation space: joint positions (6) + gripper state (1) + object pose (7)
│   ├── Action space: joint velocity targets (6) + gripper command (1)
│   └── Reward: distance-to-object + grasp-success + place-success
├── Training config: training/configs/ppo_pick_place.yaml
│   ├── PPO hyperparameters (lr, gamma, clip_range, etc.)
│   ├── Curriculum: starts easy (1 object, centered) → hard (8 objects, random)
│   └── Domain randomization settings
└── Scripts: train.py, evaluate.py, export.py
```

---

## Step 1: Get NGC Access and Pull Base Image

```bash
# Login to NGC
docker login nvcr.io -u '$oauthtoken' -p <YOUR_NGC_API_KEY>

# Pull Isaac Lab base image (warning: ~15 GB, takes 10-20 min)
docker pull nvcr.io/nvidia/isaac-lab:4.5.0
```

---

## Step 2: Build the RL Training Container

```bash
cd containers/isaac-lab

# Build on top of Isaac Lab base
docker build --platform linux/amd64 -t isaac-lab-training .

# Tag for ECR
ECR_URI=$(aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`IsaacLabRepoUri`].OutputValue' --output text)

docker tag isaac-lab-training:latest $ECR_URI:latest

# Push to ECR (warning: ~20 GB, takes 15-30 min on first push)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_URI
docker push $ECR_URI:latest
cd ../..
```

---

## Step 3: Understand the RL Environment

Before running training, understand what the RL agent sees and does:

```python
# training/envs/pick_and_place_ur3.py (simplified)

class PickAndPlaceUR3Env:
    """
    Observation (14-dim):
      - Joint positions (6 floats) — where the arm joints are
      - Gripper state (1 float) — open/closed
      - Object position (3 floats) — where the target object is
      - Object orientation (4 floats) — quaternion

    Action (7-dim):
      - Joint velocity targets (6 floats) — how fast to move each joint
      - Gripper command (1 float) — open or close

    Reward:
      - Distance reward: -distance_to_object (encourages reaching)
      - Grasp reward: +0.3 when object is grasped
      - Place reward: +1.0 when object is placed at target
      - Penalty: -0.1 per timestep (encourages speed)
      - Collision penalty: -0.5 for hitting bin walls

    Domain randomization (applied every episode reset):
      - Object position: ±5cm from center of bin
      - Object type: random from catalog (cube, cylinder, sphere)
      - Lighting: random intensity + direction
      - Camera: ±2cm position noise
    """
```

Review the training config:
```bash
cat training/configs/ppo_pick_place.yaml
```

Key hyperparameters:
- `num_envs: 4096` — parallel environments per GPU
- `max_epochs: 2000` — training iterations
- `lr: 3e-4` — learning rate
- `gamma: 0.99` — discount factor
- `curriculum.enabled: true` — starts easy, gets harder

---

## Step 4: Launch RL Refinement Training

```bash
# Get Lab 1 model location from Model Registry
MODEL_S3=$(aws sagemaker describe-model-package \
  --model-package-name $(aws sagemaker list-model-packages \
    --model-package-group-name groot-models \
    --query 'ModelPackageSummaryList[0].ModelPackageArn' --output text) \
  --query 'InferenceSpecification.Containers[0].ModelDataUrl' --output text)

echo "Lab 1 model: $MODEL_S3"

# Launch Isaac Lab RL training as SageMaker job
python training/scripts/train.py \
  --config training/configs/ppo_pick_place.yaml \
  --pretrained-model $MODEL_S3 \
  --output-dir s3://$BUCKET/isaac-lab/output/ \
  --instance-type ml.g5.12xlarge \
  --max-epochs 500
```

**What happens during RL training:**
1. Isaac Lab launches 4096 parallel simulation environments on the GPU
2. Each environment resets with randomized scene parameters
3. The policy (initialized from Lab 1 checkpoint) takes actions in all 4096 envs simultaneously
4. Reward signals are collected across all envs
5. PPO updates the policy weights to maximize expected reward
6. Every 100 epochs, a checkpoint is saved to S3
7. After 500 epochs (~2-4 hours): the policy is significantly better

---

## Step 5: Evaluate the Refined Policy

```bash
# Run evaluation: 100 episodes with random scene variations
python training/scripts/evaluate.py \
  --checkpoint s3://$BUCKET/isaac-lab/output/checkpoint_500.pt \
  --num-episodes 100 \
  --output-dir ./eval_results/
```

**Expected output:**
```json
{
  "success_rate": 0.93,
  "avg_cycle_time_sec": 2.1,
  "episodes_evaluated": 100,
  "domain_randomization": true,
  "comparison": {
    "lab1_model_success_rate": 0.72,
    "lab2_model_success_rate": 0.93,
    "improvement": "+21%"
  }
}
```

---

## Step 6: Export to TensorRT (for edge deployment)

```bash
python training/scripts/export.py \
  --checkpoint s3://$BUCKET/isaac-lab/output/checkpoint_500.pt \
  --output ./model_exported/ \
  --format tensorrt \
  --precision fp16
```

This produces `model.trt` — ready for Lab 3 (edge deployment).

---

## ✅ Lab 2 Checkpoint

You've completed Lab 2 if you can answer:
- [ ] What does RL refinement do that imitation alone can't? (handles unseen variations through trial-and-error)
- [ ] What is domain randomization? (randomized scene parameters so policy must be robust to succeed)
- [ ] How many parallel environments run simultaneously? (4096 on one GPU)
- [ ] What reward signal drives improvement? (success/failure at the manipulation task)
- [ ] How much did success rate improve? (Lab 1: ~70-80% → Lab 2: ~93-95%)
- [ ] Where is the refined model? (S3 checkpoint + optionally TensorRT export)

---

## Current Status / Known Limitations

> **⚠️ This lab requires NGC access for the Isaac Lab container base image.**
> If you don't have NGC credentials, you cannot build the container yet.
> The training scripts and environment code can be reviewed without NGC.
> Task for contributors: explore building Isaac Lab from source without NGC image.

> **⚠️ Cosmos scene generation is NOT included in this lab.**
> We use Isaac Lab's built-in procedural domain randomization (random positions, lighting, textures).
> Cosmos would add photorealistic enhancement (scratched metal, dust, realistic shadows) —
> this is a V3 feature requiring NVIDIA NIM API access.
> For most tasks, procedural randomization alone achieves good sim-to-real transfer.

---

## How This Connects to the Full Pipeline

```
Lab 1 (imitation)──────────▶ Lab 2 (RL refinement) ──────────▶ Lab 3 (edge)
                                     │
                                     ├── Domain randomization (built-in)
                                     │   Random positions, lighting, textures
                                     │   4096 parallel variations per step
                                     │
                                     └── [Future] Cosmos enhancement (V3)
                                         Photorealistic scene generation
                                         Requires NVIDIA NIM API
```

## When to Use What: Domain Randomization vs. Cosmos

This is a common question — when do you need Cosmos vs. Isaac Lab's built-in randomization?

| Approach | What it does | When to use | Cost |
|----------|-------------|-------------|------|
| **Isaac Lab procedural randomization** | Randomizes object positions, lighting intensity/direction, object colors, camera noise | **Always — this is your default.** Handles 80% of sim-to-real transfer for most manipulation tasks. | Free (built into Isaac Lab) |
| **Cosmos Transfer** | Takes your sim-rendered scene and makes it photorealistic (adds scratches, dust, realistic shadows, material imperfections) | When procedural randomization alone isn't enough — typically for tasks where **visual appearance** matters (e.g., bin picking by color, defect detection). | NIM API cost per frame |
| **Cosmos Generate** | Creates entirely new environments from text/image prompts | When you need **environment diversity** beyond what procedural generation offers (e.g., "generate 100 different warehouse layouts"). | NIM API cost per scene |

**Rule of thumb:**
1. Start with Isaac Lab procedural randomization (free, fast, good enough for most tasks)
2. If your policy fails on real hardware due to **visual domain gap** (sim looks too different from real) → add Cosmos Transfer
3. If your policy fails because it only works in **one environment layout** → add Cosmos Generate

**Most teams never need Cosmos.** Procedural randomization + a well-tuned reward function gets you to 90%+ success on real hardware for standard manipulation tasks (pick-and-place, insertion, assembly). Cosmos becomes relevant for:
- High-precision visual tasks (reading labels, color sorting)
- Environments with complex, varied backgrounds (warehouses with many objects)
- When you have very limited real-world data to validate against

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker login nvcr.io` fails | Verify NGC API key at https://ngc.nvidia.com/setup/api-key |
| Container build OOM | Isaac Lab base is ~15 GB. Ensure 30+ GB free disk space |
| Training doesn't converge | Check reward function. Try reducing domain randomization range initially |
| `CUDA out of memory` | Reduce `num_envs` in config (4096 → 2048 → 1024) |
| Policy success rate stays at 0% | Pretrained model path wrong — verify Lab 1 checkpoint loaded correctly |

---

**Previous:** [← Lab 1: Train from Demonstrations](lab-1-train-groot.md)
**Next:** [Lab 3: Edge Deployment →](lab-3-edge-deployment.md)
