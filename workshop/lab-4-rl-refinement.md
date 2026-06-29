# Lab 4: RL Policy Training in Simulation

**Goal:** Train a robust pick-and-place policy via reinforcement learning in Isaac Lab simulation with domain randomization
**Time:** 3 hours (30 min hands-on + training runs in background)
**Cost:** ~$3 for smoke test (50 iterations), ~$28 for full training (2000 iterations). Full cost breakdown is in the [main README](../README.md#cost-summary).

---

## 🏃 Quick Runbook (do this in order)

> Lab 4 is **standalone RL** — it does **not** require Lab 1 or any GR00T model. Follow top-to-bottom; each step says what to run and how you know it worked.

| # | Action | Command (summary) | ✅ Success check |
|---|--------|-------------------|-----------------|
| 0 | Confirm prerequisites | `pai doctor` | all checks pass |
| 1 | Launch the RL smoke test | `pai rl launch --max-iterations 50 --instance-type ml.g5.xlarge` | prints `Launched. Monitor: …` with a job name |
| 2 | Watch the SageMaker job | `pai rl status <name>` | status `InProgress` → `Completed` |
| 3 | (Optional) full training | `pai rl launch --task Isaac-Velocity-Flat-Anymal-D-v0 --max-iterations 1500 --instance-type ml.g5.12xlarge` | job launches; logs `Mean reward` climbing |
| 3b | (Optional) scale out across nodes | SageMaker: add `--instance-count 2`. Batch: `pai rl launch --engine batch --num-nodes 2` | job launches across N nodes (multi-node NCCL **unvalidated**) |
| 4 | Evaluate the policy (closed-loop) | runs on the **Lab 2 workstation** — `python training/scripts/eval_policy_server.py` + `eval_sim_client.py` | prints `success_rate` JSON |
| 5 | Export to TensorRT | `pai export --checkpoint … --output-onnx … --output-trt … --target-device jetson-orin` | writes `model.trt` |

**Before you start, confirm:**
- [ ] AWS credentials active for the **test account** (`pai doctor` checks this)
- [ ] `config.json` `aws.region` matches where your Foundation stack / ECR lives
- [ ] Foundation stack deployed → the `isaac-lab` training image is in your ECR (`pai doctor` checks this)
- [ ] GPU quota for `ml.g5.xlarge` (smoke test) or `ml.g5.12xlarge` (full run) — see Lab 0 quota preflight

> 💸 **Cost reminder:** the smoke test (Step 1) is ~$3; full training (Step 3) is ~$28. SageMaker tears the instance down when the job ends — no manual stop needed (unlike Lab 2's EC2 box).

> ⚠️ **Honest status:** the validated path is Isaac Lab's built-in `Isaac-Velocity-Flat-Anymal-D-v0` task. The custom UR3 env (`PickAndPlaceUR3-v0`) is registered but **not yet wired into the container**, so use the Anymal task for a green run today. Details in Step 1.

---

## What You're Building

You'll train a robot policy from scratch with reinforcement learning (RL) in Isaac Lab.
The pipeline: define the task + reward, Isaac Lab runs **4096 randomized copies** of the
env in parallel on one GPU, **PPO** updates the policy to maximize reward, checkpoint to S3.

**Procedural domain randomization** is what makes the policy robust — every env copy varies
the scene (object position ~5cm, lighting 1000-5000 lux, object color, camera noise), so the
policy only scores well if it succeeds across *all* variations, which is what transfers to
the real world. Reward shaping is the usual mix: +1.0 place, +0.3 approach, -0.1/step, -0.5
collision.

**RL vs. imitation learning (Lab 1):** two independent ways to get a policy — IL copies human
demos (best with good demos, weak sim); RL learns from reward (best with a strong sim + clear
success signal). Both feed Lab 5 edge deployment. You can run Lab 4 on its own — it doesn't read
a GR00T model or any Lab 1 output.

> **Prerequisites** are the runbook's "Before you start" checklist above (Foundation
> deployed → `isaac-lab` image in ECR, GPU quota). See [Lab 0](lab-0-prerequisites.md) to set them up.

---

## Step 1: Launch RL Training

All RL training is launched from your laptop with **`pai rl launch`** — it builds the
SageMaker job (resolving your account's ECR image, role, and output bucket) and
submits it.

**First, preview the job (free — makes no AWS calls):**

```bash
pai rl launch --dry-run
```

This prints the exact `create_training_job` request it would submit (image, role,
output path, hyperparameters) so you can sanity-check it before spending anything.

**Then launch the smoke test on the validated built-in task:**

```bash
# Smoke test: 50 iterations (~5-10 min, ~$3) on the validated Anymal task
pai rl launch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 \
  --max-iterations 50 \
  --instance-type ml.g5.xlarge
```

<details>
<summary>Under the hood (raw commands)</summary>

```bash
python training/scripts/launch_rl.py --dry-run

python training/scripts/launch_rl.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 \
  --max-iterations 50 \
  --instance-type ml.g5.xlarge
```

</details>

**What happens:**
1. Launches Isaac Lab RL training on SageMaker (ml.g5.xlarge, 4096 parallel envs)
2. Trains the policy with PPO from scratch in headless Isaac Sim
3. Policy learns through trial-and-error guided by reward signals
4. Saves a checkpoint (`model_*.pt`) + training metadata to S3

50 iterations is a *smoke test* — it proves the pipeline runs end-to-end. The
resulting policy will stumble, not perform well; a usable policy needs ~1000+
iterations (see Step 3).

> **Why Anymal and not the UR3 arm this project is about?** The UR3 pick-and-place env is the
> reference design's end goal, but it can't train today. Three concrete blockers: (1) the
> container's training entrypoint runs only Isaac Lab's built-in `train.py` and never imports
> `training.envs`, so the `PickAndPlaceUR3-v0` registration never fires *inside the job* — the
> task id won't resolve (`pai rl launch` warns if you try); (2) the env points its USD assets at
> a local `omniverse://localhost` Nucleus server that doesn't exist on a headless box — it needs
> a cloud/S3 asset root; (3) it's never run on a GPU, so more issues likely lurk. It's a moderate
> effort (wire the import, repoint assets, then iterate on a GPU), tracked as ROADMAP Feature 2 —
> not a code rewrite, but it needs the GPU iteration loop the Lab 2 workstation exists for. Until
> a real run confirms it, Anymal is the validated, load-tested path.

> Need to rebuild the Isaac Lab container after changing its Dockerfile? Trigger
> the cloud build with `pai deploy foundation` (redeploys and rebuilds all images)
> or directly with `aws codebuild start-build --project-name physical-ai-isaac-lab-build`.

## Step 2: Understand the RL Environment

**Where do tasks like `Isaac-Velocity-Flat-Anymal-D-v0` come from?** Two places:
- **Built-in tasks** ship *inside* the `isaac-lab` container as part of Isaac Lab itself
  (`/workspace/isaaclab/source/...`, registered via `omni.isaac.lab_tasks`). Anymal is one of
  these — upstream NVIDIA code you can read but don't edit here.
- **Custom tasks** live in this repo under `training/envs/`, registered in
  `training/envs/__init__.py`. That's the file you copy to add your own task. See the NVIDIA
  Isaac Lab docs linked in [Lab 2](lab-2-isaac-workstation.md#step-4-verify-the-workstation-renders-visual-smoke-test)
  for writing a custom env.

This repo's reference task is the UR3 arm. `training/envs/pick_and_place_ur3.py` defines it:

- **Observation (14-dim):** 6 joint positions + gripper state + object position (3) + object orientation (quaternion, 4).
- **Action (7-dim):** 6 joint-velocity targets + gripper open/close.
- **Reward:** `-distance` (reach) + 0.3 grasp + 1.0 place, with -0.1/step and -0.5 collision penalties.
- **Domain randomization** (per reset): object position/type, lighting, camera noise.

Hyperparameters live in `training/configs/ppo_pick_place.yaml` (`num_envs: 4096`,
`max_epochs: 2000`, `lr: 3e-4`, `gamma: 0.99`, `curriculum.enabled: true`).

---

## Step 3: Launch Full RL Training (Optional)

Same command as the smoke test, scaled up — more iterations, a bigger instance, a longer
runtime cap. Add `--dry-run` first to preview:

```bash
pai rl launch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 --max-iterations 1500 \
  --instance-type ml.g5.12xlarge --runtime-min 240
```

A checkpoint is saved to S3 every 100 epochs; after ~1500 iterations (~2-4 hours) the
policy reaches high success rates. Checkpoints land as `model_*.pt` in your output bucket.

> **`train.py` is in-container only:** it imports `from isaacsim import SimulationApp` at
> module top and crashes off-GPU — always launch via `pai rl launch`, never run it on your
> laptop. Warm-start (`--pretrained-model`) is a `train.py` hyperparameter for resuming RL
> from a same-architecture checkpoint — *not* for loading a GR00T/VLA model (incompatible shapes).

---

## Step 3b: Scale Out Across Multiple Nodes (Optional)

A single GPU instance is enough for the workshop tasks. When you outgrow one box —
bigger models, more parallel envs, faster wall-clock — RL training scales across
**multiple nodes** two ways. Both reuse the **same `physical-ai/isaac-lab` container**;
the only difference is who provisions the fleet and wires the NCCL topology.

> ⚠️ **Honest status:** both paths are **fully implemented but UNVALIDATED on hardware** — not
> placeholders. The launchers and stacks really do set up the topology (parse the cluster config,
> launch `torchrun --nnodes/--node_rank/--rdzv_endpoint`), so a job *will* launch across N nodes.
> What's unverified is **NCCL convergence across nodes on G-family GPUs.** The single-node path
> (Steps 1–3) is the proven one; don't treat a green multi-node launch as a validated distributed
> run until you've watched reward actually climb. If you try it and it converges, this label can
> flip — that's exactly the missing confirmation.

### Option A — SageMaker multi-instance (quickest)

The container's SageMaker entrypoint already parses the cluster's `resourceconfig.json`
and launches `torchrun` across however many instances SageMaker provisions — the **only**
thing you change is `--instance-count`:

```bash
# Add --dry-run to preview (no AWS calls); drop it to launch.
pai rl launch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 --max-iterations 100 \
  --instance-type ml.g5.12xlarge --instance-count 2
```

SageMaker handles inter-node networking and tears the fleet down when the job ends.

### Option B — AWS Batch Multi-Node Parallel (the reference architecture)

Batch gives you direct control over the EC2 fleet (g6.12xlarge, 4× L4 each), a shared
**EFS** filesystem for checkpoints, and a self-managed NCCL security group. It's an
**opt-in CDK stack**, not deployed by default.

```bash
# 1. Deploy the opt-in Batch stack (isaac-lab image must already be in ECR).
pai deploy batch
# Reads `batch` settings from config.json and prints a ready-to-run LaunchCommand
# with the exact queue/job-definition names.

# 2. Submit (add --dry-run to preview). Queue/def default to
#    physical-ai-dev-rl-queue / -rl-mnp; override with --job-queue / --job-definition.
pai rl launch --engine batch \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 --max-iterations 100 --num-nodes 2

# 3. Monitor
pai rl status --engine batch <job-id>
```

> ⚠️ **GPU quota:** g6.12xlarge needs *G-family On-Demand* vCPU quota you may not have by
> default — request it in Service Quotas first, or the compute env sits at 0 vCPUs and jobs
> stay `RUNNABLE` forever. Also needs a **default VPC** (same as the Lab 2 workstation).

Checkpoints persist to EFS at `/efs/models/<job-id>` — mount it to a workstation (or SSM
onto a compute node) to inspect them.

| | SageMaker multi-instance | AWS Batch MNP |
|---|---|---|
| Setup | none (just `--instance-count N`) | deploy opt-in `PhysicalAi-dev-Batch` stack |
| Provisioning | managed by SageMaker | managed EC2 compute env you own |
| Shared storage | S3 only | EFS (`/efs`) + S3 |
| Teardown | automatic on job end | job auto-terminates; stack persists until `cdk destroy` |
| Best for | quick scale-out, least moving parts | full control, EC2-priced fleets, prototyping NCCL |

---

## Step 4: Evaluate the Trained Policy (Closed-Loop)

> **⚠️ UNVALIDATED until run on the Lab 2 GPU workstation (g6e.4xlarge L40S).**
> This step drives Isaac Lab simulation with actions from a TorchScript policy server over ZMQ. You must first scriptify the checkpoint (Step 4a), then run the policy server + sim client in two terminals (Step 4b).

**Where this runs:** inside the **`isaac-lab` container** on the Lab 2 workstation — the same single environment used for visual training (Lab 2 Step 4). The sim client imports `omni.isaac.lab.*` (in the container), and the `pai` CLI is not installed on the workstation (it's the laptop-side control plane). Launch the container with `~/run-isaac-lab.sh`; your repo working tree is mounted at `/workspace/toolchain`. Open a second shell into the same container with `sudo docker exec -it isaac-lab bash`.

### Step 4a: Fetch the checkpoint, then scriptify it

**First, get the checkpoint onto the workstation.** The Lab 4 SageMaker job wrote its output
to `s3://<bucket>/isaac-lab/output/` as a `model.tar.gz` (containing `logs/.../model_*.pt`).
Nothing pulls it down automatically — copy and extract it into the mounted repo on the
workstation:

```bash
# In the container shell, in /workspace/toolchain:
BUCKET=$(aws sts get-caller-identity --query Account --output text | xargs -I{} echo physical-ai-dev-datasets-{})
aws s3 cp "s3://$BUCKET/isaac-lab/output/<job-name>/output/model.tar.gz" .
tar -xzf model.tar.gz          # extracts logs/.../model_<N>.pt
```

**Then scriptify it.** The evaluator (and `export.py`) need a **TorchScript** model. Raw rsl_rl
checkpoints from `train.py` are state dicts, not scripted modules — `torch.jit.load` rejects
them. `scriptify_policy.py` rebuilds the policy MLP, loads the weights, and `jit.script`s it
(this round-trip is the one piece covered by an automated test):

```bash
# Arch must match training/configs/ppo_pick_place.yaml.
python training/scripts/scriptify_policy.py \
  --checkpoint ./logs/rsl_rl/<run>/model_500.pt \
  --output ./model_scripted/model_scripted.pt \
  --obs-dim 12308 --action-dim 7
```

This writes `model_scripted.pt` — feed it to the policy server below (and to Step 5's export).

> **Why scriptify now, when Lab 2 didn't need it?** Lab 2 only *trains and renders* — it produces
> a checkpoint but never loads one back. Eval (and export) *consume* the policy, and both
> `eval_policy_server.py` and `export.py` call `torch.jit.load`, which requires TorchScript and
> rejects the raw rsl_rl checkpoint. Scriptify bridges that one-time gap.

### Step 4b: Run closed-loop eval (two shells in the container)

From your laptop, `pai workstation start` then `pai workstation ip`; connect via DCV,
launch the container (`~/run-isaac-lab.sh`), and open a second shell with
`sudo docker exec -it isaac-lab bash`. Both shells `cd /workspace/toolchain`.

**Shell 1 — policy server** (binds `tcp://127.0.0.1:5555`, localhost only — ZMQ has no auth):
```bash
python training/scripts/eval_policy_server.py \
  --checkpoint ./model_scripted/model_scripted.pt --device cuda
```

**Shell 2 — sim client** (boots Isaac Sim, so run via `isaaclab.sh`, not bare python):
```bash
/workspace/isaaclab/isaaclab.sh -p training/scripts/eval_sim_client.py \
  --task PickAndPlaceUR3-v0 --endpoint tcp://127.0.0.1:5555 \
  --eval-rounds 100 --output-dir ./eval_results
```

The client runs 100 episodes, querying the server for each action, and writes JSON
metrics (`success_rate_pct`, `num_episodes`, `avg_reward`, `avg_cycle_time_sec`,
`failure_modes{timeout,drop,collision}`) to `./eval_results`.

**Open-loop alternative** (policy in-process, no server/ZMQ — also boots Isaac Sim):
```bash
/workspace/isaaclab/isaaclab.sh -p training/scripts/evaluate.py \
  --checkpoint ./model_scripted/model_scripted.pt --num-episodes 100 --output-dir ./eval_results/
```

---

## Step 5: Export to TensorRT (for edge deployment)

`export.py` consumes the **TorchScript** `.pt` from Step 4a (`jit.load` → ONNX →
TensorRT), so point `--checkpoint` at `model_scripted.pt`, not the raw rsl_rl checkpoint:

```bash
pai export \
  --checkpoint ./model_scripted/model_scripted.pt \
  --output-onnx ./model_exported/model.onnx \
  --output-trt ./model_exported/model.trt \
  --target-device jetson-orin --fp16 --benchmark
```

`--target-device` ∈ `{jetson-orin, jetson-nano, gpu-pc}`; `--fp16` on by default;
`--benchmark` times inference after compile. Produces `model.trt` — ready for Lab 5.

**Where this runs:** export is lightweight — it compiles a single batch-1 MLP, taking seconds,
not a GPU training job — so running it on the Lab 2 workstation (where you already have the
scriptified checkpoint) is fine. It needs the `tensorrt` Python package; the ONNX intermediate
is portable if you'd rather compile the final engine on the Jetson itself for an exact match.

**Push the engine to S3** so Lab 5 (Greengrass → Jetson) can pull it — the local
`./model_exported/` path lives only on this instance:

```bash
aws s3 cp ./model_exported/model.trt "s3://$BUCKET/isaac-lab/exported/model.trt"
```

> The local output is fine for the box that builds it, but the pipeline expects the engine in
> S3 (or an artifact store). A production setup would version it there rather than leave it on a
> single instance's disk.

---

## ✅ Lab 4 Checkpoint

You've **run** Lab 4 if:
- [ ] `launch_rl.py --dry-run` printed a valid job spec with your account's image/role
- [ ] A real RL job ran to `Completed` on a G-family instance (no GR00T involved)
- [ ] A checkpoint (`model_*.pt`) landed in S3

And you can **explain** the concepts:
- [ ] How does RL learn? (trial-and-error guided by reward signals in simulation)
- [ ] What is domain randomization? (randomized scene parameters so policy must be robust to succeed)
- [ ] How many parallel environments run simultaneously? (4096 on one GPU)
- [ ] What reward signal drives improvement? (success/failure at the manipulation task)
- [ ] What success rate does RL with domain randomization achieve? (~93-95% on randomized tasks)
- [ ] Where is the trained model? (S3 checkpoint + optionally TensorRT export)

---

## What You'll See During Training

Isaac Lab logs each PPO iteration:

```
Learning iteration 1/500
Computation: 3741 steps/s (collection: 0.715s, learning 0.106s)
Mean reward: -0.90
Mean episode length: 26.33
```

- **steps/s** — throughput across all parallel envs; bounded by the physics sim
  ("collection"), which is why more GPUs help.
- **Mean reward** — starts negative and should climb. Flat after 50+ iterations → check
  the reward function.
- **Mean episode length** — grows as the robot survives longer before termination.

**What's validated:** the SageMaker integration (entrypoint parses `resourceconfig.json`,
reads hyperparameters, launches `torchrun`; headless render; artifacts to `/opt/ml/model/`)
is load-tested on `Isaac-Velocity-Flat-Anymal-D-v0` — 128 envs, A10G, ~3,741 steps/s.
The UR3 task is registered but not yet container-wired (see Step 1).

> **Cosmos is not part of this lab.** Isaac Lab's procedural domain randomization
> (positions, lighting, textures) handles most sim-to-real transfer on its own; Cosmos
> photorealistic enhancement is a later, NIM-gated add-on (Lab 3). Most teams don't need it.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker login nvcr.io` fails | Verify NGC API key at https://ngc.nvidia.com/setup/api-key |
| Container build OOM | Isaac Lab base is ~15 GB. Ensure 30+ GB free disk space |
| Training doesn't converge | Check reward function. Try reducing domain randomization range initially |
| `CUDA out of memory` | Reduce `num_envs` in config (4096 → 2048 → 1024) |
| Policy success rate stays at 0% | Check reward function is correct; verify environment resets properly |

---

**Previous:** [← Lab 3: Cosmos World Generation](lab-3-cosmos-world-generation.md)
**Next:** [Lab 5: Edge Deployment →](lab-5-edge-deployment.md)
