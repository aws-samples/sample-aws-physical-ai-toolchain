# Lab 4b: Train a Robot Arm (Universal Robots, optional)

**Goal:** Train a manipulation policy on a Universal Robots UR10 arm reaching task (optional alternate path to Lab 4)
**Time:** 3 hours (same pipeline, different robot)
**Cost:** Same as Lab 4 — ~$3 smoke test, ~$28 full training

---

## What This Is

This is an **optional alternate path** to [Lab 4](lab-4-rl-refinement.md) — the exact same RL pipeline (train → export → eval), but on a **robot arm** instead of a quadruped. Where Lab 4 trains the Anymal quadruped to walk, this lab trains a **Universal Robots UR10** arm to reach target positions.

**Same infrastructure, different robot.** Everything you learned in Lab 4 applies here — `pai rl launch`, checkpointing to S3, export/eval on the Lab 2 workstation. The only difference is the task ID you pass to the commands.

**Prerequisites:** same as Lab 4:
- [ ] Foundation stack deployed → `isaac-lab` image in ECR
- [ ] Lab 2 workstation (for Step 3 visual playback + eval)
- [ ] GPU quota for `ml.g5.xlarge` (smoke test) or `ml.g5.12xlarge` (full run)

> ✅ **Validated:** the full pipeline (train → export → eval → visual playback) was validated end-to-end on L40S hardware (Isaac Sim 4.5 / Isaac Lab 2.x, driver 580). Training (50 iterations) showed mean reward climb from -1.31 to -1.09 with decreasing end-effector position/orientation error. Export produced `policy.onnx`, eval ran successfully, and visual playback worked on the DCV desktop. Short smoke-test policies won't fully solve reaching (same as Anymal) — validation proves the pipeline works, not policy quality.

---

## 🏃 Quick Runbook

| # | Action | Command (summary) | ✅ Success check |
|---|--------|-------------------|-----------------|
| 1 | Smoke test (50 iter) | `pai rl launch --task Isaac-Reach-UR10-v0 --num-envs 4096 --max-iterations 50 --instance-type ml.g5.xlarge` | job launches; logs `Mean reward` (likely negative at 50 iter) |
| 2 | Full training (optional) | `pai rl launch --task Isaac-Reach-UR10-v0 --num-envs 4096 --max-iterations 1500 --instance-type ml.g5.12xlarge` | checkpoint `model_*.pt` in S3; reward climbs |
| 3 | Export + eval + watch | Same as Lab 4 Step 4: `play.py` exports `policy.onnx`, `evaluate.py` scores it, live playback on DCV | `eval_metrics.json` written; watch the arm reach in the viewport |

See [Lab 4](lab-4-rl-refinement.md) for the full walkthrough of each step — the commands above are just the task ID swapped. The rest of this doc explains what's different.

---

## What's Different from Lab 4

**The task:** `Isaac-Reach-UR10-v0` instead of `Isaac-Velocity-Flat-Anymal-D-v0`. This is a built-in Isaac Lab task (ships in the `isaac-lab` container) that trains a 6-DOF UR10 arm to reach randomized target poses. It's a **manager-based** RL task, uses **rsl_rl + PPO**, and fetches assets from NVIDIA's cloud servers (no localhost Nucleus needed) — the same class as the validated Anymal task.

**The reward:** reaching the target position (low distance from end-effector to goal) with stability penalties, not locomotion velocity tracking. Expect negative reward early in training; it should climb as the arm learns to reach.

**The physics:** 6-DOF arm control instead of quadruped gaits. Joint limits, collision avoidance, and Cartesian space reaching are the challenges here (vs. stability and foot contacts for locomotion).

**Everything else is the same:** headless Isaac Sim, 4096 parallel envs, PPO, domain randomization, SageMaker/Batch, export to ONNX, eval on the Lab 2 workstation.

---

## Step 1: Launch Training

Same `pai rl launch` command as Lab 4, with `--task Isaac-Reach-UR10-v0`:

```bash
# Smoke test first (50 iterations, ~5-10 min, ~$3)
pai rl launch \
  --task Isaac-Reach-UR10-v0 \
  --num-envs 4096 \
  --max-iterations 50 \
  --instance-type ml.g5.xlarge
```

**What happens:** trains the UR10 arm to reach randomized targets in headless Isaac Sim on SageMaker. Logs print `Mean reward` and `Mean episode length` just like Lab 4. At 50 iterations the policy will be weak (reward likely still negative); this is a smoke test to prove the pipeline runs end-to-end.

For a usable policy, scale up to **1500+ iterations**:

```bash
pai rl launch \
  --task Isaac-Reach-UR10-v0 \
  --num-envs 4096 \
  --max-iterations 1500 \
  --instance-type ml.g5.12xlarge --runtime-min 240
```

Monitor with `pai rl status <job-name>`. Checkpoints save to S3 every 100 epochs.

> **What to watch:** reward should climb from negative toward zero or positive (validated run showed -1.31 → -1.09 over 50 iterations). If it stays flat after 200+ iterations, investigate — the task should show clear improvement like Anymal in Lab 4.

---

## Step 2: Export and Evaluate

**Do this on the Lab 2 workstation** — same as [Lab 4 Step 4](lab-4-rl-refinement.md#step-4-evaluate-the-trained-policy). Enter the `isaac-lab` container:

```bash
~/run-isaac-lab.sh                      # prompt becomes /workspace/isaaclab#
cd /workspace/toolchain
```

### 2a. Pull the checkpoint (on the HOST)

Same S3 fetch as Lab 4; the checkpoint naming and path are identical:

```bash
cd ~/aws-physical-ai-toolchain
BUCKET=$(aws sts get-caller-identity --query Account --output text | xargs -I{} echo physical-ai-dev-datasets-{})
aws s3 cp "s3://$BUCKET/isaac-lab/output/<job-name>/output/model.tar.gz" .
tar -xzf model.tar.gz
ls logs/rsl_rl/*/*/model_*.pt          # confirm it extracted to ./logs
```

### 2b. Export (inside the container)

Run NVIDIA's `play.py` to export `policy.{pt,onnx}`:

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-UR10-v0 \
  --checkpoint /workspace/toolchain/logs/rsl_rl/isaac_reach_ur10/<timestamp>/model_49.pt \
  --num_envs 1 --headless --video --video_length 1
```

> Use the **absolute** `/workspace/toolchain/...` path. `--video --video_length 1` makes it exit on its own after writing. The folder name under `logs/rsl_rl/` will differ from Anymal's; check `ls logs/rsl_rl/` to find the exact path.

Copy the artifacts:

```bash
mkdir -p /workspace/toolchain/model_exported
cp /workspace/toolchain/logs/rsl_rl/isaac_reach_ur10/<timestamp>/exported/policy.* \
   /workspace/toolchain/model_exported/
```

### 2c. Evaluate (inside the container)

Same `evaluate.py` script, just pass `--env Isaac-Reach-UR10-v0`:

```bash
/workspace/isaaclab/isaaclab.sh -p /workspace/toolchain/training/scripts/evaluate.py \
  --env Isaac-Reach-UR10-v0 \
  --checkpoint /workspace/toolchain/model_exported/policy.pt \
  --num-episodes 5 --output-dir /workspace/toolchain/eval_results/
```

Writes `eval_metrics.json` with `success_rate_pct`, `avg_reward`, etc. Use **absolute** `/workspace/toolchain/...` paths.

> **Note:** `success_rate_pct` will read 0% for this reaching task because the success detector targets pick-and-place semantics (object drop/collision/cycle time). For reaching, the meaningful signals are `avg_reward` and decreasing `position_error` in training logs.

---

## Step 3: Watch Your Policy Run (Visual)

Same as [Lab 4 Step 4c](lab-4-rl-refinement.md#step-4c-watch-your-policy-run-visual) — run `play.py` **without** `--headless`/`--video` from the DCV desktop to see the trained arm reach live:

```bash
# Run from the DCV terminal (needs the X display):
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-UR10-v0 \
  --checkpoint /workspace/toolchain/logs/rsl_rl/isaac_reach_ur10/<timestamp>/model_49.pt \
  --num_envs 50
```

> Pass the **raw `model_49.pt` checkpoint** (not the exported `policy.pt`). First load takes ~2-3 min; `Ctrl+C` to quit. If the viewport looks empty, scroll out or adjust camera — the arms spawn at the origin. `--num_envs 50` shows a grid of arms; drop to `1` to watch a single one.

Compare the trained policy (arm reaches the target smoothly) to the untrained behavior at the start (flailing randomly). That's the payoff of RL.

---

## Why Not the Custom UR3 Env?

The repo ships an optional custom UR3 pick-and-place task (`training/envs/pick_and_place_ur3.py`, task ID `PickAndPlaceUR3-v0`). **It's not container-wired** and not GPU-validated, so it's not part of the validated flow today. Use `Isaac-Reach-UR10-v0` (this lab) for a working arm task on the proven infrastructure.

The custom UR3 env is the "build-your-own custom env" advanced path — see [Lab 4 Step 1/2](lab-4-rl-refinement.md#step-1-launch-rl-training) and [ROADMAP](../docs/ROADMAP.md) Feature 2 for details. Think of it as a template for when you need a task Isaac Lab doesn't provide, not a runnable lab exercise.

---

## ✅ Lab 4b Checkpoint

You've completed Lab 4b if:
- [ ] A UR10 training job ran to `Completed` on SageMaker
- [ ] Checkpoint exported to `policy.onnx`
- [ ] `evaluate.py` wrote `eval_metrics.json`
- [ ] You watched the trained arm reach targets in the DCV viewport

And you can **explain**:
- [ ] How is this different from Lab 4? (same pipeline, arm task instead of locomotion task)
- [ ] Why UR10 built-in instead of the custom UR3 env? (UR10 is container-wired and uses the validated infra; UR3 is unvalidated "build-your-own" template)
- [ ] What does the reward measure here? (distance from end-effector to target + stability penalties)

---

## Next Steps

- **Deploy to edge:** Lab 5 walks through pushing the `policy.onnx` to a Jetson + Greengrass
- **Switch back to quadrupeds:** Lab 4 (Anymal) is the validated path for the rest of the workshop
- **Build your own task:** use the custom UR3 env as a template and follow NVIDIA's [Isaac Lab custom env tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)

---

**Previous:** [← Lab 4: RL Policy Training](lab-4-rl-refinement.md)
**Next:** [Lab 5: Edge Deployment →](lab-5-edge-deployment.md)
