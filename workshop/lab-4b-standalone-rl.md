# Lab 4b: Standalone RL (no GR00T)

**Goal:** Train an Isaac Lab reinforcement-learning policy on SageMaker — and render
a video of it — **without** the GR00T imitation-learning stage (Lab 1).
**Time:** 30 min hands-on + ~10 min training
**Cost:** ~$3 for a 50-iteration smoke test on `ml.g5.xlarge`

---

## Why this lab exists

Lab 4 frames RL as *refinement* of a GR00T model (Lab 1 → Lab 4). But Isaac Lab RL
stands on its own: you can train a policy from scratch in simulation with no
demonstrations and no foundation model. This lab is the standalone path — useful
for understanding RL in isolation, demos, and CI-style smoke tests.

It uses the **same `physical-ai/isaac-lab` container** as Lab 4 (built in CodeBuild),
just launched directly rather than through the GR00T→RL bridge.

> **Task note:** the validated standalone path uses an Isaac Lab **built-in task**
> (`Isaac-Velocity-Flat-Anymal-D-v0`, a quadruped). The toolkit's custom UR3
> pick-and-place env is **not yet gym-registered** in the container, so it won't
> resolve as a task id today — see `docs/ROADMAP.md` (Feature 2). Use a built-in
> task for this lab.

---

## Prerequisites

- Foundation stack deployed (`physical-ai/isaac-lab:latest` in ECR — built by CodeBuild)
- GPU quota for a **G-family** training instance (G5/G6/G6e).
  P-family (P4/P5) has no RT Cores → Isaac Sim crashes. (No GR00T model, no HF token needed.)

---

## Step 1: Preview the job (free)

```bash
python training/scripts/launch_rl.py --dry-run
```

This prints the exact SageMaker job it would create (resolved image, role, output
path, hyperparameters) and makes **no AWS calls**.

## Step 2: Launch a smoke test

```bash
python training/scripts/launch_rl.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 \
  --num-envs 4096 \
  --max-iterations 50 \
  --instance-type ml.g5.xlarge
```

50 iterations is a *smoke test* — it proves the pipeline runs end-to-end (~5-10 min).
The resulting policy will stumble, not walk well; locomotion needs ~1000+ iterations.

Monitor:
```bash
aws sagemaker describe-training-job \
  --training-job-name <JOB_NAME> \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus}'
```

## Step 3: Render a video of the trained policy

Once the job completes, render an MP4 (this reuses the play-mode render path):
```bash
python training/scripts/groot_to_rl_bridge.py render-video \
  --model-s3 s3://<DATASETS_BUCKET>/isaac-lab/output/<JOB_NAME>/output/model.tar.gz \
  --task Isaac-Velocity-Flat-Anymal-D-v0
```
(The launch command prints the exact `render-video` line for your job.)

---

## ✅ Lab 4b Checkpoint

- [ ] `launch_rl.py --dry-run` prints a valid job spec with your account's image/role
- [ ] A real RL job ran to `Completed` on a G-family instance (no GR00T involved)
- [ ] A checkpoint (`model_*.pt`) landed in S3
- [ ] (Optional) Rendered an MP4 of the trained policy

---

## How this differs from Lab 4

| | Lab 4 (refinement) | Lab 4b (standalone) |
|---|---|---|
| Needs Lab 1 / GR00T? | Yes (frames RL as refining the imitation model) | **No** |
| Entry point | `groot_to_rl_bridge.py rl-refine` | `launch_rl.py` |
| Container | `physical-ai/isaac-lab` | same |
| Task | (intended) UR3 — currently Anymal placeholder | built-in Anymal (UR3 not yet registered) |

---

**Related:** [Lab 4: RL Refinement](lab-4-rl-refinement.md) · [docs/ROADMAP.md](../docs/ROADMAP.md) (Feature 2: UR3 env registration)
