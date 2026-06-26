# Lab 3: Cosmos World Generation

**Goal:** Generate photorealistic, diverse training environments using NVIDIA Cosmos to improve sim-to-real transfer
**Time:** 1-2 hours
**Cost:** ~$32/hr while Cosmos endpoint is running (teardown immediately after)

> ⚠️ **Workshop Note:** This lab requires `ml.p4d.24xlarge` (8× A100 80GB GPUs) for the Cosmos Transfer endpoint. This instance type is **not available in standard AWS Workshop Studio accounts** without special quota approval. In a live workshop setting, this lab is either:
> - **Instructor-led demonstration** — instructor runs the endpoint from a pre-approved account while attendees observe
> - **Self-paced only** — for customers running in their own AWS account with p4d quota approved
>
> Labs 0-2 and 4 work on standard workshop instances (g5.xlarge / g5.12xlarge). This lab is optional — Lab 4 (RL Policy Training) works without Cosmos using built-in domain randomization.

---

## What You're Building

Isaac Lab's built-in domain randomization (Lab 4) changes object positions, colors, and lighting randomly. That works for many tasks. But if your robot needs to handle visually complex environments — cluttered warehouses, varied lighting conditions, realistic material textures — you need *photorealistic* diversity.

**Cosmos generates synthetic worlds** that look real:

1. **Cosmos Transfer** — Takes your sim-rendered scene and makes it photorealistic (adds scratches, dust, realistic shadows, material imperfections)
2. **Cosmos Generate** — Creates entirely new environments from text/image prompts ("generate a warehouse shelf with metal parts under fluorescent lighting")

**Why this matters for robots:**

The #1 reason robot policies fail in the real world is the *visual domain gap* — sim looks too clean, too perfect, too uniform. Real factories have:
- Scratched metal surfaces that confuse depth estimation
- Mixed lighting (fluorescent overhead + natural from windows + task lights)
- Cluttered backgrounds the camera has never seen
- Dust, oil, and wear on objects

Cosmos closes this gap by training the policy on photorealistic variations *before* it ever sees the real world.

---

## When to Use Cosmos vs. Built-in Randomization

| Approach | Use When | Cost |
|----------|----------|------|
| **Isaac Lab procedural randomization** (Lab 4 default) | Object positions, basic lighting, simple color variation. Works for 80% of manipulation tasks. | Free |
| **Cosmos Transfer** | Your policy fails on real hardware due to visual appearance (materials, textures, lighting quality) | ~$0.01-0.05 per frame |
| **Cosmos Generate** | You need environment diversity (many different scenes, backgrounds, layouts) | ~$0.10-0.50 per scene |

**Start with built-in randomization. Add Cosmos only if sim-to-real transfer is poor.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Scene Generation Pipeline                                   │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Base USD │───▶│  Cosmos NIM  │───▶│  Photorealistic  │  │
│  │  Scenes   │    │  API         │    │  Training Scenes │  │
│  │  (Isaac   │    │              │    │  (stored in S3)  │  │
│  │   Lab)    │    │  Transfer or │    │                  │  │
│  │           │    │  Generate    │    │  Used by Lab 4   │  │
│  └──────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## What Exactly Happens (Concrete Example)

Here's the actual flow when you run Cosmos Transfer on a training scene:

**Input you send:**
1. A sim-rendered image — e.g., an Isaac Lab screenshot showing the UR3 arm reaching for a red block in a bin
2. A style prompt — e.g., "industrial warehouse with fluorescent lighting, scratched metal surfaces"

**What Cosmos does:**
- Detects the structural content (robot arm shape, object position, spatial layout)
- Replaces the "video game" textures with photorealistic materials
- Adds realistic lighting, shadows, reflections, and surface imperfections
- Preserves the exact geometry and robot pose (so training labels stay valid)

**Output you get:**
- The same scene, same robot pose, same object position — but looking like a photograph instead of a simulation screenshot

**How this fits the pipeline:**
```
Isaac Lab renders 100 frames of the UR3 picking a block (clean sim visuals)
    ↓
Cosmos Transfer generates 4 style variations of each → 400 photorealistic frames
    ↓
RL trains on all 400 frames (the robot learns to succeed regardless of visual style)
    ↓
On real hardware: the wrist camera sees "warehouse lighting" → policy already trained on it → succeeds
```

**Without Cosmos:** Robot trained only on sim's flat gray surfaces. Real factory has scratched metal → policy confused → drops object.

**With Cosmos:** Robot trained on scratched metal, dusty surfaces, mixed lighting. Real factory looks familiar → policy works.

---

## Two Modes

```
Mode A — Transfer (this lab, available):
  Isaac Lab renders a clip → Cosmos Transfer restyles it photorealistically → RL trains on it

Mode B — Generate (future, Cosmos Predict / Cosmos 3):
  Text prompt → Cosmos generates a new scene → import to Isaac Lab
  (separate model; not wired up yet — see docs/ROADMAP.md Feature 3)
```

---

## Prerequisites

- Lab 2 completed (Isaac Sim workstation for visual verification)
- **NGC API key** in Secrets Manager (`physical-ai/ngc-api-key`) — the Cosmos NIM
  pulls model weights with it at container start
- **P5 service quota** (defaults to 0 — request an increase) for the Spot p5 instance
- Sim clips rendered to **MP4** (93–480 frames) to feed Cosmos Transfer

---

## Step 1: Launch the Cosmos GPU Instance

Cosmos Transfer 2.5 runs as an NVIDIA **NIM container on an EC2 Spot p5** (8× H100),
serving `POST /v1/infer` on port 8000. **It does not run on a SageMaker real-time
endpoint** — SageMaker's managed GPUs ship NVIDIA driver 470, but Cosmos needs 580+.
The full runbook is in [`docs/cosmos-deployment-guide.md`](../docs/cosmos-deployment-guide.md).

```bash
# Preview the launch (no AWS writes):
python training/scripts/cosmos_setup.py launch --dry-run

# Launch the Spot p5 (boots scripts/cosmos-userdata.sh → driver, container, NIM):
python training/scripts/cosmos_setup.py launch
```

Prerequisites: **P5 service quota** (defaults to 0 — request an increase), an
**NGC API key** in Secrets Manager (`physical-ai/ngc-api-key`), and an instance
profile with ECR + Secrets Manager read. Bootstrap takes ~10–15 min. Cost: ~$7–8/hr
on Spot — **terminate when done.**

```bash
# Wait for the NIM to report ready (checked over SSM — no inbound port needed):
python training/scripts/cosmos_setup.py status --instance-id i-xxxx
# → Health: {"status":"ready"}
```

---

## Step 2: Restyle Sim Videos with Cosmos Transfer

Cosmos Transfer operates on **video** (MP4, 93–480 frames) — not single images. It
takes a sim-rendered clip + a style prompt + a control modality (edge/depth/seg/vis)
and returns a photorealistic clip with the same geometry/motion.

```bash
# Render sim clips to MP4 first, then:
python training/scripts/cosmos_setup.py generate \
  --instance-id i-xxxx \
  --input ./sim_videos/ \
  --output ./cosmos_out/ \
  --control edge \
  --dry-run    # prints the exact /v1/infer payload; drop --dry-run to run
```

The request shape (per the runbook), for reference:
```json
POST http://localhost:8000/v1/infer
{
  "prompt": "industrial warehouse with fluorescent lighting and metal shelving",
  "video": "<base64-encoded MP4, 93-480 frames>",
  "edge": {"enabled": true},
  "num_steps": 35,
  "guidance": 3,
  "resolution": "480"
}
```

> **Performance note (from the runbook):** on a single GPU (CP=1) a 93-frame clip
> can exceed a 10-min request timeout. The userdata starts the NIM with
> `NIM_MODEL_PROFILE=latency` to use all 8 H100s (CP=8, ~8× faster). This path is
> documented and the health/API are confirmed, but a full restyle has **not** been
> validated end-to-end in this repo (blocked on sustained p5 capacity).

When done, **terminate** to stop Spot charges:
```bash
python training/scripts/cosmos_setup.py terminate --instance-id i-xxxx
```

---

## Step 3 (Future): Generate New Environments with Cosmos *Predict*

> **Not implemented in this toolkit yet — different model from Transfer.** Creating
> *entirely new* environments from a text prompt (rather than restyling an existing
> sim clip) is the job of **Cosmos Predict / Cosmos 3** (`NVIDIA/cosmos-framework`),
> a separate model from the Cosmos *Transfer* used in Steps 1–2. The Cosmos 3
> Generator image is built in CodeBuild (`physical-ai/cosmos3` in ECR), but a
> generation runner is not wired up, and Cosmos 3 does **not** yet support the
> controlled (edge/depth/seg) *transfer* this lab needs.
>
> See [`docs/ROADMAP.md`](../docs/ROADMAP.md) (Feature 3) for the plan. For now,
> use Step 2 (Transfer) for sim→photorealistic, and Isaac Lab's built-in domain
> randomization for environment diversity.

---

## Step 4: Integrate with Isaac Lab Domain Randomization

Use Cosmos-generated scenes as background textures and environment variations in RL training:

```python
# In your Isaac Lab environment config:
domain_randomization:
  backgrounds:
    source: "s3://physical-ai-dev-datasets/cosmos-scenes/"
    mode: "random_per_episode"  # New background each episode reset

  textures:
    source: "s3://physical-ai-dev-datasets/cosmos-textures/"
    apply_to: ["bin", "table", "walls"]

  lighting:
    # Cosmos-generated HDR environment maps
    hdri_source: "s3://physical-ai-dev-datasets/cosmos-hdri/"
    randomize: true
```

---

## Step 5: Validate Visual Quality

On your Lab 2 workstation, visually verify the generated scenes look realistic:

```bash
# Render a few scenes and compare sim vs. Cosmos-enhanced
python training/scripts/generate_scenes.py \
  --mode compare \
  --output ./scene_comparison/ \
  --num-samples 10
```

**What to check:**
- Do materials look realistic? (metal should have reflections, not flat gray)
- Is lighting varied enough? (not all scenes should look the same)
- Are textures at appropriate resolution? (no obvious pixelation)
- Does the robot still look correct? (Cosmos shouldn't distort the robot itself)

---

## Step 6: Upload Scenes for Lab 4

```bash
# Resolve your datasets bucket from the Foundation stack (no hardcoded account):
BUCKET=$(aws cloudformation describe-stacks --stack-name PhysicalAi-dev-Foundation \
  --query 'Stacks[0].Outputs[?OutputKey==`DatasetsBucketName`].OutputValue' --output text)

# Upload generated scenes to S3 for RL training
aws s3 sync ./cosmos_out/ "s3://$BUCKET/cosmos-scenes/"

echo "Ready for Lab 4: RL Policy Training with Cosmos-enhanced environments"
```

---

## ✅ Lab 3 Checkpoint

- [ ] Cosmos NIM API key configured
- [ ] Generated photorealistic scene variations using Cosmos Transfer
- [ ] Generated new environments using Cosmos Generate
- [ ] Visually verified quality on the workstation (Lab 2)
- [ ] Uploaded scenes to S3 for RL training (Lab 4)
- [ ] Understand when to use Transfer vs. Generate vs. built-in randomization

---

## Cost Estimation

| Operation | Est. Cost | Typical Volume |
|-----------|-----------|---------------|
| Cosmos Transfer (per image) | ~$0.01-0.05 | 100-1000 images = $1-50 |
| Cosmos Generate (per scene) | ~$0.10-0.50 | 50-200 scenes = $5-100 |
| S3 storage (generated assets) | ~$0.023/GB | 10-50 GB = $0.23-1.15/month |

**For a typical project:** 200 Transfer images + 50 Generated scenes ≈ **$15-30 one-time cost**.

---

## Without Cosmos (Fallback)

If you don't have NIM API access, Lab 4 still works. Isaac Lab's built-in procedural domain randomization provides:
- Random object positions (±5cm)
- Random lighting intensity and direction
- Random object colors (uniform RGB sampling)
- Random camera noise

This gets you ~85-90% sim-to-real transfer for standard manipulation. Cosmos pushes it to 95%+ for visually challenging environments.

```bash
# Run Lab 4 without Cosmos (uses built-in randomization only)
python training/scripts/train.py --no-cosmos --domain-rand-only
```

---

**Previous:** [← Lab 2: Isaac Sim Workstation](lab-2-isaac-workstation.md)
**Next:** [Lab 4: RL Policy Training →](lab-4-rl-refinement.md)
