# Lab 3: Cosmos World Generation

**Goal:** Generate photorealistic, diverse training environments using NVIDIA Cosmos to improve sim-to-real transfer
**Time:** 1-2 hours
**Cost:** ~$32/hr while Cosmos endpoint is running (teardown immediately after)

> ⚠️ **Workshop Note:** This lab requires `ml.p4d.24xlarge` (8× A100 80GB GPUs) for the Cosmos Transfer endpoint. This instance type is **not available in standard AWS Workshop Studio accounts** without special quota approval. In a live workshop setting, this lab is either:
> - **Instructor-led demonstration** — instructor runs the endpoint from a pre-approved account while attendees observe
> - **Self-paced only** — for customers running in their own AWS account with p4d quota approved
>
> Labs 0-2 and 4 work on standard workshop instances (g5.xlarge / g5.12xlarge). This lab is optional — Lab 4 (RL Refinement) works without Cosmos using built-in domain randomization.

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

Mode A: Transfer (enhance existing scenes)
  Isaac Lab renders scene → Cosmos makes it photorealistic → RL trains on enhanced images

Mode B: Generate (create new environments)
  Text prompt → Cosmos generates scene → Import to Isaac Lab → RL trains in new env
```

---

## Prerequisites

- Lab 2 completed (Isaac Sim workstation for visual verification)
- NVIDIA NIM API key (for Cosmos models)
  - Sign up at https://build.nvidia.com
  - Access Cosmos models: `nvidia/cosmos-transfer` and `nvidia/cosmos-generate`
- Base Isaac Lab scenes created (at minimum, the UR3 pick-and-place environment)

---

## Step 1: Set Up Cosmos NIM Access

For the self-hosted approach (recommended for toolkit/production):

```bash
# Deploy Cosmos Transfer 2.5 as a SageMaker endpoint (~10-15 min)
python training/scripts/cosmos_setup.py deploy

# Check status
python training/scripts/cosmos_setup.py status
```

This spins up a p4d.24xlarge instance with the Cosmos NIM container. Cost: ~$32/hr while running.

For the API approach (if you have NIM enterprise access):

```bash
# Store your NIM API key
aws secretsmanager create-secret \
  --name physical-ai/nim-api-key \
  --secret-string "nvapi-YOUR_KEY_HERE" \
  --region us-east-1
```

---

## Step 2: Generate Scene Variations with Cosmos Transfer

Cosmos Transfer takes a rendered sim image and makes it look real:

```python
# training/scripts/generate_scenes.py --mode transfer

import requests
import base64

def cosmos_transfer(sim_image_path: str, style_prompt: str) -> bytes:
    """Apply photorealistic transfer to a sim-rendered image."""
    with open(sim_image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    response = requests.post(
        "https://ai.api.nvidia.com/v1/cosmos/transfer",
        headers={"Authorization": f"Bearer {NIM_API_KEY}"},
        json={
            "image": image_b64,
            "prompt": style_prompt,
            "strength": 0.7,  # 0=keep original, 1=full restyle
        }
    )
    return base64.b64decode(response.json()["image"])

# Generate variations of the pick-and-place scene
styles = [
    "industrial warehouse with fluorescent lighting and metal shelving",
    "bright factory floor with natural light from skylights",
    "dimly lit manufacturing cell with task lighting only",
    "dusty workshop with worn metal surfaces and oil stains",
]

for i, style in enumerate(styles):
    result = cosmos_transfer("sim_render_base.png", style)
    with open(f"scene_variation_{i}.png", "wb") as f:
        f.write(result)
```

---

## Step 3: Generate New Environments with Cosmos Generate

Create entirely new training environments from text descriptions:

```python
# training/scripts/generate_scenes.py --mode generate

def cosmos_generate(prompt: str, num_scenes: int = 10) -> list:
    """Generate new environment backgrounds from text prompts."""
    scenes = []
    for i in range(num_scenes):
        response = requests.post(
            "https://ai.api.nvidia.com/v1/cosmos/generate",
            headers={"Authorization": f"Bearer {NIM_API_KEY}"},
            json={
                "prompt": prompt,
                "seed": i,  # Different seed = different variation
                "resolution": "1024x1024",
            }
        )
        scenes.append(base64.b64decode(response.json()["image"]))
    return scenes

# Generate diverse warehouse backgrounds
prompts = [
    "Empty industrial warehouse floor with metal shelving units, overhead fluorescent lights",
    "Manufacturing workcell with robot arm, parts bins, and safety barriers",
    "Clean room environment with white walls and bright even lighting",
    "Automotive assembly line with conveyor belts and hanging tools",
]

for prompt in prompts:
    scenes = cosmos_generate(prompt, num_scenes=25)
    # Upload to S3 for Isaac Lab to use as background textures
```

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
# Upload generated scenes to S3 for RL training
aws s3 sync ./cosmos_scenes/ s3://physical-ai-dev-datasets-802782083985/cosmos-scenes/ \
  --region us-east-1

echo "Ready for Lab 4: RL Refinement with Cosmos-enhanced environments"
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

## Cosmos Predict (Placeholder — Future Enhancement)

> **TODO:** Implement Cosmos Predict integration

While Cosmos **Transfer** takes existing video and makes it photorealistic, Cosmos **Predict** generates entirely new video from scratch:

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| **Cosmos Transfer** (what we built above) | Sim-rendered video + style prompt | Photorealistic version of the same scene | Close visual domain gap for existing environments |
| **Cosmos Predict** | Text prompt or seed image | Brand new video that doesn't exist yet | Create novel training environments ("robot in a clean room" when you only have warehouse data) |

**When you'd use Predict:**
- You need to train for an environment you haven't built in Isaac Lab yet
- You want massive environment diversity without modeling each one in USD
- You're exploring "what if" scenarios (new factory layouts, different robot placements)

**API:** Same p5 infrastructure, different NIM container (`cosmos-predict1-7b-text2world`). Same deployment pattern as Transfer.

```bash
# Future: deploy Cosmos Predict alongside Transfer
# docker pull nvcr.io/nim/nvidia/cosmos-predict1-7b-text2world:latest
# Same --gpus all --ipc=host flags required
```

This is a V3 enhancement — Transfer alone handles the majority of sim-to-real gap for manipulation tasks.

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
**Next:** [Lab 4: RL Refinement in Simulation →](lab-4-rl-refinement.md)
