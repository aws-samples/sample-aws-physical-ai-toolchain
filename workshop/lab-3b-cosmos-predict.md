# Lab 3b: Cosmos Predict — World Generation from Text

**Goal:** Generate entirely new training environments from text descriptions using Cosmos Predict
**Time:** 1-2 hours
**Cost:** ~$8-15/hr (Spot p5 instance while running)

> ⚠️ **Status:** Placeholder lab. Content to be built. An existing immersion day is available for Cosmos Predict separately. This lab will integrate that workflow into our Physical AI pipeline context.

> ⚠️ **Same infrastructure requirement as Lab 3:** Requires `p5.48xlarge` (8× H100) with NVIDIA fabricmanager. Same workshop studio limitations apply.

---

## How This Differs from Lab 3 (Transfer)

| | Lab 3: Cosmos Transfer | Lab 3b: Cosmos Predict |
|---|---|---|
| **Input** | Existing sim-rendered video | Text prompt or seed image |
| **Output** | Photorealistic version of the same scene | Brand new video that doesn't exist yet |
| **Geometry** | Preserves original layout (robot pose, object positions) | Creates novel layouts from scratch |
| **Use case** | Close visual domain gap for environments you already have | Create training environments you haven't built in sim |
| **When to use** | You have an Isaac Lab env but it looks too "game-like" | You need diversity in environments themselves, not just textures |

---

## What Cosmos Predict Does

Given a text prompt like:
> "A robot arm picking red blocks from a metal bin on a factory floor, fluorescent overhead lighting, safety barriers visible in background"

Cosmos Predict generates a physically plausible video of that scene — without you ever building it in Isaac Lab or USD.

**Why this matters for Physical AI:**
- Training on 1 environment = overfitting to that layout
- Building 100 USD environments manually = months of work
- Cosmos Predict generating 100 environments from prompts = minutes

---

## Planned Integration

```
Text prompts (100 environment descriptions)
    ↓
Cosmos Predict (p5 H100)
    ↓
100 generated training videos
    ↓
Extract frames as domain randomization backgrounds for RL (Lab 4)
    ↓
Policy trained on diverse environments → better generalization
```

---

## Prerequisites (same as Lab 3)

- p5.48xlarge Spot instance (us-east-2 recommended)
- NVIDIA fabricmanager installed
- Docker with `--gpus all --ipc=host --ulimit memlock=-1`
- NGC API key in Secrets Manager

---

## Steps (TODO — to be implemented)

### Step 1: Deploy Cosmos Predict NIM

```bash
# Container: nvcr.io/nim/nvidia/cosmos-predict1-7b-text2world:latest
# Same deployment pattern as Lab 3 Transfer
# TODO: Add to cosmos_setup.py as a 'predict' mode
```

### Step 2: Generate Environment Videos

```bash
# TODO: Script that sends text prompts and saves generated videos
# python training/scripts/cosmos_setup.py predict \
#   --prompts prompts.txt \
#   --output ./generated_environments/
```

### Step 3: Integrate with RL Training

```bash
# TODO: Use generated videos as domain randomization source in Lab 4
```

---

## Reference

- [Cosmos Predict Paper](https://arxiv.org/abs/2501.03575)
- [NVIDIA Cosmos GitHub](https://github.com/NVIDIA/Cosmos)
- [Existing Cosmos Predict Immersion Day](TODO: add link)
- NIM container: `nvcr.io/nim/nvidia/cosmos-predict1-7b-text2world`

---

**Previous:** [← Lab 3: Cosmos Transfer](lab-3-cosmos-world-generation.md)
**Next:** [Lab 4: RL Refinement →](lab-4-rl-refinement.md)
