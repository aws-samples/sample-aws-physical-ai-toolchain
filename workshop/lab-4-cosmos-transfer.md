# Lab 4: Cosmos Transfer — Photorealistic Data Augmentation

> **Status:** In progress — validating on p4d.24xlarge Spot.

**Goal:** Use Cosmos Transfer 2.5 to restyle your existing training videos with photorealistic visual variations while preserving exact robot motion and geometry — making your action labels remain valid
**Time:** 1-2 hours
**Cost:** ~$13/hr on Spot p4d.24xlarge (or ~$37/hr via Capacity Block)

> **New to Transfer vs. Predict?** Lab 3 (Predict) generates *new* demonstrations from
> prompts. This lab (Transfer) restyles *existing* demonstrations — same motion, different
> visual appearance. Both use Cosmos, but for different purposes.

---

## What You're Building

You have 27 real demonstrations from Lab 1. The robot's motion and actions are perfect —
but the visual environment is always the same lab bench. When you deploy to a factory,
the policy fails because it's never seen scratched metal, dim lighting, or cluttered
backgrounds.

**Cosmos Transfer 2.5 solves this** by taking your existing wrist camera videos and
restyling them with different visual environments — while preserving the exact geometry,
motion, and pixel-level structure of every frame. Your action labels stay perfectly valid
because nothing moved — only the visual appearance changed.

**This is the canonical data augmentation approach** recommended by NVIDIA's Physical AI
Reference Architecture, the SO-101 Sim-to-Real tutorial, and the GR00T-Mimic Blueprint.

---

## How It Differs from Lab 3

| | Lab 3 (Predict/World Gen) | Lab 4 (Transfer) |
|---|---|---|
| Model | Cosmos 3 Super (64B) | Cosmos Transfer 2.5 (2B) |
| What it does | Generates new video from reference + prompt | Restyles existing video preserving structure |
| Actions preserved? | No — new trajectory, no action labels | **Yes — exact same motion, same labels** |
| Control signal | Text prompt only (loose) | Edge/depth/seg map (strict geometry constraint) |
| Input | Reference video + text prompt | Video + edge map + style prompt |
| Output | Novel demonstration (vision-only) | Restyled demonstration (action-paired) |
| Use case | Scale dataset with new trajectories | Scale dataset with visual diversity |
| GPU required | 8x H100 (640 GB) | 1x A100 80GB or 8x A100 40GB |
| Container | `vllm/vllm-omni:cosmos3` | `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b` |
| Cost | ~$37/hr (p5 Capacity Block) | ~$13/hr (p4d Spot) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Cosmos Transfer 2.5 — Pixel-Faithful Restyling                              │
│                                                                              │
│  ┌──────────────┐    ┌────────────────────┐    ┌───────────────┐            │
│  │ Lab 1 wrist  │    │  EC2 p4d.24xlarge   │    │  Output       │            │
│  │ camera MP4   │───▶│  (Spot instance)    │───▶│  Restyled MP4 │            │
│  │ + edge map   │    │                    │    │               │            │
│  │ + style      │    │  Cosmos Transfer   │    │  SAME motion  │            │
│  │   prompt     │    │  2.5 NIM           │    │  NEW visuals  │            │
│  └──────────────┘    │  (8x A100 40GB)    │    │  Actions stay │            │
│                      └────────────────────┘    │  valid!       │            │
│                                                └───────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘

Key: the edge map extracted from your input video tells the model WHERE everything is.
The prompt tells it HOW things should look. Geometry is locked — only appearance changes.
```

---

## Prerequisites

- **Lab 1 completed** — you have wrist camera MP4s in S3
- **NGC API key** in Secrets Manager (`physical-ai/ngc-api-key`) — Transfer NIM pulls from NGC
- **HuggingFace token** in Secrets Manager (`physical-ai/hf-token`) — for guardrail model
- **P4d quota** (or Spot access) — 8x A100 40GB

---

## Steps

> **TODO:** Full step-by-step instructions will be added once validation completes.
> The validation is running now on a Spot p4d.24xlarge in us-west-2.

### Expected workflow:
1. Launch p4d Spot instance (or use Capacity Block)
2. Pull `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest`
3. Start the NIM with all 8 GPUs
4. Copy your Lab 1 wrist camera clip to the instance
5. POST to `/v1/infer` with: video (base64) + edge control + style prompt
6. Get back: same video restyled with different visual appearance
7. Merge restyled videos into dataset (same parquet actions, new video file)
8. Re-train GR00T (Lab 1) on the augmented dataset

### Expected API call:
```json
POST http://localhost:8000/v1/infer
{
  "prompt": "industrial warehouse with fluorescent lighting, scratched metal surfaces",
  "video": "<base64-encoded MP4>",
  "edge": {},
  "num_steps": 10,
  "guidance": 3
}
```

The `edge: {}` tells Transfer to auto-extract edge maps from the input video and use
them as structural constraints — locking the geometry while restyling the appearance.

---

## ✅ Lab 4 Checkpoint

- [ ] Transfer NIM running on p4d (8x A100)
- [ ] Posted one clip with edge control
- [ ] Output preserves robot motion exactly (visual comparison confirms)
- [ ] Same video, different visual style — action labels valid
- [ ] (Optional) Generated multiple style variations
- [ ] (Optional) Merged into augmented dataset and re-trained

---

## Cost Estimation

| Resource | Cost | Notes |
|----------|------|-------|
| p4d.24xlarge Spot | ~$13/hr | Often available in us-west-2 |
| p4d.24xlarge On-Demand | ~$32/hr | If Spot unavailable |
| Capacity Block (p4d) | Varies | For guaranteed capacity |
| Generation time | ~7-8 min per clip | Single H100; faster with all 8 GPUs |

Much cheaper than Lab 3 (Predict) because Transfer 2.5 is only 2B params vs. 64B.

---

**Previous:** [← Lab 3: Cosmos World Generation](lab-3-cosmos-world-generation.md)
**Next:** [Lab 5: RL Policy Training with Isaac →](lab-5-rl-refinement-with-isaac.md)
