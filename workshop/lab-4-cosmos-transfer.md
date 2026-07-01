# Lab 4: Cosmos Transfer — Photorealistic Data Augmentation

> **Status:** Validated on p5.48xlarge (8x H100 80GB) using the NIM container.
> Geometry preservation works. Color fidelity and framerate matching require tuning.
> See notes below for known issues and workarounds.

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

> **Status:** Validated on p5.48xlarge (8x H100 80GB) with the NIM container.
> Quality tuning still needed — output colors shift and framerate needs matching.
> See `docs/cosmos3-validated-runbook.md` for the full validated Transfer NIM setup.

### Setup (validated):
1. Launch p5.48xlarge (H100 80GB required — A100 40GB OOMs, NIM needs compute cap 9.0)
2. Pull NIM: `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest` (requires NGC API key)
3. Run with `--gpus all` — NIM auto-selects H100 latency profile with FP8
4. Server ready when `/v1/health/ready` returns 200

### API call (validated):
```python
import base64, requests, json

# Encode input video (must be 93-480 frames)
with open("input_video.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode()

payload = {
    "prompt": "Industrial factory with scratched metal table and fluorescent lighting",
    "video": video_b64,
    "edge": {},           # edge control preserves geometry
    "num_steps": 35,      # full quality (10 = fast but low quality)
    "guidance": 3
}

resp = requests.post("http://localhost:8000/v1/infer", json=payload, timeout=900)

# Response is JSON with base64-encoded video
data = resp.json()
video_bytes = base64.b64decode(data["b64_video"])
with open("output.mp4", "wb") as f:
    f.write(video_bytes)
```

### Key constraints:
- **Input must be 93-480 frames** (below 93 → HTTP 422 error)
- **H100 80GB minimum** (A100 80GB OOMs on VAE decode; A100 40GB not enough)
- **NIM requires NGC API key** + HuggingFace token (for Cosmos-Guardrail1 download)
- **Output is base64 JSON**, not raw video — must decode `data["b64_video"]`
- **Output is HEVC encoded** — convert to H.264 for Mac playback: `ffmpeg -i output.mp4 -c:v libx264 output_h264.mp4`
- **Framerate mismatch:** NIM outputs at 16fps regardless of input fps. Re-encode output to match original fps.

### Known quality issues (to investigate):
- Colors shift toward the prompt's described environment (edge control doesn't preserve colors)
- `vis` control mode preserves colors better but produces artifacts at the "latency" FP8 profile
- The BF16 `throughput` profile may produce better quality (not yet tested)
- Lower `guidance` (1-2) may preserve more of the original appearance

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
| p5.48xlarge (Capacity Block) | ~$37/hr (~$574 for 16hr block) | H100 80GB required |
| Generation time | ~8-10 min per 93-frame clip (35 steps) | Faster with fewer steps (lower quality) |
| NIM container pull | ~20 min first time | 56 GB image from NGC |

Much cheaper than Lab 3 (Predict) because Transfer 2.5 is only 2B params vs. 64B.

---

**Previous:** [← Lab 3: Cosmos World Generation](lab-3-cosmos-world-generation.md)
**Next:** [Lab 5: RL Policy Training with Isaac →](lab-5-rl-refinement-with-isaac.md)
