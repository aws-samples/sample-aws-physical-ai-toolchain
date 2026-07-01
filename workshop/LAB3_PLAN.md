# Lab 3 Finalization Plan

**Status:** In progress
**Owner:** devris
**Last updated:** 2026-06-30

---

## Summary

Rewrite Lab 3 from "Cosmos World Generation" (generate scenes from scratch) to
"Cosmos Data Augmentation" (augment existing training data with photorealistic
variations). Uses **Cosmos 3 via vLLM-Omni on EKS** (the validated, published path
from awslabs/awsome-distributed-ai) instead of the old Cosmos Transfer 2.5 NIM on
Spot p5 (which never worked end-to-end).

**Key pivot:** The old NIM/Spot-p5 path (`cosmos_setup.py`) is deprecated. The new
primary path is an EKS Job running the official `vllm/vllm-omni:cosmos3` image with
a simple `curl -F input_reference=@clip.mp4` API. This is simpler, validated, and
uses the current-generation model (Cosmos 3 Super 64B, not the deprecated Transfer 2.5).

---

## Why the Reframe

The current Lab 3 frames Cosmos as a "world generator" that creates novel scenes from
text prompts. Research from NVIDIA's official references shows this is NOT the common
customer use case:

| Source | What it says |
|--------|--------------|
| [NVIDIA Physical AI Reference Architecture](https://docs.nvidia.com/enterprise-reference-architectures/physical-ai-reference-architecture-deployment-guide/latest/physical-ai-workflow.html) | Six-step pipeline: MimicGen → HDF5→MP4 → **Cosmos Transfer visual augmentation** → MP4→HDF5 → train |
| [SO-101 Sim-to-Real Tutorial](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html) | Strategy 3: "Cosmos augmentation" — augment 75 sim demos with 70 Cosmos-restyled variants |
| [GR00T-Mimic Blueprint](https://build.nvidia.com/nvidia/isaac-gr00t-synthetic-manipulation/blueprintcard) | "Cosmos Transfer to augment the data with photorealism" in the synthetic manipulation pipeline |
| [Cosmos Transfer 2.5 Docs](https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html) | Two workflows: "Simulations to Photorealism" and "Scale World State Diversity" — both are augmentation, not creation |
| [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook) | Primary recipes are post-training + augmentation, not generation from scratch |
| [awslabs/awsome-distributed-ai cosmos3](https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/pytorch/cosmos3) | Flywheel: generate augments training corpus, policy trains on augmented data |

**Key insight:** Cosmos Transfer takes existing video (sim renders or real teleop
recordings) and produces photorealistic variations. Actions/labels stay unchanged —
only the visual observations are restyled. This multiplies training data 4-10x.

---

## What Changes vs. What Stays

### Stays the same (validated infra — from awsome-distributed-ai reference)
- `kubernetes/generate-vllm-omni-super.yaml` (EKS manifest for vLLM-Omni server)
- The data augmentation concept (augment existing demos, actions stay the same)
- Connection to Lab 1 (input) and Lab 4 (output)

### Deprecated (old NIM path that never worked)
- `training/scripts/cosmos_setup.py` (Spot p5 NIM approach — kept for reference but not primary)
- `training/scripts/cosmos3_generate.py` (SageMaker batch runner — future/untested)
- `scripts/cosmos-userdata.sh` (EC2 bootstrap — fragile, driver issues)

### Changes (lab narrative)
| Section | Old | New |
|---------|-----|-----|
| Title | "Cosmos World Generation" | "Cosmos Data Augmentation" |
| Infra | EC2 Spot p5 + NIM container (never worked) | EKS Job + vLLM-Omni (validated, published) |
| Model | Cosmos Transfer 2.5 (deprecated) | Cosmos 3 Super 64B (current) |
| API | `POST /v1/infer` with base64 video + edge control | `curl -F input_reference=@clip.mp4 /v1/videos/sync` |
| Demo | Never completed | One clip in, one clip out, visual comparison |
| Input | "Base USD scenes from Isaac Lab" | MP4 wrist camera clip from Lab 1 dataset |
| Output | "USD scenes stored in S3 for Lab 4 backgrounds" | Augmented MP4 clips → merged into LeRobot dataset → retrain |
| Cost | ~$7-8/hr Spot p5 (theoretical) | ~$98/hr p5en EKS (real) |

---

## New Pipeline Narrative

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Cosmos Data Augmentation Pipeline                                            │
│                                                                              │
│  ┌──────────────┐    ┌───────────────┐    ┌───────────────┐    ┌──────────┐ │
│  │ Lab 1 demos  │    │               │    │  Augmented    │    │ Re-train │ │
│  │ (MP4 wrist   │───▶│ Cosmos        │───▶│  MP4 clips    │───▶│ GR00T    │ │
│  │  camera)     │    │ Transfer 2.5  │    │  (5 styles ×  │    │ (Lab 1)  │ │
│  ├──────────────┤    │               │    │   each clip)  │    │   or     │ │
│  │ Lab 2 sim    │───▶│ POST /v1/infer│    │               │    │ RL       │ │
│  │ renders      │    │ on Spot p5    │    │  Same actions, │    │ (Lab 4)  │ │
│  │ (MP4 clips)  │    │ (8x H100)    │    │  new visuals  │    │          │ │
│  └──────────────┘    └───────────────┘    └───────────────┘    └──────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Tasks

- [x] Research best practices (NVIDIA docs, Cosmos Cookbook, SO-101, awsome-distributed-ai)
- [x] Identify framing gap (current lab vs. best practice)
- [x] Pull latest from repo (Adam/Ignacio updates to Labs 1, 2, 4)
- [x] Review updated cosmos_setup.py and cosmos3_generate.py
- [x] Read awsome-distributed-ai code (Dockerfile, generate manifest, env_vars, build-push.sh)
- [x] Write this plan doc (LAB3_PLAN.md)
- [x] Rewrite lab-3-cosmos-world-generation.md (vLLM-Omni / EKS primary path)
- [x] Verify consistency with updated Labs 1, 2, 4 (CLI patterns, bucket refs, etc.)
- [ ] Add `kubernetes/generate-vllm-omni-super.yaml` manifest to repo
- [ ] Test: mirror vllm-omni image to ECR
- [ ] Test: deploy manifest on EKS, verify health endpoint
- [ ] Test: POST one clip, verify output MP4
- [ ] Upload a pre-generated augmented sample to S3 (workshop fallback)
- [ ] (Optional) Write a merge script to automate dataset augmentation
- [ ] (Optional) Mark `cosmos_setup.py` as deprecated in its docstring
- [ ] Update workshop/README.md cost estimate (done: ~$30-50 → realistic ~$50-100)

---

## References

- NVIDIA Cosmos Transfer 2.5: https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html
- NVIDIA Physical AI Workflow: https://docs.nvidia.com/enterprise-reference-architectures/physical-ai-reference-architecture-deployment-guide/latest/physical-ai-workflow.html
- SO-101 Sim-to-Real (Strategy 3): https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html
- Cosmos Cookbook: https://github.com/nvidia-cosmos/cosmos-cookbook
- awsome-distributed-ai Cosmos 3: https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/pytorch/cosmos3
- GR00T-Mimic Blueprint: https://build.nvidia.com/nvidia/isaac-gr00t-synthetic-manipulation/blueprintcard
- Our deployment runbook: docs/cosmos-deployment-guide.md
