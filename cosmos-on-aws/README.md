# Cosmos 3 on AWS

Generate synthetic robot demonstration videos with [NVIDIA Cosmos 3](https://www.nvidia.com/en-us/ai/cosmos/) (Cosmos3-Super, a World Foundation Model) — conditioned on your own robot data.

Two deployment paths are documented:

| Path | Guide | Best for |
|------|-------|----------|
| **EC2** | [EC2 Deployment Guide](ec2-deployment-guide.md) | Single-node experiments, workshops, one-off generation runs. **Validated end-to-end.** |
| **EKS** | [EKS Deployment Guide](eks-deployment-guide.md) | Production-scale generation, multi-replica flywheel, persistent shared storage. **Planned — not yet validated.** |

Both paths run the same server (`vllm/vllm-omni:cosmos3` serving `nvidia/Cosmos3-Super`) on the same hardware (p5.48xlarge / p5en.48xlarge, 8x H100+). The difference is orchestration: EC2 is a single instance you manage directly; EKS lets you run multiple generation jobs in parallel across a GPU node pool with shared storage (FSx).

---

## Can Cosmos 3 generate world output from UR3 data? — Yes, validated

Cosmos3-Super's **V2V (video-to-video)** mode takes a reference video as the starting condition and generates a new plausible continuation — this works with any input video, including your own UR3 teleoperation footage.

**Validated end-to-end (on EC2):**
- **Input:** `episode_000000.mp4` — a real UR3 wrist-camera pick-and-place episode (862 KB, from the Lab 1 dataset)
- **Output:** A new 189-frame, 1280×720, 24fps synthetic video (6.0 MB) — the UR3 arm performing a novel pick-and-place trajectory, photorealistic quality
- **Generation time:** ~2-8 min on 8x H100 (first request includes JIT warmup)
- **Mechanism:** The first 1-2 frames of your input video anchor the scene (table, blocks, lighting); the text prompt + seed control what action gets generated

**Artifacts in S3 (for your own validation):**
```bash
aws s3 cp s3://physical-ai-dev-datasets-804152302157/cosmos-samples/original_episode_000000.mp4 ./
aws s3 cp s3://physical-ai-dev-datasets-804152302157/cosmos-samples/augmented_episode_000000.mp4 ./
```
Play both side by side — `original_episode_000000.mp4` (861.7 KiB, the real input) vs `augmented_episode_000000.mp4` (5.9 MiB, the Cosmos 3 output) — to see the generated trajectory against the source scene it was conditioned on.

This means: give it any wrist-camera clip from your robot, describe the task in the prompt, and it generates a new synthetic demonstration starting from your actual scene — not a generic stock scene.

**What this is / isn't good for:**
- ✅ Vision pre-training — teach a VLA backbone what your task looks like across many visual variations
- ✅ Dataset diversity — same task, different lighting/block-color/approach-angle/speed
- ✅ Policy evaluation — visually sanity-check whether a generated rollout looks physically plausible
- ❌ **No action labels** — output is video only (no joint positions/velocities). For action-paired augmentation (restyle existing video while preserving the original actions), use Cosmos Transfer 2.5 instead — different model, different workflow.

Full step-by-step generation, validation, and troubleshooting: **[→ EC2 Deployment Guide](ec2-deployment-guide.md)**.

---

## Choosing EC2 vs EKS

- **Start with EC2** if you're experimenting, running a workshop, or need to generate a batch of videos during a single Capacity Block window. Simpler — one instance, no cluster.
- **Move to EKS** when you need multiple concurrent generation jobs (a continuous "flywheel" producing synthetic data), shared FSx storage across replicas, or integration with a broader Kubernetes-orchestrated pipeline (e.g. [OSMO](../osmo-on-aws/)).

---

## Files in This Directory

| File | Purpose |
|------|---------|
| [`ec2-deployment-guide.md`](ec2-deployment-guide.md) | Full EC2 deployment walkthrough — Capacity Block, launch, server setup, generation, S3 validation |
| [`eks-deployment-guide.md`](eks-deployment-guide.md) | EKS deployment plan and open work items (not yet validated) |
| [`launch-cosmos3.sh`](launch-cosmos3.sh) | Launch p5.48xlarge into an active Capacity Block |
| [`setup-cosmos3-server.sh`](setup-cosmos3-server.sh) | Pull the container and start the Cosmos3-Super server |
| [`generate-v2v.sh`](generate-v2v.sh) | Generate a V2V synthetic demonstration from a reference video |
| `infra/` | Terraform for the CodeBuild-based container build path (Cosmos 3 + Cosmos Transfer 2.5) |

---

## Next Steps

- **Action-paired augmentation** — for restyling existing video while preserving ground-truth actions, see Cosmos Transfer 2.5 (`containers/cosmos/`)
- **Feed generated videos into training** — use Cosmos 3 output for vision pre-training or dataset diversity ahead of [GR00T fine-tuning](../isaac-gr00t-on-aws/)
- **Full CLI runbook** — every command run during validation is in [`docs/cosmos3-validated-runbook.md`](../docs/cosmos3-validated-runbook.md)
