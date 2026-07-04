# Lab 3 — Cosmos Transfer: Photorealistic Scene Generation

Generate photorealistic training scenes from Isaac Lab sim footage using NVIDIA
Cosmos Transfer 2.5-2B. This lab is optional — Lab 4 works without it.

---

## What this lab does

Lab 3 launches an EC2 `p5.48xlarge` Spot instance (8× H100) and runs NVIDIA's
Cosmos Transfer NIM to restyle sim-rendered video clips into photorealistic
training data. It bridges the sim-to-real visual gap for RL policies trained in
Lab 4. The lab also includes an experimental Lab 3b notebook for Cosmos 3
Generator (text-to-video), which produces entirely new trajectories.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Lab3_Cosmos_World_Generation.ipynb` | Launches a p5.48xlarge Spot instance, waits for the Cosmos Transfer NIM to become ready (~23 min), sends input video clips for restyling, and uploads output to S3. |
| `Lab3b_Cosmos3_World_Generation.ipynb` | Experimental: generates new pick-and-place trajectories from Lab 1 seed frames using Cosmos 3 Generator (not yet validated end-to-end). |

## Supporting scripts

| Script | Description |
|--------|-------------|
| `test_cosmos_e2e.py` | End-to-end test runner for Cosmos Transfer (launch → NIM → infer → terminate). |
| `fix_cosmos_nim.py` | One-shot fix script to pull and restart the correct NIM image on a running instance. |
| `restart_nim.py` | Restarts the Cosmos NIM container with the correct H100 fp8 latency profile. |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| AWS account | With EC2, SSM, Secrets Manager, and S3 access |
| `p5.48xlarge` Spot quota | Service Quotas console → EC2 → "Running On-Demand P instances" (or Spot quota) |
| NGC API key | Must be an `nvapi-...` Personal Key from https://org.ngc.nvidia.com/setup/personal-keys with **NGC Catalog** scope. Store in Secrets Manager as `ngc-api-key`. |
| IAM instance profile | `physical-ai-dev-cosmos-profile` with ECR pull + Secrets Manager read |

---

## Dependencies

Lab 3 has no hard dependency on other labs but works best with Lab 4:
- Lab 3 output (photorealistic MP4s) can feed Lab 4 as domain randomization input.
- Lab 4 works without Lab 3 using Isaac Lab's built-in domain randomization.

---

## Instance types launched

| Resource | Instance | Cost |
|----------|----------|------|
| EC2 Spot | `p5.48xlarge` (8× H100 80 GB) | ~$13.58/hr (us-east-2a) |

---

## Key notes

- **Always terminate the instance when done** — ~$13.58/hr.
- Bootstrap takes ~23 min (NGC pull ~15 min + model load ~8 min).
- Validated fast test: `num_steps=5`, `resolution=256` → ~47 seconds per 100-frame clip.
- Production quality: `num_steps=35`, `resolution=480`.
- Add Cosmos only if sim-to-real transfer fails on real hardware due to visual domain gap.
