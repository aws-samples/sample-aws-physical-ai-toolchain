# Testing Guide — Fresh Clone Validation

This document captures known issues and areas that need verification when testing the full workshop from a fresh clone in a new AWS account.

## How to Test

1. Fresh AWS account (or `cdk destroy --all` first)
2. Clone the repo
3. Follow labs sequentially
4. Document every failure and fix

---

## Known Issues & Watch Points

### Lab 0: Prerequisites
- **Status:** Should work cleanly
- **Watch:** CDK bootstrap needs to run in `us-east-1` (not us-west-2)

### Lab 1: Train from Demonstrations

| Step | Risk | Issue | Potential Fix |
|------|------|-------|--------------|
| Step 2 (zarr conversion) | Medium | Script needs `zarr`, `numpy`, `opencv-python`, `pandas`, `pyarrow` installed. These aren't in a `requirements.txt` yet. | Add `training/requirements.txt` |
| Step 4 (container build) | Medium | HF_TOKEN must be set as env var before build. Docs mention it but the actual `train_entrypoint.py` reads it from SageMaker env. May need to pass via `--environment` in the training job. | Verify HF_TOKEN flow end-to-end |
| Step 5 (pipeline.py) | Low | Proven to work. But `dataset-prefix` must match exactly (`groot-data/ur3` not `groot-data/ur3/dataset`). | Already documented but easy to get wrong |

### Lab 2: Isaac Sim Workstation

| Step | Risk | Issue | Potential Fix |
|------|------|-------|--------------|
| Deploy | High | `g5.4xlarge` capacity varies by AZ. We hit `InsufficientCapacity` multiple times. | Add fallback logic: try g5.4xlarge → g5.2xlarge → g5.xlarge. Or let user specify via context param. |
| DCV connect | Medium | VPN blocks port 8443. Users must disconnect VPN to access DCV. | Documented in lab, but easy to miss. Add a prominent warning. |
| Isaac Sim launch | Medium | `isaacsim` command works in `~/isaac-env` venv but not globally. User must `source ~/isaac-env/bin/activate` first. | Add to convenience script or .bashrc |

### Lab 3: Cosmos Transfer

| Step | Risk | Issue | Potential Fix |
|------|------|-------|--------------|
| `cosmos_setup.py deploy` | High | **This will fail.** SageMaker endpoint approach doesn't work (driver too old). Script needs a `deploy-ec2` mode for Spot p5. | Add `cosmos_setup.py deploy-spot` command that handles the full EC2 flow |
| EC2 Spot setup | High | Multiple manual SSM commands were needed (driver, fabricmanager, Docker config, container pull). Not automated in one script. | Consolidate into `scripts/cosmos-userdata.sh` and make it robust (it currently has issues with heredocs via SSM) |
| Inference test | Not tested | We confirmed health endpoint but never sent an actual video through. | Need to complete task 11c |

### Lab 4: RL Refinement

| Step | Risk | Issue | Potential Fix |
|------|------|-------|--------------|
| Step 1 (bridge script) | Low | Proven to work. | — |
| Step 2 (understand env) | Medium | References `training/configs/ppo_pick_place.yaml` which may not exist in the repo. | Create the config file or remove the reference |
| UR3 env in Isaac Lab | Not tested | `pick_and_place_ur3.py` was written but never validated in Isaac Lab. The RL job runs the built-in Anymal-D task, not our UR3 env. | Validate on workstation (needs g5 capacity) |

### Labs 5-6: Edge + OSMO
- **Status:** Placeholders only. Not testable without hardware (Lab 5) or EKS cluster (Lab 6).

---

## Quick Fixes Needed Before Clean Testing

Priority order:

1. **Create `training/requirements.txt`** — list all Python dependencies for conversion/pipeline scripts
2. **Add `training/configs/ppo_pick_place.yaml`** — referenced in Lab 4 docs but doesn't exist
3. **Update `cosmos_setup.py`** — add `deploy-spot` mode that automates the EC2 approach
4. **Add `run-path-a.sh`** verification — the quick-start script referenced in README
5. **Test HF_TOKEN flow** — verify the token makes it from local env → SageMaker training job → container

---

## Integration Test Script (Future)

Create `scripts/test-labs.sh` that non-interactively runs:
```bash
# Lab 0: Verify prereqs
# Lab 1: Convert data + run pipeline (100 steps)
# Lab 4: Run bridge script (50 iterations)
# Verify: model in registry, RL artifacts in S3
```

This would catch regressions automatically.

---

## What's Been Proven End-to-End

These specific commands/flows are confirmed working in our account:

```bash
# Zarr conversion (proven)
python training/groot/convert_zarr_to_lerobot.py \
  --episodes-dir training/data/ur3_episodes/episodes \
  --output-dir training/data/ur3_lerobot_dataset

# S3 upload (proven)
aws s3 sync training/data/ur3_lerobot_dataset/ s3://physical-ai-dev-datasets-802782083985/groot-data/ur3/dataset/

# GR00T training (proven — multiple times)
python training/groot/pipeline.py --execute --max-steps 100 --dataset-prefix groot-data/ur3

# Isaac Lab RL (proven — multiple times)
python training/scripts/groot_to_rl_bridge.py end-to-end

# Isaac Lab direct (proven)
aws sagemaker create-training-job with isaac-lab container + Anymal-D task

# Cosmos health (proven)
Spot p5 + fabricmanager + --ipc=host → container health: ready
```
