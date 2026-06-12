# AWS Physical AI Toolchain — Build & Test Tracker

## Status Key
- ✅ Done
- 🔲 Not started
- 🚧 In progress

---

## Code Completion

### Infrastructure (CDK)
- ✅ `cdk/bin/app.ts` — App entry point, stack wiring
- ✅ `cdk/lib/network-stack.ts` — VPC, subnets, security groups, VPC endpoints
- ✅ `cdk/lib/storage-stack.ts` — S3 buckets, ECR repos
- ✅ `cdk/lib/eks-cluster-stack.ts` — EKS cluster, GPU node pool, autoscaler, NVIDIA plugin
- ✅ `cdk/lib/osmo-stack.ts` — OSMO Helm chart, RDS PostgreSQL, ElastiCache Redis
- ✅ `cdk/lib/edge-stack.ts` — IoT Core, Greengrass components, IAM roles
- ✅ `cdk/config/dev.ts` — Dev environment config (g5.xlarge)
- ✅ `cdk/config/prod.ts` — Prod environment config (P5e)
- ✅ `cdk/package.json` + `tsconfig.json` + `cdk.json`

### Pipeline (OSMO Workflow)
- ✅ `workflows/pick-and-place.yaml` — Full pipeline definition

### Training
- ✅ `training/scripts/train.py` — Training entry point (rl_games PPO runner)
- ✅ `training/scripts/evaluate.py` — Evaluation + video rendering
- ✅ `training/scripts/export.py` — PyTorch → ONNX → TensorRT export + benchmark
- ✅ `training/scripts/generate_scenes.py` — Cosmos + procedural scene generation
- ✅ `training/configs/ppo_pick_place.yaml` — PPO hyperparameters + curriculum + domain rand
- ✅ `training/envs/__init__.py` — Environment registration
- ✅ `training/envs/pick_and_place_ur3.py` — Isaac Lab RL environment (obs/action/reward)

### Containers
- ✅ `containers/isaac-sim/Dockerfile` — Scene generation container (Isaac Sim + Cosmos)
- ✅ `containers/isaac-lab/Dockerfile` — Training container (Isaac Lab + rl_games)
- ✅ `containers/inference/Dockerfile` — Edge inference (x86 + Jetson multi-stage)

### Edge
- ✅ `edge/ros2-workspace/src/ur3_inference/ur3_inference_node.py` — ROS 2 TensorRT inference node
- ✅ `edge/entrypoint.sh` — Container entrypoint (sources ROS 2, launches node)

### Docs
- ✅ `README.md` — Quick start guide
- ✅ `../aws-physical-ai-architecture.md` — Full architecture doc

---

## Testing Phases

### Phase 1: CDK Synth (local, no AWS)
- ✅ `npm install` in cdk/
- ✅ `npx cdk synth` — validates TypeScript compiles and generates valid CloudFormation
- ✅ Fixed: kubectlLayer required for EKS construct
- ✅ Fixed: Cyclic dependency between EKS and OSMO stacks
- **Result:** All 5 stacks synthesize successfully

### Phase 2: Deploy Network + Storage (cheap, fast)
- 🔲 `cdk deploy PhysicalAi-dev-Network PhysicalAi-dev-Storage`
- 🔲 Verify VPC created with correct subnets
- 🔲 Verify S3 buckets and ECR repos exist
- **Expected issues:** Bucket naming conflicts (globally unique), region availability

### Phase 3: Deploy EKS (~20 min)
- ✅ `cdk deploy PhysicalAi-dev-Eks`
- ✅ Cluster is ACTIVE (v1.31)
- ✅ 3 nodes running (2 control + 1 GPU)
- ✅ kubectl access working (via cluster-admin role)
- ✅ GPU AMI includes NVIDIA device plugin (no separate Helm install needed)
- ✅ Fixed: Removed NVIDIA device plugin Helm chart (conflicts with GPU AMI)
- ✅ Fixed: Removed OSMO Helm chart from CDK (repo URL was wrong, install manually)
- ✅ Fixed: Added SteveAdmin IAM user to cluster masters
- ⚠️ Manual fix needed in CDK: EKS-managed cluster SG → RDS/Redis SG rules (done manually, needs automation)

### Phase 4: Deploy OSMO + Edge
- ✅ RDS PostgreSQL deployed and accessible from EKS pods
- ✅ ElastiCache Redis deployed and accessible from EKS pods
- ✅ OSMO Helm repo accessible (`helm.ngc.nvidia.com/nvidia/osmo`)
- ✅ OSMO database created (`osmo_db`)
- ⚠️ OSMO Helm chart values need tuning (postgres password env var injection)
- ❌ Edge stack skipped (Greengrass — not needed for demo)
- **Next:** Follow OSMO minimal deployment guide with correct values structure

### Phase 5: Build & Push Containers
- 🔲 Build isaac-lab container locally (or in CodeBuild)
- 🔲 Push to ECR
- 🔲 Build inference container
- 🔲 Push to ECR
- **Expected issues:** NGC base image pull (needs NGC API key), large image sizes

### Phase 6: Run Training Pipeline
- 🔲 `osmo workflow submit -f workflows/pick-and-place.yaml`
- 🔲 Verify GPU node scales up (cluster autoscaler)
- 🔲 Verify training pod starts and logs show progress
- 🔲 Wait for training to complete (~4-8 hours on g5, faster on P5e)
- 🔲 Verify eval_video.mp4 in S3
- 🔲 Verify eval_metrics.json shows >80% success rate
- **Expected issues:** Container image pull failures, S3 permissions, Isaac Lab env registration, GPU memory

### Phase 7: Edge Deployment (optional — needs hardware)
- 🔲 Provision Jetson/GPU PC as Greengrass core device
- 🔲 Deploy inference component
- 🔲 Verify ROS 2 node publishes joint commands
- 🔲 Connect to UR3 and test real pick-and-place
- **Expected issues:** Network connectivity, ROS 2 driver config, TensorRT version mismatch

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| OSMO Helm chart not publicly available yet | Can't deploy OSMO | Fall back to manual K8s manifests from OSMO GitHub repo |
| P5e instance quota = 0 in account | Can't train at full scale | Use g5.xlarge for dev (slower but works) |
| Isaac Lab UR3 env needs tuning | Policy doesn't converge | Start with Franka (well-tested), port to UR3 after |
| Cosmos API access unclear | Can't generate scenes | Skip Cosmos, use Isaac Sim procedural generation instead |
| OSMO edge operator on Jetson untested | Edge deploy fails | Deploy via Greengrass only (skip OSMO operator on edge) |
