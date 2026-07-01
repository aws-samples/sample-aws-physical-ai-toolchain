# Lab 6: OSMO Orchestration

> **Status:** This lab is a placeholder. Contributions welcome.

**Goal:** Use NVIDIA OSMO to orchestrate multi-stage Physical AI workflows across cloud and edge
**Time:** 2-3 hours
**Cost:** ~$50-100 (EKS cluster + GPU nodes for orchestrated training)

---

## What You're Building

OSMO (Orchestration System for Machine Operations) coordinates the full Physical AI pipeline as a single managed workflow:

1. **Data ingestion** — Teleop recordings arrive in S3
2. **Training** — GR00T fine-tuning on SageMaker (Lab 1)
3. **Simulation** — Isaac Lab RL refinement (Lab 3)
4. **Evaluation** — Automated sim rollouts with pass/fail gates
5. **Deployment** — Push to edge fleet via Greengrass (Lab 4)

Without OSMO, you run each stage manually. With OSMO, you trigger a single workflow and it handles sequencing, resource scheduling, failure recovery, and multi-node GPU orchestration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  NVIDIA OSMO (on EKS)                                           │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Stage 1  │───▶│  Stage 2  │───▶│  Stage 3  │───▶│  Stage 4  │ │
│  │  Data     │    │  GR00T   │    │  Isaac   │    │  Deploy  │ │
│  │  Prep     │    │  Train   │    │  Lab RL  │    │  (edge)  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │                │                │                │      │
│       ▼                ▼                ▼                ▼      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Kubernetes GPU Scheduling (Kueue + DRA)                 │  │
│  │  - Allocates GPU nodes on demand                          │  │
│  │  - Multi-node training when needed                        │  │
│  │  - Preemption and priority queues                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │   S3    │         │SageMaker│         │   IoT   │
    │(datasets│         │(training│         │Greengrass│
    │ models) │         │  jobs)  │         │ (deploy)│
    └─────────┘         └─────────┘         └─────────┘
```

---

## Why OSMO

| Without OSMO | With OSMO |
|-------------|-----------|
| Run each lab manually in sequence | Single trigger runs the full pipeline |
| Manually check if training converged before proceeding | Quality gates auto-evaluate and block bad models |
| Restart failed stages from scratch | Automatic retry with checkpointing |
| One training run at a time | Queue multiple experiments with priority scheduling |
| Manual GPU allocation | Dynamic GPU scheduling across cluster |
| No visibility into pipeline status | Dashboard shows all stages, logs, metrics |

**When to use OSMO:** Once you've validated each stage individually (Labs 1-4), OSMO ties them together for production. Think of it as "CI/CD for robot learning."

---

## Prerequisites

- Labs 1-4 completed and validated individually
- EKS cluster deployed (CDK stack: `PhysicalAi-dev-Eks`)
- Helm 3 installed
- NVIDIA GPU Operator installed on EKS
- OSMO Helm chart access (requires NVIDIA Enterprise license or eval)

---

## Steps (placeholder — to be detailed)

### Step 1: Deploy OSMO on EKS

```bash
# Add NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install OSMO
helm install osmo nvidia/osmo \
  --namespace osmo-system \
  --create-namespace \
  --values osmo_values.yaml
```

### Step 2: Define the Pipeline

```yaml
# workflows/osmo-pipeline.yaml
apiVersion: osmo.nvidia.com/v1
kind: Workflow
metadata:
  name: physical-ai-training-pipeline
spec:
  stages:
    - name: data-prep
      container: physical-ai/data-prep:latest
      resources:
        cpu: 4
        memory: 16Gi
      inputs:
        - s3://bucket/raw-teleop-data/

    - name: groot-finetune
      container: physical-ai/groot-training:latest
      resources:
        gpu: 1
        gpu_type: A10G
      depends_on: [data-prep]
      params:
        epochs: 10
        batch_size: 32

    - name: rl-refinement
      container: physical-ai/isaac-lab:latest
      resources:
        gpu: 1
        gpu_type: A10G
      depends_on: [groot-finetune]
      params:
        task: PickAndPlaceUR3-v0
        num_envs: 4096
        max_iterations: 500

    - name: evaluation
      container: physical-ai/isaac-lab:latest
      resources:
        gpu: 1
      depends_on: [rl-refinement]
      quality_gate:
        metric: success_rate
        threshold: 0.90
        action_on_fail: block

    - name: deploy-edge
      container: physical-ai/deployer:latest
      depends_on: [evaluation]
      params:
        target_group: ur3-robots
        rollout_strategy: canary
```

### Step 3: Submit the Workflow

```bash
osmoctl submit workflows/osmo-pipeline.yaml \
  --param data_path=s3://bucket/new-teleop-recordings/ \
  --param experiment_name=ur3-pickplace-v2
```

### Step 4: Monitor Progress

```bash
# CLI
osmoctl status physical-ai-training-pipeline

# Or Kubernetes dashboard
kubectl port-forward svc/osmo-dashboard 8080:80 -n osmo-system
# Open http://localhost:8080
```

### Step 5: Iterate

```bash
# Re-run with different hyperparameters
osmoctl submit workflows/osmo-pipeline.yaml \
  --param num_envs=8192 \
  --param max_iterations=1000 \
  --param experiment_name=ur3-pickplace-v3

# Compare experiments
osmoctl compare ur3-pickplace-v2 ur3-pickplace-v3
```

---

## Integration with AWS Services

OSMO on EKS integrates with:

| AWS Service | Role in Pipeline |
|-------------|-----------------|
| S3 | Dataset storage, model artifacts, checkpoints |
| ECR | Container images for each pipeline stage |
| SageMaker | Can delegate training to SageMaker jobs (hybrid mode) |
| IoT Greengrass | Edge deployment target |
| CloudWatch | Pipeline metrics, logs, alarms |
| EventBridge | Trigger pipelines on S3 uploads or schedules |

---

## Cost Considerations

| Component | Cost | Notes |
|-----------|------|-------|
| EKS control plane | $0.10/hr ($73/month) | Always on |
| GPU nodes (g5.xlarge) | $1.41/hr per node | Only when training — autoscale to zero |
| OSMO itself | Included with NVIDIA AI Enterprise | Requires license |
| S3 storage | ~$0.023/GB/month | Datasets + checkpoints |

**Cost optimization:**
- Use Karpenter to autoscale GPU nodes to zero when idle
- Spot instances for non-critical training (saves ~70%)
- Schedule training during off-peak hours (lower Spot prices)

---

## ✅ Lab 5 Checkpoint

- [ ] OSMO deployed on EKS cluster
- [ ] Pipeline definition created with all stages
- [ ] Submitted a full pipeline run successfully
- [ ] Quality gate blocked a bad model (tested with low iteration count)
- [ ] Monitored pipeline progress via dashboard
- [ ] Understand how to iterate on experiments

---

## When to Use OSMO vs. SageMaker Pipelines

| Criteria | SageMaker Pipelines | OSMO on EKS |
|----------|--------------------:|------------:|
| Simple train/eval/register | ✅ Best choice | Overkill |
| Multi-node GPU training | ✅ Built-in | ✅ Built-in |
| Isaac Lab simulation stages | ❌ Not native | ✅ Native GPU scheduling |
| Mixed CPU/GPU workflows | Limited | ✅ Full K8s scheduling |
| Edge deployment integration | Manual | ✅ Pipeline stage |
| NVIDIA ecosystem tools | Separate | ✅ Integrated |
| No K8s expertise needed | ✅ Managed | Requires K8s knowledge |

**Our recommendation:** Use SageMaker Pipelines (Lab 1 already does this) for the GR00T training stage. Use OSMO when you need Isaac Lab simulation + multi-stage orchestration + edge deployment as a unified pipeline.

---

**Previous:** [← Lab 5: Edge Deployment](lab-5-edge-deployment.md)
