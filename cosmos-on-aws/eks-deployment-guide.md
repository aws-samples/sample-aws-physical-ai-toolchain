# Cosmos 3 on EKS — Deployment Guide

Deploy the Cosmos3-Super V2V generation server on a dedicated Amazon EKS cluster. **Validated end-to-end**, including fully automatic GPU node scaling — a real UR3 clip generated a synthetic video after Cluster Autoscaler brought up a GPU node group launched into an EC2 Capacity Block, then scaled it back down automatically when idle. Output matched the EC2 path.

See [`README.md`](README.md) for the overview and the EC2-vs-EKS comparison.

---

## Why EKS (vs EC2)

- **Multiple concurrent generation jobs.** Run several Cosmos3-Super replicas in parallel across a GPU node pool instead of one instance handling requests serially.
- **Fits a continuous flywheel.** If generation becomes an ongoing pipeline (not a one-off experiment), Kubernetes gives you scheduling, retries, and scaling primitives EC2 doesn't.
- **Integrates with a broader orchestration layer.** If your training/simulation pipeline already runs on EKS (see [OSMO](../osmo-on-aws/)), running Cosmos 3 generation as Kubernetes Jobs keeps everything in one control plane.

Use the [EC2 guide](ec2-deployment-guide.md) instead if you're experimenting, running a single workshop session, or don't need more than one generation job at a time.

**Current scope:** this guide covers a single GPU node running one Cosmos3-Super replica — the same 8-GPU-per-replica model validated on EC2, just orchestrated by Kubernetes instead of a raw EC2 instance. Multi-replica scaling and shared FSx storage are documented as [Future Work](#future-work-multi-replica--shared-storage) — they add real cost/complexity that only pays off once you need concurrent generation jobs.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Dedicated EKS cluster (physical-ai-dev-cosmos3)                          │
│                                                                              │
│   ┌─────────────────────┐    ┌──────────────────────────────────────┐    │
│   │ System node group    │    │ GPU node group (p5.48xlarge)          │    │
│   │ t3.medium, always on │    │ desired=0 by default                  │    │
│   │ CoreDNS, addons,      │    │ Cluster Autoscaler scales 0→1 on a    │    │
│   │ Cluster Autoscaler    │    │ pending pod, 1→0 after ~10 min idle   │    │
│   └─────────────────────┘    │ Launched into an EC2 Capacity Block    │    │
│                                │ via capacity_reservation_specification │    │
│                                │ + instance_market_options              │    │
│                                │                                        │    │
│                                │  Job: cosmos3-vllm-omni                │    │
│                                │  ┌──────────────────────────────┐    │    │
│                                │  │ vllm/vllm-omni:cosmos3        │    │    │
│                                │  │ vllm serve nvidia/Cosmos3-Super│    │    │
│                                │  │ --cfg-parallel-size 2          │    │    │
│                                │  │ --ulysses-degree 4             │    │    │
│                                │  │ --use-hsdp --hsdp-shard-size 8 │    │    │
│                                │  └──────────────────────────────┘    │    │
│                                │  ServiceAccount: cosmos3-generator     │    │
│                                │  (IRSA → S3 write + Secrets read)     │    │
│                                └──────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │ kubectl port-forward svc/cosmos3-vllm-omni 8000:8000
                           ▼
                    POST /v1/videos/sync
                    -F input_reference=@your_ur3_clip.mp4
                           │
                           ▼
                    S3 (cosmos-samples/)
```

---

## Prerequisites

Same as the [EC2 guide's prerequisites](ec2-deployment-guide.md#prerequisites) — Foundation deployed, an active P5 Capacity Block, HuggingFace token with both gated licenses accepted, stored in Secrets Manager. Additionally:

| Requirement | Why |
|-------------|-----|
| `kubectl` installed | Manage the cluster |
| AWS provider `< 6.0.0` pinned | The `terraform-aws-modules/eks/aws` module requires it — `terraform init -upgrade` handles this automatically |

---

## Step 1: Reserve a Capacity Block

Same as [EC2 guide Step 1](ec2-deployment-guide.md#step-1-reserve-a-capacity-block). Note the AZ — the GPU node group is pinned to it (see Step 2).

> **p5.48xlarge Capacity Blocks are 24-hour minimum.** If none are available at that duration, check for shorter/nearer-term offerings across durations — availability and pricing vary significantly by start time.

---

## Step 2: Deploy the EKS Cluster

The cluster is defined in [`infra/eks.tf`](infra/eks.tf), gated behind `enable_eks_cluster` (default `false`) — safe to leave in any `terraform apply` since it adds zero resources when disabled. **Once created, the control plane bills continuously (~$0.10/hr) until destroyed**, independent of GPU node scaling.

```bash
cd cosmos-on-aws/infra
terraform init -upgrade   # pulls the eks module + downgrades the AWS provider to <6.0.0
terraform apply \
  -var="aws_region=<REGION>" \
  -var="enable_eks_cluster=true"
```

This creates:
- The EKS control plane + OIDC provider
- A `system` node group (`t3.medium`, always on — runs CoreDNS/addons)
- A `gpu` node group (`p5.48xlarge`, **`desired_size=0` by default** — no GPU cost while idle)
- Addons: CoreDNS, kube-proxy, VPC CNI, EBS CSI driver
- **Cluster Autoscaler** (Helm release) — automatically scales the GPU node group 0→1 when a Job needs it and back to 0 when idle. See [Step 4](#step-4-how-autoscaling-works-no-manual-steps-needed).
- IRSA role for a `cosmos3-generator` service account (S3 write to the datasets bucket, Secrets Manager read for the HF token)

```bash
aws eks update-kubeconfig --name $(terraform output -raw eks_cluster_name) --region <REGION>
kubectl get nodes
```

You should see one `t3.medium` node, `Ready`.

### Install the NVIDIA device plugin (one-time, cluster-wide)

The GPU node's AMI (`AL2_x86_64_GPU`) has drivers baked in, but Kubernetes still needs the device plugin to expose GPUs as an allocatable resource:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

This runs as a daemonset with a built-in toleration for the GPU taint, so it schedules automatically once a GPU node joins.

---

## Step 3: Create the ServiceAccount

The IRSA role was created by Terraform in Step 2; bind it to a Kubernetes ServiceAccount:

```bash
IRSA_ARN=$(cd cosmos-on-aws/infra && terraform output -raw eks_pod_irsa_role_arn)

kubectl create serviceaccount cosmos3-generator -n default --dry-run=client -o yaml | \
  kubectl annotate -f - --local -o yaml eks.amazonaws.com/role-arn="$IRSA_ARN" | \
  kubectl apply -f -
```

Create the HF token secret in the cluster:

```bash
HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id physical-ai/hf-token \
  --region <REGION> --query 'SecretString' --output text)
kubectl create secret generic hf-token -n default --from-literal=token="$HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## Step 4: How Autoscaling Works (No Manual Steps Needed)

**Validated end-to-end.** Cluster Autoscaler is deployed by Terraform in Step 2 and handles GPU node scaling automatically — you don't need to run `aws eks update-nodegroup-config` at all in normal operation:

- **Scale up:** when you `kubectl apply` the Job (Step 5) and its pod can't schedule (no GPU node exists), Cluster Autoscaler detects the pending pod within ~1 min, matches its `nvidia.com/gpu: 8` request against the GPU node group's node-template tags, and scales the ASG from 0→1.
- **Scale down:** once you delete the Job (Step 8) and the GPU node sits idle past `scale-down-unneeded-time` (10 min, configured in `eks.tf`), Cluster Autoscaler removes the node automatically — no action needed.

**Validated timings:** pending pod → ASG desired=1 in ~44s; idle node → scaled back to 0 in ~14 min after Job deletion.

This works because the GPU node group carries `k8s.io/cluster-autoscaler/node-template/*` tags on its ASG (set in `eks.tf`) that tell Cluster Autoscaler what a *not-yet-existing* node would offer — its GPU count, labels, and taints — so it can evaluate whether a pending pod would fit before actually launching anything. Without these tags, autoscaler can't reason about a node group sitting at 0 instances and will never scale it up.

### First-time / Capacity Block setup

Autoscaling only works once the GPU node group's launch template is correctly targeting an active Capacity Block. Set this via Terraform:

```bash
terraform apply \
  -var="aws_region=<REGION>" -var="enable_eks_cluster=true" \
  -var="eks_gpu_capacity_reservation_id=<CR_ID>" \
  -var="eks_gpu_availability_zone=<AZ>"
```

**Switching to a new Capacity Block after the previous one expires:** re-run the same command with the new `eks_gpu_capacity_reservation_id`. This updates the launch template in place (no node group replacement) as long as the AZ hasn't changed. If the GPU node group has a running instance at the time (`capacity_type=CAPACITY_BLOCK`), the update will fail — scale it to 0 first:
```bash
aws eks update-nodegroup-config --cluster-name <CLUSTER_NAME> \
  --nodegroup-name <GPU_NODEGROUP_NAME> \
  --scaling-config minSize=0,maxSize=1,desiredSize=0 \
  --region <REGION>
# wait for the instance to terminate (can take several minutes for p5.48xlarge), then re-apply
```

### If you ever need to scale manually (bypassing the autoscaler)

For debugging or to pre-warm a node before a Job, you can still scale directly:
```bash
aws eks update-nodegroup-config --cluster-name <CLUSTER_NAME> \
  --nodegroup-name <GPU_NODEGROUP_NAME> \
  --scaling-config minSize=0,maxSize=1,desiredSize=1 \
  --region <REGION>
```
Get the exact node group name (EKS appends a random suffix): `aws eks list-nodegroups --cluster-name <CLUSTER_NAME> --region <REGION>`.

Confirm 8 GPUs are exposed once the node is `Ready`:
```bash
kubectl get node <GPU_NODE_NAME> -o jsonpath='{.status.capacity.nvidia\.com/gpu}'
```

### Terraform gotchas hit during validation (already fixed in `eks.tf`, documented for context)

1. **AZ mismatch.** EKS node groups by default span all private subnets across AZs. A Capacity Block is reserved in exactly one AZ — if the node group launches in a different AZ than the reservation, EC2 rejects it: `Capacity Reservation's attribute does not match with requested instance parameter. Attribute: AvailabilityZone`. Fix: `eks.tf` pins the GPU node group's `subnet_ids` to a single AZ-filtered subnet via `eks_gpu_availability_zone`.
2. **Missing `instance_market_options`.** `capacity_reservation_specification` alone is not sufficient for Capacity Block launches — the launch template also needs `instance_market_options { market_type = "capacity-block" }`, or the API rejects the node group with `market type (purchasing) option is not valid`. Fix: `eks.tf` sets both together whenever a Capacity Block ID is provided.
3. **Updating a `CAPACITY_BLOCK` node group while it has a running instance fails.** `AutoScalingGroupInvalidConfiguration: Upgrade of the node group is not allowed when the current capacity of the Auto Scaling group is not zero and the capacity type is set to CAPACITY_BLOCK.` Scale to 0 and wait for termination before changing the launch template (e.g. pointing at a new Capacity Block ID).
4. **Autoscaler can't scale a zero-node group without node-template tags.** Without `k8s.io/cluster-autoscaler/node-template/resources/nvidia.com/gpu` (and the label/taint equivalents) on the ASG, Cluster Autoscaler has no way to know what a not-yet-launched node offers, and silently never scales up. Fix: `eks.tf` sets these tags directly on the GPU node group.

---

## Step 5: Deploy the Cosmos3 Job

The Job manifest (`cosmos-on-aws/cosmos3-job.yaml`) runs the same server configuration validated on EC2 — `vllm/vllm-omni:cosmos3` serving `nvidia/Cosmos3-Super` with `--cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8` — as a Kubernetes `Job` + `Service`, using `emptyDir` for the model weight cache (re-downloaded per pod; see [Future Work](#future-work-multi-replica--shared-storage) for shared caching) and the `cosmos3-generator` ServiceAccount for IRSA.

```bash
cd cosmos-on-aws
kubectl apply -f cosmos3-job.yaml
```

Watch the pod schedule onto the GPU node and start:
```bash
kubectl get pods -l app=cosmos3-vllm-omni -o wide
kubectl logs -l app=cosmos3-vllm-omni -f
```

**Model weight download is ~126 GB from HuggingFace** — same as EC2, takes 15-20 min on first pull since the `emptyDir` cache doesn't persist across pod restarts.

Port-forward to reach the server (do this in a background terminal or separate session):
```bash
kubectl port-forward svc/cosmos3-vllm-omni 8000:8000
```

Check readiness:
```bash
curl -s http://localhost:8000/v1/models
```
Ready when it returns `{"data":[{"id":"nvidia/Cosmos3-Super",...}]}`.

**Validated timing (including autoscaler-driven node launch):** pod `Pending` → `Running` in ~9 min (autoscaler scale-up ~44s + node join/join-ready ~2-3 min + image pull ~5-6 min); weight download + model init added another ~15 min on top, consistent with the EC2 path.

> **Avoid `kubectl exec -it` in non-interactive/scripted environments** (CI, automation, or a tool-driven session without a real TTY) — it panics with a Go runtime nil-pointer crash trying to manage terminal resize. Use `kubectl port-forward` + `curl` from outside the pod instead, as shown above; it doesn't need a TTY and is what this guide validated against.

---

## Step 6: Generate a World from Your UR3 Data

With the port-forward running, generate directly against `localhost:8000` — no need to copy the reference video into the pod first (unlike the EC2/Docker path):

```bash
cd cosmos-on-aws
aws s3 cp s3://<DATASETS_BUCKET>/groot-data/ur3/dataset/videos/chunk-000/observation.images.wrist/episode_000000.mp4 \
  ./episode_000000.mp4 --region <REGION>

PROMPT="A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small red wooden block, lifts it slowly, and places it onto a yellow sticky note approximately 6 inches away. Top-down wrist camera view. Colorful wooden blocks are scattered on the table. Smooth deliberate motion."

curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" \
  -F "prompt=${PROMPT}" \
  -F "size=1280x720" \
  -F "num_frames=189" \
  -F "fps=24" \
  -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" \
  -F "flow_shift=10.0" \
  -F "extra_params={\"condition_frame_indexes_vision\":[0,1],\"condition_video_keep\":\"first\"}" \
  -F "seed=100" \
  -F "input_reference=@episode_000000.mp4;type=video/mp4" \
  -o output_eks_v2v.mp4 \
  --max-time 600 \
  -w "\nHTTP:%{http_code} SIZE:%{size_download}\n"
```

Same required parameters as the [EC2 path](ec2-deployment-guide.md#step-4-generate-a-world-from-your-ur3-data) — this is the same server, just reached via `kubectl port-forward` instead of SSM.

**Validated result:** HTTP 200, 6.1 MB output, generated in a few minutes with a warm model — matching the EC2 path's output size closely (5.9-6.1 MB range across both paths).

---

## Step 7: Validate the Output

Same validation approach as [EC2 Step 4b](ec2-deployment-guide.md#step-4b-validate-the-output-in-s3):

```bash
ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,codec_name \
  output_eks_v2v.mp4
```

Expect `width=1280`, `height=720`, `codec_name=h264`, `duration≈7.875` (189 frames at 24fps). **Validated exactly this on the EKS path.**

Upload to S3 and confirm:
```bash
aws s3 cp output_eks_v2v.mp4 s3://<DATASETS_BUCKET>/cosmos-samples/augmented_episode_000000_eks.mp4 --region <REGION>
aws s3 ls s3://<DATASETS_BUCKET>/cosmos-samples/ --region <REGION> --human-readable
```

---

## Step 8: Tear Down

**Delete the Job** when done generating — Cluster Autoscaler takes care of scaling the GPU node back to 0 automatically (~10-15 min after the node goes idle, no action needed):
```bash
kubectl delete -f cosmos-on-aws/cosmos3-job.yaml
```

Confirm it scaled down (optional):
```bash
watch -n 30 "aws eks describe-nodegroup --cluster-name <CLUSTER_NAME> --nodegroup-name <GPU_NODEGROUP_NAME> --region <REGION> --query 'nodegroup.scalingConfig.desiredSize'"
```

If you need the GPU node gone immediately (don't want to wait for the idle timer), scale it manually instead:
```bash
aws eks update-nodegroup-config --cluster-name <CLUSTER_NAME> \
  --nodegroup-name <GPU_NODEGROUP_NAME> \
  --scaling-config minSize=0,maxSize=1,desiredSize=0 \
  --region <REGION>
```

**Delete the entire cluster** when you're done with EKS entirely (stops control-plane billing too):
```bash
cd cosmos-on-aws/infra
terraform destroy -var="aws_region=<REGION>" -var="enable_eks_cluster=true"
```

The Capacity Block itself expires automatically at its end time regardless of node group state.

---

## Cost

| Resource | Cost | Notes |
|----------|------|-------|
| EKS control plane | ~$0.10/hr | Bills continuously once created, independent of node scaling |
| System node (`t3.medium`) | ~$0.04/hr | Always on |
| GPU node (`p5.48xlarge`, Capacity Block) | Same as [EC2 pricing](ec2-deployment-guide.md#cost) | Only while `desired_size=1` |
| Per generation | ~$0 marginal | Same as EC2 — block cost is fixed |

**EKS adds ~$0.14/hr over the EC2 path** for the control plane + system node, whether or not you're generating. Worth it once you need multiple concurrent Jobs; pure overhead for a single one-off generation — use EC2 for that case.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Capacity Reservation's attribute does not match... Attribute: AvailabilityZone` | GPU node group subnets span multiple AZs, Capacity Block is single-AZ | Set `eks_gpu_availability_zone` so the node group only uses that AZ's subnet (already the default behavior in `eks.tf`) |
| `market type (purchasing) option is not valid` | `capacity_reservation_specification` set without `instance_market_options` | Both must be set together — already handled in `eks.tf` when `eks_gpu_capacity_reservation_id` is non-empty |
| `terraform apply` with a new `-var eks_gpu_node_desired_size=1` doesn't scale the node | Module sets `ignore_changes` on `desired_size` (intentional — Cluster Autoscaler owns this after initial creation) | Let Cluster Autoscaler handle it by submitting a Job (Step 5); for manual override use `aws eks update-nodegroup-config` |
| Job's pod stays `Pending` for more than ~2 min with no node appearing | Cluster Autoscaler not running, or missing node-template tags on the ASG | `kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-cluster-autoscaler` — confirm it's `Running`; check its logs for `failed to find place for` messages, which confirm it saw the pod. If it's not scaling, verify the GPU node group's ASG has the `k8s.io/cluster-autoscaler/node-template/*` tags (`eks.tf` sets these) |
| GPU node `Ready` but pod stuck `Pending` | NVIDIA device plugin not installed, or node's GPU capacity not yet advertised | `kubectl apply` the device plugin daemonset; wait ~30s after node `Ready` for capacity to appear |
| GPU node doesn't scale down after deleting the Job | Still within the 10-min `scale-down-unneeded-time` window, or another pod is still using the node | Wait — validated at ~14 min end-to-end from Job deletion to `desiredSize=0`; check `kubectl get pods -A -o wide` for anything still scheduled on the GPU node |
| Pod scheduled but never becomes `Running` / image pull slow | ~30GB `vllm/vllm-omni:cosmos3` pull on first schedule to a fresh node | Normal — same pull time as the EC2 path; check `kubectl describe pod` for pull progress |
| `terraform init` fails with a provider version conflict | EKS module requires AWS provider `< 6.0.0`, lock file has a newer version | `terraform init -upgrade` |
| IAM role name length error on `terraform plan`/`apply` | EKS module appends `-eks-node-group-<suffix>` to node group names; long prefixes exceed IAM's 38-char `name_prefix` limit | Keep node group `name` short (this repo uses `cosmos3-system` / `cosmos3-gpu`, not prefixed with the full stack name) |
| `kubectl exec -it ... -- curl ...` panics with a Go nil-pointer crash | No real TTY available (scripted/automated shell) | Use `kubectl port-forward` + `curl` from outside the pod instead |
| GPU node stuck at `NotReady` after joining | Normal for the first ~10-30s while kubelet/CNI initialize | Wait; check `kubectl describe node` if it persists past a minute |

---

## Future Work: Multi-Replica + Shared Storage

Not needed for a single generation job (what this guide validates), but worth doing if Cosmos 3 becomes a continuous production pipeline:

1. **FSx for Lustre** for shared `HF_HOME` — avoids each new GPU node/pod re-downloading the full ~126GB of model weights. Requires provisioning FSx, mounting it via a PVC instead of `emptyDir`, and IAM for the FSx CSI driver.
2. **Multiple GPU node group replicas** (`max_size > 1`, one Job per replica) for concurrent generation — each replica still needs its own full 8-GPU allocation (Cosmos3-Super's parallelism is within-node, not across nodes), so this is horizontal scaling by replica count, not by making a single generation faster.
3. **Capacity for multiple P5 nodes simultaneously** — either multiple Capacity Blocks or a larger single reservation; `InstanceMatchCriteria=targeted` reservations are for exactly the reserved count.
4. **Request routing across replicas** if exposing a single endpoint for many replicas — standard Kubernetes `Service` load balancing works for stateless HTTP, but each generation call is long-running (minutes) and stateful within a request, so validate behavior under concurrent load before relying on it in production.
