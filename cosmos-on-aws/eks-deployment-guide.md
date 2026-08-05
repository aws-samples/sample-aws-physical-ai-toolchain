# Cosmos 3 on EKS — Deployment Guide

*Status: planned — not yet validated end-to-end.*

Deploy the Cosmos3-Super V2V generation server on Amazon EKS for production-scale, multi-replica generation with shared storage. This is the production counterpart to the [EC2 Deployment Guide](ec2-deployment-guide.md).

See [`README.md`](README.md) for the overview and the EC2-vs-EKS comparison.

---

## Why EKS (vs EC2)

- **Multiple concurrent generation jobs.** Run several Cosmos3-Super replicas in parallel across a GPU node pool instead of one instance handling requests serially.
- **Shared persistent storage.** FSx for Lustre shared across replicas — no per-instance HuggingFace cache duplication, shared output directory for generated videos.
- **Fits a continuous flywheel.** If generation becomes an ongoing pipeline (not a one-off experiment), Kubernetes gives you scheduling, retries, and scaling primitives EC2 doesn't.
- **Integrates with a broader orchestration layer.** If your training/simulation pipeline already runs on EKS (see [OSMO](../osmo-on-aws/)), running Cosmos 3 generation as Kubernetes Jobs keeps everything in one control plane.

Use the [EC2 guide](ec2-deployment-guide.md) instead if you're experimenting, running a single workshop session, or don't need more than one generation job at a time.

---

## Starting Point

A validated-pattern Kubernetes manifest already exists at [`kubernetes/generate-vllm-omni-super.yaml`](../kubernetes/generate-vllm-omni-super.yaml), adapted from the [awslabs/awsome-distributed-ai](https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/pytorch/cosmos3) reference (MIT-0). It defines:

- A `Job` running `vllm/vllm-omni:cosmos3` with the same `--cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8` parallelism validated on EC2
- A `Service` exposing port 8000 for the `/v1/videos/sync` generation endpoint
- `nodeSelector` targeting a GPU instance type (parameterized via `${INSTANCE_TYPE}`, e.g. `p5en.48xlarge`)
- `emptyDir` volumes for workspace and `/dev/shm` (see [Open Work](#open-work) below — this should move to FSx for a real shared-storage deployment)

Render it with `envsubst` (the `a8m/envsubst` variant, not GNU gettext's — see the header comment in the manifest for the install command):

```bash
export NAMESPACE=default IMAGE_URI=<ECR_URI>/vllm-omni:cosmos3 \
       INSTANCE_TYPE=p5en.48xlarge GPUS_PER_NODE=8 \
       CFG_PARALLEL=2 ULYSSES=4 HSDP_SHARD=8 \
       HF_HOME=/fsx/hf-cache OUT=/fsx/results/generate

kubectl create secret generic hf-token -n $NAMESPACE --from-literal=token="$HF_TOKEN"
envsubst < kubernetes/generate-vllm-omni-super.yaml | kubectl apply -f -
kubectl port-forward svc/cosmos3-vllm-omni 8000:8000
curl -F input_reference=@clip.mp4 http://localhost:8000/v1/videos/sync -o out.mp4
```

---

## Open Work

This path has **not been run end-to-end** on this project's infrastructure. Before treating it as validated, the following need to be worked out and tested:

1. **EKS cluster with a GPU node pool** — sized for `p5.48xlarge` / `p5en.48xlarge` (8x GPU nodes), with the NVIDIA device plugin installed. No Terraform/CDK for this exists in this repo yet — [`osmo-on-aws/`](../osmo-on-aws/) has an EKS cluster pattern that could be adapted, but Cosmos 3's GPU node requirements should be scoped separately.
2. **FSx for Lustre** — for shared `HF_HOME` (avoid re-downloading ~126GB of weights per replica) and shared output directory. The current manifest uses `emptyDir`, which does NOT persist or share across pods — this needs to change to an FSx-backed PVC for a real multi-replica deployment.
3. **Capacity for P5/P5en on EKS-managed node groups** — same Capacity Block constraints as EC2 apply; a managed node group or Karpenter provisioner needs to target a Capacity Block or on-demand capacity reservation, not just an instance type.
4. **IAM for pod-level S3/Secrets Manager access** — IRSA (IAM Roles for Service Accounts) setup for pods to read the HF token from Secrets Manager and write generated videos to S3, mirroring what the EC2 instance profile does today.
5. **Multi-replica request routing** — if running multiple Cosmos3-Super replicas, decide how generation requests get distributed (a `Service` with multiple backing pods needs each pod to hold a full 8-GPU replica; validate whether standard k8s load balancing works cleanly given each generation is stateful and long-running).
6. **Validate the manifest itself** — it's adapted from an external reference and has not been applied/tested against a real cluster in this project.

---

## Next Steps

Once a GPU-enabled EKS cluster is available:
1. Deploy the NVIDIA device plugin and confirm GPU nodes are schedulable
2. Provision FSx for Lustre and update the manifest's volumes accordingly
3. Set up IRSA for the pod's S3 + Secrets Manager access
4. Apply the manifest, port-forward, and repeat the same [validation steps used on EC2](ec2-deployment-guide.md#step-4b-validate-the-output-in-s3) (S3 existence check, `ffprobe`, visual comparison) to confirm parity with the EC2 path
5. Update this guide with validated timings, costs, and troubleshooting once end-to-end testing is complete
