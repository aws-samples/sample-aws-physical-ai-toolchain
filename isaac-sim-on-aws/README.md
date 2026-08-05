# Isaac Sim on AWS

Deploy [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) on AWS for physics-accurate robot simulation. Isaac Sim provides photorealistic rendering, accurate physics, and sensor simulation for developing and testing robot policies before deploying to real hardware.

---

## Two Modes of Operation

| Mode | Use Case | Guide |
|------|----------|-------|
| **GUI Mode** | Visual development, debugging, scene building, watching policies run | [GUI Mode Guide →](gui-mode-guide.md) |
| **Headless Mode** | Large-scale training, batch rendering, CI/CD validation | [Headless Mode Guide →](headless-mode-guide.md) |

---

## When to Use Each Mode

**GUI Mode** — deploy a GPU workstation with NICE DCV remote desktop:
- Iterate on RL environments visually
- Debug physics issues (objects clipping, falling through surfaces)
- Watch trained policies execute in real-time
- Build and compose simulation scenes
- Validate camera views and sensor placements

**Headless Mode** — run simulation without rendering:
- Train RL policies at scale (4096+ parallel environments)
- Generate synthetic data (domain randomization)
- Run automated evaluation pipelines
- CI/CD simulation testing

---

## Infrastructure

Both modes use the same Terraform to deploy a GPU workstation:

```bash
cd isaac-sim-on-aws/infra
terraform init
terraform apply -var="aws_region=us-east-2"
```

| Resource | Description |
|----------|-------------|
| EC2 Instance | g6e.4xlarge (L40S GPU, 48 GB VRAM) |
| AMI | NVIDIA Isaac Sim Marketplace AMI (pre-installed) |
| EBS | 512 GB gp3 |
| Security Group | DCV (8443) + SSH (22) |
| IAM Role | SSM + ECR access |

---

## Prerequisites

- Foundation infrastructure deployed
- [Isaac Sim Marketplace AMI subscription](https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw) (free, one-time per account)
- EC2 GPU quota for g6e.4xlarge
- NVIDIA NGC API key (for pulling container images)

---

## Cost

| Item | Cost | Notes |
|------|------|-------|
| g6e.4xlarge (running) | ~$1.86/hr | Stop when not in use |
| EBS 512 GB | ~$40/month | Persists when instance is stopped |

**Always stop the instance when done** — it bills per second while running.

---

## Agentic Orchestration with Strands Agents

[**strands-robots**](https://github.com/strands-labs/robots) exposes a full **Isaac Sim
backend** (`strands_robots.simulation.isaac`) behind the same `Robot()` /
`Simulation()` interface as its default MuJoCo backend. A
[Strands Agent](https://strandsagents.com) can therefore build worlds, run a candidate
policy, and regression-gate it in Isaac Sim using natural language — the SIL validation
step of the flywheel — then promote the policy to hardware with the same code. See
[strands-agents-on-aws](../strands-agents-on-aws/) for the orchestration layer.

---

## License

Apache 2.0 — see [LICENSE](../LICENSE).
