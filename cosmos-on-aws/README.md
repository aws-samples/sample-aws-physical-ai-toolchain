# Cosmos on AWS

Deploy [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) World Foundation
Models on AWS for **synthetic data generation** — the first pillar of the Physical AI
flywheel. Two capabilities:

- **Cosmos 3 (Predict)** — generate entirely new synthetic demonstrations from text
  prompts + a reference video (`cosmos-framework`, `p5.48xlarge`).
- **Cosmos Transfer 2.5** — restyle existing training videos (lighting, materials,
  surfaces) while preserving exact geometry and motion, for sim-to-real diversity
  (`g6e.12xlarge`).

Deploy with `cosmos-on-aws/infra/` (Terraform). Sources and pinned refs are in
[`infra/variables.tf`](infra/variables.tf).

---

## Agentic Orchestration with Strands Agents

[**strands-robots**](https://github.com/strands-labs/robots) ships a **Cosmos 3
trainer** (`strands_robots.training.cosmos3`) that drives the same
`cosmos-framework` SFT pipeline as a Python library. This lets a
[Strands Agent](https://strandsagents.com) orchestrate the data-generation stage —
scripting scene prompts, launching Cosmos runs, and curating the generated episodes
into a LeRobot v2 dataset for downstream GR00T fine-tuning. See
[strands-agents-on-aws](../strands-agents-on-aws/) for the orchestration layer.

---

## License

Apache 2.0 — see [LICENSE](../LICENSE).
