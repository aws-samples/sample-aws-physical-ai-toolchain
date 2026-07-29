# Isaac Sim on AWS

Deploy an [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
GPU workstation on AWS for **physics-accurate simulation** — the SIL (software-in-the-loop)
pillar of the Physical AI flywheel. Isaac Sim provides gravity, friction, collisions,
and RTX camera rendering to build scenes, debug physics, and validate policies before
they touch hardware.

Deploy with `isaac-sim-on-aws/infra/` (Terraform): a `g6e.4xlarge` instance from the
NVIDIA Isaac Sim Marketplace AMI, reachable over NICE DCV (browser, port 8443).
Variables are in [`infra/variables.tf`](infra/variables.tf).

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
