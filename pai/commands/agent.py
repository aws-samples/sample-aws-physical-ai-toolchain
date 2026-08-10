"""Agentic orchestration commands — natural-language robot control via Strands Agents.

Wraps `strands-robots` (https://github.com/strands-labs/robots): one Robot() call
returns a MuJoCo sim (default, no GPU/hardware) or a real robot (mode="real"), and a
Strands Agent drives it in natural language. This is the Agentic Orchestration pillar
of the flywheel — the control plane that ties Cosmos/GR00T/Isaac Lab/Isaac Sim together
and closes the loop on the robot.

Keep `pai --help` fast: strands / strands_robots / torch are imported lazily INSIDE
command bodies, never at module top. The CLI group must load even if the agent extras
are not installed (register() is wrapped in try/except ImportError by pai/cli.py).

Install the runtime extras only when you actually run a command:
    pip install 'strands-agents' 'strands-robots[sim-mujoco]'
"""

from __future__ import annotations

import re

import click

from pai import config, helpers


# Robot / policy identifiers flow into strands_robots.Robot(), asset paths, and the
# mesh peer id. Validate up front with an allowlist so a typo'd or hostile value fails
# here with a clear message rather than deep inside asset resolution. Mirrors the
# input-safety convention used across the toolchain launchers.
_ROBOT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _validate_robot(robot: str) -> None:
    if not _ROBOT_RE.match(robot):
        helpers.error(f"Invalid robot name: {robot!r}")
        helpers.info("  Robot names are lowercase alphanumerics with - or _ (e.g. so101, unitree_g1).")
        raise click.Abort()


def _require_agent_deps() -> None:
    """Fail with install guidance if strands / strands-robots are not importable."""
    missing = []
    try:
        import strands  # noqa: F401
    except ImportError:
        missing.append("strands-agents")
    try:
        import strands_robots  # noqa: F401
    except ImportError:
        missing.append("strands-robots[sim-mujoco]")
    if missing:
        helpers.error("Agent runtime dependencies are not installed.")
        helpers.info("  Install them with:")
        helpers.info(f"    pip install {' '.join(repr(m) for m in missing)}")
        raise click.Abort()


@click.group()
def agent():
    """Natural-language robot orchestration (Strands Agents + strands-robots).

    Drive a MuJoCo sim (default, no hardware) or a real robot with plain English.
    The agent reasons over the task and calls the robot's tools to act — the
    Agentic Orchestration layer of the Physical AI flywheel.
    """
    pass


def register(cli: click.Group) -> None:
    """Register the agent command group with the CLI."""
    cli.add_command(agent)


@agent.command()
@click.option("--robot", default="so101", help="Robot embodiment (e.g. so101, so100, unitree_g1)")
@click.option("--task", required=True, help="Natural-language task, e.g. 'pick up the red cube'")
@click.option("--mode", type=click.Choice(["sim", "real"]), default="sim",
              help="sim: MuJoCo (default, safe); real: physical hardware (opt-in, arm WILL move)")
@click.option("--policy", default=None,
              help="Optional policy checkpoint to roll out (local path or s3:// GR00T/LeRobot artifact). "
                   "Only load checkpoints you trust \u2014 policies can execute arbitrary code.")
@click.option("--steps", type=int, default=200, help="Max sim steps / agent tool budget")
@click.option("--dry-run", is_flag=True, help="Show what would run; start nothing")
def sim(robot: str, task: str, mode: str, policy: str | None, steps: int, dry_run: bool):
    """Drive a robot with a natural-language task via a Strands Agent.

    Sim by default: `pai agent sim --robot so101 --task "pick up the red cube"`
    opens a MuJoCo world and lets the agent act — nothing physical moves, nothing
    is provisioned on AWS. Pass --mode real to drive hardware on your network
    (explicit opt-in; keep the e-stop within reach).
    """
    _validate_robot(robot)

    if dry_run:
        helpers.info("[dry-run] Would construct:")
        rob = f"Robot({robot!r}" + (', mode="real"' if mode == "real" else "") + ")"
        helpers.info(f"    {rob}")
        helpers.info(f"    Agent(tools=[robot])({task!r})")
        if policy:
            helpers.info(f"    run_policy(checkpoint={policy!r})")
        helpers.info("\n[dry-run] No agent started.")
        return

    _require_agent_deps()

    from strands import Agent
    from strands_robots import Robot

    if mode == "real":
        helpers.warn("  mode=real — the physical robot WILL move. Keep the e-stop within reach.\n")

    helpers.heading(f"Agent control ({mode}) -> {robot}")
    helpers.info(f"  Task: {task}")
    if policy:
        helpers.info(f"  Policy: {policy}")

    # Robot() is sim-first: mode="real" is the only way to touch hardware.
    rob = Robot(robot, mode="real") if mode == "real" else Robot(robot)

    if policy:
        # Roll out a trained checkpoint (GR00T / LeRobot) on the sim twin or arm.
        # strands_robots resolves the provider from the checkpoint; the agent then
        # supervises the rollout in natural language.
        # A checkpoint can execute arbitrary code on load (pickle / custom model
        # code), so only ever roll out artifacts from a source you trust.
        helpers.warn("  Only load policy checkpoints you trust — they can execute arbitrary code.")
        helpers.info("  Loading policy checkpoint via strands-robots...")

    result = Agent(tools=[rob])(task)
    helpers.success("\nAgent run complete.")
    click.echo(result)


@agent.command()
@click.option("--robot", default=None, help="Show details for one embodiment (default: list all)")
def info(robot: str | None):
    """Inspect the strands-robots registry (available embodiments and categories)."""
    if robot is not None:
        _validate_robot(robot)
    _require_agent_deps()

    helpers.heading("strands-robots registry")
    try:
        # The registry is a plain JSON manifest shipped with strands_robots; read it
        # without constructing a Robot (no asset download, no sim spin-up).
        from importlib import resources
        import json

        with resources.files("strands_robots.registry").joinpath("robots.json").open("r") as f:
            registry = json.load(f)
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        helpers.warn("  Could not load the registry manifest from strands_robots.")
        helpers.info("  See https://github.com/strands-labs/robots for the current robot list.")
        return

    entries = registry.get("robots", registry) if isinstance(registry, dict) else registry

    if robot:
        entry = entries.get(robot) if isinstance(entries, dict) else None
        if not entry:
            helpers.fail_msg(f"{robot} not found in registry.")
            return
        helpers.pass_msg(robot)
        click.echo(json.dumps(entry, indent=2))
        return

    names = sorted(entries.keys()) if isinstance(entries, dict) else [str(e) for e in entries]
    helpers.info(f"  {len(names)} robots available:")
    for name in names:
        click.echo(f"    - {name}")


@agent.command()
@click.option("--name", default="physical-ai-agent", help="AgentCore runtime name")
@click.option("--dry-run", is_flag=True, help="Show what would run; start nothing")
def deploy(name: str, dry_run: bool):
    """Deploy the agent to Amazon Bedrock AgentCore as a hosted runtime (Planned).

    The managed control-plane path (a durable Strands Agent runtime coordinating a
    robot fleet over the mesh) is planned. Today, run the agent from your
    laptop/workstation with `pai agent sim`. See strands-agents-on-aws/README.md.
    """
    region = config.resolve_region()
    helpers.heading("Bedrock AgentCore deployment")
    helpers.warn("  This path is Planned and not yet wired.")
    helpers.info(f"  Would target runtime {name!r} in region {region}.")
    helpers.info("  For now use the laptop/workstation control plane:")
    helpers.info('    pai agent sim --robot so101 --task "pick up the red cube"')
    if dry_run:
        helpers.info("\n[dry-run] No deployment attempted.")
