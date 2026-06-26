"""pai eval — Policy evaluation commands (open-loop and closed-loop)."""

from pathlib import Path

import click

from pai import _scriptpath, helpers


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--checkpoint", help="Path or s3:// URI to checkpoint")
@click.option("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
@click.option("--render-video", is_flag=True, help="Render evaluation video (MP4)")
@click.option("--output-dir", default="./evaluation", help="Output directory for results")
@click.option("--env", default="PickAndPlaceUR3-v0", help="Gym environment ID")
@click.option("--seed", type=int, default=0, help="Random seed")
@click.option("--closed-loop", is_flag=True, help="Use closed-loop eval (policy server + sim client over ZMQ)")
@click.option("--endpoint", default="tcp://localhost:5555", help="Policy server endpoint (closed-loop mode only)")
@click.option("--eval-rounds", type=int, default=100, help="Number of eval rounds (closed-loop mode only)")
@click.option("--max-steps", type=int, default=500, help="Max steps per episode (closed-loop mode only)")
def eval(ctx, checkpoint, num_episodes, render_video, output_dir, env, seed, closed_loop, endpoint, eval_rounds, max_steps):
    """Evaluate trained policies (open-loop by default, --closed-loop for distributed eval)."""
    # Ensure training/scripts is on sys.path
    _scriptpath.ensure_scripts_on_path()

    # If a subcommand was invoked (e.g., 'serve'), don't run evaluation
    if ctx.invoked_subcommand is not None:
        return

    # Checkpoint is required when running evaluation (not when invoking subcommands)
    if checkpoint is None:
        raise click.UsageError("--checkpoint is required for evaluation")

    output_path = Path(output_dir)

    if closed_loop:
        # Closed-loop: policy server + sim client
        helpers.info(f"Running closed-loop evaluation: {eval_rounds} rounds")
        helpers.warn("\n⚠️  UNVALIDATED on GPU hardware — requires Lab 2 workstation (g6e.4xlarge L40S).")
        helpers.info("For production use, run server and client in separate terminals:")
        helpers.info("  Terminal 1: pai eval serve --checkpoint <ckpt>")
        helpers.info("  Terminal 2: python training/scripts/eval_sim_client.py --endpoint tcp://localhost:5555\n")

        try:
            from eval_protocol import ZmqTransport
            from eval_sim_client import run_eval as run_eval_closed

            transport = ZmqTransport(endpoint)
            try:
                metrics = run_eval_closed(
                    task=env,
                    transport=transport,
                    eval_rounds=eval_rounds,
                    max_steps=max_steps,
                    output_dir=output_path,
                )
                helpers.success("\nEvaluation complete!")
                click.echo(f"\nMetrics: {metrics}")
            finally:
                transport.close()
        except Exception as e:
            helpers.error(f"Closed-loop evaluation failed: {e}")
            raise
    else:
        # Open-loop: policy runs in-process with env
        helpers.info(f"Running open-loop evaluation: {num_episodes} episodes")
        helpers.warn("⚠️  Requires Isaac Sim (GPU) — run on Lab 2 workstation")

        try:
            from evaluate import run_open_loop

            metrics = run_open_loop(
                checkpoint=checkpoint,
                num_episodes=num_episodes,
                render_video=render_video,
                output_dir=str(output_path),
                env=env,
                seed=seed,
            )
            helpers.success("\nEvaluation complete!")
            click.echo(f"\nMetrics: {metrics}")
        except Exception as e:
            helpers.error(f"Open-loop evaluation failed: {e}")
            raise


@eval.command()
@click.option("--checkpoint", required=True, help="Path or s3:// URI to TorchScript checkpoint")
@click.option("--endpoint", default="tcp://127.0.0.1:5555", help="ZMQ bind endpoint (default: localhost only)")
@click.option("--device", default=None, help="Inference device (cuda/cpu, auto-detected if omitted)")
def serve(checkpoint, endpoint, device):
    """Start a policy server for closed-loop evaluation."""
    # Ensure training/scripts is on sys.path
    _scriptpath.ensure_scripts_on_path()

    helpers.info(f"Starting policy server on {endpoint}")
    helpers.warn("⚠️  Checkpoint must be TorchScript (not raw .pt) — use 'pai export' to convert")

    # Lazy-import (pulls torch)
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from eval_policy_server import serve as serve_policy

        serve_policy(checkpoint=checkpoint, endpoint=endpoint, device=device)
    except KeyboardInterrupt:
        helpers.info("\nShutting down policy server")
    except Exception as e:
        helpers.error(f"Policy server failed: {e}")
        raise


def register(cli: click.Group):
    """Register eval commands with the CLI."""
    cli.add_command(eval)
