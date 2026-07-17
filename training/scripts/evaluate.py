"""
UR3 Pick-and-Place Policy Evaluation — Metrics aggregation wrapper

This script computes success-rate / failure-mode / cycle-time metrics that Isaac Lab
does not provide natively. For visualization and video capture, use NVIDIA's stock
`play.py` with `--video` (the native, documented evaluation path).

Supports two modes:
  - Open-loop (default): policy runs directly in-process with env
  - Closed-loop (--closed-loop): policy server + sim client over ZMQ transport
    (mirrors NVIDIA GR00T's PolicyServer/PolicyClient pattern on port 5555)

IMPORTANT: Expects a TorchScript checkpoint (native `policy.pt` from Isaac Lab's
exporter, which bakes in the observation normalizer). NOT the raw rsl_rl .pt.

Usage:
    # Open-loop metrics (validated path)
    python evaluate.py --checkpoint logs/.../exported/policy.pt --num-episodes 100

    # Video capture: use NVIDIA's stock play.py instead
    # python IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
    #        --task=Isaac-Velocity-Flat-Anymal-D-v0 --checkpoint=logs/.../model_1000.pt --video

    # Closed-loop (two-terminal recommended; in-thread unvalidated convenience mode also available)
    # Terminal 1: python eval_policy_server.py --checkpoint policy.pt
    # Terminal 2: python eval_sim_client.py --endpoint tcp://localhost:5555
    # OR in-thread mode: python evaluate.py --checkpoint policy.pt --closed-loop --num-episodes 100
"""

import argparse
import json
import sys
from pathlib import Path


def _lazy_isaac_imports():
    """Lazy-import Isaac Sim deps (GPU-only). Call inside main() if not --closed-loop.

    IMPORTANT: bootstrap Isaac via Isaac Lab's AppLauncher, NOT a raw
    SimulationApp({...}). AppLauncher populates the nucleus asset-root setting
    (/persistent/isaac/asset_root/cloud) that Isaac Lab's task configs read to
    locate robot USDs; a raw SimulationApp leaves it unset, so ISAACLAB_NUCLEUS_DIR
    resolves to the literal string "None" and env creation fails with
    "USD file not found at path: 'None/Isaac/IsaacLab/.../anymal_d.usd'". This
    matches how Lab 2's validated `./isaaclab.sh -p .../train.py` boots, and how
    the closed-loop eval_sim_client.py (validated on L40S) boots.
    """
    import argparse as _argparse

    from isaaclab.app import AppLauncher

    # Headless launch through AppLauncher (no GUI; rendering still available for video).
    _p = _argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(_p)
    _app_args, _ = _p.parse_known_args([])
    _app_args.headless = True
    app_launcher = AppLauncher(_app_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch
    # Isaac Lab 2.x renamed this package `isaaclab_tasks` (was `omni.isaac.lab_tasks`
    # in 1.x). Import whichever the container ships — both just register built-in envs.
    try:
        import isaaclab_tasks  # noqa: F401 — Isaac Lab 2.x
    except ImportError:
        import omni.isaac.lab_tasks  # noqa: F401 — Isaac Lab 1.x fallback

    # Register our custom UR3 env with gymnasium (training/envs/__init__.py).
    # Ensure the repo root is on sys.path so this works when run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import training.envs  # noqa: F401 — registers PickAndPlaceUR3-v0

    return simulation_app, gym, torch, np


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained policy — compute success rate + failure mode metrics'
    )
    parser.add_argument('--env', type=str, default='Isaac-Velocity-Flat-Anymal-D-v0',
                        help='Isaac Lab task ID (default: validated Anymal task)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to native TorchScript policy.pt (with normalizer baked in)')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--output-dir', type=str, default='./evaluation',
                        help='Output directory for eval_metrics.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--closed-loop', action='store_true',
                        help='Use closed-loop eval (policy server + sim client over ZMQ, mirrors GR00T)')
    parser.add_argument('--endpoint', type=str, default='tcp://localhost:5555',
                        help='Policy server endpoint (closed-loop mode only)')
    args = parser.parse_args()

    # Closed-loop mode: orchestrate client (policy server assumed running separately)
    if args.closed_loop:
        print("⚠️  CLOSED-LOOP MODE (in-thread server): UNVALIDATED until run on Lab 2 GPU")
        print("    Two-terminal separate-process is the documented safe default:")
        print("      Terminal 1: python eval_policy_server.py --checkpoint <ckpt>")
        print("      Terminal 2: python eval_sim_client.py --endpoint tcp://localhost:5555\n")

        # In-thread convenience mode (unvalidated): start server + client together
        # This MAY conflict with Isaac GIL; separate-process is safer.
        import threading
        from eval_protocol import ZmqTransport
        import eval_sim_client

        # Start policy server in background thread
        def run_server():
            import eval_policy_server
            sys.argv = [
                "eval_policy_server.py",
                "--checkpoint", args.checkpoint,
                "--endpoint", args.endpoint,
            ]
            eval_policy_server.main()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for the ZMQ server to become reachable before starting the client.
        import time

        def _wait_for_server(endpoint: str, timeout: float = 10.0, interval: float = 0.5):
            """Poll the ZMQ endpoint until it accepts a connection or timeout."""
            import zmq
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVTIMEO, int(interval * 1000))
            sock.connect(endpoint)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    sock.send(b"ping")
                    sock.recv()
                    sock.close()
                    ctx.term()
                    return
                except zmq.Again:
                    pass  # server not ready yet — retry
            sock.close()
            ctx.term()
            raise TimeoutError(
                f"Policy server at {endpoint} did not respond within {timeout}s"
            )

        _wait_for_server(args.endpoint)

        # Run client (this blocks until eval completes)
        transport = ZmqTransport(args.endpoint)
        try:
            eval_sim_client.run_eval(
                args.env,
                transport,
                args.num_episodes,
                max_steps=500,
                output_dir=Path(args.output_dir),
            )
        finally:
            transport.close()
        return

    # Open-loop mode (original path): policy runs in-process with env
    metrics = run_open_loop(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        env=args.env,
        seed=args.seed,
    )


def run_open_loop(checkpoint, num_episodes, output_dir, env, seed):
    """Run open-loop evaluation: policy in-process with Isaac Sim env.

    This computes success-rate / failure-mode / cycle-time metrics that Isaac Lab
    does not provide natively. The rollout loop mirrors NVIDIA's
    `scripts/tutorials/03_envs/policy_inference_in_usd.py` pattern.

    For video capture, use NVIDIA's stock play.py instead:
      python IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
             --task=<task> --checkpoint=logs/.../model_1000.pt --video

    Args:
        checkpoint: Path to native TorchScript policy.pt (with normalizer baked in)
        num_episodes: Number of evaluation episodes
        output_dir: Directory to save eval_metrics.json
        env: Gym environment ID (default: Isaac-Velocity-Flat-Anymal-D-v0)
        seed: Random seed

    Returns:
        dict: Metrics (success_rate_pct, num_episodes, avg_reward, avg_cycle_time_sec, failure_modes)
    """
    # Lazy-import Isaac Sim deps (GPU-only, keep module importable on laptop)
    simulation_app, gym, torch, np = _lazy_isaac_imports()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Policy Evaluation — Metrics Aggregation")
    print(f"  Task: {env}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Episodes: {num_episodes}")
    print(f"{'='*60}")

    # Load native TorchScript policy (normalizer baked in)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = torch.jit.load(checkpoint, map_location=device)
    policy.eval()
    print(f"  Native policy loaded on {device} (normalizer in-graph)")

    # Create environment (single env for evaluation). 2.x = isaaclab_tasks, 1.x fallback.
    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except ImportError:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(env)
    env_cfg.scene.num_envs = 1  # Single env for clear video

    # Instantiate the Isaac Lab environment (GPU-only, unvalidated on hardware).
    # This path requires a G-family GPU with Isaac Sim — cannot run locally.
    # The UR3 task is registered but GPU-untested; validated path uses built-in tasks.
    env_instance = gym.make(env, cfg=env_cfg)

    # Reuse the validated observation flattener from the closed-loop client so both
    # eval paths handle Anymal's "policy" group (and the UR3 dict) identically.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_sim_client import _flatten_obs, _detect_success

    # Run evaluation episodes
    successes = 0
    total_rewards = []
    cycle_times = []
    failure_modes = {'timeout': 0, 'drop': 0, 'collision': 0}

    # Isaac Lab's `info` mixes per-env tensors with nested dicts (e.g. "log",
    # "observations"); only index array-likes to pull env 0 — indexing a dict with
    # [0] raises KeyError, and plain scalars/dicts pass through unchanged.
    def _env0(v):
        if isinstance(v, dict):
            return v
        if hasattr(v, 'shape') and getattr(v, 'ndim', 0) >= 1:  # torch/np tensor
            return v[0]
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return v[0]
        return v

    for episode in range(num_episodes):
        obs_raw, _ = env_instance.reset()
        obs = _flatten_obs(obs_raw)  # handles Anymal "policy" group + UR3 dict

        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(obs_tensor).squeeze(0).cpu().numpy()

            # Isaac Lab manager-based envs expect a torch tensor on the sim device,
            # shaped (num_envs, action_dim) — convert from the numpy policy output.
            action_input = torch.as_tensor(
                action.reshape(1, -1), dtype=torch.float32, device=env_instance.unwrapped.device
            )
            obs_raw, reward, terminated, truncated, info = env_instance.step(action_input)
            done = bool(terminated[0] or truncated[0])

            # Extract scalar reward from vectorized return
            if hasattr(reward, '__getitem__'):
                reward = float(reward[0])
            episode_reward += reward
            steps += 1

            # Flatten obs for next iteration
            obs = _flatten_obs(obs_raw)

        # Track metrics. Pull env 0 from the vectorized info dict safely.
        info_single = {k: _env0(v) for k, v in info.items()}
        success = _detect_success(bool(terminated[0]), bool(truncated[0]), info_single, env_instance)
        if success:
            successes += 1
            cycle_times.append(steps * 0.02)  # 50Hz control = 20ms per step

        if info_single.get('failure_mode'):
            failure_modes[info_single['failure_mode']] = failure_modes.get(info_single['failure_mode'], 0) + 1

        total_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            current_rate = successes / (episode + 1) * 100
            print(f"  Episode {episode+1}/{num_episodes} | "
                  f"Success rate: {current_rate:.1f}% | "
                  f"Avg reward: {np.mean(total_rewards):.2f}")

    # Compute final metrics
    success_rate = successes / num_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_cycle_time = np.mean(cycle_times) if cycle_times else 0

    metrics = {
        'success_rate_pct': round(success_rate, 1),
        'num_episodes': num_episodes,
        'successes': successes,
        'avg_reward': round(float(avg_reward), 2),
        'avg_cycle_time_sec': round(float(avg_cycle_time), 2),
        'failure_modes': failure_modes,
        'checkpoint': checkpoint,
    }

    # Save metrics
    metrics_path = output_dir / 'eval_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg cycle time: {avg_cycle_time:.2f}s")
    print(f"  Failure modes: {failure_modes}")
    print(f"  Metrics saved: {metrics_path}")
    print(f"{'='*60}")
    print(f"\n  For video capture, use NVIDIA's stock play.py with --video flag")

    simulation_app.close()
    return metrics


if __name__ == '__main__':
    main()
