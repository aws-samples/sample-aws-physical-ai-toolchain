"""
UR3 Pick-and-Place Policy Evaluation

Loads a trained policy and runs evaluation episodes in Isaac Sim.
Optionally renders video for demo purposes.

Supports two modes:
  - Open-loop (default): policy runs directly in-process with env
  - Closed-loop (--closed-loop): policy server + sim client over ZMQ transport

Usage:
    # Open-loop (original)
    python evaluate.py --checkpoint checkpoints/best_policy.pt --render-video
    python evaluate.py --checkpoint checkpoints/best_policy.pt --num-episodes 100

    # Closed-loop (two-terminal recommended; in-thread unvalidated convenience mode also available)
    # Terminal 1: python eval_policy_server.py --checkpoint model.pt
    # Terminal 2: python eval_sim_client.py --endpoint tcp://localhost:5555
    # OR in-thread mode: python evaluate.py --checkpoint model.pt --closed-loop --num-episodes 100
"""

import argparse
import json
import sys
from pathlib import Path


def _lazy_isaac_imports():
    """Lazy-import Isaac Sim deps (GPU-only). Call inside main() if not --closed-loop."""
    from isaacsim import SimulationApp

    # Enable rendering for video capture
    simulation_app = SimulationApp({"headless": True, "enable_livestream": False})

    import gymnasium as gym
    import numpy as np
    import omni.isaac.lab_tasks  # noqa: F401
    import torch
    from omni.isaac.lab.app import AppLauncher

    # Register our custom UR3 env with gymnasium (training/envs/__init__.py).
    # Ensure the repo root is on sys.path so this works when run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import training.envs  # noqa: F401 — registers PickAndPlaceUR3-v0

    return simulation_app, gym, torch, np


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained pick-and-place policy')
    parser.add_argument('--env', type=str, default='PickAndPlaceUR3-v0')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained policy checkpoint')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render-video', action='store_true',
                        help='Render evaluation video (MP4)')
    parser.add_argument('--output-dir', type=str, default='./evaluation',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--closed-loop', action='store_true',
                        help='Use closed-loop eval (policy server + sim client over ZMQ)')
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

        # Give server time to bind
        import time
        time.sleep(2)

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
        render_video=args.render_video,
        output_dir=args.output_dir,
        env=args.env,
        seed=args.seed,
    )


def run_open_loop(checkpoint, num_episodes, render_video, output_dir, env, seed):
    """Run open-loop evaluation: policy in-process with Isaac Sim env.

    Args:
        checkpoint: Path to TorchScript checkpoint
        num_episodes: Number of evaluation episodes
        render_video: If True, record MP4 of first 10 episodes
        output_dir: Directory to save metrics JSON and video
        env: Gym environment ID (default: PickAndPlaceUR3-v0)
        seed: Random seed

    Returns:
        dict: Metrics (success_rate_pct, num_episodes, avg_reward, avg_cycle_time_sec, failure_modes)
    """
    # Lazy-import Isaac Sim deps (GPU-only, keep module importable on laptop)
    simulation_app, gym, torch, np = _lazy_isaac_imports()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  UR3 Pick-and-Place Evaluation")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Render video: {render_video}")
    print(f"{'='*60}")

    # Load policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = torch.jit.load(checkpoint, map_location=device)
    policy.eval()
    print(f"  Policy loaded on {device}")

    # Create environment (single env for evaluation)
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(env)
    env_cfg.scene.num_envs = 1  # Single env for clear video

    # Instantiate the Isaac Lab environment (GPU-only, unvalidated on hardware).
    # This path requires a G-family GPU with Isaac Sim — cannot run locally.
    # The UR3 task is registered but GPU-untested; validated path uses built-in tasks.
    env_instance = gym.make(env, cfg=env_cfg)

    # Video recording setup
    if render_video:
        from omni.isaac.lab.utils.video import VideoRecorder
        video_path = output_dir / 'eval_video.mp4'
        recorder = VideoRecorder(
            output_path=str(video_path),
            fps=30,
            resolution=(1280, 720),
        )
        print(f"  Recording video to: {video_path}")

    # Run evaluation episodes
    successes = 0
    total_rewards = []
    cycle_times = []
    failure_modes = {'timeout': 0, 'drop': 0, 'collision': 0}

    for episode in range(num_episodes):
        obs, _ = env_instance.reset()
        # Isaac Lab manager-based envs return vectorized obs even for num_envs=1
        if isinstance(obs, dict):
            obs = {k: v[0] if hasattr(v, '__getitem__') else v for k, v in obs.items()}
        elif hasattr(obs, '__getitem__'):
            obs = obs[0]

        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Get action from policy
            with torch.no_grad():
                if isinstance(obs, dict):
                    # Flatten observation dict to tensor (Isaac Lab convention)
                    obs_flat = torch.cat([torch.FloatTensor(v).flatten() for v in obs.values()])
                    obs_tensor = obs_flat.unsqueeze(0).to(device)
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = policy(obs_tensor).squeeze(0).cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env_instance.step(action.reshape(1, -1))
            done = terminated[0] or truncated[0]

            # Extract scalar reward from vectorized return
            if hasattr(reward, '__getitem__'):
                reward = float(reward[0])
            episode_reward += reward
            steps += 1

            # Extract single-env obs for next iteration
            if isinstance(obs, dict):
                obs = {k: v[0] if hasattr(v, '__getitem__') else v for k, v in obs.items()}
            elif hasattr(obs, '__getitem__'):
                obs = obs[0]

            # Record frame for video
            if render_video and episode < 10:  # Record first 10 episodes
                frame = env_instance.render()
                recorder.add_frame(frame)

        # Track metrics (extract from vectorized info dict)
        info_single = {k: v[0] if hasattr(v, '__getitem__') else v for k, v in info.items()}
        if info_single.get('success', False):
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

    # Finalize video
    if render_video:
        recorder.close()
        print(f"\n  Video saved: {video_path}")

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
    if render_video:
        print(f"  Demo video: {video_path}")
    print(f"{'='*60}")

    simulation_app.close()
    return metrics


if __name__ == '__main__':
    main()
