"""
UR3 Pick-and-Place Policy Evaluation

Loads a trained policy and runs evaluation episodes in Isaac Sim.
Optionally renders video for demo purposes.

Usage:
    python evaluate.py --checkpoint checkpoints/best_policy.pt --render-video
    python evaluate.py --checkpoint checkpoints/best_policy.pt --num-episodes 100
"""

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp

# Enable rendering for video capture
simulation_app = SimulationApp({"headless": True, "enable_livestream": False})

import torch
import numpy as np
from omni.isaac.lab.app import AppLauncher
import omni.isaac.lab_tasks  # noqa: F401


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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  UR3 Pick-and-Place Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Render video: {args.render_video}")
    print(f"{'='*60}")

    # Load policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = torch.jit.load(args.checkpoint, map_location=device)
    policy.eval()
    print(f"  Policy loaded on {device}")

    # Create environment (single env for evaluation)
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(args.env)
    env_cfg.scene.num_envs = 1  # Single env for clear video

    # Video recording setup
    if args.render_video:
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

    for episode in range(args.num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action = policy(obs_tensor).squeeze(0).cpu().numpy()

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Record frame for video
            if args.render_video and episode < 10:  # Record first 10 episodes
                frame = env.render()
                recorder.add_frame(frame)

        # Track metrics
        if info.get('success', False):
            successes += 1
            cycle_times.append(steps * 0.02)  # 50Hz control = 20ms per step

        if info.get('failure_mode'):
            failure_modes[info['failure_mode']] = failure_modes.get(info['failure_mode'], 0) + 1

        total_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            current_rate = successes / (episode + 1) * 100
            print(f"  Episode {episode+1}/{args.num_episodes} | "
                  f"Success rate: {current_rate:.1f}% | "
                  f"Avg reward: {np.mean(total_rewards):.2f}")

    # Finalize video
    if args.render_video:
        recorder.close()
        print(f"\n  Video saved: {video_path}")

    # Compute final metrics
    success_rate = successes / args.num_episodes * 100
    avg_reward = np.mean(total_rewards)
    avg_cycle_time = np.mean(cycle_times) if cycle_times else 0

    metrics = {
        'success_rate_pct': round(success_rate, 1),
        'num_episodes': args.num_episodes,
        'successes': successes,
        'avg_reward': round(float(avg_reward), 2),
        'avg_cycle_time_sec': round(float(avg_cycle_time), 2),
        'failure_modes': failure_modes,
        'checkpoint': args.checkpoint,
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
    if args.render_video:
        print(f"  Demo video: {video_path}")
    print(f"{'='*60}")

    simulation_app.close()


if __name__ == '__main__':
    main()
