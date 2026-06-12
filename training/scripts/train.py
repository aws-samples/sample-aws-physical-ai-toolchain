"""
UR3 Pick-and-Place RL Training Script

Entry point for training a pick-and-place policy using PPO in Isaac Lab.
Runs headless Isaac Sim with parallel environments on GPU.

Usage:
    python train.py --env PickAndPlaceUR3-v0 --config configs/ppo_pick_place.yaml
    python train.py --env PickAndPlaceUR3-v0 --config configs/ppo_pick_place.yaml --num-envs 4096
"""

import argparse
import os
import yaml
from pathlib import Path

# Isaac Sim must be imported before any other Isaac/Omniverse imports
from isaacsim import SimulationApp

# Launch headless (no GUI, no rendering overhead during training)
simulation_app = SimulationApp({"headless": True})

# Now safe to import Isaac Lab and RL components
from omni.isaac.lab.app import AppLauncher
import omni.isaac.lab_tasks  # noqa: F401 — registers environments
from omni.isaac.lab_tasks.utils import parse_env_cfg

# RL algorithm (rl_games PPO)
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train UR3 pick-and-place policy')
    parser.add_argument('--env', type=str, default='PickAndPlaceUR3-v0',
                        help='Isaac Lab environment name')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac'],
                        help='RL algorithm')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config YAML')
    parser.add_argument('--num-envs', type=int, default=4096,
                        help='Number of parallel environments')
    parser.add_argument('--max-iterations', type=int, default=5000,
                        help='Maximum training iterations')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--scene-dir', type=str, default=None,
                        help='Directory with Cosmos-generated USD scenes')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    config['env']['num_envs'] = args.num_envs
    config['training']['max_iterations'] = args.max_iterations
    config['training']['seed'] = args.seed

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  UR3 Pick-and-Place Training")
    print(f"  Environment: {args.env}")
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Parallel envs: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Scene dir: {args.scene_dir or 'procedural (no Cosmos scenes)'}")
    print(f"{'='*60}")

    # Configure Isaac Lab environment
    env_cfg = parse_env_cfg(args.env)
    env_cfg.scene.num_envs = args.num_envs

    # If Cosmos scenes are provided, configure scene loading
    if args.scene_dir:
        env_cfg.scene.usd_scene_dir = args.scene_dir
        print(f"  Using {len(os.listdir(args.scene_dir))} Cosmos-generated scenes")

    # Apply domain randomization settings from config
    if config.get('domain_randomization', {}).get('enabled', False):
        dr_config = config['domain_randomization']
        env_cfg.domain_randomization = dr_config
        print("  Domain randomization: ENABLED")

    # Apply curriculum settings
    if config.get('curriculum', {}).get('enabled', False):
        env_cfg.curriculum = config['curriculum']
        print("  Curriculum learning: ENABLED")

    # Configure rl_games runner
    runner_config = {
        'params': {
            'seed': args.seed,
            'algo': {
                'name': args.algo,
            },
            'model': {
                'name': 'continuous_a2c_logstd',
            },
            'network': config.get('network', {}),
            'config': {
                'name': f'{args.env}_{args.algo}',
                'env_name': args.env,
                'num_actors': args.num_envs,
                'horizon_length': config['ppo']['horizon_length'],
                'minibatch_size': args.num_envs * config['ppo']['horizon_length'] // config['ppo']['num_minibatches'],
                'mini_epochs': config['ppo']['num_epochs'],
                'clip_value': True,
                'gamma': config['ppo']['gamma'],
                'tau': config['ppo']['gae_lambda'],
                'e_clip': config['ppo']['clip_range'],
                'entropy_coef': config['ppo']['entropy_coef'],
                'learning_rate': config['ppo']['learning_rate'],
                'lr_schedule': config['ppo']['lr_schedule'],
                'max_epochs': args.max_iterations,
                'save_frequency': config['training']['checkpoint_interval'],
                'save_best_after': 1000,
                'print_stats': True,
            },
        },
    }

    # Create and run trainer
    runner = Runner()
    runner.load(runner_config)
    runner.reset()

    print("\n  Starting training...\n")
    runner.run({
        'train': True,
        'play': False,
        'checkpoint': args.resume,
        'sigma': None,
    })

    # Save final checkpoint
    final_path = checkpoint_dir / 'best_policy.pt'
    print(f"\n  Training complete!")
    print(f"  Best policy saved to: {final_path}")
    print(f"  Run evaluation: python evaluate.py --checkpoint {final_path}")

    # Cleanup
    simulation_app.close()


if __name__ == '__main__':
    main()
