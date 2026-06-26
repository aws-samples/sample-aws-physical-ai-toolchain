"""Closed-loop sim client: drives Isaac Lab env with actions from policy server.

Lazy-imports isaacsim/Isaac Lab INSIDE functions (so importing this module on a
laptop doesn't crash). Runs on the Lab 2 workstation (g6e.4xlarge L40S).

Per eval round: reset env → loop {obs → transport.send → action → step} until
term/trunc → record success. Success detection fallback: prefer info['success'],
else terminated and not truncated, else object-height probe if exposed.

Usage:
    python eval_sim_client.py --task PickAndPlaceUR3-v0 --endpoint tcp://localhost:5555 \
      --eval-rounds 100 --output-dir ./eval_results
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _lazy_imports():
    """Lazy-import Isaac Lab deps (GPU-only). Call inside functions that need them."""
    # This is invoked ONLY when running eval (not at module-import time), so a
    # laptop with no isaacsim can still import eval_sim_client.py for tests.
    try:
        from isaacsim import SimulationApp
    except ImportError as e:
        print(
            "❌ ERROR: Isaac Sim not found. This script requires Isaac Lab on a GPU.\n"
            "Run this on the Lab 2 workstation (g6e.4xlarge).",
            file=sys.stderr,
        )
        raise e

    # Enable headless (no GUI, no rendering)
    simulation_app = SimulationApp({"headless": True, "enable_livestream": False})

    import gymnasium as gym
    import omni.isaac.lab_tasks  # noqa: F401 — registers built-in envs

    # Register custom UR3 env
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import training.envs  # noqa: F401 — registers PickAndPlaceUR3-v0

    return simulation_app, gym


def _flatten_obs(obs: Any) -> np.ndarray:
    """Flatten observation (handle BOTH dict and pre-flattened tensor).

    Isaac Lab manager-based envs can return:
      - dict obs (camera, joint_pos, etc.) → flatten in export.py order
      - pre-flattened tensor (if env already concatenated)

    export.py order: joint_pos (6) + joint_vel (6) + gripper (1) + object_pose (7) + camera (12288)
    Total: 12308 floats.

    GPU-UNVALIDATED: These keys match pick_and_place_ur3.py ObservationTermCfg names,
    but are untested on a real GPU run. If a key is missing, a warning is logged.
    """
    if isinstance(obs, dict):
        # Dict obs: flatten in export.py order (matches pick_and_place_ur3.py)
        expected_keys = ["joint_pos", "joint_vel", "gripper_state", "object_pos_relative", "wrist_camera"]
        parts = []
        for key in expected_keys:
            val = obs.get(key)
            if val is None:
                # Defensive: log missing key (GPU-unvalidated path)
                print(f"  ⚠️  WARNING: Expected obs key '{key}' missing from env obs dict. Available: {list(obs.keys())}")
                continue
            # Vectorized env: extract [0]
            if hasattr(val, "__getitem__") and hasattr(val, "shape") and val.shape[0] > 1:
                val = val[0]
            # Flatten
            parts.append(np.array(val).flatten())
        return np.concatenate(parts)
    else:
        # Pre-flattened tensor (or already vectorized)
        arr = np.array(obs)
        if arr.ndim > 1:
            arr = arr[0]  # extract single env
        return arr.flatten()


def _detect_success(terminated: bool, truncated: bool, info: Dict[str, Any], env: Any) -> bool:
    """Success detection fallback (RISK: env may not write info['success']).

    Prefer info['success'], else terminated and not truncated (task succeeded
    without timeout), else object-height probe if exposed.
    """
    # 1. Prefer info['success'] if present
    if "success" in info:
        return bool(info["success"])

    # 2. Fallback: terminated without truncation
    if terminated and not truncated:
        return True

    # 3. Last resort: object height probe (PickAndPlaceUR3-v0 success = obj z > 0.15)
    try:
        obj_pos = env.scene["object"].data.root_pos_w
        if obj_pos.ndim > 1:
            obj_pos = obj_pos[0]  # single env
        return obj_pos[2] > 0.15
    except Exception:
        # No object probe available; can't infer success
        return False


def run_eval(
    task: str,
    transport,  # eval_protocol.Transport
    eval_rounds: int,
    max_steps: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run closed-loop eval: policy server drives Isaac Lab env for N rounds."""
    simulation_app, gym = _lazy_imports()

    print(f"  Creating environment: {task}")
    # Parse env config (Isaac Lab convention)
    from omni.isaac.lab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task)
    env_cfg.scene.num_envs = 1  # Single env for clear eval
    env = gym.make(task, cfg=env_cfg)

    successes = 0
    total_rewards = []
    cycle_times = []
    failure_modes = {"timeout": 0, "drop": 0, "collision": 0}

    print(f"  Running {eval_rounds} evaluation rounds...")
    for episode in range(eval_rounds):
        obs_raw, _ = env.reset()
        obs = _flatten_obs(obs_raw)

        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # Query policy server
            action = transport.send(obs)

            # Step env (Isaac Lab expects (num_envs, action_dim))
            obs_raw, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            done = terminated[0] or truncated[0]

            # Extract scalar reward
            if hasattr(reward, "__getitem__"):
                reward = float(reward[0])
            episode_reward += reward
            steps += 1

            # Flatten obs for next iteration
            obs = _flatten_obs(obs_raw)

        # Detect success
        info_single = {k: v[0] if hasattr(v, "__getitem__") else v for k, v in info.items()}
        success = _detect_success(terminated[0], truncated[0], info_single, env)
        if success:
            successes += 1
            cycle_times.append(steps * 0.02)  # 50Hz control = 20ms/step

        # Track failure mode
        if info_single.get("failure_mode"):
            failure_modes[info_single["failure_mode"]] = failure_modes.get(info_single["failure_mode"], 0) + 1

        total_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            current_rate = successes / (episode + 1) * 100
            print(
                f"  Episode {episode+1}/{eval_rounds} | "
                f"Success rate: {current_rate:.1f}% | "
                f"Avg reward: {np.mean(total_rewards):.2f}"
            )

    # Compute metrics
    success_rate = successes / eval_rounds * 100
    avg_reward = np.mean(total_rewards)
    avg_cycle_time = np.mean(cycle_times) if cycle_times else 0

    metrics = {
        "success_rate_pct": round(success_rate, 1),
        "num_episodes": eval_rounds,
        "successes": successes,
        "avg_reward": round(float(avg_reward), 2),
        "avg_cycle_time_sec": round(float(avg_cycle_time), 2),
        "failure_modes": failure_modes,
    }

    # Save metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg cycle time: {avg_cycle_time:.2f}s")
    print(f"  Failure modes: {failure_modes}")
    print(f"  Metrics saved: {metrics_path}")
    print(f"{'='*60}")

    simulation_app.close()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Closed-loop sim client (Isaac Lab)")
    parser.add_argument("--task", type=str, default="PickAndPlaceUR3-v0", help="Isaac Lab task ID")
    parser.add_argument("--endpoint", type=str, default="tcp://localhost:5555", help="Policy server endpoint")
    parser.add_argument("--eval-rounds", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory for metrics")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"{'='*60}")
    print(f"  Closed-loop Evaluation Client")
    print(f"  Task: {args.task}")
    print(f"  Policy server: {args.endpoint}")
    print(f"  Eval rounds: {args.eval_rounds}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Create transport
    from eval_protocol import ZmqTransport

    transport = ZmqTransport(args.endpoint)

    try:
        run_eval(args.task, transport, args.eval_rounds, args.max_steps, output_dir)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
