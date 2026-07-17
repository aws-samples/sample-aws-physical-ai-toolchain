"""Closed-loop sim client: drives Isaac Lab env with actions from policy server.

Mirrors NVIDIA GR00T's PolicyClient pattern (ZMQ REQ/REP on port 5555). Lazy-imports
isaacsim/Isaac Lab INSIDE functions (so importing this module on a laptop doesn't crash).
Runs on the Lab 2 workstation (g6e.4xlarge L40S).

Per eval round: reset env → loop {obs → transport.send → action → step} until
term/trunc → record success. Success detection fallback: prefer info['success'],
else terminated and not truncated, else object-height probe if exposed.

The policy server loads a native TorchScript policy.pt (from Isaac Lab's exporter,
with normalizer baked in). This client sends unnormalized observations; the server's
policy handles normalization in-graph.

Usage:
    python eval_sim_client.py --task Isaac-Velocity-Flat-Anymal-D-v0 \
                               --endpoint tcp://localhost:5555 \
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
    #
    # IMPORTANT: bootstrap Isaac via Isaac Lab's AppLauncher, NOT a raw
    # SimulationApp({...}). AppLauncher populates the nucleus asset-root setting
    # (/persistent/isaac/asset_root/cloud) that Isaac Lab's task configs read to
    # locate robot USDs. A raw SimulationApp leaves it unset, so ISAACLAB_NUCLEUS_DIR
    # resolves to the literal string "None" and env creation fails with
    # "USD file not found at path: 'None/Isaac/IsaacLab/.../anymal_d.usd'".
    # This matches how Lab 2's validated `./isaaclab.sh -p .../train.py` boots.
    try:
        import argparse as _argparse

        from isaaclab.app import AppLauncher
    except ImportError as e:
        print(
            "❌ ERROR: Isaac Lab not found. This script requires Isaac Lab on a GPU.\n"
            "Run this on the Lab 2 workstation (g6e.4xlarge) via isaaclab.sh -p.",
            file=sys.stderr,
        )
        raise e

    # Headless launch through AppLauncher (no GUI, no rendering).
    _p = _argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(_p)
    _app_args, _ = _p.parse_known_args([])
    _app_args.headless = True
    app_launcher = AppLauncher(_app_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    # Isaac Lab 2.x renamed this package `isaaclab_tasks` (was `omni.isaac.lab_tasks`
    # in 1.x). Import whichever the container ships — both just register built-in envs.
    try:
        import isaaclab_tasks  # noqa: F401 — Isaac Lab 2.x
    except ImportError:
        import omni.isaac.lab_tasks  # noqa: F401 — Isaac Lab 1.x fallback

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
        # Standard Isaac Lab manager-based envs (e.g. Anymal) nest the actor
        # observation under a "policy" group — already concatenated to the
        # policy's input dim (48 for Anymal). Use it directly when present.
        if "policy" in obs:
            val = obs["policy"]
            arr = np.array(val.cpu() if hasattr(val, "cpu") else val)
            if arr.ndim > 1:
                arr = arr[0]  # single env from the vectorized batch
            return arr.flatten()

        # UR3 pick-and-place dict obs: flatten in export.py order
        # (matches pick_and_place_ur3.py ObservationTermCfg names). UNVALIDATED.
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
            parts.append(np.array(val.cpu() if hasattr(val, "cpu") else val).flatten())
        if not parts:
            raise ValueError(
                f"No known observation keys found in env obs dict. Available keys: "
                f"{list(obs.keys())}. For a manager-based task expose a 'policy' group; "
                f"for UR3 expose {expected_keys}."
            )
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
    # Parse env config (Isaac Lab convention). 2.x = isaaclab_tasks, 1.x = omni.isaac.lab_tasks.
    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except ImportError:
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
            # Query policy server (returns a numpy action array)
            action = transport.send(obs)

            # Isaac Lab manager-based envs expect a torch tensor on the sim device,
            # shaped (num_envs, action_dim). Convert from the numpy action the
            # transport returns. (gym.Env subclasses also accept numpy, but the
            # underlying Isaac Lab env does not — convert explicitly.)
            action_input = action.reshape(1, -1)
            try:
                import torch as _torch
                action_input = _torch.as_tensor(
                    action_input, dtype=_torch.float32, device=env.unwrapped.device
                )
            except Exception:
                pass  # fall back to numpy (e.g. non-Isaac gym env in tests)

            obs_raw, reward, terminated, truncated, info = env.step(action_input)
            done = bool(terminated[0] or truncated[0])

            # Extract scalar reward
            if hasattr(reward, "__getitem__"):
                reward = float(reward[0])
            episode_reward += reward
            steps += 1

            # Flatten obs for next iteration
            obs = _flatten_obs(obs_raw)

        # Detect success. Isaac Lab's `info` mixes per-env tensors/arrays with
        # nested dicts (e.g. "log", "observations"). Only index into array-like
        # values to pull env 0 — indexing a dict with [0] raises KeyError, and
        # plain scalars/dicts pass through unchanged.
        def _env0(v):
            if isinstance(v, dict):
                return v
            if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:  # torch/np tensor
                return v[0]
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return v[0]
            return v
        info_single = {k: _env0(v) for k, v in info.items()}
        success = _detect_success(bool(terminated[0]), bool(truncated[0]), info_single, env)
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
