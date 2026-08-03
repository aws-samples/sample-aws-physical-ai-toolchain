#!/usr/bin/env python3
"""
Convert Zarr episodes to LeRobot v2 format for GR00T fine-tuning.

Reads raw teleop episodes (Zarr) and writes a LeRobot v2 dataset in the
format expected by NVIDIA GR00T's LeRobotSingleDataset.

Usage (from the repo root):
    # Convert all episodes (after: unzip ur3_episodes_001_027.zip -d training/data/episodes)
    python training/gr00t/convert_zarr_to_lerobot.py \
        --episodes-dir training/data/episodes/episodes \
        --output-dir   training/data/ur3_lerobot_dataset

    # Convert specific episodes
    python training/gr00t/convert_zarr_to_lerobot.py \
        --episodes-dir training/data/episodes/episodes \
        --output-dir   training/data/ur3_lerobot_dataset \
        -e episode_001_pick episode_002_pick

    # Then upload to S3 (BUCKET = DatasetsBucketName stack output)
    aws s3 sync training/data/ur3_lerobot_dataset/ "s3://$BUCKET/groot-data/ur3/dataset/"

    # Then train (see workshop/lab-1-train-groot.md)
    python training/gr00t/pipeline.py --execute --dataset-prefix groot-data/ur3 --max-steps 100

Bring your own data: the input Zarr schema is documented in docs/zarr-schema.md.
Point --episodes-dir at your own recordings and the same script applies.

Prerequisites:
    pip install -r training/requirements.txt   # zarr, numpy, opencv-python, pandas, pyarrow
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import zarr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Canonical paths (match README unzip target + Lab 1 Step 2). The zip's internal
# root is `episodes/`, so `unzip ... -d training/data/episodes` yields
# training/data/episodes/episodes/episode_*.
EPISODES_DIR = Path("training/data/episodes/episodes")
OUTPUT_DIR = Path("training/data/ur3_lerobot_dataset")

# UR3 has 6 joints + 1 gripper = 7D state/action
STATE_DIM = 7   # 6 joints + 1 gripper
ACTION_DIM = 7  # 6 Cartesian velocity × dt + 1 gripper


def _require_zarr_v3() -> None:
    """Fail fast with one clear message if zarr is too old.

    The bundled episodes are zarr v3 format (zarr.json metadata). zarr 2.x can't
    read them — it looks for .zgroup, finds none, and raises PathNotFoundError
    ("nothing found at path '") for every episode. Catch that up front rather than
    emitting 27 identical tracebacks.
    """
    major = int(zarr.__version__.split(".")[0])
    if major < 3:
        print(
            f"ERROR: zarr {zarr.__version__} is too old — the episodes are zarr v3 "
            f"format and need zarr>=3.\n"
            f"  Upgrade in this environment:  pip install 'zarr>=3'\n"
            f"  (or reinstall the toolchain:  pip install -e .)",
            file=sys.stderr,
        )
        sys.exit(1)


def list_episodes(episodes_dir: Path) -> list[str]:
    if not episodes_dir.exists():
        return []
    return sorted(
        p.name for p in episodes_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )


def _parse_commands(episode_dir: Path) -> list[dict] | None:
    """Load commands.json and parse velocity/gripper from URScript strings."""
    cmd_file = episode_dir / "commands.json"
    if not cmd_file.exists():
        return None

    import re
    with open(cmd_file) as f:
        raw = json.load(f)

    parsed = []
    for entry in raw:
        t = entry["t"]
        prog = entry["prog"]

        m = re.search(r"speedl\(\[([^\]]+)\]", prog)
        if m:
            vals = [float(x) for x in m.group(1).split(",")]
            parsed.append({"t": t, "type": "speedl", "vel": vals})
            continue

        if "stopj" in prog or "stopl" in prog:
            parsed.append({"t": t, "type": "stop", "vel": [0.0] * 6})
            continue

        m = re.search(r'socket_set_var\("POS",\s*(\d+)', prog)
        if m:
            pos = int(m.group(1))
            parsed.append({"t": t, "type": "gripper", "position": pos})
            continue

    return parsed


def convert_episode(episodes_dir: Path, episode_id: str, episode_index: int,
                    output_dir: Path, global_frame_offset: int,
                    task_to_index: dict[str, int],
                    camera_key: str = "wrist", output_camera_key: str = "wrist") -> dict:
    """Convert one Zarr episode to LeRobot v2 parquet + video.

    When commands.json is available, uses the commanded Cartesian velocities
    scaled by dt as EEF actions (matching speedl teleop). Observation joint
    state comes from Zarr telemetry.

    `task_to_index` maps each episode's task description to its global task index
    (the same ordering written to tasks.jsonl), so the per-frame `task_index` is
    the episode's REAL task — not a hardcoded 0 (which would collapse a multi-task
    dataset to a single task and drop the language label a VLA conditions on).

    Returns metadata dict for episodes.jsonl.
    """
    episode_dir = episodes_dir / episode_id
    root = zarr.open(str(episode_dir), mode="r")
    task_name = root.attrs.get("task_name", "manipulation task").replace("_", " ")
    # Defensive: the pre-pass (build_task_index) normally contains every task, but
    # if an episode was unreadable there yet readable here, fall back to appending
    # it rather than KeyError-ing. Keeps the map and tasks.jsonl consistent.
    if task_name not in task_to_index:
        task_to_index[task_name] = len(task_to_index)
    task_index = task_to_index[task_name]

    # Load telemetry
    telem_ts = np.array(root["observations"]["timestamps"])
    joints = np.array(root["observations"]["joints"])       # (N, 6) rad
    gripper = np.array(root["observations"]["gripper_position"])  # (N,) 0-255

    # Load camera
    cam_frames = root["images"][camera_key]                  # (M, H, W, 3)
    cam_ts = np.array(root["images"][f"{camera_key}_timestamps"])
    n_frames = len(cam_ts)

    # Parse commanded actions from commands.json
    commands = _parse_commands(episode_dir)

    # Normalize gripper: 0-255 → 0.0-1.0
    gripper_norm = gripper.astype(np.float64) / 255.0

    # Build action lookup from commands
    if commands:
        vel_cmds = [(c["t"], c["vel"]) for c in commands if c["type"] in ("speedl", "stop")]
        grip_cmds = [(c["t"], c["position"]) for c in commands if c["type"] == "gripper"]
    else:
        vel_cmds = []
        grip_cmds = []

    def _action_at(t: float, dt: float) -> list[float]:
        """Get 7D action at time t: 6 velocity*dt deltas + 1 gripper absolute."""
        vel = [0.0] * 6
        for ct, v in vel_cmds:
            if ct > t:
                break
            vel = v
        # Scale velocity by timestep to get delta (units: m/step for position, rad/step for rotation)
        deltas = [v * dt for v in vel]

        grip = 0.0
        for ct, p in grip_cmds:
            if ct > t:
                break
            grip = float(p) / 255.0

        return deltas + [grip]

    fps = root.attrs.get("camera_hz", 5)
    step_dt = 1.0 / fps

    rows = []
    for frame_idx in range(n_frames):
        ft = cam_ts[frame_idx]
        telem_idx = int(np.argmin(np.abs(telem_ts - ft)))

        # State: 6 joints + 1 gripper
        state = np.concatenate([joints[telem_idx], [gripper_norm[telem_idx]]])

        # Action: 6 deltas + 1 gripper (absolute)
        if vel_cmds:
            action = _action_at(ft, step_dt)
        else:
            # Fallback for old episodes without commands.json
            if telem_idx < len(joints) - 1:
                deltas = (joints[telem_idx + 1] - joints[telem_idx]).tolist()
            else:
                deltas = [0.0] * 6
            grip_abs = float(gripper_norm[telem_idx])
            action = deltas + [grip_abs]

        is_last = frame_idx == n_frames - 1

        rows.append({
            "observation.state": state.tolist(),
            "action": action,
            "episode_index": episode_index,
            "frame_index": frame_idx,
            "timestamp": float(ft - cam_ts[0]),
            "next.reward": 1.0 if is_last else 0.0,
            "next.done": is_last,
            "next.success": is_last,
            "index": global_frame_offset + frame_idx,
            "task_index": task_index,
        })

    # Write parquet
    df = pd.DataFrame(rows)
    parquet_dir = output_dir / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"
    df.to_parquet(str(parquet_path), index=False)

    # Write video — use output_camera_key ("wrist") as directory name for GR00T compatibility
    video_dir = output_dir / "videos" / "chunk-000" / f"observation.images.{output_camera_key}"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"episode_{episode_index:06d}.mp4"

    h, w = cam_frames.shape[1], cam_frames.shape[2]
    fps = root.attrs.get("camera_hz", 5)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for i in range(n_frames):
        frame_rgb = np.array(cam_frames[i])
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

    action_src = f"{len(vel_cmds)} commands" if vel_cmds else "joint deltas (no commands.json)"
    print(f"  {episode_id} → episode_{episode_index:06d} "
          f"({n_frames} frames, {len(telem_ts)} telem, actions from {action_src})")

    return {
        "episode_index": episode_index,
        "tasks": [task_name],
        "length": n_frames,
    }


def write_metadata(output_dir: Path, episode_metas: list[dict],
                    task_to_index: dict[str, int],
                    output_camera_key: str = "wrist",
                    image_shape: tuple = (480, 640, 3), fps: int = 5):
    """Write meta/ directory: modality.json, episodes.jsonl, info.json, tasks.jsonl.

    `task_to_index` is the SAME map used to stamp per-frame task_index in the
    parquet, so tasks.jsonl and the data agree.
    """
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # modality.json — use output_camera_key to match GR00T's expected "wrist"
    video_config = {
        output_camera_key: {"original_key": f"observation.images.{output_camera_key}"}
    }

    modality = {
        "state": {
            "arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "action": {
            "arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": video_config,
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for meta in episode_metas:
            f.write(json.dumps(meta) + "\n")

    # info.json — must include "features" for GR00T's generate_stats
    total_frames = sum(m["length"] for m in episode_metas)
    total_eps = len(episode_metas)

    features = {
        f"observation.images.{output_camera_key}": {
            "dtype": "video",
            "shape": list(image_shape),
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float64",
            "shape": [STATE_DIM],
            "names": [
                "shoulder_pan", "shoulder_lift", "elbow",
                "wrist_1", "wrist_2", "wrist_3", "gripper",
            ],
        },
        "action": {
            "dtype": "float64",
            "shape": [ACTION_DIM],
            "names": [
                "vx", "vy", "vz",
                "rx", "ry", "rz", "gripper",
            ],
        },
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "next.reward": {"dtype": "float64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "next.success": {"dtype": "bool", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "ur3",
        "total_episodes": total_eps,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_eps,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # tasks.jsonl — written from the shared task_to_index map (same indices the
    # parquet task_index column uses), ordered by index.
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")


def _compute_stats(arr: np.ndarray) -> dict:
    """Per-dimension stats (mean/std/min/max/q01/q99). Matches the proven reference."""
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.percentile(arr, 1, axis=0).tolist(),
        "q99": np.percentile(arr, 99, axis=0).tolist(),
    }


def write_stats(output_dir: Path):
    """Write meta/stats.json by reading back the written parquet (states+actions).

    GR00T's data pipeline requires meta/stats.json for normalization, and the
    inference loader asserts it exists. (Our earlier converter dropped this — it
    is present in the proven reference pipeline.)
    """
    meta_dir = output_dir / "meta"
    parquet_files = sorted((output_dir / "data").rglob("*.parquet"))
    if not parquet_files:
        print("  WARNING: no parquet files found, skipping stats.json")
        return

    states, actions, timestamps = [], [], []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        if "observation.state" in df.columns:
            states.append(np.stack(df["observation.state"].values))
        if "action" in df.columns:
            actions.append(np.stack(df["action"].values))
        if "timestamp" in df.columns:
            timestamps.append(np.asarray(df["timestamp"].values, dtype=np.float64))

    stats = {}
    if states:
        stats["observation.state"] = _compute_stats(np.concatenate(states))
    if actions:
        stats["action"] = _compute_stats(np.concatenate(actions))
    if timestamps:
        stats["timestamp"] = _compute_stats(np.concatenate(timestamps).reshape(-1, 1))

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote meta/stats.json ({len(stats)} feature(s))")


def build_task_index(episodes_dir: Path, episode_ids: list[str]) -> dict[str, int]:
    """Pre-pass: read each episode's task_name and build a deterministic
    task -> index map (sorted by description). This is the single source of truth
    for both the per-frame task_index in the parquet and the tasks.jsonl entries,
    so a multi-task dataset keeps its distinct task/language labels.
    """
    tasks = set()
    for ep_id in episode_ids:
        try:
            root = zarr.open(str(episodes_dir / ep_id), mode="r")
            tasks.add(root.attrs.get("task_name", "manipulation task").replace("_", " "))
        except Exception:
            # Unreadable episodes are skipped here; convert_episode will also error
            # on them and they'll be dropped from the output.
            pass
    return {task: i for i, task in enumerate(sorted(tasks))}


def main():
    parser = argparse.ArgumentParser(description="Convert Zarr episodes to LeRobot v2")
    parser.add_argument("--episodes-dir", default=str(EPISODES_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("-e", "--episodes", nargs="+", default=None,
                        help="Specific episode IDs (default: all)")
    parser.add_argument("--camera", default="wrist",
                        help="Primary camera key in Zarr images group (default: wrist)")
    args = parser.parse_args()

    _require_zarr_v3()

    episodes_dir = Path(args.episodes_dir)
    output_dir = Path(args.output_dir)

    if args.episodes:
        episode_ids = args.episodes
    else:
        episode_ids = list_episodes(episodes_dir)

    if not episode_ids:
        print(f"No episodes found in {episodes_dir}")
        sys.exit(1)

    print(f"Converting {len(episode_ids)} episodes to LeRobot v2 format")
    print(f"  Source: {episodes_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Pre-pass: build the task -> index map so per-frame task_index is the real task.
    task_to_index = build_task_index(episodes_dir, episode_ids)
    print(f"  Tasks: {len(task_to_index)} distinct")

    episode_metas = []
    global_frame_offset = 0

    for i, ep_id in enumerate(episode_ids):
        ep_index = i
        try:
            meta = convert_episode(
                episodes_dir, ep_id, ep_index, output_dir,
                global_frame_offset, task_to_index, camera_key=args.camera,
            )
            global_frame_offset += meta["length"]
            episode_metas.append(meta)
        except Exception as e:
            print(f"  ERROR converting {ep_id}: {e}")
            import traceback
            traceback.print_exc()

    if not episode_metas:
        print("No episodes converted successfully.")
        sys.exit(1)

    # Get image shape and fps from first episode
    first_ep = zarr.open(str(episodes_dir / episode_ids[0]), mode="r")
    image_shape = tuple(first_ep["images"][args.camera].shape[1:])  # (H, W, 3)
    fps = first_ep.attrs.get("camera_hz", 5)

    write_metadata(output_dir, episode_metas, task_to_index,
                   output_camera_key="wrist",
                   image_shape=image_shape, fps=fps)
    write_stats(output_dir)

    total_frames = sum(m["length"] for m in episode_metas)
    print(f"\nDone! Converted {len(episode_metas)} episodes ({total_frames} total frames)")
    print(f"\nOutput: {output_dir}/")
    print(f"  data/chunk-000/       ({len(episode_metas)} parquet files)")
    print(f"  videos/chunk-000/     ({len(episode_metas)} mp4 files per camera)")
    print(f"  meta/                 (modality.json, episodes.jsonl, info.json, tasks.jsonl, stats.json)")
    print(f"\nNext steps:")
    print(f'  aws s3 sync {output_dir}/ "s3://$BUCKET/groot-data/ur3/dataset/"')
    print(f"  python training/gr00t/pipeline.py --execute --dataset-prefix groot-data/ur3 --max-steps 100")


if __name__ == "__main__":
    main()
