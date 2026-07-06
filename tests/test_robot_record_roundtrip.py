"""Record -> convert round-trip for the UR3 hardware track (no hardware, no AWS).

The seam this guards: `pai groot record` (robot/ur3/recorder.py) writes Zarr
episodes, and `pai groot convert` (training/groot/convert_zarr_to_lerobot.py) must
consume them. A schema drift between the two would silently produce a broken
training dataset — so we drive the REAL recorder flush with a fake robot + synthetic
frames, then run the REAL converter on the output and assert the LeRobot v2 result.

Everything runs on a laptop: no UR3, no camera, no threads, no GPU, no AWS.
"""
import importlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The converter needs zarr>=3 + opencv + pandas + pyarrow. Skip cleanly if absent.
_MISSING = [m for m in ("zarr", "cv2", "pandas", "pyarrow")
            if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"data-conversion deps missing: {_MISSING}"
)


def _record_synthetic_episode(episodes_dir):
    """Write one episode via the real EpisodeRecorder flush path (no I/O threads)."""
    from robot.ur3.recorder import EpisodeRecorder

    rec = EpisodeRecorder(robot=None, task_name="pick up the red cube",
                          episodes_dir=episodes_dir, telemetry_hz=10, camera_hz=5)
    rec._episode_id = "episode_001_pick_up_the_red_cube"

    t0 = 1_000_000.0
    # 20 telemetry samples @ 10 Hz
    rec._telemetry = [
        {"t": t0 + i * 0.1,
         "joints": [0.05 * i, -1.57, 1.57, -1.57, -1.57, 0.0],
         "tcp": [0.0] * 6, "force": [0.0] * 6}
        for i in range(20)
    ]
    # gripper opens, then closes at t0+1.0
    rec._gripper_events = [(t0 + 0.0, 0), (t0 + 1.0, 255)]
    # 10 camera frames @ 5 Hz, small RGB
    rng = np.random.RandomState(0)
    rec._cam_frames = {
        "wrist": [(t0 + 0.05 + i * 0.2,
                   (rng.rand(48, 64, 3) * 255).astype(np.uint8))
                  for i in range(10)]
    }
    # URScript command log exactly as the teleop sender emits it
    command_log = [
        (t0 + 0.05, "def cmd():\n  speedl([0,0,-0.035,0,0,0],1.0,0.12)\nend\n"),
        (t0 + 1.0,  'def cmd():\n  socket_set_var("POS", 255)\nend\n'),
    ]
    meta = rec._flush_to_zarr(notes="test", command_log=command_log)
    return rec._episode_id, meta


def test_recorder_writes_documented_zarr_schema(tmp_path):
    """The recorder emits exactly the keys docs/ZARR_SCHEMA.md promises."""
    import zarr

    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    ep_id, meta = _record_synthetic_episode(episodes_dir)

    root = zarr.open(str(episodes_dir / ep_id), mode="r")

    # observations: the load-bearing keys the converter reads
    obs = set(root["observations"].keys())
    assert {"timestamps", "joints", "gripper_position"} <= obs
    assert np.array(root["observations"]["joints"]).shape == (20, 6)
    assert np.array(root["observations"]["gripper_position"]).shape == (20,)

    # images: wrist frames + their own clock
    imgs = set(root["images"].keys())
    assert {"wrist", "wrist_timestamps"} <= imgs
    assert np.array(root["images"]["wrist"]).shape == (10, 48, 64, 3)

    # attrs the converter uses
    assert root.attrs["task_name"] == "pick up the red cube"
    assert root.attrs["camera_hz"] == 5

    # commands.json present and shaped {"t","prog"}
    cmds = json.load(open(episodes_dir / ep_id / "commands.json"))
    assert cmds and set(cmds[0]) == {"t", "prog"}


def test_record_then_convert_produces_valid_lerobot_dataset(tmp_path):
    """Full seam: recorded Zarr -> real converter -> valid LeRobot v2 dataset."""
    import pandas as pd

    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    ep_id, _ = _record_synthetic_episode(episodes_dir)

    conv = importlib.import_module("training.groot.convert_zarr_to_lerobot")
    out_dir = tmp_path / "lerobot"

    task_map = conv.build_task_index(episodes_dir, [ep_id])
    assert task_map == {"pick up the red cube": 0}

    em = conv.convert_episode(episodes_dir, ep_id, 0, out_dir, 0, task_map)
    conv.write_metadata(out_dir, [em], task_map, image_shape=(48, 64, 3), fps=5)
    conv.write_stats(out_dir)

    # --- parquet: 10 frames, 7D state, 7D action ---
    df = pd.read_parquet(out_dir / "data" / "chunk-000" / "episode_000000.parquet")
    assert len(df) == 10
    assert len(df.iloc[0]["observation.state"]) == 7
    assert len(df.iloc[0]["action"]) == 7

    # --- the URScript command actually flows into the action ---
    # speedl vz = -0.035 m/s, dt = 1/5 s  =>  z-delta = -0.007 in the first frame.
    assert df.iloc[0]["action"][2] == pytest.approx(-0.035 * (1.0 / 5.0))

    # --- every meta file GR00T's loader requires exists ---
    meta_dir = out_dir / "meta"
    for name in ("modality.json", "episodes.jsonl", "info.json",
                 "tasks.jsonl", "stats.json"):
        assert (meta_dir / name).is_file(), f"missing meta/{name}"

    info = json.load(open(meta_dir / "info.json"))
    assert info["robot_type"] == "ur3"
    assert info["fps"] == 5
    assert info["total_frames"] == 10

    # --- video written under the GR00T-expected key ---
    video = (out_dir / "videos" / "chunk-000"
             / "observation.images.wrist" / "episode_000000.mp4")
    assert video.is_file() and video.stat().st_size > 0

    # --- stats cover state + action (normalization inputs) ---
    stats = json.load(open(meta_dir / "stats.json"))
    assert "observation.state" in stats and "action" in stats
