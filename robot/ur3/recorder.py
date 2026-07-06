#!/usr/bin/env python3
"""
Episode recorder for imitation learning.

Records synchronized robot telemetry (joints, TCP pose, force/torque, gripper)
and camera frames into Zarr episode stores.  Replaces the old mimic_loop.py
with a proper multi-modal, episode-based recording system.

Usage as library:
    recorder = EpisodeRecorder(robot, "pick_part", camera_url="http://localhost:8554/")
    episode_id = recorder.start()
    # ... human demonstrates ...
    metadata = recorder.stop(notes="first attempt")

Usage as CLI:
    uv run src/recorder.py list
    uv run src/recorder.py info episode_001_pick_part
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zarr

log = logging.getLogger("recorder")

# Where recorded episodes land. Defaults to the location `pai groot convert`
# reads from, so captured data flows straight into the Lab 1 pipeline. Override
# with PAI_EPISODES_DIR (the `pai groot record` command sets this).
_DEFAULT_EPISODES_DIR = Path(
    os.environ.get("PAI_EPISODES_DIR")
    or Path(__file__).resolve().parents[2] / "training" / "data" / "episodes" / "episodes"
)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# EpisodeRecorder
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """Records a single demonstration episode (telemetry + camera)."""

    def __init__(
        self,
        robot,
        task_name: str,
        episodes_dir: str | Path = _DEFAULT_EPISODES_DIR,
        camera_url: str | None = None,
        cameras: dict[str, str] | None = None,
        telemetry_hz: int = 10,
        camera_hz: int = 5,
    ):
        self._robot = robot
        self._task_name = task_name
        self._episodes_dir = Path(episodes_dir)
        self._telemetry_hz = telemetry_hz
        self._camera_hz = camera_hz

        # Build camera dict — cameras param takes priority over legacy camera_url
        if cameras:
            self._cameras = dict(cameras)
        elif camera_url:
            self._cameras = {"camera": camera_url}
        else:
            self._cameras = {}

        self._stop_event = threading.Event()
        self._recording = False
        self._episode_id: str | None = None

        # In-memory buffers (filled by threads, flushed on stop)
        self._telemetry: list[dict] = []
        self._gripper_events: list[tuple[float, int]] = []  # (timestamp, position)
        self._cam_frames: dict[str, list[tuple[float, np.ndarray]]] = {}

        self._telemetry_thread: threading.Thread | None = None
        self._camera_threads: list[threading.Thread] = []

    def start(self, external_telemetry: bool = False) -> str:
        """Start recording. Returns the episode ID.

        If external_telemetry=True, the caller must push telemetry via
        record_telemetry() instead of the internal polling thread.
        """
        if self._recording:
            raise RuntimeError("Already recording")

        self._episode_id = self._next_episode_id()
        self._episodes_dir.mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self._telemetry.clear()
        self._gripper_events.clear()
        self._cam_frames.clear()
        self._camera_threads.clear()
        self._recording = True

        if not external_telemetry:
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop, daemon=True, name="recorder-telemetry"
            )
            self._telemetry_thread.start()
        else:
            self._telemetry_thread = None

        for cam_name, cam_url in self._cameras.items():
            self._cam_frames[cam_name] = []
            t = threading.Thread(
                target=self._camera_loop, args=(cam_name, cam_url),
                daemon=True, name=f"recorder-{cam_name}",
            )
            t.start()
            self._camera_threads.append(t)

        log.info("Recording started: %s", self._episode_id)
        return self._episode_id

    def stop(self, notes: str = "", command_log: list[tuple[float, str]] | None = None) -> dict:
        """Stop recording, flush to Zarr, return metadata summary."""
        if not self._recording:
            raise RuntimeError("Not recording")

        self._stop_event.set()
        self._recording = False

        if self._telemetry_thread:
            self._telemetry_thread.join(timeout=5.0)
        for t in self._camera_threads:
            t.join(timeout=5.0)

        metadata = self._flush_to_zarr(notes, command_log=command_log)
        log.info(
            "Recording stopped: %s — %d telemetry, %d commands, %d frames, %.1fs",
            self._episode_id,
            metadata["telemetry_samples"],
            metadata.get("commands", 0),
            metadata["camera_frames"],
            metadata["duration_s"],
        )
        return metadata

    def is_recording(self) -> bool:
        return self._recording

    def record_telemetry(self, joints: list[float], tcp: list[float],
                         force: list[float] | None = None) -> None:
        """Push a telemetry sample directly (call from the control loop)."""
        if self._recording:
            self._telemetry.append({
                "t": time.time(),
                "joints": joints or [0.0] * 6,
                "tcp": tcp or [0.0] * 6,
                "force": force or [0.0] * 6,
            })

    def record_gripper_event(self, position: int) -> None:
        """Record a gripper event during active recording."""
        if self._recording:
            self._gripper_events.append((time.time(), position))

    # -- threads -------------------------------------------------------------

    def _telemetry_loop(self) -> None:
        interval = 1.0 / self._telemetry_hz
        while not self._stop_event.is_set():
            try:
                joints = self._robot.get_joints()
                tcp = self._robot.get_tcp_pose()
                force = self._robot.get_force()
                self._telemetry.append(
                    {
                        "t": time.time(),
                        "joints": joints or [0.0] * 6,
                        "tcp": tcp or [0.0] * 6,
                        "force": force or [0.0] * 6,
                    }
                )
            except Exception:
                log.debug("Telemetry read failed", exc_info=True)
            self._stop_event.wait(interval)

    def _camera_loop(self, cam_name: str, cam_url: str) -> None:
        if cam_url == "realsense" or cam_url.startswith("realsense:"):
            self._realsense_loop(cam_name, cam_url)
        else:
            self._rtsp_loop(cam_name, cam_url)

    def _rtsp_loop(self, cam_name: str, cam_url: str) -> None:
        import cv2

        interval = 1.0 / self._camera_hz
        cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            log.warning("Camera '%s' unavailable at %s — skipping", cam_name, cam_url)
            return

        try:
            while not self._stop_event.is_set():
                cap.grab()
                cap.grab()
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._cam_frames[cam_name].append((time.time(), rgb))
                self._stop_event.wait(interval)
        finally:
            cap.release()

    def _realsense_loop(self, cam_name: str, cam_url: str) -> None:
        from robot.ur3.realsense_capture import RealSenseCapture

        suffix = cam_url.split(":", 1)[1] if ":" in cam_url else None
        interval = 1.0 / self._camera_hz

        kwargs = {"fps": max(self._camera_hz, 15)}
        if suffix is not None and suffix.isdigit():
            kwargs["device_index"] = int(suffix)
        elif suffix is not None:
            kwargs["serial"] = suffix

        cam = RealSenseCapture(**kwargs)
        try:
            cam.start()
            print(f"  RealSense '{cam_name}' started (opencv_fallback={cam._use_cv})")
        except Exception as e:
            print(f"  WARNING: RealSense '{cam_name}' unavailable: {e}")
            return

        try:
            while not self._stop_event.is_set():
                frame = cam.grab()
                if frame is not None:
                    self._cam_frames[cam_name].append((time.time(), frame))
                self._stop_event.wait(interval)
        finally:
            cam.stop()

    # -- Zarr flush ----------------------------------------------------------

    @staticmethod
    def _save_array(group, name: str, data: np.ndarray, **kwargs):
        """Zarr 3 compatible dataset creation from numpy array."""
        group.create_dataset(name, shape=data.shape, dtype=data.dtype, **kwargs)
        group[name][:] = data

    def _flush_to_zarr(self, notes: str, command_log: list[tuple[float, str]] | None = None) -> dict:
        episode_dir = self._episodes_dir / self._episode_id
        root = zarr.open(str(episode_dir), mode="w")

        n_telem = len(self._telemetry)
        total_frames = sum(len(frames) for frames in self._cam_frames.values())

        # -- observations ----------------------------------------------------
        obs = root.create_group("observations")

        if n_telem > 0:
            timestamps = np.array([s["t"] for s in self._telemetry], dtype=np.float64)
            joints = np.array([s["joints"] for s in self._telemetry], dtype=np.float64)
            tcp_pose = np.array([s["tcp"] for s in self._telemetry], dtype=np.float64)
            force_torque = np.array([s["force"] for s in self._telemetry], dtype=np.float64)

            self._save_array(obs, "timestamps", timestamps)
            self._save_array(obs, "joints", joints)
            self._save_array(obs, "tcp_pose", tcp_pose)
            self._save_array(obs, "force_torque", force_torque)

            gripper_pos = self._build_gripper_array(timestamps)
            self._save_array(obs, "gripper_position", gripper_pos)
        else:
            self._save_array(obs, "timestamps", np.array([], dtype=np.float64))
            obs.create_dataset("joints", shape=(0, 6), dtype=np.float64)
            obs.create_dataset("tcp_pose", shape=(0, 6), dtype=np.float64)
            obs.create_dataset("force_torque", shape=(0, 6), dtype=np.float64)
            self._save_array(obs, "gripper_position", np.array([], dtype=np.uint8))

        # -- images ----------------------------------------------------------
        images = root.create_group("images")

        for cam_name, frames in self._cam_frames.items():
            if not frames:
                self._save_array(images, f"{cam_name}_timestamps", np.array([], dtype=np.float64))
                continue
            cam_timestamps = np.array([f[0] for f in frames], dtype=np.float64)
            frame_stack = np.stack([f[1] for f in frames])
            self._save_array(images, cam_name, frame_stack,
                             chunks=(1, *frame_stack.shape[1:]))
            self._save_array(images, f"{cam_name}_timestamps", cam_timestamps)

        # -- actions ---------------------------------------------------------
        actions = root.create_group("actions")

        if n_telem > 1:
            action_joints = np.array([s["joints"] for s in self._telemetry], dtype=np.float64)
            action_joints[:-1] = action_joints[1:]
            self._save_array(actions, "joints", action_joints)
        elif n_telem == 1:
            self._save_array(actions, "joints",
                             np.array([self._telemetry[0]["joints"]], dtype=np.float64))
        else:
            actions.create_dataset("joints", shape=(0, 6), dtype=np.float64)

        # -- URScript command log (raw commands for replay) ------------------
        n_cmds = 0
        if command_log:
            import json as _json
            cmd_file = episode_dir / "commands.json"
            cmd_data = [{"t": t, "prog": p} for t, p in command_log]
            with open(cmd_file, "w") as f:
                _json.dump(cmd_data, f)
            n_cmds = len(command_log)

        # -- metadata --------------------------------------------------------
        duration = 0.0
        if n_telem >= 2:
            duration = self._telemetry[-1]["t"] - self._telemetry[0]["t"]

        metadata = {
            "episode_id": self._episode_id,
            "task_name": self._task_name,
            "duration_s": round(duration, 2),
            "telemetry_hz": self._telemetry_hz,
            "camera_hz": self._camera_hz,
            "telemetry_samples": n_telem,
            "commands": n_cmds,
            "camera_frames": total_frames,
            "notes": notes,
            "schema_version": SCHEMA_VERSION,
        }
        root.attrs.update(metadata)

        metadata["path"] = str(episode_dir)
        return metadata

    def _build_gripper_array(self, timestamps: np.ndarray) -> np.ndarray:
        """Build gripper position array aligned to telemetry timestamps."""
        gripper = np.zeros(len(timestamps), dtype=np.uint8)
        if not self._gripper_events:
            return gripper

        # Sort events by time
        events = sorted(self._gripper_events, key=lambda e: e[0])
        event_idx = 0
        current_pos = 0

        for i, t in enumerate(timestamps):
            while event_idx < len(events) and events[event_idx][0] <= t:
                current_pos = events[event_idx][1]
                event_idx += 1
            gripper[i] = current_pos

        return gripper

    # -- helpers -------------------------------------------------------------

    def _next_episode_id(self) -> str:
        """Generate the next episode ID by scanning existing episodes."""
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self._episodes_dir.iterdir()) if self._episodes_dir.exists() else []
        max_num = 0
        for p in existing:
            if p.is_dir() and p.name.startswith("episode_"):
                try:
                    num = int(p.name.split("_")[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    pass
        next_num = max_num + 1
        safe_name = self._task_name.replace(" ", "_").replace("/", "_")
        return f"episode_{next_num:03d}_{safe_name}"


# ---------------------------------------------------------------------------
# EpisodeStore (read-only access)
# ---------------------------------------------------------------------------


class EpisodeStore:
    """Read-only access to recorded episodes."""

    def __init__(self, episodes_dir: str | Path = _DEFAULT_EPISODES_DIR):
        self._episodes_dir = Path(episodes_dir)

    def list_episodes(self) -> list[dict]:
        """List all episodes with metadata."""
        if not self._episodes_dir.exists():
            return []
        episodes = []
        for p in sorted(self._episodes_dir.iterdir()):
            if not p.is_dir() or not p.name.startswith("episode_"):
                continue
            try:
                root = zarr.open(str(p), mode="r")
                attrs = dict(root.attrs)
                attrs["episode_id"] = p.name
                attrs["path"] = str(p)
                episodes.append(attrs)
            except Exception:
                log.debug("Skipping invalid episode dir: %s", p)
        return episodes

    def load_episode(self, episode_id: str) -> zarr.Group:
        """Load an episode as a Zarr group."""
        path = self._episodes_dir / episode_id
        if not path.exists():
            raise FileNotFoundError(f"Episode not found: {episode_id}")
        return zarr.open(str(path), mode="r")

    def get_trajectory(self, episode_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load (timestamps, joints) for replay."""
        root = self.load_episode(episode_id)
        timestamps = np.array(root["observations"]["timestamps"])
        joints = np.array(root["observations"]["joints"])
        return timestamps, joints

    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode directory. Returns True if deleted."""
        path = self._episodes_dir / episode_id
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run src/recorder.py list")
        print("  uv run src/recorder.py info <episode_id>")
        sys.exit(1)

    cmd = sys.argv[1]
    store = EpisodeStore()

    if cmd == "list":
        episodes = store.list_episodes()
        if not episodes:
            print("No episodes recorded yet.")
            print(f"  (episodes dir: {store._episodes_dir})")
            return
        print(f"{len(episodes)} episode(s):\n")
        for ep in episodes:
            print(f"  {ep.get('episode_id', '?'):30s}  "
                  f"task={ep.get('task_name', '?'):15s}  "
                  f"duration={ep.get('duration_s', 0):6.1f}s  "
                  f"samples={ep.get('telemetry_samples', 0):5d}  "
                  f"frames={ep.get('camera_frames', 0):4d}")
            if ep.get("notes"):
                print(f"    notes: {ep['notes']}")

    elif cmd == "info":
        if len(sys.argv) < 3:
            print("Usage: uv run src/recorder.py info <episode_id>")
            sys.exit(1)
        episode_id = sys.argv[2]
        try:
            root = store.load_episode(episode_id)
        except FileNotFoundError as e:
            print(str(e))
            sys.exit(1)

        print(f"Episode: {episode_id}\n")
        print("Metadata:")
        for k, v in sorted(root.attrs.items()):
            print(f"  {k}: {v}")
        print(f"\nStructure:")
        print(root.tree())

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: list, info")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _cli()
