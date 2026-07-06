#!/usr/bin/env python3
"""
RealSense D405 RGB capture for episode recording and teleop.

Provides a thread-safe capture class that grabs RGB frames from a
USB-connected Intel RealSense D405 (short-range, wrist-mounted).
Only uses the RGB stream — GR00T N1 does not support depth.

On Linux with pyrealsense2 installed, uses the native RealSense SDK.
On macOS (or if pyrealsense2 is missing), falls back to OpenCV since
the D405 is UVC-compliant and appears as a standard webcam.

Usage:
    from robot.ur3.realsense_capture import RealSenseCapture

    cam = RealSenseCapture()
    cam.start()
    frame = cam.grab()   # numpy (H, W, 3) RGB uint8
    cam.stop()
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("realsense")

_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_FPS = 15

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    _HAS_RS = False


class RealSenseCapture:
    """Thread-safe RGB capture from a RealSense D405."""

    def __init__(
        self,
        serial: str | None = None,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps: int = _DEFAULT_FPS,
        device_index: int | None = None,
    ):
        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._device_index = device_index
        self._pipeline = None
        self._cv_cap = None
        self._use_cv = False
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        if _HAS_RS:
            try:
                self._start_realsense()
                return
            except Exception as e:
                log.warning("pyrealsense2 failed, falling back to OpenCV: %s", e)

        self._start_opencv()

    def _start_realsense(self) -> None:
        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)
        config.enable_stream(
            rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps
        )

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)

        device = profile.get_device()
        log.info(
            "RealSense started: %s (serial: %s)",
            device.get_info(rs.camera_info.name),
            device.get_info(rs.camera_info.serial_number),
        )

        for _ in range(30):
            self._pipeline.wait_for_frames(timeout_ms=1000)

        self._use_cv = False
        self._started = True

    def _start_opencv(self) -> None:
        import cv2

        idx = self._device_index if self._device_index is not None else _find_realsense_index()
        self._cv_cap = cv2.VideoCapture(idx)
        if not self._cv_cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {idx}")

        self._cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cv_cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Warm up auto-exposure
        for _ in range(30):
            self._cv_cap.read()

        self._use_cv = True
        self._started = True
        log.info("RealSense (OpenCV fallback) started at index %d", idx)

    def grab(self, timeout_ms: int = 1000) -> np.ndarray | None:
        """Return the latest RGB frame as (H, W, 3) uint8, or None on failure."""
        with self._lock:
            if not self._started:
                return None

            if self._use_cv:
                return self._grab_opencv()
            return self._grab_realsense(timeout_ms)

    def _grab_realsense(self, timeout_ms: int) -> np.ndarray | None:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color = frames.get_color_frame()
            if not color:
                return None
            return np.asarray(color.get_data())
        except Exception:
            log.debug("RealSense grab failed", exc_info=True)
            return None

    def _grab_opencv(self) -> np.ndarray | None:
        import cv2
        try:
            ret, frame = self._cv_cap.read()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            log.debug("OpenCV grab failed", exc_info=True)
            return None

    def stop(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None
            if self._cv_cap is not None:
                try:
                    self._cv_cap.release()
                except Exception:
                    pass
                self._cv_cap = None
            self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def __enter__(self) -> RealSenseCapture:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


def _find_realsense_index() -> int:
    """Try camera indices 0-4, return the first that opens. Default 0."""
    import cv2
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            name = cap.getBackendName()
            cap.release()
            log.debug("Camera index %d available (backend: %s)", i, name)
            return i
        cap.release()
    return 0


def find_devices() -> list[dict]:
    """List connected RealSense devices (requires pyrealsense2)."""
    if not _HAS_RS:
        return [{"name": "RealSense (OpenCV fallback)", "serial": "n/a"}]

    ctx = rs.context()
    devices = []
    for d in ctx.devices:
        devices.append({
            "name": d.get_info(rs.camera_info.name),
            "serial": d.get_info(rs.camera_info.serial_number),
        })
    return devices


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"pyrealsense2 available: {_HAS_RS}")
    print("Connected devices:")
    devs = find_devices()
    if not devs:
        print("  (none)")
    else:
        for d in devs:
            print(f"  {d['name']} — serial {d['serial']}")

    print("\nCapturing test frame...")
    with RealSenseCapture() as cam:
        frame = cam.grab()
        if frame is not None:
            print(f"  Got frame: {frame.shape} dtype={frame.dtype}")
        else:
            print("  Failed to capture frame")
