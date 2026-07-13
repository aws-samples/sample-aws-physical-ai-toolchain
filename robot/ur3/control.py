#!/usr/bin/env python3.10
"""
Closed-loop control using GR00T N1 VLA.

Captures wrist camera frame and joint state, sends to the GR00T endpoint,
receives a 16-step action chunk, executes the first N steps at 5Hz using
speedl (same streaming approach as gamepad_teleop), then re-queries.

Usage:
    python3 examples/08_groot_control.py "pick up the cube"
    python3 examples/08_groot_control.py --max-queries 5 "push the block left"
    python3 examples/08_groot_control.py --endpoint groot-finetuned "stack the blocks"
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import socket
import threading
import time

import cv2
import numpy as np

from robot.ur3.safe_controller import SafeUR3Controller
from robot.ur3.config import UR3_CONFIG

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROBOT_IP = os.environ.get("ROBOT_IP", "127.0.0.1")
ENDPOINT_NAME = os.environ.get("GROOT_ENDPOINT", "groot-ur3")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

START_JOINTS_DEG = [38.5, -90.4, 78.5, -90.2, -92.7, 12.5]


# ---------------------------------------------------------------------------
# Persistent URScript sender (same as gamepad_teleop.py)
# ---------------------------------------------------------------------------
class URScriptSender:
    def __init__(self, host, port=30002):
        self.host = host
        self.port = port
        self._sock = None
        self._running = False

    def connect(self):
        self.close()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(2.0)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((self.host, self.port))
        self._running = True
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        while self._running and self._sock:
            try:
                data = self._sock.recv(4096)
                if not data:
                    break
            except (socket.timeout, OSError):
                continue

    def send(self, program):
        try:
            self._sock.sendall(program.encode("utf-8"))
        except (socket.error, OSError, AttributeError):
            self.connect()
            self._sock.sendall(program.encode("utf-8"))

    def send_speedl(self, velocities, accel=1.0, duration=0.25):
        v = "[" + ",".join(f"{x:.5f}" for x in velocities) + "]"
        self.send(f"def cmd():\n  speedl({v},{accel},{duration:.2f})\nend\n")

    def send_stopj(self, decel=3.0):
        self.send(f"def cmd():\n  stopj({decel})\nend\n")

    def close(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------
class WristCameraCapture:
    def __init__(self):
        self._cam = None

    def start(self):
        from robot.ur3.realsense_capture import RealSenseCapture
        self._cam = RealSenseCapture(fps=15)
        try:
            self._cam.start()
        except Exception as e:
            print(f"  WARNING: Wrist camera unavailable: {e}")
            self._cam = None

    def capture(self) -> dict[str, bytes]:
        images = {}
        if self._cam is not None:
            frame = self._cam.grab()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                images["wrist"] = jpeg.tobytes()
        return images

    def stop(self):
        if self._cam is not None:
            self._cam.stop()
            self._cam = None


# ---------------------------------------------------------------------------
# GR00T endpoint
# ---------------------------------------------------------------------------
def query_groot(runtime, images: dict[str, bytes], state: list[float],
                task: str, endpoint: str) -> np.ndarray:
    import requests as _requests

    images_b64 = {k: base64.b64encode(v).decode("ascii") for k, v in images.items()}
    payload = {"images": images_b64, "state": state, "task": task}

    if endpoint.startswith("http"):
        resp = _requests.post(f"{endpoint}/invocations", json=payload)
        body = resp.json()
    else:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        body = json.loads(response["Body"].read())

    actions = np.array(body["actions"], dtype=np.float32)
    if actions.ndim == 3:
        actions = actions[0]
    return actions


# ---------------------------------------------------------------------------
# Action execution via persistent socket (like gamepad_teleop)
# ---------------------------------------------------------------------------
def execute_action_chunk(robot, sender: URScriptSender, actions: np.ndarray,
                         gripper_state: float, execute_steps: int) -> float:
    step_interval = 1.0 / UR3_CONFIG.control_hz
    max_d = UR3_CONFIG.max_vel_delta
    max_vel = 0.3  # m/s — conservative cap for Cartesian velocity

    for i in range(min(execute_steps, len(actions))):
        step_start = time.time()

        # Force safety check
        force = robot.get_force()
        if force:
            force_mag = math.sqrt(sum(f * f for f in force[:3]))
            if force_mag > UR3_CONFIG.force_limit:
                print(f"  SAFETY: Force {force_mag:.1f}N > {UR3_CONFIG.force_limit}N")
                sender.send_stopj()
                return gripper_state

        # Model outputs Cartesian velocity × dt deltas; convert back to velocities
        deltas = actions[i, :6].astype(float)
        clipped = np.clip(deltas, -max_d, max_d)
        velocities = [float(d) / step_interval for d in clipped]
        velocities = [max(-max_vel, min(max_vel, v)) for v in velocities]

        sender.send_speedl(velocities, accel=1.0)

        # Gripper
        gripper_cmd = float(actions[i, 6])
        if gripper_cmd > 0.5 and gripper_state <= 0.5:
            robot.gripper_close()
            gripper_state = 1.0
            print(f"  Step {i}: gripper CLOSE")
        elif gripper_cmd <= 0.5 and gripper_state > 0.5:
            robot.gripper_open()
            gripper_state = 0.0
            print(f"  Step {i}: gripper OPEN")

        elapsed = time.time() - step_start
        remaining = step_interval - elapsed
        if remaining > 0:
            time.sleep( # nosemgrep: arbitrary-sleep # nosemgrep: arbitrary-sleep
remaining)

    return gripper_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(task: str, max_queries: int = 20, endpoint: str = "",
        save_images: bool = False):
    import boto3

    if not endpoint:
        endpoint = ENDPOINT_NAME

    debug_dir = None
    if save_images:
        debug_dir = f"/tmp/groot_debug_{int(time.time())}"
        os.makedirs(debug_dir, exist_ok=True)

    print(f"=== GR00T VLA Control ===")
    print(f"  Task:       {task}")
    print(f"  Endpoint:   {endpoint} ({REGION})")
    print(f"  Horizon:    {UR3_CONFIG.action_horizon} predict, {UR3_CONFIG.execute_horizon} execute")
    print(f"  Control Hz: {UR3_CONFIG.control_hz}")
    if debug_dir:
        print(f"  Debug imgs: {debug_dir}")
    print()

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    cameras = WristCameraCapture()
    cameras.start()
    gripper_state = 0.0

    sender = URScriptSender(ROBOT_IP)

    try:
        with SafeUR3Controller(ROBOT_IP) as robot:
            start_rad = [math.radians(d) for d in START_JOINTS_DEG]
            print("Moving to start position...")
            robot.move_joints(start_rad, vel=0.3, accel=0.3)
            robot.gripper_open()
            time.sleep( # nosemgrep: arbitrary-sleep # nosemgrep: arbitrary-sleep
0.5)

            sender.connect()
            print("=== Starting control loop ===\n")
            total_steps = 0
            prev_joints = None
            stall_count = 0

            for query_idx in range(max_queries):
                print(f"--- Query {query_idx} ---")

                joints = robot.get_joints()
                if joints is None:
                    print("  ERROR: Cannot read joints")
                    break
                state = joints + [gripper_state]

                if prev_joints is not None:
                    joint_delta = max(abs(a - b) for a, b in zip(joints, prev_joints))
                    if joint_delta < 0.001:
                        stall_count += 1
                        print(f"  WARNING: Robot stalled ({stall_count} cycles, max joint delta={joint_delta:.5f})")
                        if stall_count >= 3:
                            print("  ABORT: Robot stuck for 3 queries — possible joint limit or protective stop")
                            sender.send_stopj()
                            break
                    else:
                        stall_count = 0
                prev_joints = joints[:]

                images = cameras.capture()
                if not images:
                    print("  ERROR: No camera frames")
                    break

                img_sizes = {k: len(v) for k, v in images.items()}
                print(f"  State: [{', '.join(f'{j:.3f}' for j in joints)}] grip={gripper_state:.0f}")
                print(f"  Image: {img_sizes}")

                if debug_dir:
                    for cam_name, jpeg_bytes in images.items():
                        img_path = os.path.join(debug_dir, f"q{query_idx:03d}_{cam_name}.jpg")
                        with open(img_path, "wb") as f:
                            f.write(jpeg_bytes)

                try:
                    actions = query_groot(runtime, images, state, task, endpoint)
                except Exception as e:
                    print(f"  GR00T error: {e}")
                    break

                print(f"  Got {len(actions)} steps, "
                      f"Delta[0]: [{', '.join(f'{a:.4f}' for a in actions[0,:6])}] "
                      f"grip={actions[0,6]:.2f}")

                gripper_state = execute_action_chunk(
                    robot, sender, actions, gripper_state, UR3_CONFIG.execute_horizon)
                total_steps += min(UR3_CONFIG.execute_horizon, len(actions))

            sender.send_stopj()
            print(f"\nDone. {query_idx + 1} queries, {total_steps} steps executed.")
            if debug_dir:
                print(f"Debug images saved to: {debug_dir}")
    finally:
        sender.close()
        cameras.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-loop GR00T VLA control for UR3")
    parser.add_argument("task", help="Natural language task description")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--save-images", action="store_true",
                        help="Save camera frames to /tmp for debugging")
    args = parser.parse_args()
    run(args.task, max_queries=args.max_queries, endpoint=args.endpoint,
        save_images=args.save_images)
