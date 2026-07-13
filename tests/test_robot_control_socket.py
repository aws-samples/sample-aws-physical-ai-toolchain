"""On-robot control path for the UR3 hardware track (no real robot, no AWS).

The UR3 speaks plain-text URScript over a TCP socket (port 30002). We can exercise
the exact command path that was validated against real hardware by:

  1. standing up a fake UR3 TCP server on localhost and asserting the exact bytes
     `URScriptSender` puts on the wire, and
  2. driving `execute_action_chunk` with a fake robot + a fake sender and asserting
     the model-output -> velocity math, the safety clamps, and the gripper logic, and
  3. calling `query_groot` against a mocked sagemaker-runtime client and asserting the
     request/response contract matches what the groot-inference serve.py accepts.

No UR3, no camera, no GPU, no AWS. What this does NOT cover: a physical arm actually
moving, and a real SageMaker endpoint responding (needs a deployed endpoint).
"""
import base64
import json
import pathlib
import socket
import sys
import threading
import time

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from robot.ur3 import control


class _FakeUR3Server:
    """Accepts one TCP connection and records every byte it receives."""

    def __init__(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self.port = self._srv.getsockname()[1]
        self._srv.listen(1)
        self.received = bytearray()
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        conn.settimeout(0.5)
        while not self._stop:
            try:
                data = conn.recv(4096)
                if not data:
                    break
                self.received += data
            except socket.timeout:
                continue
            except OSError:
                break
        conn.close()

    def close(self):
        self._stop = True
        try:
            self._srv.close()
        except OSError:
            pass


class _CapturingSender:
    """Stand-in for URScriptSender that records calls instead of using a socket."""

    def __init__(self):
        self.speedls = []
        self.stops = 0

    def send_speedl(self, velocities, accel=1.0, duration=0.25):
        self.speedls.append([float(v) for v in velocities])

    def send_stopj(self, decel=3.0):
        self.stops += 1


class _FakeRobot:
    def __init__(self, force=None):
        self._force = force or [0.0] * 6
        self.gripper_calls = []

    def get_force(self):
        return self._force

    def gripper_open(self):
        self.gripper_calls.append("open")

    def gripper_close(self):
        self.gripper_calls.append("close")


def test_sender_puts_valid_urscript_on_the_wire():
    """URScriptSender emits the exact speedl/stopj URScript a UR3 parses."""
    fake = _FakeUR3Server()
    try:
        sender = control.URScriptSender("127.0.0.1", port=fake.port)
        sender.connect()
        sender.send_speedl([0.1, 0.0, -0.2, 0.0, 0.0, 0.05], accel=1.0, duration=0.25)
        sender.send_stopj()
        time.sleep(0.3) # nosemgrep: arbitrary-sleep
        sender.close()
    finally:
        fake.close()

    wire = bytes(fake.received).decode("utf-8")
    # speedl: wrapped in a def cmd(): ... end block, 6-vector, 5-decimal formatting
    assert "def cmd():" in wire
    assert "speedl([0.10000,0.00000,-0.20000,0.00000,0.00000,0.05000],1.0,0.25)" in wire
    assert "stopj(3.0)" in wire
    assert wire.count("end\n") == 2  # one per command block


def test_execute_action_chunk_converts_deltas_to_velocities():
    """Model outputs velocity*dt deltas; the loop divides by dt to recover velocity."""
    robot = _FakeRobot()
    sender = _CapturingSender()

    dt = 1.0 / control.UR3_CONFIG.control_hz  # 0.2 s at 5 Hz
    actions = np.array([
        [0.0, 0.0, -0.035, 0.0, 0.0, 0.0, 0.0],  # vz delta -0.035 -> vel -0.175
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    control.execute_action_chunk(robot, sender, actions,
                                 gripper_state=0.0, execute_steps=1)

    assert len(sender.speedls) == 1
    assert sender.speedls[0][2] == pytest.approx(-0.035 / dt)  # -0.175


def test_execute_action_chunk_clamps_large_deltas():
    """A delta above max_vel_delta is clamped before the velocity conversion."""
    robot = _FakeRobot()
    sender = _CapturingSender()

    dt = 1.0 / control.UR3_CONFIG.control_hz
    max_d = control.UR3_CONFIG.max_vel_delta  # 0.05
    actions = np.array([
        [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # vx delta 0.10 -> clamp to 0.05
    ], dtype=np.float32)

    control.execute_action_chunk(robot, sender, actions,
                                 gripper_state=0.0, execute_steps=1)

    # clamped to max_d, then / dt
    assert sender.speedls[0][0] == pytest.approx(max_d / dt)  # 0.25


def test_execute_action_chunk_gripper_transition():
    """Gripper actuates only on a crossing of the 0.5 threshold, once."""
    robot = _FakeRobot()
    sender = _CapturingSender()

    actions = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 0 -> close
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # stays closed: no repeat
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # -> open
    ], dtype=np.float32)

    final = control.execute_action_chunk(robot, sender, actions,
                                         gripper_state=0.0, execute_steps=3)

    assert robot.gripper_calls == ["close", "open"]
    assert final == pytest.approx(0.0)


def test_execute_action_chunk_force_limit_trips_stop():
    """Force above the limit sends stopj and halts the chunk immediately."""
    over = control.UR3_CONFIG.force_limit + 50.0
    robot = _FakeRobot(force=[over, 0.0, 0.0, 0.0, 0.0, 0.0])
    sender = _CapturingSender()

    actions = np.zeros((3, 7), dtype=np.float32)
    control.execute_action_chunk(robot, sender, actions,
                                 gripper_state=0.0, execute_steps=3)

    assert sender.stops == 1        # emergency stop issued
    assert len(sender.speedls) == 0  # bailed before commanding motion


# ---------------------------------------------------------------------------
# query_groot — the SageMaker endpoint round-trip (mocked runtime, no real endpoint)
# ---------------------------------------------------------------------------


class _FakeBody:
    def __init__(self, raw):
        self._raw = raw

    def read(self):
        return self._raw


class _FakeRuntime:
    """Stand-in for a boto3 sagemaker-runtime client."""

    def __init__(self, actions):
        self._actions = actions
        self.last_request = None

    def invoke_endpoint(self, **kwargs):
        self.last_request = kwargs
        body = json.dumps({"actions": self._actions, "action_dim": 7})
        return {"Body": _FakeBody(body.encode())}


def test_query_groot_request_matches_serving_contract():
    """The payload query_groot sends is exactly what groot-inference serve.py accepts.

    serve.py accepts 'images' as a {camera: base64} dict; control.py must send that
    shape (keyed by camera name) plus 'state' and 'task', as application/json.
    """
    rt = _FakeRuntime(actions=[[0.0, 0.0, -0.035, 0.0, 0.0, 0.0, 0.0]] * 16)
    raw_jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    images = {"wrist": raw_jpeg}

    out = control.query_groot(rt, images, state=[0.0] * 7,
                              task="pick up the cube", endpoint="groot-ur3")

    req = rt.last_request
    assert req["EndpointName"] == "groot-ur3"
    assert req["ContentType"] == "application/json"

    body = json.loads(req["Body"])
    assert sorted(body) == ["images", "state", "task"]
    # images keyed by camera name (serve.py's dict branch), base64 round-trips
    assert isinstance(body["images"], dict) and list(body["images"]) == ["wrist"]
    assert base64.b64decode(body["images"]["wrist"]) == raw_jpeg
    assert body["task"] == "pick up the cube"

    # response parsed into a (16, 7) float array
    assert out.shape == (16, 7)
    assert out.dtype == np.float32


def test_query_groot_squeezes_batched_action_response():
    """A (1, T, D) response from the endpoint is squeezed to (T, D)."""
    rt = _FakeRuntime(actions=[[[1.0] * 7] * 16])  # shape (1, 16, 7)
    out = control.query_groot(rt, {"wrist": b"x"}, state=[0.0] * 7,
                              task="t", endpoint="groot-ur3")
    assert out.shape == (16, 7)
