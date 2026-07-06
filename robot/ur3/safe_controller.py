#!/usr/bin/env python3
"""
Safe UR3 Controller - a single "dummy proof" controller for the UR3 robot.

Socket-only (no RTDE dependency), enforces all safety bounds automatically,
and uses the Dashboard Server to recover from any error state without user
intervention.

Architecture:
    SafeUR3Controller  (public API)
      +-- _BoundsChecker     (clamp joints, validate workspace, cap velocity)
      +-- _StateReader        (port 30003: binary state packets)
      +-- _RecoveryManager    (port 29999: wraps DashboardClient)
      +-- RobotiqGripper      (imported, socket mode)

Ports:
    30002 - URScript commands (ephemeral connections)
    30003 - Real-time state reading (persistent connection)
    29999 - Dashboard Server for recovery (persistent connection)

Usage as library:
    with SafeUR3Controller("127.0.0.1") as robot:
        robot.move_to("home")
        robot.move_joints([0, -1.57, 1.57, -1.57, -1.57, 0])
        robot.gripper_open()

Usage as CLI:
    uv run src/safe_controller.py status
    uv run src/safe_controller.py home
    uv run src/safe_controller.py move salute
    uv run src/safe_controller.py joints 0 -90 90 -90 -90 0
    uv run src/safe_controller.py gripper open
    uv run src/safe_controller.py freedrive
    uv run src/safe_controller.py positions
    uv run src/safe_controller.py recover
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import socket
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("safe_controller")

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ControllerError(Exception):
    """Base exception for SafeUR3Controller."""


class ControllerNotReadyError(ControllerError):
    """Command issued while the controller is not in READY state."""


class OutOfBoundsError(ControllerError):
    """Cartesian target is outside the workspace envelope."""


class RecoveryFailedError(ControllerError):
    """Automatic recovery exhausted all retries."""


# ---------------------------------------------------------------------------
# Controller state machine
# ---------------------------------------------------------------------------


class ControllerState(enum.Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    POWER_OFF = "POWER_OFF"
    BOOTING = "BOOTING"
    READY = "READY"
    MOVING = "MOVING"
    FREEDRIVE = "FREEDRIVE"
    PROTECTIVE_STOP = "PROTECTIVE_STOP"
    SAFETY_FAULT = "SAFETY_FAULT"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Bounds configuration
# ---------------------------------------------------------------------------


@dataclass
class BoundsConfig:
    """All safety limits in one place, fully configurable."""

    # Per-joint min/max in radians  [base, shoulder, elbow, wrist1, wrist2, wrist3]
    joint_min: list[float] = field(
        default_factory=lambda: [-math.pi, -math.pi, 0.0, -math.pi, -math.pi, -math.pi]
    )
    joint_max: list[float] = field(
        default_factory=lambda: [math.pi, 0.0, math.pi, math.pi, math.pi, math.pi]
    )

    # Cartesian workspace envelope (meters)
    max_reach: float = 0.45
    min_z: float = 0.02
    max_z: float = 0.60

    # Velocity / acceleration caps
    max_joint_vel: float = 1.0  # rad/s
    max_joint_accel: float = 1.0  # rad/s²
    max_linear_vel: float = 0.25  # m/s
    max_linear_accel: float = 0.25  # m/s²

    # Rate limiting
    min_command_interval: float = 0.1  # seconds between commands


# ---------------------------------------------------------------------------
# _BoundsChecker  (pure validation logic, no I/O)
# ---------------------------------------------------------------------------


class _BoundsChecker:
    def __init__(self, cfg: BoundsConfig):
        self.cfg = cfg

    def clamp_joints(self, joints: list[float]) -> list[float]:
        """Clamp each joint to its configured limit. Logs a warning on clamp."""
        clamped = []
        for i, angle in enumerate(joints):
            lo, hi = self.cfg.joint_min[i], self.cfg.joint_max[i]
            if angle < lo:
                log.warning("Joint %d clamped: %.3f → %.3f (min)", i, angle, lo)
                clamped.append(lo)
            elif angle > hi:
                log.warning("Joint %d clamped: %.3f → %.3f (max)", i, angle, hi)
                clamped.append(hi)
            else:
                clamped.append(angle)
        return clamped

    def validate_cartesian(self, pose: list[float]) -> tuple[bool, str]:
        """Return (ok, reason) for a Cartesian target [x, y, z, rx, ry, rz]."""
        x, y, z = pose[0], pose[1], pose[2]
        reach = math.sqrt(x * x + y * y)
        if reach > self.cfg.max_reach:
            return False, f"reach {reach:.3f}m exceeds max {self.cfg.max_reach}m"
        if z < self.cfg.min_z:
            return False, f"z={z:.3f}m below min {self.cfg.min_z}m"
        if z > self.cfg.max_z:
            return False, f"z={z:.3f}m above max {self.cfg.max_z}m"
        return True, ""

    def clamp_velocity(self, vel: float, is_linear: bool = False) -> float:
        cap = self.cfg.max_linear_vel if is_linear else self.cfg.max_joint_vel
        if vel > cap:
            log.warning("Velocity clamped: %.3f → %.3f", vel, cap)
            return cap
        return max(0.01, vel)

    def clamp_acceleration(self, accel: float, is_linear: bool = False) -> float:
        cap = self.cfg.max_linear_accel if is_linear else self.cfg.max_joint_accel
        if accel > cap:
            log.warning("Acceleration clamped: %.3f → %.3f", accel, cap)
            return cap
        return max(0.01, accel)


# ---------------------------------------------------------------------------
# _StateReader  (port 30003: read binary state packets)
# ---------------------------------------------------------------------------

# UR CB-series real-time data packet offsets (~1108 bytes at 125 Hz)
# Layout: 4-byte header (uint32 packet length) then doubles in fixed order.
_RT_PACKET_MIN_SIZE = 300
_RT_PACKET_FULL_SIZE = 1108
_6D = 48  # 6 × float64 (8 bytes each)

_Q_TARGET_OFFSET = 12  # Target joint positions (what the controller is aiming for)
_Q_ACTUAL_OFFSET = 252  # Actual joint positions
_QD_ACTUAL_OFFSET = 300  # Actual joint velocities (key for motion detection)
_Q_ACTUAL_SIZE = _6D
_TCP_OFFSET = 444  # Tool Center Point [x, y, z, rx, ry, rz]
_TCP_SIZE = _6D
_TCP_FORCE_OFFSET = 540  # TCP force/torque [Fx, Fy, Fz, Mx, My, Mz]
_ROBOT_MODE_OFFSET = 756  # float64 (POWER_OFF=2, BOOTING=3, IDLE=5, BACKDRIVE=6, RUNNING=7)
# CB3: bytes 764-811 = joint_modes[6], then safety_mode at 812
_SAFETY_MODE_OFFSET = 812  # float64 (CB3 ≥ v3.0: NORMAL=1, PROTECTIVE_STOP=3, FAULT=9, etc.)


class _StateReader:
    """Persistent connection to port 30003 for reading robot state.

    Hardened with:
    - Automatic reconnection on socket errors
    - Packet framing validation (length prefix sanity check)
    - Drains stale data on reconnect
    - Thread-safe reads
    """

    def __init__(self, host: str, port: int = 30003, timeout: float = 2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    @staticmethod
    def _configure_socket(s: socket.socket, timeout: float) -> None:
        """Apply hardened settings: timeout, keepalive, no-delay."""
        s.settimeout(timeout)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect(self) -> None:
        with self._lock:
            self._close_socket()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._configure_socket(s, self.timeout)
            try:
                s.connect((self.host, self.port))
            except (socket.timeout, OSError) as e:
                s.close()
                raise ControllerError(f"StateReader connect to {self.host}:{self.port} failed: {e}") from e
            # Drain any buffered data from a previous session
            s.setblocking(False)
            try:
                while s.recv(4096):
                    pass
            except (BlockingIOError, OSError):
                pass
            s.setblocking(True)
            s.settimeout(self.timeout)
            self._sock = s
            self._consecutive_failures = 0
            log.info("StateReader connected to %s:%d", self.host, self.port)

    def disconnect(self) -> None:
        with self._lock:
            self._close_socket()

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _reconnect(self) -> bool:
        """Attempt a reconnection. Returns True on success."""
        self._close_socket()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._configure_socket(s, self.timeout)
            s.connect((self.host, self.port))
            # Drain stale data
            s.setblocking(False)
            try:
                while s.recv(4096):
                    pass
            except (BlockingIOError, OSError):
                pass
            s.setblocking(True)
            s.settimeout(self.timeout)
            self._sock = s
            self._consecutive_failures = 0
            log.info("StateReader reconnected to %s:%d", self.host, self.port)
            return True
        except (socket.timeout, OSError) as e:
            log.debug("StateReader reconnect failed: %s", e)
            return False

    def _read_packet(self) -> bytes | None:
        """Read one full real-time data packet with reconnect-on-failure."""
        with self._lock:
            if self._sock is None:
                if not self._reconnect():
                    return None
            try:
                # Read packet size (first 4 bytes = int32 big-endian)
                header = self._recv_exact(4)
                if header is None:
                    self._handle_failure("empty header")
                    return None
                pkt_len = struct.unpack("!I", header)[0]
                if pkt_len < _RT_PACKET_MIN_SIZE or pkt_len > 2048:
                    log.warning("StateReader bad packet length: %d, draining", pkt_len)
                    self._drain_and_reconnect()
                    return None
                body = self._recv_exact(pkt_len - 4)
                if body is None:
                    self._handle_failure("incomplete body")
                    return None
                self._consecutive_failures = 0
                return header + body
            except (socket.timeout, OSError) as e:
                self._handle_failure(str(e))
                return None

    def _read_packet_nonblocking(self) -> bytes | None:
        """Try to read a packet without blocking. Returns None if no data ready."""
        with self._lock:
            if self._sock is None:
                return None
            try:
                self._sock.setblocking(False)
                try:
                    header = self._sock.recv(4)
                except (BlockingIOError, socket.timeout):
                    return None
                finally:
                    self._sock.setblocking(True)
                    self._sock.settimeout(self.timeout)
                if len(header) < 4:
                    return None
                pkt_len = struct.unpack("!I", header)[0]
                if pkt_len < _RT_PACKET_MIN_SIZE or pkt_len > 2048:
                    return None
                body = self._recv_exact(pkt_len - 4)
                if body is None:
                    return None
                return header + body
            except OSError:
                return None

    def _recv_exact(self, n: int) -> bytes | None:
        buf = b""
        deadline = time.monotonic() + self.timeout
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                self._sock.settimeout(max(0.1, remaining))
                chunk = self._sock.recv(n - len(buf))
            except socket.timeout:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _handle_failure(self, reason: str) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            log.warning(
                "StateReader %d consecutive failures (%s), reconnecting",
                self._consecutive_failures,
                reason,
            )
            self._drain_and_reconnect()

    def _drain_and_reconnect(self) -> None:
        self._close_socket()
        self._reconnect()

    def read_state(self) -> dict | None:
        """Read the LATEST packet, discarding any stale buffered data.

        The robot sends at 125 Hz but we only poll at ~10 Hz, so packets
        accumulate in the socket buffer.  This method drains all buffered
        packets and parses only the most recent one, ensuring we always
        have fresh data.

        Returns dict with keys: joints, joint_velocities, target_joints,
        tcp_pose, robot_mode, safety_mode.  All values sanity-checked.
        """
        # Read one packet (blocking — ensures we get at least one)
        data = self._read_packet()
        if data is None:
            return None
        # Drain any buffered packets, keep only the latest
        while True:
            more = self._read_packet_nonblocking()
            if more is None:
                break
            data = more
        result: dict = {}
        try:
            # Target joint positions (what the controller is tracking toward)
            if len(data) >= _Q_TARGET_OFFSET + _6D:
                q_target = list(struct.unpack("!6d", data[_Q_TARGET_OFFSET : _Q_TARGET_OFFSET + _6D]))
                if all(abs(j) < 2 * math.pi for j in q_target):
                    result["target_joints"] = q_target

            # Actual joint positions
            if len(data) >= _Q_ACTUAL_OFFSET + _6D:
                joints = list(struct.unpack("!6d", data[_Q_ACTUAL_OFFSET : _Q_ACTUAL_OFFSET + _6D]))
                if all(abs(j) < 2 * math.pi for j in joints):
                    result["joints"] = joints

            # Actual joint velocities (rad/s) — primary motion detection signal
            if len(data) >= _QD_ACTUAL_OFFSET + _6D:
                qd = list(struct.unpack("!6d", data[_QD_ACTUAL_OFFSET : _QD_ACTUAL_OFFSET + _6D]))
                # Velocities should be < 10 rad/s even in worst case
                if all(abs(v) < 10.0 for v in qd):
                    result["joint_velocities"] = qd

            # TCP pose
            if len(data) >= _TCP_OFFSET + _6D:
                tcp = list(struct.unpack("!6d", data[_TCP_OFFSET : _TCP_OFFSET + _6D]))
                if all(abs(v) < 2.0 for v in tcp[:3]):
                    result["tcp_pose"] = tcp

            # TCP force/torque [Fx, Fy, Fz, Mx, My, Mz]
            if len(data) >= _TCP_FORCE_OFFSET + _6D:
                ft = list(struct.unpack("!6d", data[_TCP_FORCE_OFFSET : _TCP_FORCE_OFFSET + _6D]))
                if all(abs(v) < 500.0 for v in ft):  # sanity: < 500 N/Nm
                    result["tcp_force"] = ft

            # Robot mode
            if len(data) >= _ROBOT_MODE_OFFSET + 8:
                rm = struct.unpack("!d", data[_ROBOT_MODE_OFFSET : _ROBOT_MODE_OFFSET + 8])[0]
                if 0 <= rm <= 11:
                    result["robot_mode"] = rm

            # Safety mode (CB3: offset 812, after 6 joint_mode doubles at 764)
            if len(data) >= _SAFETY_MODE_OFFSET + 8:
                sm = struct.unpack("!d", data[_SAFETY_MODE_OFFSET : _SAFETY_MODE_OFFSET + 8])[0]
                if 0 <= sm <= 11:
                    result["safety_mode"] = sm
        except struct.error as e:
            log.warning("StateReader unpack error: %s", e)
            return None
        return result if result else None


# ---------------------------------------------------------------------------
# _RecoveryManager  (port 29999: wraps DashboardClient)
# ---------------------------------------------------------------------------

# Import from sibling module
_dashboard_client_mod = None


def _get_dashboard_client_class():
    global _dashboard_client_mod
    if _dashboard_client_mod is None:
        from robot.ur3.dashboard_client import DashboardClient

        _dashboard_client_mod = DashboardClient
    return _dashboard_client_mod


# Robot mode values from real-time data packet (float64 at offset 756)
_ROBOT_MODE_RUNNING = 7.0
_ROBOT_MODE_IDLE = 5.0
_ROBOT_MODE_POWER_OFF = 2.0
_ROBOT_MODE_BOOTING = 3.0
_ROBOT_MODE_BACKDRIVE = 6.0

# Safety mode values from real-time data packet (float64 at offset 812, CB3)
_SAFETY_NORMAL = 1.0
_SAFETY_PROTECTIVE_STOP = 3.0
_SAFETY_RECOVERY = 4.0
_SAFETY_SAFEGUARD_STOP = 5.0
_SAFETY_SYSTEM_ESTOP = 6.0
_SAFETY_ROBOT_ESTOP = 7.0
_SAFETY_VIOLATION = 8.0
_SAFETY_FAULT = 9.0


class _RecoveryManager:
    """Wraps DashboardClient for automatic error recovery.

    Hardened with:
    - Auto-reconnect on dashboard socket errors
    - All commands wrapped with _safe_cmd() that catches and reconnects
    - Thread-safe access
    """

    def __init__(self, host: str, max_attempts: int = 3):
        self.host = host
        self.max_attempts = max_attempts
        self._db = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            self._close()
            DashboardClient = _get_dashboard_client_class()
            self._db = DashboardClient(host=self.host)
            try:
                banner = self._db.connect()
                log.info("RecoveryManager connected: %s", banner)
            except (socket.timeout, OSError) as e:
                self._db = None
                raise ControllerError(f"Dashboard connect failed: {e}") from e

    def disconnect(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._db is not None:
            try:
                self._db.disconnect()
            except Exception:
                pass
            self._db = None

    def _reconnect(self) -> bool:
        """Try to re-establish dashboard connection. Returns True on success."""
        self._close()
        try:
            DashboardClient = _get_dashboard_client_class()
            self._db = DashboardClient(host=self.host)
            banner = self._db.connect()
            log.info("RecoveryManager reconnected: %s", banner)
            return True
        except Exception as e:
            log.warning("RecoveryManager reconnect failed: %s", e)
            self._db = None
            return False

    def _safe_cmd(self, method_name: str, *args, **kwargs) -> str:
        """Call a DashboardClient method, reconnecting once on failure."""
        if self._db is None:
            if not self._reconnect():
                return ""
        try:
            return getattr(self._db, method_name)(*args, **kwargs)
        except (socket.timeout, OSError, RuntimeError) as e:
            log.warning("Dashboard command '%s' failed: %s, reconnecting", method_name, e)
            if self._reconnect():
                try:
                    return getattr(self._db, method_name)(*args, **kwargs)
                except Exception as e2:
                    log.error("Dashboard command '%s' failed after reconnect: %s", method_name, e2)
            return ""

    @property
    def connected(self) -> bool:
        return self._db is not None

    def robot_mode(self) -> str:
        with self._lock:
            return self._safe_cmd("robot_mode")

    def safety_mode(self) -> str:
        with self._lock:
            return self._safe_cmd("safety_mode")

    def power_on_and_release_brakes(self) -> None:
        with self._lock:
            if self._db is None and not self._reconnect():
                raise ControllerError("Dashboard not connected")
            log.info("Powering on and releasing brakes...")
            try:
                self._db.power_on_and_release_brakes(settle_time=8.0)
            except (socket.timeout, OSError) as e:
                log.warning("power_on_and_release_brakes failed: %s, retrying after reconnect", e)
                if self._reconnect():
                    self._db.power_on_and_release_brakes(settle_time=8.0)
                else:
                    raise ControllerError("Cannot power on: dashboard connection lost") from e

    def recover_protective_stop(self) -> bool:
        """Wait 5.5s → close popup → unlock → verify NORMAL. Returns True on success."""
        with self._lock:
            if self._db is None and not self._reconnect():
                return False
            log.info("Recovering from protective stop...")
            time.sleep(5.5)
            self._safe_cmd("close_safety_popup")
            time.sleep(0.5)
            resp = self._safe_cmd("unlock_protective_stop")
            log.info("unlock_protective_stop: %s", resp)
            time.sleep(1.0)
            mode = self._safe_cmd("safety_mode")
            log.info("Safety mode after recovery: %s", mode)
            return "NORMAL" in mode.upper()

    def recover_safety_fault(self) -> bool:
        """Close popup → restart safety → wait → power on → verify."""
        with self._lock:
            if self._db is None and not self._reconnect():
                return False
            log.info("Recovering from safety fault/violation...")
            self._safe_cmd("close_safety_popup")
            time.sleep(0.5)
            self._safe_cmd("restart_safety")
            log.info("Safety restarting, waiting 10s...")
            time.sleep(10.0)
            # Need fresh connection after safety restart
            self._reconnect()
            try:
                if self._db:
                    self._db.power_on_and_release_brakes(settle_time=8.0)
            except Exception as e:
                log.warning("Power on after safety restart failed: %s", e)
                return False
            time.sleep(2.0)
            mode = self._safe_cmd("robot_mode")
            log.info("Robot mode after fault recovery: %s", mode)
            return "RUNNING" in mode.upper()

    def recover_power_off(self) -> bool:
        """Power on + brake release → verify RUNNING."""
        with self._lock:
            if self._db is None and not self._reconnect():
                return False
            log.info("Recovering from unexpected power off...")
            try:
                if self._db:
                    self._db.power_on_and_release_brakes(settle_time=8.0)
            except (socket.timeout, OSError) as e:
                log.warning("Power on failed: %s, retrying", e)
                if self._reconnect() and self._db:
                    self._db.power_on_and_release_brakes(settle_time=8.0)
                else:
                    return False
            time.sleep(2.0)
            mode = self._safe_cmd("robot_mode")
            log.info("Robot mode after power recovery: %s", mode)
            return "RUNNING" in mode.upper()

    def get_info(self) -> dict:
        """Return a dict of dashboard information."""
        with self._lock:
            if self._db is None:
                return {}
            try:
                return {
                    "robot_mode": self._safe_cmd("robot_mode"),
                    "safety_mode": self._safe_cmd("safety_mode"),
                }
            except Exception:
                return {}


# ---------------------------------------------------------------------------
# Predefined safe positions (all validated at module load time)
# ---------------------------------------------------------------------------

_DEFAULT_BOUNDS = BoundsConfig()
_bounds_checker = _BoundsChecker(_DEFAULT_BOUNDS)

POSITIONS: dict[str, list[float]] = {
    "home": [0, -1.57, 1.57, -1.57, -1.57, 0],
    "forward": [0, -1.0, 0.8, -1.3, -1.57, 0],
    "up": [0, -1.8, 2.2, -1.9, -1.57, 0],
    "left": [-1.2, -1.2, 1.5, -1.8, -1.57, 0],
    "right": [1.2, -1.2, 1.5, -1.8, -1.57, 0],
    "reach_forward": [0, -1.0, 0.8, -1.3, -1.57, 0],
    "reach_up": [0, -1.8, 2.2, -1.9, -1.57, 0],
    "reach_left": [-1.2, -1.2, 1.5, -1.8, -1.57, 0],
    "reach_right": [1.2, -1.2, 1.5, -1.8, -1.57, 0],
    "point_forward": [0, -1.1, 0.5, -0.9, -1.57, 0],
    "wave_high": [0.8, -1.9, 2.0, -1.6, -1.57, 0],
    "wave_side": [1.0, -1.3, 1.8, -2.0, -1.57, 0],
    "salute": [0.3, -1.6, 2.1, -2.0, -1.57, 0],
    "thinking": [0.2, -1.4, 1.9, -2.0, -1.57, 0],
    "welcome": [-0.5, -1.0, 1.2, -1.7, -1.57, 0],
    "stop_gesture": [0, -1.2, 1.0, -1.3, -1.57, 0],
}

# Validate all at import time
for _name, _joints in POSITIONS.items():
    _clamped = _bounds_checker.clamp_joints(_joints)
    if _clamped != _joints:
        raise ValueError(
            f"Predefined position '{_name}' fails default bounds: {_joints} → {_clamped}"
        )


# ---------------------------------------------------------------------------
# Gripper import helper
# ---------------------------------------------------------------------------

_gripper_class = None


def _get_gripper_class():
    global _gripper_class
    if _gripper_class is None:
        from robot.ur3.robotiq_gripper_control import RobotiqGripper

        _gripper_class = RobotiqGripper
    return _gripper_class


# ---------------------------------------------------------------------------
# SafeUR3Controller  (public API)
# ---------------------------------------------------------------------------


class SafeUR3Controller:
    """
    Safe, socket-only UR3 controller with automatic bounds enforcement
    and error recovery.

    Usage::

        with SafeUR3Controller("127.0.0.1") as robot:
            robot.move_to("home")
            robot.move_joints([0, -1.57, 1.57, -1.57, -1.57, 0])
            robot.gripper_open()
    """

    def __init__(
        self,
        robot_ip: str = "127.0.0.1",
        *,
        auto_power: bool = True,
        bounds: BoundsConfig | None = None,
        max_recovery_attempts: int = 3,
    ):
        self.robot_ip = robot_ip
        self._auto_power = auto_power
        self._bounds_cfg = bounds or BoundsConfig()
        self._checker = _BoundsChecker(self._bounds_cfg)
        self._state_reader = _StateReader(robot_ip)
        self._recovery = _RecoveryManager(robot_ip, max_attempts=max_recovery_attempts)
        self._max_recovery = max_recovery_attempts

        self._state = ControllerState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._recovery_lock = threading.Lock()  # prevents concurrent auto-recovery
        self._last_command_time: float = 0.0

        # Health monitor thread
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        # Cached state from monitor
        self._cached_joints: list[float] | None = None
        self._cached_joint_velocities: list[float] | None = None
        self._cached_target_joints: list[float] | None = None
        self._cached_tcp: list[float] | None = None
        self._cached_force: list[float] | None = None
        self._cached_robot_mode: float | None = None
        self._cached_safety_mode: float | None = None

        # Motion tracking
        self._command_target_joints: list[float] | None = None
        self._moving_since: float = 0.0
        self._saw_motion = False
        self._prev_joints: list[float] | None = None  # for position-delta motion detection

        # Shared state file for external consumers (e.g. visualizer)
        self._state_file = Path(tempfile.gettempdir()) / "ur3_state.json"

        # Gripper (lazy init)
        self._gripper = None
        self._gripper_activated = False

        # User-added positions (on top of POSITIONS)
        self._custom_positions: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to all three ports and start the health monitor."""
        self._set_state(ControllerState.CONNECTING)
        try:
            self._state_reader.connect()
            self._recovery.connect()
        except Exception as e:
            self._set_state(ControllerState.DISCONNECTED)
            raise ControllerError(f"Connection failed: {e}") from e

        # Read initial state — retry a few times because _state_reader.connect()
        # drains stale data, so the first fresh packet may not arrive immediately.
        for _ in range(10):
            self._poll_state()
            if self._state != ControllerState.CONNECTING:
                break
            time.sleep(0.1)

        # Auto-power if needed
        if self._auto_power:
            self._ensure_powered()

        # Start health monitor
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._health_monitor, daemon=True, name="ur3-health"
        )
        self._monitor_thread.start()
        log.info("SafeUR3Controller connected to %s", self.robot_ip)

    def disconnect(self) -> None:
        """Stop monitor, close all sockets."""
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            # Recovery operations can take up to ~20s (safety fault recovery),
            # so allow enough time for the monitor to finish gracefully.
            self._monitor_thread.join(timeout=25.0)
            if self._monitor_thread.is_alive():
                log.warning("Health monitor thread did not exit in time")
            self._monitor_thread = None
        self._state_reader.disconnect()
        self._recovery.disconnect()
        self._gripper = None
        self._gripper_activated = False
        self._set_state(ControllerState.DISCONNECTED)
        log.info("SafeUR3Controller disconnected")

    def __enter__(self) -> SafeUR3Controller:
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Movement (all bounds-checked, all auto-recover on fault)
    # ------------------------------------------------------------------

    def move_joints(
        self,
        angles: list[float],
        vel: float = 0.3,
        accel: float = 0.3,
        *,
        wait: bool = True,
    ) -> bool:
        """Move to joint angles (radians). Auto-clamps to bounds.

        If robot is currently moving, sends stopj first and waits for it to
        settle before issuing the new command.  This prevents the UR controller
        from abruptly swapping programs mid-trajectory (which can cause
        protective stops).
        """
        self._require_ready()
        clamped = self._checker.clamp_joints(list(angles))
        vel = self._checker.clamp_velocity(vel)
        accel = self._checker.clamp_acceleration(accel)
        # Skip if already at target (avoids 30s timeout waiting for motion)
        current = self._cached_joints
        if current is not None:
            max_delta = max(abs(a - b) for a, b in zip(clamped, current))
            if max_delta < 0.01:  # ~0.6° — close enough
                log.debug("Already at target, skipping move")
                return True
        self._ensure_stopped()
        self._rate_limit()
        self._command_target_joints = clamped
        # Atomically enter MOVING — prevents clobbering a fault state that
        # the health monitor set between _require_ready() and now.
        if not self._try_enter_moving():
            return False
        try:
            self._send_movej(clamped, vel, accel)
            if wait:
                return self.wait_until_idle()
            return True
        except Exception as e:
            log.error("move_joints failed: %s", e)
            self._try_auto_recover()
            return False

    def move_to(
        self,
        position_name: str,
        vel: float = 0.3,
        accel: float = 0.3,
        *,
        wait: bool = True,
    ) -> bool:
        """Move to a pre-validated named position."""
        all_positions = self.list_positions()
        if position_name not in all_positions:
            raise ControllerError(
                f"Unknown position '{position_name}'. "
                f"Available: {sorted(all_positions.keys())}"
            )
        return self.move_joints(all_positions[position_name], vel, accel, wait=wait)

    def move_linear(
        self,
        pose: list[float],
        vel: float = 0.1,
        accel: float = 0.1,
        *,
        wait: bool = True,
    ) -> bool:
        """Move linearly to a Cartesian pose [x,y,z,rx,ry,rz]. Rejects out-of-bounds."""
        self._require_ready()
        ok, reason = self._checker.validate_cartesian(pose)
        if not ok:
            raise OutOfBoundsError(reason)
        vel = self._checker.clamp_velocity(vel, is_linear=True)
        accel = self._checker.clamp_acceleration(accel, is_linear=True)
        self._ensure_stopped()
        self._rate_limit()
        self._command_target_joints = None  # movel targets are Cartesian, use velocity-only check
        # Atomically enter MOVING — prevents clobbering a fault state that
        # the health monitor set between _require_ready() and now.
        if not self._try_enter_moving():
            return False
        try:
            self._send_movel(pose, vel, accel)
            if wait:
                return self.wait_until_idle()
            return True
        except Exception as e:
            log.error("move_linear failed: %s", e)
            self._try_auto_recover()
            return False

    def stop(self) -> None:
        """Send a stop command to the robot.  Deceleration = 3.0 rad/s²."""
        program = "def prog():\n  stopj(3.0)\nend\n"
        try:
            self._send_urscript(program)
        except ControllerError:
            pass  # Best-effort stop; don't raise if socket is down

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        """Block until the robot stops moving or timeout expires.

        Waits to see actual motion (non-zero velocity) first, then waits for
        the robot to come to rest.  This handles SSH tunnel latency — the
        command may take a second or more to reach the robot.

        Note: does NOT call _poll_state() — the health monitor thread handles
        all packet reading to avoid competing for port 30003.
        """
        deadline = time.time() + timeout
        saw_motion = False
        still_count = 0
        required_still = 3  # ~240ms of stillness
        prev_joints = None

        while time.time() < deadline:
            # Fault detection — bail immediately
            if self._state in (
                ControllerState.PROTECTIVE_STOP,
                ControllerState.SAFETY_FAULT,
                ControllerState.ERROR,
            ):
                log.warning("wait_until_idle: robot entered %s", self._state.value)
                return False

            if self._state == ControllerState.READY:
                return True

            # Detect motion via position delta (local tracking, no shared state)
            joints = self._cached_joints
            if joints is not None:
                if prev_joints is not None:
                    max_delta = max(abs(a - p) for a, p in zip(joints, prev_joints))
                    if max_delta > 0.001:  # ~0.06° — robot is moving
                        saw_motion = True
                        still_count = 0
                    elif saw_motion:
                        still_count += 1
                        if still_count >= required_still:
                            self._set_state(ControllerState.READY)
                            self._command_target_joints = None
                            return True
                prev_joints = list(joints)

            time.sleep(0.08)  # ~12 Hz check rate

        log.warning(
            "wait_until_idle timed out after %.1fs (state=%s, saw_motion=%s)",
            timeout, self._state.value, saw_motion,
        )
        return False

    def _check_motion_complete(self) -> bool:
        """Return True if the robot has moved and then stopped.

        Detects motion by comparing joint positions between consecutive
        readings (> 0.001 rad delta = moving).  Requires seeing motion
        before declaring complete, preventing false "done" when the command
        hasn't reached the robot yet over the SSH tunnel.
        """
        actual = self._cached_joints
        if actual is None:
            return False

        prev = self._prev_joints
        self._prev_joints = list(actual)

        if prev is not None:
            max_delta = max(abs(a - p) for a, p in zip(actual, prev))
            if max_delta > 0.001:  # ~0.06° — robot is moving
                self._saw_motion = True
                return False

        return self._saw_motion  # Only done if we saw motion first

    # ------------------------------------------------------------------
    # Named positions
    # ------------------------------------------------------------------

    def list_positions(self) -> dict[str, list[float]]:
        """Return all named positions (built-in + custom)."""
        merged = dict(POSITIONS)
        merged.update(self._custom_positions)
        return merged

    def add_position(self, name: str, joints: list[float]) -> None:
        """Register a custom named position. Validates against bounds."""
        clamped = self._checker.clamp_joints(list(joints))
        if clamped != list(joints):
            log.warning(
                "Position '%s' was clamped to fit bounds: %s → %s", name, joints, clamped
            )
        self._custom_positions[name] = clamped

    # ------------------------------------------------------------------
    # Freedrive / teach mode
    # ------------------------------------------------------------------

    def freedrive(self, enable: bool = True) -> None:
        """Enable or disable freedrive (teach) mode."""
        if enable:
            self._require_ready()
            program = "def prog():\n  freedrive_mode()\n  sleep(3600)\nend\n"
            self._send_urscript(program)
            self._set_state(ControllerState.FREEDRIVE)
            log.info("Freedrive enabled")
        else:
            program = "def prog():\n  end_freedrive_mode()\nend\n"
            self._send_urscript(program)
            self._set_state(ControllerState.READY)
            log.info("Freedrive disabled")

    def capture_position(self) -> list[float] | None:
        """Read and return current joint positions (e.g. during freedrive)."""
        return self.get_joints()

    # ------------------------------------------------------------------
    # Gripper (delegates to RobotiqGripper in socket mode)
    # ------------------------------------------------------------------

    def _ensure_gripper(self) -> None:
        if self._gripper is None:
            GripperClass = _get_gripper_class()
            self._gripper = GripperClass(robot_ip=self.robot_ip)

    def gripper_activate(self) -> bool:
        self._ensure_gripper()
        result = self._gripper.activate()
        self._gripper_activated = True
        return result

    def gripper_open(self) -> bool:
        self._ensure_gripper()
        if not self._gripper_activated:
            self.gripper_activate()
        return self._gripper.open()

    def gripper_close(self) -> bool:
        self._ensure_gripper()
        if not self._gripper_activated:
            self.gripper_activate()
        return self._gripper.close()

    def gripper_move(self, position_mm: float) -> bool:
        self._ensure_gripper()
        if not self._gripper_activated:
            self.gripper_activate()
        return self._gripper.move(position_mm)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> ControllerState:
        return self._state

    def get_joints(self) -> list[float] | None:
        """Return latest joint positions in radians, or None."""
        return list(self._cached_joints) if self._cached_joints else None

    def get_tcp_pose(self) -> list[float] | None:
        """Return latest TCP pose [x,y,z,rx,ry,rz], or None."""
        return list(self._cached_tcp) if self._cached_tcp else None

    def get_force(self) -> list[float] | None:
        """Return latest TCP force/torque [Fx,Fy,Fz,Mx,My,Mz], or None."""
        return list(self._cached_force) if self._cached_force else None

    def get_status(self) -> dict:
        """Return a comprehensive status dict."""
        joints = self.get_joints()
        tcp = self.get_tcp_pose()
        dashboard = {}
        try:
            dashboard = self._recovery.get_info()
        except Exception:
            pass
        return {
            "state": self._state.value,
            "robot_ip": self.robot_ip,
            "joints_rad": joints,
            "joints_deg": [round(math.degrees(j), 1) for j in joints] if joints else None,
            "tcp_pose": tcp,
            "dashboard": dashboard,
            "state_reader_connected": self._state_reader.connected,
            "dashboard_connected": self._recovery.connected,
        }

    def is_ready(self) -> bool:
        return self._state == ControllerState.READY

    def is_moving(self) -> bool:
        return self._state == ControllerState.MOVING

    # ------------------------------------------------------------------
    # Internal: URScript sending (port 30002)
    # ------------------------------------------------------------------

    # Minimum time in MOVING state before allowing transition to READY.
    # Prevents false "motion complete" when the robot hasn't physically
    # started yet (command still travelling over SSH tunnel).
    _MOVING_GRACE_PERIOD = 1.0  # seconds

    _SEND_MAX_RETRIES = 2
    _SEND_TIMEOUT = 5.0

    def _send_urscript(self, program: str) -> None:
        """Send a URScript program to port 30002 using an ephemeral socket.

        Retries once on transient socket errors (tunnel hiccup, etc.).
        Uses TCP_NODELAY and SO_KEEPALIVE for reliability over SSH tunnels.
        """
        encoded = program.encode("utf-8")
        last_err: Exception | None = None
        with self._send_lock:
            for attempt in range(self._SEND_MAX_RETRIES):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self._SEND_TIMEOUT)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    s.connect((self.robot_ip, 30002))
                    s.sendall(encoded)
                    # Brief read to detect immediate connection reset (the UR
                    # controller closes the socket if it rejects the program).
                    s.settimeout(0.3)
                    try:
                        s.recv(1)
                    except socket.timeout:
                        pass  # Expected — UR doesn't normally send a response
                    return
                except (socket.timeout, OSError, ConnectionError) as e:
                    last_err = e
                    log.warning(
                        "URScript send attempt %d/%d failed: %s",
                        attempt + 1,
                        self._SEND_MAX_RETRIES,
                        e,
                    )
                    time.sleep(0.5)
                finally:
                    try:
                        s.close()
                    except OSError:
                        pass
        raise ControllerError(f"URScript send failed after {self._SEND_MAX_RETRIES} attempts: {last_err}")


    def _send_movej(self, joints: list[float], vel: float, accel: float) -> None:
        angles_str = "[" + ", ".join(f"{a:.6f}" for a in joints) + "]"
        program = f"def prog():\n  movej({angles_str}, a={accel:.4f}, v={vel:.4f})\nend\n"
        self._send_urscript(program)

    def _send_movel(self, pose: list[float], vel: float, accel: float) -> None:
        pose_str = "p[" + ", ".join(f"{v:.6f}" for v in pose) + "]"
        program = f"def prog():\n  movel({pose_str}, a={accel:.4f}, v={vel:.4f})\nend\n"
        self._send_urscript(program)

    # ------------------------------------------------------------------
    # Internal: state management
    # ------------------------------------------------------------------

    def _set_state(self, new_state: ControllerState) -> None:
        with self._state_lock:
            if self._state != new_state:
                log.debug("State: %s → %s", self._state.value, new_state.value)
                self._state = new_state
                if new_state == ControllerState.MOVING:
                    self._moving_since = time.time()
                    self._saw_motion = False
                    self._prev_joints = None

    def _publish_state(self) -> None:
        """Write latest state to a shared JSON file for external consumers."""
        try:
            data = {
                "state": self._state.value,
                "joints": self._cached_joints,
                "tcp_pose": self._cached_tcp,
                "robot_mode": self._cached_robot_mode,
                "safety_mode": self._cached_safety_mode,
                "timestamp": time.time(),
            }
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(str(tmp), str(self._state_file))
        except Exception:
            pass  # Non-critical

    def _try_enter_moving(self) -> bool:
        """Atomically transition to MOVING only if currently READY or MOVING.

        Returns True if the transition succeeded.  This prevents clobbering a
        fault state (PROTECTIVE_STOP, SAFETY_FAULT, etc.) that the health
        monitor set between _require_ready() and the actual command send.
        """
        with self._state_lock:
            if self._state not in (ControllerState.READY, ControllerState.MOVING):
                log.warning(
                    "Cannot enter MOVING: state is %s", self._state.value
                )
                return False
            if self._state != ControllerState.MOVING:
                log.debug("State: %s → MOVING", self._state.value)
                self._state = ControllerState.MOVING
                self._moving_since = time.time()
            return True

    def _require_ready(self) -> None:
        if self._state == ControllerState.FREEDRIVE:
            # Auto-exit freedrive for explicit moves
            self.freedrive(enable=False)
            time.sleep(0.5)
        if self._state not in (ControllerState.READY, ControllerState.MOVING):
            raise ControllerNotReadyError(
                f"Controller is in {self._state.value} state, not READY. "
                "Call connect() or wait for recovery."
            )

    def _ensure_stopped(self) -> None:
        """If the robot is currently moving, send stopj and wait for stillness.

        This prevents sending a new movej while a previous one is executing,
        which would cause the UR controller to abruptly swap programs and can
        trigger protective stops.

        Raises ControllerError if the robot does not stop within 5 seconds.
        """
        if self._state != ControllerState.MOVING:
            return
        log.info("Robot still moving — sending stopj before new command")
        self.stop()
        # Wait for the robot to actually decelerate to a stop
        for _ in range(50):  # up to 5s
            self._poll_state()
            if self._state in (
                ControllerState.PROTECTIVE_STOP,
                ControllerState.SAFETY_FAULT,
                ControllerState.ERROR,
            ):
                raise ControllerNotReadyError(
                    f"Robot entered {self._state.value} while stopping"
                )
            if self._cached_joint_velocities is not None:
                if max(abs(v) for v in self._cached_joint_velocities) < 0.01:
                    self._set_state(ControllerState.READY)
                    return
            time.sleep(0.1)
        raise ControllerError("Robot did not stop within 5s after stopj")

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_command_time
        remaining = self._bounds_cfg.min_command_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_command_time = time.time()

    # ------------------------------------------------------------------
    # Internal: state polling
    # ------------------------------------------------------------------

    def _poll_state(self) -> None:
        """Read one packet from port 30003 and update cached state + controller state."""
        state_data = self._state_reader.read_state()
        if state_data is None:
            # Connection might be lost
            if self._state not in (ControllerState.DISCONNECTED, ControllerState.CONNECTING):
                log.debug("State read returned None")
            return

        if "joints" in state_data:
            self._cached_joints = state_data["joints"]
        if "joint_velocities" in state_data:
            self._cached_joint_velocities = state_data["joint_velocities"]
        if "target_joints" in state_data:
            self._cached_target_joints = state_data["target_joints"]
        if "tcp_pose" in state_data:
            self._cached_tcp = state_data["tcp_pose"]
        if "tcp_force" in state_data:
            self._cached_force = state_data["tcp_force"]
        if "robot_mode" in state_data:
            self._cached_robot_mode = state_data["robot_mode"]
        if "safety_mode" in state_data:
            self._cached_safety_mode = state_data["safety_mode"]

        # Derive controller state from raw modes
        self._update_controller_state()

    def _update_controller_state(self) -> None:
        """Map raw robot_mode / safety_mode to ControllerState.

        Uses joint velocities (from port 30003) as the authoritative signal
        for whether the robot is still moving, rather than position-diffing.
        """
        sm = self._cached_safety_mode
        rm = self._cached_robot_mode

        # Safety issues take priority
        if sm is not None:
            if sm == _SAFETY_PROTECTIVE_STOP:
                self._set_state(ControllerState.PROTECTIVE_STOP)
                return
            if sm in (_SAFETY_VIOLATION, _SAFETY_FAULT):
                self._set_state(ControllerState.SAFETY_FAULT)
                return
            if sm in (_SAFETY_SYSTEM_ESTOP, _SAFETY_ROBOT_ESTOP, _SAFETY_SAFEGUARD_STOP):
                self._set_state(ControllerState.ERROR)
                return

        if rm is not None:
            if rm == _ROBOT_MODE_POWER_OFF:
                self._set_state(ControllerState.POWER_OFF)
                return
            if rm == _ROBOT_MODE_BOOTING:
                self._set_state(ControllerState.BOOTING)
                return
            if rm == _ROBOT_MODE_BACKDRIVE:
                if self._state == ControllerState.FREEDRIVE:
                    return
                self._set_state(ControllerState.FREEDRIVE)
                return
            if rm in (_ROBOT_MODE_RUNNING, _ROBOT_MODE_IDLE):
                if self._state == ControllerState.MOVING:
                    # Don't allow MOVING→READY until the grace period has
                    # elapsed.  This prevents false "motion complete" when the
                    # robot hasn't physically started yet (command still
                    # travelling over SSH tunnel).
                    elapsed = time.time() - self._moving_since
                    if elapsed >= self._MOVING_GRACE_PERIOD and self._check_motion_complete():
                        self._set_state(ControllerState.READY)
                elif self._state == ControllerState.FREEDRIVE:
                    pass  # Keep freedrive state
                else:
                    self._set_state(ControllerState.READY)
                return

    # ------------------------------------------------------------------
    # Internal: power-up sequence
    # ------------------------------------------------------------------

    def _ensure_powered(self) -> None:
        """If robot is powered off, power on and release brakes."""
        self._poll_state()
        if self._state in (ControllerState.POWER_OFF, ControllerState.BOOTING):
            log.info("Robot not powered, auto-powering on...")
            self._recovery.power_on_and_release_brakes()
            # Wait for it to come up
            for _ in range(30):
                time.sleep(1.0)
                self._poll_state()
                if self._state == ControllerState.READY:
                    log.info("Robot powered on and ready")
                    return
            log.warning("Robot did not reach READY after power-on")
        elif self._state in (ControllerState.PROTECTIVE_STOP, ControllerState.SAFETY_FAULT):
            self._try_auto_recover()

    # ------------------------------------------------------------------
    # Internal: automatic recovery
    # ------------------------------------------------------------------

    def _try_auto_recover(self) -> bool:
        """Attempt automatic recovery based on current state. Returns True on success.

        Uses _recovery_lock to prevent concurrent recovery from the health
        monitor and main thread (e.g. if the main thread's move_joints fails
        while the health monitor also detects the fault).
        """
        if not self._recovery_lock.acquire(blocking=False):
            log.debug("Recovery already in progress on another thread, skipping")
            return False
        try:
            return self._do_recover()
        finally:
            self._recovery_lock.release()

    def _do_recover(self) -> bool:
        """Inner recovery loop (must be called while holding _recovery_lock)."""
        for attempt in range(1, self._max_recovery + 1):
            if self._monitor_stop.is_set():
                log.info("Recovery aborted: shutdown requested")
                return False

            current = self._state
            log.info(
                "Auto-recovery attempt %d/%d for %s",
                attempt,
                self._max_recovery,
                current.value,
            )

            ok = False
            try:
                if current == ControllerState.PROTECTIVE_STOP:
                    ok = self._recovery.recover_protective_stop()
                elif current == ControllerState.SAFETY_FAULT:
                    ok = self._recovery.recover_safety_fault()
                elif current == ControllerState.POWER_OFF:
                    ok = self._recovery.recover_power_off()
                else:
                    log.warning("No recovery procedure for state %s", current.value)
                    break
            except Exception as e:
                log.error("Recovery attempt %d failed with exception: %s", attempt, e)
                ok = False

            if self._monitor_stop.is_set():
                log.info("Recovery aborted: shutdown requested")
                return False

            if ok:
                # Re-poll to confirm
                time.sleep(1.0)
                self._poll_state()
                if self._state == ControllerState.READY:
                    log.info("Recovery successful")
                    return True
                log.warning("Recovery reported success but state is %s", self._state.value)

            time.sleep(2.0)
            self._poll_state()

        log.error("Auto-recovery exhausted after %d attempts", self._max_recovery)
        self._set_state(ControllerState.ERROR)
        return False

    # ------------------------------------------------------------------
    # Internal: health monitor thread
    # ------------------------------------------------------------------

    def _health_monitor(self) -> None:
        """Background thread: poll state and trigger recovery on faults."""
        state_poll_interval = 0.1  # 100ms
        dashboard_poll_interval = 1.0
        last_dashboard_poll = 0.0

        while not self._monitor_stop.is_set():
            try:
                self._poll_state()
                self._publish_state()

                # Periodic dashboard cross-check (authoritative safety mode
                # source, in case port 30003 offset is wrong for this firmware).
                now = time.time()
                if now - last_dashboard_poll > dashboard_poll_interval:
                    last_dashboard_poll = now
                    try:
                        safety_str = self._recovery.safety_mode()
                        if safety_str:
                            upper = safety_str.upper()
                            if "PROTECTIVE_STOP" in upper:
                                self._set_state(ControllerState.PROTECTIVE_STOP)
                            elif "FAULT" in upper or "VIOLATION" in upper:
                                self._set_state(ControllerState.SAFETY_FAULT)
                            elif "EMERGENCY" in upper or "SAFEGUARD" in upper:
                                self._set_state(ControllerState.ERROR)
                    except Exception:
                        pass  # Dashboard unavailable; rely on port 30003

                # Auto-recover on detected faults
                if self._state == ControllerState.PROTECTIVE_STOP:
                    log.warning("Health monitor detected PROTECTIVE_STOP, recovering...")
                    self._try_auto_recover()
                elif self._state == ControllerState.SAFETY_FAULT:
                    log.warning("Health monitor detected SAFETY_FAULT, recovering...")
                    self._try_auto_recover()
                elif self._state == ControllerState.POWER_OFF:
                    if self._auto_power:
                        log.warning("Health monitor detected POWER_OFF, recovering...")
                        self._try_auto_recover()

            except Exception as e:
                log.error("Health monitor error: %s", e)
                if self._monitor_stop.is_set():
                    break
                # Try to reconnect state reader
                try:
                    self._state_reader.connect()
                except Exception:
                    pass

            self._monitor_stop.wait(state_poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = sys.argv[1:]
    if not args:
        _print_cli_usage()
        return 0

    command = args[0].lower()

    # Determine robot IP (allow override via env or --ip flag)
    import os

    robot_ip = os.environ.get("ROBOT_IP", "127.0.0.1")

    if command == "status":
        return _cli_status(robot_ip)
    elif command == "home":
        return _cli_move(robot_ip, "home")
    elif command == "move" and len(args) >= 2:
        return _cli_move(robot_ip, args[1])
    elif command == "joints" and len(args) >= 7:
        degrees = [float(a) for a in args[1:7]]
        radians = [math.radians(d) for d in degrees]
        return _cli_joints(robot_ip, radians)
    elif command == "gripper" and len(args) >= 2:
        return _cli_gripper(robot_ip, args[1])
    elif command == "freedrive":
        return _cli_freedrive(robot_ip)
    elif command == "positions":
        return _cli_positions()
    elif command == "recover":
        return _cli_recover(robot_ip)
    else:
        _print_cli_usage()
        return 1


def _print_cli_usage() -> None:
    print("Safe UR3 Controller CLI")
    print()
    print("Usage:")
    print("  uv run src/safe_controller.py status")
    print("  uv run src/safe_controller.py home")
    print("  uv run src/safe_controller.py move <position_name>")
    print("  uv run src/safe_controller.py joints <j0> <j1> <j2> <j3> <j4> <j5>  (degrees)")
    print("  uv run src/safe_controller.py gripper <open|close|activate>")
    print("  uv run src/safe_controller.py freedrive")
    print("  uv run src/safe_controller.py positions")
    print("  uv run src/safe_controller.py recover")
    print()
    print("Environment:")
    print("  ROBOT_IP=127.0.0.1  (default, override with this env var)")


def _cli_status(robot_ip: str) -> int:
    print(f"Connecting to {robot_ip}...")
    try:
        with SafeUR3Controller(robot_ip, auto_power=False) as robot:
            status = robot.get_status()
            print()
            print(f"  State:      {status['state']}")
            print(f"  Robot IP:   {status['robot_ip']}")
            print(f"  Port 30003: {'connected' if status['state_reader_connected'] else 'FAILED'}")
            print(f"  Port 29999: {'connected' if status['dashboard_connected'] else 'FAILED'}")
            if status["dashboard"]:
                print(f"  Robot mode: {status['dashboard'].get('robot_mode', '?')}")
                print(f"  Safety:     {status['dashboard'].get('safety_mode', '?')}")
            if status["joints_deg"]:
                print(f"  Joints (°): {status['joints_deg']}")
            if status["tcp_pose"]:
                tcp = status["tcp_pose"]
                print(
                    f"  TCP (m):    x={tcp[0]:.3f} y={tcp[1]:.3f} z={tcp[2]:.3f}"
                )
            print()
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    return 0


def _cli_move(robot_ip: str, position_name: str) -> int:
    print(f"Moving to '{position_name}'...")
    try:
        with SafeUR3Controller(robot_ip) as robot:
            ok = robot.move_to(position_name)
            if ok:
                print("  Done.")
            else:
                print("  Move did not complete within timeout.")
                return 1
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    return 0


def _cli_joints(robot_ip: str, radians: list[float]) -> int:
    deg_str = ", ".join(f"{math.degrees(r):.1f}" for r in radians)
    print(f"Moving to joints [{deg_str}]° ...")
    try:
        with SafeUR3Controller(robot_ip) as robot:
            ok = robot.move_joints(radians)
            if ok:
                print("  Done.")
            else:
                print("  Move did not complete within timeout.")
                return 1
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    return 0


def _cli_gripper(robot_ip: str, action: str) -> int:
    action = action.lower()
    try:
        with SafeUR3Controller(robot_ip) as robot:
            if action == "open":
                print("Opening gripper...")
                robot.gripper_open()
            elif action == "close":
                print("Closing gripper...")
                robot.gripper_close()
            elif action == "activate":
                print("Activating gripper...")
                robot.gripper_activate()
            else:
                print(f"Unknown gripper action: {action}")
                return 1
            print("  Done.")
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    return 0


def _cli_freedrive(robot_ip: str) -> int:
    print("Entering freedrive mode. Press Ctrl+C to exit.")
    try:
        with SafeUR3Controller(robot_ip) as robot:
            robot.freedrive(enable=True)
            try:
                while True:
                    joints = robot.capture_position()
                    if joints:
                        deg = [f"{math.degrees(j):.1f}" for j in joints]
                        print(f"\r  Joints (°): {deg}  ", end="", flush=True)
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n  Exiting freedrive...")
                robot.freedrive(enable=False)
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    return 0


def _cli_positions() -> int:
    print("Predefined positions:")
    print()
    for name, joints in sorted(POSITIONS.items()):
        deg = [f"{math.degrees(j):.1f}" for j in joints]
        print(f"  {name:18s} {deg}")
    print()
    return 0


def _cli_recover(robot_ip: str) -> int:
    print(f"Running recovery on {robot_ip}...")
    try:
        with SafeUR3Controller(robot_ip, auto_power=True) as robot:
            if robot.is_ready():
                print("  Robot is already READY.")
            else:
                ok = robot._try_auto_recover()
                if ok:
                    print("  Recovery successful.")
                else:
                    print("  Recovery failed.")
                    return 1
    except Exception as e:
        print(f"  Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
