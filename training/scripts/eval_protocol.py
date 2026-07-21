"""Closed-loop evaluation protocol: message encoding + transport abstraction.

Mirrors NVIDIA GR00T's PolicyServer/PolicyClient pattern (ZMQ REQ/REP on port 5555).
Defines the wire format for policy server ↔ sim client communication and provides
both a ZMQ socket transport (for real eval on Lab 2 workstation) and an in-process
transport (for laptop/CI tests that have no GPU/sockets).

NOTE: For built-in RL, in-process evaluation via `play.py` is the simpler documented
path. On a real robot, the transport is ROS2/NITROS, not ZMQ. This ZMQ pattern is a
teaching example that aligns with GR00T's architecture.

Message schema v1: JSON body (obs/action are float arrays). ~150KB/msg for 12308-dim
obs is acceptable for episodic eval (not real-time edge inference — swap to msgpack
later behind same signatures if needed).
"""
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np


def encode_obs(obs: np.ndarray) -> bytes:
    """Encode observation array to bytes (JSON v1)."""
    return json.dumps({"obs": obs.tolist()}).encode("utf-8")


def decode_obs(data: bytes) -> np.ndarray:
    """Decode observation from bytes."""
    msg = json.loads(data.decode("utf-8"))
    return np.array(msg["obs"], dtype=np.float32)


def encode_action(action: np.ndarray) -> bytes:
    """Encode action array to bytes (JSON v1)."""
    return json.dumps({"action": action.tolist()}).encode("utf-8")


def decode_action(data: bytes) -> np.ndarray:
    """Decode action from bytes."""
    msg = json.loads(data.decode("utf-8"))
    return np.array(msg["action"], dtype=np.float32)


class Transport(ABC):
    """Abstract transport for policy queries (obs → action)."""

    @abstractmethod
    def send(self, obs: np.ndarray) -> np.ndarray:
        """Send observation, receive action."""
        pass

    def close(self):
        """Release resources (optional)."""
        pass


class ZmqTransport(Transport):
    """ZMQ REQ socket transport (lazy-import pyzmq)."""

    def __init__(self, endpoint: str):
        """Connect to policy server at endpoint (e.g. 'tcp://localhost:5555')."""
        try:
            import zmq
        except ImportError as e:
            raise ImportError(
                "pyzmq is required for ZmqTransport. Install: pip install pyzmq>=25"
            ) from e

        self._endpoint = endpoint
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(endpoint)

    def send(self, obs: np.ndarray) -> np.ndarray:
        """Send obs, block until action received."""
        self._socket.send(encode_obs(obs))
        action_bytes = self._socket.recv()
        return decode_action(action_bytes)

    def close(self):
        """Close socket and context."""
        self._socket.close()
        self._context.term()


class InProcessTransport(Transport):
    """In-process transport for tests (no sockets, calls policy_fn directly)."""

    def __init__(self, policy_fn: Callable[[np.ndarray], np.ndarray]):
        """policy_fn: obs array → action array."""
        self._policy_fn = policy_fn

    def send(self, obs: np.ndarray) -> np.ndarray:
        """Call policy_fn synchronously."""
        return self._policy_fn(obs)
