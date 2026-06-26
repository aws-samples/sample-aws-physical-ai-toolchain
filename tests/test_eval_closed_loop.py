"""Tests for closed-loop evaluator (no GPU/sockets — uses InProcessTransport + mock env)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training" / "scripts"))

from eval_protocol import encode_obs, decode_obs, encode_action, decode_action, InProcessTransport
from eval_mock_env import MockPickEnv


def test_message_round_trip():
    """Obs/action encode/decode round-trip."""
    obs_orig = np.random.randn(12308).astype(np.float32)
    obs_bytes = encode_obs(obs_orig)
    obs_decoded = decode_obs(obs_bytes)
    np.testing.assert_allclose(obs_orig, obs_decoded, rtol=1e-5)

    action_orig = np.random.randn(7).astype(np.float32)
    action_bytes = encode_action(action_orig)
    action_decoded = decode_action(action_bytes)
    np.testing.assert_allclose(action_orig, action_decoded, rtol=1e-5)


def test_in_process_transport():
    """InProcessTransport calls policy_fn directly (no sockets)."""
    def dummy_policy(obs: np.ndarray) -> np.ndarray:
        # Echo first 7 dims as action
        return obs[:7]

    transport = InProcessTransport(dummy_policy)
    obs = np.random.randn(12308).astype(np.float32)
    action = transport.send(obs)
    np.testing.assert_allclose(action, obs[:7])


def test_eval_loop_control_flow():
    """Eval loop over mock env: reset → step until term/trunc → record success."""
    env = MockPickEnv(success_threshold=0.5)

    # Policy that always outputs action[0]=0.6 (above threshold → success)
    def always_succeed_policy(obs: np.ndarray) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[0] = 0.6  # "close gripper"
        return action

    transport = InProcessTransport(always_succeed_policy)

    successes = 0
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action = transport.send(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("success"):
            successes += 1

    # All 10 episodes should succeed (action[0]=0.6 > threshold=0.5)
    assert successes == 10


def test_success_rate_math():
    """7/10 scripted successes → 70.0% success rate."""
    env = MockPickEnv(success_threshold=0.5)

    # Policy that outputs action[0] based on episode number (modulo 10)
    # Episodes 0,3,7,8,9 → action[0]=0.3 (fail)
    # Episodes 1,2,4,5,6 → action[0]=0.6 (success) = 5 successes
    # Let's aim for 7/10: episodes 0,1,2,3,4,5,6 succeed, 7,8,9 fail
    episode_num = {"i": 0}

    def scripted_policy(obs: np.ndarray) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        # First 7 episodes: action[0]=0.6 (success), last 3: action[0]=0.3 (fail)
        if episode_num["i"] < 7:
            action[0] = 0.6  # success
        else:
            action[0] = 0.3  # fail
        return action

    transport = InProcessTransport(scripted_policy)

    successes = 0
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action = transport.send(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("success"):
            successes += 1
        episode_num["i"] += 1

    # 7 successes out of 10 → 70%
    assert successes == 7
    success_rate_pct = round(successes / 10 * 100, 1)
    assert success_rate_pct == 70.0


def test_zmq_transport_import_guarded():
    """ZmqTransport fails gracefully if pyzmq not installed (tested by import guard)."""
    # This test verifies the import guard exists; actual ZMQ socket test would need pyzmq.
    # The lazy import in eval_protocol.py ensures ZmqTransport.__init__ only imports zmq
    # when called, so the module loads cleanly even without pyzmq.
    from eval_protocol import ZmqTransport  # Should not crash on import

    # Attempting to instantiate without pyzmq would raise ImportError (tested manually)
