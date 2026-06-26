"""Mock gym-like env for closed-loop eval tests (no Isaac Sim/GPU required).

Tiny stub: fixed obs dim, scripted reward/termination so a known policy yields
a known success rate. Used by tests/test_eval_closed_loop.py.
"""
import numpy as np


class MockPickEnv:
    """Fake pick-and-place env: 12308-dim obs, 7-dim action, scripted termination."""

    def __init__(self, obs_dim: int = 12308, action_dim: int = 7, success_threshold: float = 0.5):
        """success_threshold: action[0] > threshold → success."""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.success_threshold = success_threshold
        self._step_count = 0
        self._episode_reward = 0.0

    def reset(self):
        """Reset to random obs."""
        self._step_count = 0
        self._episode_reward = 0.0
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        """Step: scripted termination (action[0] > threshold OR step > 10)."""
        self._step_count += 1

        # Scripted reward: action[0] encodes "approach quality"
        reward = float(action[0])
        self._episode_reward += reward

        # Success if action[0] > threshold (policy "closes gripper")
        success = action[0] > self.success_threshold

        # Terminate on success or timeout
        terminated = success
        truncated = self._step_count > 10

        info = {"success": success}

        obs = np.random.randn(self.obs_dim).astype(np.float32)
        return obs, reward, terminated, truncated, info
