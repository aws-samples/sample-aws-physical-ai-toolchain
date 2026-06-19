"""training.envs must import cleanly on a bare box and expose the task id /
registration logic without hard-failing when Isaac Lab isn't installed."""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_envs_import_clean_without_isaac():
    import training.envs as e
    # Bare import must not raise even though omni.isaac.lab is absent.
    assert e.TASK_ID == "PickAndPlaceUR3-v0"
    # REGISTERED is a bool (False on a box without gymnasium, True where present).
    assert isinstance(e.REGISTERED, bool)


def test_registration_idempotent_when_gym_present():
    import importlib
    try:
        import gymnasium as gym
    except Exception:
        import pytest
        pytest.skip("gymnasium not installed on this box")
    import training.envs as e
    # If gym is present, the task must be registered and re-import must not double-register.
    assert e.REGISTERED is True
    assert e.TASK_ID in gym.registry
    importlib.reload(e)
    assert e.TASK_ID in gym.registry
