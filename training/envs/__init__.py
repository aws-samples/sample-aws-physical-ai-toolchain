"""UR3 pick-and-place environment registration.

Registers `PickAndPlaceUR3-v0` as a gymnasium / Isaac Lab task so that
`gym.make(...)` and Isaac Lab's `parse_env_cfg("PickAndPlaceUR3-v0")` can resolve
it (previously the env existed but was never registered, so the task id didn't
resolve and any RL launch against it failed).

⚠️ NOT YET VALIDATED ON A GPU. The env body (pick_and_place_ur3.py) still has
known gaps that can only be confirmed/fixed on a G-family GPU with Isaac Sim:
  - uses the legacy `omni.isaac.lab.*` namespace (newer Isaac Lab is `isaaclab.*`)
  - robot/bin USD paths use `omniverse://localhost/...` (a local Nucleus server
    that is not present in headless SageMaker — needs a cloud asset root)
  - the DifferentialIKController is constructed per-step in `_apply_action`
    (should be built once in __init__)
See docs/ROADMAP.md (Feature 2). Until those are resolved, use a built-in task
(e.g. Isaac-Velocity-Flat-Anymal-D-v0) for validated RL runs — see
training/scripts/launch_rl.py.

The registration is guarded so importing this package never hard-fails when
gymnasium or the Isaac Lab deps aren't installed (e.g. on a plain dev box).
"""

# Eagerly expose the env classes WHEN Isaac Lab is importable. On a bare box
# (no `omni.isaac.lab`), importing the env module raises ModuleNotFoundError —
# we tolerate that so `import training.envs` (and registration via string entry
# point) still works for tooling/tests that don't need the live classes.
try:
    from .pick_and_place_ur3 import PickAndPlaceUR3Env, PickAndPlaceUR3EnvCfg
except ModuleNotFoundError:
    PickAndPlaceUR3Env = None  # type: ignore
    PickAndPlaceUR3EnvCfg = None  # type: ignore

__all__ = ["PickAndPlaceUR3Env", "PickAndPlaceUR3EnvCfg", "TASK_ID", "REGISTERED"]

TASK_ID = "PickAndPlaceUR3-v0"


def _register() -> bool:
    """Register the task with gymnasium. Returns True on success, False if the
    registry/deps are unavailable (so a bare import never crashes)."""
    try:
        import gymnasium as gym
    except Exception:
        return False

    # Idempotent: don't re-register if already present.
    try:
        if TASK_ID in gym.registry:
            return True
    except Exception:
        pass

    gym.register(
        id=TASK_ID,
        entry_point=f"{__name__}.pick_and_place_ur3:PickAndPlaceUR3Env",
        disable_env_checker=True,
        kwargs={
            # Isaac Lab resolves the cfg from this entry point.
            "env_cfg_entry_point": f"{__name__}.pick_and_place_ur3:PickAndPlaceUR3EnvCfg",
        },
    )
    return True


REGISTERED = _register()
