"""UR3 runtime control config for the closed-loop GR00T policy runner.

This holds only the *runtime* knobs the on-robot control loop needs — action
chunking, control rate, and safety clamps. It deliberately does NOT define the
GR00T dataset modality (state/action/video index ranges): that lives in one
canonical place, `containers/groot-training/ur3_modality_config.py`, and is
written into each dataset's `meta/modality.json` by
`training/groot/convert_zarr_to_lerobot.py`. Keeping dataset schema out of here
avoids the two-sources-of-truth drift that misaligns training and inference.

Action space (for reference — enforced at dataset build time, not here):
    arm     6D Cartesian velocity x dt (RELATIVE, from `speedl` teleop)
    gripper 1D absolute position (0.0 = open, 1.0 = closed)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UR3ControlConfig:
    """Runtime parameters for the on-robot GR00T control loop."""

    # Action chunking: the policy predicts `action_horizon` steps; we execute the
    # first `execute_horizon` before re-querying (receding-horizon control).
    action_horizon: int = 16
    execute_horizon: int = 4

    # Control rate (Hz) — must match the camera/dataset rate the model trained on.
    control_hz: int = 5

    # Safety clamps applied to every commanded step before it reaches the arm.
    max_vel_delta: float = 0.05   # m/step — clamp on Cartesian velocity x dt
    force_limit: float = 120.0    # N — protective-stop threshold (~88N static from gravity)


# Singleton used by the control loop.
UR3_CONFIG = UR3ControlConfig()
