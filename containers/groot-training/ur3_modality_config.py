"""
GR00T modality config for the UR3 arm (NEW_EMBODIMENT) — reference copy.

NOTE: the training job does NOT import this file. `train_entrypoint.py` writes an
identical config inline (write_embodiment_config) so the container is self-contained.
This file is kept as the canonical, readable reference for customizing the embodiment
when bringing your own robot (see docs/zarr-schema.md). Keep the two in sync.

Ported from the validated lab-cloud-env GR00T N1.6 reference. The action space is
EEF (end-effector): the arm actions are commanded Cartesian velocity × dt from
`speedl` teleop (RELATIVE deltas), and the gripper is an ABSOLUTE 0..1 position.

Keys + index ranges must match what convert_zarr_to_lerobot.py writes into the
dataset's meta/modality.json:
    state:  arm [0:6], gripper [6:7]   (UR3 is 6-DOF + 1-DOF gripper = 7D)
    action: arm [0:6], gripper [6:7]   (6 Cartesian-velocity deltas + 1 gripper)
    video:  wrist                       (single wrist camera)
    language: annotation.human.action.task_description

For a different robot, change modality_keys/index ranges here AND in
convert_zarr_to_lerobot.py (STATE_DIM/ACTION_DIM + meta/modality.json), and keep
the embodiment tag NEW_EMBODIMENT.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.data.embodiment_tags import EmbodimentTag


ur3_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            # arm: 6-value EEF action = 3 Cartesian translation + 3 rotation-vector
            # (vx,vy,vz,rx,ry,rz from speedl). MUST be XYZ_ROTVEC: DEFAULT makes
            # GR00T's pose loader reshape to a 4x4 (16-value) matrix and crash on
            # our 6 values. XYZ_ROTVEC parses translation=data[:3], rotation=data[3:].
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
            ),
            # gripper: absolute 0..1 position
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(ur3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
