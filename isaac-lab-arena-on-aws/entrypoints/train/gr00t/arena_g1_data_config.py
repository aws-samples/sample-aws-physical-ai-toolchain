# G1 whole-body-control (WBC) GR00T modality config for fine-tuning on the
# Isaac Lab Arena G1 loco-manipulation dataset (nvidia/Arena-G1-Loco-Manipulation-Task).
#
# This MIRRORS Isaac Lab Arena's N1.7 config
#   isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py
# EXCEPT it registers under EmbodimentTag.NEW_EMBODIMENT instead of GR1/UNITREE_G1.
#
# WHY NEW_EMBODIMENT: our pinned gr00t commit (376ba890, defaults.json repo_commit,
# N1.7 / Gr00tN1d7Pipeline) exposes the custom-embodiment finetuning slot as
# NEW_EMBODIMENT (register_modality_config() defaults to it). At eval, the GR00T
# server is launched with the SAME tag (EMBODIMENT_TAG=new_embodiment) so it matches
# the checkpoint's metadata.json. This mirrors the proven GR1 arms-only path.
#
# WHY WBC (NOT full 43-DoF): G1 loco-manipulation uses whole-body control — a
# separate WBC policy drives the legs. GR00T outputs ONLY the upper body + waist
# joints plus two high-level WBC commands (base_height, navigate). The dataset's
# lerobot/meta/modality.json exposes the full 43-DoF body split (incl. legs), but
# Arena's N1.7 policy config trains on the WBC subset below. Verified against the
# authoritative Arena source (raw.githubusercontent.com/isaac-sim/IsaacLab-Arena
# .../embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py), NOT guessed.
#
# Action horizon = 40 steps (delta_indices range(40)), matching the N1.7 config
# (the N1.6 g1_locomanip_gr00t_closedloop_config.yaml action_horizon=50 pairs with
# the N1.6 data config; our N1.7 fine-tune uses this N1.7 data config's range(40)).
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# State: upper-body joints + waist (WBC handles the legs separately).
_STATE_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
# Action: upper-body joints + waist + two high-level WBC command groups.
_ACTION_KEYS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
]

unitree_g1_sim_wbc_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=list(_STATE_KEYS),
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),  # 40-step horizon (matches Arena N1.7 g1 WBC config)
        modality_keys=list(_ACTION_KEYS),
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in _ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Side-effect on import: launch_finetune.py's load_modality_config() imports this
# module, registering the config into MODALITY_CONFIGS["new_embodiment"].
register_modality_config(unitree_g1_sim_wbc_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
