# GR1 arms-only GR00T modality config for fine-tuning on the Isaac Lab Arena GR1
# manipulation dataset (nvidia/Arena-GR1-Manipulation-Task).
#
# This MIRRORS Isaac Lab Arena's
#   isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py
# EXCEPT it registers under EmbodimentTag.NEW_EMBODIMENT instead of GR1.
#
# WHY: our pinned gr00t commit (376ba890, defaults.json repo_commit) has NO
# `GR1` member in EmbodimentTag (verified against the pinned source: the enum has
# LIBERO_PANDA/NEW_EMBODIMENT/ROBOCASA_GR1_TABLETOP/UNITREE_G1/... but no bare
# GR1). Arena's own file uses `EmbodimentTag.GR1`, which would AttributeError at
# 376ba890. gr00t@376ba890's custom-embodiment finetuning slot is NEW_EMBODIMENT,
# and register_modality_config() defaults to it. So we register the identical
# arms-only spec under NEW_EMBODIMENT and fine-tune with --embodiment-tag
# new_embodiment. At eval, the GR00T server is launched with the SAME tag
# (EMBODIMENT_TAG=new_embodiment) so it matches the checkpoint's metadata.json.
#
# The key layout + 16-step action horizon EXACTLY match the dataset's
# lerobot/meta/modality.json (state/action split left_arm[0:7] right_arm[7:14]
# left_hand[14:20] right_hand[20:26] = 26-dim; video ego_view; language
# annotation.human.action.task_description).
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

gr1_arms_only_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        sin_cos_embedding_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16-step horizon (matches Arena gr1_manip config)
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in range(4)
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Side-effect on import: launch_finetune.py's load_modality_config() imports this
# module, which registers the config into MODALITY_CONFIGS["new_embodiment"].
register_modality_config(gr1_arms_only_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
