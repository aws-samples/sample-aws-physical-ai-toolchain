# GR1 arms-only GR00T modality config for fine-tuning on the Isaac Lab Arena GR1
# manipulation dataset (nvidia/Arena-GR1-Manipulation-Task) with GR00T N1.6.
#
# This MIRRORS Isaac Lab Arena's
#   isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py
# and registers under EmbodimentTag.GR1 -- the NATIVE pretrained GR1 head.
#
# WHY (vs the N1.7 sibling arena_gr1_data_config.py which uses NEW_EMBODIMENT):
# our N1.6 pin (defaults.json repo_commit_n16 = 5dc80c4, n1.6.1-release) HAS a
# native `GR1` member in EmbodimentTag ("gr1"), verified against the pinned
# source (gr00t/data/embodiment_tags.py). N1.6's pretrained GR00T-N1.6-3B ships
# the GR1 head, so we fine-tune with --embodiment-tag gr1 (no new_embodiment
# custom slot, no n17->n16 action shim at eval). "gr1" is NOT pre-registered in
# MODALITY_CONFIGS (keys: unitree_g1/libero_panda/oxe_*/behavior_r1_pro), so
# registering here is collision-free. At eval the GR00T server is launched with
# the SAME tag (EMBODIMENT_TAG=gr1) so it matches the checkpoint metadata.
#
# The key layout + 16-step action horizon EXACTLY match the dataset's
# lerobot/meta/modality.json (state/action split left_arm[0:7] right_arm[7:14]
# left_hand[14:20] right_hand[20:26] = 26-dim; video ego_view; language
# annotation.human.action.task_description) and are byte-identical to the N1.7
# sibling config -- only the registration tag differs.
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
# module, which registers the config into MODALITY_CONFIGS["gr1"].
register_modality_config(gr1_arms_only_config, embodiment_tag=EmbodimentTag.GR1)
