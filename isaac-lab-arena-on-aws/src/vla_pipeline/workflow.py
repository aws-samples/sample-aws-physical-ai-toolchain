"""The dependency order and parameter contract of the supported pipeline prefixes."""
from __future__ import annotations

STEPS = ("FineTune", "SimEval", "Validate", "SuccessGate", "RegisterModel")


def selected_steps(through="RegisterModel", *, checkpoint=False):
    if through not in STEPS:
        raise ValueError(f"Unknown stopping step {through!r}; choose {', '.join(STEPS)}")
    start = 1 if checkpoint else 0
    end = STEPS.index(through) + 1
    if start >= end:
        raise ValueError("A supplied checkpoint omits FineTune; select SimEval or a later step")
    return list(STEPS[start:end])


def parameter_keys(steps):
    """Internal build_parameters keys referenced by these steps.

    Full graphs retain the historical parameter declarations. Prefixes omit
    unused required images/source/hardware so callers never stage omitted work.
    """
    keys = set()
    if "FineTune" in steps:
        keys.update({
            "model_family", "train_steps", "save_steps", "train_suite", "dataset_s3_uri",
            "dataset_revision", "train_image", "train_instance", "volume_size",
            "max_runtime_seconds", "train_source_dir", "gr00t_version",
        })
    if "SimEval" in steps:
        keys.update({
            "model_family", "suite", "eval_image", "eval_instance", "eval_volume_size",
            "max_runtime_seconds", "eval_seed", "eval_trials", "eval_task_ids",
            "eval_dose_steps", "eval_source_dir", "use_groot_server", "arena_connector",
            "eval_sim_config", "gr00t_version",
        })
    if "Validate" in steps:
        keys.update({
            "volume_size", "expected_embodiment_tag", "expected_arena_embodiment",
            "expected_arena_object", "expected_policy_config", "success_threshold",
        })
    if "RegisterModel" in steps:
        keys.add("registry_group")
    return keys
