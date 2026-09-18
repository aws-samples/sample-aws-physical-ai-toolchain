"""Tests for vla_pipeline.dose_curve (dose-extraction submit layer).

Covers the default-OFF semantics, the enabled predicate, the WARN-and-downgrade
single-GPU precondition (result-preserving, never a hard error), and the
pure-passthrough-when-disabled parameter injection.
"""
from vla_pipeline.dose_curve import (
    DOSE_PARAM_NAMES,
    FINAL_ONLY_SAVE_STEPS,
    DoseCurveConfig,
)


def test_default_is_dose_off():
    # save_steps defaults to the final-only sentinel (>> max_steps) => disabled.
    cfg = DoseCurveConfig(max_steps=2000)
    assert cfg.save_steps == FINAL_ONLY_SAVE_STEPS
    assert cfg.enabled is False


def test_enabled_when_save_below_max():
    assert DoseCurveConfig(max_steps=2000, save_steps=250).enabled is True
    # boundary: save_steps == max_steps is final-only (NOT a curve).
    assert DoseCurveConfig(max_steps=2000, save_steps=2000).enabled is False


def test_preconditions_noop_when_disabled():
    cfg = DoseCurveConfig(max_steps=2000)  # disabled
    gpus, warns = cfg.validate_preconditions(requested_num_gpus=4)
    assert gpus == 4 and warns == []


def test_preconditions_warn_and_downgrade_multi_gpu():
    cfg = DoseCurveConfig(max_steps=2000, save_steps=250)  # enabled
    gpus, warns = cfg.validate_preconditions(requested_num_gpus=8)
    assert gpus == 1  # forced single-GPU
    assert len(warns) == 1 and "single-GPU" in warns[0]
    # It is a WARNING, not a raise -- the call returns normally.


def test_preconditions_single_gpu_no_warning():
    cfg = DoseCurveConfig(max_steps=2000, save_steps=250)
    gpus, warns = cfg.validate_preconditions(requested_num_gpus=1)
    assert gpus == 1 and warns == []


def test_preconditions_escape_hatch_keeps_multi_gpu():
    cfg = DoseCurveConfig(max_steps=2000, save_steps=250,
                          supports_multi_gpu_intermediates=True)
    gpus, warns = cfg.validate_preconditions(requested_num_gpus=8)
    assert gpus == 8 and warns == []


def test_inject_params_noop_when_disabled():
    cfg = DoseCurveConfig(max_steps=2000)  # disabled
    base = {"ModelFamily": "gr00t", "TrainSteps": "2000"}
    out = cfg.inject_dose_params(base)
    assert out == base  # unchanged
    for k in DOSE_PARAM_NAMES:
        assert k not in out


def test_inject_params_sets_three_knobs_when_enabled():
    cfg = DoseCurveConfig(max_steps=2000, save_steps=250,
                          eval_dose_steps="250,500,2000", volume_size_gb=500)
    out = cfg.inject_dose_params({"ModelFamily": "gr00t"})
    assert out["TrainSaveSteps"] == "250"
    assert out["EvalDoseSteps"] == "250,500,2000"
    assert out["VolumeSizeInGB"] == "500"
    # original untouched keys preserved
    assert out["ModelFamily"] == "gr00t"


def test_inject_params_does_not_mutate_input():
    cfg = DoseCurveConfig(max_steps=2000, save_steps=250)
    base = {"ModelFamily": "gr00t"}
    cfg.inject_dose_params(base)
    assert "TrainSaveSteps" not in base  # input dict unmutated
