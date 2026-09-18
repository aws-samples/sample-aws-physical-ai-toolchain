"""Audit: the dose-curve is OFF by default.

The pipeline's ``TrainSaveSteps`` default must keep every run final-only (dose
DISABLED) unless a caller explicitly lowers ``save_steps`` below ``max_steps``.
Per the effective-default analysis, this checks the EFFECTIVE default semantics
via ``dose_curve.DoseCurveConfig`` -- NOT a literal ``SAVE_STEPS == MAX_STEPS``
requirement (defaults.json carries no step keys; the default lives in pipeline.py).
"""
from vla_pipeline.dose_curve import FINAL_ONLY_SAVE_STEPS, DoseCurveConfig


def test_pipeline_default_save_steps_matches_final_only_sentinel():
    """The pipeline's TrainSaveSteps default_value must equal the dose_curve
    final-only sentinel, so an un-overridden execution is dose-OFF."""
    from vla_pipeline.pipeline import build_parameters
    params = build_parameters()
    assert params["save_steps"].default_value == FINAL_ONLY_SAVE_STEPS


def test_dose_off_by_default_across_realistic_max_steps():
    """With save_steps at its default (final-only sentinel), dose is OFF for any
    realistic training length (sample 5 .. NVIDIA reference 20000)."""
    for m in (5, 100, 2000, 20000):
        cfg = DoseCurveConfig(max_steps=m)  # save_steps defaults to FINAL_ONLY_SAVE_STEPS
        assert cfg.enabled is False, f"dose unexpectedly ON at max_steps={m}"


def test_dose_on_only_when_save_below_max():
    """Sanity: dose turns ON exactly when a caller lowers save_steps below max."""
    assert DoseCurveConfig(max_steps=2000, save_steps=250).enabled is True
