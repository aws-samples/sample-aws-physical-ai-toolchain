"""A GENERIC component must not choose a FAMILY's hardware.

The family is selected per execution, so a value correct for one is wrong for another. Two places
chose anyway:

  pipeline.py declared default_value="ml.g6e.12xlarge" for both instance roles and 100 for both
  volumes. A caller who omitted them got those figures silently -- which is how a family declaring
  300 GB ran on 100, and how the launcher fix that made manifests authoritative could still be
  bypassed by submitting directly to the pipeline. One of those defaults (the eval volume) was added
  by the very commit that removed the launcher defaults, which is what made this worth a test rather
  than a note.

  dose_curve.DoseCurveConfig.volume_size_gb defaulted to 100 and inject_dose_params() wrote it
  UNCONDITIONALLY over whatever the caller had already resolved. So enabling a dose curve reverted a
  300 GB family to 100 -- while a curve run legitimately needs MORE disk, since it stages N
  intermediate checkpoints. Exactly backwards.

Both are checked behaviourally: the parameters' declared defaults are read off the real pipeline
objects, and the dose injection is executed.
"""
import pytest

from vla_pipeline.dose_curve import DoseCurveConfig


def test_no_hardware_parameter_carries_a_default():
    """Read off the REAL parameter objects, so this cannot drift from the declaration."""
    from vla_pipeline.pipeline import build_parameters

    params = build_parameters()
    hardware = {
        "train_instance": "TrainInstanceType",
        "eval_instance": "EvalInstanceType",
        "volume_size": "VolumeSizeInGB",
        "eval_volume_size": "EvalVolumeSizeInGB",
    }
    for key, declared_name in hardware.items():
        param = params[key]
        assert param.name == declared_name, f"{key} is not {declared_name}"
        assert getattr(param, "default_value", None) is None, (
            f"{declared_name} declares default {param.default_value!r}. A generic pipeline cannot "
            f"choose a family's hardware -- the family is selected per execution. A caller who omits "
            f"it must FAIL, not silently receive a figure nobody chose for this family")


def test_the_check_sees_a_parameter_that_legitimately_has_a_default():
    """Guards the guard: if no parameter anywhere has a default, the assertion above proves nothing."""
    from vla_pipeline.pipeline import build_parameters

    params = build_parameters()
    with_defaults = [k for k, v in params.items() if getattr(v, "default_value", None) is not None]
    assert with_defaults, (
        "no pipeline parameter declares a default at all, so the hardware check above cannot "
        "distinguish 'deliberately required' from 'nothing has defaults'")


def test_a_dose_curve_does_not_overwrite_a_resolved_volume():
    dose = DoseCurveConfig(max_steps=1000, save_steps=250)
    assert dose.enabled, "a curve run is the case under test"
    assert dose.volume_size_gb is None, "absence must be None, so the caller's value survives"

    resolved = {"VolumeSizeInGB": "300", "TrainSteps": "1000"}
    out = dose.inject_dose_params(resolved)
    assert out["VolumeSizeInGB"] == "300", (
        "the dose curve overwrote a volume the caller had already resolved from the family manifest; "
        "a curve run needs MORE disk, not the generic default")
    assert out["TrainSaveSteps"] == "250", "the dose settings it OWNS must still be applied"


def test_an_explicit_dose_volume_is_applied_and_validated():
    dose = DoseCurveConfig(max_steps=1000, save_steps=250, volume_size_gb=500)
    out = dose.inject_dose_params({"VolumeSizeInGB": "300"})
    assert out["VolumeSizeInGB"] == "500", "an explicit override must win"

    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            DoseCurveConfig(max_steps=1000, save_steps=250,
                            volume_size_gb=bad).inject_dose_params({})


def test_a_disabled_dose_curve_touches_nothing():
    dose = DoseCurveConfig(max_steps=1000)
    assert not dose.enabled
    resolved = {"VolumeSizeInGB": "300"}
    assert dose.inject_dose_params(resolved) == resolved
