"""I2/I3: UNOBSERVED success telemetry must invalidate the episode -- at the right boundary.

Pinned upstream accepts success from three places, in order: per step from top-level
``env_infos["success"]``, per step from ``env_infos["final_info"][i]["success"]``, and again
from ``final_info`` when the episode ends. When the top-level key was absent it did
``current_successes[i] = False``, which turned a missing metric signal into an ordinary zero
score and -- because it assigns rather than accumulates -- erased a success already recorded
for that episode.

A first attempt at this raised in that else-branch. That was wrong: it fired before the
other two sources could contribute, so a run reporting success only at episode end (the
normal terminal-telemetry pattern) crashed instead of scoring. The absence decision belongs
at the EPISODE boundary, once every source has had its chance.

These tests EXECUTE the patched pinned function against fake environments, so they check
behaviour rather than the shape of the edit.
"""
from __future__ import annotations

import importlib.util
import os
import ast
import pathlib
import re
import shutil
import sys
import tempfile
import types

import numpy as np
import pytest

_ENTRY = (pathlib.Path(__file__).resolve().parents[1]
          / "entrypoints/eval/libero/gr00t/eval_entry.py")
_DEFAULTS = (pathlib.Path(__file__).resolve().parents[1]
             / "entrypoints/train/gr00t/defaults.json")
# Verbatim from NVIDIA/Isaac-GR00T at the pinned commit
# (376ba890cff8c9de64d71d982772a9c36185fdd7), gr00t/eval/rollout_policy.py. Production
# refuses to run if its anchors are absent, so this copy going stale cannot silently
# disable the guarantee.
_PINNED = pathlib.Path(__file__).with_name("data") / "rollout_policy_pinned.py"


def _module():
    os.environ["EVAL_GR00T_VERSION"] = "n17"
    for name, attrs in (("digest", {"weights_digest": lambda *a, **k: "0" * 64}),
                        ("validator", {"validate_report": lambda *a, **k: 0.0})):
        if name not in sys.modules:
            sys.modules[name] = types.SimpleNamespace(**attrs)
    staged = pathlib.Path(tempfile.mkdtemp(prefix="libero-telemetry-"))
    shutil.copy(_ENTRY, staged / "eval_entry.py")
    shutil.copy(_DEFAULTS, staged / "defaults.json")
    spec = importlib.util.spec_from_file_location(
        "libero_telemetry_under_test", staged / "eval_entry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _grab(name: str, text: str) -> str:
    """A top-level def, up to the NEXT top-level def/class/decorator."""
    start = text.index(f"def {name}(")
    lines = text[start:].splitlines(keepends=True)
    out = [lines[0]]
    for line in lines[1:]:
        if re.match(r"^(def |class |@)", line):
            break
        out.append(line)
    return "".join(out)


def _patched_rollout():
    """Apply the PRODUCTION patch to the pinned source and return the live function."""
    pytest.importorskip("tqdm")
    import tqdm
    from collections import defaultdict

    patched = _module()._patch_success_telemetry(_PINNED.read_text())
    helper = _grab("_macro_step_env_steps", patched).replace(
        "def _macro_step_env_steps(env_infos: dict, env_idx: int) -> int:",
        "def _macro_step_env_steps(env_infos, env_idx):")
    main = _grab("_collect_rollout_episodes", patched)
    for annotated, plain in (
        ("    policy: BasePolicy,\n", "    policy,\n"),
        ("    n_episodes: int,\n", "    n_episodes,\n"),
        ("    n_envs: int,\n", "    n_envs,\n"),
        ("    seed: int | None,\n", "    seed,\n"),
    ):
        main = main.replace(annotated, plain)
    namespace = {"np": np, "tqdm": tqdm.tqdm, "defaultdict": defaultdict}
    exec(compile(helper + "\n" + main, "pinned_rollout", "exec"), namespace)
    return namespace["_collect_rollout_episodes"]


class _Policy:
    def get_action(self, observations):
        return ({}, None)

    def reset(self):
        pass


def _env(info_factory):
    class _Env:
        def reset(self, seed=None):
            return ({}, {})

        def step(self, actions):
            return ({}, [0.0], [True], [False], info_factory())

    return _Env()


def _run(info_factory):
    return _patched_rollout()(_env(info_factory), _Policy(), 1, 1, 1)


def test_top_level_success_is_recorded():
    successes, *_ = _run(lambda: {"success": [True],
                                  "final_info": [{"success": [False]}]})
    assert successes == [True]


def test_a_legitimate_failure_still_records_as_failure():
    """An all-false outcome is a MEASUREMENT, not missing telemetry."""
    successes, *_ = _run(lambda: {"success": [False],
                                  "final_info": [{"success": [False]}]})
    assert successes == [False]


def test_terminal_only_telemetry_is_recorded():
    """I3: the first version of this patch raised here.

    Upstream accepts success from final_info in later branches, so raising in the
    top-level else-branch crashed runs that report success only at episode end.
    """
    successes, *_ = _run(lambda: {"final_info": [{"success": [True]}]})
    assert successes == [True]


def test_an_episode_with_no_telemetry_at_all_is_rejected():
    """I2: this used to be recorded as an ordinary unsuccessful episode."""
    with pytest.raises(RuntimeError, match="no success telemetry was observed"):
        _run(lambda: {})


def test_a_changed_upstream_stops_the_run():
    """If any anchor moves, the guarantee cannot be made -- fail, do not proceed."""
    module = _module()
    moved = _PINNED.read_text().replace(
        "    current_successes = [False] * n_envs\n", "    pass\n", 1)
    with pytest.raises(RuntimeError, match="patch anchor 1 occurs 0 times"):
        module._patch_success_telemetry(moved)


def test_the_patch_is_applied_on_both_baked_and_unbaked_paths():
    """A baked image carries the same unpatched upstream file."""
    source = _ENTRY.read_text()
    assert "_patch_success_telemetry(_rollout_src)" in source
    assert 'if "no success telemetry was observed" not in _rollout_src:' in source
    assert source.index("The commit check runs on BOTH paths") < source.index(
        "_patch_success_telemetry(_rollout_src)")


def _gr00t_shape_validator():
    """The validator the GR00T patch injects, executed rather than inspected."""
    namespace = {}
    exec(compile(_module()._TELEMETRY_VALIDATOR, "<validator>", "exec"), namespace)
    return namespace["_require_boolean_success"]


@pytest.mark.parametrize("value", [
    [[]],        # Astra's probe: recorded a completed FAILED episode
    [],          # no outcome reported
    2,           # Astra's probe: recorded SUCCESS
    1,
    "true",
    None,
])
def test_malformed_gr00t_telemetry_is_not_an_outcome(value):
    """C4: upstream's bool()/any() turned invalid values into ordinary outcomes.

    Probes against the pinned rollout showed [[]] recorded a completed failed episode and
    the integer 2 recorded a success. Marking telemetry observed AFTER that conversion could
    not recover the original value, so validation moved to the raw read.
    """
    with pytest.raises(ValueError):
        _gr00t_shape_validator()(value, "probe")


def test_a_gr00t_vector_containing_nan_is_rejected():
    """Astra's probe: NaN is truthy, so any() recorded a SUCCESS."""
    numpy = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="not boolean"):
        _gr00t_shape_validator()(numpy.array([True, numpy.nan]), "probe")


def test_genuine_gr00t_telemetry_is_accepted():
    numpy = pytest.importorskip("numpy")
    validator = _gr00t_shape_validator()
    assert validator(True, "probe") is True
    assert validator(numpy.bool_(False), "probe") is False
    assert validator([False, True], "probe") is True
    assert validator(numpy.array([False, False]), "probe") is False


def test_the_gr00t_patch_order_is_the_one_that_composes():
    """The order is load-bearing and NOT the intuitive one.

    Both patches touch the episode-end `any(final_info[...]["success"])` block. The telemetry
    patch APPENDS after it, leaving the text intact, so the shape patch can still find it.
    The shape patch REPLACES it, so running shape first destroys the telemetry patch's
    anchor. Applying them in the wrong order hard-fails the job at startup rather than
    silently skipping a guard -- but the right order must be pinned so nobody "tidies" it.
    """
    source = _ENTRY.read_text()
    assert source.index("_patch_success_telemetry(_rollout_src)") < source.index(
        "_patch_telemetry_shape(_rollout_src)")


def test_the_gr00t_patches_compose_in_production_order():
    """Executed, not asserted from text: both patches apply and the result is valid Python."""
    module = _module()
    pinned = (pathlib.Path(__file__).with_name("data")
              / "rollout_policy_pinned.py").read_text()
    combined = module._patch_telemetry_shape(
        module._patch_success_telemetry(pinned))
    ast.parse(combined)
    assert "_require_boolean_success" in combined
    assert "no success telemetry was observed" in combined


def test_the_reverse_gr00t_order_fails_loudly():
    """Proving the ordering constraint is real rather than superstition."""
    module = _module()
    pinned = (pathlib.Path(__file__).with_name("data")
              / "rollout_policy_pinned.py").read_text()
    with pytest.raises(RuntimeError, match="success-telemetry patch anchor 4 occurs 0"):
        module._patch_success_telemetry(module._patch_telemetry_shape(pinned))


def test_a_changed_upstream_stops_the_gr00t_shape_patch():
    module = _module()
    pinned = (pathlib.Path(__file__).with_name("data")
              / "rollout_policy_pinned.py").read_text()
    moved = pinned.replace(
        '                    env_success = env_infos["success"][env_idx]\n', "", 1)
    with pytest.raises(RuntimeError, match="telemetry-shape patch anchor 1 occurs 0 times"):
        module._patch_telemetry_shape(moved)
