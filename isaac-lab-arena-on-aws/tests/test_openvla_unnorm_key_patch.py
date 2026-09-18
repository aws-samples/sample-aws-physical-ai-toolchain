"""I3: the manifest's unnorm_key must reach upstream VERBATIM.

The entrypoint patched only upstream's initial assignment, so upstream's `_no_noops`
fallback still fired: a manifest key ABSENT from norm_stats whose `_no_noops` variant
existed was silently substituted, and the recorded input configuration then described a
different normalization from the one actually applied to actions.

These tests run the PRODUCTION patcher over a verbatim copy of the pinned upstream
function and execute the result, rather than re-implementing the patch -- a test that
pasted its own version of the edit would keep passing with production broken.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _REPO_ROOT / "entrypoints/eval/libero/openvla/eval_entry.py"

# Copied verbatim from moojink/openvla-oft at the commit this component pins
# (OFT_COMMIT = e4287e94541f459edc4feabc4e181f537cd569a8),
# experiments/robot/libero/run_libero_eval.py :: check_unnorm_key.
# Production fails loud if the real file no longer contains the patch targets, so this
# copy going stale cannot silently disable the guarantee.
_PINNED_UPSTREAM = '''
def check_unnorm_key(cfg, model):
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key
'''


class _Cfg:
    def __init__(self, unnorm_key, suite="libero_spatial"):
        self.unnorm_key = unnorm_key
        self.task_suite_name = suite


class _Model:
    def __init__(self, norm_stats):
        self.norm_stats = norm_stats


def _entry_module():
    """Load the entrypoint far enough to reach its module-level patcher."""
    # The module reads its configuration at import time; these values are irrelevant to
    # the patcher but must be present for the import to succeed.
    os.environ.setdefault("EVAL_CHECKPOINT", "/opt/ml/input/data/model")
    os.environ.setdefault("EVAL_SUITE", "libero_spatial")
    os.environ.setdefault("EVAL_SEED", "1000")
    os.environ.setdefault("EVAL_TRIALS", "3")
    os.environ.setdefault("EVAL_TASK_IDS", "all")
    for name, attrs in (("digest", {"weights_digest": lambda *a, **k: "0" * 64}),
                        ("validator", {"validate_report": lambda *a, **k: 0.0})):
        if name not in sys.modules:
            sys.modules[name] = types.SimpleNamespace(**attrs)
    spec = importlib.util.spec_from_file_location("openvla_eval_entry_under_test", _ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patched_check():
    """Apply the PRODUCTION patch to the pinned upstream text and return the function."""
    patched = _entry_module()._patch_unnorm_key(_PINNED_UPSTREAM)
    namespace: dict = {}
    exec(patched, namespace)
    return namespace["check_unnorm_key"]


def test_an_absent_manifest_key_is_not_silently_replaced():
    """Astra's probe: manifest key absent, only its _no_noops variant present.

    This previously resolved to the _no_noops key without a word.
    """
    check = _patched_check()
    cfg = _Cfg("claimed_exact_key")
    with pytest.raises(AssertionError, match="claimed_exact_key not found"):
        check(cfg, _Model({"claimed_exact_key_no_noops": {}}))


def test_a_present_manifest_key_is_applied_verbatim():
    """Even when a _no_noops sibling exists, the manifest value must win."""
    check = _patched_check()
    cfg = _Cfg("claimed_exact_key")
    check(cfg, _Model({"claimed_exact_key": {}, "claimed_exact_key_no_noops": {}}))
    assert cfg.unnorm_key == "claimed_exact_key"


def test_upstream_fallback_is_preserved_without_an_explicit_key():
    """With no manifest key the upstream _no_noops behaviour must be unchanged."""
    check = _patched_check()
    cfg = _Cfg("")
    check(cfg, _Model({"libero_spatial_no_noops": {}}))
    assert cfg.unnorm_key == "libero_spatial_no_noops"


@pytest.mark.parametrize("removed,expected", [
    ("    unnorm_key = cfg.task_suite_name", "assignment not found"),
    ('        unnorm_key = f"{unnorm_key}_no_noops"', "_no_noops fallback not found"),
])
def test_a_changed_upstream_stops_the_run(removed, expected):
    """If upstream moves, the verbatim guarantee cannot be made -- fail, do not proceed."""
    module = _entry_module()
    with pytest.raises(RuntimeError, match=expected):
        module._patch_unnorm_key(_PINNED_UPSTREAM.replace(removed, "    pass"))


def test_the_patch_targets_still_match_the_pinned_upstream_text():
    """Guards the fixture: each target must appear exactly once in the pinned function."""
    for target in ("    unnorm_key = cfg.task_suite_name",
                   '        unnorm_key = f"{unnorm_key}_no_noops"'):
        assert _PINNED_UPSTREAM.count(target) == 1, target
