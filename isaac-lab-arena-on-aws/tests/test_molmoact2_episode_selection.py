"""I12: the recorded dataset provenance must describe the data actually consumed.

Training selects a PREFIX of the dataset -- capped at 100 episodes, floored at 5, scaled
by the requested step count -- but the manifest recorded only the dataset's full
episode_count. A run over the first five episodes of a 1693-episode dataset therefore
carried the same provenance as a run over all of them, and at smaller doses changing the
step count silently changed the training data too.

These tests exercise the production selection function, so the recorded rule and the
argv handed to training cannot drift apart.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types

import pytest

_TRAIN_ENTRY = (pathlib.Path(__file__).resolve().parents[1]
                / "entrypoints/train/molmoact2/train_entry.py")


def _module():
    """Load the trainer's module-level helpers.

    The module reads configuration at import; these values do not affect the selection
    function but must be present for the import to succeed.
    """
    import os
    os.environ.setdefault("TRAIN_MAX_STEPS", "200")
    os.environ.setdefault("TRAIN_DATASET_S3URI", "s3://bucket/datasets/fridge")
    os.environ.setdefault("TRAIN_SUITE", "libero_spatial")
    os.environ.setdefault("SM_MODEL_DIR", "/tmp/model")
    for name, attrs in (("digest", {"weights_digest": lambda *a, **k: "0" * 64}),
                        ("validator", {"validate_manifest": lambda *a, **k: None})):
        if name not in sys.modules:
            sys.modules[name] = types.SimpleNamespace(**attrs)
    spec = importlib.util.spec_from_file_location("molmo_train_under_test", _TRAIN_ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("max_steps,episode_count,expected", [
    (200, 1693, 100),    # cap applies
    (20, 1693, 10),      # steps // 2
    (2, 1693, 5),        # floor applies
    (200, 7, 7),         # dataset smaller than the cap
    (10000, 1693, 100),  # cap still applies at large doses
])
def test_the_selection_is_a_bounded_prefix(max_steps, episode_count, expected):
    mod = _module()
    selected = mod._select_episodes(max_steps, episode_count)
    assert selected == list(range(expected))
    assert len(selected) <= episode_count


def test_the_dose_changes_the_dataset_which_is_why_it_must_be_recorded():
    """The finding's substance: step count and training data are coupled."""
    mod = _module()
    assert len(mod._select_episodes(20, 1693)) != len(mod._select_episodes(200, 1693))


def test_an_empty_dataset_is_rejected():
    mod = _module()
    with pytest.raises(ValueError, match="episode_count must be >= 1"):
        mod._select_episodes(200, 0)


def test_the_manifest_records_the_selection_and_its_rule():
    """The manifest must carry both the consumed count and the rule that produced it.

    Asserted over the source because the manifest is assembled deep inside a training
    run that cannot be executed here; the validator tests cover the schema side, and the
    selection function above covers the value.
    """
    src = _TRAIN_ENTRY.read_text()
    tree = ast.parse(src)
    # Find the dataset_manifest dict literal and check its keys.
    keys: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "dataset_manifest" \
                        and isinstance(value, ast.Dict):
                    keys = [k.value for k in value.keys
                            if isinstance(k, ast.Constant)]
    assert keys, "dataset_manifest dict literal not found"
    assert "episodes_selected" in keys, keys
    assert "episode_selection" in keys, keys


def test_the_argv_and_the_manifest_use_the_same_selection():
    """Both must read the single selected_episodes list, so they cannot disagree."""
    src = _TRAIN_ENTRY.read_text()
    assert "selected_episodes = _select_episodes(" in src
    assert "','.join(str(i) for i in selected_episodes)" in src
    assert '"episodes_selected": len(selected_episodes)' in src
