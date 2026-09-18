"""I1: the normalization file the loader reads is NAMED BY base/config.json.

Pinned MolmoAct2 reads config.json, takes norm_stats_filename, and loads statistics from the
resulting path (lerobot @ a4f15bf3, processor_molmoact2.py:144-152). Both preflights checked that
the CONVENTIONAL base/norm_stats.json exists, which proved nothing about the file actually used --
a checkpoint could redirect to different statistics and still pass, because the conventional file
sat there satisfying the check.

Line 152 is a pathlib join, so an absolute value REPLACES the base directory and reads from
outside the checkpoint entirely.
"""
from __future__ import annotations

import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EVAL = _ROOT / "entrypoints/eval/libero/molmoact2/eval_entry.py"
_TRAIN = _ROOT / "entrypoints/train/molmoact2/train_entry.py"


def _helper():
    """Exec only the helper: both module bodies do real work at import."""
    src = _TRAIN.read_text()
    start = src.index("def _resolve_norm_stats")
    end = src.index("\ndef ", start + 10)
    ns: dict = {}
    exec(src[start:end], ns)
    return ns["_resolve_norm_stats"]


class _Stop(Exception):
    pass


def _run(tmp_path, config, files):
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")

    def fail(msg):
        raise _Stop(msg)

    return _helper()(str(tmp_path), lambda m: None, fail)


def test_the_real_published_shape_is_accepted(tmp_path):
    """allenai/MolmoAct2-LIBERO config.json carries norm_stats_filename="norm_stats.json"."""
    assert _run(tmp_path, {"norm_stats_filename": "norm_stats.json"}, ["norm_stats.json"])


def test_an_absent_key_falls_back_to_the_conventional_name(tmp_path):
    assert _run(tmp_path, {}, ["norm_stats.json"])


def test_a_redirect_to_a_real_in_checkpoint_file_is_allowed(tmp_path):
    """Legitimate: the digest now covers whatever config.json names."""
    assert _run(tmp_path, {"norm_stats_filename": "alt.json"}, ["alt.json"])


def test_an_absolute_filename_is_refused(tmp_path):
    with pytest.raises(_Stop, match="ABSOLUTE"):
        _run(tmp_path, {"norm_stats_filename": "/etc/n.json"}, ["norm_stats.json"])


def test_an_escaping_filename_is_refused(tmp_path):
    with pytest.raises(_Stop, match="resolves outside"):
        _run(tmp_path, {"norm_stats_filename": "../n.json"}, ["norm_stats.json"])


def test_a_redirect_to_a_missing_file_is_refused(tmp_path):
    """The conventional file existing must NOT satisfy a redirect elsewhere.

    This is the exact hole: base/norm_stats.json present, config.json naming something else.
    """
    with pytest.raises(_Stop, match="missing"):
        _run(tmp_path, {"norm_stats_filename": "other.json"}, ["norm_stats.json"])


def test_a_non_string_filename_is_refused(tmp_path):
    with pytest.raises(_Stop, match="not a usable filename"):
        _run(tmp_path, {"norm_stats_filename": 5}, ["norm_stats.json"])


def test_an_unreadable_config_is_refused(tmp_path):
    (tmp_path / "config.json").write_text("{not json")

    def fail(msg):
        raise _Stop(msg)

    with pytest.raises(_Stop, match="unreadable"):
        _helper()(str(tmp_path), lambda m: None, fail)


def test_both_preflights_resolve_rather_than_hardcode():
    """Fixing one of two paths has been a recurring defect in this component."""
    for path in (_EVAL, _TRAIN):
        source = path.read_text()
        assert "_resolve_norm_stats(base_dir, log" in source, f"{path.name} does not resolve"
    # The hardcoded existence check must be gone from both.
    for path in (_EVAL, _TRAIN):
        source = path.read_text()
        assert 'os.path.join(base_dir, "norm_stats.json")' not in source, (
            f"{path.name} still hardcodes the conventional filename")
