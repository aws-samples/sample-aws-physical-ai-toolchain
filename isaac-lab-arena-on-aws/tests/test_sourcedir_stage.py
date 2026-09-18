"""Tests for vla_pipeline.common.sourcedir.stage (layout refactor).

stage() is a behaviour-preserving no-op: it must produce the same
staged file set the old inline runner.stage_sourcedir produced, and it adds a
hard fail on duplicate basenames (never last-write-wins).
"""
import os
import tempfile
from pathlib import Path

import pytest

from vla_pipeline.common import sourcedir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _staged_names(family: str) -> set:
    d = sourcedir.stage(family)
    try:
        return {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}
    finally:
        pass


def test_stage_gr00t_includes_train_eval_shared_and_arena_cfg():
    names = _staged_names("gr00t")
    # family entries + defaults + shared modules the entries import
    assert "train_entry.py" in names
    assert "eval_entry.py" in names
    assert "defaults.json" in names
    assert "digest.py" in names
    assert "validator.py" in names
    # gr00t train-time Arena GR1 modality config, pulled from steps/isaac_arena/
    assert "arena_gr1_data_config.py" in names


def test_stage_openvla_includes_shared_modules():
    names = _staged_names("openvla")
    assert "train_entry.py" in names
    assert "eval_entry.py" in names
    assert "digest.py" in names
    assert "validator.py" in names


def test_stage_includes_required_inventory():
    """stage() must include the required files per family. This is an EXPLICIT
    inventory check -- not a comparison against runner.stage_sourcedir, which now
    delegates to stage() and would be tautological (a == a)."""
    required = {
        "gr00t": {"train_entry.py", "eval_entry.py", "defaults.json",
                  "digest.py", "validator.py", "rlds_validator.py",
                  "arena_gr1_data_config.py"},
        "openvla": {"train_entry.py", "eval_entry.py", "defaults.json",
                    "digest.py", "validator.py", "rlds_validator.py"},
        "molmoact2": {"train_entry.py", "eval_entry.py", "defaults.json",
                      "digest.py", "validator.py", "rlds_validator.py"},
        "dummy": {"train_entry.py", "eval_entry.py", "defaults.json",
                  "digest.py", "validator.py", "rlds_validator.py"},
    }
    for family, req in required.items():
        d = sourcedir.stage(family)
        staged = {f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))}
        missing = req - staged
        assert not missing, f"{family}: stage() missing required files {missing}"


def test_stage_missing_family_hard_fails():
    with pytest.raises(FileNotFoundError) as ei:
        sourcedir.stage("no_such_family")
    assert "no_such_family" in str(ei.value)


def test_stage_duplicate_basename_hard_fails(monkeypatch):
    """A manifest with two sources of the same basename must raise, not clobber."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"
        b = Path(tmp) / "b"
        a.mkdir()
        b.mkdir()
        (a / "utils.py").write_text("# a\n")
        (b / "utils.py").write_text("# b\n")

        monkeypatch.setattr(
            sourcedir, "_staging_manifest",
            lambda root, family: [a / "utils.py", b / "utils.py"],
        )
        with pytest.raises(ValueError) as ei:
            sourcedir.stage("whatever")
        assert "duplicate basename" in str(ei.value)
        assert "utils.py" in str(ei.value)
