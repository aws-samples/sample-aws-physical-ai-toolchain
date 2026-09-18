"""Behavioral tests for N1.6 Arena eval_entry embodiment-tag guard.

Verifies that an incompatible tag is rejected before extraction/setup/rollout,
and that GR1 proceeds and reports GR1 in effective_eval_config.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from unittest.mock import MagicMock

import pytest

_EVAL_ENTRY = (pathlib.Path(__file__).resolve().parents[1]
               / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py")


def _make_arena_module(monkeypatch, tag, version="n16"):
    monkeypatch.setenv("EVAL_GR00T_VERSION", version)
    monkeypatch.setenv("SM_HP_EMBODIMENT_TAG", tag)
    monkeypatch.setenv("EVAL_SEED", "100")
    monkeypatch.setenv("EVAL_TRIALS", "1")
    monkeypatch.setenv("EVAL_TASK_IDS", "all")
    monkeypatch.setenv("SM_HP_NUM_STEPS", "10")
    monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", "/workspace/test.yaml")
    monkeypatch.setenv("EVAL_ARENA_EMBODIMENT", "gr1_joint")
    monkeypatch.setenv("EVAL_OBJECT", "NONE")
    monkeypatch.setenv("EVAL_SIM_CONFIG", json.dumps({"task_name": "test_task"}))
    monkeypatch.setenv("SM_HP_USE_GROOT_SERVER", "true")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    digest_mock = MagicMock()
    digest_mock.weights_digest = MagicMock(return_value="sha256:" + "a" * 64)
    monkeypatch.setitem(sys.modules, "digest", digest_mock)
    validator_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "validator", validator_mock)
    spec = importlib.util.spec_from_file_location("n16_tag_test", _EVAL_ENTRY)
    mod = importlib.util.module_from_spec(spec)
    return spec, mod


def test_incompatible_n16_tag_rejected(monkeypatch):
    spec, mod = _make_arena_module(monkeypatch, "UNITREE_G1")
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit) as exc_info:
        mod.main()
    assert exc_info.value.code == 1


def test_gr1_tag_accepted(monkeypatch):
    spec, mod = _make_arena_module(monkeypatch, "GR1")
    spec.loader.exec_module(mod)
    assert mod._DECLARED_POLICY_CONFIG == "/workspace/test.yaml"


def test_version_is_normalized_at_import(monkeypatch):
    """An unnormalized value must not make the guards, the launcher and the report
    disagree with each other.

    Previously the version was re-read raw at each site while only the launcher applied
    .strip().lower(). With EVAL_GR00T_VERSION="N16": the early tag guard compared raw
    ("N16" != "n16") and was skipped, the launcher normalized and started the N1.6 server
    (which hardcodes --embodiment-tag GR1), and the reporter also compared raw and so
    recorded the REQUESTED tag. The report then claimed an embodiment the server never
    served -- and the digest chain cannot detect that.
    """
    spec, mod = _make_arena_module(monkeypatch, "GR1", version="  N16  ")
    spec.loader.exec_module(mod)
    assert mod.EVAL_GR00T_VERSION == "n16"


def test_unsupported_version_rejected_at_import(monkeypatch):
    spec, mod = _make_arena_module(monkeypatch, "GR1", version="n18")
    with pytest.raises(RuntimeError, match="unsupported EVAL_GR00T_VERSION"):
        spec.loader.exec_module(mod)


def test_incompatible_n16_tag_rejected_even_when_version_is_uppercase(monkeypatch):
    """The guard must fire on the normalized value.

    With the raw comparison, `EVAL_GR00T_VERSION="N16"` skipped this guard entirely and
    an incompatible tag reached the server.
    """
    spec, mod = _make_arena_module(monkeypatch, "UNITREE_G1", version="N16")
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.main()
