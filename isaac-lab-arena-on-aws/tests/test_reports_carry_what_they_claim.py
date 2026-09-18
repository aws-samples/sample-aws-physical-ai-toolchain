"""R4#M1: output assertions replacing source-text assertions that could not fail.

Four tests asserted that a literal string appeared in a source file. A source-text assertion
passes whether or not the behaviour is correct, so each of these stayed green under a mutation
that made the produced report wrong. Astra's R4 review demonstrated the mutations; this module
asserts the OUTPUT instead, and each test here fails under the mutation its docstring names.

The source-text originals are deleted rather than kept alongside. A test that cannot fail is
not coverage, and leaving it in place implies a check that does not exist.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tarfile
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ARENA_ENTRY = _REPO_ROOT / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"
_ARENA_SHARED = _REPO_ROOT / "entrypoints/eval/isaac_arena/_shared"
_COMMON = _REPO_ROOT / "src/vla_pipeline/common"


def _load_arena(monkeypatch):
    """Load the Arena evaluator with the flat import layout the image gives it."""
    monkeypatch.setenv("SM_HP_TASK_NAME", "fixture_task")
    monkeypatch.setenv("SM_HP_EMBODIMENT_TAG", "GR1")
    monkeypatch.syspath_prepend(str(_COMMON))
    monkeypatch.syspath_prepend(str(_ARENA_SHARED))
    spec = importlib.util.spec_from_file_location(f"arena_m1_{len(sys.modules)}", _ARENA_ENTRY)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Arena evaluator needs a module absent here: {exc!r}")
    return module


def _snapshot_tree(tmp_path, name="posctrl"):
    root = tmp_path / f"{name}_tree"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "gr00t"}')
    return root


def test_the_positive_control_can_publish_at_all(tmp_path, monkeypatch):
    """R4#I1: the positive control was refused publication before reaching its own branch.

    The positive control serves NVIDIA's published HF snapshot through a remote GR00T server, so:

      line 1835  run_arena_eval(policy_type="gr00t_remote")
      line  701  is_remote = True
      line  841  results["policy_type"] = "checkpoint"      <- overwritten
      line 1206  if _RESOLVED is None and policy_type != "positive_control": raise
      line 1242  the positive-control branch                 <- never reached

    _RESOLVED is only set by archive extraction, and there is no mounted archive in this mode, so
    the guard fires on every positive-control run. The branch that knows how to build a snapshot
    report sits below the guard that refuses to let it get there.

    This is what four review passes reported. It stayed invisible because the gate fixture staged
    an incomplete layout and failed earlier, on an import.
    """
    module = _load_arena(monkeypatch)
    checkpoint_root = _snapshot_tree(tmp_path)

    monkeypatch.setenv("EVAL_POSCTRL_N16", "true")
    monkeypatch.setattr(module, "EVAL_TRIALS", 3)
    monkeypatch.setattr(module, "_RESOLVED", None)
    monkeypatch.setitem(sys.modules, "validator", types.SimpleNamespace(
        validate_report=lambda *a, **k: 0.0, validate_manifest=lambda *a, **k: None))

    # Exactly what run_arena_eval returns for this mode: is_remote rewrote policy_type.
    results = {"eval_backend": "isaac_lab_arena", "episodes": 3, "success_rate": 0.6666666666666666,
               "policy_type": "checkpoint", "task_name": "fixture_task"}
    output = tmp_path / "out"
    module.write_metrics(results, str(output), str(checkpoint_root),
                         posctrl_commit="c" * 40)

    assert (output / "metrics.json").exists(), (
        "the positive control produced no metrics.json: publication was refused before the "
        "positive-control branch could build the snapshot report")


def test_the_positive_control_report_names_the_resolved_commit_it_was_given(tmp_path, monkeypatch):
    """R4#M1: was `assert "positive_control_commit" in source`.

    Mutation that the old assertion allowed: replace the resolved commit with a constant. The
    string still appeared in the source, so the test passed while the report attested a commit
    the run never resolved -- the exact untraceability the positive control exists to avoid.

    This asserts the report carries the commit PASSED IN, so a constant fails it.
    """
    module = _load_arena(monkeypatch)
    resolved_commit = "c" * 40
    checkpoint_root = _snapshot_tree(tmp_path)

    monkeypatch.setenv("EVAL_POSCTRL_N16", "true")
    monkeypatch.setattr(module, "EVAL_TRIALS", 3)
    monkeypatch.setattr(module, "_RESOLVED", None)
    monkeypatch.setitem(sys.modules, "validator", types.SimpleNamespace(
        validate_report=lambda *a, **k: 0.0, validate_manifest=lambda *a, **k: None))

    results = {"eval_backend": "isaac_lab_arena", "episodes": 3, "success_rate": 0.6666666666666666,
               "policy_type": "checkpoint", "task_name": "fixture_task"}
    output = tmp_path / "out"
    module.write_metrics(results, str(output), str(checkpoint_root),
                         posctrl_commit=resolved_commit)

    report = json.loads((output / "metrics.json").read_text())
    carried = json.dumps(report)
    assert resolved_commit in carried, (
        "the positive-control report does not carry the commit the Hub resolved for these "
        f"weights; a constant or the requested reference would be attested instead. {carried[:400]}")
    assert "source_archive" not in report, (
        "the positive control serves an HF snapshot, not a mounted archive; an archive field "
        "means the report names bytes that did not supply the inference weights")
