"""Behavioral tests for verify_failure.py negative-run verifier.

Tests the actual verify_failure.main() with mocked SageMaker responses.
Verifies that: terminal failures pass, nonterminal executions fail,
wrong failure reasons fail, downstream activity fails, ARN-bearing
non-success registration fails, and pagination is exercised.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_failure.py")
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])


def _load_module():
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    if "src" not in sys.path:
        sys.path.insert(0, str(pathlib.Path(_REPO_ROOT) / "src"))
    spec = importlib.util.spec_from_file_location("verify_failure", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mock_sm(execution_status, steps, pages=1):
    sm = MagicMock()
    sm.describe_pipeline_execution.return_value = {
        "PipelineExecutionStatus": execution_status
    }
    if pages == 1:
        sm.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": steps
        }
    else:
        first = {"PipelineExecutionSteps": steps[:1], "NextToken": "page2"}
        second = {"PipelineExecutionSteps": steps[1:]}
        sm.list_pipeline_execution_steps.side_effect = [first, second]
    return sm


def _clean_failure_steps():
    return [
        {"StepName": "Validate", "StepStatus": "Failed",
         "FailureReason": "eval_seed mismatch: expected=100 got=999"},
    ]


def test_clean_terminal_failure_passes(monkeypatch):
    mod = _load_module()
    sm = _mock_sm("Failed", _clean_failure_steps())
    monkeypatch.setattr(mod.boto3, "client", lambda *a, **kw: sm)
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(region="us-east-1"))
    monkeypatch.setattr(sys, "argv", ["verify_failure.py", "arn:test", "Validate", "eval_seed mismatch"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0


def test_nonterminal_execution_fails(monkeypatch):
    mod = _load_module()
    sm = _mock_sm("Executing", _clean_failure_steps())
    monkeypatch.setattr(mod.boto3, "client", lambda *a, **kw: sm)
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(region="us-east-1"))
    monkeypatch.setattr(sys, "argv", ["verify_failure.py", "arn:test", "Validate", "eval_seed mismatch"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_wrong_failure_reason_fails(monkeypatch):
    mod = _load_module()
    steps = [{"StepName": "Validate", "StepStatus": "Failed",
              "FailureReason": "image pull error"}]
    sm = _mock_sm("Failed", steps)
    monkeypatch.setattr(mod.boto3, "client", lambda *a, **kw: sm)
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(region="us-east-1"))
    monkeypatch.setattr(sys, "argv", ["verify_failure.py", "arn:test", "Validate", "eval_seed mismatch"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_stopped_gate_causes_failure(monkeypatch):
    mod = _load_module()
    steps = _clean_failure_steps() + [
        {"StepName": "SuccessGate", "StepStatus": "Stopped"},
    ]
    sm = _mock_sm("Failed", steps)
    monkeypatch.setattr(mod.boto3, "client", lambda *a, **kw: sm)
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(region="us-east-1"))
    monkeypatch.setattr(sys, "argv", ["verify_failure.py", "arn:test", "Validate", "eval_seed mismatch"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_failed_register_with_arn_causes_failure(monkeypatch):
    mod = _load_module()
    steps = _clean_failure_steps() + [
        {"StepName": "RegisterModel", "StepStatus": "Failed",
         "Metadata": {"RegisterModel": {"Arn": "arn:aws:sagemaker:us-east-1:123:model-package/pkg"}}},
    ]
    sm = _mock_sm("Failed", steps)
    monkeypatch.setattr(mod.boto3, "client", lambda *a, **kw: sm)
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(region="us-east-1"))
    monkeypatch.setattr(sys, "argv", ["verify_failure.py", "arn:test", "Validate", "eval_seed mismatch"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_pagination_exercised(monkeypatch):
    mod = _load_module()
    steps = _clean_failure_steps() + [
        {"StepName": "SimEval", "StepStatus": "Succeeded",
         "StartTime": "2026-01-01T00:00:00Z"},
    ]
    sm = _mock_sm("Failed", steps, pages=2)
    monkeypatch.setattr(mod.boto3, "client", lambda *a, **kw: sm)
    monkeypatch.setattr(mod, "load_config", lambda: MagicMock(region="us-east-1"))
    monkeypatch.setattr(sys, "argv", ["verify_failure.py", "arn:test", "Validate", "eval_seed mismatch"])
    with pytest.raises(SystemExit):
        mod.main()
    assert sm.list_pipeline_execution_steps.call_count == 2
