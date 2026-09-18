"""Behavioral tests for MolmoAct2 eval_entry startup rejection.

Verifies that unsupported published-checkpoint mode is rejected before any
subprocess/setup work, and that the pipeline's local-checkpoint mode passes.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from unittest.mock import MagicMock

import pytest

def _load_with_env(monkeypatch, env_overrides):
    """Load MolmoAct2 eval_entry with controlled env vars, intercepting exit."""
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, v)
    for k in ("EVAL_CHECKPOINT", "EVAL_CKPT_REV", "EVAL_MODEL_SOURCE_URI",
              "EVAL_SUITE", "EVAL_SEED", "EVAL_TRIALS", "EVAL_TASK_IDS"):
        if k not in env_overrides:
            monkeypatch.delenv(k, raising=False)
    digest_mock = MagicMock()
    digest_mock.weights_digest = MagicMock(return_value="sha256:" + "a" * 64)
    monkeypatch.setitem(sys.modules, "digest", digest_mock)
    validator_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "validator", validator_mock)
    # Import the actual staged layout; never recreate an orphan in the source tree.
    from vla_pipeline.common.sourcedir import stage
    staged_entry = pathlib.Path(stage("molmoact2")) / "eval_entry.py"
    spec = importlib.util.spec_from_file_location("molmo_startup_test", staged_entry)
    mod = importlib.util.module_from_spec(spec)
    return spec, mod


def _loaded_module(monkeypatch):
    """Load the module far enough to reach its module-level helpers."""
    spec, mod = _load_with_env(monkeypatch, {
        "EVAL_CHECKPOINT": "/opt/ml/input/data/model",
        "EVAL_SUITE": "libero_spatial", "EVAL_SEED": "1000",
        "EVAL_TRIALS": "3", "EVAL_TASK_IDS": "all",
        "EVAL_MODEL_SOURCE_URI": "s3://bucket/key/model.tar.gz",
    })
    spec.loader.exec_module(mod)
    return mod


def test_adapter_audit_gate_reads_the_saved_policy_config(monkeypatch, tmp_path):
    """C2: the gate was an OPTIONAL manifest field, so deleting it skipped the audit.

    policy/config.json is what the loader itself reads -- it is why __init__ re-wraps
    with PEFT -- so it is the authoritative source. A manifest with no train_mode_vlm
    declaration must NOT be able to turn the audit off.
    """
    mod = _loaded_module(monkeypatch)
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(json.dumps({"train_mode_vlm": "lora"}))
    assert mod._resolve_train_mode(str(policy_dir), {}) == "lora"
    assert mod._resolve_train_mode(str(policy_dir), {"train_recipe": {}}) == "lora"


def test_an_omitted_train_mode_falls_back_to_the_upstream_default(monkeypatch, tmp_path):
    """An omitted field is not an absent mode.

    The upstream configuration class defaults train_mode_vlm to "lora", so the evaluator
    installs adapters even when the saved config says nothing. Reading the field with a
    bare .get() returned None and skipped the adapter audit on a run that WAS using
    LoRA -- the same shape of bypass as gating on the optional manifest field.
    """
    mod = _loaded_module(monkeypatch)
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(json.dumps({"some_other_field": 1}))
    assert mod._resolve_train_mode(str(policy_dir), {}) == "lora"
    assert mod.UPSTREAM_DEFAULT_TRAIN_MODE_VLM == "lora"


def test_a_null_train_mode_falls_back_to_the_upstream_default(monkeypatch, tmp_path):
    mod = _loaded_module(monkeypatch)
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(json.dumps({"train_mode_vlm": None}))
    assert mod._resolve_train_mode(str(policy_dir), {}) == "lora"


def test_adapter_audit_gate_rejects_manifest_disagreement(monkeypatch, tmp_path):
    """A provenance record describing a different policy must not be tolerated."""
    mod = _loaded_module(monkeypatch)
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(json.dumps({"train_mode_vlm": "lora"}))
    with pytest.raises(SystemExit):
        mod._resolve_train_mode(
            str(policy_dir), {"train_recipe": {"train_mode_vlm": "fft"}})


def test_adapter_audit_gate_fails_without_a_policy_config(monkeypatch, tmp_path):
    """An unresolvable mode must fail, not silently skip the audit."""
    mod = _loaded_module(monkeypatch)
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    with pytest.raises(SystemExit):
        mod._resolve_train_mode(str(policy_dir), {})


def test_published_checkpoint_rejected_before_work(monkeypatch):
    env = {
        "EVAL_CHECKPOINT": "allenai/MolmoAct2-LIBERO",
        "EVAL_CKPT_REV": "0d24a92bd1faf321ef497c3bbd5681af97c65aa2",
        "EVAL_SUITE": "libero_spatial",
        "EVAL_SEED": "1000",
        "EVAL_TRIALS": "1",
        "EVAL_TASK_IDS": "all",
    }
    import subprocess as real_subprocess
    call_count = {"run": 0, "popen": 0}
    orig_run = real_subprocess.run
    orig_popen = real_subprocess.Popen

    def counting_run(*a, **kw):
        call_count["run"] += 1
        return orig_run(*a, **kw)

    def counting_popen(*a, **kw):
        call_count["popen"] += 1
        return orig_popen(*a, **kw)

    monkeypatch.setattr(real_subprocess, "run", counting_run)
    monkeypatch.setattr(real_subprocess, "Popen", counting_popen)
    spec, mod = _load_with_env(monkeypatch, env)
    with pytest.raises(SystemExit) as exc_info:
        spec.loader.exec_module(mod)
    assert exc_info.value.code == 1
    assert call_count["run"] == 0, f"subprocess.run called {call_count['run']} times"
    assert call_count["popen"] == 0, f"subprocess.Popen called {call_count['popen']} times"


def test_quoted_identifier_rejected_before_work(monkeypatch):
    env = {
        "EVAL_CHECKPOINT": "'); print('injected'); #",
        "EVAL_CKPT_REV": "0d24a92bd1faf321ef497c3bbd5681af97c65aa2",
        "EVAL_SUITE": "libero_spatial",
        "EVAL_SEED": "1000",
        "EVAL_TRIALS": "1",
        "EVAL_TASK_IDS": "all",
    }
    import subprocess as real_subprocess
    call_count = {"run": 0, "popen": 0}
    orig_run = real_subprocess.run
    orig_popen = real_subprocess.Popen

    def counting_run(*a, **kw):
        call_count["run"] += 1
        return orig_run(*a, **kw)

    def counting_popen(*a, **kw):
        call_count["popen"] += 1
        return orig_popen(*a, **kw)

    monkeypatch.setattr(real_subprocess, "run", counting_run)
    monkeypatch.setattr(real_subprocess, "Popen", counting_popen)
    spec, mod = _load_with_env(monkeypatch, env)
    with pytest.raises(SystemExit) as exc_info:
        spec.loader.exec_module(mod)
    assert exc_info.value.code == 1
    assert call_count["run"] == 0, f"subprocess.run called {call_count['run']} times"
    assert call_count["popen"] == 0, f"subprocess.Popen called {call_count['popen']} times"


def test_pipeline_local_checkpoint_does_not_exit(monkeypatch):
    env = {
        "EVAL_CHECKPOINT": "/opt/ml/input/data/model",
        "EVAL_CKPT_REV": "",
        "EVAL_MODEL_SOURCE_URI": "s3://bucket/key/model.tar.gz",
        "EVAL_SUITE": "libero_spatial",
        "EVAL_SEED": "1000",
        "EVAL_TRIALS": "1",
        "EVAL_TASK_IDS": "all",
    }
    spec, mod = _load_with_env(monkeypatch, env)
    spec.loader.exec_module(mod)
    assert mod.PIPELINE_MODE is True
    assert mod.HF_REV == ""
