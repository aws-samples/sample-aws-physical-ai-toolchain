"""Tests for the GR00T fine-tuning entrypoint + customer-data ingestion.

Laptop-validatable contract (no GPU / SDK / AWS):
  - the entrypoint FAILS LOUD (non-zero exit + failure file) when the SDK is
    absent, instead of the old silent exit-0 stub;
  - the embodiment config it writes is valid Python with the proven UR3 shape
    (EEF action space, arm+gripper keys, wrist cam);
  - eval is honest — it writes dataset baselines clearly labelled as NOT a model
    eval, and never fabricates a model number;
  - the ingestion --dry-run makes no AWS calls and converts nothing.

The entrypoint module imports on a bare box because gr00t/torch imports live
inside functions.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO / "containers" / "groot-training" / "train_entrypoint.py"
INGEST = REPO / "training" / "groot" / "ingest_customer_data.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def entry():
    return _load(ENTRYPOINT, "_groot_entrypoint")


@pytest.fixture
def ingest():
    return _load(INGEST, "_groot_ingest")


# --------------------------------------------------------------------------- #
# Fail-loud
# --------------------------------------------------------------------------- #

def test_fails_loud_when_sdk_missing(entry, tmp_path, monkeypatch):
    """No SDK present → non-zero exit + failure file, never a green empty model.

    require_sdk() runs first in __main__, before torch import / torchrun relaunch,
    so the failure reason is always written."""
    failure = tmp_path / "failure"
    monkeypatch.setattr(entry, "SDK_DIR", str(tmp_path / "no-sdk-here"))
    monkeypatch.setattr(entry, "FAILURE_FILE", str(failure))
    monkeypatch.setattr(entry, "MODEL_DIR", str(tmp_path / "model"))

    with pytest.raises(SystemExit) as exc:
        entry.require_sdk()
    assert exc.value.code == 1
    assert failure.exists()
    assert "did NOT train" in failure.read_text()
    # No model artifact pretending success.
    assert not (tmp_path / "model" / "training_metadata.json").exists()


def test_no_silent_stub_behavior(entry):
    """Guard against the old bugs: no isaac-groot pip pkg, no silent fallback marker,
    no fabricated model-eval number, no longer the dead launch_finetune CLI path."""
    src = ENTRYPOINT.read_text()
    # Strip comment lines — the module legitimately MENTIONS the old API in prose
    # to explain why it's avoided; we only forbid actual usage.
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "pip install isaac-groot" not in code
    assert "TRAINING_FALLBACK_USED" not in code
    # No invocation of the dead n1.7 launch_finetune.py CLI path.
    assert "launch_finetune.py" not in code
    # The proven N1.6 core must be present.
    assert "from gr00t.experiment.experiment import run" in src
    assert "get_default_config" in code
    assert "gradient_checkpointing" in code


# --------------------------------------------------------------------------- #
# Embodiment config writer (proven UR3 shape)
# --------------------------------------------------------------------------- #

def test_embodiment_config_is_valid_and_correct_shape(entry, tmp_path):
    """write_embodiment_config emits importable Python with the proven UR3 shape."""
    path = entry.write_embodiment_config(str(tmp_path))
    code = pathlib.Path(path).read_text()
    # Compiles.
    compile(code, path, "exec")
    # Proven imports + registration.
    assert "from gr00t.configs.data.embodiment_configs import register_modality_config" in code
    assert "register_modality_config(ur3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)" in code
    # EEF action space (the proven choice — NOT NON_EEF).
    assert "ActionType.EEF" in code
    assert "ActionType.NON_EEF" not in code
    # Action format MUST be XYZ_ROTVEC (3 translation + 3 rotation-vector = our 6
    # speedl values). ActionFormat.DEFAULT makes GR00T reshape to a 4x4 (16-value)
    # matrix and crash on our 6-value actions — a real g5 failure we hit and fixed.
    assert "ActionFormat.XYZ_ROTVEC" in code
    assert "ActionFormat.DEFAULT" not in code
    # UR3 keys.
    assert '"arm", "gripper"' in code
    assert '"wrist"' in code
    assert "annotation.human.action.task_description" in code


def test_reference_modality_config_matches_entrypoint(entry, tmp_path):
    """The standalone reference config and the inline one must agree on action type."""
    ref = (REPO / "containers" / "groot-training" / "ur3_modality_config.py").read_text()
    inline = pathlib.Path(entry.write_embodiment_config(str(tmp_path))).read_text()
    for token in ("ActionType.EEF", '"arm", "gripper"', '"wrist"',
                  "annotation.human.action.task_description"):
        assert token in ref, f"{token} missing from reference config"
        assert token in inline, f"{token} missing from inline config"


# --------------------------------------------------------------------------- #
# Honest eval
# --------------------------------------------------------------------------- #

def test_baselines_only_report_is_labelled_not_a_model_eval(entry, tmp_path):
    """The eval report must clearly say it is NOT a trained-model result."""
    # A tiny LeRobot-ish parquet with an array-valued 'action' column.
    pd = pytest.importorskip("pandas")
    import numpy as np
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    df = pd.DataFrame({"action": [np.zeros(7).tolist() for _ in range(20)]})
    df.to_parquet(data_dir / "episode_000000.parquet")

    entry.write_baselines_only(str(tmp_path), tmp_path, reason="unit test")
    report = json.loads((tmp_path / "eval_report.json").read_text())
    assert report["status"] == "dataset_baselines_only"
    assert "do NOT reflect the trained model" in report["warning"]
    assert "baseline_mse_overall" in report
    # Crucially: no key implying a real model score.
    assert "model_mse_overall" not in report


# --------------------------------------------------------------------------- #
# Ingestion dry-run
# --------------------------------------------------------------------------- #

def test_ingest_dry_run_makes_no_aws_calls(ingest, monkeypatch, capsys):
    monkeypatch.setattr(ingest.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no calls in dry-run")))
    sys.argv = ["ingest", "--episodes-dir", "./my_eps", "--prefix", "groot-data/mine", "--dry-run"]
    ingest.main()
    out = capsys.readouterr().out
    assert "[dry-run] No AWS calls made" in out
    assert "convert" in out and "upload" in out
    assert '"step": "train"' not in out


def test_ingest_dry_run_includes_train_step_when_requested(ingest, monkeypatch, capsys):
    monkeypatch.setattr(ingest.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no calls in dry-run")))
    sys.argv = ["ingest", "--episodes-dir", "./e", "--prefix", "groot-data/mine",
                "--train", "--max-steps", "100", "--dry-run"]
    ingest.main()
    out = capsys.readouterr().out
    assert '"step": "train"' in out
    assert "launch_training.py" in out
