"""C6: OpenVLA must not attribute a stale or unrelated evaluation log to the current run.

The wrapper selected the newest `EVAL-*` file from a shared directory, matching no run
identifier, suite, or checkpoint. An older log with a later mtime could win even when a new
log existed, and its internally consistent numbers would satisfy Validate because the wrapper
supplied the current run's labels around them.

Pinned upstream (moojink/openvla-oft@e4287e94) exposes `--local_log_dir` and `--run_id_note`
and builds the log name as `EVAL-<suite>-<family>-<timestamp>[--<run_id_note>].txt`, so the
log can be identified rather than guessed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _REPO_ROOT / "entrypoints/eval/libero/openvla/eval_entry.py"


def _module(monkeypatch, **environment):
    for name in ("digest", "validator"):
        sys.modules.setdefault(name, MagicMock())
    defaults = {
        "EVAL_CHECKPOINT": "/opt/ml/input/data/model",
        "EVAL_SUITE": "libero_spatial",
        "EVAL_SEED": "1000",
        "EVAL_TRIALS": "3",
        "EVAL_MODEL_SOURCE_URI": "s3://bucket/key/model.tar.gz",
    }
    defaults.update(environment)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("openvla_c6_under_test", _ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_identity_prefers_the_sagemaker_job_name(monkeypatch):
    module = _module(monkeypatch, TRAINING_JOB_NAME="simeval-openvla-20260912-1")
    assert module._run_identity() == "simeval-openvla-20260912-1"


def test_the_identity_falls_back_to_the_training_env_job_name(monkeypatch):
    monkeypatch.delenv("TRAINING_JOB_NAME", raising=False)
    module = _module(monkeypatch,
                     SM_TRAINING_ENV=json.dumps({"job_name": "from-training-env"}))
    assert module._run_identity() == "from-training-env"


def test_a_malformed_training_env_does_not_crash_identity(monkeypatch):
    """A broken env var must not take down the run; it falls back and says so."""
    monkeypatch.delenv("TRAINING_JOB_NAME", raising=False)
    module = _module(monkeypatch, SM_TRAINING_ENV="{not json")
    identity = module._run_identity()
    assert identity.startswith("local-")


def test_the_identity_is_filename_safe(monkeypatch):
    """Upstream embeds this in the log filename, so unsafe characters must be replaced."""
    module = _module(monkeypatch, TRAINING_JOB_NAME="job/with spaces:and*chars")
    identity = module._run_identity()
    assert "/" not in identity and " " not in identity and "*" not in identity
    assert identity == "job-with-spaces-and-chars"


def test_the_identity_is_bounded(monkeypatch):
    module = _module(monkeypatch, TRAINING_JOB_NAME="j" * 500)
    assert len(module._run_identity()) <= 96


def test_the_command_requests_a_per_run_directory_and_identity():
    source = _ENTRY.read_text()
    assert '"--local_log_dir", logs_dir,' in source
    assert '"--run_id_note", run_identity,' in source
    # And the directory must be per-run, not the shared upstream default.
    assert 'os.path.join(OFT_DIR, "experiments", "logs", run_identity)' in source


def test_selection_by_modification_time_is_gone():
    """The whole defect was newest-wins. It must not survive anywhere in selection."""
    source = _ENTRY.read_text()
    assert "os.path.getmtime" not in source, (
        "log selection must not depend on modification time")


def test_an_existing_directory_is_refused():
    """A directory that already exists cannot be proven free of foreign logs."""
    source = _ENTRY.read_text()
    assert "already exists; refusing to reuse" in source
    assert source.index("if os.path.exists(logs_dir):") < source.index("os.makedirs(logs_dir)")


def _log_dir(tmp_path, *names):
    directory = tmp_path / "logs"
    directory.mkdir()
    for name in names:
        (directory / name).write_text("contents")
    return str(directory)


def test_the_run_own_log_is_selected(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    identity = "simeval-job-1"
    name = f"EVAL-libero_spatial-openvla-2026_09_12--{identity}.txt"
    chosen = module._select_run_log(_log_dir(tmp_path, name), identity)
    assert pathlib.Path(chosen).name == name


def test_a_stale_log_with_a_later_timestamp_cannot_win(monkeypatch, tmp_path):
    """The exact defect: newest-wins selected a log from an unrelated run.

    The stale file is given a LATER modification time than this run's, so a
    mtime-based selection would choose it.
    """
    module = _module(monkeypatch)
    identity = "simeval-job-2"
    mine = f"EVAL-libero_spatial-openvla-2026_09_12--{identity}.txt"
    stale = "EVAL-libero_spatial-openvla-2020_01_01--other-run.txt"
    directory = _log_dir(tmp_path, mine, stale)
    os.utime(os.path.join(directory, stale), (10 ** 10, 10 ** 10))
    assert os.path.getmtime(os.path.join(directory, stale)) > os.path.getmtime(
        os.path.join(directory, mine)), "the stale log must look newer for this to bite"
    with pytest.raises(ValueError, match="Exactly one is required"):
        module._select_run_log(directory, identity)


def test_a_log_from_another_run_alone_is_refused(monkeypatch, tmp_path):
    """Not merely ambiguous -- a single log that is not ours must not be parsed."""
    module = _module(monkeypatch)
    directory = _log_dir(
        tmp_path, "EVAL-libero_spatial-openvla-2020_01_01--someone-else.txt")
    with pytest.raises(ValueError, match="does not carry this run's identity"):
        module._select_run_log(directory, "simeval-job-3")


def test_no_log_at_all_is_refused(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    with pytest.raises(ValueError, match="no EVAL-.*log produced"):
        module._select_run_log(_log_dir(tmp_path), "simeval-job-4")


def test_unrelated_files_are_ignored_but_do_not_satisfy_selection(monkeypatch, tmp_path):
    module = _module(monkeypatch)
    directory = _log_dir(tmp_path, "notes.txt", "EVAL-partial.log")
    with pytest.raises(ValueError, match="no EVAL-.*log produced"):
        module._select_run_log(directory, "simeval-job-5")


def test_more_than_one_log_is_refused():
    source = _ENTRY.read_text()
    assert "Exactly one is required" in source


def test_a_log_without_this_runs_identity_is_refused():
    source = _ENTRY.read_text()
    assert "does not carry this run's identity" in source
