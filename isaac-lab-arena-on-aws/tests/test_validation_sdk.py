import json
import pathlib
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vla_pipeline.runner import stage_validate_code
from vla_pipeline.validation_sdk import _locked_wheels


def test_staged_sdk_runs_without_an_installer(tmp_path):
    staged = stage_validate_code()
    probe = """
import json, os, runpy, subprocess, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Validate tried to launch an external installer")
subprocess.Popen = forbidden
entry = runpy.run_path(sys.argv[1])
sdk = entry["_ensure_conditional_write_support"]()
import botocore
print(json.dumps({
    "boto3": sdk.__version__, "botocore": botocore.__version__,
    "module": sdk.__file__, "bundle": os.environ["VLA_VALIDATION_SDK_SHA256"],
}))
"""
    result = subprocess.run(
        [sys.executable, "-", staged], input=probe, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    runtime = json.loads(result.stdout)
    assert runtime["boto3"] == runtime["botocore"] == "1.42.97"
    assert "vla-validation-sdk-" in runtime["module"]
    assert len(runtime["bundle"]) == 64


@pytest.mark.parametrize("missing", ["PutObject", "CompleteMultipartUpload"])
def test_old_sdk_fails_without_attempting_install(monkeypatch, missing, capsys):
    import botocore.session
    entry = runpy.run_path(str(
        pathlib.Path(__file__).resolve().parents[1] / "entrypoints/validate_entry.py"))
    model = SimpleNamespace(operation_model=lambda operation: SimpleNamespace(
        input_shape=SimpleNamespace(members={} if operation == missing else {"IfNoneMatch": None})))
    monkeypatch.setattr(botocore.session, "get_session", lambda: SimpleNamespace(
        get_service_model=lambda service: model))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Runtime installer called"))
    with pytest.raises(SystemExit):
        entry["_ensure_conditional_write_support"]()
    assert f"{missing}.IfNoneMatch" in capsys.readouterr().err


def test_tampered_cached_dependency_is_rejected(tmp_path):
    (tmp_path / "package.whl").write_bytes(b"tampered dependency")
    with pytest.raises(ValueError, match="hash-locked closure"):
        _locked_wheels(tmp_path, "package==1 --hash=sha256:" + "a" * 64 + "\n")
