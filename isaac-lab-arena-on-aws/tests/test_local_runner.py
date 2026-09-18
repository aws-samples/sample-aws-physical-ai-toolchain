"""Local control-plane failures must not become successful workload reports."""
import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

COMPONENT = Path(__file__).resolve().parents[1]
SCRIPT = COMPONENT / "scripts/local/run_local_pipeline.py"
spec = importlib.util.spec_from_file_location("arena_local_runner_test", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def execution(status="Succeeded", gate=True, omit=None, failed_step=None):
    rows = [
        {"StepName": name, "StepStatus": "Failed" if name == failed_step else "Succeeded"}
        for name in ("FineTune", "SimEval", "Validate", "SuccessGate") if name != omit
    ]
    for row in rows:
        if row["StepName"] == "SuccessGate":
            row["Metadata"] = {"Condition": {"Outcome": gate}}
    return SimpleNamespace(
        describe=lambda: {"PipelineExecutionStatus": status, "FailureReason": "test failure"},
        list_steps=lambda: {"PipelineExecutionSteps": rows},
    )


def test_success_checker_accepts_complete_run():
    description, rows = runner.assert_success(execution())
    assert description["PipelineExecutionStatus"] == "Succeeded"
    assert len(rows) == 4


@pytest.mark.parametrize("kwargs", [
    {"status": "Failed"},  # All four steps may say Succeeded while the execution failed.
    {"status": "Executing"},
    {"gate": False},
    {"omit": "Validate"},
    {"omit": "SuccessGate"},
    {"failed_step": "FineTune"},
])
def test_success_checker_rejects_bad_run(kwargs):
    with pytest.raises(RuntimeError):
        runner.assert_success(execution(**kwargs))


@pytest.mark.parametrize("as_dict", [False, True])
def test_processing_receipt_handles_both_sdk_shapes(as_dict):
    output = {"OutputName": "validated", "S3Output": {"S3Uri": "s3://scratch/validated"}}
    outputs = {"validated": output} if as_dict else [output]
    description = {"ProcessingOutputConfig": {"Outputs": outputs}}
    assert runner.validated_output_uri(description) == "s3://scratch/validated/validated_metrics.json"


@dataclasses.dataclass
class Config:
    bucket: str = "managed-models"
    handoff_bucket: str = "managed-handoff"
    trust_bucket: str = "managed-trust"
    pipeline_name: str = "managed"
    prefix: str = "pipeline"


@pytest.mark.parametrize("bucket", ["managed-models", "managed-handoff", "managed-trust"])
def test_local_storage_rejects_every_managed_bucket(bucket):
    with pytest.raises(RuntimeError, match="differ from all managed"):
        runner.development_config(Config(), bucket, "test")


def test_local_storage_redirects_all_three_references_without_mutating_managed():
    original = Config()
    local = runner.development_config(original, "dedicated-dev-bucket", "test")
    assert local.bucket == local.handoff_bucket == local.trust_bucket == "dedicated-dev-bucket"
    assert original == Config()


def test_complete_recipe_keeps_typed_parameters_and_resolves_deployment():
    from vla_pipeline.pipeline import build_parameters

    recipe = SCRIPT.with_name("arena-gr1-smoke.json")
    params = runner.load_parameters(
        recipe, build_parameters(), "111122223333.dkr.ecr.us-west-2.amazonaws.com")
    assert len(params) == 30
    assert params["TrainSteps"] == 200
    assert params["EvalTrials"] == 3
    assert params["SuccessThreshold"] == 0.0
    assert params["ExpectedPolicyConfig"].endswith("_closedloop_config.yaml")
    assert params["EvalImageUri"].startswith("111122223333.dkr.ecr.us-west-2.amazonaws.com/")
    assert "@sha256:" in params["EvalImageUri"]


def test_duplicate_recipe_parameter_is_rejected(tmp_path):
    recipe = tmp_path / "parameters.json"
    recipe.write_text(json.dumps({"PipelineParameters": [
        {"Name": "TrainSteps", "Value": "200"}, {"Name": "TrainSteps", "Value": "1"}]}))
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.load_parameters(recipe, {}, "registry")


def test_reusing_run_directory_preserves_historical_status(tmp_path):
    status = tmp_path / "status.json"
    status.write_text('{"status":"Failed"}')
    process = subprocess.run([
        sys.executable, str(SCRIPT), "--run-dir", str(tmp_path), "--run-id", "retry",
        "--expected-commit", "unused", "--development-bucket", "scratch",
        "--expected-role", "instance-role",
    ], capture_output=True, text=True)
    assert process.returncode == 2
    assert "not empty" in process.stderr
    assert json.loads(status.read_text()) == {"status": "Failed"}


def test_actual_local_negative_control_rejects_failure(monkeypatch):
    # No managed API or containers: execute real local Condition/Fail steps.
    import boto3
    from sagemaker.workflow.pipeline_context import LocalPipelineSession

    monkeypatch.setenv("SAGEMAKER_TELEMETRY_OPT_OUT", "1")
    boto = boto3.Session(
        aws_access_key_id="test", aws_secret_access_key="test", region_name="us-east-1")
    session = LocalPipelineSession(boto_session=boto, default_bucket="local-control-test")
    result = runner.negative_control(session, "arn:aws:iam::111122223333:role/test", "regression")
    assert result["description"]["PipelineExecutionStatus"] == "Failed"
    assert result["required_steps_present"] and result["checker_rejected"]
    assert len(result["steps"]["PipelineExecutionSteps"]) == 5


@pytest.mark.parametrize("probe_exit", [0, 1])
def test_compose_adapter_requires_a_real_successful_command(tmp_path, monkeypatch, probe_exit):
    from sagemaker.local.image import _SageMakerContainer

    original = _SageMakerContainer._get_compose_cmd_prefix
    monkeypatch.setattr(_SageMakerContainer, "_get_compose_cmd_prefix", staticmethod(original))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "5.5.1\n")
    commands = []

    def popen(command, stdout, stderr):
        commands.append(command)
        if "up" in command:
            stdout.write("local Compose probe passed\n")
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.return_value = probe_exit if "up" in command else 0
        return process

    def inventory(command, **kwargs):
        assert command[:3] == ["docker", "ps", "-a"]
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.subprocess, "run", inventory)
    if probe_exit:
        with pytest.raises(RuntimeError, match="Compose cpu-check exited 1"):
            runner.check_compose("cpu-test-image", tmp_path, "test")
        assert _SageMakerContainer._get_compose_cmd_prefix is original
    else:
        assert runner.check_compose("cpu-test-image", tmp_path, "test")["version"] == "5.5.1"
        assert _SageMakerContainer._get_compose_cmd_prefix() == ["docker", "compose"]
    assert commands[-1][-1] == "down"
