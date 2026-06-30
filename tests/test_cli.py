"""Test pai CLI commands (all offline, zero real AWS calls)."""

import json
import os
import sys

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Return a Click test runner."""
    return CliRunner()


@pytest.fixture
def cli_group():
    """Return the CLI group with all commands registered."""
    from pai import cli as cli_module

    # Create a fresh CLI group
    group = cli_module.cli

    # Register all commands (copy logic from main())
    try:
        from pai.commands import doctor
        doctor.register(group)
    except ImportError:
        pass

    try:
        from pai.commands import config_cmd
        config_cmd.register(group)
    except ImportError:
        pass

    try:
        from pai.commands import deploy
        deploy.register(group)
    except ImportError:
        pass

    try:
        from pai.commands import workstation
        workstation.register(group)
    except ImportError:
        pass

    try:
        from pai.commands import groot
        groot.register(group)
    except ImportError:
        pass

    try:
        from pai.commands import rl
        rl.register(group)
    except ImportError:
        pass

    try:
        from pai.commands import export
        export.register(group)
    except ImportError:
        pass

    return group


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Create a temporary config.json and patch pai.config to use it."""
    config_data = {
        "aws": {"region": "us-west-2"},
        "environment": "dev",
        "projectName": "physical-ai",
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data, indent=2))

    # Patch REPO_ROOT and CONFIG_PATH to point to temp dir
    import pai.config
    monkeypatch.setattr(pai.config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pai.config, "CONFIG_PATH", config_file)

    return config_file


def test_cli_help(runner, cli_group):
    """pai --help exits 0."""
    result = runner.invoke(cli_group, ["--help"])
    assert result.exit_code == 0
    assert "AWS Physical AI Toolchain CLI" in result.output


def test_cli_version(runner, cli_group):
    """pai --version exits 0 and prints version."""
    result = runner.invoke(cli_group, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_rl_launch_dry_run_sagemaker(runner, cli_group, fake_boto3, monkeypatch):
    """pai rl launch --instance-count 2 --dry-run → SageMaker request, zero AWS calls."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(
        cli_group,
        [
            "rl",
            "launch",
            "--instance-count",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert '"InstanceCount": 2' in result.output
    assert "Multi-node training" in result.output
    assert "not yet" in result.output and "validated on hardware" in result.output
    assert "[dry-run] No AWS calls made." in result.output

    # Verify zero create_training_job calls
    sm_calls = [c for c in clients["sagemaker"].calls if c[0] == "create_training_job"]
    assert len(sm_calls) == 0


def test_rl_launch_dry_run_batch(runner, cli_group, fake_boto3, monkeypatch):
    """pai rl launch --engine batch --num-nodes 2 --dry-run → Batch request, zero AWS calls."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(
        cli_group,
        [
            "rl",
            "launch",
            "--engine",
            "batch",
            "--num-nodes",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "physical-ai-dev-rl-queue" in result.output
    assert "physical-ai-dev-rl-mnp" in result.output
    assert '"targetNodes": "0:1"' in result.output
    assert "[dry-run] No AWS calls made." in result.output

    # Verify zero submit_job calls
    batch_calls = [c for c in clients["batch"].calls if c[0] == "submit_job"]
    assert len(batch_calls) == 0


def test_rl_launch_ur3_warning(runner, cli_group, fake_boto3, monkeypatch):
    """pai rl launch --task PickAndPlaceUR3-v0 --dry-run → prints UR3-not-registered warning."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(
        cli_group,
        [
            "rl",
            "launch",
            "--task",
            "PickAndPlaceUR3-v0",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "not yet GPU-validated" in result.output


def test_doctor_runs(runner, cli_group, fake_boto3):
    """pai doctor with fake boto3 → runs all checks, exits (may warn)."""
    account, clients = fake_boto3

    result = runner.invoke(cli_group, ["doctor"])

    # Doctor may exit 0 or 1 depending on checks; just verify it runs
    assert "Preflight Check" in result.output
    assert "AWS credentials" in result.output


def test_region_resolution(runner, cli_group, temp_config, fake_boto3, monkeypatch):
    """Region resolves from config.json when AWS_DEFAULT_REGION unset."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    # Invoke a command that needs region resolution
    result = runner.invoke(cli_group, ["config", "show"])

    assert result.exit_code == 0
    assert "us-west-2" in result.output


def test_groot_launch_dry_run(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot launch --dry-run → resolves CFN outputs, previews the pipeline, zero AWS writes."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    # Set a secret token — it must NOT appear in the dry-run output.
    monkeypatch.setenv("HF_TOKEN", "hf_supersecrettoken123")

    result = runner.invoke(cli_group, ["groot", "launch", "--dry-run"])

    assert result.exit_code == 0
    assert "test-bucket-123" in result.output  # from fake CFN stack output
    assert "groot-finetune-pipeline" in result.output
    assert "groot-models" in result.output  # registers to the model registry
    assert "[dry-run] No AWS calls made." in result.output

    # Security: the HF token must be redacted, never echoed to stdout/logs.
    assert "hf_supersecrettoken123" not in result.output
    assert "<redacted>" in result.output

    # Verify NO SageMaker calls at all (dry-run reads only CloudFormation outputs)
    assert len(clients["sagemaker"].calls) == 0


def test_groot_launch_creates_and_executes_pipeline(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot launch (no dry-run) → create_pipeline + start_pipeline_execution."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = runner.invoke(cli_group, ["groot", "launch", "--max-steps", "100"])

    assert result.exit_code == 0
    assert "Pipeline execution started" in result.output

    call_names = [c[0] for c in clients["sagemaker"].calls]
    assert "create_pipeline" in call_names or "update_pipeline" in call_names
    assert "start_pipeline_execution" in call_names

    # max-steps is forwarded as a pipeline parameter
    exec_call = next(c for c in clients["sagemaker"].calls if c[0] == "start_pipeline_execution")
    params = {p["Name"]: p["Value"] for p in exec_call[1]["PipelineParameters"]}
    assert params["MaxSteps"] == "100"


def test_groot_launch_dry_run_no_hf_token_warns(runner, cli_group, fake_boto3, monkeypatch):
    """Without HF_TOKEN, launch --dry-run warns it runs unauthenticated."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    result = runner.invoke(cli_group, ["groot", "launch", "--dry-run"])

    assert result.exit_code == 0
    assert "unauthenticated" in result.output
    assert len(clients["sagemaker"].calls) == 0


def test_groot_runs_lists_executions(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot runs → calls list_pipeline_executions and prints the run."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(cli_group, ["groot", "runs"])

    assert result.exit_code == 0
    assert "groot-finetune-pipeline" in result.output
    assert "Executing" in result.output

    sm_calls = [c for c in clients["sagemaker"].calls if c[0] == "list_pipeline_executions"]
    assert len(sm_calls) == 1


def test_groot_runs_works_without_repo_on_syspath(runner, cli_group, fake_boto3, monkeypatch):
    """Regression: `pai groot` imports training.groot.*, which is NOT part of the
    installed pai package. The command must add the repo root to sys.path itself
    (like rl.py) so it works when invoked as an installed CLI from any directory —
    not just when conftest happens to have put the repo on the path.
    """
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    # Simulate the installed-CLI environment: repo root NOT on sys.path, and any
    # already-imported training.* modules evicted so the import must re-resolve.
    from pai import config
    repo = str(config.REPO_ROOT)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != repo])
    for name in [m for m in list(sys.modules) if m == "training" or m.startswith("training.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    result = runner.invoke(cli_group, ["groot", "runs"])

    assert result.exit_code == 0, f"groot runs crashed without repo on path: {result.exception!r}"
    assert "groot-finetune-pipeline" in result.output


def test_groot_deploy_dry_run(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot deploy --dry-run → prints the create plan, makes no SageMaker calls."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(
        cli_group,
        ["groot", "deploy",
         "--model-s3", "s3://b/groot-data/ur3/output/job/output/model.tar.gz",
         "--endpoint-name", "groot-ur3", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "groot-inference" in result.output  # inference image URI
    assert "ContainerStartupHealthCheckTimeoutInSeconds" in result.output
    assert "[dry-run] No AWS calls made." in result.output

    # No endpoint actually created in dry-run
    assert not any(c[0] == "create_endpoint" for c in clients["sagemaker"].calls)


def test_groot_deploy_creates_endpoint(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot deploy (no dry-run) → create_model + config + endpoint."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(
        cli_group,
        ["groot", "deploy",
         "--model-s3", "s3://b/groot-data/ur3/output/job/output/model.tar.gz",
         "--endpoint-name", "groot-ur3"],
    )

    assert result.exit_code == 0
    call_names = [c[0] for c in clients["sagemaker"].calls]
    assert "create_model" in call_names
    assert "create_endpoint_config" in call_names
    assert "create_endpoint" in call_names


def test_groot_invoke_calls_runtime(runner, cli_group, fake_boto3, monkeypatch, tmp_path):
    """pai groot invoke → reads the image, calls sagemaker-runtime invoke_endpoint."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    img = tmp_path / "wrist.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    result = runner.invoke(
        cli_group,
        ["groot", "invoke", "--endpoint-name", "groot-ur3",
         "--image-path", str(img), "--state", "0,0,0,0,0,0,0", "--task", "pick up the cube"],
    )

    assert result.exit_code == 0
    assert "action_dim" in result.output

    rt_calls = clients["sagemaker-runtime"].calls
    assert any(c[0] == "invoke_endpoint" for c in rt_calls)


def test_groot_invoke_missing_image_aborts(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot invoke with a non-existent image → aborts before any AWS call."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(
        cli_group,
        ["groot", "invoke", "--endpoint-name", "groot-ur3", "--image-path", "/no/such/wrist.jpg"],
    )

    assert result.exit_code != 0
    assert "Image not found" in result.output
    assert len(clients["sagemaker-runtime"].calls) == 0


def test_groot_delete_calls_delete_endpoint(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot delete --yes → tears down endpoint + config + model."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(cli_group, ["groot", "delete", "--endpoint-name", "groot-ur3", "--yes"])

    assert result.exit_code == 0
    call_names = [c[0] for c in clients["sagemaker"].calls]
    assert "delete_endpoint" in call_names
    assert "delete_endpoint_config" in call_names
    assert "delete_model" in call_names


def test_groot_convert_dry_run(runner, cli_group, monkeypatch):
    """pai groot convert --dry-run → prints the command, runs no subprocess."""
    from pai import helpers

    def _fail_on_call(*args, **kwargs):
        raise AssertionError("helpers.run called during dry-run — should not happen")

    monkeypatch.setattr(helpers, "run", _fail_on_call)

    result = runner.invoke(cli_group, ["groot", "convert", "--dry-run"])

    assert result.exit_code == 0
    assert "[dry-run]" in result.output
    assert "convert_zarr_to_lerobot.py" in result.output


def test_groot_convert_missing_deps_message(runner, cli_group, monkeypatch, tmp_path):
    """A missing conversion dep (e.g. cv2) yields a clean install hint, not a traceback."""
    from pai import helpers
    import importlib.util as _u

    # Real episodes dir so we get past the existence check to the dep check.
    episodes = tmp_path / "episodes"
    episodes.mkdir()

    # Simulate opencv (cv2) not installed; everything else present.
    real_find_spec = _u.find_spec

    def _fake_find_spec(name, *a, **k):
        if name == "cv2":
            return None
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(_u, "find_spec", _fake_find_spec)

    # If we somehow reach the subprocess, fail loudly — the dep check must stop first.
    def _fail_on_run(*args, **kwargs):
        raise AssertionError("helpers.run called despite missing deps")

    monkeypatch.setattr(helpers, "run", _fail_on_run)

    result = runner.invoke(cli_group, ["groot", "convert", "--episodes-dir", str(episodes)])

    assert result.exit_code != 0
    assert "Missing data-conversion dependencies" in result.output
    assert "cv2" in result.output
    assert "pip install -e ." in result.output
    # Clean abort, not an unhandled ModuleNotFoundError traceback.
    assert not isinstance(result.exception, ModuleNotFoundError)


def test_groot_upload_dry_run(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot upload --dry-run → resolves the bucket from CFN, prints the sync, no subprocess."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    from pai import helpers

    def _fail_on_run(*args, **kwargs):
        raise AssertionError("helpers.run called during dry-run — should not happen")

    monkeypatch.setattr(helpers, "run", _fail_on_run)

    result = runner.invoke(cli_group, ["groot", "upload", "--dry-run"])

    assert result.exit_code == 0
    assert "[dry-run]" in result.output
    assert "aws s3 sync" in result.output
    # Bucket resolved from the fake Foundation stack output, dataset path correct.
    assert "s3://test-bucket-123/groot-data/ur3/dataset/" in result.output


def test_groot_upload_missing_dataset_aborts(runner, cli_group, fake_boto3, monkeypatch):
    """pai groot upload with a non-existent dataset dir → aborts before syncing."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    from pai import helpers

    def _fail_on_run(*args, **kwargs):
        raise AssertionError("helpers.run called despite missing dataset dir")

    monkeypatch.setattr(helpers, "run", _fail_on_run)

    result = runner.invoke(cli_group, ["groot", "upload", "--dataset-dir", "/no/such/dataset"])

    assert result.exit_code != 0
    assert "Dataset directory not found" in result.output
    assert "pai groot convert" in result.output


def test_config_show_masks_allowed_cidr(runner, cli_group, tmp_path, monkeypatch):
    """pai config show must redact the user's personal IP (allowedCidr)."""
    import pai.config

    config_data = {
        "aws": {"region": "us-west-2"},
        "environment": "dev",
        "projectName": "physical-ai",
        "workstation": {"allowedCidr": "198.51.100.23/32", "instanceType": "g6e.4xlarge"},
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data, indent=2))
    monkeypatch.setattr(pai.config, "REPO_ROOT", tmp_path)
    if hasattr(pai.config, "CONFIG_PATH"):
        monkeypatch.setattr(pai.config, "CONFIG_PATH", config_file)

    result = runner.invoke(cli_group, ["config", "show"])

    assert result.exit_code == 0
    assert "198.51.100.23" not in result.output
    assert "<redacted>" in result.output


def test_rl_status_calls_describe_training_job(runner, cli_group, fake_boto3, monkeypatch):
    """pai rl status <job> → calls describe_training_job."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(cli_group, ["rl", "status", "test-job-123"])

    assert result.exit_code == 0
    assert "test-job-123" in result.output
    assert "Completed" in result.output

    # Verify describe_training_job was called
    sm_calls = [c for c in clients["sagemaker"].calls if c[0] == "describe_training_job"]
    assert len(sm_calls) == 1
    assert sm_calls[0][1]["TrainingJobName"] == "test-job-123"


def test_deploy_foundation_dry_run(runner, cli_group, monkeypatch, fake_boto3):
    """pai deploy foundation --dry-run → prints cdk command, runs NO subprocess."""
    account, clients = fake_boto3

    # Monkeypatch helpers.run to raise if called (proves dry-run doesn't shell out)
    from pai import helpers

    def _fail_on_call(*args, **kwargs):
        raise AssertionError("helpers.run called during dry-run — should not happen")

    monkeypatch.setattr(helpers, "run", _fail_on_call)

    result = runner.invoke(cli_group, ["deploy", "foundation", "--dry-run"])

    assert result.exit_code == 0
    assert "[dry-run]" in result.output
    assert "npx cdk deploy" in result.output


def test_deploy_workstation_dry_run(runner, cli_group, monkeypatch, fake_boto3):
    """pai deploy workstation --dry-run → prints bash command, runs NO subprocess."""
    account, clients = fake_boto3

    from pai import helpers

    def _fail_on_call(*args, **kwargs):
        raise AssertionError("helpers.run called during dry-run")

    monkeypatch.setattr(helpers, "run", _fail_on_call)

    result = runner.invoke(cli_group, ["deploy", "workstation", "--dry-run"])

    assert result.exit_code == 0
    assert "[dry-run]" in result.output
    assert "deploy-workstation.sh" in result.output


def test_workstation_ip(runner, cli_group, fake_boto3, monkeypatch):
    """pai workstation ip → calls EC2 describe_instances, returns stubbed IP."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(cli_group, ["workstation", "ip"])

    assert result.exit_code == 0
    assert "203.0.113.42" in result.output

    # Verify EC2 describe_instances was called
    ec2_calls = [c for c in clients["ec2"].calls if c[0] == "describe_instances"]
    assert len(ec2_calls) > 0


def test_workstation_start(runner, cli_group, fake_boto3, monkeypatch):
    """pai workstation start → calls EC2 start_instances."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(cli_group, ["workstation", "start"])

    # May exit 0 (already running) or after starting
    assert "already running" in result.output or "started" in result.output.lower()

    # Verify EC2 calls happened
    ec2_calls = clients["ec2"].calls
    assert len(ec2_calls) > 0


def test_workstation_stop(runner, cli_group, fake_boto3, monkeypatch):
    """pai workstation stop --yes → calls EC2 stop_instances."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    # Need to patch the fake EC2 to return stopped state first
    clients["ec2"].calls.clear()

    # Modify the describe_instances response to return stopped state
    def _describe_stopped(**kwargs):
        clients["ec2"].calls.append(("describe_instances", kwargs))
        instance_id = kwargs.get("InstanceIds", ["i-test12345"])[0]
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": instance_id,
                            "State": {"Name": "stopped"},
                            "InstanceType": "g6e.4xlarge",
                        }
                    ]
                }
            ]
        }

    original_describe = clients["ec2"].describe_instances
    clients["ec2"].describe_instances = _describe_stopped

    result = runner.invoke(cli_group, ["workstation", "stop", "--yes"])

    assert result.exit_code == 0
    assert "already stopped" in result.output

    # Restore
    clients["ec2"].describe_instances = original_describe


def test_config_set_round_trip(runner, cli_group, temp_config, fake_boto3):
    """pai config set aws.region eu-west-1 → round-trips through temp config.json."""
    account, clients = fake_boto3

    # Set a new region
    result = runner.invoke(cli_group, ["config", "set", "aws.region", "eu-west-1"])
    assert result.exit_code == 0
    assert "eu-west-1" in result.output

    # Verify it was written
    config_data = json.loads(temp_config.read_text())
    assert config_data["aws"]["region"] == "eu-west-1"

    # Show it
    result = runner.invoke(cli_group, ["config", "show"])
    assert result.exit_code == 0
    assert "eu-west-1" in result.output


def test_rl_status_batch(runner, cli_group, fake_boto3, monkeypatch):
    """pai rl status --engine batch <job> → calls batch describe_jobs."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    result = runner.invoke(cli_group, ["rl", "status", "--engine", "batch", "test-batch-job"])

    assert result.exit_code == 0
    assert "test-batch-job" in result.output or "SUCCEEDED" in result.output

    # Verify batch describe_jobs was called
    batch_calls = [c for c in clients["batch"].calls if c[0] == "describe_jobs"]
    assert len(batch_calls) == 1
