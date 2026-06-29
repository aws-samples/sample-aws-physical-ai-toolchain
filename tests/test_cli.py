"""Test pai CLI commands (all offline, zero real AWS calls)."""

import json
import os

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
    assert "UNVALIDATED on hardware" in result.output
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
    assert '"numNodes": 2' in result.output
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
    assert "not yet wired into the isaac-lab" in result.output


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


def test_groot_launch_dry_run(runner, cli_group, fake_boto3, monkeypatch, tmp_path):
    """pai groot launch --dry-run → resolves CFN outputs, prints request, zero AWS calls."""
    account, clients = fake_boto3
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    # Create fake dataset dir
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dummy.txt").write_text("test")

    # Set a secret token — it must NOT appear in the dry-run output.
    monkeypatch.setenv("HF_TOKEN", "hf_supersecrettoken123")

    result = runner.invoke(
        cli_group,
        [
            "groot",
            "launch",
            "--dataset-dir",
            str(dataset_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "create_training_job" in result.output
    assert "[dry-run] No AWS calls made." in result.output
    assert "test-bucket-123" in result.output  # from fake CFN stack output

    # Security: the HF token must be redacted, never echoed to stdout/logs.
    assert "hf_supersecrettoken123" not in result.output
    assert "<redacted>" in result.output

    # Verify zero create_training_job calls
    sm_calls = [c for c in clients["sagemaker"].calls if c[0] == "create_training_job"]
    assert len(sm_calls) == 0


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
