"""The launcher scripts must produce valid output in --dry-run with NO real AWS
calls. fake_boto3 stubs sts (account resolution) and fails any other client use."""
import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _run(path, argv, capsys):
    spec = importlib.util.spec_from_file_location(f"_dry_{pathlib.Path(path).stem}", REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.argv
    sys.argv = [pathlib.Path(path).name] + argv
    try:
        mod.main()
    finally:
        sys.argv = old
    return capsys.readouterr().out


def test_launch_rl_dry_run(fake_boto3, capsys):
    out = _run("training/scripts/launch_rl.py", ["--dry-run"], capsys)
    assert "create_training_job" in out
    assert fake_boto3 in out  # resolved account appears in the image URI
    assert "physical-ai/isaac-lab:latest" in out
    assert "[dry-run] No AWS calls made." in out


def test_launch_rl_warns_on_unregistered_ur3(fake_boto3, capsys):
    out = _run("training/scripts/launch_rl.py", ["--task", "PickAndPlaceUR3-v0", "--dry-run"], capsys)
    assert "not yet gym-registered" in out


def test_cosmos_launch_dry_run(fake_boto3, capsys):
    out = _run("training/scripts/cosmos_setup.py", ["launch", "--dry-run"], capsys)
    assert "run_instances" in out
    assert "p5.48xlarge" in out
    assert "[dry-run] No AWS calls made." in out


def test_cosmos_generate_dry_run(fake_boto3, capsys):
    out = _run("training/scripts/cosmos_setup.py",
               ["generate", "--instance-id", "i-test", "--input", "/tmp", "--output", "/tmp/o", "--dry-run"],
               capsys)
    assert "/v1/infer" in out
    assert "93-480" in out


def test_cosmos3_generate_dry_run(fake_boto3, capsys):
    out = _run("training/scripts/cosmos3_generate.py",
               ["--mode", "text2video", "--prompt", "a robot arm", "--dry-run"], capsys)
    assert "cosmos_framework.scripts.inference" in out
    assert "Cosmos3-Nano" in out
    assert "physical-ai/cosmos3:latest" in out


def test_edge_scripts_dry_run(fake_boto3, capsys):
    out = _run("edge/create_component.py", ["--model", "/tmp/x.trt", "--dry-run"], capsys)
    assert "com.physicalai.dev.inference" in out
    assert fake_boto3 in out  # ECR image URI uses resolved account
    out2 = _run("edge/deploy_to_fleet.py", ["--dry-run"], capsys)
    assert "physical-ai-dev-robots" in out2
    assert "No AWS writes" in out2


def test_groot_deploy_endpoint_dry_run(fake_boto3, capsys):
    """deploy_endpoint --dry-run prints the create_model/config/endpoint plan and
    makes NO sagemaker calls (only sts for account resolution, stubbed)."""
    out = _run("training/groot/deploy_endpoint.py",
               ["--model-s3", "s3://b/groot-data/ur3/output/job/output/model.tar.gz",
                "--endpoint-name", "groot-ur3", "--dry-run"], capsys)
    assert "create_endpoint_config" in out
    assert "groot-inference" in out          # uses the inference image URI
    assert fake_boto3 in out                 # resolved account in image/role
    assert "ContainerStartupHealthCheckTimeoutInSeconds" in out  # GR00T slow-load handling
    assert "[dry-run] No AWS calls made." in out
