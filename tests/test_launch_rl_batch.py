"""Test launch_rl_batch.py dry-run (no AWS calls)."""
import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _run(path, argv, capsys, fake_boto3_fixture):
    """Import and run a script module, capturing stdout."""
    spec = importlib.util.spec_from_file_location(
        f"_dry_{pathlib.Path(path).stem}", REPO / path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.argv
    sys.argv = [pathlib.Path(path).name] + argv
    try:
        mod.main()
    finally:
        sys.argv = old
    return capsys.readouterr().out


def test_launch_rl_batch_dry_run(fake_boto3, capsys):
    """launch_rl_batch.py --dry-run prints submit_job request, no AWS calls."""
    _, _ = fake_boto3
    out = _run("training/scripts/launch_rl_batch.py", ["--dry-run"], capsys, fake_boto3)
    assert "submit_job" in out
    assert "physical-ai-dev-rl-queue" in out
    assert "physical-ai-dev-rl-mnp" in out
    assert "[dry-run] No AWS calls made." in out
    # Verify nodeOverrides structure
    assert "nodePropertyOverrides" in out
    assert "numNodes" in out


def test_launch_rl_batch_multinode_disclaimer(fake_boto3, capsys):
    """Multi-node jobs print the UNVALIDATED disclaimer."""
    _, _ = fake_boto3
    out = _run(
        "training/scripts/launch_rl_batch.py",
        ["--num-nodes", "4", "--dry-run"],
        capsys,
        fake_boto3,
    )
    assert "[dry-run] No AWS calls made." in out
    assert '"numNodes": 4' in out
    assert "Multi-node training" in out
    assert "not yet" in out and "validated on hardware" in out


def test_launch_rl_batch_warns_on_ur3(fake_boto3, capsys):
    """UR3 task triggers not-yet-container-wired warning."""
    _, _ = fake_boto3
    out = _run(
        "training/scripts/launch_rl_batch.py",
        ["--task", "PickAndPlaceUR3-v0", "--dry-run"],
        capsys,
        fake_boto3,
    )
    assert "not yet GPU-validated" in out


def test_launch_rl_batch_node_overrides(fake_boto3, capsys):
    """Verify hyperparameters are passed via nodePropertyOverrides."""
    _, _ = fake_boto3
    out = _run(
        "training/scripts/launch_rl_batch.py",
        [
            "--task",
            "Isaac-Custom-Task",
            "--num-envs",
            "2048",
            "--max-iterations",
            "200",
            "--framework",
            "skrl",
            "--num-nodes",
            "3",
            "--dry-run",
        ],
        capsys,
        fake_boto3,
    )
    assert "Isaac-Custom-Task" in out
    assert '"value": "2048"' in out
    assert '"value": "200"' in out
    assert '"value": "skrl"' in out
    assert '"numNodes": 3' in out
