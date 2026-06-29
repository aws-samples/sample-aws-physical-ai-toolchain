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
    # Verify nodeOverrides structure. No top-level numNodes override — the job
    # definition's closed range fixes the node count; we only target the range.
    assert "nodePropertyOverrides" in out
    assert '"targetNodes": "0:1"' in out


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
    assert '"targetNodes": "0:3"' in out
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
    assert '"targetNodes": "0:2"' in out


def test_launch_rl_batch_rejects_mismatched_num_nodes(fake_boto3, capsys):
    """A real submit with --num-nodes != job-def node count errors, no submit_job."""
    _, clients = fake_boto3
    # Fake job def is 2 nodes; ask for 4 (no --dry-run → guard runs).
    with pytest.raises(SystemExit) as exc:
        _run(
            "training/scripts/launch_rl_batch.py",
            ["--num-nodes", "4"],
            capsys,
            fake_boto3,
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "does not match job definition" in err
    # Guard must fire before submit_job is ever called.
    submits = [c for c in clients["batch"].calls if c[0] == "submit_job"]
    assert len(submits) == 0


def test_launch_rl_batch_matching_num_nodes_submits(fake_boto3, capsys):
    """A real submit with --num-nodes matching the job def calls submit_job."""
    _, clients = fake_boto3
    _run(
        "training/scripts/launch_rl_batch.py",
        ["--num-nodes", "2"],
        capsys,
        fake_boto3,
    )
    submits = [c for c in clients["batch"].calls if c[0] == "submit_job"]
    assert len(submits) == 1
