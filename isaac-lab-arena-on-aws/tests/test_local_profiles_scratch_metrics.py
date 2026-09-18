"""Local profiles must preserve model contracts, mounts and measurement meaning."""
import importlib.util
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/local"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiles = load("local_profiles")
scratch = load("local_scratch")
metrics = load("local_metrics")


@pytest.mark.parametrize("profile,family,train_suite,volumes", [
    ("openvla-libero", "openvla", "libero_spatial", (300, 200)),
    ("molmoact2-libero", "molmoact2", "unified", (250, 250)),
    ("gr00t-libero", "gr00t", "libero_spatial", (150, 150)),
])
def test_libero_recipes_preserve_managed_model_contract(profile, family, train_suite, volumes):
    from vla_pipeline.pipeline import build_parameters

    declarations = build_parameters()
    parameters = profiles.libero_parameters(
        profile, declarations, "111122223333.dkr.ecr.us-east-1.amazonaws.com",
        lambda image: image.rsplit(":", 1)[0] + "@sha256:" + "1" * 64)
    assert set(parameters) == {p.name for p in declarations.values()}
    assert parameters["ModelFamily"] == family
    assert parameters["TrainSuite"] == train_suite
    assert parameters["Suite"] == "libero_spatial"
    assert parameters["Gr00tVersion"] == "n17"
    assert parameters["TrainSteps"] == 200 and parameters["EvalTrials"] == 3
    assert parameters["EvalSeed"] == 1000
    assert profiles.expected_episodes(parameters) == 30
    assert (parameters["VolumeSizeInGB"], parameters["EvalVolumeSizeInGB"]) == volumes
    assert parameters["TrainInstanceType"] == parameters["EvalInstanceType"] == "local_gpu"
    assert parameters["UseGrootServer"] == "false"
    assert parameters["EvalSimConfig"] == "{}"


@pytest.mark.parametrize("profile,family", [
    ("openvla-libero", "openvla"),
    ("molmoact2-libero", "molmoact2"),
    ("gr00t-libero", "gr00t"),
])
def test_fresh_account_image_overrides_do_not_resolve_missing_historic_tags(profile, family):
    from vla_pipeline.pipeline import build_parameters

    registry = "111122223333.dkr.ecr.us-east-1.amazonaws.com"
    new_image = f"{registry}/vla/{family}@sha256:" + "2" * 64
    lookups = []

    def fresh_account_resolver(image):
        lookups.append(image)
        if image != new_image:
            raise RuntimeError(f"Image does not exist in this fresh account: {image}")
        return image

    parameters = profiles.libero_parameters(
        profile, build_parameters(), registry, fresh_account_resolver,
        train_image=new_image, eval_image=new_image)
    assert parameters["TrainImageUri"] == parameters["EvalImageUri"] == new_image
    assert lookups == [new_image]


def test_arena_count_and_libero_task_subset_are_not_interchangeable():
    assert profiles.expected_episodes({"Suite": "arena_gr1_fridge", "EvalTrials": 3}) == 3
    with pytest.raises(RuntimeError, match="complete spatial suite"):
        profiles.expected_episodes({
            "Suite": "libero_spatial", "EvalTrials": 3, "EvalTaskIds": "[0]"})


def test_empty_declared_defaults_are_not_sent_as_api_overrides(monkeypatch):
    import boto3
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionEquals
    from sagemaker.workflow.parameters import ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import LocalPipelineSession

    monkeypatch.setenv("SAGEMAKER_TELEMETRY_OPT_OUT", "1")
    parameter = ParameterString(name="ArenaOnlyField", default_value="")
    session = LocalPipelineSession(boto_session=boto3.Session(
        aws_access_key_id="test", aws_secret_access_key="test", region_name="us-east-1"))
    gate = ConditionStep(name="UsesDeclaredDefault", conditions=[
        ConditionEquals(left=parameter, right="")], if_steps=[], else_steps=[])
    pipeline = Pipeline(name="empty-default-regression", parameters=[parameter],
                        steps=[gate], sagemaker_session=session)
    pipeline.create(role_arn="arn:aws:iam::111122223333:role/test")
    complete = {"ArenaOnlyField": ""}
    overrides = profiles.execution_parameters(complete, {"field": parameter})
    result = pipeline.start(parameters=overrides)
    assert result.describe()["PipelineExecutionStatus"] == "Succeeded"
    assert result.list_steps()["PipelineExecutionSteps"][0]["Metadata"]["Condition"]["Outcome"] is True
    assert complete == {"ArenaOnlyField": ""}
    with pytest.raises(RuntimeError, match="not the declared default"):
        profiles.execution_parameters({"ArenaOnlyField": ""}, {
            "field": ParameterString(name="ArenaOnlyField", default_value="required")})


def test_legacy_lineage_cannot_be_promoted_to_independent_verification():
    params = {"ModelFamily": "molmoact2", "Suite": "libero_spatial", "EvalTaskIds": "all"}
    fields = {"train_steps": "matched", "dataset_source": "recorded_only",
              "dataset_revision": "recorded_only", "train_suite": "unavailable"}
    receipt = {"model_family": "molmoact2", "suite": "libero_spatial",
               "training_contract": {"fields": [
                   {"field": field, "status": status} for field, status in fields.items()]}}
    profiles.check_training_contract(receipt, params)
    receipt["training_contract"]["verified"] = ["dataset_source"]
    with pytest.raises(RuntimeError, match="stronger lineage"):
        profiles.check_training_contract(receipt, params)


def test_absent_scratch_mount_does_not_silently_use_root(tmp_path):
    if tmp_path.stat().st_dev != Path("/").stat().st_dev:
        pytest.skip("This test needs a temporary directory on the root filesystem")
    with pytest.raises(RuntimeError, match="separate from root"):
        scratch.prepare(tmp_path / "unmounted", tmp_path / "run", "openvla-libero")
    assert not (tmp_path / "unmounted").exists()


@pytest.mark.parametrize("command", ["train", "process"])
def test_real_sdk_compose_preserves_generated_and_explicit_scratch_mounts(tmp_path, monkeypatch, command):
    import boto3
    from sagemaker.local.image import _SageMakerContainer, _Volume
    from sagemaker.workflow.pipeline_context import LocalPipelineSession

    original = _SageMakerContainer._generate_compose_file
    monkeypatch.setattr(_SageMakerContainer, "_generate_compose_file", original)
    monkeypatch.setattr(scratch, "free_gib", lambda path: 500)
    monkeypatch.setattr(scratch.shutil, "disk_usage", lambda path: SimpleNamespace(
        total=1000 * 1024**3, used=500 * 1024**3, free=500 * 1024**3))
    session = LocalPipelineSession(
        boto_session=boto3.Session(
            aws_access_key_id="test", aws_secret_access_key="test", region_name="us-east-1"),
        default_bucket="local-test")
    session.config = {"local": {"container_config": {"shm_size": "8g", "volumes": ["wrong:mount"]}}}
    layout = {"root": str(tmp_path), "work": str(tmp_path / "scratch"),
              "cache": str(tmp_path / "cache")}
    job = tmp_path / "job"
    job.mkdir()
    container = SimpleNamespace(
        container_root=str(job), image="local-test-image", hosts=["algo-1"],
        sagemaker_session=session, is_studio=False, instance_type="local_gpu",
        container_entrypoint=None, container_arguments=None,
        _build_optml_volumes=lambda host, dirs: [_Volume(str(tmp_path / "model"), "/opt/ml/model")],
    )
    container._create_docker_host = MethodType(_SageMakerContainer._create_docker_host, container)
    scratch.install_mounts(layout, tmp_path)
    result = _SageMakerContainer._generate_compose_file(
        container, command, [_Volume(str(tmp_path / "code"), "/opt/ml/code")], {})
    mounts = result["services"]["algo-1"]["volumes"]
    assert str(tmp_path / "model") + ":/opt/ml/model" in mounts
    assert str(tmp_path / "code") + ":/opt/ml/code" in mounts
    assert any(mount.endswith(":/tmp") for mount in mounts)
    assert not any(mount.endswith(":/opt/vla") or mount.endswith(":/workspace") for mount in mounts)
    assert "wrong:mount" not in mounts
    assert json.loads((tmp_path / "scratch-mounts.json").read_text())[0]["command"] == command


def test_loss_records_do_not_invent_optimizer_steps():
    records, summary = metrics.training_records(
        "noise\n{'loss': 0.4, 'learning_rate': 0.001}\n"
        "{'loss': 0.1, 'step': 20}\n"
        "{'train_runtime': 12.5, 'train_loss': 0.2}\n")
    assert records[0]["optimizer_step"] is None
    assert records[1]["optimizer_step"] == 20
    assert summary["train_runtime"] == 12.5


def test_metrics_accept_local_sdk_epochs_and_docker_nanoseconds(tmp_path):
    (tmp_path / "train.log").write_text("{'loss': 0.1, 'step': 200}\n")
    started = metrics._timestamp("2026-09-15T23:04:30.627803Z")
    metrics.write_metrics(
        tmp_path, {"elapsed_seconds": 2},
        [{"StepName": "FineTune", "StartTime": started, "EndTime": started + 1.5}],
        [{"kind": "train", "state": {
            "StartedAt": "2026-09-15T23:04:30.627803457Z",
            "FinishedAt": "2026-09-15T23:04:32.127803457Z"}}],
        {"TrainSteps": 200}, {}, 0.2)
    timing = json.loads((tmp_path / "timings.json").read_text())
    assert timing["step_wall_seconds"] == {"FineTune": 1.5}
    assert timing["container_wall_seconds"] == {"train": 1.5}
    assert len(timing["exporter_sha256"]) == 64


@pytest.mark.parametrize("loss", ["nan", "inf", "-inf", "1e999"])
def test_nonfinite_training_loss_is_not_accepted(loss):
    with pytest.raises(RuntimeError, match="non-finite"):
        metrics.training_records("{'loss': " + loss + "}")


def test_cli_rejects_reused_run_before_aws_or_gpu_work(tmp_path):
    import subprocess

    (tmp_path / "status.json").write_text('{"status": "Failed"}')
    process = subprocess.run([
        sys.executable, str(SCRIPTS / "run_local_pipeline.py"),
        "--profile", "openvla-libero", "--run-dir", str(tmp_path), "--run-id", "duplicate",
        "--expected-commit", "unused", "--development-bucket", "scratch",
        "--expected-role", "role",
    ], text=True, capture_output=True)
    assert process.returncode == 2 and "not empty" in process.stderr


@pytest.mark.parametrize("bad_readback,wrong_owner", [(False, False), (True, False), (False, True)])
def test_container_removal_requires_archived_bytes_and_run_ownership(
        tmp_path, monkeypatch, bad_readback, wrong_owner):
    import io

    archiver = load("archive_run")
    payloads = {
        "manifest.json": {
            "run_id": "sample", "canonical_commit": "abc", "region": "us-east-1",
            "caller": {"Account": "111122223333"}, "development_bucket": "dev-test"},
        "status.json": {"status": "Succeeded"},
        "independent-verification.json": {
            "status": "Succeeded", "run_id": "sample", "canonical_commit": "abc"},
        "container-exits.json": [{"id": str(index)} for index in range(3)],
    }
    for name, value in payloads.items():
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(archiver, "EVIDENCE_FILES", tuple(payloads))
    stored, removed = {}, []

    def put(**kwargs):
        stored[kwargs["Key"]] = kwargs["Body"]
        return {"VersionId": "test-version"}

    s3 = SimpleNamespace(
        put_object=put,
        get_object=lambda **kw: {
            "Body": io.BytesIO(b"corrupt" if bad_readback else stored[kw["Key"]]),
            "VersionId": "test-version"},
    )
    monkeypatch.setattr(archiver.boto3, "Session", lambda **kw: SimpleNamespace(
        client=lambda service: s3 if service == "s3" else SimpleNamespace(
            get_caller_identity=lambda: {"Account": "111122223333"})))
    actual = [{
        "Config": {"Labels": {"vla.local.run": "other" if wrong_owner else "sample"}},
        "State": {"Running": False, "ExitCode": 0},
    } for _ in range(3)]
    monkeypatch.setattr(archiver.subprocess, "check_output", lambda *a, **kw: json.dumps(actual))
    monkeypatch.setattr(archiver.subprocess, "run", lambda command, **kw: removed.append(command))
    if bad_readback or wrong_owner:
        with pytest.raises(RuntimeError):
            archiver.archive_run(tmp_path, remove_containers=True)
        assert removed == []
    else:
        result = archiver.archive_run(tmp_path, remove_containers=True)
        assert result["status"] == "ArchivedAndReadBack"
        assert removed == [["docker", "rm", "0", "1", "2"]]


def test_disk_report_records_minimum_without_claiming_exact_peak(tmp_path, monkeypatch):
    monitor = scratch.DiskMeasurements({"root": str(tmp_path)}, tmp_path)
    free = iter([100, 500, 80, 480, 90, 490])
    monkeypatch.setattr(scratch.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(free)))
    for _ in range(3):
        monitor.sample()
    result = json.loads((tmp_path / "disk-usage.json").read_text())
    assert result["minimum_free_bytes"] == {"root": 80, "scratch": 480}
    assert len(result["samples"]) == 3


@pytest.mark.parametrize("exit_code,oom,log,emits_output", [
    (0, False, "eval_seed mismatch: expected=1001 got=1000", False),
    (1, True, "eval_seed mismatch: expected=1001 got=1000", False),
    (1, False, "SDK bootstrap failed", False),
    (1, False, "eval_seed mismatch: expected=1001 got=1000", True),
    (1, False, "eval_seed mismatch: expected=1001 got=2000", False),
])
def test_negative_validation_must_fail_for_the_injected_reason(
        tmp_path, exit_code, oom, log, emits_output):
    control = load("check_bad_eval_seed")
    if emits_output:
        (tmp_path / "validated_metrics.json").write_text("{}")
    with pytest.raises(RuntimeError):
        control.check_rejection(
            {"Running": False, "ExitCode": exit_code, "OOMKilled": oom},
            log, tmp_path, 1001, 1000)


def test_negative_validation_accepts_only_the_expected_rejection(tmp_path):
    load("check_bad_eval_seed").check_rejection(
        {"Running": False, "ExitCode": 1, "OOMKilled": False},
        "[validate] FATAL: eval_seed mismatch: expected=1001 got=1000", tmp_path, 1001, 1000)
