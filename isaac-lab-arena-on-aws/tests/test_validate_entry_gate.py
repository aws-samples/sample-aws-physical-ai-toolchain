"""Load-bearing Validate-gate tests.

Runs the STAGED validate_entry.py (as stage_validate_code produces it, validator
embedded) as a subprocess against crafted metrics.json, asserting it FAILS CLOSED
(exit != 0) on the cases that make the gated-registry claim true:
  * policy_type=zero_action  -> not a real eval
  * episodes=0               -> nothing ran
  * missing metrics.json     -> no result to validate
  * missing EXPECTED_* env   -> gate refuses to validate vs invented expectations
(Missing-validator fail-closed is covered separately; the runner always
embeds the validator here.)

Note: the gate now REQUIRES the pipeline-owned expectations (EXPECTED_EVAL_SEED /
TRIALS / TASK_IDS / SUITE); pipeline.py always sets them. The _run helper supplies
matching values so the structural-failure cases reach their intended check, and
the crafted metrics carry the matching eval_seed / eval_trials so Check 2 passes.
"""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
_VALIDATE_ENTRY = pathlib.Path(__file__).resolve().parents[1] / "entrypoints/validate_entry.py"
import os
import subprocess
import shutil
import sys
import tarfile
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# Pipeline-owned expectations the gate now requires. Match the crafted metrics.
_EXPECT_ENV = {
    "EXPECTED_EVAL_SEED": "1000",
    # These fixtures' manifests carry no training provenance (max_steps and
    # dataset_manifest are null), i.e. they represent checkpoints supplied as input
    # rather than trained by the execution, so the eval-only provenance contract is
    # the correct one. The train-path contract is exercised separately.
    "EXPECTED_TRAIN_PATH": "eval_only",
    "EXPECTED_EVAL_TRIALS": "1",
    "EXPECTED_EVAL_TASK_IDS": "all",
    "EXPECTED_SUITE": "libero_spatial",
    "EXPECTED_MODEL_FAMILY": "molmoact2",
    "EXPECTED_FAMILY_VERSION": "n17",
    "EXPECTED_MODEL_SOURCE_URI": "s3://test-bucket/test-key/model.tar.gz",
    "EVALUATOR_IMAGE_URI": "123456789012.dkr.ecr.us-east-1.amazonaws.com/vla-eval@sha256:abababababababababababababababababababababababababababababababab",
    "_VALIDATE_S3_HEAD_STUB": json.dumps({
        "VersionId": "fixture-version", "ETag": '"fixture-etag"',
    }),
}


@pytest.fixture(scope="module")
def staged_validator():
    from vla_pipeline.runner import stage_validate_code
    return stage_validate_code(repo_root=_REPO_ROOT)


def _run(staged, eval_dir, out_dir, expectations=_EXPECT_ENV, checkpoint_dir=None):
    env = {**os.environ, "EVAL_OUTPUT_DIR": str(eval_dir), "OUTPUT_DIR": str(out_dir)}
    env["CHECKPOINT_DIR"] = str(checkpoint_dir or eval_dir.parent / "missing-checkpoint")
    # Clear any ambient EXPECTED_* first, then apply the requested set.
    for k in _EXPECT_ENV:
        env.pop(k, None)
    env.update(expectations)
    # C5: Validate promotes the validated bytes before emitting its receipt, so an
    # offline run needs somewhere to publish. The promotion record marks itself
    # local_test_publication, so it cannot pass for a real promotion downstream.
    env["_VALIDATE_PROMOTION_LOCAL_DIR"] = str(out_dir.parent / "trust")
    # schema 3: the version GET measures CONTENT, so the stub must carry the bytes of the
    # archive this run is validating. The stub is module-level and cannot know the per-test
    # path, so it is completed here where the checkpoint dir is known.
    #
    # I5: this used to overwrite BodyPath UNCONDITIONALLY, which meant no test could express a
    # mismatched or absent body -- the harness forced every run to agree with itself, so the
    # version-content check could not be exercised at all. Filled in only when the test has not
    # spoken: an explicit path is kept, and an explicit None means "genuinely absent".
    if "_VALIDATE_S3_HEAD_STUB" in env:
        stub = json.loads(env["_VALIDATE_S3_HEAD_STUB"])
        if "BodyPath" not in stub:
            stub["BodyPath"] = str(pathlib.Path(env["CHECKPOINT_DIR"]) / "model.tar.gz")
        elif stub["BodyPath"] is None:
            del stub["BodyPath"]
        env["_VALIDATE_S3_HEAD_STUB"] = json.dumps(stub)
    return subprocess.run(
        [sys.executable, staged], env=env, capture_output=True, text=True)



_ARENA_DOCKERFILE = (Path(_REPO_ROOT).parent / "containers" / "isaac-lab-arena" / "Dockerfile")


def _image_workspace_files():
    """Every file the Arena image COPYs into /workspace, read from the Dockerfile itself.

    R4#I1: this used to be a hand-written list of TWO files while the Dockerfile copied ten,
    six of which the evaluator imports by bare name. The staged layout therefore did not
    resemble the image at all, and the tests only passed by borrowing sys.path from unrelated
    test modules that happened to expose the shared directories at collection time. The bug
    was reported in four consecutive review passes and re-fixed three times, because a
    hand-written list drifts silently every time the Dockerfile gains a COPY.

    Deriving the list removes the drift instead of guarding against it. Returns
    [(source_path_relative_to_build_context, destination_basename)].

    COPY paths in this Dockerfile are relative to the build context, which is the toolchain
    root -- one level above this component -- so they carry the component directory as a
    prefix.
    """
    if not _ARENA_DOCKERFILE.exists():
        raise AssertionError(
            f"cannot derive the Arena image layout: {_ARENA_DOCKERFILE} is absent. The gate "
            f"tests stage what the image ships, so a missing Dockerfile is a hard failure -- "
            f"falling back to a hand-written list is what caused R4#I1.")
    files = []
    for raw in _ARENA_DOCKERFILE.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("COPY "):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        src, dest = parts[-2], parts[-1]
        if not dest.startswith("/workspace/"):
            continue
        files.append((src, dest[len("/workspace/"):]))
    if not files:
        raise AssertionError(
            f"parsed no /workspace COPY lines out of {_ARENA_DOCKERFILE}; the parser and the "
            f"Dockerfile have diverged")
    return files


def _load_staged_arena(tmp_path, arena_root, monkeypatch, module_name):
    """Stage the Arena entry as the image lays it out and load it with the image's import path.

    The image runs `python3 /workspace/eval_entry.py`, so CPython puts /workspace at
    sys.path[0] and the evaluator's bare-name sibling imports resolve there.
    `spec_from_file_location` + `exec_module` does NOT do that, so a staged load without an
    explicit path prepend cannot resolve those imports and instead picks up whatever an
    earlier test left on sys.path. That is the second half of R4#I1: the staged file set was
    incomplete AND the import context was wrong, so the tests reported on a layout that
    exists nowhere.

    monkeypatch.syspath_prepend is undone at teardown, so this does not leak to other tests.

    Both gate call sites go through here. Two sites loading the staged entry with slightly
    different setup is how one of them kept being fixed while the other was not.
    """
    entry = _stage_arena_entry(tmp_path, arena_root)
    monkeypatch.syspath_prepend(str(entry.parent))
    spec = importlib.util.spec_from_file_location(module_name, entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_arena_entry(tmp_path, arena_root):
    """Stage the Arena entry with EVERY file the image puts beside it in /workspace.

    The image runs `python3 /workspace/eval_entry.py`, which makes /workspace the script
    import directory, so the evaluator's bare-name imports -- source_identity, digest,
    validator, capped_reader, checkpoint_compat, gr00t_seeded_server -- resolve to siblings.
    Reproducing that layout is the whole point of staging: a partial copy tests a layout
    production never uses.
    """
    staged = tmp_path / "arena-staged"
    staged.mkdir(exist_ok=True)
    context_root = Path(_REPO_ROOT).parent
    missing = []
    for src_rel, dest_name in _image_workspace_files():
        src = context_root / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        shutil.copy(src, staged / dest_name)
    if missing:
        raise AssertionError(
            f"the Dockerfile COPYs paths that do not exist in the tree: {missing}. The image "
            f"build would fail, so the test must too.")
    return staged / "eval_entry.py"

def _write_metrics(eval_dir, metrics):
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


@pytest.fixture
def valid_pipeline_report():
    digest = "sha256:" + "a" * 64
    commit = "b" * 40
    return {
        "schema_version": 3,
        "source_archive": {"sha256": "b" * 64, "size_bytes": 4096},
        "policy_type": "checkpoint",
        "model_family": "molmoact2",
        "checkpoint": "/opt/ml/input/data/model",
        "checkpoint_revision": None,
        "checkpoint_manifest": {
            "manifest_version": 1,
            "model_family": "molmoact2",
            "base_checkpoint": "allenai/MolmoAct2-LIBERO",
            "base_revision": commit,
            "train_seed": 42,
            "input_config": {
                "norm_tag": "libero", "model_dtype": "float32",
                "inference_action_mode": "continuous",
                # The value the evaluator actually applies. This was "{}" -- a
                # placeholder the schema accepted because these fields were
                # unconstrained strings, which is the defect I14 describes.
                "camera_name_mapping": (
                    '{"agentview_image":"image",'
                    '"robot0_eye_in_hand_image":"wrist_image"}'),
            },
            "train_recipe": {
                "repo": "https://github.com/allenai/lerobot.git",
                "commit": commit, "max_steps": 100,
            },
            "weights_digest": digest,
            "dataset_manifest": {"source": "fixture", "revision": commit, "episode_count": 10},
        },
        "weights_digest_recomputed_by_eval": digest,
        "model_artifact_identity": {
            "s3_uri": "s3://test-bucket/test-key/model.tar.gz",
            "bucket": "test-bucket", "key": "test-key/model.tar.gz",
            "version_id": "fixture-version", "etag": "fixture-etag",
        },
        "suite": "libero_spatial",
        # C2: the protocol comparison had never executed, because the resolved suite table
        # discarded evaluation_protocol and the check gated on its presence. With the table
        # fixed, the check runs -- and immediately caught that this fixture never recorded
        # what the evaluator consumed. These are libero_spatial's declared values.
        "effective_eval_config": {"n_action_steps": 8, "max_episode_steps": 720},
        "task_ids": list(range(10)),
        "num_trials_per_task": 1,
        "eval_seed": 1000,
        "train_seed": 42,
        "success_rate": 1.0,
        "episodes": 10,
        "episodes_reported_by_evaluator": 10,
        "per_task": [
            {"task_id": i, "task": f"task_{i}", "episodes": 1, "success_rate": 1.0}
            for i in range(10)
        ],
        "provenance": {
            "recipe_repo": "https://github.com/allenai/lerobot.git", "recipe_commit": commit,
            "mujoco_gl": "egl", "lerobot_branch": "fixture", "python_version": "3.12",
        },
    }


def _pack_checkpoint(root, channel, report=None):
    """Pack the checkpoint and, when given a report, record the archive it actually built.

    schema 3: Validate measures the archive and requires it to equal what the evaluation
    reported. A hardcoded fixture value would prove only that the comparison runs.
    """
    from vla_pipeline.common.digest import measure_archive

    channel.mkdir()
    archive_path = channel / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
    if report is not None:
        sha256, size_bytes = measure_archive(str(archive_path))
        report["source_archive"] = {"sha256": sha256, "size_bytes": size_bytes}
    return channel


@pytest.fixture
def checkpoint_tree(tmp_path, valid_pipeline_report):
    from vla_pipeline.common.digest import weights_digest

    root = tmp_path / "weights"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"original checkpoint")
    digest = weights_digest(str(root))
    valid_pipeline_report["checkpoint_manifest"]["weights_digest"] = digest
    valid_pipeline_report["weights_digest_recomputed_by_eval"] = digest
    (root / "checkpoint_manifest.json").write_text(
        json.dumps(valid_pipeline_report["checkpoint_manifest"]))
    return root


def test_matching_family_passes(staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel",
                               valid_pipeline_report)
    _write_metrics(ed, valid_pipeline_report)
    result = _run(staged_validator, ed, od, checkpoint_dir=channel)
    assert result.returncode == 0, result.stdout + result.stderr
    validated = json.loads((od / "validated_metrics.json").read_text())
    assert validated["validation_passed"] is True
    assert validated["success_rate"] == 1.0
    for field in ("model_family", "suite", "task_ids", "eval_seed", "episodes", "per_task"):
        assert validated[field] == valid_pipeline_report[field]
    assert validated["eval_trials"] == valid_pipeline_report["num_trials_per_task"]
    assert validated["budget_type"] == "fixed_trials"
    assert validated["weights_digest_recomputed_by_validate"] == valid_pipeline_report[
        "weights_digest_recomputed_by_eval"]


@pytest.mark.parametrize("selection", ["[false]", "[0.0]", "[0,0]", "[10]", "[-1]", "[]", "null"])
def test_packaged_validator_rejects_invalid_task_expectation(
    staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree, selection
):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel",
                               valid_pipeline_report)
    task_id = 10 if selection == "[10]" else -1 if selection == "[-1]" else 0
    valid_pipeline_report.update(
        task_ids=[task_id], episodes=1, episodes_reported_by_evaluator=1,
        per_task=[{"task_id": task_id, "task": "fixture", "episodes": 1, "success_rate": 1.0}],
    )
    _write_metrics(ed, valid_pipeline_report)
    result = _run(staged_validator, ed, od, {
        **_EXPECT_ENV, "EXPECTED_EVAL_TASK_IDS": selection,
    }, checkpoint_dir=channel)
    assert result.returncode != 0
    assert "invalid task selection" in result.stdout + result.stderr
    assert not (od / "validated_metrics.json").exists()

@pytest.mark.parametrize("counts", [(1, 1), (6, 6), (4, 6), (5, 5)])
def test_packaged_validator_enforces_fixed_trial_budget(
    staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree, counts
):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel",
                               valid_pipeline_report)
    valid_pipeline_report.update(
        task_ids=[0, 1], num_trials_per_task=5,
        episodes=sum(counts), episodes_reported_by_evaluator=sum(counts),
        per_task=[
            {"task_id": index, "task": f"task_{index}", "episodes": count, "success_rate": 1.0}
            for index, count in enumerate(counts)
        ],
    )
    _write_metrics(ed, valid_pipeline_report)
    expectations = {
        **_EXPECT_ENV, "EXPECTED_EVAL_TRIALS": "5", "EXPECTED_EVAL_TASK_IDS": "[0,1]",
    }
    result = _run(staged_validator, ed, od, expectations, checkpoint_dir=channel)
    if counts == (5, 5):
        assert result.returncode == 0, result.stdout + result.stderr
        validated = json.loads((od / "validated_metrics.json").read_text())
        assert validated["episodes"] == 10
        assert validated["weights_digest_recomputed_by_validate"] == valid_pipeline_report[
            "weights_digest_recomputed_by_eval"]
    else:
        assert result.returncode != 0
        assert "!= expected trials 5" in result.stdout + result.stderr
        assert not (od / "validated_metrics.json").exists()


@pytest.mark.parametrize("input_state", ["missing_directory", "missing_archive"])
def test_missing_checkpoint_input_fails_closed(
    staged_validator, tmp_path, valid_pipeline_report, input_state
):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = tmp_path / "channel"
    if input_state == "missing_archive":
        channel.mkdir()
    _write_metrics(ed, valid_pipeline_report)
    result = _run(staged_validator, ed, od, checkpoint_dir=channel)
    assert result.returncode != 0
    assert "Full schema-v3 validation PASSED" in result.stdout
    # schema 3: the archive measurement runs first and names the cause, rather than the run
    # reaching extraction and surfacing a bare FileNotFoundError. Still fails closed, and now
    # says WHY -- which is the point of measuring before extracting.
    assert "checkpoint archive is not a file" in result.stderr
    assert str(channel) in result.stderr
    assert not (od / "validated_metrics.json").exists()


@pytest.mark.parametrize("change", ["weights", "manifest", "substituted_report"])
def test_independent_checkpoint_rejects_mismatched_evidence(
    staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree, change
):
    from vla_pipeline.common.digest import weights_digest

    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    if change == "weights":
        (checkpoint_tree / "weights.bin").write_bytes(b"corrupted checkpoint")
    elif change == "manifest":
        manifest_path = checkpoint_tree / "checkpoint_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["base_revision"] = "c" * 40
        manifest_path.write_text(json.dumps(manifest))
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel",
                               valid_pipeline_report)
    if change == "substituted_report":
        other = tmp_path / "other"
        other.mkdir()
        (other / "weights.bin").write_bytes(b"another valid checkpoint")
        digest = weights_digest(str(other))
        valid_pipeline_report["checkpoint_manifest"]["weights_digest"] = digest
        valid_pipeline_report["weights_digest_recomputed_by_eval"] = digest
    elif change == "missing_archive":
        channel = tmp_path / "empty-channel"
        channel.mkdir()
    _write_metrics(ed, valid_pipeline_report)
    result = _run(staged_validator, ed, od, checkpoint_dir=channel)
    assert result.returncode != 0
    assert "Full schema-v3 validation PASSED" in result.stdout
    expected = {
        "weights": "independent checkpoint digest differs",
        "manifest": "checkpoint manifest differs",
        "substituted_report": "checkpoint manifest differs",
    }[change]
    assert expected in result.stdout + result.stderr
    assert not (od / "validated_metrics.json").exists()


@pytest.mark.parametrize("policy_type", [None, "", "zero_action", "positive_control", "dummy", "unknown"])
def test_non_checkpoint_policy_rejected(staged_validator, tmp_path, valid_pipeline_report, policy_type):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    if policy_type is None:
        del valid_pipeline_report["policy_type"]
    else:
        valid_pipeline_report["policy_type"] = policy_type
    _write_metrics(ed, valid_pipeline_report)
    result = _run(staged_validator, ed, od)
    assert result.returncode != 0
    assert "policy_type must be 'checkpoint'" in result.stdout + result.stderr
    assert not (od / "validated_metrics.json").exists()


def test_arena_report_not_published_when_checkpoint_changed_during_rollout(
    staged_validator, tmp_path, valid_pipeline_report, monkeypatch
):
    """A digest mismatch found AFTER the rollout must block publication.

    The producer recomputes the checkpoint digest once the evaluation finishes, but the
    result was only ever compared in a log line -- so a mismatch was recorded in the
    report and published anyway, leaving the downstream Validate step as the only thing
    standing between a mutated checkpoint and a registered model. Here the checkpoint is
    mutated after its manifest is written, i.e. the bytes evaluated are not the bytes the
    manifest describes, and write_metrics must refuse to emit metrics.json.
    """
    from vla_pipeline.common.digest import weights_digest

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"fixture checkpoint")
    manifest = valid_pipeline_report["checkpoint_manifest"]
    manifest["model_family"] = "gr00t"
    manifest["input_config"] = {
        "embodiment_tag": "GR1", "n_action_steps": "8", "max_episode_steps": "720",
    }
    manifest["weights_digest"] = weights_digest(str(checkpoint))
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    # ...the checkpoint changes underneath the running evaluation.
    (checkpoint / "weights.bin").write_bytes(b"MUTATED after the manifest was written")

    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text("model_path: /placeholder\n")
    eval_dir = tmp_path / "eval"
    monkeypatch.setenv("EVAL_MODEL_FAMILY", "gr00t")
    monkeypatch.setenv("EVAL_SUITE", "arena_gr1_fridge")
    monkeypatch.setenv("EVAL_SEED", "1000")
    monkeypatch.setenv("EVAL_TRIALS", "1")
    monkeypatch.setenv("EVAL_TASK_IDS", "all")
    monkeypatch.setenv("EVAL_GR00T_VERSION", "n16")
    monkeypatch.setenv("SM_HP_EMBODIMENT_TAG", "GR1")
    monkeypatch.setenv("EVAL_ARENA_EMBODIMENT", "gr1_joint")
    monkeypatch.setenv("EVAL_OBJECT", "NONE")
    monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", str(policy_config))
    monkeypatch.setenv("EVAL_SIM_CONFIG", json.dumps(
        {"task_name": "put_item_in_fridge_and_close_door"}))
    monkeypatch.setenv("MUJOCO_GL", "egl")
    arena_root = Path(_REPO_ROOT) / "entrypoints/eval/isaac_arena"
    digest_spec = importlib.util.spec_from_file_location(
        "digest", arena_root / "_shared/digest.py")
    digest_module = importlib.util.module_from_spec(digest_spec)
    digest_spec.loader.exec_module(digest_module)
    monkeypatch.setitem(sys.modules, "digest", digest_module)
    # The producer now runs the SHARED report validator before publishing (I6), so
    # supply the REAL module rather than a stub -- the image bakes validator.py beside
    # the entrypoint, and a stub would defeat the check this exercises.
    validator_spec = importlib.util.spec_from_file_location(
        "validator", Path(_REPO_ROOT) / "src/vla_pipeline/common/validator.py")
    validator_module = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator_module)
    monkeypatch.setitem(sys.modules, "validator", validator_module)
    arena = _load_staged_arena(tmp_path, arena_root, monkeypatch, "arena_mutation_fixture")
    # This test exercises the PRODUCER only -- it never reaches Validate's archive
    # comparison -- so a literal measurement is right here. Without one the new
    # publication guard fires first and masks the digest mismatch under test.
    arena._RESOLVED = _resolved_over(tmp_path, "producer_only")
    # schema 3: the producer measures the archive before extraction and refuses to publish
    # without it. These tests drive write_metrics directly, so supply the measurement the
    # real extraction step would have taken.
    monkeypatch.setattr(arena, "_s3_head_identity",
                        lambda uri: valid_pipeline_report["model_artifact_identity"])

    results = {"eval_backend": "isaac_lab_arena", "episodes": 1, "success_rate": 1.0,
               "policy_type": "checkpoint",
               "task_name": "put_item_in_fridge_and_close_door"}
    with pytest.raises(RuntimeError, match="weights digest mismatch"):
        # posctrl_commit is REQUIRED for a positive control: the report must name the commit the Hub
        # RESOLVED, not the requested reference. Passing it unconditionally is harmless for the
        # checkpoint path, which ignores it.
        arena.write_metrics(results, str(eval_dir), str(checkpoint),
                            posctrl_commit="c" * 40)
    assert not (eval_dir / "metrics.json").exists(), (
        "a report must NOT be published when the producer already knows the digest "
        "does not match")


@pytest.mark.parametrize("task", ["put_item_in_fridge_and_close_door", "gr1_open_microwave"])
@pytest.mark.parametrize("mode", ["checkpoint", "zero_action", "positive_control"])
def test_arena_generated_report_registration_gate(
    staged_validator, tmp_path, valid_pipeline_report, monkeypatch, mode, task
):
    from vla_pipeline.common.digest import weights_digest

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"fixture checkpoint")
    manifest = valid_pipeline_report["checkpoint_manifest"]
    manifest["model_family"] = "gr00t"
    manifest["input_config"] = {
        "embodiment_tag": "GR1", "n_action_steps": "8", "max_episode_steps": "720",
    }
    manifest["weights_digest"] = weights_digest(str(checkpoint))
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text("model_path: /placeholder\n")
    _fridge_policy = "/workspace/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml"
    eval_dir, output_dir = tmp_path / "eval", tmp_path / "validated"
    expectations = {
        **_EXPECT_ENV, "EXPECTED_MODEL_FAMILY": "gr00t", "EXPECTED_SUITE": "arena_gr1_fridge",
        "EXPECTED_FAMILY_VERSION": "n16",
        "EXPECTED_EMBODIMENT_TAG": "GR1",
        "EXPECTED_ARENA_EMBODIMENT": "gr1_joint",
        "EXPECTED_ARENA_OBJECT": "NONE",
        "EXPECTED_POLICY_CONFIG": _fridge_policy,
    }
    monkeypatch.setenv("EVAL_MODEL_FAMILY", "gr00t")
    monkeypatch.setenv("EVAL_SUITE", "arena_gr1_fridge")
    monkeypatch.setenv("EVAL_SEED", "1000")
    monkeypatch.setenv("EVAL_TRIALS", "1")
    monkeypatch.setenv("EVAL_TASK_IDS", "all")
    monkeypatch.setenv("EVAL_GR00T_VERSION", "n16")
    monkeypatch.setenv("EVAL_POSCTRL_N16", str(mode == "positive_control").lower())
    monkeypatch.setenv("SM_HP_EMBODIMENT_TAG", "GR1")
    monkeypatch.setenv("EVAL_ARENA_EMBODIMENT", "gr1_joint")
    _fridge_policy = "/workspace/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml"
    monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", _fridge_policy)
    monkeypatch.setenv("EVAL_SIM_CONFIG", json.dumps({"task_name": task}))
    monkeypatch.setenv("MUJOCO_GL", "egl")
    arena_root = Path(_REPO_ROOT) / "entrypoints/eval/isaac_arena"
    digest_spec = importlib.util.spec_from_file_location("digest", arena_root / "_shared/digest.py")
    digest_module = importlib.util.module_from_spec(digest_spec)
    digest_spec.loader.exec_module(digest_module)
    monkeypatch.setitem(sys.modules, "digest", digest_module)
    # The producer now runs the SHARED report validator before publishing (I6), so
    # supply the REAL module rather than a stub -- the image bakes validator.py beside
    # the entrypoint, and a stub would defeat the check this exercises.
    validator_spec = importlib.util.spec_from_file_location(
        "validator", Path(_REPO_ROOT) / "src/vla_pipeline/common/validator.py")
    validator_module = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator_module)
    monkeypatch.setitem(sys.modules, "validator", validator_module)
    arena = _load_staged_arena(tmp_path, arena_root, monkeypatch, "arena_gate_fixture")
    # schema 3: the producer measures the archive before extraction and refuses to publish
    # without it. These tests drive write_metrics directly, so supply the measurement the
    # real extraction step would have taken.
    monkeypatch.setenv("EVAL_POLICY_CONFIG_YAML", str(policy_config))
    monkeypatch.setattr(arena, "_s3_head_identity", lambda uri: valid_pipeline_report[
        "model_artifact_identity"])
    # run_arena_eval writes a real patched policy file. Let the production digest
    # measure it so the saved config and its recorded identity agree.
    # Pack the archive BEFORE evaluating, as a real run has it: measured at extraction, then
    # evaluated. Seeding a literal here and packing afterwards made the producer and Validate
    # disagree for a reason the test was not about.
    from vla_pipeline.common.digest import measure_archive as _measure
    channel = _pack_checkpoint(checkpoint, tmp_path / "channel")
    _sha, _size = _measure(str(channel / "model.tar.gz"))
    from vla_pipeline.common.source_identity import from_archive as _from_archive
    arena._RESOLVED = _from_archive(load_root=str(checkpoint),
                                    archive_path=str(channel / "model.tar.gz"),
                                    checkpoint=str(checkpoint),
                                    sha256=_sha, size_bytes=_size)
    with monkeypatch.context() as runtime:
        runtime.setattr(arena.shutil, "which", lambda name: None)
        runtime.setattr(threading, "Timer", Mock())
        process = Mock(stdout=io.StringIO(
            "[Rank 0/1] Metrics: {'num_episodes': 1, 'success_rate': 1.0}\n"))
        process.wait.return_value = 0
        runtime.setattr(arena.subprocess, "Popen", Mock(return_value=process))
        results = arena.run_arena_eval(
            task, 1, num_episodes=1,
            policy_type="zero_action" if mode == "zero_action" else "gr00t_remote",
            checkpoint_path=str(checkpoint),
        )
        # posctrl_commit is REQUIRED for a positive control: the report must name the commit the Hub
        # RESOLVED, not the requested reference. Passing it unconditionally is harmless for the
        # checkpoint path, which ignores it.
        report = arena.write_metrics(results, str(eval_dir), str(checkpoint),
                                     posctrl_commit="c" * 40)
    assert report["policy_type"] == mode
    assert json.loads((eval_dir / "metrics.json").read_text()) == report
    result = _run(staged_validator, eval_dir, output_dir, expectations, checkpoint_dir=channel)
    if mode == "checkpoint" and task != "put_item_in_fridge_and_close_door":
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "suite/task mismatch" in combined or "coherence" in combined or "VALIDATION FAILED" in combined
        assert not (output_dir / "validated_metrics.json").exists()
    elif mode == "checkpoint":
        assert result.returncode == 0, result.stdout + result.stderr
        validated = json.loads((output_dir / "validated_metrics.json").read_text())
        assert validated["validation_passed"] is True
        assert validated["policy_type"] == "checkpoint"
    else:
        assert result.returncode != 0
        assert "policy_type must be 'checkpoint'" in result.stdout + result.stderr
        assert not (output_dir / "validated_metrics.json").exists()


def test_wrong_family_rejected(staged_validator, tmp_path, valid_pipeline_report):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    _write_metrics(ed, valid_pipeline_report)
    expectations = {**_EXPECT_ENV, "EXPECTED_MODEL_FAMILY": "gr00t"}
    result = _run(staged_validator, ed, od, expectations)
    assert result.returncode != 0
    assert "model_family mismatch" in result.stdout + result.stderr
    assert not (od / "validated_metrics.json").exists()


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_family_expectation_rejected(
    staged_validator, tmp_path, valid_pipeline_report, value
):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    _write_metrics(ed, valid_pipeline_report)
    expectations = dict(_EXPECT_ENV)
    if value is None:
        del expectations["EXPECTED_MODEL_FAMILY"]
    else:
        expectations["EXPECTED_MODEL_FAMILY"] = value
    result = _run(staged_validator, ed, od, expectations)
    assert result.returncode != 0
    assert "EXPECTED_MODEL_FAMILY is not set" in result.stdout + result.stderr
    assert not (od / "validated_metrics.json").exists()


def test_zero_action_hard_fails(staged_validator, tmp_path):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()
    _write_metrics(ed, {"schema_version": 3, "policy_type": "zero_action",
                        "success_rate": 0.0, "episodes": 1,
                        "eval_seed": 1000, "eval_trials": 1,
                        "model_family": "molmoact2"})
    r = _run(staged_validator, ed, od)
    assert r.returncode != 0
    assert "zero_action" in (r.stdout + r.stderr)


def test_episodes_zero_hard_fails(staged_validator, tmp_path, valid_pipeline_report,
                                  checkpoint_tree):
    """A run that produced no episodes must be rejected for THAT reason.

    I20: this used a report missing most required fields and asserted only that the word
    "episodes" appeared in the output -- which a missing-key error also satisfies, so
    removing the zero-episode guard would not have failed the test. Built from the valid
    fixture with only the episode count zeroed, the rejection is attributable.
    """
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()
    report = dict(valid_pipeline_report)
    report["episodes"] = 0
    report["episodes_reported_by_evaluator"] = 0
    report["per_task"] = []
    _write_metrics(ed, report)
    r = _run(staged_validator, ed, od,
             checkpoint_dir=checkpoint_tree)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    # The explicit zero-episode guard, not an incidental missing-field error.
    assert "episodes=0 means no evaluation actually ran" in out, out
    assert not os.path.exists(os.path.join(od, "validated_metrics.json")), out


def test_missing_metrics_hard_fails(staged_validator, tmp_path):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()  # no metrics.json written
    r = _run(staged_validator, ed, od)
    assert r.returncode != 0
    assert "metrics.json not found" in (r.stdout + r.stderr)


def test_missing_expectations_hard_fails(staged_validator, tmp_path):
    # Fail-loud audit: with no EXPECTED_* set, the gate must refuse rather than
    # validate against an invented default (trials=2 / seed=1000 / ...).
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()
    _write_metrics(ed, {"schema_version": 3, "policy_type": "checkpoint",
                        "success_rate": 0.5, "episodes": 10,
                        "eval_seed": 1000, "eval_trials": 1,
                        "model_family": "molmoact2"})
    r = _run(staged_validator, ed, od, expectations={})  # cleared
    assert r.returncode != 0
    assert "EXPECTED_EVAL_SEED" in (r.stdout + r.stderr)


def test_no_validated_output_on_failure(staged_validator, tmp_path, valid_pipeline_report,
                                        checkpoint_tree):
    # A failed validation must NOT emit validated_metrics.json (gate can't read it).
    #
    # I20: this used a zero_action report missing many required fields, so the absence of
    # output could have been caused by anything -- removing the intended behaviour would
    # not have failed it. Built from the valid fixture with ONE thing wrong, so the
    # rejection is attributable and the absent output is attributable to it.
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()
    report = dict(valid_pipeline_report)
    report["policy_type"] = "zero_action"
    _write_metrics(ed, report)
    r = _run(staged_validator, ed, od, checkpoint_dir=checkpoint_tree)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "policy_type must be 'checkpoint' for registration" in out, out
    assert not os.path.exists(os.path.join(od, "validated_metrics.json")), out


# --- I5: the version-content check must be EXECUTED, not merely present in the source -------
#
# My first tests for this asserted that the error string appeared in validate_entry.py. Astra
# disabled the rejection by replacing its condition with `if False` and every one of them still
# passed, because the string was untouched. These run staged Validate instead.


def test_a_version_whose_body_differs_is_rejected(
        staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree):
    """The named S3 version must CONTAIN the bytes the evaluation measured.

    This is the whole point of the version GET replacing the unversioned HEAD: a peer that
    replaced the object would preserve the reported identity while changing the bytes.
    """
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel", valid_pipeline_report)
    _write_metrics(ed, valid_pipeline_report)
    # A version body that is NOT the archive the evaluation measured.
    other = tmp_path / "other-version.tar.gz"
    other.write_bytes(b"different bytes entirely")
    expectations = dict(_EXPECT_ENV)
    stub = json.loads(expectations["_VALIDATE_S3_HEAD_STUB"])
    stub["BodyPath"] = str(other)
    expectations["_VALIDATE_S3_HEAD_STUB"] = json.dumps(stub)
    result = _run(staged_validator, ed, od, expectations=expectations, checkpoint_dir=channel)
    assert result.returncode != 0, "a substituted version body was accepted"
    combined = result.stdout + result.stderr
    assert "S3 version content mismatch" in combined
    # And no receipt may exist: a rejected run must not leave anything downstream can consume.
    assert not (od / "validated_metrics.json").exists()


def test_a_matching_version_body_passes(
        staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree):
    """The control. Without it, the rejection test could pass for an unrelated reason."""
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel", valid_pipeline_report)
    _write_metrics(ed, valid_pipeline_report)
    result = _run(staged_validator, ed, od, checkpoint_dir=channel)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "S3 version verified BY CONTENT" in result.stdout


def test_a_changed_version_id_is_rejected(
        staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree):
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel", valid_pipeline_report)
    _write_metrics(ed, valid_pipeline_report)
    expectations = dict(_EXPECT_ENV)
    stub = json.loads(expectations["_VALIDATE_S3_HEAD_STUB"])
    stub["VersionId"] = "some-other-version"
    expectations["_VALIDATE_S3_HEAD_STUB"] = json.dumps(stub)
    result = _run(staged_validator, ed, od, expectations=expectations, checkpoint_dir=channel)
    assert result.returncode != 0
    assert "VersionId changed" in result.stdout + result.stderr


def test_a_stub_without_a_body_is_rejected(
        staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree):
    """A body-less stub would exercise a weaker check than production."""
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel", valid_pipeline_report)
    _write_metrics(ed, valid_pipeline_report)
    expectations = dict(_EXPECT_ENV)
    stub = json.loads(expectations["_VALIDATE_S3_HEAD_STUB"])
    stub["BodyPath"] = None  # the harness reads None as "genuinely absent"
    expectations["_VALIDATE_S3_HEAD_STUB"] = json.dumps(stub)
    result = _run(staged_validator, ed, od, expectations=expectations, checkpoint_dir=channel)
    assert result.returncode != 0
    assert "has no BodyPath" in result.stdout + result.stderr


def test_a_changed_etag_is_rejected(
        staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree):
    """A mutation campaign found this check unprotected: disabling it left the suite green.

    It is partly redundant with the content comparison, since different bytes produce a
    different sha256. But an unprotected check is one nobody would notice breaking, and if the
    content comparison were ever removed this would be the only remaining binding between the
    reported identity and the version Validate read.
    """
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel", valid_pipeline_report)
    _write_metrics(ed, valid_pipeline_report)
    expectations = dict(_EXPECT_ENV)
    stub = json.loads(expectations["_VALIDATE_S3_HEAD_STUB"])
    stub["ETag"] = '"a-different-etag"'
    expectations["_VALIDATE_S3_HEAD_STUB"] = json.dumps(stub)
    result = _run(staged_validator, ed, od, expectations=expectations, checkpoint_dir=channel)
    assert result.returncode != 0, "a changed ETag was accepted"
    assert "ETag changed" in result.stdout + result.stderr
    assert not (od / "validated_metrics.json").exists()


def test_a_report_claiming_a_different_source_location_is_rejected(
        staged_validator, tmp_path, valid_pipeline_report, checkpoint_tree):
    """Also found unprotected by mutation: disabling it left the suite green.

    The report is produced by the evaluation worker; the expectation comes from the pipeline. If
    the two locations are not compared, a report can name an artifact the pipeline never asked
    about and the version checks would then verify THAT artifact faithfully -- confirming the
    wrong object with full rigour.
    """
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    report = dict(valid_pipeline_report)
    report["model_artifact_identity"] = {
        **report["model_artifact_identity"], "bucket": "some-other-bucket"}
    channel = _pack_checkpoint(checkpoint_tree, tmp_path / "channel", report)
    _write_metrics(ed, report)
    result = _run(staged_validator, ed, od, checkpoint_dir=channel)
    assert result.returncode != 0, "a report naming a different bucket was accepted"
    assert "Model source location mismatch" in result.stdout + result.stderr


def test_the_receipt_carries_the_producers_seed_qualification():
    """I2: the receipt kept effective_eval_config but DROPPED seed_scope.

    Arena records what the eval seed actually bound, read from the seeding wrapper's evidence
    rather than asserted, and says so when that evidence is absent. Omitting it from the
    immutable receipt meant a run whose policy RNG was not demonstrably bound registered
    indistinguishably from one where it was -- the qualification was lost exactly where it
    matters most.
    """
    source = _VALIDATE_ENTRY.read_text()
    start = source.index("All checks passed: emit validated_metrics.json")
    block = source[start:source.index("json.dump", start)]
    assert '"seed_scope"' in block
    # Carried verbatim, never defaulted: a fabricated scope is worse than an absent one.
    assert 'metrics.get("seed_scope")' in block


def _resolved_over(tmp_path, name="seed"):
    """A real ResolvedCheckpoint over a real tiny archive, for tests that drive write_metrics directly.

    These used to seed a module-level dict with an invented measurement. The constructor now measures
    the archive it is given and refuses a disagreement, so an invented value cannot build an identity
    -- which is the point: a fixture able to fabricate one is how a producer bug stays invisible.
    """
    import tarfile
    from vla_pipeline.common.digest import measure_archive
    from vla_pipeline.common.source_identity import from_archive
    root = tmp_path / f"{name}_tree"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "gr00t"}')
    arc = tmp_path / f"{name}.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        tf.add(root, arcname="ckpt")
    sha, size = measure_archive(str(arc))
    return from_archive(load_root=str(root), archive_path=str(arc),
                        checkpoint=str(root), sha256=sha, size_bytes=size)
