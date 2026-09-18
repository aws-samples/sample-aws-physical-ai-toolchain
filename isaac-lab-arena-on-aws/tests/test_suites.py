"""Suite manifests are the single source of truth for what a suite means.

A "suite" used to be declared in four unsynchronised places:

  * ``scripts/run_arena.py`` CLI defaults        -> the Arena runtime knobs
  * ``entrypoints/train/gr00t/train_entry.py``   -> dataset + embodiment tag +
                                                    modality config
  * ``config/simulators/<sim>.yaml`` ``suites:`` -> canonical task ids
  * ``validator.VALID_SUITES``                   -> the registrable allowlist

Nothing tied them together, so a suite could be half-declared (``arena_g1``:
trainable but not validatable; ``dummy_suite``: referenced by a pair manifest but
declared nowhere) and a run could train one task while evaluating another.

``config/suites/<suite>.yaml`` now declares all of it once. The copies that must
still exist -- because they ship INTO containers where ``config/`` is absent --
are pinned to the manifests by the cross-check tests here, so drift fails CI
rather than surfacing as a mis-wired GPU run.
"""
from __future__ import annotations

# The N1.6 commit this component pins, so the fixture agrees with reality.
_N16_COMMIT = "5dc80c4afd726b34faad1d8f7e007a13b34e4c88"

import ast
import base64
import json
import os
import re
import subprocess
import sys
import tarfile

from vla_pipeline.common.digest import measure_archive as _measure_archive

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from vla_pipeline import registry as reg  # noqa: E402
from vla_pipeline.common import validator as val  # noqa: E402

# --------------------------------------------------------------------------
# Manifest schema / integrity
# --------------------------------------------------------------------------

def test_every_suite_manifest_resolves():
    """No suite manifest is malformed, and every one names a real simulator."""
    suites = reg.list_suites()
    assert suites, "config/suites/ is empty"
    known_sims = set(reg.list_simulators())
    for name in suites:
        s = reg.resolve_suite(name)
        assert s.name == name
        assert s.simulator in known_sims
        assert s.status in reg.SUITE_STATUSES
        assert s.canonical_task_ids, f"{name}: canonical_task_ids must be non-empty"


def test_arena_suites_declare_the_required_runtime_fields():
    """An isaac_arena suite must declare the fields the registry requires.

    Scope, deliberately: `task` is required; `embodiment` and
    `object` are legitimately null (the task takes no such flag); and
    `policy_config` may be null -- no shipped suite has one, so a `supported` Arena
    suite is currently NOT runnable without an operator-supplied config. That gap is
    enforced at submit rather than here (see the test below) and is recorded in
    README.md#notes-and-limitations.
    """
    arena_suites = reg.suites_for_simulator("isaac_arena", include_experimental=True)
    assert arena_suites, "no Arena suites declared"
    for name in arena_suites:
        a = reg.resolve_suite(name).arena
        assert a is not None, f"{name}: isaac_arena suite with no arena block"
        assert a.task, f"{name}: arena.task is required"
        # No budget field: the sample size is an episode count carried as
        # EvalTrials, so a manifest cannot declare one that disagrees with it.
        assert not hasattr(a, "num_steps"), (
            f"{name}: arena.num_steps is retired")


def test_supported_arena_suites_without_a_policy_config_are_refused_at_submit():
    """`status: supported` does not imply runnable -- but it must never run wrong.

    Every shipped Arena suite has `policy_config: null`, so submission must be
    refused rather than proceeding with an auto-discovered placeholder config
    (which would report a meaningless success_rate). This pins the actual
    invariant instead of overclaiming that `supported` means "fully configured".
    """
    from vla_pipeline.arena import ArenaConfigError, resolve_runtime
    for name in reg.suites_for_simulator("isaac_arena", include_experimental=True):
        suite = reg.resolve_suite(name)
        if suite.arena.policy_config:
            continue                      # a config was supplied; nothing to prove
        with pytest.raises(ArenaConfigError, match="policy config"):
            resolve_runtime(suite, "gr00t", "n17")


def test_non_arena_suites_have_no_arena_block():
    for name in reg.list_suites():
        s = reg.resolve_suite(name)
        if s.simulator != "isaac_arena":
            assert s.arena is None, f"{name}: arena block on a {s.simulator} suite"


def test_declared_modality_configs_exist_on_disk():
    """A modality config named by a manifest must be a real staged file.

    ``train_entry`` resolves it next to itself in the sourcedir; a typo here would
    only surface as a FineTune crash. NOTE the field is train-side only today -- the
    connector image bakes exactly one config (``arena_gr1_data_config.py``) and the
    eval entry has its own hardcoded default, so this value does NOT reach SimEval.
    See README.md#notes-and-limitations.
    """
    for name in reg.list_suites():
        s = reg.resolve_suite(name)
        for family, versions in s.family_overrides.items():
            for version, ov in versions.items():
                mc = ov.get("modality_config")
                if not mc:
                    continue
                path = os.path.join(_REPO_ROOT, "entrypoints", "train", family, mc)
                assert os.path.isfile(path), (
                    f"suite {name} family_overrides.{family}.{version}."
                    f"modality_config={mc!r} does not exist at {path}")


def test_pair_default_suites_resolve_to_their_own_simulator():
    """Every pair manifest's default_suite exists and runs on that simulator."""
    for family, simulator in reg.list_supported_pairs():
        spec = reg.resolve(family, simulator)
        s = reg.resolve_suite(spec.default_suite)   # raises if undeclared
        assert s.simulator == simulator, (
            f"pair {family}--{simulator} default_suite={spec.default_suite!r} "
            f"runs on {s.simulator!r}")


# --------------------------------------------------------------------------
# `status: experimental` is excluded from everything registrable
# --------------------------------------------------------------------------

def test_experimental_suites_are_not_registrable():
    experimental = [s for s in reg.list_suites() if not reg.resolve_suite(s).supported]
    assert experimental, "expected at least one experimental suite (arena_g1)"
    for name in experimental:
        assert name not in reg.valid_suites()
        assert name not in reg.suite_canonical_task_ids()
        assert name not in reg.suites_for_simulator(
            reg.resolve_suite(name).simulator)


def test_arena_g1_is_explicitly_experimental():
    """The half-wired suite is now declared half-wired instead of by accident.

    Before suite manifests, ``arena_g1`` had a dataset and a modality config in
    the train entry but was absent from the validator allowlist -- a run would
    fine-tune, evaluate, then fail Validate with "unknown suite".
    """
    assert reg.resolve_suite("arena_g1").status == "experimental"


# --------------------------------------------------------------------------
# Cross-checks: the in-container copies must mirror the manifests
# --------------------------------------------------------------------------

def test_validator_allowlist_mirrors_manifests():
    """validator.VALID_SUITES is the no-specs fallback; keep it accurate.

    validator.py is base64-embedded into the Validate container where config/ is
    absent, so it carries a literal fallback. That literal must equal the
    manifests' supported set or the fallback would accept/reject the wrong suites.
    """
    assert val.VALID_SUITES == set(reg.valid_suites())


def _dict_literal(path: str, name: str) -> dict:
    """Extract a module-level dict literal without importing the module.

    ``train_entry.py`` reads env and ``sys.exit()``s at import, so it cannot be
    imported in a test; parse it instead.
    """
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


_TRAIN_ENTRY = os.path.join(_REPO_ROOT, "entrypoints", "train", "gr00t",
                            "train_entry.py")


def test_train_entry_datasets_mirror_manifests():
    """The gr00t train entry's suite->dataset map must equal the manifests.

    The map still lives in the entry script because it ships in the sourcedir and
    runs where config/ is absent (rewiring it to read a staged suites.json is a
    portability item -- see README.md#notes-and-limitations). Pinning it here means the manifests
    stay authoritative: FineTune and SimEval cannot disagree about the suite.
    """
    datasets = _dict_literal(_TRAIN_ENTRY, "SUITE_DATASETS")
    meta = _dict_literal(_TRAIN_ENTRY, "SUITE_META")
    assert set(datasets) == set(meta), "SUITE_DATASETS/SUITE_META key drift"

    for suite, repo_id in datasets.items():
        s = reg.resolve_suite(suite)          # raises if the suite is undeclared
        assert s.dataset is not None, (
            f"train_entry declares a dataset for {suite!r} but "
            f"config/suites/{suite}.yaml has dataset: null")
        assert s.dataset.repo_id == repo_id
        m = meta[suite]
        assert m["subdir"] == s.dataset.subdir
        assert m["revision"] == s.dataset.revision
        assert m["copy_libero_modality"] == s.dataset.copy_libero_modality
        # embodiment_tag in SUITE_META is the n17 value; n16 is applied by a
        # separate branch in train_entry (see the n16 cross-check below).
        assert m["embodiment_tag"] == s.family_override("gr00t", "n17").get(
            "embodiment_tag")


def test_manifest_datasets_not_silently_missing_from_train_entry():
    """A suite with a dataset must be trainable -- no declared-but-unreachable data."""
    datasets = _dict_literal(_TRAIN_ENTRY, "SUITE_DATASETS")
    for name in reg.list_suites():
        s = reg.resolve_suite(name)
        if s.dataset is not None:
            assert name in datasets, (
                f"config/suites/{name}.yaml declares a dataset but the gr00t "
                f"train entry has no SUITE_DATASETS entry, so FineTune would "
                f"reject the suite")


def _train_entry_n16_suites() -> set:
    """The suite set in train_entry's `GR00T_VERSION == "n16"` branch, via ast.

    A whole-file `in src` grep would be VACUOUS: every suite declaring an n16 GR1
    tag is also a SUITE_DATASETS key, so its name appears in the file regardless of
    the n16 branch. Parse the actual branch condition.
    """
    with open(_TRAIN_ENTRY) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        # Match `GR00T_VERSION == "n16" and SUITE in (...)`
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
            continue
        has_n16 = any(
            isinstance(v, ast.Compare) and isinstance(v.left, ast.Name)
            and v.left.id == "GR00T_VERSION"
            and any(getattr(c, "value", None) == "n16" for c in v.comparators)
            for v in node.values)
        if not has_n16:
            continue
        for v in node.values:
            if (isinstance(v, ast.Compare) and isinstance(v.ops[0], ast.In)
                    and isinstance(v.left, ast.Name) and v.left.id == "SUITE"):
                return set(ast.literal_eval(v.comparators[0]))
    raise AssertionError(
        "could not find train_entry's `GR00T_VERSION == 'n16' and SUITE in (...)` "
        "branch -- the n16 embodiment pin cannot be verified")


def test_train_entry_n16_embodiment_tag_matches_manifest():
    """The n16 native-GR1 branch in train_entry must EQUAL the manifests' set.

    Set equality both ways: a suite declaring n16 `GR1` but missing from the branch
    trains under the wrong embodiment, and a suite in the branch without the
    declaration means the manifest is wrong. The one-directional substring version
    of this test was mutation-proven vacuous (dropping `arena_gr1_fridge` from the
    branch left the suite green while silently mismatching train/eval embodiment on
    a proven cell).

    Note this is the CI half of the defence; the load-bearing half is the
    validator's embodiment coherence gate, which reads the tag FineTune actually
    recorded (see test_embodiment_gate_rejects_a_checkpoint_trained_under_the_wrong_tag).
    """
    manifest_n16 = {
        name for name in reg.list_suites()
        if reg.resolve_suite(name).family_override("gr00t", "n16").get(
            "embodiment_tag") == "GR1"
    }
    assert manifest_n16, "expected at least one suite declaring n16 embodiment GR1"
    assert _train_entry_n16_suites() == manifest_n16


# --------------------------------------------------------------------------
# The coherence gate
# --------------------------------------------------------------------------

_DIG = "sha256:" + "a" * 64
_COMMIT = "b" * 40
# The REAL gr00t schemas: the end-to-end tests run the staged validate_entry,
# which loads registry.family_schemas(), so the fixture must satisfy the actual
# defaults.json contract (input_config.embodiment_tag et al), not a stub.
_FAMILY_SCHEMAS = reg.family_schemas()
_INPUT_CONFIG = {"embodiment_tag": "new_embodiment", "n_action_steps": "8",
                 "max_episode_steps": "720"}
_PROVENANCE_EXTRA = {"repo_commit": _COMMIT, "python_version": "3.12"}


def _arena_report(task: str, suite: str = "arena_gr1_fridge",
                  pipeline_mode: bool = False, num_episodes: int = 2,
                  num_envs: int = 1) -> dict:
    """A schema-v3 report shaped like the Arena eval's output.

    Otherwise fully valid, so the ONLY thing under test is the coherence gate: the
    digest matches, the episode arithmetic is consistent, the suite matches the
    pipeline expectation. ``pipeline_mode=True`` additionally gives it the shape a
    real pipeline run has (local checkpoint + populated S3 identity), which
    ``validate_entry.py`` requires.

    The rate is DERIVED from a whole number of successful episodes rather than
    hardcoded. This fixture previously declared success_rate=0.5 over a single
    episode -- half a successful episode, which cannot happen -- and the schema
    accepted it, so the fixture was itself evidence of the gap it was meant to
    exclude. The default episode count is 2 so the familiar 0.5 stays realizable.
    """
    # A whole number of successful episodes, and the rate that count implies.
    _successes = num_episodes // 2
    _rate = _successes / num_episodes
    report = {
        "schema_version": 3,
        # The source variant must MATCH THE MODE, and its fields must agree with the report. This
        # fixture carried an archive while naming an HF repo -- an archive is a mounted FineTune
        # artifact, which HF mode does not have -- and an 8-character revision that pins nothing.
        # Both passed because every field was checked only against its own shape. pipeline_mode
        # below swaps in the archive form.
        "source_snapshot": {"repo_id": "nvidia/GR00T-N1.6-3B",
                            "resolved_commit": _N16_COMMIT,
                            "tree_sha256": "a" * 64},
        "policy_type": "checkpoint",
        "model_family": "gr00t",
        "checkpoint": "nvidia/GR00T-N1.6-3B",
        "checkpoint_revision": _N16_COMMIT,
        "checkpoint_manifest": {
            "manifest_version": 1,
            "model_family": "gr00t",
            "base_checkpoint": "nvidia/GR00T-N1.6-3B",
            "base_revision": "5dc80c4a",
            "train_seed": None,
            "input_config": dict(_INPUT_CONFIG),
            "train_recipe": {"repo": "https://github.com/NVIDIA/Isaac-GR00T.git",
                             "commit": _COMMIT, "max_steps": None},
            "weights_digest": _DIG,
            "dataset_manifest": None,
        },
        "weights_digest_recomputed_by_eval": _DIG,
        "model_artifact_identity": None,
        "suite": suite,
        "task_ids": [0],
        "num_trials_per_task": num_episodes,
        "eval_seed": 100,
        "train_seed": None,
        "success_rate": _rate,
        "episodes": num_episodes,
        "episodes_reported_by_evaluator": num_episodes,
        "per_task": [
            {"task_id": 0, "task": task, "episodes": num_episodes,
             "success_rate": _rate, "successes": _successes}],
        "provenance": {"recipe_repo": "https://github.com/NVIDIA/Isaac-GR00T.git",
                       "recipe_commit": _COMMIT, "mujoco_gl": "egl",
                       **_PROVENANCE_EXTRA},
        "effective_eval_config": {
            "budget_type": "fixed_trials",
            "num_episodes": num_episodes,
            "num_envs": num_envs,
            "embodiment_tag": "new_embodiment",
            "arena_embodiment": "gr1_joint",
            "arena_object": "",
            "policy_config": "/workspace/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml",
            # I5: the CONTENT of the consumed config, not just its path -- the same path
            # can carry a different protocol between runs.
            "policy_config_digest": "sha256:" + "c" * 64,
            "gr00t_version": "n17",
        },
    }
    if pipeline_mode:
        # What a real pipeline run looks like: the checkpoint is the mounted local
        # channel and the S3 identity leg is populated. validate_entry.py always
        # runs pipeline_mode=True, so the end-to-end tests need this shape.
        report["checkpoint"] = "/opt/ml/input/data/model"
        report["checkpoint_revision"] = None
        # A mounted archive, not a snapshot: the validator binds the variant to the mode.
        del report["source_snapshot"]
        report["source_archive"] = {"sha256": "b" * 64, "size_bytes": 4096}
        report["model_artifact_identity"] = {
            "s3_uri": "s3://b/k", "bucket": "b", "key": "k",
            "version_id": "v", "etag": "e",
        }
    return report


_EXPECT = dict(expected_suite="arena_gr1_fridge", expected_trials=2,
               expected_eval_seed=100, expected_task_ids=[0],
               family_schemas=_FAMILY_SCHEMAS, pipeline_mode=False,
               expected_budget_type="steps")


def test_a_fractional_number_of_successful_episodes_is_rejected():
    """Half a successful episode cannot happen, so the schema must not accept it.

    Bounds, sums and weighted means are relations BETWEEN independently supplied
    summaries; they cannot establish that a summary could have arisen from the trials
    it claims. A ten-task report with one episode each and success_rate=0.5 everywhere
    satisfied every one of those relations and validated.
    """
    report = _arena_report("put_item_in_fridge_and_close_door", num_episodes=1)
    report["success_rate"] = 0.5
    report["per_task"] = [{"task_id": 0, "task": "put_item_in_fridge_and_close_door",
                           "episodes": 1, "success_rate": 0.5}]
    with pytest.raises(val.ReportInvalid, match="not realizable"):
        val.validate_report(report, suite_specs=reg.suites_json(),
                            **dict(_EXPECT, expected_trials=1))


def test_an_explicit_success_count_must_agree_with_the_rate():
    """When the producer states the count outright, it must match the rate."""
    report = _arena_report("put_item_in_fridge_and_close_door", num_episodes=2)
    report["per_task"][0]["successes"] = 2      # rate says 1 of 2
    with pytest.raises(val.ReportInvalid, match="disagrees with"):
        val.validate_report(report, suite_specs=reg.suites_json(), **_EXPECT)


def test_a_repeating_fraction_from_three_episodes_is_accepted():
    """One success out of three is legitimate and must not trip the realizability check.

    1/3 round-trips through JSON as 0.3333333333333333, so a naive integrality test
    would reject a perfectly valid report.
    """
    report = _arena_report("put_item_in_fridge_and_close_door", num_episodes=3)
    assert report["per_task"][0]["successes"] == 1
    rate = val.validate_report(report, suite_specs=reg.suites_json(),
                               **dict(_EXPECT, expected_trials=3))
    assert abs(rate - 1 / 3) < 1e-9


def _trained_report(**overrides):
    """A report whose checkpoint declares real training provenance (train path)."""
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    manifest = report["checkpoint_manifest"]
    manifest["train_recipe"]["max_steps"] = 200
    # The schema requires train_seed and train_recipe.max_steps to be both null (a
    # published checkpoint) or both set (one this pipeline trained), so a fine-tuned
    # manifest must carry the seed too.
    manifest["train_seed"] = 42
    report["train_seed"] = 42
    manifest["dataset_manifest"] = {
        "source": "hf:nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task",
        "revision": "sha256:" + "d" * 64,
        "episode_count": 100,
    }
    manifest.update(overrides)
    return report


_TRAIN_CONTRACT = {
    "EXPECTED_TRAIN_PATH": "train",
    "EXPECTED_TRAIN_STEPS": "200",
    "EXPECTED_TRAIN_SUITE": "arena_gr1_fridge",
    "EXPECTED_DATASET_S3URI": "__LIBERO_DEFAULT__",
    "EXPECTED_DATASET_REVISION": "a" * 40,
}


def _training_lineage():
    return {
        "schema_version": 1, "train_suite": "arena_gr1_fridge",
        "source_parameter": "__LIBERO_DEFAULT__", "revision_parameter": "a" * 40,
        "dataset": {
            "source": "hf:nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task",
            "requested_revision": "a" * 40, "resolved_revision": "a" * 40,
            "subdirectory": "ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot",
            "content_digest": "sha256:" + "d" * 64,
        },
    }


def test_training_contract_verifies_the_checkpoint_lineage(tmp_path):
    """Requested refs, resolved commits and content digests retain separate meanings."""
    result, od = _run_staged_validate(
        tmp_path, _trained_report(), "arena_gr1_fridge", extra_env=_TRAIN_CONTRACT)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    contract = json.loads((od / "validated_metrics.json").read_text())["training_contract"]
    assert contract["train_path"] == "train"
    by_field = {record["field"]: record for record in contract["fields"]}
    assert contract["matched"] == ["train_steps"]
    assert contract["verified"] == ["dataset_revision"]
    assert contract["attested"] == [
        "dataset_content_digest", "dataset_source", "dataset_subdirectory", "train_suite"]
    assert by_field["dataset_revision"]["verification_scope"] == "revision_identity_only"
    assert by_field["dataset_revision"]["observed"] == "a" * 40
    assert by_field["dataset_content_digest"]["observed"] == "sha256:" + "d" * 64
    assert by_field["dataset_content_digest"]["dataset_recomputed_by_validate"] is False
    for field in ("dataset_source", "dataset_subdirectory", "train_suite"):
        assert by_field[field]["status"] == "attested"
        assert by_field[field]["training_consumption_verified"] is False


@pytest.mark.parametrize("field,value,error", [
    ("train_suite", "arena_gr1", "train_suite mismatch"),
    ("source", "hf:another/dataset", "dataset_source mismatch"),
    ("resolved_revision", "b" * 40, "dataset_revision mismatch"),
    ("subdirectory", "different/data", "dataset_subdirectory mismatch"),
    ("content_digest", "sha256:" + "e" * 64, "dataset_content_digest mismatch"),
])
def test_staged_validate_rejects_wrong_training_lineage(tmp_path, field, value, error):
    lineage = _training_lineage()
    target = lineage if field == "train_suite" else lineage["dataset"]
    target[field] = value
    # The fixture recomputes the checkpoint digest, so these failures must come
    # from the independent training expectations, not a stale digest.
    result, output = _run_staged_validate(
        tmp_path, _trained_report(), "arena_gr1_fridge",
        extra_env=_TRAIN_CONTRACT, training_lineage=lineage)
    assert result.returncode != 0
    assert error in result.stdout + result.stderr
    assert not (output / "validated_metrics.json").exists()


def test_staged_validate_rejects_missing_training_lineage(tmp_path):
    result, output = _run_staged_validate(
        tmp_path, _trained_report(), "arena_gr1_fridge",
        extra_env=_TRAIN_CONTRACT, training_lineage=False)
    assert result.returncode != 0
    assert "requires training_lineage.json" in result.stdout + result.stderr
    assert not (output / "validated_metrics.json").exists()


def test_eval_only_explicitly_leaves_training_lineage_outside_this_execution(tmp_path):
    result, output = _run_staged_validate(
        tmp_path, _trained_report(), "arena_gr1_fridge",
        extra_env={"EXPECTED_TRAIN_PATH": "eval_only"}, training_lineage=False)
    assert result.returncode == 0, result.stdout + result.stderr
    contract = json.loads((output / "validated_metrics.json").read_text())["training_contract"]
    assert contract["train_path"] == "eval_only"
    assert contract["fields"] == contract["verified"] == contract["attested"] == []
    assert contract["reason"] == "checkpoint_training_is_outside_this_execution"


@pytest.mark.parametrize("override,expected_error", [
    ({"EXPECTED_TRAIN_STEPS": "1000"}, "train_steps mismatch"),
    ({"EXPECTED_TRAIN_PATH": ""}, "must be 'train' or 'eval_only'"),
    ({"EXPECTED_TRAIN_PATH": "maybe"}, "must be 'train' or 'eval_only'"),
    ({"EXPECTED_TRAIN_STEPS": ""}, "cannot be checked"),
    ({"EXPECTED_TRAIN_SUITE": "arena_gr1"}, "train_suite mismatch"),
    ({"EXPECTED_DATASET_S3URI": "s3://another/dataset"}, "does not support"),
    ({"EXPECTED_DATASET_REVISION": "b" * 40}, "dataset_revision_parameter mismatch"),
])
def test_training_contract_rejects_a_checkpoint_from_another_contract(
    tmp_path, override, expected_error
):
    """I4: nothing compared the checkpoint's provenance with the execution's contract.

    A checkpoint declaring a different training dose or a different dataset satisfied
    every check, and because TrainSuite is exposed independently of Suite a manually
    started execution could train on one GR1 task and evaluate another while passing
    the embodiment checks. The dataset determines which task the policy was trained
    for, which is why the dataset comparison is the one that catches it.
    """
    result, od = _run_staged_validate(
        tmp_path, _trained_report(), "arena_gr1_fridge",
        extra_env={**_TRAIN_CONTRACT, **override})
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert expected_error in out, out
    assert not (od / "validated_metrics.json").exists(), out


def test_the_receipt_carries_the_promoted_artifact(tmp_path):
    """C5: registration must point at the bytes that passed validation.

    RegisterModel used to point at the FineTune artifact's plain, unversioned URI, so the
    package resolved to whatever occupied that key at resolution time. Validate recorded
    the source identity but never promoted the object.
    """
    result, od = _run_staged_validate(
        tmp_path,
        _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True),
        "arena_gr1_fridge")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    validated = json.loads((od / "validated_metrics.json").read_text())
    promotion = validated["promotion"]
    # Content-addressed by the COMPLETE archive digest, which is distinct from the
    # weights digest (that one excludes the manifest, logs and hidden paths).
    assert promotion["archive_sha256"]
    assert f"artifacts/v1/sha256/{promotion['archive_sha256']}/model.tar.gz" \
        in promotion["model_uri"]
    assert "evidence/v1/" in promotion["attestation_uri"]
    assert promotion["attestation_sha256"]
    # Offline publication must announce itself so it cannot pass for a real promotion.
    assert promotion["local_test_publication"] is True


def test_promotion_refuses_to_register_without_a_trust_bucket(tmp_path):
    """Fail closed: registering the unpromoted source URI is the defect.

    With neither a trust bucket nor the offline publication directory, Validate must
    refuse rather than fall back to the source URI.
    """
    result, od = _run_staged_validate(
        tmp_path,
        _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True),
        "arena_gr1_fridge",
        extra_env={"_VALIDATE_PROMOTION_LOCAL_DIR": "", "TRUST_BUCKET": ""})
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert "TRUST_BUCKET is not set" in out, out
    # A failed promotion must prevent a successful receipt.
    assert not (od / "validated_metrics.json").exists(), out


def test_the_attestation_records_the_lineage_classification(tmp_path):
    """Promotion must not upgrade a self-supplied manifest into training provenance."""
    trust = tmp_path / "trust"
    result, od = _run_staged_validate(
        tmp_path,
        _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True),
        "arena_gr1_fridge")
    assert result.returncode == 0, result.stdout + result.stderr
    attestations = list(trust.rglob("validated_metrics.json"))
    assert attestations, f"no attestation published under {trust}"
    attestation = json.loads(attestations[0].read_text())
    assert attestation["attestation_version"] == 1
    # These fixtures are eval-only: bytes verified, training lineage NOT established here.
    assert attestation["lineage"]["train_path"] == "eval_only"
    assert attestation["promoted_artifact"]["archive_sha256"]
    # The checkpoint-tree digest is kept SEPARATE from the archive digest.
    assert attestation["promoted_artifact"]["weights_digest_recomputed_by_validate"]
    assert "requested" in attestation["protocol"]
    assert "consumed" in attestation["protocol"]


def test_republishing_identical_bytes_is_idempotent(tmp_path):
    """A retried Validate step must not fail on its own previous publication.

    Both runs share ONE trust destination but use separate working directories, so the
    second genuinely re-publishes to the same content-addressed key.
    """
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    shared_trust = {"_VALIDATE_PROMOTION_LOCAL_DIR": str(tmp_path / "shared-trust")}
    first_dir = tmp_path / "run1"
    first_dir.mkdir()
    second_dir = tmp_path / "run2"
    second_dir.mkdir()
    first, _ = _run_staged_validate(first_dir, report, "arena_gr1_fridge",
                                    extra_env=shared_trust)
    assert first.returncode == 0, first.stdout + first.stderr
    second, od = _run_staged_validate(second_dir, report, "arena_gr1_fridge",
                                      extra_env=shared_trust)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (od / "validated_metrics.json").exists()


def test_arena_report_on_the_declared_task_is_otherwise_valid():
    """Baseline: with the suite's own task the report validates end to end.

    Without this the rejection test below would prove nothing -- a report that
    fails for an unrelated reason would look like a working gate.
    """
    report = _arena_report("put_item_in_fridge_and_close_door")
    assert val.validate_report(
        report, suite_specs=reg.suites_json(), **_EXPECT) == 0.5


def test_coherence_gate_rejects_a_task_the_suite_does_not_declare():
    """A digest-clean report whose task LABEL is not the suite's task must fail.

    Scope, precisely: `per_task[].task` is `EVAL_SIM_CONFIG["task_name"]` echoed
    back by the evaluator (no code path sets `results["task_name"]`), so for a
    `run_arena.py` submission -- where `Suite` and `EvalSimConfig` come from the
    same resolved suite -- this gate compares the manifest against a value derived
    from it and cannot fail. What closes the old fridge/`cube_goal_pose` mismatch
    for launcher runs is the REMOVAL of `--task-name`.

    The gate is the backstop for a hand-started StartPipelineExecution, where
    `Suite` and `EvalSimConfig` are set independently, and for cross-simulator
    suite misuse. The embodiment gate below is the stronger check: it reads a value
    FineTune wrote rather than one the evaluator echoed.
    """
    with pytest.raises(val.ReportInvalid) as ei:
        val.validate_report(_arena_report("cube_goal_pose"),
                            suite_specs=reg.suites_json(), **_EXPECT)
    msg = str(ei.value)
    assert "incoherence" in msg
    assert "put_item_in_fridge_and_close_door" in msg
    assert "cube_goal_pose" in msg


def test_coherence_gate_is_skipped_without_suite_specs():
    """Offline callers that pass no suite_specs keep the previous behavior.

    In-pipeline this state is unreachable: validate_entry._require_suite_specs()
    fails closed on an empty table (see
    test_staged_validate_fails_closed_on_empty_suite_table).
    """
    assert val.validate_report(_arena_report("cube_goal_pose"), **_EXPECT) == 0.5


def test_coherence_gate_does_not_fire_on_a_non_arena_suite():
    """Negative control: a LIBERO suite has no arena block, so the gate is inert."""
    report = _arena_report("libero_spatial_0", suite="libero_spatial")
    report["task_ids"] = [0]
    assert val.validate_report(
        report, suite_specs=reg.suites_json(),
        **dict(_EXPECT, expected_suite="libero_spatial")) == 0.5


def test_experimental_suite_is_rejected_when_suite_specs_present():
    """An experimental suite cannot register even with a well-formed report."""
    report = _arena_report("galileo_g1_locomanip_pick_and_place", suite="arena_g1")
    with pytest.raises(val.ReportInvalid, match="unknown suite"):
        val.validate_report(report, suite_specs=reg.suites_json(),
                            **dict(_EXPECT, expected_suite="arena_g1"))


# --------------------------------------------------------------------------
# The embodiment coherence gate
# --------------------------------------------------------------------------

def _embodiment_expect(version: str) -> dict:
    return dict(_EXPECT, expected_family_version=version)


def test_embodiment_gate_accepts_the_declared_tag():
    """Baseline for the negative test below: n17 trains under `new_embodiment`."""
    report = _arena_report("put_item_in_fridge_and_close_door")
    assert report["checkpoint_manifest"]["input_config"]["embodiment_tag"] == \
        "new_embodiment"
    assert val.validate_report(report, suite_specs=reg.suites_json(),
                               **_embodiment_expect("n17")) == 0.5


def test_embodiment_gate_rejects_a_checkpoint_trained_under_the_wrong_tag():
    """This is the C2 scenario, now caught by the GATE rather than only by CI.

    `arena_gr1_fridge` x n16 must train under the native GR1 head (`GR1`). If the
    train entry's n16 branch stops covering this suite, FineTune trains under
    `new_embodiment` while the eval server is told `--embodiment-tag GR1`. The
    weights digest still matches (the bytes ARE the trained bytes) and the
    suite/task check still passes (the task label is right), so nothing else in
    the trust chain sees it. The value read here is written by FineTune, not
    echoed by the evaluator.
    """
    report = _arena_report("put_item_in_fridge_and_close_door")
    # What the mutated train_entry would have produced for n16.
    report["checkpoint_manifest"]["input_config"]["embodiment_tag"] = "new_embodiment"
    with pytest.raises(val.ReportInvalid) as ei:
        val.validate_report(report, suite_specs=reg.suites_json(),
                            **_embodiment_expect("n16"))
    msg = str(ei.value)
    assert "embodiment incoherence" in msg
    assert "GR1" in msg and "new_embodiment" in msg


def test_embodiment_gate_is_inert_when_the_suite_declares_no_override():
    """LIBERO suites declare embodiment_tag: null -- the family default applies."""
    report = _arena_report("libero_spatial_0", suite="libero_spatial")
    report["checkpoint_manifest"]["input_config"]["embodiment_tag"] = "LIBERO_PANDA"
    assert val.validate_report(
        report, suite_specs=reg.suites_json(),
        **dict(_EXPECT, expected_suite="libero_spatial",
               expected_family_version="n17")) == 0.5


def test_embodiment_gate_skipped_without_expected_family_version():
    """Offline callers keep the previous behavior."""
    report = _arena_report("put_item_in_fridge_and_close_door")
    report["checkpoint_manifest"]["input_config"]["embodiment_tag"] = "wrong_tag"
    assert val.validate_report(report, suite_specs=reg.suites_json(),
                               **_EXPECT) == 0.5


# --------------------------------------------------------------------------
# Container delivery
# --------------------------------------------------------------------------

def test_suites_json_is_serializable_and_complete():
    blob = reg.suites_json()
    assert set(blob) == set(reg.list_suites())
    round_tripped = json.loads(json.dumps(blob))       # must survive the b64 hop
    fridge = round_tripped["arena_gr1_fridge"]
    assert fridge["arena"]["task"] == "put_item_in_fridge_and_close_door"
    assert fridge["status"] == "supported"


def test_stage_validate_code_embeds_the_suite_table():
    """The Validate container receives the real suite table, or the gate is vacuous.

    Asserting `"VLA_SUITES_JSON" in code` would be VACUOUS: the staged artifact is
    validate_entry.py *plus* the injected bootstrap, and validate_entry.py itself
    contains `os.environ.get("VLA_SUITES_JSON", ...)`. The substring is present
    whether or not runner.py injects anything -- so the whole injection (and with
    it the coherence gate and the authoritative allowlist) could be deleted with
    the suite green. Decode the payload and pin it to the registry instead.
    """
    from vla_pipeline.runner import stage_validate_code
    with open(stage_validate_code()) as fh:
        code = fh.read()
    m = re.search(r'VLA_SUITES_JSON"\]\s*=\s*_b64\.b64decode\("([A-Za-z0-9+/=]+)"\)',
                  code)
    assert m, ("stage_validate_code() did not embed the suite table -- the Validate "
               "container would fall back to suite_specs={} and the Arena coherence "
               "gate would never fire")
    assert json.loads(base64.b64decode(m.group(1))) == reg.suites_json()


def _run_staged_validate(tmp_path, metrics: dict, suite: str,
                         family_version: str = "n17",
                         expected_eval_trials: str = "2",
                         extra_env: dict | None = None,
                         training_lineage: dict | bool | None = None):
    """Drive the STAGED validate_entry.py as a subprocess, as Validate really does.

    This is the only test that covers the whole chain -- runner.stage_validate_code
    -> base64 bootstrap -> validate_entry -> validate_report -> the gate. Every
    other coherence-gate test calls validate_report() directly and so would still
    pass if the injection were removed.
    """
    from vla_pipeline.runner import stage_validate_code
    staged = stage_validate_code(repo_root=_REPO_ROOT)
    from vla_pipeline.common.digest import weights_digest

    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"suite checkpoint fixture")
    if (metrics["checkpoint_manifest"]["train_recipe"]["max_steps"] is not None
            and training_lineage is not False):
        (checkpoint / "training_lineage.json").write_text(
            json.dumps(training_lineage if training_lineage is not None else _training_lineage()))
    digest = weights_digest(str(checkpoint))
    metrics["checkpoint_manifest"]["weights_digest"] = digest
    metrics["weights_digest_recomputed_by_eval"] = digest
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(metrics["checkpoint_manifest"]))
    channel = tmp_path / "channel"
    channel.mkdir()
    with tarfile.open(channel / "model.tar.gz", "w:gz") as archive:
        for path in sorted(checkpoint.iterdir()):
            archive.add(path, arcname=path.name)
    # schema 3: Validate measures the archive and requires it to equal what the
    # evaluation reported. A hardcoded value would prove only that the comparison
    # runs, so this records the measurement of the archive actually built above.
    metrics["source_archive"] = dict(zip(
        ("sha256", "size_bytes"),
        _measure_archive(str(channel / "model.tar.gz"))))
    with open(ed / "metrics.json", "w") as fh:
        json.dump(metrics, fh)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("EXPECTED_", "VLA_"))}
    env.update({
        "EVAL_OUTPUT_DIR": str(ed), "OUTPUT_DIR": str(od),
        "EXPECTED_EVAL_SEED": "100", "EXPECTED_TRAIN_PATH": "eval_only",
        # C5: Validate promotes the validated bytes before emitting its
        # receipt. Offline runs publish locally; the promotion record marks
        # itself local_test_publication so it cannot pass as a real one.
        "_VALIDATE_PROMOTION_LOCAL_DIR": str(tmp_path / "trust"), "EXPECTED_EVAL_TRIALS": "2",
        "EXPECTED_EVAL_TASK_IDS": "all", "EXPECTED_SUITE": suite,
        "EXPECTED_MODEL_FAMILY": "gr00t",
        "EXPECTED_FAMILY_VERSION": family_version,
        "EXPECTED_MODEL_SOURCE_URI": "s3://b/k",
        "EVALUATOR_IMAGE_URI": "123456789012.dkr.ecr.us-east-1.amazonaws.com/vla-eval@sha256:abababababababababababababababababababababababababababababababab",
        "_VALIDATE_S3_HEAD_STUB": json.dumps({
            "VersionId": "v", "ETag": '"e"',
            # The version GET measures CONTENT, so the stub carries the bytes this fixture
            # packed. Omitting it would fail the comparison for the wrong reason.
            "BodyPath": str(channel / "model.tar.gz"),
        }),
        "CHECKPOINT_DIR": str(channel),
    })
    if "arena" in suite:
        env.update({
            "EXPECTED_EVAL_TRIALS": expected_eval_trials,
            "EXPECTED_EMBODIMENT_TAG": "new_embodiment" if family_version == "n17" else "GR1",
            "EXPECTED_ARENA_EMBODIMENT": "gr1_joint",
            "EXPECTED_ARENA_OBJECT": "NONE",
            "EXPECTED_POLICY_CONFIG": "/workspace/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml",
        })
    if extra_env:
        # Applied last so a test can override any expectation, including removing one
        # by setting it empty.
        env.update(extra_env)
    return subprocess.run([sys.executable, staged], env=env,
                          capture_output=True, text=True, check=False), od


def test_end_to_end_staged_validate_rejects_incoherent_task(tmp_path):
    """The gate fires through the REAL delivery path, not just an in-process call."""
    # No policy_type / eval_trials keys: validate_entry reads both with .get()
    # fallbacks (num_trials_per_task covers trials), and the strict schema rejects
    # unknown top-level keys.
    result, od = _run_staged_validate(
        tmp_path, _arena_report("cube_goal_pose", pipeline_mode=True),
        "arena_gr1_fridge")
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert "incoherence" in out, out
    # A failed validation must not emit the artifact the gate reads.
    assert not (od / "validated_metrics.json").exists()


def test_end_to_end_staged_validate_accepts_coherent_task(tmp_path):
    """Positive control: same report, suite's own task -> passes and emits output.

    Without this the rejection above could be passing for an unrelated reason.
    """
    result, od = _run_staged_validate(
        tmp_path,
        _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True),
        "arena_gr1_fridge")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert (od / "validated_metrics.json").exists(), out
    # I9: eval_backend was read from the report, but the schema's strict key check
    # forbids that field, so the read returned "" on every valid run and the validated
    # artifact advertised an empty backend. It is now derived from the suite's declared
    # simulator, so consumers can actually filter and compare on it.
    validated = json.loads((od / "validated_metrics.json").read_text())
    assert validated["eval_backend"] == "isaac_arena", validated["eval_backend"]


@pytest.mark.parametrize("dropped_key", [
    "EXPECTED_EMBODIMENT_TAG",
    "EXPECTED_ARENA_EMBODIMENT", "EXPECTED_ARENA_OBJECT",
    "EXPECTED_POLICY_CONFIG",
])
def test_arena_missing_expected_value_rejected(tmp_path, dropped_key):
    """Each missing Arena expected value must fail the staged validator."""
    from vla_pipeline.common.digest import weights_digest
    from vla_pipeline.runner import stage_validate_code
    staged = stage_validate_code(repo_root=_REPO_ROOT)
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"arena ckpt fixture")
    digest = weights_digest(str(checkpoint))
    report["checkpoint_manifest"]["weights_digest"] = digest
    report["weights_digest_recomputed_by_eval"] = digest
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(report["checkpoint_manifest"]))
    channel = tmp_path / "channel"
    channel.mkdir()
    with tarfile.open(channel / "model.tar.gz", "w:gz") as archive:
        for p in sorted(checkpoint.iterdir()):
            archive.add(p, arcname=p.name)
    with open(ed / "metrics.json", "w") as fh:
        json.dump(report, fh)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("EXPECTED_", "VLA_"))}
    env.update({
        "EVAL_OUTPUT_DIR": str(ed), "OUTPUT_DIR": str(od),
        "EXPECTED_EVAL_SEED": "100", "EXPECTED_TRAIN_PATH": "eval_only",
        # C5: Validate promotes the validated bytes before emitting its
        # receipt. Offline runs publish locally; the promotion record marks
        # itself local_test_publication so it cannot pass as a real one.
        "_VALIDATE_PROMOTION_LOCAL_DIR": str(tmp_path / "trust"), "EXPECTED_EVAL_TRIALS": "2",
        "EXPECTED_EVAL_TASK_IDS": "all", "EXPECTED_SUITE": "arena_gr1_fridge",
        "EXPECTED_MODEL_FAMILY": "gr00t",
        "EXPECTED_FAMILY_VERSION": "n17",
        "EXPECTED_MODEL_SOURCE_URI": "s3://b/k",
        "EVALUATOR_IMAGE_URI": "123456789012.dkr.ecr.us-east-1.amazonaws.com/vla-eval@sha256:abababababababababababababababababababababababababababababababab",
        "_VALIDATE_S3_HEAD_STUB": json.dumps(
            {"VersionId": "v", "ETag": '"e"',
             "BodyPath": str(channel / "model.tar.gz")}),
        "CHECKPOINT_DIR": str(channel),
        "EXPECTED_EMBODIMENT_TAG": "new_embodiment",
        "EXPECTED_ARENA_EMBODIMENT": "gr1_joint",
        "EXPECTED_ARENA_OBJECT": "NONE",
        "EXPECTED_POLICY_CONFIG": "/workspace/isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml",
    })
    env.pop(dropped_key, None)
    result = subprocess.run([sys.executable, staged], env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0, f"should fail when {dropped_key} is empty"
    assert "empty" in (result.stdout + result.stderr).lower()


def test_arena_multi_episode_budget_passes_staged(tmp_path):
    """A larger requested sample passes when the observed count matches it."""
    result, od = _run_staged_validate(
        tmp_path,
        _arena_report("put_item_in_fridge_and_close_door",
                      pipeline_mode=True, num_episodes=3),
        "arena_gr1_fridge", expected_eval_trials="3")
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert (od / "validated_metrics.json").exists(), out


@pytest.mark.parametrize("field,bad_value", [
    ("num_steps", 999),
    ("embodiment_tag", "WRONG_TAG"),
    ("arena_embodiment", "wrong_emb"),
    ("policy_config", "/wrong/path.yaml"),
    ("gr00t_version", "n99"),
])
def test_arena_mismatched_reported_field_rejected(tmp_path, field, bad_value):
    """Each individually mismatched Arena reported field must fail validation."""
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    report["effective_eval_config"][field] = bad_value
    result, od = _run_staged_validate(tmp_path, report, "arena_gr1_fridge")
    assert result.returncode != 0, f"should fail on mismatched {field}={bad_value}"
    assert not (od / "validated_metrics.json").exists()


def test_arena_missing_reported_field_rejected(tmp_path):
    """A report missing a required Arena field must fail validation."""
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    del report["effective_eval_config"]["embodiment_tag"]
    result, od = _run_staged_validate(tmp_path, report, "arena_gr1_fridge")
    assert result.returncode != 0
    assert "missing required Arena fields" in (result.stdout + result.stderr)
    assert not (od / "validated_metrics.json").exists()


@pytest.mark.parametrize("field", [
    "budget_type", "num_episodes", "num_envs", "embodiment_tag",
    "arena_embodiment", "policy_config", "gr00t_version", "arena_object",
])
def test_arena_null_reported_field_rejected(tmp_path, field):
    """A null/None value for any required Arena field must fail validation."""
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    report["effective_eval_config"][field] = None
    result, od = _run_staged_validate(tmp_path, report, "arena_gr1_fridge")
    assert result.returncode != 0, f"null {field} should fail"
    assert not (od / "validated_metrics.json").exists()


def test_arena_episode_count_must_match_requested_trials(tmp_path):
    """The evaluated sample must equal the requested one. An identical report is
    accepted when the expectation matches and rejected when it does not -- this is
    the gate Arena previously disabled by declaring itself step-budgeted."""
    pass_dir = tmp_path / "pass_case"
    pass_dir.mkdir()
    report = _arena_report("put_item_in_fridge_and_close_door",
                          pipeline_mode=True, num_episodes=3)
    result_pass, od_pass = _run_staged_validate(
        pass_dir, report, "arena_gr1_fridge", expected_eval_trials="3")
    assert result_pass.returncode == 0
    assert (od_pass / "validated_metrics.json").exists()

    fail_dir = tmp_path / "fail_case"
    fail_dir.mkdir()
    report2 = _arena_report("put_item_in_fridge_and_close_door",
                           pipeline_mode=True, num_episodes=3)
    result_fail, od_fail = _run_staged_validate(
        fail_dir, report2, "arena_gr1_fridge", expected_eval_trials="5")
    assert result_fail.returncode != 0
    assert not (od_fail / "validated_metrics.json").exists()


def test_staged_validate_fails_closed_if_the_suite_table_injection_is_lost(tmp_path):
    """If runner.py ever stops embedding the suite table, Validate must FAIL.

    This is the threat model behind the payload assertion above. `{}` is falsy, so
    without the guard validate_report reverts the allowlist to its literal fallback
    AND skips both coherence gates -- and the step then prints "PASSED". A degraded
    gate must never look like a passing one.

    The staged artifact sets VLA_SUITES_JSON itself, so an env override cannot
    reach the degraded state; strip the injected line to reproduce it exactly.
    """
    from vla_pipeline.runner import stage_validate_code
    with open(stage_validate_code(repo_root=_REPO_ROOT)) as fh:
        code = fh.read()
    stripped = re.sub(r'^_os\.environ\["VLA_SUITES_JSON"\].*\n', "", code,
                      flags=re.MULTILINE)
    assert stripped != code, "expected to strip the VLA_SUITES_JSON injection"
    mutated = tmp_path / "validate_entry_no_suites.py"
    mutated.write_text(stripped)

    ed, od = tmp_path / "eval", tmp_path / "out"
    ed.mkdir()
    od.mkdir()
    with open(ed / "metrics.json", "w") as fh:
        json.dump(_arena_report("cube_goal_pose", pipeline_mode=True), fh)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("EXPECTED_", "VLA_"))}
    env.update({
        "EVAL_OUTPUT_DIR": str(ed), "OUTPUT_DIR": str(od),
        "EXPECTED_EVAL_SEED": "100", "EXPECTED_TRAIN_PATH": "eval_only",
        # C5: Validate promotes the validated bytes before emitting its
        # receipt. Offline runs publish locally; the promotion record marks
        # itself local_test_publication so it cannot pass as a real one.
        "_VALIDATE_PROMOTION_LOCAL_DIR": str(tmp_path / "trust"), "EXPECTED_EVAL_TRIALS": "2",
        "EXPECTED_EVAL_TASK_IDS": "all", "EXPECTED_SUITE": "arena_gr1_fridge",
        "EXPECTED_MODEL_FAMILY": "gr00t",
        "EXPECTED_FAMILY_VERSION": "n17",
        "EXPECTED_MODEL_SOURCE_URI": "s3://b/k",
        "EVALUATOR_IMAGE_URI": "123456789012.dkr.ecr.us-east-1.amazonaws.com/vla-eval@sha256:abababababababababababababababababababababababababababababababab",
        "_VALIDATE_S3_HEAD_STUB": json.dumps({
            "VersionId": "v", "ETag": '"e"',
            # The comment above about content applies where the archive check is
            # reached. This test fails closed on the missing suite table long before
            # that, and has no archive, so no body is supplied.
        }),
    })
    r = subprocess.run([sys.executable, str(mutated)], env=env,
                       capture_output=True, text=True, check=False)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "VLA_SUITES_JSON is empty" in out, out
    assert "validation PASSED" not in out, out


# --------------------------------------------------------------------------
# Gate DELIVERY: the plumbing, not just the gate function (I18)
# --------------------------------------------------------------------------

def _offline_cfg():
    from vla_pipeline.config import PipelineConfig
    return PipelineConfig(account_id="000000000000", region="us-east-1",
                          training_role_arn="arn:aws:iam::000000000000:role/train", workload_role_arn="arn:aws:iam::000000000000:role/wl", validation_role_arn="arn:aws:iam::000000000000:role/val", trust_bucket="trust-bucket", handoff_bucket="handoff-bucket",
                          role_arn="arn:aws:iam::000000000000:role/r", bucket="b")


def test_validate_step_receives_the_family_version_on_both_variants():
    """The embodiment gate's own delivery had zero coverage.

    Deleting EXPECTED_FAMILY_VERSION from pipeline.py's Validate env left the whole
    suite green, and the obvious "fix" for the resulting loud runtime break --
    relaxing validate_entry's _require_expectation to .get() -- would silently kill
    the gate. Both pipeline variants must carry it.
    """
    from vla_pipeline.pipeline import build_pipeline
    for kwargs in ({}, {"checkpoint_s3_uri": "s3://b/k/model.tar.gz"}):
        definition = build_pipeline(_offline_cfg(), **kwargs).definition()
        assert "EXPECTED_FAMILY_VERSION" in definition, kwargs
        assert "Gr00tVersion" in definition, kwargs


def test_gr00t_version_parameter_is_enum_constrained():
    """A hand-started execution must not be able to pass a value no suite keys.

    An unconstrained ParameterString reaching Validate is what let the embodiment
    gate silently no-op (I17); rejecting it at StartPipelineExecution closes the
    same hole one layer earlier.
    """
    from vla_pipeline.pipeline import build_parameters
    param = build_parameters()["gr00t_version"]
    assert sorted(param.enum_values) == ["n16", "n17"]


# --------------------------------------------------------------------------
# I17: the embodiment gate must not fail open on an odd family version
# --------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["n16", "N16", " n16 ", "N16 "])
def test_embodiment_gate_normalises_the_family_version(version):
    """Every other consumer .strip().lower()s this; the validator must too.

    Without normalisation `"N16"` silently skipped the check while the rest of the
    stack happily ran the n16 route -- an inert gate on the exact hand-started path
    the gate exists for.
    """
    report = _arena_report("put_item_in_fridge_and_close_door")
    report["checkpoint_manifest"]["input_config"]["embodiment_tag"] = "new_embodiment"
    with pytest.raises(val.ReportInvalid, match="embodiment incoherence"):
        val.validate_report(report, suite_specs=reg.suites_json(),
                            **_embodiment_expect(version))


def test_embodiment_gate_rejects_an_unknown_family_version():
    """Fail CLOSED: an unkeyed version means the gate cannot be evaluated."""
    report = _arena_report("put_item_in_fridge_and_close_door")
    with pytest.raises(val.ReportInvalid, match="unknown family version"):
        val.validate_report(report, suite_specs=reg.suites_json(),
                            **_embodiment_expect("n99"))


def test_embodiment_gate_error_names_the_remedy():
    """S16: the message must tell an eval-only caller how to fix it."""
    report = _arena_report("put_item_in_fridge_and_close_door")
    report["checkpoint_manifest"]["input_config"]["embodiment_tag"] = "GR1"
    with pytest.raises(val.ReportInvalid, match="--gr00t-version"):
        val.validate_report(report, suite_specs=reg.suites_json(),
                            **_embodiment_expect("n17"))


# --------------------------------------------------------------------------
# S7: the positive-control report must never be registrable
# --------------------------------------------------------------------------

def _posctrl_expect():
    return dict(_EXPECT, pipeline_mode=True, expected_family_version="n16")


def test_positive_control_report_is_rejected_for_its_extra_keys():
    """A posctrl report carries keys schema-v3 does not allow.

    `eval_entry.py` adds `positive_control` / `positive_control_repo`, which are not
    in TOP_KEYS -- so the strict validator rejects the report on its shape alone.
    Asserted SEPARATELY from the trust-chain reason below: the previous version of
    this test used a bare `pytest.raises(ReportInvalid)` and passed on THIS reason
    while its docstring claimed the digest one. An unqualified `raises` cannot tell
    you which guard fired.
    """
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    report["checkpoint_manifest"] = None
    report["positive_control"] = True
    report["positive_control_repo"] = "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task"
    with pytest.raises(val.ReportInvalid, match="unknown keys"):
        val.validate_report(report, suite_specs=reg.suites_json(), **_posctrl_expect())


def test_a_manifestless_report_is_not_registrable():
    """The trust-chain reason, isolated: no checkpoint_manifest -> no digest chain.

    `--posctrl-n16` deliberately bypasses the train->eval trust chain (it serves
    NVIDIA's published checkpoint), so it emits `checkpoint_manifest: None`. Even
    with the extra keys stripped -- i.e. even if a future change made the posctrl
    report schema-clean -- it must still be unregistrable, because there is nothing
    to verify the evaluated bytes against.

    The path is only reachable from submit_simeval.py, which runs a standalone job
    with no Validate step, so there is no live risk today; this pins the boundary so
    routing it through the pipeline cannot quietly register it.
    """
    report = _arena_report("put_item_in_fridge_and_close_door", pipeline_mode=True)
    report["checkpoint_manifest"] = None
    with pytest.raises(val.ReportInvalid, match="checkpoint_manifest|missing keys"):
        val.validate_report(report, suite_specs=reg.suites_json(), **_posctrl_expect())
