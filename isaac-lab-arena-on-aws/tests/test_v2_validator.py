"""Unit tests for the shared schema-v2 validator (models/common/validator.py)."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vla_pipeline", "common"))
from validator import ReportInvalid, parse_report, validate_report  # noqa: E402

FAMILY_SCHEMAS = {
    "molmoact2": {
        "input_config_schema": {
            "norm_tag": {"type": "str", "required": True, "enum": ["libero"]},
            "model_dtype": {"type": "str", "required": True},
            "inference_action_mode": {"type": "str", "required": False},
        },
        "provenance_keys": {
            "lerobot_branch": {"type": "str", "required": False},
        },
    },
    "openvla": {
        "input_config_schema": {
            "num_images_in_input": {"type": "int", "required": True},
            "use_proprio": {"type": "bool", "required": True},
            "unnorm_key": {"type": "str", "required": True},
        },
        "provenance_keys": {},
    },
}

DIG = "sha256:" + "a" * 64
COMMIT = "b" * 40


def pipeline_report() -> dict:
    """A coherent PIPELINE-mode report: local mounted checkpoint + measured archive.

    good_report() previously served both modes while carrying an archive and an HF repo id, which is
    a contradiction -- an archive is a mounted FineTune artifact and HF mode has none. The validator
    now binds the variant to the mode, so the two cases need separate coherent fixtures rather than
    one that is wrong for whichever mode it is not being used in.
    """
    r = good_report()
    del r["source_snapshot"]
    r["source_archive"] = {"sha256": "b" * 64, "size_bytes": 4096}
    r["checkpoint"] = "/opt/ml/input/data/model"      # local: pipeline mode requires it
    r["checkpoint_revision"] = None                   # a local artifact has no upstream revision
    # Pipeline mode requires the S3 identity leg; HF mode leaves it null, so it must be added here
    # rather than inherited.
    r["model_artifact_identity"] = {
        "s3_uri": "s3://b/k", "bucket": "b", "key": "k", "version_id": "v", "etag": "e",
    }
    return r


def good_report() -> dict:
    return {
        "schema_version": 3,
        # A COHERENT HF-mode report. This carried a source_archive while naming an HF repo as its
        # checkpoint -- an archive is a mounted FineTune artifact, which HF mode does not have -- and
        # an 8-character checkpoint_revision that pins nothing. Both passed because each field was
        # checked only for its own shape, never against the others. The snapshot now agrees with the
        # report: repo_id IS the checkpoint, resolved_commit IS checkpoint_revision, and tree_sha256
        # is the bare form of the digest the evaluator recomputed.
        "source_snapshot": {"repo_id": "allenai/MolmoAct2-LIBERO",
                            "resolved_commit": "0d24a92b" + "0" * 32,
                            "tree_sha256": DIG.split(":", 1)[1]},
        "policy_type": "checkpoint",
        "model_family": "molmoact2",
        "checkpoint": "allenai/MolmoAct2-LIBERO",
        "checkpoint_revision": "0d24a92b" + "0" * 32,
        "checkpoint_manifest": {
            "manifest_version": 1,
            "model_family": "molmoact2",
            "base_checkpoint": "allenai/MolmoAct2-LIBERO",
            "base_revision": "0d24a92b",
            "train_seed": None,
            "input_config": {"norm_tag": "libero", "model_dtype": "float32"},
            "train_recipe": {"repo": "https://github.com/allenai/lerobot.git",
                             "commit": COMMIT, "max_steps": None},
            "weights_digest": DIG,
            "dataset_manifest": None,
        },
        "weights_digest_recomputed_by_eval": DIG,
        "model_artifact_identity": None,
        "suite": "libero_goal",
        "task_ids": [0, 1],
        "num_trials_per_task": 5,
        "eval_seed": 1000,
        "train_seed": None,
        "success_rate": 0.9,
        "episodes": 10,
        "episodes_reported_by_evaluator": 10,
        "per_task": [
            {"task_id": 0, "task": "libero_goal_0", "episodes": 5, "success_rate": 1.0},
            {"task_id": 1, "task": "libero_goal_1", "episodes": 5, "success_rate": 0.8},
        ],
        "provenance": {"recipe_repo": "https://github.com/allenai/lerobot.git",
                       "recipe_commit": COMMIT, "mujoco_gl": "egl"},
    }


EXPECT = dict(expected_suite="libero_goal", expected_trials=5,
              expected_eval_seed=1000, expected_task_ids=[0, 1],
              family_schemas=FAMILY_SCHEMAS, pipeline_mode=False,
              expected_budget_type="fixed_trials")


def test_good_report_passes():
    assert validate_report(good_report(), **EXPECT) == 0.9


def _expect_fail(report, match, expect_overrides=None):
    kw = dict(EXPECT)
    if expect_overrides:
        kw.update(expect_overrides)
    with pytest.raises(ReportInvalid, match=match):
        validate_report(report, **kw)


def test_task_subset_bypass_rejected():
    # Report claims tasks [0,1] but the pipeline expected the full suite.
    _expect_fail(good_report(), "task-set mismatch",
                 {"expected_task_ids": list(range(10))})


def test_task_superset_rejected():
    r = good_report()
    _expect_fail(r, "task-set mismatch", {"expected_task_ids": [0, 1, 2]})


def test_duplicate_per_task_rejected():
    r = good_report()
    r["per_task"][1]["task_id"] = 0
    _expect_fail(r, "duplicate task_ids")


def test_substituted_per_task_rejected():
    r = good_report()
    r["per_task"][1]["task_id"] = 7
    _expect_fail(r, "per_task task set")


def test_digest_mismatch_rejected():
    r = good_report()
    r["weights_digest_recomputed_by_eval"] = "sha256:" + "c" * 64
    _expect_fail(r, "content binding failed")


def test_episodes_from_config_not_evaluator_rejected():
    r = good_report()
    r["episodes_reported_by_evaluator"] = 8
    _expect_fail(r, "!= episodes_reported_by_evaluator")


def test_weighted_mean_mismatch_rejected():
    r = good_report()
    r["success_rate"] = 0.95
    _expect_fail(r, "weighted mean")


def test_unknown_top_key_rejected():
    r = good_report()
    r["bonus"] = 1
    _expect_fail(r, "unknown keys")


def test_unknown_nested_key_rejected():
    r = good_report()
    r["checkpoint_manifest"]["extra"] = 1
    _expect_fail(r, "unknown keys")


def test_bool_not_int():
    r = good_report()
    r["num_trials_per_task"] = True
    _expect_fail(r, "num_trials_per_task invalid")


def test_nan_rejected_at_parse():
    with pytest.raises(ReportInvalid, match="non-finite"):
        parse_report('{"success_rate": NaN}')


def test_conditional_nulls_travel_together():
    r = good_report()
    r["checkpoint_manifest"]["train_seed"] = 42  # max_steps still null
    _expect_fail(r, "both be null")


def test_seed_mismatch_rejected():
    _expect_fail(good_report(), "eval_seed mismatch", {"expected_eval_seed": 100})


def test_trials_mismatch_rejected():
    _expect_fail(good_report(), "trials mismatch", {"expected_trials": 20})


def test_suite_mismatch_rejected():
    _expect_fail(good_report(), "suite mismatch", {"expected_suite": "libero_spatial"})


def test_input_config_enum_enforced():
    r = good_report()
    r["checkpoint_manifest"]["input_config"]["norm_tag"] = "wrong"
    _expect_fail(r, "not in enum")


def test_input_config_unknown_key_rejected():
    r = good_report()
    r["checkpoint_manifest"]["input_config"]["mystery"] = 1
    _expect_fail(r, "unknown keys")


def test_provenance_extra_must_be_enumerated():
    r = good_report()
    r["provenance"]["unlisted"] = "x"
    _expect_fail(r, "unknown keys")


def test_provenance_enumerated_extra_ok():
    r = good_report()
    r["provenance"]["lerobot_branch"] = "molmoact2-policy"
    assert validate_report(r, **EXPECT) == 0.9


def test_pipeline_mode_rejects_hf_checkpoint():
    # Mode/locality coupling (coherence): pipeline mode must consume the
    # mounted FineTune artifact, never an HF repo id.
    r = good_report()
    _expect_fail(r, "requires a local checkpoint", {"pipeline_mode": True})


def test_hf_mode_rejects_local_checkpoint():
    r = good_report()
    r["checkpoint"] = "/opt/ml/input/data/model"
    _expect_fail(r, "forbids a local checkpoint")


def test_pipeline_mode_requires_identity():
    r = pipeline_report()
    r["model_artifact_identity"] = None      # the fixture supplies one; this test removes it
    _expect_fail(r, "not an object", {"pipeline_mode": True})


def test_train_seed_report_manifest_disagreement():
    r = good_report()
    r["checkpoint_manifest"]["train_seed"] = 42
    r["checkpoint_manifest"]["train_recipe"]["max_steps"] = 4000
    r["checkpoint_manifest"]["dataset_manifest"] = {
        "source": "s", "revision": "r", "episode_count": 1}
    _expect_fail(r, "report.train_seed != manifest.train_seed")


def test_dataset_manifest_null_coupling():
    # Published ckpt (train_seed null) with a NON-null dataset_manifest: fatal.
    r = good_report()
    r["checkpoint_manifest"]["dataset_manifest"] = {
        "source": "s", "revision": "r", "episode_count": 1}
    _expect_fail(r, "dataset_manifest must be null iff")


def _finetuned_report(dataset_manifest):
    """A report for a checkpoint this pipeline trained (train_seed and dataset present).

    dataset_manifest is null exactly for published checkpoints, travelling with
    train_seed, so both must be set together to exercise dataset provenance.
    """
    r = good_report()
    r["train_seed"] = 42
    r["checkpoint_manifest"]["train_seed"] = 42
    r["checkpoint_manifest"]["train_recipe"]["max_steps"] = 200
    r["checkpoint_manifest"]["dataset_manifest"] = dataset_manifest
    return r


def test_recorded_episode_selection_is_accepted():
    """I12: training used a PREFIX subset while the manifest reported the full count.

    Recording only episode_count meant a run over the first 5 episodes of a
    1693-episode dataset carried the same provenance as a run over all of them, and at
    smaller doses changing the step count silently changed the dataset too.
    """
    r = _finetuned_report({"source": "s3://b/d", "revision": "rev1",
                           "episode_count": 1693, "episodes_selected": 100,
                           "episode_selection": "prefix: range(...)"})
    assert validate_report(r, **EXPECT) == 0.9


def test_a_selection_larger_than_the_dataset_is_rejected():
    r = _finetuned_report({"source": "s3://b/d", "revision": "rev1",
                           "episode_count": 10, "episodes_selected": 50,
                           "episode_selection": "prefix: range(...)"})
    _expect_fail(r, "exceeds episode_count")


def test_a_selection_count_without_its_rule_is_rejected():
    """A count with no rule cannot be reproduced or audited."""
    r = _finetuned_report({"source": "s3://b/d", "revision": "rev1",
                           "episode_count": 1693, "episodes_selected": 100})
    _expect_fail(r, "without episode_selection")


def test_a_selection_rule_without_its_count_is_rejected():
    r = _finetuned_report({"source": "s3://b/d", "revision": "rev1",
                           "episode_count": 1693,
                           "episode_selection": "prefix: range(...)"})
    _expect_fail(r, "without episodes_selected")


@pytest.mark.parametrize("bad", [0, -1, "100", 1.5, None])
def test_an_invalid_selection_count_is_rejected(bad):
    r = _finetuned_report({"source": "s3://b/d", "revision": "rev1",
                           "episode_count": 1693, "episodes_selected": bad,
                           "episode_selection": "prefix: range(...)"})
    _expect_fail(r, "episodes_selected invalid")


@pytest.mark.parametrize("field", [
    "model_dtype", "inference_action_mode", "camera_name_mapping",
])
@pytest.mark.parametrize("bad", ["", "   ", "float16", "{}"])
def test_input_config_settings_must_be_supported_values(field, bad):
    """I14: these were unconstrained strings, so an empty value bypassed every check.

    The evaluator hardcodes float32, continuous action mode and its own camera map, so a
    manifest that declares an empty or different value describes an evaluation that will
    not happen. The schema now pins each to the single supported value.

    This loads the REAL family schema from the shipped defaults.json rather than the
    hand-rolled FAMILY_SCHEMAS fixture above: a test carrying its own copy of the schema
    would keep passing with the shipped one unconstrained, which is exactly the gap
    being closed.
    """
    import json
    import pathlib
    defaults = json.loads(
        (pathlib.Path(__file__).resolve().parents[1]
         / "entrypoints/train/molmoact2/defaults.json").read_text())
    schemas = {"molmoact2": {
        "input_config_schema": defaults["input_config_schema"],
        "provenance_keys": defaults.get("provenance_keys", {}),
    }}
    r = good_report()
    r["model_family"] = "molmoact2"
    r["checkpoint_manifest"]["model_family"] = "molmoact2"
    r["checkpoint_manifest"]["input_config"] = {
        "norm_tag": "libero", "model_dtype": "float32",
        "inference_action_mode": "continuous",
        "camera_name_mapping": ('{"agentview_image":"image",'
                                '"robot0_eye_in_hand_image":"wrist_image"}'),
    }
    r["checkpoint_manifest"]["input_config"][field] = bad
    _expect_fail(r, "not in enum", {"family_schemas": schemas})


def test_the_shipped_molmoact2_schema_constrains_every_inference_setting():
    """Guards the fix itself: an unconstrained string field would reopen I14."""
    import json
    import pathlib
    schema = json.loads(
        (pathlib.Path(__file__).resolve().parents[1]
         / "entrypoints/train/molmoact2/defaults.json").read_text()
    )["input_config_schema"]
    for field in ("norm_tag", "model_dtype", "inference_action_mode",
                  "camera_name_mapping"):
        assert schema[field].get("enum"), f"{field} is not constrained to known values"


def test_source_archive_must_be_a_real_measurement():
    """schema 3 replaced checksum_sha256 with a measured archive.

    S3 returned checksum_sha256 only when the object was uploaded with a checksum algorithm, so
    it was absent in practice while its presence in the key set implied a guarantee the identity
    never carried. The archive measurement is evidence the producer actually takes.
    """
    for bad in ({"sha256": "x" * 64, "size_bytes": 1},          # not hex
                {"sha256": "A" * 64, "size_bytes": 1},          # not lowercase
                {"sha256": "a" * 63, "size_bytes": 1},          # wrong length
                {"sha256": "a" * 64, "size_bytes": 0},          # no bytes
                {"sha256": "a" * 64, "size_bytes": -1},
                {"sha256": "a" * 64, "size_bytes": True},       # bool is not a size
                {"sha256": "a" * 64},                           # incomplete
                {"sha256": "a" * 64, "size_bytes": 1, "extra": 1}):
        r = good_report()
        r["source_archive"] = bad
        _expect_fail(r, "source_archive")
    r = good_report()
    r["source_archive"] = "not an object"
    _expect_fail(r, "source_archive")

def test_required_provenance_extra_enforced_when_absent():
    # Review HIGH: a family-declared required provenance key must fail even
    # when the report ships zero extras.
    schemas = {
        "molmoact2": {
            "input_config_schema": FAMILY_SCHEMAS["molmoact2"]["input_config_schema"],
            "provenance_keys": {
                "lerobot_branch": {"type": "str", "required": True},
            },
        },
    }
    _expect_fail(good_report(), "missing required key lerobot_branch",
                 {"family_schemas": schemas})


def test_exponent_overflow_rejected_at_parse():
    with pytest.raises(ReportInvalid, match="overflows"):
        parse_report('{"x": 1e400}')


def test_pipeline_mode_identity_validated():
    r = pipeline_report()
    r["model_artifact_identity"] = {
        "s3_uri": "s3://b/k", "bucket": "b", "key": "k", "version_id": "v", "etag": "e",
    }
    assert validate_report(r, **dict(EXPECT, pipeline_mode=True)) == 0.9


@pytest.mark.parametrize("marker", ["null", "NULL", " null ", "None"])
def test_pipeline_mode_rejects_an_unversioned_bucket_marker(marker):
    """S3 returns the literal string "null" for an object in an unversioned bucket.

    A nonempty-string check accepts it, so the schema recorded a value that provides
    none of the immutable version selection it claims to require: "null" cannot
    retrieve one specific generation of the bytes.
    """
    r = pipeline_report()
    r["model_artifact_identity"] = {
        "s3_uri": "s3://b/k", "bucket": "b", "key": "k", "version_id": marker, "etag": "e",
    }
    with pytest.raises(ReportInvalid, match="UNVERSIONED"):
        validate_report(r, **dict(EXPECT, pipeline_mode=True))


def test_hf_mode_identity_must_be_null():
    r = good_report()
    r["model_artifact_identity"] = {"s3_uri": "s3://b/k", "bucket": "b", "key": "k",
                                    "version_id": "v", "etag": "e"}
    _expect_fail(r, "must be null")


def test_revision_null_for_nonlocal_rejected():
    r = good_report()
    r["checkpoint_revision"] = None
    _expect_fail(r, "checkpoint_revision null")


def test_task_ids_null_fatal():
    # Review HIGH: null must not be canonicalized into an assumed full suite.
    r = good_report()
    r["task_ids"] = None
    _expect_fail(r, "task_ids malformed")


def test_full_suite_explicit_ok():
    r = good_report()
    r["task_ids"] = list(range(10))
    r["episodes"] = 50
    r["episodes_reported_by_evaluator"] = 50
    r["num_trials_per_task"] = 5
    r["per_task"] = [
        {"task_id": i, "task": f"libero_goal_{i}", "episodes": 5, "success_rate": 1.0}
        for i in range(10)
    ]
    r["success_rate"] = 1.0
    assert validate_report(r, **dict(EXPECT, expected_task_ids=list(range(10)))) == 1.0


def test_openvla_family_schema():
    r = good_report()
    r["model_family"] = "openvla"
    r["checkpoint_manifest"]["model_family"] = "openvla"
    r["checkpoint_manifest"]["train_seed"] = 42
    r["train_seed"] = 42
    r["checkpoint_manifest"]["train_recipe"]["max_steps"] = 4000
    r["checkpoint_manifest"]["dataset_manifest"] = {
        "source": "hf:openvla/modified_libero_rlds",
        "revision": "abc", "episode_count": 432}
    r["checkpoint_manifest"]["input_config"] = {
        "num_images_in_input": 1, "use_proprio": False, "unnorm_key": "libero_spatial"}
    assert validate_report(r, **EXPECT) == 0.9


def test_json_roundtrip_via_parse_report():
    raw = json.dumps(good_report())
    assert validate_report(parse_report(raw), **EXPECT) == 0.9


def test_schema_version_float_rejected():
    r = good_report()
    r["schema_version"] = 2.0
    _expect_fail(r, "schema_version")


def test_manifest_version_bool_rejected():
    r = good_report()
    r["checkpoint_manifest"]["manifest_version"] = True
    _expect_fail(r, "manifest_version")


def test_huge_int_in_family_float_rejected_cleanly():
    schemas = {"molmoact2": {
        "input_config_schema": dict(
            FAMILY_SCHEMAS["molmoact2"]["input_config_schema"],
            temperature={"type": "float", "required": False}),
        "provenance_keys": {}}}
    r = good_report()
    r["checkpoint_manifest"]["input_config"]["temperature"] = 10 ** 400
    _expect_fail(r, "expected float", {"family_schemas": schemas})


def test_relative_local_path_rejected_in_hf_mode():
    r = good_report()
    r["checkpoint"] = "./model"
    _expect_fail(r, "forbids a local checkpoint")


def test_valid_checksum_accepted():
    import base64
    r = pipeline_report()
    r["model_artifact_identity"] = {
        "s3_uri": "s3://b/k", "bucket": "b", "key": "k", "version_id": "v",
        "etag": "e",
    }
    assert validate_report(r, **dict(EXPECT, pipeline_mode=True)) == 0.9


def test_null_dataset_manifest_with_populated_seed_rejected():
    r = good_report()
    r["checkpoint_manifest"]["train_seed"] = 42
    r["train_seed"] = 42
    r["checkpoint_manifest"]["train_recipe"]["max_steps"] = 4000
    # dataset_manifest stays null -> must fail the iff coupling
    _expect_fail(r, "dataset_manifest must be null iff")


def test_null_max_steps_with_populated_seed_rejected():
    r = good_report()
    r["checkpoint_manifest"]["train_seed"] = 42
    # max_steps stays null -> the seed/steps pair coupling fails first
    _expect_fail(r, "both be null")


@pytest.mark.parametrize("episodes", [1, 2, 7])
def test_step_budgeted_sim_variable_episodes_ok(episodes):
    r = good_report()
    r["suite"] = "arena_gr1_fridge"
    r["task_ids"] = [0]
    r["num_trials_per_task"] = 1
    r["episodes"] = episodes
    r["episodes_reported_by_evaluator"] = episodes
    r["success_rate"] = 1.0
    r["per_task"] = [{
        "task_id": 0, "task": "put_item_in_fridge_and_close_door",
        "episodes": episodes, "success_rate": 1.0,
    }]
    assert validate_report(r, **dict(
        EXPECT, expected_suite="arena_gr1_fridge", expected_trials=1,
        expected_task_ids=[0], expected_budget_type="steps",
    )) == 1.0


@pytest.mark.parametrize("counts", [(1, 1), (6, 6), (4, 6)])
def test_fixed_trials_require_each_task_budget(counts):
    r = good_report()
    r["episodes"] = r["episodes_reported_by_evaluator"] = sum(counts)
    r["success_rate"] = 1.0
    for task, episodes in zip(r["per_task"], counts):
        task.update(episodes=episodes, success_rate=1.0)
    _expect_fail(r, "!= expected trials")


@pytest.mark.parametrize("budget", [None, "", "unknown"])
def test_invalid_budget_contract_rejected(budget):
    _expect_fail(good_report(), "unknown expected_budget_type", {
        "expected_budget_type": budget,
    })
