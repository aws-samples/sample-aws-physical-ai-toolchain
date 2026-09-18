"""Shared schema-v3 report validator.

Pure stdlib, no AWS calls: this module validates a PARSED metrics report
against pipeline EXPECTATIONS. The Validate processing entrypoint
supplies the expectations from execution-resolved pipeline properties and
performs the S3/HeadObject identity checks; everything content-shaped is
here so it can be unit-tested exhaustively.

Trust model: the validator never trusts the report for anything the
pipeline already knows -- every check is equality against an expectation or
internal consistency of evaluator-derived values.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

TOL = 1e-6
# Tolerance for "is this rate realizable as a whole number of successful episodes".
# Loose enough for a faithfully-stored repeating fraction (1/3 round-trips through JSON
# as 0.3333333333333333, so rate*3 differs from 1 by ~1e-16), tight enough to reject a
# rate that genuinely could not have come from the reported episode count.
SUCCESS_COUNT_TOL = 1e-6
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHA256_PREFIXED = re.compile(r"^sha256:[0-9a-f]{64}$")
# Fallback suite allowlist. This module is delivered INTO containers (base64-
# embedded by stage_validate_code), where config/ does not exist, so it cannot
# import the registry. The authoritative list is registry.valid_suites(), passed
# in as `suite_specs`; this literal is the offline/no-specs fallback and is
# CI-pinned to the manifests by tests/test_suites.py::test_validator_allowlist_mirrors_manifests.
# TODO(port): derive this from a staged suite table instead of a literal, once the
# train entry does the same. See README.md#notes-and-limitations.
VALID_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "arena_gr1_fridge"}
VALID_GL = {"egl", "osmesa"}

TOP_KEYS = {
    "schema_version", "policy_type", "model_family", "checkpoint", "checkpoint_revision",
    "checkpoint_manifest", "weights_digest_recomputed_by_eval",
    "model_artifact_identity", "suite", "task_ids", "num_trials_per_task",
    "eval_seed", "train_seed", "success_rate", "episodes",
    "episodes_reported_by_evaluator", "per_task", "provenance",
    # schema 3: the measured identity of the ARCHIVE the evaluator read. A tree digest
    # cannot establish that SimEval and Validate read the same bytes -- it excludes
    # manifests, logs and hidden paths, so two different archives can share one. And
    # HeadObject reports the object current when HEAD runs, not the object the job had
    # already downloaded. REQUIRED, not optional: evidence that may be absent is not
    # evidence.
}
# Optional, NON-GATED diagnostic evidence. `aux_metrics` carries the auxiliary values
# the simulator reported alongside the gated score (per-subtask rates, subtask success
# vectors, ...) under their ORIGINAL names. They are preserved rather than discarded so
# a run can be diagnosed after the fact, and they travel inside the validated report so
# they are covered by the same schema/digest chain -- but no gate reads them, and a
# non-finite auxiliary value is recorded as null rather than a fabricated number.
TOP_OPTIONAL_KEYS = {"effective_eval_config", "seed_scope", "aux_metrics",
                     # The two source-identity variants. Neither is individually
                     # required; EXACTLY ONE must be present (checked below), because a
                     # report must say which bytes it read and cannot claim both modes.
                     "source_archive", "source_snapshot",
                     # cycle-16 C1: the resolved identity of the external VLM backbone, whose
                     # snapshot determines preprocessing. Optional because not every family loads
                     # an external backbone -- but its SHAPE is now checked when present (below).
                     # It was previously allowlisted and never read, then copied verbatim into the
                     # attestation, so a report naming a backbone the run never loaded was accepted
                     # in silence.
                     "backbone_identity"}
#: The exact keys a backbone identity carries. A partial record is worse than none: "repo_id"
#: without a commit names a MOVING reference and reads as though it pinned something.
BACKBONE_IDENTITY_KEYS = {"repo_id", "resolved_commit"}
MANIFEST_KEYS = {
    "manifest_version", "model_family", "base_checkpoint", "base_revision",
    "train_seed", "input_config", "train_recipe", "weights_digest",
    "dataset_manifest",
}
# Optional provenance keys (allowed, not required) added for the self-contained
# Option-C layout (MolmoAct2 base+adapter): resolved base/tokenizer SHAs and the
# on-disk subdir layout. Optional so pre-Option-C manifests still validate.
MANIFEST_OPTIONAL_KEYS = {
    "base_snapshot_resolved_sha", "fast_tokenizer_resolved_sha",
    "checkpoint_layout",
}
RECIPE_KEYS = {"repo", "commit", "max_steps"}
# Optional recipe provenance (LoRA hyperparameters + training dtype), allowed on
# fine-tune manifests that record the exact PEFT config.
RECIPE_OPTIONAL_KEYS = {
    "batch_size", "train_mode_vlm", "train_model_dtype", "lora_rank",
    "lora_alpha", "lora_dropout", "lora_bias", "lora_target_modules",
}
DATASET_KEYS = {"source", "revision", "episode_count"}
# Optional dataset provenance: which episodes training actually consumed. `episode_count`
# alone describes the DATASET, not the data used -- a run over the first 5 episodes of a
# 1693-episode dataset recorded the same value as a run over all of them. Optional
# because producers already baked into published images predate these keys.
DATASET_OPTIONAL_KEYS = {"episodes_selected", "episode_selection"}
# schema 3: checksum_sha256 is removed. S3 only returns it when the object was uploaded
# with a checksum algorithm, so it was absent in practice and its presence in the key
# set implied a guarantee the identity never carried. The archive measurement below
# is the content evidence.
IDENTITY_KEYS = {"s3_uri", "bucket", "key", "version_id", "etag"}
SOURCE_ARCHIVE_KEYS = {"sha256", "size_bytes"}
# cycle-14 I4: an HF-mode evaluation downloads a SNAPSHOT rather than an archive, so it has
# no tarball to measure and used to publish source_archive: {} -- which this validator
# rejects. The run spent its entire inference budget and then could not publish.
#
# The fix is an explicit VARIANT, not a fabricated archive measurement: a downloaded
# directory is a different claim from a measured tarball, and pretending otherwise would
# put a made-up digest in the evidence chain. resolved_commit is what pins the source;
# tree_sha256 is measured over the downloaded contents.
SOURCE_SNAPSHOT_KEYS = {"repo_id", "resolved_commit", "tree_sha256"}
PER_TASK_KEYS = {"task_id", "task", "episodes", "success_rate", "successes"}
# `successes` is ALLOWED but not REQUIRED: an explicit integer count is the better
# contract, and a producer that supplies it gets checked against the rate. Existing
# producers (including bytes already baked into published images) predate it, so
# requiring it would reject valid historical reports. The realizability check below
# applies either way -- a rate that cannot arise from a whole number of successful
# episodes out of the reported episodes is rejected whether or not `successes` is given.
PER_TASK_REQUIRED = {"task_id", "task", "episodes", "success_rate"}
PROVENANCE_REQUIRED = {"recipe_repo", "recipe_commit", "mujoco_gl"}


class ReportInvalid(ValueError):
    pass


def _fail(msg: str) -> None:
    raise ReportInvalid(msg)


def parse_report(raw: str) -> dict:
    """Parse JSON with NaN/Infinity rejected (contract numeric strictness)."""
    def _reject(value: str) -> None:
        _fail(f"non-finite JSON number: {value}")

    def _float(value: str) -> float:
        f = float(value)
        if not math.isfinite(f):
            _fail(f"JSON number overflows to non-finite: {value}")
        return f
    return json.loads(raw, parse_constant=_reject, parse_float=_float)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_finite_number(x: Any) -> bool:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _is_rate(x: Any) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)) and 0.0 <= float(x) <= 1.0)


def _nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _check_keys(obj: dict, allowed: set, where: str, required: set | None = None) -> None:
    if not isinstance(obj, dict):
        _fail(f"{where}: not an object")
    unknown = set(obj) - allowed
    if unknown:
        _fail(f"{where}: unknown keys {sorted(unknown)}")
    missing = (required if required is not None else allowed) - set(obj)
    if missing:
        _fail(f"{where}: missing keys {sorted(missing)}")


def _check_family_schema(obj: dict, schema: dict, where: str) -> None:
    """Validate against the shared {type, required, enum?, pattern?} grammar."""
    type_map = {"int": _is_int, "bool": lambda x: isinstance(x, bool),
                "str": lambda x: isinstance(x, str),
                "float": _is_finite_number}
    allowed = set(schema)
    unknown = set(obj) - allowed
    if unknown:
        _fail(f"{where}: unknown keys {sorted(unknown)}")
    for key, spec in schema.items():
        if key not in obj:
            if spec.get("required", False):
                _fail(f"{where}: missing required key {key}")
            continue
        val = obj[key]
        if not type_map[spec["type"]](val):
            _fail(f"{where}.{key}: expected {spec['type']}, got {val!r}")
        if "enum" in spec and val not in spec["enum"]:
            _fail(f"{where}.{key}: {val!r} not in enum {spec['enum']}")
        if "pattern" in spec and isinstance(val, str) and not re.fullmatch(spec["pattern"], val):
            _fail(f"{where}.{key}: {val!r} fails pattern")


def validate_manifest(m: dict, family_schemas: dict) -> None:
    _check_keys(m, MANIFEST_KEYS | MANIFEST_OPTIONAL_KEYS, "checkpoint_manifest",
                required=MANIFEST_KEYS)
    if not _is_int(m["manifest_version"]) or m["manifest_version"] != 1:
        _fail(f"manifest_version: {m['manifest_version']!r}")
    if not _nonempty_str(m["model_family"]):
        _fail("manifest.model_family empty")
    if not _nonempty_str(m["base_checkpoint"]):
        _fail("manifest.base_checkpoint empty")
    if not _nonempty_str(m["base_revision"]):
        _fail("manifest.base_revision empty")
    if not SHA256_PREFIXED.fullmatch(m["weights_digest"] or ""):
        _fail(f"manifest.weights_digest malformed: {m['weights_digest']!r}")
    recipe = m["train_recipe"]
    _check_keys(recipe, RECIPE_KEYS | RECIPE_OPTIONAL_KEYS, "train_recipe",
                required=RECIPE_KEYS)
    if not _nonempty_str(recipe["repo"]):
        _fail("train_recipe.repo empty")
    if not (isinstance(recipe["commit"], str) and HEX40.fullmatch(recipe["commit"])):
        _fail(f"train_recipe.commit not 40-hex: {recipe['commit']!r}")
    # Conditional nulls travel together (contract rev6)
    seed_null = m["train_seed"] is None
    steps_null = recipe["max_steps"] is None
    if seed_null != steps_null:
        _fail("train_seed and train_recipe.max_steps must both be null (published ckpt) or both set")
    if not seed_null and not _is_int(m["train_seed"]):
        _fail(f"train_seed invalid: {m['train_seed']!r}")
    if not steps_null and not (_is_int(recipe["max_steps"]) and recipe["max_steps"] >= 1):
        _fail(f"max_steps invalid: {recipe['max_steps']!r}")
    ds = m["dataset_manifest"]
    # rev6: null exactly for published checkpoints (travels with train_seed)
    if (ds is None) != seed_null:
        _fail("dataset_manifest must be null iff train_seed is null (published ckpt)")
    if ds is not None:
        _check_keys(ds, DATASET_KEYS | DATASET_OPTIONAL_KEYS, "dataset_manifest",
                    required=DATASET_KEYS)
        if not _nonempty_str(ds["source"]) or not _nonempty_str(ds["revision"]):
            _fail("dataset_manifest source/revision empty")
        if not (_is_int(ds["episode_count"]) and ds["episode_count"] >= 1):
            _fail(f"dataset_manifest.episode_count invalid: {ds['episode_count']!r}")
        # When the producer records what it actually consumed, it must be coherent: a
        # selection cannot be empty, and cannot exceed the dataset it was drawn from.
        if "episodes_selected" in ds:
            selected = ds["episodes_selected"]
            if not (_is_int(selected) and selected >= 1):
                _fail(f"dataset_manifest.episodes_selected invalid: {selected!r}")
            if selected > ds["episode_count"]:
                _fail(f"dataset_manifest.episodes_selected {selected} exceeds "
                      f"episode_count {ds['episode_count']}")
            if not _nonempty_str(ds.get("episode_selection")):
                _fail("dataset_manifest.episodes_selected is recorded without "
                      "episode_selection: a count without the rule that produced it "
                      "cannot be reproduced or audited")
        elif "episode_selection" in ds:
            _fail("dataset_manifest.episode_selection is recorded without "
                  "episodes_selected")
    schema = family_schemas.get(m["model_family"], {}).get("input_config_schema")
    if schema is None:
        _fail(f"no input_config_schema registered for family {m['model_family']!r}")
    _check_family_schema(m["input_config"], schema, "input_config")
    # These protocol settings apply to LIBERO, not Arena. Older Arena artifacts
    # may still carry them; the schema continues to validate their types.
    if (m["model_family"] == "gr00t"
            and m["input_config"].get("embodiment_tag") == "LIBERO_PANDA"):
        _require_gr00t_libero_protocol(m["input_config"])


def _require_gr00t_libero_protocol(config: dict) -> None:
    for key in ("n_action_steps", "max_episode_steps"):
        value = config.get(key)
        if not isinstance(value, str) or not value.isdecimal() or int(value) < 1:
            _fail(f"input_config.{key}: GR00T/LIBERO requires a positive integer string")


def validate_report(
    report: dict,
    *,
    expected_suite: str,
    expected_trials: int,
    expected_eval_seed: int,
    expected_task_ids: list[int],
    family_schemas: dict,
    pipeline_mode: bool,
    expected_budget_type: str = "fixed_trials",
    suite_specs: dict | None = None,
    expected_family_version: str | None = None,
) -> float:
    """Full schema-v3 validation. Returns the gated success_rate.

    ``suite_specs`` is ``registry.suites_json()`` -- the resolved suite manifests,
    injected via ``VLA_SUITES_JSON``. When present it supplies the authoritative
    suite allowlist (``status: supported`` only) and enables the coherence checks
    below. When absent (offline callers) the module-level ``VALID_SUITES`` fallback
    applies and the coherence checks are skipped.

    ``expected_family_version`` is the pipeline-owned family version selector
    (``Gr00tVersion``: ``n16``/``n17``), which selects WHICH ``family_overrides``
    entry the run was supposed to train under. Required for the embodiment
    coherence check; without it that check is skipped.
    """
    if expected_budget_type not in ("fixed_trials", "steps"):
        _fail(f"unknown expected_budget_type: {expected_budget_type!r}")
    _check_keys(report, TOP_KEYS | TOP_OPTIONAL_KEYS, "report", required=TOP_KEYS)
    if report["policy_type"] != "checkpoint":
        _fail(f"policy_type must be 'checkpoint', got {report['policy_type']!r}")
    if not _is_int(report["schema_version"]) or report["schema_version"] != 3:
        _fail(f"schema_version: {report['schema_version']!r} (this validator requires 3; "
              f"version 2 reports carry no measured archive identity, so they cannot show "
              f"that evaluation and validation read the same bytes)")
    if report["model_family"] not in family_schemas:
        _fail(f"unknown model_family: {report['model_family']!r}")
    allowed = ({s for s, spec in suite_specs.items()
                if spec.get("status", "supported") == "supported"}
               if suite_specs else VALID_SUITES)
    if report["suite"] not in allowed:
        _fail(f"unknown suite: {report['suite']!r} (allowed: {sorted(allowed)})")
    if report["suite"] != expected_suite:
        _fail(f"suite mismatch: report={report['suite']!r} expected={expected_suite!r}")

    # Task-set binding: exact equality with the pipeline expectation.
    # null is FATAL (hardening): a report that omits its task set could
    # otherwise be canonicalized into whatever the validator assumes. The
    # adapter must write the explicit sorted list it actually ran.
    task_ids = report["task_ids"]
    if (not isinstance(task_ids, list) or not all(_is_int(t) for t in task_ids)
            or len(set(task_ids)) != len(task_ids) or sorted(task_ids) != task_ids):
        _fail(f"task_ids malformed: {report['task_ids']!r}")
    if task_ids != sorted(set(expected_task_ids)):
        _fail(f"task-set mismatch: report={task_ids} expected={sorted(set(expected_task_ids))}")

    trials = report["num_trials_per_task"]
    if not _is_int(trials) or trials < 1:
        _fail(f"num_trials_per_task invalid: {trials!r}")
    if trials != expected_trials:
        _fail(f"trials mismatch: report={trials} expected={expected_trials}")
    if not _is_int(report["eval_seed"]):
        _fail(f"eval_seed invalid: {report['eval_seed']!r}")
    if report["eval_seed"] != expected_eval_seed:
        _fail(f"eval_seed mismatch: report={report['eval_seed']} expected={expected_eval_seed}")
    if report["train_seed"] is not None and not _is_int(report["train_seed"]):
        _fail(f"train_seed invalid: {report['train_seed']!r}")

    # Checkpoint locality must agree with the pipeline mode (coherence):
    # pipeline mode consumes the FineTune artifact (local mounted path);
    # HF mode consumes a repo id (never a local path).
    ckpt = report["checkpoint"]
    if not _nonempty_str(ckpt):
        _fail(f"checkpoint invalid: {ckpt!r}")
    # Local = absolute path OR relative-path shapes ("./x", "../x", bare
    # "x/y" is ambiguous with HF org/name so only dot-forms count).
    is_local = ckpt.startswith(("/", "./", "../"))
    if pipeline_mode and not is_local:
        _fail(f"pipeline mode requires a local checkpoint path, got {ckpt!r}")
    if not pipeline_mode and is_local:
        _fail(f"HF mode forbids a local checkpoint path: {ckpt!r}")

    sr = report["success_rate"]
    if not _is_rate(sr):
        _fail(f"success_rate invalid: {sr!r}")
    episodes = report["episodes"]
    reported = report["episodes_reported_by_evaluator"]
    if not _is_int(episodes) or not _is_int(reported):
        _fail("episodes fields must be ints")
    # NOTE: the blanket `episodes == num_trials_per_task * len(task_ids)` equality is not
    # asserted here, because it does not hold for every budget protocol. It IS enforced
    # per task under `budget_type='fixed_trials'` below, which is the mode Arena now uses:
    # the rollout is asked for a fixed number of COMPLETE episodes (--num_episodes), so
    # the observed count must equal the requested one. (Arena was previously declared
    # step-budgeted, which DISABLED that check -- a step budget shorter than the
    # environment's episode length silently produced zero episodes.) The integrity legs
    # that make the gated-registry claim true are preserved: weights-digest match
    # (trained checkpoint == evaluated checkpoint, below), episodes>0 (per_task is
    # required non-empty), and the internal-consistency checks that follow
    # (episodes == episodes_reported_by_evaluator, per-task episode-sum == episodes,
    # weighted mean == success_rate).
    if episodes != reported:
        _fail(f"episodes {episodes} != episodes_reported_by_evaluator {reported}")

    per_task = report["per_task"]
    if not isinstance(per_task, list) or not per_task:
        _fail("per_task missing or empty")
    seen_ids = []
    total_eps = 0
    weighted = 0.0
    for rec in per_task:
        _check_keys(rec, PER_TASK_KEYS, "per_task record", required=PER_TASK_REQUIRED)
        if not _is_int(rec["task_id"]):
            _fail(f"per_task.task_id invalid: {rec['task_id']!r}")
        if not _nonempty_str(rec["task"]):
            _fail("per_task.task empty")
        if not (_is_int(rec["episodes"]) and rec["episodes"] >= 1):
            _fail(f"per_task.episodes invalid: {rec['episodes']!r}")
        if expected_budget_type == "fixed_trials" and rec["episodes"] != expected_trials:
            _fail(f"per_task.episodes {rec['episodes']} != expected trials {expected_trials}")
        if not _is_rate(rec["success_rate"]):
            _fail(f"per_task.success_rate invalid: {rec['success_rate']!r}")
        # Success is a BINARY per-episode outcome, so a task's rate must be a whole
        # number of successful episodes divided by its episode count. Bounds, sums and
        # weighted means are all consistency relations BETWEEN independently supplied
        # summaries -- they cannot tell that a summary could not have arisen from the
        # trials it claims. Without this, a ten-task report with one episode each and
        # success_rate=0.5 everywhere validated: half a successful episode per task.
        _succ_exact = rec["success_rate"] * rec["episodes"]
        _succ = round(_succ_exact)
        if abs(_succ_exact - _succ) > SUCCESS_COUNT_TOL:
            _fail(
                f"per_task.success_rate {rec['success_rate']!r} is not realizable from "
                f"{rec['episodes']} episode(s) of task {rec['task_id']}: it implies "
                f"{_succ_exact} successful episodes, which is not a whole number. "
                f"Success is binary per episode, so the rate must be successes/episodes.")
        if not 0 <= _succ <= rec["episodes"]:
            _fail(f"per_task implied successes {_succ} outside [0, {rec['episodes']}] "
                  f"for task {rec['task_id']}")
        # When the producer supplies the count explicitly, it must agree with the rate.
        if "successes" in rec:
            if not _is_int(rec["successes"]):
                _fail(f"per_task.successes must be an int: {rec['successes']!r}")
            if rec["successes"] != _succ:
                _fail(f"per_task.successes {rec['successes']} disagrees with "
                      f"success_rate {rec['success_rate']!r} over {rec['episodes']} "
                      f"episode(s) (implies {_succ}) for task {rec['task_id']}")
            if not 0 <= rec["successes"] <= rec["episodes"]:
                _fail(f"per_task.successes {rec['successes']} outside "
                      f"[0, {rec['episodes']}] for task {rec['task_id']}")
        seen_ids.append(rec["task_id"])
        total_eps += rec["episodes"]
        weighted += rec["success_rate"] * rec["episodes"]
    if len(set(seen_ids)) != len(seen_ids):
        _fail(f"duplicate task_ids in per_task: {seen_ids}")
    if sorted(seen_ids) != task_ids:
        _fail(f"per_task task set {sorted(seen_ids)} != task_ids {task_ids}")
    if total_eps != reported:
        _fail(f"per_task episode sum {total_eps} != reported {reported}")

    # --- Suite/task coherence gate (Isaac Lab Arena) -------------------------
    # The digest chain proves the EVALUATED BYTES are the TRAINED BYTES. It says
    # nothing about whether the policy was rolled out on the task it was trained
    # for: a run could fine-tune on the fridge dataset, roll out `cube_goal_pose`,
    # and still pass every integrity leg. For an Arena suite the manifest names
    # exactly one task, so the report's task label must equal it -- otherwise the
    # success_rate answers a different question than the suite claims.
    _spec = (suite_specs or {}).get(report["suite"]) or {}
    _arena = _spec.get("arena")
    # `_arena.get("task")` is non-empty for every resolvable Arena suite
    # (registry._parse_arena requires it); the check keeps this function safe for
    # a hand-built suite_specs dict.
    if _arena and _arena.get("task"):
        _expected_task = _arena["task"]
        _reported = sorted({r["task"] for r in per_task})
        if _reported != [_expected_task]:
            _fail(
                f"suite/task incoherence: suite {report['suite']!r} declares Arena "
                f"task {_expected_task!r} but the report rolled out {_reported}. "
                f"The evaluated task must be the suite's task -- a matching "
                f"weights digest does not make a mismatched rollout valid.")

    # --- Embodiment coherence gate ------------------------------------------
    # Stronger than the task check above, because it reads a value FineTune wrote
    # rather than a label the evaluator echoed back: the fine-tune recipe records
    # the embodiment tag it trained under in
    # `checkpoint_manifest.input_config.embodiment_tag`. The suite manifest
    # declares, per (family, version), what that tag must be. If they disagree the
    # policy was trained for one embodiment and rolled out configured for another
    # -- e.g. GR00T N1.6 trained under `new_embodiment` while the eval server is
    # told `--embodiment-tag GR1`. The digest chain cannot see this (the bytes are
    # genuinely the trained bytes) and neither can the task check.
    _ov = ((_spec.get("family_overrides") or {})
           .get(report["model_family"]) or {})
    # Normalise the version the way EVERY other consumer does -- train_entry and
    # eval_entry both `.strip().lower()` it -- and then fail CLOSED on a value the
    # suite does not key. An exact dict lookup on a raw, unconstrained
    # ParameterString silently no-ops the gate for "N16", "n16 " or "n99", and a
    # hand-started StartPipelineExecution is precisely the threat model this gate
    # exists for. On the eval-only path there is no FineTune to fail loud first, so
    # a skipped gate is the only thing left looking.
    _fv = (expected_family_version or "").strip().lower()
    if _fv and _ov:
        _entry = _ov.get(_fv)
        if _entry is None:
            _fail(
                f"unknown family version {expected_family_version!r} for "
                f"{report['model_family']!r}: suite {report['suite']!r} declares "
                f"{sorted(_ov)}. The embodiment coherence gate cannot be "
                f"evaluated, so this run is not registrable. If this run used "
                f"--embodiment-tag to override the manifest, add the "
                f"family_overrides.{report['model_family']}.{_fv} entry to the "
                f"suite instead.")
        _declared = _entry.get("embodiment_tag")
        _manifest = report.get("checkpoint_manifest")
        if _declared and isinstance(_manifest, dict):
            _trained = (_manifest.get("input_config") or {}).get("embodiment_tag")
            if _trained is not None and _trained != _declared:
                _fail(
                    f"suite/embodiment incoherence: suite {report['suite']!r} "
                    f"declares embodiment_tag {_declared!r} for "
                    f"{report['model_family']}/{_fv}, but the checkpoint was "
                    f"fine-tuned under {_trained!r}. The policy was trained for a "
                    f"different embodiment than the suite evaluates. "
                    f"Pass --gr00t-version <n16|n17> to match the checkpoint.")
    if abs(weighted / total_eps - float(sr)) > TOL:
        _fail(f"weighted mean {weighted / total_eps} != success_rate {sr}")

    validate_manifest(report["checkpoint_manifest"], family_schemas)
    if report["model_family"] == "gr00t" and (
            _spec.get("simulator") == "libero" or expected_suite.startswith("libero_")):
        _require_gr00t_libero_protocol(report["checkpoint_manifest"]["input_config"])
    if report["checkpoint_manifest"]["model_family"] != report["model_family"]:
        _fail("manifest.model_family != report.model_family")
    if report["checkpoint_manifest"]["train_seed"] != report["train_seed"]:
        _fail("report.train_seed != manifest.train_seed")

    digest = report["weights_digest_recomputed_by_eval"]
    if not (isinstance(digest, str) and SHA256_PREFIXED.fullmatch(digest)):
        _fail(f"weights_digest_recomputed_by_eval malformed: {digest!r}")
    if digest != report["checkpoint_manifest"]["weights_digest"]:
        _fail("digest mismatch: eval-recomputed != manifest (content binding failed)")

    # checkpoint_revision nullability: null only for local pipeline checkpoints
    rev = report["checkpoint_revision"]
    if rev is None and not is_local:
        _fail("checkpoint_revision null for a non-local checkpoint")
    if rev is not None and not _nonempty_str(rev):
        _fail(f"checkpoint_revision malformed: {rev!r}")

    # schema 3: the measured archive. Validated for EVERY report, not only pipeline mode --
    # a standalone evaluation that cannot say which bytes it read is exactly the evidence gap
    # this field exists to close.
    # EXACTLY ONE source-identity variant, decided by PRESENCE of the key rather than by whether its
    # value happens to be a populated dict. Selecting on truthiness meant junk in the unused key was
    # silently ignored: source_snapshot="unavailable", or {}, or None, alongside a valid archive all
    # passed, so a report could carry a second contradictory claim that nothing looked at.
    present = {k for k in ("source_archive", "source_snapshot") if k in report}
    if len(present) == 2:
        _fail("report carries BOTH source_archive and source_snapshot; exactly one source identity "
              "is permitted, and a report claiming two cannot establish which bytes it read. The "
              "unused variant must be ABSENT, not empty or null -- a key present with junk in it is "
              "a second claim about the same evaluation")
    if not present:
        _fail("report carries neither source_archive nor source_snapshot; an evaluation that cannot "
              "say which bytes it read is exactly the evidence gap these fields exist to close")
    has_archive = "source_archive" in present
    if not isinstance(report[next(iter(present))], dict) or not report[next(iter(present))]:
        _fail(f"{next(iter(present))} must be a populated object, got "
              f"{report[next(iter(present))]!r}")

    # The variant must match the MODE. Either satisfied either before, so a pipeline-mode report
    # could publish a snapshot and pass here while verify_checkpoint went on to require the archive --
    # an inconsistent contract that fails late, after the evaluation has been paid for.
    if pipeline_mode and not has_archive:
        _fail("a pipeline-mode report must carry source_archive: the checkpoint arrived as a mounted "
              "archive that Validate measures independently, and a snapshot claim cannot be "
              "cross-checked against it")
    if not pipeline_mode and has_archive:
        _fail("a non-pipeline report must carry source_snapshot: there is no mounted archive in that "
              "mode, so an archive measurement describes something other than what was evaluated")

    if has_archive:
        archive = report["source_archive"]
        _check_keys(archive, SOURCE_ARCHIVE_KEYS, "source_archive")
        sha = archive["sha256"]
        if not (isinstance(sha, str) and len(sha) == 64
                and all(c in "0123456789abcdef" for c in sha)):
            _fail(f"source_archive.sha256 must be 64 lowercase hex characters, got {sha!r}")
        size = archive["size_bytes"]
        if not _is_int(size) or size <= 0:
            _fail(f"source_archive.size_bytes must be a positive integer, got {size!r}")
    else:
        snap = report["source_snapshot"]
        _check_keys(snap, SOURCE_SNAPSHOT_KEYS, "source_snapshot")
        repo_id = snap["repo_id"]
        if not (isinstance(repo_id, str) and repo_id.strip()):
            _fail(f"source_snapshot.repo_id must be a non-empty string, got {repo_id!r}")
        commit = snap["resolved_commit"]
        # A RESOLVED commit, never a branch or tag: those move, so they cannot pin bytes.
        if not (isinstance(commit, str) and len(commit) == 40
                and all(c in "0123456789abcdef" for c in commit)):
            _fail(f"source_snapshot.resolved_commit must be a 40-character hex commit, got "
                  f"{commit!r} -- a branch or tag name does not pin the bytes that were read")
        tree = snap["tree_sha256"]
        if not (isinstance(tree, str) and len(tree) == 64
                and all(c in "0123456789abcdef" for c in tree)):
            _fail(f"source_snapshot.tree_sha256 must be 64 lowercase hex characters, got {tree!r}")

        # AGREEMENT, not just format. Each field above was well-formed in isolation, so a report
        # could name a repo, a commit and a tree digest belonging to three different things and pass
        # every check. These bind the snapshot to the report that carries it.
        if repo_id != report["checkpoint"]:
            _fail(f"source_snapshot.repo_id {repo_id!r} is not the report's own checkpoint "
                  f"{report['checkpoint']!r}; a source identity that names a different artifact than "
                  f"the report describes establishes nothing about what was evaluated")
        if commit != report["checkpoint_revision"]:
            _fail(f"source_snapshot.resolved_commit {commit!r} disagrees with checkpoint_revision "
                  f"{report['checkpoint_revision']!r}; the report would pin two different revisions "
                  f"of the same checkpoint")
        # The evaluator's own recomputed digest is prefixed (SHA256_PREFIXED, enforced above); the
        # snapshot field is bare. Compared with an inline strip so this module stays import-free --
        # it is delivered to bare containers through a two-file bootstrap.
        recomputed = report["weights_digest_recomputed_by_eval"]
        recomputed_bare = (recomputed.split(":", 1)[1]
                           if isinstance(recomputed, str) and recomputed.startswith("sha256:")
                           else recomputed)
        if tree != recomputed_bare:
            _fail(f"source_snapshot.tree_sha256 {tree!r} disagrees with the digest the evaluator "
                  f"recomputed over what it loaded ({recomputed!r}); the snapshot would describe a "
                  f"different tree than the one that was evaluated")

    # backbone_identity: OPTIONAL, but never half-stated. An empty object means "this run loaded no
    # external backbone", which is a legitimate claim. A populated one must name both the repository
    # and the resolved commit, because a repo id alone is a moving reference that reads like a pin.
    #
    # This was allowlisted and never read while being copied verbatim into the attestation, so a
    # report claiming a backbone the run never loaded -- which is exactly what a module-level default
    # produced for every N1.6 run -- passed validation and was attested.
    if "backbone_identity" in report:
        backbone = report["backbone_identity"]
        if not isinstance(backbone, dict):
            _fail(f"backbone_identity must be an object, got {backbone!r}")
        if backbone:
            missing = BACKBONE_IDENTITY_KEYS - set(backbone)
            extra = set(backbone) - BACKBONE_IDENTITY_KEYS
            if missing or extra:
                _fail(f"backbone_identity keys must be exactly {sorted(BACKBONE_IDENTITY_KEYS)}, "
                      f"missing {sorted(missing)} unexpected {sorted(extra)} -- a partial record "
                      f"names a moving reference while reading as though it pinned one")
            repo = backbone["repo_id"]
            if not (isinstance(repo, str) and repo.strip() and repo == repo.strip()):
                _fail(f"backbone_identity.repo_id must be a non-blank string, got {repo!r}")
            commit = backbone["resolved_commit"]
            if not (isinstance(commit, str) and len(commit) == 40
                    and all(c in "0123456789abcdef" for c in commit)):
                _fail(f"backbone_identity.resolved_commit must be a 40-character hex commit, got "
                      f"{commit!r} -- a tag or branch does not identify the weights that were read")

    ident = report["model_artifact_identity"]
    if pipeline_mode:
        _check_keys(ident, IDENTITY_KEYS, "model_artifact_identity")
        for k in ("bucket", "key", "version_id", "etag"):
            if not _nonempty_str(ident[k]):
                _fail(f"model_artifact_identity.{k} empty")
        # S3 returns the LITERAL string "null" as the VersionId of an object in a
        # bucket that is not versioned. A nonempty check therefore accepts a value
        # that provides none of the immutable version selection the identity claims:
        # "null" cannot be used to retrieve one specific generation of the bytes. The
        # schema asserts versioned artifact identity, so an unversioned marker must be
        # rejected here rather than recorded as if it were proof.
        if ident["version_id"].strip().lower() in {"null", "none"}:
            _fail(
                f"model_artifact_identity.version_id is {ident['version_id']!r}, which "
                f"is the marker S3 returns for an object in an UNVERSIONED bucket. It "
                f"selects no particular generation of the bytes, so it cannot serve as "
                f"versioned artifact identity. Enable bucket versioning (or promote the "
                f"artifact to a versioned location) before registering.")
    elif ident is not None:
        _fail("model_artifact_identity must be null for HF checkpoints")

    prov = report["provenance"]
    if not isinstance(prov, dict):
        _fail("provenance not an object")
    fam = family_schemas[report["model_family"]]
    prov_schema = fam.get("provenance_keys", {})
    allowed_prov = PROVENANCE_REQUIRED | set(prov_schema)
    _check_keys(prov, allowed_prov, "provenance", required=PROVENANCE_REQUIRED)
    if not _nonempty_str(prov["recipe_repo"]):
        _fail("provenance.recipe_repo empty")
    if not (isinstance(prov["recipe_commit"], str) and HEX40.fullmatch(prov["recipe_commit"])):
        _fail(f"provenance.recipe_commit not 40-hex: {prov['recipe_commit']!r}")
    if prov["mujoco_gl"] not in VALID_GL:
        _fail(f"provenance.mujoco_gl invalid: {prov['mujoco_gl']!r}")
    # ALWAYS validate extras against the family schema -- an empty extras dict
    # must still fail if the family declares a required provenance key
    # (hardening: input-shaping skip).
    extras = {k: v for k, v in prov.items() if k not in PROVENANCE_REQUIRED}
    _check_family_schema(extras, prov_schema, "provenance extras")

    return float(sr)
