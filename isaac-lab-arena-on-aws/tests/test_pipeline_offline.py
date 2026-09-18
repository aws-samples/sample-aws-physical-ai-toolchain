"""The pipeline definition builds and serializes OFFLINE -- no AWS account.

This is the property that makes the artifact real rather than aspirational: a
reviewer (or CI) can clone, `pip install sagemaker`, and prove the workflow is
well-formed without any credentials or cloud calls.

Architecture:
  FineTune (Training) -> SimEval (Training) -> Validate (Processing, CPU) -> Gate
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vla_pipeline.config import PipelineConfig  # noqa: E402
from vla_pipeline.pipeline import build_parameters, build_pipeline  # noqa: E402

_CFG = {
    "account_id": "123456789012",
    "region": "us-west-2",
    "role_arn": "arn:aws:iam::123456789012:role/Exec",
    "bucket": "my-bucket",
    # Component trust boundary (C4 + C5): distinct runtime identities and the
    # create-only bucket promoted artifacts are published to.
    "training_role_arn": "arn:aws:iam::123456789012:role/Training",
    "workload_role_arn": "arn:aws:iam::123456789012:role/Workload",
    "validation_role_arn": "arn:aws:iam::123456789012:role/Validation",
    "trust_bucket": "my-trust-bucket",
    "handoff_bucket": "my-handoff-bucket",
}


def _cfg():
    return PipelineConfig(**_CFG)


def _definition():
    return json.loads(build_pipeline(_cfg()).definition())


def _step(d, name):
    for s in d["Steps"]:
        if s["Name"] == name:
            return s
    raise AssertionError(f"step {name!r} not in definition: "
                         f"{[s['Name'] for s in d['Steps']]}")


# --- Parameter tests -------------------------------------------------------

def test_parameters_present():
    params = build_parameters()
    names = {p.name for p in params.values()}
    # Core parameters ("model as a parameter")
    assert {"ModelFamily", "TrainSteps", "Suite", "DatasetS3Uri",
            "SuccessThreshold"} <= names
    # Eval protocol parameters (validator checks AGAINST these)
    assert {"EvalSeed", "EvalTrials", "EvalTaskIds"} <= names


def test_image_params_are_required_no_default():
    """A start with images unset must fail at submission, not deep in job."""
    d = _definition()
    by_name = {p["Name"]: p for p in d["Parameters"]}
    for req in ("TrainImageUri", "EvalImageUri", "DatasetS3Uri",
                "TrainSourceDirUri", "EvalSourceDirUri"):
        assert "DefaultValue" not in by_name[req], \
            f"{req} must be required (no default) -- fail loud at submit"


def test_eval_protocol_params_have_defaults():
    """EvalSeed/Trials/TaskIds have sensible defaults (full protocol)."""
    d = _definition()
    by_name = {p["Name"]: p for p in d["Parameters"]}
    assert by_name["EvalSeed"]["DefaultValue"] == 100
    assert by_name["EvalTrials"]["DefaultValue"] == 200
    assert by_name["EvalTaskIds"]["DefaultValue"] == "all"  # v3: "all" not "" (API rejects empty)


# --- Topology tests --------------------------------------------------------

def test_definition_serializes_without_aws():
    d = _definition()
    assert d["Version"] == "2020-12-01"
    step_names = {s["Name"] for s in d["Steps"]}
    # 4-step architecture: FineTune -> SimEval -> Validate -> Gate
    assert {"FineTune", "SimEval", "Validate", "SuccessGate"} <= step_names


def test_pipeline_parameters_declared_in_definition():
    d = _definition()
    declared = {p["Name"] for p in d["Parameters"]}
    assert "ModelFamily" in declared and "SuccessThreshold" in declared
    assert "EvalSeed" in declared and "EvalTrials" in declared


def test_simeval_is_training_step():
    """SimEval must be a TrainingStep (reuses Training quota, matches
    old harness topology, eval_entry uses SM_CHANNEL_MODEL conventions)."""
    d = _definition()
    sim = _step(d, "SimEval")
    # TrainingSteps have AlgorithmSpecification; ProcessingSteps have AppSpecification
    assert "AlgorithmSpecification" in sim.get("Arguments", sim), \
        "SimEval must be a TrainingStep (not ProcessingStep)"


def test_validate_is_processing_step():
    """Validate must be a ProcessingStep (CPU, emits PropertyFile)."""
    d = _definition()
    val = _step(d, "Validate")
    assert "AppSpecification" in val.get("Arguments", val), \
        "Validate must be a ProcessingStep"


def test_simeval_has_model_input_channel():
    """SimEval TrainingStep must have a 'model' input channel whose S3Uri
    is the FineTune step's ModelArtifacts reference. without this,
    the model channel is empty at runtime (add_depends_on is ordering only)."""
    d = _definition()
    sim = _step(d, "SimEval")
    # TrainingStep with inputs has InputDataConfig in Arguments
    args = sim.get("Arguments", sim)
    input_config = args.get("InputDataConfig", [])
    channel_names = [ch.get("ChannelName") for ch in input_config]
    assert "model" in channel_names, \
        f"SimEval must have a 'model' input channel, got: {channel_names}"
    # The model channel's S3Uri must reference FineTune's ModelArtifacts
    model_ch = [ch for ch in input_config if ch["ChannelName"] == "model"][0]
    s3_uri_blob = json.dumps(model_ch.get("DataSource", {}))
    assert "FineTune" in s3_uri_blob and "ModelArtifacts" in s3_uri_blob, \
        f"model channel must reference FineTune's ModelArtifacts, got: {s3_uri_blob[:200]}"


@pytest.mark.parametrize("eval_only", [False, True])
def test_every_parameter_reference_is_declared(eval_only):
    """C6: the eval-only definition referenced parameters it had removed.

    Validate's environment unconditionally read TrainSteps, TrainSuite, DatasetS3Uri and
    DatasetRevision, while the eval-only branch removes those declarations -- so the
    definition carried undeclared Parameters.* references and would be rejected at upsert,
    before any job runs. A Python-side early return for eval_only cannot fix a definition
    that must be accepted first.

    This walks the WHOLE definition rather than checking known sites, so any future
    reference to an undeclared parameter fails here regardless of where it is added.
    """
    definition = _eval_only_definition() if eval_only else _definition()
    declared = {param["Name"] for param in definition["Parameters"]}

    referenced: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            target = node.get("Get")
            if isinstance(target, str) and target.startswith("Parameters."):
                referenced.add(target.split(".", 1)[1])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(definition)
    undeclared = sorted(referenced - declared)
    assert not undeclared, (
        f"definition references undeclared parameters {undeclared}; SageMaker rejects "
        f"this at upsert. Declared: {sorted(declared)}")
    # Sanity: the walk must actually find references, or it would pass vacuously.
    assert referenced, "no Parameters.* references found -- the walk is not working"


def test_stages_run_under_distinct_component_roles():
    """Every stage shared the Foundation role, which can overwrite the validator.

    The Foundation role holds Put/DeleteObject across the models bucket that also carries
    the staged validation code, so a training or eval worker could overwrite the validator
    that judges it and the model bytes after validation. Asserting distinct names alone is
    weak, so this also pins WHICH role each stage gets.

    Later: FineTune and SimEval were split too. One shared workload role scoped writes away
    from the validator but still let a TRAINING worker write under eval/, manufacturing or
    replacing the raw evaluation evidence Validate judges. All three stages now differ.
    """
    definition = _definition()
    finetune = _step(definition, "FineTune")["Arguments"]["RoleArn"]
    simeval = _step(definition, "SimEval")["Arguments"]["RoleArn"]
    validate = _step(definition, "Validate")["Arguments"]["RoleArn"]
    assert finetune == _CFG["training_role_arn"]
    assert simeval == _CFG["workload_role_arn"]
    assert validate == _CFG["validation_role_arn"]
    # The workers must NOT hold the identity that publishes promoted artifacts.
    assert validate != finetune
    # And producing the weights must not be the same identity as producing the evidence.
    assert finetune != simeval, (
        "a training worker sharing SimEval's identity can write the evaluation evidence")
    assert len({finetune, simeval, validate}) == 3
    # And no stage may run as the Foundation orchestration role.
    assert _CFG["role_arn"] not in {finetune, simeval, validate}


def test_validate_receives_the_trust_bucket():
    """C5: without it, Validate refuses to emit a receipt rather than registering the
    unpromoted source URI."""
    definition = _definition()
    env = _step(definition, "Validate")["Arguments"]["Environment"]
    assert env["TRUST_BUCKET"] == _CFG["trust_bucket"]
    assert "PIPELINE_EXECUTION_ID" in env


@pytest.mark.parametrize("eval_only", [False, True])
def test_validation_and_registration_use_evaluated_checkpoint(eval_only):
    definition = _eval_only_definition() if eval_only else _definition()
    expected_checkpoint = {
        "Get": "Parameters.CheckpointS3Uri" if eval_only
        else "Steps.FineTune.ModelArtifacts.S3ModelArtifacts",
    }
    inputs = _step(definition, "Validate")["Arguments"]["ProcessingInputs"]
    data_inputs = {
        item["S3Input"]["LocalPath"]: item["S3Input"]["S3Uri"]
        for item in inputs if item["S3Input"]["LocalPath"] != "/opt/ml/processing/input/code"
    }
    assert data_inputs == {
        "/opt/ml/processing/checkpoint": expected_checkpoint,
        "/opt/ml/processing/eval_output": {
            "Get": "Steps.SimEval.ModelArtifacts.S3ModelArtifacts",
        },
    }
    channels = _step(definition, "SimEval")["Arguments"]["InputDataConfig"]
    model = next(channel for channel in channels if channel["ChannelName"] == "model")
    assert model["DataSource"]["S3DataSource"]["S3Uri"] == expected_checkpoint
    registration = _step(definition, "SuccessGate")["Arguments"]["IfSteps"][0]["Arguments"]
    model_data_url = registration["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    # C5: registration must resolve to the PROMOTED artifact from Validate's receipt, not
    # to the raw source URI. This assertion previously required the opposite -- it
    # encoded the defect, because the source URI is unversioned and therefore resolves to
    # whatever occupies that key at resolution time rather than the validated bytes.
    assert model_data_url == {
        "Std:JsonGet": {
            "PropertyFile": {"Get": "Steps.Validate.PropertyFiles.ValidatedMetrics"},
            "Path": "promotion.model_uri",
        }
    }
    assert model_data_url != expected_checkpoint, (
        "registration must not point at the unpromoted source artifact")


@pytest.mark.parametrize("eval_only", [False, True])
def test_validation_storage_uses_execution_volume_parameter(eval_only):
    definition = _eval_only_definition() if eval_only else _definition()
    validation = _step(definition, "Validate")
    resources = validation["Arguments"]["ProcessingResources"]["ClusterConfig"]
    assert resources["VolumeSizeInGB"] == {"Get": "Parameters.VolumeSizeInGB"}


def test_gate_reads_validator_not_eval():
    """ConditionStep gates on Validate's PropertyFile, never raw eval output.
    This makes it structurally impossible to gate on an unvalidated number."""
    d = _definition()
    gate = _step(d, "SuccessGate")
    conditions_blob = json.dumps(gate)
    # The gate must reference "Validate" step (not "SimEval")
    assert "Validate" in conditions_blob, \
        "Gate must reference the Validate step's PropertyFile"
    # And must NOT directly reference SimEval's output in conditions
    # (it CAN reference SimEval in DependsOn, but not in conditions)
    conditions_only = json.dumps(gate.get("Conditions", gate.get("Arguments", {}).get("Conditions", [])))
    assert "SimEval" not in conditions_only, \
        "Gate must NOT gate on SimEval's raw output directly"


# --- Caching tests ---------------------------------------------------------

@pytest.mark.parametrize("eval_only", [False, True])
def test_configured_secret_reaches_each_runtime(eval_only):
    from dataclasses import replace

    cfg = replace(_cfg(), hf_secret_name="robotics/custom-hf")
    definition = json.loads(build_pipeline(
        cfg, checkpoint_s3_uri=_CKPT if eval_only else None).definition())
    steps = ["SimEval"] if eval_only else ["FineTune", "SimEval"]
    for name in steps:
        assert _step(definition, name)["Arguments"]["Environment"]["HF_SECRET_NAME"] == cfg.hf_secret_name
    assert "vla-pipeline/hf-token" not in json.dumps(definition)


def test_training_identity_is_independent_of_evaluation_suite():
    definition = _definition()
    train = _step(definition, "FineTune")["Arguments"]
    assert "Parameters.Suite" not in json.dumps(train)
    assert "Parameters.TrainSuite" in json.dumps(train["Environment"]["TRAIN_SUITE"])
    assert "Parameters.Suite" in json.dumps(
        _step(definition, "SimEval")["Arguments"]["Environment"]["EVAL_SUITE"])
    for parameter in ("TrainSteps", "DatasetS3Uri", "TrainSourceDirUri", "TrainImageUri"):
        assert f"Parameters.{parameter}" in json.dumps(train)
    assert "TrainSuite" not in {p["Name"] for p in _eval_only_definition()["Parameters"]}


def test_finetune_caching_disabled():
    """FineTune caching OFF until DatasetRevision carries an immutable SHA
    at submission time. A named ref in the cache key can match across
    different actual dataset bytes."""
    d = _definition()
    fine = _step(d, "FineTune")
    assert fine["CacheConfig"]["Enabled"] is False


def test_simeval_caching_disabled():
    """CONTRACT (contract.md rev6): SimEval caching MUST be off. A cached eval
    resolves its model-artifact reference to a PRIOR run's output -- breaking
    'evaluated bytes == produced bytes'. Also: stochastic eval must never be
    cache-reused (measurement hygiene)."""
    d = _definition()
    sim = _step(d, "SimEval")
    assert sim["CacheConfig"]["Enabled"] is False, \
        "SimEval caching must be disabled (contract rev6)"


def test_validate_caching_disabled():
    """Validate must always re-validate fresh eval output."""
    d = _definition()
    val = _step(d, "Validate")
    assert val["CacheConfig"]["Enabled"] is False, \
        "Validate caching must be disabled"


# --- Portability tests -----------------------------------------------------

def test_no_hardcoded_bucket_in_definition():
    """Output paths must use the CONFIGURED bucket, not a baked-in one."""
    d = _definition()
    body = json.dumps(d)
    for bucket in (_CFG["bucket"], _CFG["handoff_bucket"], _CFG["trust_bucket"]):
        assert bucket in body


def test_simeval_input_channel_implies_dependency():
    """When SimEval has FineTune's ModelArtifacts as a TrainingInput, the SDK
    automatically establishes the DAG dependency. No separate add_depends_on
    needed (and relying on it alone was that bug)."""
    d = _definition()
    sim = _step(d, "SimEval")
    # The presence of the model input channel with FineTune reference
    # is sufficient -- SDK resolves DAG ordering from it
    args = sim.get("Arguments", sim)
    input_config = args.get("InputDataConfig", [])
    model_channels = [ch for ch in input_config if ch.get("ChannelName") == "model"]
    assert len(model_channels) == 1


def test_register_model_behind_gate_only():
    """RegisterModel fires only on gate-pass (if_steps), never unconditionally.
    Every gate pass registers a model package."""
    d = _definition()
    gate = _step(d, "SuccessGate")
    # ConditionStep has IfSteps and ElseSteps
    if_steps = gate.get("Arguments", {}).get("IfSteps", [])
    else_steps = gate.get("Arguments", {}).get("ElseSteps", [])
    # RegisterModel must be in if_steps
    if_names = [s.get("Name", "") for s in if_steps]
    assert any("Register" in n for n in if_names), \
        f"RegisterModel must be in SuccessGate if_steps, got: {if_names}"
    # Nothing in else_steps should register
    else_names = [s.get("Name", "") for s in else_steps]
    assert not any("Register" in n for n in else_names), \
        f"RegisterModel must NOT be in else_steps: {else_names}"


# --- Eval-only / checkpoint-skip variant tests (D2) ------------------------

_CKPT = "s3://my-bucket/vla-pipeline/train/exec-abc/output/model.tar.gz"


def _eval_only_definition():
    return json.loads(build_pipeline(_cfg(), checkpoint_s3_uri=_CKPT).definition())


def test_eval_only_omits_finetune():
    """Checkpoint-skip variant is a distinct static graph: NO FineTune step."""
    d = _eval_only_definition()
    names = {s["Name"] for s in d["Steps"]}
    assert "FineTune" not in names, f"eval-only must omit FineTune, got {names}"
    assert {"SimEval", "Validate", "SuccessGate"} <= names


def test_eval_only_declares_checkpoint_param_and_drops_train_only():
    d = _eval_only_definition()
    declared = {p["Name"] for p in d["Parameters"]}
    assert "CheckpointS3Uri" in declared
    # FineTune-only params must NOT be declared (else the runner would have to
    # pass unused/undeclared values at StartPipelineExecution).
    for train_only in ("TrainImageUri", "DatasetS3Uri", "TrainSteps",
                       "TrainSaveSteps", "TrainSourceDirUri", "TrainInstanceType"):
        assert train_only not in declared, \
            f"{train_only} must be dropped in eval-only, still declared"


def test_eval_only_simeval_sources_checkpoint_not_finetune():
    """SimEval's model channel must reference the CheckpointS3Uri param, never
    a (nonexistent) FineTune ModelArtifacts property."""
    d = _eval_only_definition()
    sim = _step(d, "SimEval")
    blob = json.dumps(sim.get("Arguments", sim).get("InputDataConfig", []))
    assert "CheckpointS3Uri" in blob, f"SimEval must source CheckpointS3Uri: {blob[:200]}"
    assert "FineTune" not in blob, "eval-only SimEval must not reference FineTune"


def test_eval_only_records_input_checkpoint_uri_provenance():
    definition = _eval_only_definition()
    gate = _step(definition, "SuccessGate")
    registration = gate["Arguments"]["IfSteps"][0]["Arguments"]
    checkpoint = {"Get": "Parameters.CheckpointS3Uri"}
    meta = registration["CustomerMetadataProperties"]
    assert meta["input_checkpoint_uri"] == checkpoint
    assert meta["eval_only"] == "true"
    assert "model_family" in meta
    # C5: eval-only registers the PROMOTED artifact too -- both paths use the same
    # publication path, so neither can register unpromoted bytes. The input checkpoint URI
    # stays on the package as provenance: verifying a supplied checkpoint's bytes is not
    # the same claim as authenticating its training lineage, and promotion must not
    # upgrade one into the other.
    model_data_url = registration[
        "InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    assert model_data_url == {
        "Std:JsonGet": {
            "PropertyFile": {"Get": "Steps.Validate.PropertyFiles.ValidatedMetrics"},
            "Path": "promotion.model_uri",
        }
    }
    assert model_data_url != checkpoint


def test_train_default_has_lineage_metadata():
    gate = _step(_definition(), "SuccessGate")
    registration = gate["Arguments"]["IfSteps"][0]["Arguments"]
    meta = registration["CustomerMetadataProperties"]
    assert "model_family" in meta
    assert "suite" in meta
    assert "eval_only" not in meta


def test_eval_only_serializes_without_aws():
    d = _eval_only_definition()
    assert d["Version"] == "2020-12-01"


@pytest.mark.parametrize("eval_only", [False, True])
def test_validation_output_is_execution_scoped(eval_only):
    definition = _eval_only_definition() if eval_only else _definition()
    validation = _step(definition, "Validate")
    outputs = validation["Arguments"]["ProcessingOutputConfig"]["Outputs"]
    output = next(item for item in outputs if item["OutputName"] == "validated")
    assert output["S3Output"]["S3Uri"] == {
        "Std:Join": {
            "On": "/",
            "Values": [
                _cfg().handoff_uri("validated/v1"),
                {"Get": "Execution.PipelineExecutionId"},
            ],
        }
    }
    assert validation["PropertyFiles"] == [{
        "PropertyFileName": "ValidatedMetrics",
        "OutputName": "validated",
        "FilePath": "validated_metrics.json",
    }]


def test_no_job_creation_site_uses_the_shared_foundation_role():
    """C1: standalone SimEval created its training job with the Foundation role.

    The pipeline path already used workload_role_arn, so the same container ran with
    different privileges depending on how it was launched, and the standalone path bypassed
    the trust boundary the component owns. cfg.role_arn remains correct for pipeline UPSERT,
    which is orchestration rather than job execution.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for relative in ("scripts/submit_simeval.py", "src/vla_pipeline/runner.py",
                     "src/vla_pipeline/pipeline.py"):
        for number, line in enumerate((root / relative).read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # A job's execution identity is passed as role=/RoleArn=; the upsert role is
            # passed positionally to upsert_versioned or as role_arn=.
            if ("role=cfg.role_arn" in stripped or "RoleArn=cfg.role_arn" in stripped):
                offenders.append(f"{relative}:{number}: {stripped}")
    # pipeline.py's Model role is deliberately excluded from this rule: it is the identity a
    # deployed endpoint would assume, not a job execution identity, and changing it would
    # alter registration rather than job privilege.
    offenders = [o for o in offenders if "pipeline.py" not in o]
    assert not offenders, (
        "job execution must use the component workload/validation role:\n" + "\n".join(offenders))


def test_the_standalone_launcher_uses_the_workload_role():
    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "scripts/submit_simeval.py").read_text()
    assert "cfg.workload_role_arn" in source
    assert "cfg.role_arn, cfg.bucket" not in source


def test_the_plumbing_graph_exercises_the_same_identity_split_as_production():
    """A rehearsal under different identities proves nothing about real permissions."""
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src/vla_pipeline/runner.py").read_text()
    assert "role=cfg.workload_role_arn" in source
    assert "role=cfg.validation_role_arn" in source
