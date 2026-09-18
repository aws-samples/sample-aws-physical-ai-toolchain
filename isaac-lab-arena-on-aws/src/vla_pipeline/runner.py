"""Runner: upload code, deploy (upsert) the pipeline, start executions.

This is the interface between local developer workflow and AWS. It handles:
  - Content-addressed code upload (sha256 in the S3 key)
  - Pipeline upsert (build definition + deploy)
  - Execution start + wait + step-status retrieval

All identity comes from PipelineConfig (no hardcoded account/bucket/region).
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from sagemaker.workflow.pipeline import Pipeline

    from .config import PipelineConfig


def stage_validate_code(repo_root: str | None = None) -> str:
    """Create a self-contained validate_entry.py with validator embedded.

    ProcessingStep `code=` accepts only a single .py file S3 URI.
    SageMaker downloads that one file and runs it -- no source_dir for Processing
    in pipeline mode. Staging embeds the shared modules as base64 constants in
    validate_entry.py. At runtime it restores them in /tmp and imports them.

    The pinned AWS SDK is also embedded at staging time. The first staging
    downloads its hash-locked wheels into a host cache; jobs never run pip.
    Returns path to the staged validate_entry.py (fully self-contained).
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    validate_entry = root / "entrypoints" / "validate_entry.py"
    validator_src = root / "src" / "vla_pipeline" / "common" / "validator.py"

    if not validate_entry.exists():
        raise FileNotFoundError(f"validate_entry.py not found at {validate_entry}")

    entry_code = validate_entry.read_text()

    if validator_src.exists():
        import base64
        import json as _json
        validator_code = validator_src.read_bytes()
        b64 = base64.b64encode(validator_code).decode()
        digest_code = (validator_src.parent / "digest.py").read_bytes()
        digest_b64 = base64.b64encode(digest_code).decode()
        # The Validate step downloads ONE file, so every module validate_entry imports
        # flatly has to travel inside this bootstrap. tests/test_shared_module_shipping.py
        # derives that list from the actual imports and fails if one is missing here.
        capped_code = (validator_src.parent / "capped_reader.py").read_bytes()
        capped_b64 = base64.b64encode(capped_code).decode()
        lineage_b64 = base64.b64encode(
            (validator_src.parent / "training_lineage.py").read_bytes()).decode()

        # embed per-family schemas (input_config_schema +
        # provenance_keys) from each steps/<family>/defaults.json so the strict
        # validator can verify manifest.input_config and provenance per family.
        # Without this, validate_report rejects EVERY family ("unknown
        # model_family"). Single-file code= delivery means we inject these the
        # same way we inject validator.py (base64 into the bootstrap).
        # family_schemas + the canonical suite->task_ids map now
        # come from the registry (single source of truth), NOT an inline glob /
        # hardcoded dict. The registry fails loud on a malformed defaults.json,
        # preserving the old fail-closed behavior (a dropped family would later
        # surface as "unknown model_family" instead of the real cause).
        from .registry import family_schemas as _family_schemas
        from .registry import suite_canonical_task_ids as _suite_canonical
        from .registry import suites_json as _suites_json
        fam_schemas = _family_schemas()
        fs_b64 = base64.b64encode(_json.dumps(fam_schemas).encode()).decode()
        suite_ids_b64 = base64.b64encode(
            _json.dumps(_suite_canonical()).encode()).decode()
        # The full resolved suite table: the validator's suite allowlist plus the
        # per-simulator coherence expectations (Arena task). config/ does not exist
        # inside the Validate container, so it travels the same base64 route as the
        # family schemas rather than being re-derived there.
        suites_b64 = base64.b64encode(
            _json.dumps(_suites_json()).encode()).decode()

        # Inject the validator bootstrap at the top of the file (after docstring/imports)
        bootstrap = f'''
# --- Embedded validator module (injected by stage_validate_code) ---
import base64 as _b64, tempfile as _tf, os as _os
_validator_b64 = "{b64}"
_module_dir = _tf.TemporaryDirectory(prefix="vla-validation-modules-")
_validator_path = _os.path.join(_module_dir.name, "validator.py")
with open(_validator_path, "wb") as _vf:
    _vf.write(_b64.b64decode(_validator_b64))
with open(_os.path.join(_module_dir.name, "digest.py"), "wb") as _df:
    _df.write(_b64.b64decode("{digest_b64}"))
with open(_os.path.join(_module_dir.name, "capped_reader.py"), "wb") as _cf:
    _cf.write(_b64.b64decode("{capped_b64}"))
with open(_os.path.join(_module_dir.name, "training_lineage.py"), "wb") as _lf:
    _lf.write(_b64.b64decode("{lineage_b64}"))
import sys as _sys
_sys.path.insert(0, _module_dir.name)
# per-family schemas for the strict validator (embedded from defaults.json)
_os.environ["VLA_FAMILY_SCHEMAS_JSON"] = _b64.b64decode("{fs_b64}").decode()
# canonical suite->task_ids map (registry-sourced) for "all" expansion
_os.environ["VLA_SUITE_CANONICAL_TASK_IDS_JSON"] = _b64.b64decode("{suite_ids_b64}").decode()
# Resolved suite manifests: suite allowlist + Arena suite/task coherence gate
_os.environ["VLA_SUITES_JSON"] = _b64.b64decode("{suites_b64}").decode()
# --- END embedded validator ---
'''
        from .validation_sdk import sdk_bootstrap
        bootstrap = sdk_bootstrap() + bootstrap
        # Insert before the entrypoint imports or constructs any AWS SDK clients.
        injection_point = "import tarfile\n"
        if injection_point in entry_code:
            entry_code = entry_code.replace(injection_point, injection_point + bootstrap, 1)
        else:
            # Fallback: prepend after docstring
            entry_code = bootstrap + entry_code

        # Deterministic filename (NOT a random tempfile name). upload_code keys on
        # {prefix}/code/<sha256>/<filename>; a random name would defeat the
        # content-addressed skip-if-exists AND change the pipeline-definition code=
        # URI on every deploy even when the bytes are identical. Same content ->
        # same key.
        staged_dir = tempfile.mkdtemp(prefix="vla-validate-")
        staged_path = os.path.join(staged_dir, "validate_entry.py")
        with open(staged_path, "w") as vf:
            vf.write(entry_code)
        return staged_path
    else:
        raise FileNotFoundError(
            f"stage_validate_code: validator.py not found at {validator_src} -- "
            f"the Validate step MUST embed it (single-file code= delivery). "
            f"Refusing to return an unembedded validate_entry.py, which would "
            f"break the trust chain (fail-closed).")


def _deterministic_targz(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as tar:
            for entry in sorted(dir_path.rglob("*")):
                if entry.is_file() and not entry.name.startswith("."):
                    info = tarfile.TarInfo(name=str(entry.relative_to(dir_path)))
                    info.size = entry.stat().st_size
                    info.mtime = 0
                    with entry.open("rb") as source:
                        tar.addfile(info, source)
    return buf.getvalue()


def _publish_content_addressed(cfg: PipelineConfig, content: bytes, filename: str) -> str:
    """Publish immutable code to the component's TRUST bucket, verifying its bytes.

    C4: this wrote the shared Foundation models bucket. The Foundation SageMaker role holds
    PutObject and DeleteObject across that bucket, and the documented toolchain has peer
    components submitting jobs under it -- so a peer worker could replace the validator before
    Validate downloaded it, and the replacement would then execute under the privileged
    validation role. Scoping THIS component's workers away from it did not close that, because
    the exposure was never this component's workers.

    It also skipped the upload whenever the key existed, treating a content-addressed key as a
    guarantee about content. A key is a name: if different bytes were already there, they were
    used unverified. The published bytes are now read back and hashed, so the digest in the key
    is CHECKED against the object rather than assumed.
    """
    digest = hashlib.sha256(content).hexdigest()
    bucket = cfg.trust_bucket
    s3_key = f"code/v1/{digest}/{filename}"
    s3 = boto3.client("s3", region_name=cfg.region)

    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        exists = True
    except s3.exceptions.ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        exists = False

    if not exists:
        try:
            # Conditional creation: a concurrent publisher must not be overwritten, and an
            # object that appeared since the head must not be silently replaced.
            # Sent unconditionally, and that is deliberate. The trust bucket's
            # DenyUnconditionalWritesToProtectedNamespaces statement denies s3:PutObject whenever
            # s3:if-none-match is absent, so omitting it does not weaken the publish -- it makes the
            # bucket refuse it, with an authorization error that hides the real cause. This runs from a
            # developer machine or a launcher, whose boto3 is current; validate_entry runs in an old
            # SageMaker image and guarantees support explicitly before publishing.
            s3.put_object(
                Bucket=bucket, Key=s3_key, Body=content,
                ChecksumSHA256=base64.b64encode(
                    hashlib.sha256(content).digest()).decode("ascii"),
                IfNoneMatch="*")
        except s3.exceptions.ClientError as error:
            if error.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
                raise
            # Published concurrently; fall through to verification.

    published = s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read()
    actual = hashlib.sha256(published).hexdigest()
    if actual != digest:
        raise RuntimeError(
            f"published code at s3://{bucket}/{s3_key} hashes to {actual}, not the {digest} "
            f"named by its key. The object does not contain the code that was published, so "
            f"executing it would run something this deployment did not produce.")
    return f"s3://{bucket}/{s3_key}"


def upload_code(cfg: PipelineConfig, local_path: str) -> str:
    """Publish a single code file to the trust bucket under a content-addressed key."""
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"upload_code: {local_path} does not exist")
    return _publish_content_addressed(cfg, path.read_bytes(), path.name)


def upload_directory(cfg: PipelineConfig, local_dir: str) -> str:
    """Publish a directory as a deterministic tar.gz to the trust bucket."""
    dir_path = Path(local_dir)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"upload_directory: {local_dir} is not a directory")
    return _publish_content_addressed(
        cfg, _deterministic_targz(dir_path), "sourcedir.tar.gz")


def upsert_versioned(pipeline, role_arn: str) -> dict:
    result = pipeline.upsert(role_arn=role_arn)
    if "PipelineVersionId" not in result:
        # CreatePipeline omits the version; update our graph to obtain an atomic version ID.
        result = pipeline.update(role_arn=role_arn)
    # Fail if neither API response identifies the immutable version to execute.
    result["PipelineVersionId"]
    return result


def deploy(cfg: PipelineConfig, validate_code_uri: str | None = None) -> dict:
    """Build the pipeline definition and upsert it. Returns the pipeline ARN.

    `validate_code_uri` is the content-addressed S3 URI for validate_entry.py.
    If None, uses cfg.s3_uri("code", "validate_entry.py") (the definition
    default -- requires the code to already be at that path).
    """
    # Build with a real PipelineSession for upsert. PipelineSession (subclass of
    # Session) is required so the ModelStep's Model.register() CAPTURES step args
    # instead of firing a create_model_package API call at build time.
    from sagemaker.workflow.pipeline_context import PipelineSession

    from .pipeline import build_pipeline

    session = PipelineSession(
        boto_session=boto3.Session(region_name=cfg.region))

    pipeline = build_pipeline(cfg, session=session,
                              validate_code_uri=validate_code_uri)

    return upsert_versioned(pipeline, cfg.role_arn)


def start(
    cfg: PipelineConfig,
    params: dict[str, str],
    pipeline_name: str | None = None,
    *,
    pipeline_version_id: int,
) -> dict:
    """Start a pipeline execution with the given parameters.

    Returns a dict with 'ExecutionArn' and convenience fields.
    """
    sm = boto3.client("sagemaker", region_name=cfg.region)
    name = pipeline_name or cfg.pipeline_name

    # Convert params to the API format
    pipeline_params = [
        {"Name": k, "Value": str(v)} for k, v in params.items()
    ]

    response = sm.start_pipeline_execution(
        PipelineName=name,
        PipelineVersionId=pipeline_version_id,
        PipelineParameters=pipeline_params,
    )

    return {
        "ExecutionArn": response["PipelineExecutionArn"],
        "PipelineName": name,
        "Parameters": params,
    }


def wait_for_execution(
    cfg: PipelineConfig,
    execution_arn: str,
    poll_seconds: int = 30,
    timeout_seconds: int = 14400,  # 4 hours
) -> str:
    """Poll until execution finishes. Returns final status string."""
    sm = boto3.client("sagemaker", region_name=cfg.region)
    start_time = time.time()

    while True:
        resp = sm.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn)
        status = resp["PipelineExecutionStatus"]

        if status in ("Succeeded", "Failed", "Stopped"):
            return status

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Execution {execution_arn} still {status} after "
                f"{elapsed:.0f}s (timeout={timeout_seconds}s)")

        time.sleep(poll_seconds)


def describe_steps(cfg: PipelineConfig, execution_arn: str) -> list[dict]:
    """List all steps in an execution with their statuses."""
    sm = boto3.client("sagemaker", region_name=cfg.region)
    resp = sm.list_pipeline_execution_steps(
        PipelineExecutionArn=execution_arn)
    return resp.get("PipelineExecutionSteps", [])


def describe_execution(cfg: PipelineConfig, execution_arn: str) -> dict:
    """Get full execution metadata."""
    sm = boto3.client("sagemaker", region_name=cfg.region)
    return sm.describe_pipeline_execution(
        PipelineExecutionArn=execution_arn)


def build_plumbing_pipeline(cfg: PipelineConfig, session, plumbing_dir: str,
                            sabotage: bool = False) -> Pipeline:
    """Build a plumbing variant of the pipeline for M1 testing.

    Same graph shape (FineTune -> SimEval -> Validate -> SuccessGate) but:
    - All instances ml.m5.xlarge (CPU, pennies)
    - Uses SKLearn estimator in script mode (entry_point = plumbing scripts)
    - Validate code = plumb_validate.py

    This validates the pipeline MECHANICS (artifact hand-off, PropertyFile/JsonGet
    resolution, gate branching) without any GPU spend.
    """
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
    from sagemaker.workflow.functions import JsonGet
    from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

    plumb = Path(plumbing_dir)
    eval_script = "plumb_eval_sabotage.py" if sabotage else "plumb_eval.py"

    # Parameters (same as production but with plumbing defaults)
    threshold = ParameterFloat(name="SuccessThreshold", default_value=0.5)
    eval_seed = ParameterInteger(name="EvalSeed", default_value=1000)
    # A single episode is not a measurement: the rate can only be 0.0 or 1.0, and a sample run must still produce a rate that means something. Three is the floor for any episode count.
    eval_trials = ParameterInteger(name="EvalTrials", default_value=3)

    # --- FineTune (Training, CPU) ---
    train_estimator = SKLearn(
        entry_point="plumb_train.py",
        source_dir=str(plumb),
        # I15: the rehearsal must exercise the SAME identity split as production, or it
        # proves nothing about the permissions the real run needs. FineTune has its own
        # training role; using the evaluation role here rehearsed a graph that cannot
        # exist in production.
        role=cfg.training_role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        framework_version="1.2-1",
        output_path=cfg.s3_uri("plumbing/train"),
        sagemaker_session=session,
    )
    train_step = TrainingStep(
        name="FineTune", estimator=train_estimator,
        cache_config=CacheConfig(enable_caching=False))

    # --- SimEval (Training, CPU) ---
    # Wire FineTune's ModelArtifacts as the "model" input channel so eval
    # can read it at SM_CHANNEL_MODEL
    from sagemaker.inputs import TrainingInput

    eval_estimator = SKLearn(
        entry_point=eval_script,
        source_dir=str(plumb),
        # C1: job execution identity -- the workload role, matching production.
        role=cfg.workload_role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        framework_version="1.2-1",
        output_path=cfg.s3_uri("plumbing/eval"),
        sagemaker_session=session,
        environment={
            "EVAL_SEED": eval_seed.to_string(),
            "EVAL_TRIALS": eval_trials.to_string(),
            "EVAL_TASK_IDS": "[0,1]",
        },
    )
    eval_step = TrainingStep(
        name="SimEval", estimator=eval_estimator,
        inputs={
            "model": TrainingInput(
                s3_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            ),
        },
        cache_config=CacheConfig(enable_caching=False))

    # --- Validate (Processing, CPU) ---
    validated_prop = PropertyFile(
        name="ValidatedMetrics", output_name="validated",
        path="validated_metrics.json")
    from sagemaker.image_uris import retrieve as _retrieve_image
    _sklearn_image = _retrieve_image("sklearn", cfg.region, version="1.2-1", instance_type="ml.m5.xlarge")
    validator = ScriptProcessor(
        image_uri=_sklearn_image,
        command=["python3"],
        # C1: Validate runs under the VALIDATION role in production, and the
        # plumbing graph must exercise the same identity split -- otherwise the
        # rehearsal proves nothing about the permissions the real run needs.
        role=cfg.validation_role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
    )
    validate_step = ProcessingStep(
        name="Validate",
        processor=validator,
        inputs=[ProcessingInput(
            source=eval_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/eval_output")],
        outputs=[ProcessingOutput(
            output_name="validated",
            source="/opt/ml/processing/output",
            # I15: the validation role may write validated/* only, so "plumbing/validated"
            # would have been DENIED at runtime. Kept inside the allowed namespace
            # rather than widening the grant to accommodate a rehearsal.
            destination=cfg.s3_uri("validated/plumbing"))],
        property_files=[validated_prop],
        cache_config=CacheConfig(enable_caching=False),
        code=str(plumb / "plumb_validate.py"),
    )

    # --- Gate ---
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=validate_step.name,
                     property_file=validated_prop,
                     json_path="success_rate"),
        right=threshold,
    )
    # I15: both branches were empty, so a FALSE threshold condition still produced a
    # successful pipeline execution -- the rehearsal could not demonstrate that the
    # gate blocks anything, which is the one behaviour it exists to rehearse.
    from sagemaker.workflow.fail_step import FailStep

    fail_step = FailStep(
        name="PlumbingBelowThreshold",
        error_message="plumbing success_rate did not meet SuccessThreshold",
    )
    gate_step = ConditionStep(
        name="SuccessGate",
        conditions=[condition],
        if_steps=[],
        else_steps=[fail_step],
    )

    return Pipeline(
        name=f"{cfg.pipeline_name}-plumbing",
        parameters=[threshold, eval_seed, eval_trials],
        steps=[train_step, eval_step, validate_step, gate_step],
        sagemaker_session=session,
    )
