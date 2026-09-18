"""The VLA model-evaluation SageMaker Pipeline -- the evaluation workflow.

This builds an actual `sagemaker.workflow.pipeline.Pipeline` from typed SDK
step objects (NOT a hand-assembled JSON dict). The graph:

    [FineTune]  TrainingStep  -- LoRA fine-tune at (ModelFamily, TrainSteps)
        |         produces model.tar.gz (step output, resolved at runtime)
        v
    [SimEval]   TrainingStep  -- mount the trained model, run LIBERO or Arena episodes,
        |         write metrics.json + evidence to SM_MODEL_DIR so the output
        |         is exposed as a step property (ModelArtifacts.S3ModelArtifacts)
        v
    [Validate]  ProcessingStep (CPU) -- mounts SimEval's ModelArtifacts,
        |         runs schema-v3 validation + digest re-check against
        |         pipeline-owned expectations (EvalSeed/Trials/TaskIds params),
        |         emits validated_metrics.json + PropertyFile(success_rate)
        |         ONLY if all checks pass. Load-bearing, not advisory.
        v
    [Gate]      ConditionStep -- success_rate >= threshold ?
        |  yes                                 |  no
        v                                      v
    [Register]  ModelStep(RegisterModel)     (stop; nothing registered)

Key properties (all verifiable offline, no AWS account needed):
  * ModelFamily / TrainSteps / Suite / DatasetS3Uri / EvalSeed / EvalTrials /
    EvalTaskIds are Pipeline PARAMETERS.
  * SimEval consumes FineTune's ModelArtifacts via step-property reference.
  * Validate consumes SimEval's ModelArtifacts via step-property reference.
  * The gate reads the VALIDATOR'S output (validated_metrics.json), never the
    raw eval output directly -- structurally impossible to gate on unvalidated.
  * TRAIN_SUITE reaches FineTune for all families and records the requested suite.
    Training cache reuse is disabled.
  * Content-addressed code URIs for both train + eval: .../code/<sha256>/...
    identify the staged source independently of mutable image tags.
  * `build_pipeline(cfg).definition()` serializes to JSON with no credentials.

Architecture: SimEval runs as a TrainingStep (reuses the g6e training
quota) followed by a CPU Validate ProcessingStep. Nothing fabricates a result.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

if TYPE_CHECKING:
    from .config import PipelineConfig


def _register_session(session, cfg):
    """Return a PipelineSession for the register Model so ``Model.register()``
    CAPTURES step args instead of firing a real ``create_model_package`` API call.

    ``Model.register()`` only defers (captures) under a ``PipelineSession``; with a
    plain ``Session`` or ``None`` it executes against AWS at build time -- that was
    the bug that broke offline ``.definition()`` (22 tests) and would hit AWS on a
    live deploy. We reuse the caller's ``PipelineSession`` when given one; otherwise
    we build a local, capture-only ``PipelineSession`` pinned to ``cfg.region`` (with
    an explicit ``default_bucket`` so no STS / default-bucket network resolution
    happens offline). ``PipelineSession`` is a subclass of ``Session``, so this is
    safe everywhere a ``Session`` is expected.
    """
    from sagemaker.workflow.pipeline_context import PipelineSession
    if isinstance(session, PipelineSession):
        return session
    import boto3
    boto_sess = getattr(session, "boto_session", None) or boto3.Session(
        region_name=cfg.region)
    return PipelineSession(boto_session=boto_sess, default_bucket=cfg.bucket)


def build_parameters(*, through="RegisterModel", checkpoint_s3_uri=None) -> dict:
    """Pipeline parameters -- the knobs a run sets. 'model as a parameter.'"""
    params = {
        # Default aligned with the registry_group default below (both gr00t) so
        # an all-defaults execution is internally coherent; launchers pass the
        # real family + group from the resolved pair manifest.
        "model_family": ParameterString(
            name="ModelFamily", default_value="gr00t"),
        "registry_group": ParameterString(
            name="ModelPackageGroupName", default_value="vla-eval-gr00t"),
        "train_steps": ParameterInteger(name="TrainSteps", default_value=2000),
        # Dose-curve: checkpoint save interval for the single-run-multi-checkpoint
        # design. Default 1000000 (>> any real MAX_STEPS) => one final checkpoint =
        # historical final-only behavior, so existing sample/showcase runs are unchanged.
        # A curve run sets this to an interval (e.g. 100) so ONE trajectory saves
        # intermediates (train_entry stages them under hidden .dose_checkpoints/).
        "save_steps": ParameterInteger(name="TrainSaveSteps", default_value=1000000),
        "suite": ParameterString(
            name="Suite", default_value="libero_spatial"),
        "train_suite": ParameterString(name="TrainSuite"),
        # REQUIRED (no default): fail LOUDLY at StartPipelineExecution, not
        # deep in job creation.
        "dataset_s3_uri": ParameterString(name="DatasetS3Uri"),
        "dataset_revision": ParameterString(
            name="DatasetRevision", default_value="__FROM_SUITE_MANIFEST__"),
        "train_image": ParameterString(name="TrainImageUri"),
        "eval_image": ParameterString(name="EvalImageUri"),
        # NO DEFAULTS on any of the four hardware parameters. A GENERIC pipeline graph cannot choose
        # a FAMILY's resources: the family is selected per execution, and the value that is correct
        # for one is wrong for another. Defaults here meant a caller who omitted them got
        # ml.g6e.12xlarge and 100 GB silently -- which is how a family declaring 300 GB ran on 100,
        # and how the launcher fix that made manifests authoritative could still be bypassed by
        # submitting directly. Required parameters force the caller to resolve them through
        # registry.select_instance / select_volume_gb, which validate against the declaration.
        #
        # A ParameterString/Integer with no default_value is REQUIRED at StartPipelineExecution:
        # omitting it fails the call rather than substituting a figure nobody chose.
        "train_instance": ParameterString(name="TrainInstanceType"),
        "eval_instance": ParameterString(name="EvalInstanceType"),
        # Per ROLE: train and eval run on different instance types with different storage limits, so
        # one shared figure cannot be checked against both. openvla declares 300 train / 200 eval.
        "volume_size": ParameterInteger(name="VolumeSizeInGB"),
        "eval_volume_size": ParameterInteger(name="EvalVolumeSizeInGB"),
        # I12: defaults.json declares max_runtime_seconds (28800) and max_run appeared
        # ZERO times here, so the declared budget bound nothing -- a stalled job retained
        # paid accelerator capacity until a service limit intervened, long after runner.py
        # stopped waiting. A caller-side wait timeout ends the wait, not the bill.
        "max_runtime_seconds": ParameterInteger(
            name="MaxRuntimeSeconds", default_value=28800),
        # --- Eval protocol parameters (validator checks AGAINST these) ---
        # These make expectations pipeline-owned, never derived from the report.
        # Also: sample path (2 tasks x 3 trials) is a parameter change, not code.
        "eval_seed": ParameterInteger(name="EvalSeed", default_value=100),
        "eval_trials": ParameterInteger(name="EvalTrials", default_value=200),
        # "all" = all tasks (full protocol, registry-eligible).
        # Non-"all" = task subset (sample/debug, informational only).
        # NOTE: SageMaker rejects empty strings for parameters (min length 1).
        "eval_task_ids": ParameterString(name="EvalTaskIds", default_value="all"),
        # Dose-curve: comma list of intermediate checkpoint steps to roll out
        # (subset of what train saved), or "all" = eval every staged checkpoint.
        # NOTE: must be NON-EMPTY -- SageMaker StartPipelineExecution rejects an
        # empty-string parameter value ("string too short"). "all" + no
        # .dose_checkpoints/ => today's single-checkpoint eval.
        "eval_dose_steps": ParameterString(name="EvalDoseSteps", default_value="all"),
        # The capability gate threshold. Default 0.0 = plumbing proof (any score
        # passes); launchers set a real bar explicitly. Matches the documented
        # SuccessGate contract in the README.
        "success_threshold": ParameterFloat(
            name="SuccessThreshold", default_value=0.0),
        # --- Arena eval per-run knobs, collapsed into ONE JSON blob ---
        # The five discrete Arena knobs (embodiment_tag, task_name,
        # policy_config_yaml, arena_embodiment, object) are now one
        # EvalSimConfig JSON string the eval entry unpacks. Default "{}" -> the
        # eval entry keeps its own hardcoded defaults (the LIBERO/sample path
        # never set these). use_groot_server + arena_connector stay discrete
        # (below): the shell entrypoint routes on them before eval_entry runs.
        "eval_sim_config": ParameterString(
            name="EvalSimConfig", default_value="{}"),
        # Individual resolved Arena knobs for Validate comparison. Each value
        # is bounded (<256 chars for Processing env). LIBERO path leaves these
        # at their empty defaults; Arena launchers populate them from the
        # resolver output (same values as EvalSimConfig, unbundled).
        "expected_embodiment_tag": ParameterString(
            name="ExpectedEmbodimentTag", default_value=""),
        "expected_arena_embodiment": ParameterString(
            name="ExpectedArenaEmbodiment", default_value=""),
        "expected_arena_object": ParameterString(
            name="ExpectedArenaObject", default_value=""),
        "expected_policy_config": ParameterString(
            name="ExpectedPolicyConfig", default_value=""),
        # --- Per-family code delivery (content-addressed sourcedir tarballs) ---
        # The runner uploads family-specific sourcedirs and passes these URIs.
        # Images = environment; sourcedir = entry scripts.
        "train_source_dir": ParameterString(name="TrainSourceDirUri"),
        "eval_source_dir": ParameterString(name="EvalSourceDirUri"),
        # --- Isaac Arena backend control ---
        # "true" = start GR00T inference server in container (real eval).
        # "false" = zero_action POC mode (validates infra, always 0% success).
        "use_groot_server": ParameterString(
            name="UseGrootServer", default_value="false"),
        # The only shipped Arena connector is the native GR00T server.
        "arena_connector": ParameterString(
            name="ArenaConnector", default_value="groot", enum_values=["groot"]),
        # GR00T version selector. n17 (default, UNTOUCHED) = base GR00T-N1.7-3B
        # @376ba890, Arena GR1 -> new_embodiment + n17->n16 action shim at eval.
        # n16 = base GR00T-N1.6-3B @5dc80c4 (n1.6.1-release), native EmbodimentTag.GR1,
        # native Gr00tN1d6 eval server (no shim). Reaches FineTune as GR00T_VERSION and
        # SimEval as EVAL_GR00T_VERSION.
        # enum_values so a hand-started StartPipelineExecution is rejected at
        # submission rather than reaching Validate with a value no suite manifest
        # keys (which would silently no-op the embodiment coherence gate).
        "gr00t_version": ParameterString(
            name="Gr00tVersion", default_value="n17",
            enum_values=["n16", "n17"]),
    }
    from .workflow import parameter_keys, selected_steps

    steps = selected_steps(through, checkpoint=checkpoint_s3_uri is not None)
    if through != "RegisterModel":
        used = parameter_keys(steps)
        params = {key: value for key, value in params.items() if key in used}
    elif checkpoint_s3_uri is not None:
        train_only = {"train_steps", "save_steps", "train_image", "train_source_dir",
                      "train_instance", "dataset_s3_uri", "train_suite"}
        params = {key: value for key, value in params.items() if key not in train_only}
    if checkpoint_s3_uri is not None:
        params["checkpoint_s3_uri"] = ParameterString(
            name="CheckpointS3Uri", default_value=checkpoint_s3_uri)
    return params


def build_pipeline(cfg: PipelineConfig, session=None, validate_code_uri=None,
                   checkpoint_s3_uri=None, through="RegisterModel") -> Pipeline:
    """Assemble the Pipeline object for `cfg`.

    `session` is an optional PipelineSession; when None the pipeline can still
    be constructed and `.definition()`-serialized offline (this is what the
    tests exercise).

    `checkpoint_s3_uri` selects the pipeline VARIANT (eval-only mode):
      * None (default) -> full train-default graph
        (FineTune -> SimEval -> Validate -> SuccessGate -> RegisterModel).
      * an S3 URI (the exact model.tar.gz object key) -> eval-only / checkpoint-skip
        graph: NO FineTune. SimEval mounts the provided checkpoint; RegisterModel
        points to its validated, promoted copy. The weights-digest chain still runs over
        the mounted bytes (integrity of evaluated bytes vs the checkpoint's bundled
        manifest -- NOT provenance: on this path the manifest is part of the input,
        so the PendingManualApproval approver must verify lineage out-of-band; the
        input_checkpoint_uri is recorded on the registered package to enable that).
        Precondition: the checkpoint lives in a versioned bucket (the identity leg
        needs a VersionId). SageMaker DAGs are static, so this is a distinct graph
        the runner selects -- not a runtime skip.

    `through` selects a dependency-preserving prefix. The default retains the
    existing complete graph. Omitted steps are never constructed or submitted.
    """
    from sagemaker.estimator import Estimator
    from sagemaker.model_metrics import MetricsSource, ModelMetrics
    from sagemaker.processing import (
        ProcessingInput,
        ProcessingOutput,
        ScriptProcessor,
    )

    from .workflow import selected_steps

    requested = selected_steps(through, checkpoint=checkpoint_s3_uri is not None)
    params = build_parameters(through=through, checkpoint_s3_uri=checkpoint_s3_uri)

    def finish(steps):
        return Pipeline(name=cfg.pipeline_name, parameters=list(params.values()),
                        steps=steps, sagemaker_session=session)

    # --- Caching policy (SECURITY-RELEVANT, not just performance) -----------
    # FineTune: OFF -- dataset revisions are resolved inside the trainer,
    # AFTER SageMaker decides cache eligibility. A named ref (e.g. "main")
    # can point to different bytes across runs while producing the same
    # cache key. Disabled until DatasetRevision carries an immutable SHA
    # at submission time.
    # SimEval: OFF -- MANDATORY. "Evaluated bytes == produced bytes" holds
    # only with caching off; a cached SimEval resolves Get-refs to a PRIOR
    # run's output. Also: stochastic eval must never be cache-reused.
    # Validate: OFF -- must always re-validate fresh eval output.
    train_cache = CacheConfig(enable_caching=False)
    eval_cache = CacheConfig(enable_caching=False)
    validate_cache = CacheConfig(enable_caching=False)

    # --- Mode: train-default vs eval-only (checkpoint-skip) -----------------
    eval_only = checkpoint_s3_uri is not None
    ckpt_param = params.get("checkpoint_s3_uri")

    # --- FineTune (TrainingStep) -- OMITTED in eval-only --------------------
    # TRAIN_SUITE records the training selection independently of the eval suite.
    # OpenVLA/GR00T use per-suite datasets; MolmoAct2 uses "unified".
    # Training cache reuse remains disabled for every family.
    train_step = None
    estimator = None
    if not eval_only:
        estimator = Estimator(
            image_uri=params["train_image"],
            # C4: FineTune runs as the component WORKLOAD role, not the
            # Foundation role. The Foundation role holds Put/DeleteObject over
            # the models bucket that also carries the staged validation code, so
            # a training worker could overwrite the validator that judges it.
            #
            # FineTune has its OWN identity. Sharing one workload role with SimEval
            # scoped writes away from the validator but still let a training worker
            # write under eval/ -- manufacturing or replacing the raw evaluation
            # evidence Validate judges. This role cannot write eval/ at all.
            role=cfg.training_role_arn,
            instance_count=1,
            instance_type=params["train_instance"],
            volume_size=params["volume_size"],
            # I12: the declared runtime budget, actually enforced by the service.
            max_run=params["max_runtime_seconds"],
            output_path=cfg.s3_uri("train"),
            sagemaker_session=session,
            # Entry-script delivery: sourcedir tarball passed via hyperparameters.
            # The DLC training toolkit fetches sagemaker_submit_directory at job
            # start and runs sagemaker_program as the entry point.
            hyperparameters={
                "sagemaker_submit_directory": params["train_source_dir"].to_string(),
                "sagemaker_program": "train_entry.py",
            },
            environment={
                "TRAIN_MODEL_FAMILY": params["model_family"].to_string(),
                "TRAIN_MAX_STEPS": params["train_steps"].to_string(),
                "TRAIN_SAVE_STEPS": params["save_steps"].to_string(),
                "TRAIN_SUITE": params["train_suite"].to_string(),
                "TRAIN_DATASET_S3URI": params["dataset_s3_uri"].to_string(),
                "GR00T_VERSION": params["gr00t_version"].to_string(),
                "TRAIN_DATASET_REVISION": params["dataset_revision"].to_string(),
                "HF_SECRET_NAME": cfg.hf_secret_name,
                # I12: the SAME declared budget max_run enforces, handed to the container so a
                # subprocess timeout can be derived from it instead of a second, disagreeing
                # constant. Without this the entrypoints cannot see the budget at all.
                "VLA_MAX_RUNTIME_SECONDS": params["max_runtime_seconds"].to_string(),
            },
        )
        train_step = TrainingStep(
            name="FineTune", estimator=estimator, cache_config=train_cache)

    steps = [train_step] if train_step is not None else []
    if "SimEval" not in requested:
        return finish(steps)

    # SimEval + RegisterModel source the model from FineTune's output (train
    # mode) or the provided checkpoint param (eval-only).
    model_source = (train_step.properties.ModelArtifacts.S3ModelArtifacts
                    if not eval_only else ckpt_param)

    # --- SimEval (TrainingStep -- GPU eval) ---------------------------------
    # Why TrainingStep, not ProcessingStep:
    #   1. Reuses the proven g6e Training quota (Processing quota is 0).
    #   2. Matches old harness topology (eval ran as training jobs) so
    #      eval_entry ports with SM_CHANNEL_MODEL / SM_MODEL_DIR conventions.
    #   3. PropertyFile is Processing-only, so we need a separate validation
    #      step anyway -- that's where the gate proof lives.
    #
    # The eval writes metrics.json + small evidence to SM_MODEL_DIR (NOT
    # SM_OUTPUT_DATA_DIR) so the output is exposed as a step property
    # (ModelArtifacts.S3ModelArtifacts) and the validation step consumes it
    # via a structural step-property reference. Videos/logs go to
    # SM_OUTPUT_DATA_DIR and don't pollute the model channel.
    eval_estimator = Estimator(
        image_uri=params["eval_image"],
        # SimEval is the EVALUATION worker: it writes eval/ and cannot write train/.
        # Distinct from FineTune so producing the evidence and producing the weights
        # are attributable to different identities.
        role=cfg.workload_role_arn,
        instance_count=1,
        instance_type=params["eval_instance"],
        volume_size=params["eval_volume_size"],
        # I12: the declared runtime budget, actually enforced by the service.
        max_run=params["max_runtime_seconds"],
        # C3: raw evaluation evidence goes to the component-owned handoff bucket, not the
        # shared Foundation models bucket. Foundation holds PutObject and DeleteObject
        # across that bucket and peer components submit jobs under it, so a peer could
        # replace this archive between SimEval finishing and Validate downloading --
        # keeping the genuine checkpoint identity while substituting the score. Every
        # check Validate performs authenticates the model BYTES; none authenticated the
        # ORIGIN of the result.
        output_path=cfg.handoff_uri("eval/v1"),
        sagemaker_session=session,
        hyperparameters={
            "sagemaker_submit_directory": params["eval_source_dir"].to_string(),
            "sagemaker_program": "eval_entry.py",
            "USE_GROOT_SERVER": params["use_groot_server"].to_string(),
        },
        environment={
            "EVAL_MODEL_FAMILY": params["model_family"].to_string(),
            # I12: the SAME declared budget max_run enforces, handed to the container so a
            # subprocess timeout can be derived from it instead of a second, disagreeing
            # constant. Without this the entrypoints cannot see the budget at all.
            "VLA_MAX_RUNTIME_SECONDS": params["max_runtime_seconds"].to_string(),
            "EVAL_SUITE": params["suite"].to_string(),
            "EVAL_SEED": params["eval_seed"].to_string(),
            "EVAL_TRIALS": params["eval_trials"].to_string(),
            "EVAL_TASK_IDS": params["eval_task_ids"].to_string(),
            "EVAL_DOSE_STEPS": params["eval_dose_steps"].to_string(),
            # eval_entry requires these for pipeline mode
            "EVAL_CHECKPOINT": "/opt/ml/input/data/model",
            "EVAL_CKPT_REV": "",  # local checkpoint, not HF
            "EVAL_MODEL_SOURCE_URI": model_source.to_string(),
            "MUJOCO_GL": "egl",
            "HF_SECRET_NAME": cfg.hf_secret_name,
            "USE_GROOT_SERVER": params["use_groot_server"].to_string(),
            # Isaac Arena v5's custom ENTRYPOINT bypasses SageMaker toolkit,
            # so SM_HP_* vars from hyperparameters are never set. The baked
            # eval_entry.py reads SM_HP_USE_GROOT_SERVER, so we set it directly.
            "SM_HP_USE_GROOT_SERVER": params["use_groot_server"].to_string(),
            # n16 -> eval_entry serves the checkpoint via the native Gr00tN1d6 GR1
            # server (no shim); n17 (default) unchanged.
            "EVAL_GR00T_VERSION": params["gr00t_version"].to_string(),
            # Arena connector routing -- docker_entrypoint_multi.sh reads this
            "ARENA_CONNECTOR": params["arena_connector"].to_string(),
            # Arena per-run knobs collapsed into one JSON blob the eval entry
            # unpacks. Only this one var is injected -- no more
            # all-sim union of discrete SM_HP_*/EVAL_* knobs. eval_entry seeds the
            # individual SM_HP_*/EVAL_* keys from the blob at import.
            "EVAL_SIM_CONFIG": params["eval_sim_config"].to_string(),
        },
    )
    # Feed FineTune's trained model as an input channel so the eval mounts it
    # at SM_CHANNEL_MODEL. add_depends_on establishes ordering but
    # NOT data flow -- the model channel must be an explicit TrainingInput.
    from sagemaker.inputs import TrainingInput
    eval_step = TrainingStep(
        name="SimEval", estimator=eval_estimator,
        inputs={
            "model": TrainingInput(
                s3_data=model_source,
            ),
        },
        cache_config=eval_cache)
    steps.append(eval_step)
    if "Validate" not in requested:
        return finish(steps)

    # --- Validate (ProcessingStep -- CPU, load-bearing gate proof) ----------
    # Consumes SimEval's ModelArtifacts (which contains metrics.json +
    # evidence). Runs schema-v3 validation + digest re-check against
    # pipeline-owned expectations (EvalSeed/Trials/TaskIds from params).
    # Emits validated_metrics.json + PropertyFile ONLY if checks pass.
    # Structurally impossible to gate on an unvalidated number.
    validated_prop = PropertyFile(
        name="ValidatedMetrics", output_name="validated",
        path="validated_metrics.json")
    # Use the SDK to resolve the sklearn image URI (region-aware, no hardcoded account).
    from sagemaker.image_uris import retrieve as _retrieve_image
    _sklearn_image = _retrieve_image("sklearn", cfg.region, version="1.2-1", instance_type="ml.m5.large")
    validator = ScriptProcessor(
        image_uri=_sklearn_image,
        command=["python3"],
        # C4/C5: Validate runs as the VALIDATION role -- the only identity
        # permitted to publish into the trust bucket, and one that cannot replace
        # the validation code it runs.
        role=cfg.validation_role_arn,
        instance_count=1,
        instance_type="ml.m5.large",
        volume_size_in_gb=params["volume_size"],
        sagemaker_session=session,
        # Pipeline-owned expectations passed to the validator via env.
        # Without these, every expectation check silently skips (vacuous pass).
        env={
            "EXPECTED_EVAL_SEED": params["eval_seed"].to_string(),
            "EXPECTED_EVAL_TRIALS": params["eval_trials"].to_string(),
            "EXPECTED_EVAL_TASK_IDS": params["eval_task_ids"].to_string(),
            "EXPECTED_MODEL_FAMILY": params["model_family"].to_string(),
            "EXPECTED_SUITE": params["suite"].to_string(),
            "EXPECTED_MODEL_SOURCE_URI": model_source.to_string(),
            # cycle-15 I1: evaluator_image_uri does not identify the complete LIBERO evaluator --
            # the graph supplies its executable sourcedir separately, so two runs sharing an image
            # but running different eval code attested identically. This is the URI the graph passed
            # to SimEval, so the receipt names the code that actually ran.
            "VLA_EVAL_SOURCEDIR_URI": params["eval_source_dir"].to_string(),
            # I8: the immutable attestation certified a score without naming the code
            # that produced it -- validate_entry.py had ZERO references to any image
            # identity. An immutable tag prevents overwriting a tag; it does not record
            # WHICH image ran, so a current launcher and validator could evaluate with an
            # older evaluator image and the receipt would look identical.
            "EVALUATOR_IMAGE_URI": params["eval_image"].to_string(),
            "EXPECTED_FAMILY_VERSION": params["gr00t_version"].to_string(),
            "EXPECTED_EMBODIMENT_TAG": params["expected_embodiment_tag"].to_string(),
            "EXPECTED_ARENA_EMBODIMENT": params["expected_arena_embodiment"].to_string(),
            "EXPECTED_ARENA_OBJECT": params["expected_arena_object"].to_string(),
            "EXPECTED_POLICY_CONFIG": params["expected_policy_config"].to_string(),
            # The execution's TRAINING contract. Validate previously received only
            # evaluation expectations, so nothing compared the checkpoint's own
            # provenance against what this execution actually asked FineTune to do: a
            # checkpoint could declare a different training dose or a different
            # dataset and still satisfy every check. TrainSuite is exposed
            # independently of Suite, so a manually started execution could train on
            # one GR1 task and evaluate another while passing the embodiment checks.
            #
            # EXPECTED_TRAIN_PATH distinguishes the two provenance contracts: on
            # "train" the checkpoint was produced by THIS execution and must match
            # these parameters; on "eval_only" it was supplied from outside, so these
            # parameters describe nothing about it and comparing them would be
            # meaningless. Validate records which contract it applied rather than
            # silently skipping the checks.
            "EXPECTED_TRAIN_PATH": "eval_only" if eval_only else "train",
            # The training expectations are supplied ONLY on the train graph. The eval-only
            # graph removes TrainSteps/TrainSuite/DatasetS3Uri from its
            # declared parameters, so referencing them here produced a definition with
            # undeclared Parameters.* references -- rejected at upsert, before any job
            # runs. check_training_contract()'s early return for eval_only cannot fix a
            # definition that must be accepted first.
            **({} if eval_only else {
                "EXPECTED_TRAIN_STEPS": params["train_steps"].to_string(),
                "EXPECTED_TRAIN_SUITE": params["train_suite"].to_string(),
                "EXPECTED_DATASET_S3URI": params["dataset_s3_uri"].to_string(),
                "EXPECTED_DATASET_REVISION": params["dataset_revision"].to_string(),
            }),
            # C5: where Validate publishes the promoted artifact and its attestation, and
            # the execution the evidence belongs to. Absent, Validate refuses to emit a
            # receipt rather than letting registration fall back to the unpromoted source.
            "TRUST_BUCKET": cfg.trust_bucket,
            "PIPELINE_EXECUTION_ID": ExecutionVariables.PIPELINE_EXECUTION_ID,
            "EXPECTED_SUCCESS_THRESHOLD": params["success_threshold"].to_string(),
        },
    )
    validate_step = ProcessingStep(
        name="Validate",
        processor=validator,
        inputs=[ProcessingInput(
            source=eval_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/eval_output"),
            ProcessingInput(source=model_source,
                            destination="/opt/ml/processing/checkpoint")],
        outputs=[ProcessingOutput(
            output_name="validated",
            source="/opt/ml/processing/output",
            destination=Join(on="/", values=[
                # C3: the gate's own receipt was equally replaceable, so a peer worker
                # could manufacture a validated result. Only the validation role may
                # publish this namespace.
                cfg.handoff_uri("validated/v1"),
                ExecutionVariables.PIPELINE_EXECUTION_ID,
            ]))],
        property_files=[validated_prop],
        cache_config=validate_cache,
        # Consume the content-addressed URI the runner uploaded via
        # stage_validate_code() + upload_code() (immutable, fresh-account-resolvable).
        # upload_code writes {prefix}/code/<sha256>/validate_entry.py, NOT the flat
        # {prefix}/code/validate_entry.py this used to point at -- so the old key was
        # never written and a fresh-account deploy had no object there. Fall back to
        # the flat convention key ONLY for offline .definition()/tests (no upload);
        # a real deploy MUST pass validate_code_uri.
        code=validate_code_uri or cfg.s3_uri("code", "validate_entry.py"),
    )
    steps.append(validate_step)
    if "SuccessGate" not in requested:
        return finish(steps)

    # --- Gate (ConditionStep) -----------------------------------------------
    # Gates on the VALIDATOR'S output, not the raw eval metrics.
    # Two conditions must BOTH be true:
    #   1. success_rate >= threshold
    #   2. validation_passed == true (the validator already rejects zero_action
    #      and episodes=0, so if validated_metrics.json exists the run is real)
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=validate_step.name,
                     property_file=validated_prop,
                     json_path="success_rate"),
        right=params["success_threshold"],
    )
    gate_fail = FailStep(
        name="EvaluationBelowThreshold",
        error_message="Evaluation success rate below the configured threshold.",
    )
    if "RegisterModel" not in requested:
        return finish([*steps, ConditionStep(
            name="SuccessGate", conditions=[condition], if_steps=[], else_steps=[gate_fail],
        )])

    # --- RegisterModel via ModelStep (fires on every gate pass) --------------
    # The model-package group is supplied by the caller as a pipeline parameter
    # (ModelPackageGroupName), resolved from the pair manifest's
    # registry_group_prefix + model_family. The runner ensures that same group
    # exists before starting the execution, so ensured group == registered group.
    # full_suite is metadata on the package, not a condition for registration.
    #
    # Migrated off the deprecated ``step_collections.RegisterModel`` to
    # ``ModelStep`` fed by ``Model.register()``. ``Model.register()`` CAPTURES
    # step args (rather than firing a real ``create_model_package`` API call)
    # ONLY when the Model's session is a ``PipelineSession`` -- so the register
    # Model is built on ``_register_session(session, cfg)`` (the caller's
    # PipelineSession on a live deploy; a local capture-only one offline). The
    # emitted sub-step is named ``RegisterModel-RegisterModel``.
    from sagemaker.model import Model
    from sagemaker.workflow.model_step import ModelStep

    model_package_group = params["registry_group"]

    # The registered model's inference image + weights:
    #   * train path: train image + FineTune's ModelArtifacts.
    #   * eval-only:  eval image + the provided checkpoint param (train_image /
    #     the FineTune estimator do not exist in this variant; train_image is
    #     dropped from the param list below, so referencing it would be an
    #     undeclared-parameter error).
    register_image = params["eval_image"] if eval_only else params["train_image"]
    # Register the PROMOTED artifact, read from Validate's receipt.
    #
    # This used to be the FineTune artifact's plain, unversioned URI (or the checkpoint
    # parameter on the eval-only path), so the registered package resolved to whatever
    # occupied that key at resolution time rather than the bytes that passed validation.
    # Validate verified the mounted bytes and recorded their identity but never promoted
    # the object.
    #
    # The model-package API has no VersionId field -- neither
    # ModelPackageContainerDefinition nor S3ModelDataSource exposes one -- so pinning a
    # version in the URI is not available. Instead Validate publishes the verified bytes
    # to a create-only, content-addressed key in the component's trust bucket, which the
    # Foundation role has no grant on, and registration points there. A key that can never
    # be replaced is what gives an ordinary ModelDataUrl a stable meaning.
    #
    # Both the train and eval-only paths use this same publication path, so neither can
    # register unpromoted bytes.
    register_model_data = JsonGet(
        step_name=validate_step.name,
        property_file=validated_prop,
        json_path="promotion.model_uri",
    )

    register_model = Model(
        # NOTE: intentionally NO entry_point / source_dir / dependencies here.
        # Adding any of them makes ModelStep inject a _RepackModelStep sub-step
        # (extra step "RegisterModel-RepackModel" + a repack job), changing the
        # step graph. The registered checkpoint needs no repack, so keep this a
        # bare image+model_data Model.
        image_uri=register_image,
        model_data=register_model_data,
        role=cfg.role_arn,
        sagemaker_session=_register_session(session, cfg),
    )
    # Point ModelMetrics at the IMMUTABLE attestation Validate published, not at the
    # ProcessingOutput copy. The processing output lives under an ordinary coordination
    # prefix that is not create-only, so it does not carry the guarantee the attestation
    # does. ContentDigest is populated so a consumer can verify the evidence it fetches,
    # and the verification script requires this linkage -- "6/6 checks passed" previously
    # did not establish that the package pointed at the evidence being inspected, so a
    # package with missing metrics, or another execution's, passed.
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            content_type="application/json",
            s3_uri=JsonGet(
                step_name=validate_step.name,
                property_file=validated_prop,
                json_path="promotion.attestation_uri",
            ),
            # N1: the API requires "sha256:<hex>", not bare hex. The producer emits
            # this pre-formatted field; the raw hex remains available as
            # promotion.attestation_sha256 for content addressing.
            content_digest=JsonGet(
                step_name=validate_step.name,
                property_file=validated_prop,
                json_path="promotion.attestation_content_digest",
            ),
        ),
    )
    register_args = dict(
        content_types=["application/x-tar"],
        response_types=["application/json"],
        inference_instances=["ml.g5.xlarge"],
        transform_instances=["ml.g5.xlarge"],
        model_package_group_name=model_package_group,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
        description=(
            "Simulation-evaluated policy checkpoint. Artifact-only record; "
            "no SageMaker inference endpoint is provided by this component. "
            "The verified S3 VersionId/ETag/digest are in ModelMetrics "
            "(the published validation attestation). The model-data URI is unversioned; "
            "verify against ModelMetrics before consumption."
        ),
    )
    metadata = {
        "model_family": params["model_family"],
        "suite": params["suite"],
    }
    if eval_only:
        metadata["input_checkpoint_uri"] = ckpt_param
        metadata["eval_only"] = "true"
    register_args["customer_metadata_properties"] = metadata
    register_step = ModelStep(
        name="RegisterModel",
        step_args=register_model.register(**register_args),
    )

    gate_step = ConditionStep(
        name="SuccessGate",
        conditions=[condition],
        if_steps=[register_step],
        else_steps=[gate_fail],
    )

    return finish([*steps, gate_step])
