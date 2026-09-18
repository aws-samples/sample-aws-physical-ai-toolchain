#!/usr/bin/env python3
"""Run a sample cell for a given family through the real pipeline.

Usage:
    PYTHONPATH=src python scripts/run_libero.py --family openvla --train-steps 2000
    PYTHONPATH=src python scripts/run_libero.py --family molmoact2 --suite libero_object --train-steps 2000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import boto3

sys.path.insert(0, "src")
from vla_pipeline.common.sourcedir import stage
from vla_pipeline.config import load_config
from vla_pipeline.pipeline import build_pipeline
from vla_pipeline.registry import (
    require_image_capability,
    resolve_image_digest,
    check_storage_fits,
    select_instance,
    select_volume_gb,
    DATASET_FROM_SUITE_MANIFEST,
    list_supported_pairs,
    resolve,
    resolve_suite,
    suites_for_simulator,
)
from vla_pipeline.runner import (
    describe_steps,
    stage_validate_code,
    upload_code,
    upload_directory,
    upsert_versioned,
    wait_for_execution,
)

# Image URIs + registry group now come from the registry (single source of
# truth). resolve(family, "libero") replaces the old hardcoded _get_images map;
# adding a model is a manifest, NOT an edit here.
_LIBERO_FAMILIES = sorted({f for f, s in list_supported_pairs() if s == "libero"})


def _normalize_task_ids(s: str) -> str:
    """Normalize --eval-task-ids to what validate_entry expects: the sentinel
    'all' or a JSON int list. Accepts 'all', a JSON array ('[0,1]'), or a comma
    list ('0,1'). Fails loud otherwise -- validate_entry does json.loads() on any
    non-'all' value, so a bare comma string would JSONDecodeError there."""
    s = (s or "").strip()
    if s == "" or s.lower() == "all":
        return "all"
    try:
        raw = json.loads(s) if s.startswith("[") else [int(x) for x in s.split(",") if x.strip()]
        if not raw:
            raise SystemExit("--eval-task-ids produced an empty list")
        for v in raw:
            if not isinstance(v, int) or isinstance(v, bool):
                raise SystemExit(
                    f"--eval-task-ids values must be integers, got {v!r} ({type(v).__name__})")
            if v < 0:
                raise SystemExit(f"--eval-task-ids values must be non-negative, got {v}")
        ids = sorted(set(raw))
    except (ValueError, TypeError) as e:
        raise SystemExit(
            f"--eval-task-ids must be 'all', a JSON int list ('[0,1]'), or a comma "
            f"list ('0,1'); got {s!r}: {e}")
    return json.dumps(ids)


def ensure_model_package_group(sm, group_name: str):
    """Create the Model Package Group if it doesn't exist."""
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"  Model Package Group exists: {group_name}")
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=f"VLA sample-run packages for {group_name}",
        )
        print(f"  Model Package Group created: {group_name}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=_LIBERO_FAMILIES)
    parser.add_argument("--suite", default="libero_spatial",
                        help="LIBERO suite (config/suites/*.yaml with simulator: "
                             f"libero). Supported: "
                             f"{', '.join(suites_for_simulator('libero'))}.")
    parser.add_argument("--allow-experimental", action="store_true",
                        help="permit a suite whose manifest declares "
                             "status: experimental (Validate will reject the report)")
    def _positive_int(v):
        iv = int(v)
        if iv < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
        return iv

    def _finite_float(v):
        fv = float(v)
        if not math.isfinite(fv):
            raise argparse.ArgumentTypeError(f"must be finite, got {v}")
        return fv

    parser.add_argument("--threshold", type=_finite_float, default=0.0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--acknowledge-instance-override", action="store_true",
                        help="State deliberately that an instance type differing from the family "
                             "manifest's declaration is intended. Without it a differing "
                             "--train-instance/--eval-instance is refused, because a silent "
                             "substitution reached the training forward pass on half the declared "
                             "GPU memory and died after being billed.")
    parser.add_argument("--train-instance", default=None,
                        help="Override the training instance type. Defaults to the family "
                             "manifest's train_instance.")
    parser.add_argument("--eval-instance", default=None,
                        help="Override the evaluation instance type. Defaults to the family "
                             "manifest's eval_instance.")
    parser.add_argument("--train-image", default=None,
                        help="Override the training image with repository:tag or "
                             "repository@sha256:digest, without the ECR registry hostname.")
    parser.add_argument("--eval-image", default=None,
                        help="Override the evaluation image with repository:tag or "
                             "repository@sha256:digest, without the ECR registry hostname.")
    parser.add_argument("--train-steps", type=_positive_int, default=None,
                        help="FineTune step budget for a train-default run (e.g. 2000, 20000). "
                             "Required unless --checkpoint-s3 is given (eval-only skips FineTune).")
    # A single episode is not a measurement: the rate can only be 0.0 or 1.0, and a sample run must still produce a rate that means something. Three is the floor for any episode count.
    parser.add_argument("--eval-trials", type=_positive_int, default=3,
                        help="EvalTrials = episodes per task (default 3 = the sample "
                             "floor; use 20 for paper-grade LIBERO curves).")
    parser.add_argument("--eval-task-ids", default="all",
                        help="'all' (default = full suite), a JSON int list like "
                             "'[0,1]', or a comma list like '0,1'. Normalized to "
                             "what the Validate gate expects (json.loads).")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit and return the execution ARN immediately (do not block).")
    parser.add_argument("--eval-volume-gb", type=int, default=None,
                        help="Override the EVALUATION volume. Defaults to the family manifest's "
                             "eval_volume_gb, which is a separate figure because eval runs on a "
                             "different instance with different storage limits.")
    parser.add_argument("--volume-gb", type=int, default=None,
        help="Override the TRAINING (and Validate) volume only -- NOT eval, which has its own "
             "--eval-volume-gb. Defaults to the family manifest's train_volume_gb. The manifest "
             "field is train_volume_gb/eval_volume_gb; the single shared volume_gb this flag used to "
             "name no longer exists, because one figure cannot be checked against two different "
             "instances' storage limits.")
    parser.add_argument("--checkpoint-s3", default=None,
                        help="EVAL-ONLY mode (D2): S3 URI of an existing checkpoint model.tar.gz "
                             "(exact object key, in a versioned bucket). Skips FineTune -- SimEval "
                             "mounts this checkpoint and RegisterModel records it as "
                             "input_checkpoint_uri. Omit for the default train-then-eval flow.")
    args = parser.parse_args()
    if not args.checkpoint_s3 and args.train_steps is None:
        parser.error("--train-steps is required for a train-default run "
                     "(omit it only with --checkpoint-s3 eval-only).")

    # Resolve the suite through the registry, exactly as run_arena.py does. A
    # free-form --suite string used to reach StartPipelineExecution unchecked:
    # `--suite arena_gr1` would fine-tune on the GR1 dataset, evaluate in MuJoCo,
    # and only be rejected by Validate afterwards -- after the full GPU spend. A
    # typo (`libero_spatal`) failed just as late.
    resolved_suite = resolve_suite(args.suite)
    if resolved_suite.simulator != "libero":
        raise SystemExit(
            f"FATAL: suite {args.suite!r} runs on simulator "
            f"{resolved_suite.simulator!r}, not 'libero'. LIBERO suites: "
            f"{', '.join(suites_for_simulator('libero'))}")
    if not resolved_suite.supported and not args.allow_experimental:
        raise SystemExit(
            f"FATAL: suite {args.suite!r} is status: experimental -- it is excluded "
            f"from the validator's suite allowlist, so Validate would reject the "
            f"report. Pass --allow-experimental to submit anyway (diagnostic only).")
    # GR00T train_entry resolves datasets from SUITE_DATASETS (a hardcoded
    # suite->repo map); a suite with `dataset: null` is not in that map, so
    # train_entry hard-exits. Catch it at submit time. OpenVLA and MolmoAct2
    # resolve datasets from TRAIN_DATASET_S3URI / their own HF repos, so
    # suite.dataset is irrelevant for them -- a null here does not block training.
    if (args.family == "gr00t"
            and not args.checkpoint_s3 and resolved_suite.dataset is None):
        raise SystemExit(
            f"FATAL: suite {args.suite!r} declares no fine-tune dataset "
            f"(config/suites/{args.suite}.yaml dataset: null), so GR00T FineTune "
            f"would reject it. Use --checkpoint-s3 for an eval-only run.")
    suite = resolved_suite.name
    task_ids_str = _normalize_task_ids(args.eval_task_ids)
    if task_ids_str != "all":
        canonical = resolved_suite.canonical_task_ids
        submitted = json.loads(task_ids_str)
        out_of_range = [t for t in submitted if t not in canonical]
        if out_of_range:
            raise SystemExit(
                f"FATAL: task IDs {out_of_range} are not in suite {suite!r}'s "
                f"canonical set {canonical}. Fix --eval-task-ids or use 'all'.")

    if args.family == "openvla" and task_ids_str != "all":
        raise SystemExit(
            "FATAL: OpenVLA does not support task subsetting (--eval-task-ids). "
            "The pinned evaluator rejects every non-'all' selection. Use 'all'.")
    if args.family == "openvla" and args.eval_trials > 50:
        raise SystemExit(
            f"FATAL: --eval-trials={args.eval_trials} exceeds the LIBERO spatial "
            "initial-state bank (50). The pinned OpenVLA evaluator indexes states "
            "directly; trials > 50 will crash. Use --eval-trials 50 or fewer.")

    cfg = load_config()
    family = args.family
    spec = resolve(family, "libero")
    base = f"{cfg.account_id}.dkr.ecr.{cfg.region}.amazonaws.com"
    # Resolved per ROLE from the spec, not assumed equal. The comment here used to read
    # "train == eval for LIBERO" and one variable was submitted for both parameters -- true for the
    # three LIBERO pairs today, and NOT true in general: gr00t x isaac_arena declares
    # different training and Arena evaluator repositories. So a pair manifest that
    # legitimately diverged would have been silently ignored, submitting the TRAINING image as the
    # evaluator, and the only symptom would be an evaluator that cannot serve the suite.
    image = f"{base}/{args.train_image or spec.train_image_repo}"
    eval_image = f"{base}/{args.eval_image or spec.eval_image_repo}"
    if not args.checkpoint_s3:
        image = resolve_image_digest(image)

    print(f"{'='*60}")
    print(f"  SMOKE CELL: {family} x {suite}")
    print(f"  Train image: {image}")
    print(f"  Eval image:  {eval_image}")

    # Digest-form for attestation, then a CAPABILITY check -- both before anything is submitted. The
    # capability read costs about a second against ECR and replaces discovering a missing LIBERO client
    # after a GPU node has been provisioned and the image pulled. An image with no capability label
    # says so and proceeds: unlabelled images predate the label, and the runtime guard still fails
    # closed on the interpreter's real presence.
    _checked_eval_image = resolve_image_digest(eval_image)
    # Read from the SIMULATOR manifest, never hardcoded here. This launcher demanded
    # "libero-client-verified" from whatever image it was given, which is right for LIBERO and wrong for
    # the Arena connector -- a different image, with a different stamp and no LIBERO client at all.
    if spec.required_image_capability:
        require_image_capability(_checked_eval_image, spec.required_image_capability)
    print(f"{'='*60}")

    # 1. Stage and upload sourcedir tarballs (content-addressed)
    # stage() assembles entry scripts + shared modules (digest, validator,
    # defaults.json) into one directory that SageMaker extracts as the working dir.
    print("\n[1/5] Staging and uploading sourcedirs...")
    staged_dir = stage(family)
    train_source_uri = upload_directory(cfg, staged_dir)
    eval_source_uri = train_source_uri  # Same sourcedir has both entries
    print(f"  staged: {staged_dir}")
    print(f"  train_source: {train_source_uri}")
    print(f"  eval_source: {eval_source_uri}")

    # 2. Upload validate_entry.py to the fixed key the definition references
    # (must be done BEFORE deploy -- the definition points at this exact path)
    # 1a fix: embed the full validator.py into validate_entry.py (base64 bootstrap)
    # so the Validate step actually loads it. Raw upload left validator.py as dead
    # code and Validate silently ran basic checks only.
    staged_validate = stage_validate_code()
    validate_uri = upload_code(cfg, staged_validate)
    print(f"  validate_code (validator embedded): {validate_uri}")

    # 3. Deploy the pipeline
    print("\n[2/5] Deploying pipeline...")
    from sagemaker.workflow.pipeline_context import PipelineSession
    session = PipelineSession(boto_session=boto3.Session(region_name=cfg.region))
    # 1c fix: pass the content-addressed validate code URI so code= resolves to a
    # real object (the flat convention key was never written by upload_code).
    pipeline = build_pipeline(cfg, session=session, validate_code_uri=validate_uri,
                              checkpoint_s3_uri=args.checkpoint_s3)
    result = upsert_versioned(pipeline, cfg.role_arn)
    pipeline_arn = result["PipelineArn"]
    print(f"  Pipeline: {pipeline_arn}")

    # 4. Ensure Model Package Group
    print("\n[3/5] Ensuring Model Package Group...")
    sm = boto3.client("sagemaker", region_name=cfg.region)
    group_name = spec.registry_group
    ensure_model_package_group(sm, group_name)

    # 5. Start execution
    print("\n[4/5] Starting execution...")
    _train_instance = select_instance(spec, "train", args.train_instance,
                                     args.acknowledge_instance_override)
    _eval_instance = select_instance(spec, "eval", args.eval_instance,
                                    args.acknowledge_instance_override)
    _train_volume = select_volume_gb(spec, "train", args.volume_gb)
    _eval_volume = select_volume_gb(spec, "eval", args.eval_volume_gb)
    # Storage capacity, BEFORE submission. A 300 GB request was rejected at CreateTrainingJob by an
    # instance with 250 GB of local NVMe -- fast and free, but it silently removed that candidate from
    # a capacity fan-out. Checked here so the command fails naming the instance and its limit.
    # I4: both checks were unconditional, so an EVAL-ONLY run was rejected over a trainer that never
    # runs -- the eval-only graph omits FineTune entirely (pipeline.py: `eval_only = checkpoint_s3_uri
    # is not None`). A supplied checkpoint with an unused large-volume trainer override could not be
    # submitted at all.
    #
    # The SAME predicate as the graph, deliberately: a launcher deciding "is this eval-only?" by its own
    # rule is a second declaration that would drift from the graph it submits to.
    _eval_only = args.checkpoint_s3 is not None
    if not _eval_only:
        check_storage_fits(_train_instance, _train_volume)
    else:
        print("  eval-only: FineTune is omitted from the graph, so the training instance and volume "
              "are not checked -- nothing will use them")
    check_storage_fits(_eval_instance, _eval_volume)
    params = {
        "ModelFamily": family,
        "ModelPackageGroupName": group_name,
        "Suite": suite,
        "TrainSuite": "unified" if family == "molmoact2" else suite,
        "TrainImageUri": image,
        # Digest-form, resolved from ECR: the attestation must name the BYTES that ran, and a tag
        # cannot. Validate REFUSES a non-digest reference, so this is required, not decorative.
        "EvalImageUri": _checked_eval_image,
        # "resolve the dataset from the suite manifest" -- not an S3 URI.
        "DatasetS3Uri": DATASET_FROM_SUITE_MANIFEST,
        # cycle-17 R2 (found by a real launch, not by any test): sending "" here made
        # StartPipelineExecution reject the whole call -- the parameter declares
        # minLength 1, and its pipeline default is the "__FROM_SUITE_MANIFEST__"
        # sentinel. An empty override is not "no override", it is an invalid one, so
        # the override is OMITTED when the suite carries no revision and the
        # pipeline default applies. Set below, after the dict.
        "TrainInstanceType": _train_instance,
        "EvalInstanceType": _eval_instance,
        "TrainSourceDirUri": train_source_uri,
        "EvalSourceDirUri": eval_source_uri,
        "SuccessThreshold": str(args.threshold),
        "EvalSeed": str(args.eval_seed),
        "TrainSteps": str(args.train_steps),
        "EvalTrials": str(args.eval_trials),
        "EvalTaskIds": _normalize_task_ids(args.eval_task_ids),
    }
    _revision = (resolved_suite.dataset.revision if resolved_suite.dataset else None)
    if _revision:
        params["DatasetRevision"] = _revision
    if args.checkpoint_s3:
        params["CheckpointS3Uri"] = args.checkpoint_s3

    # cycle-15 I7: this whole resolution used to sit INSIDE `if args.checkpoint_s3:`, so a
    # train-default run omitted VolumeSizeInGB entirely and silently took the pipeline's 100 GB
    # default -- even when --volume-gb was passed explicitly. The families needing 250-300 GB were
    # exactly the ones affected. Volume has nothing to do with whether a checkpoint was supplied,
    # so it is resolved for every mode.
    #
    # I10: the family manifests declare per-role volumes (train_volume_gb/eval_volume_gb) and the
    # launcher passed nothing, so every run took the pipeline default of 100 GB. The registry
    # appeared to state an operational requirement that the launcher ignored -- and for the
    # families needing 250-300 GB that is a disk-exhaustion risk part-way through a paid run,
    # discovered only when the volume fills.
    #
    # Resolved from the manifest rather than hardcoded, and overridable, so the declared figure
    # stays the single source of truth. Same shape as the max_run fix: a declared number that
    # governed nothing.
    params["VolumeSizeInGB"] = str(_train_volume)
    params["EvalVolumeSizeInGB"] = str(_eval_volume)


    # Filter to the parameters the (variant-selected) pipeline actually declares.
    # In eval-only mode the FineTune-only params (TrainImageUri/DatasetS3Uri/
    # TrainSteps/TrainSaveSteps/TrainSourceDirUri/TrainInstanceType) are NOT
    # declared, and passing an undeclared parameter fails StartPipelineExecution.
    declared = {p.name for p in pipeline.parameters}
    dropped = sorted(set(params) - declared)
    if dropped:
        print(f"  (eval-only) dropping non-declared params: {dropped}")
    pipeline_params = [{"Name": k, "Value": v} for k, v in params.items() if k in declared]
    resp = sm.start_pipeline_execution(
        PipelineName=cfg.pipeline_name,
        PipelineVersionId=result["PipelineVersionId"],
        PipelineParameters=pipeline_params,
    )
    execution_arn = resp["PipelineExecutionArn"]
    print(f"  Execution: {execution_arn}")
    print(f"  Params: {json.dumps(params, indent=2)}")

    if args.no_wait:
        print("\n  Submitted; training may still be waiting for capacity.")
        print(f"  Status: aws sagemaker describe-pipeline-execution --region {cfg.region} "
              f"--pipeline-execution-arn {execution_arn}")
        return

    # 6. Wait and report
    print("\n[5/5] Waiting for execution...")
    start_time = time.time()
    status = wait_for_execution(cfg, execution_arn, poll_seconds=30)
    elapsed = time.time() - start_time
    print(f"\n  Final status: {status} (elapsed: {elapsed/60:.1f} min)")

    steps = describe_steps(cfg, execution_arn)
    print("  Steps:")
    for s in steps:
        name = s.get("StepName", "?")
        st = s.get("StepStatus", "?")
        reason = s.get("FailureReason", "")
        cond = s.get("Metadata", {}).get("Condition", {}).get("Outcome", "")
        extra = f" (condition={cond})" if cond else ""
        extra += f" REASON: {reason[:200]}" if reason else ""
        print(f"    {name}: {st}{extra}")

    print(f"\n{'='*60}")
    print(f"  RESULT: {status}")
    print(f"  Execution ARN: {execution_arn}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"{'='*60}")

    if status != "Succeeded":
        print(f"\n  FAILURE: Pipeline ended with status '{status}', expected 'Succeeded'")
        sys.exit(1)


if __name__ == "__main__":
    main()
