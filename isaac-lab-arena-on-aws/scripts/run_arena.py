#!/usr/bin/env python3
"""Submit a <family> x Isaac Lab Arena pipeline execution.

Unlike run_libero.py (LIBERO/MuJoCo, train image == eval image), the Arena path
uses a SEPARATE eval image (an Isaac Sim + Isaac Lab Arena container) and the
family's Arena policy connector. FineTune runs on the family TRAIN image and
produces the checkpoint; SimEval runs on the Arena eval image routed by
ARENA_CONNECTOR.

Only GR00T x Arena is shipped. OpenVLA and MolmoAct2 are 7-DoF Franka policies
that cannot emit the 26-dim GR1 action, so those cells are embodiment-blocked and
excluded -- `docker_entrypoint_multi.sh` hard-rejects those connectors, and no
pair manifest exists for them (see the README compatibility matrix).

WHERE THE ARENA KNOBS COME FROM
-------------------------------
The Arena runtime contract (`policy_runner` task, --embodiment, --object,
--num_episodes, --policy_config_yaml_path) and the GR00T embodiment tag are resolved
from the SUITE MANIFEST (`config/suites/<suite>.yaml`), not from CLI defaults.
Selecting `--suite` therefore selects a coherent train+eval cell. The
corresponding flags below are OVERRIDES: unset means "use the manifest".

Previously these were independent flags whose defaults belonged to other cells
(`--embodiment-tag LIBERO_PANDA`, `--task-name cube_goal_pose`), so
`--suite arena_gr1_fridge` would fine-tune on the fridge dataset and then
evaluate a different task on a Franka embodiment tag -- a silent train/eval
mismatch that the digest gate cannot catch.

Usage:
    PYTHONPATH=src python scripts/run_arena.py --family gr00t \
        --suite arena_gr1_fridge --train-steps 2000

Submits and returns the execution ARN immediately (does NOT block). Monitor the
SimEval log tail (never trust step status alone).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import boto3

sys.path.insert(0, "src")
from vla_pipeline.arena import ArenaConfigError, gate_suite, resolve_runtime
from vla_pipeline.common.sourcedir import stage
from vla_pipeline.config import load_arena_repository, load_config
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
from vla_pipeline.runner import stage_validate_code, upload_code, upload_directory, upsert_versioned

# Adapter identity (train/eval images, connector, use_groot_server, default
# suite, registry group) comes from the registry
# (config/{families,simulators,pairs}) -- single source of truth. Adding a model
# is a manifest, NOT an edit here.
_ARENA_FAMILIES = sorted({f for f, s in list_supported_pairs() if s == "isaac_arena"})
_SIMULATOR = "isaac_arena"


def resolve_arena_knobs(suite, args) -> dict:
    """The EVAL_SIM_CONFIG payload for `suite`, with this run's CLI overrides.

    Thin adapter over vla_pipeline.arena.resolve_runtime -- the ONE implementation,
    shared with scripts/submit_simeval.py so the two Arena launchers cannot drift
    apart again. Manifest defects and unrunnable suites surface as a submit-time
    SystemExit instead of a failed GPU job.
    """
    try:
        return resolve_runtime(
            suite, args.family, args.gr00t_version,
            embodiment_tag=args.embodiment_tag,
            policy_config=args.policy_config_yaml,
            arena_embodiment=args.arena_embodiment,
            obj=args.object,
        )
    except ArenaConfigError as e:
        raise SystemExit(f"FATAL: {e}") from e


def build_eval_sim_config(suite, args) -> str:
    """Collapse the per-run Arena eval knobs into one JSON blob.

    The Arena eval_entry unpacks EVAL_SIM_CONFIG and seeds the discrete
    SM_HP_*/EVAL_* env keys. use_groot_server + arena_connector are NOT here --
    the shell entrypoint (docker_entrypoint_multi.sh) routes on them before the
    Python eval entry runs, so they stay discrete pipeline params.
    """
    return json.dumps(resolve_arena_knobs(suite, args), sort_keys=True)


def ensure_model_package_group(sm, group_name: str):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"  Model Package Group exists: {group_name}")
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=f"VLA Arena packages for {group_name}",
        )
        print(f"  Model Package Group created: {group_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="gr00t", choices=_ARENA_FAMILIES)
    parser.add_argument("--suite", default=None,
                        help="Arena suite (config/suites/*.yaml with simulator: "
                             f"isaac_arena). Supported: "
                             f"{', '.join(suites_for_simulator(_SIMULATOR))}. "
                             "Default: the pair manifest's default_suite.")
    parser.add_argument("--allow-experimental", action="store_true",
                        help="permit a suite whose manifest declares "
                             "status: experimental (never proven end-to-end, and "
                             "excluded from the validator's suite allowlist, so "
                             "Validate will reject the report).")
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

    parser.add_argument("--train-steps", type=_positive_int, default=None,
                        help="FineTune step budget for a train-default run (e.g. 2000, 20000). "
                             "Required unless --checkpoint-s3 is given (eval-only skips FineTune).")
    parser.add_argument("--save-steps", type=int, default=1000000,
                        help="checkpoint save interval for a single-run dose curve; default "
                             "1000000 = final-only (unchanged). Set e.g. 100 for a dose curve.")
    parser.add_argument("--eval-dose-steps", default="all",
                        help="comma list of intermediate checkpoint steps to roll out (subset "
                             "of what train saved), or 'all' = every staged checkpoint. "
                             "Must be non-empty (SageMaker rejects empty param values).")
    parser.add_argument("--eval-volume-size", type=int, default=None,
                        help="Override the EVALUATION volume. Defaults to the family manifest's "
                             "eval_volume_gb.")
    parser.add_argument("--volume-size", type=int, default=None,
                        help="Override the TRAINING (and Validate) volume only -- NOT eval, which "
                             "has its own --eval-volume-size. Defaults to the family manifest's "
                             "train_volume_gb. Raise it for a dose curve that stages many "
                             "checkpoints to /opt/ml/model (e.g. 500).")
    # A single episode is not a measurement: the rate can only be 0.0 or 1.0, and a sample run must still produce a rate that means something. Three is the floor for any episode count.
    parser.add_argument("--eval-trials", type=_positive_int, default=3)
    parser.add_argument("--eval-task-ids", default="all")
    parser.add_argument("--eval-seed", type=int, default=100)
    parser.add_argument("--threshold", type=_finite_float, default=0.0)
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
                        help="Override the Arena image with repository:tag or "
                             "repository@sha256:digest, without the ECR registry hostname.")
    parser.add_argument("--gr00t-version", default="n17", choices=["n16", "n17"],
                        help="n16 = N1.6-3B@5dc80c4 native GR1 (matches NVIDIA reference); n17 = default")
    # --- Arena runtime OVERRIDES ------------------------------------------
    # All default to None = "use config/suites/<suite>.yaml". Pass one only to
    # deviate from the suite's declared contract for a single run; deviating from
    # the manifest's task/embodiment breaks train/eval coherence, which Validate
    # now rejects.
    ov = parser.add_argument_group(
        "Arena overrides (default: resolved from the suite manifest)")
    ov.add_argument("--embodiment-tag", default=None,
                    help="override the GR00T server --embodiment-tag")
    ov.add_argument("--policy-config-yaml", default=None,
                    help="container path to the Arena closed-loop config YAML "
                         "(overrides arena.policy_config)")
    ov.add_argument("--arena-embodiment", default=None,
                    help="override policy_runner --embodiment (pass 'NONE' to omit)")
    ov.add_argument("--object", dest="object", default=None,
                    help="override policy_runner --object (pass 'NONE' to omit)")
    parser.add_argument("--checkpoint-s3", default=None,
                        help="EVAL-ONLY mode (D2): S3 URI of an existing checkpoint model.tar.gz "
                             "(exact object key, versioned bucket). Skips FineTune -- SimEval mounts "
                             "this checkpoint; RegisterModel records it as input_checkpoint_uri. "
                             "Omit for the default train-then-eval flow.")
    args = parser.parse_args()
    if args.eval_task_ids not in ("all", "[0]"):
        parser.error(f"--eval-task-ids must be 'all' or '[0]' for Arena single-task "
                     f"suites, got {args.eval_task_ids!r}")
    if args.gr00t_version == "n16" and args.save_steps < (args.train_steps or 1000000):
        parser.error("--save-steps < --train-steps (dose-curve mode) is not supported "
                     "for N1.6: the pinned FinetuneConfig lacks --save_only_model. "
                     "Use --save-steps >= --train-steps for a single final checkpoint.")
    if not args.checkpoint_s3 and args.train_steps is None:
        parser.error("--train-steps is required for a train-default run "
                     "(omit it only with --checkpoint-s3 eval-only).")
    # No bounds check here: argparse type=int rejects a non-integer, and
    # arena.resolve_runtime rejects a non-positive one with the same message the
    # manifest field gets. A duplicate parser.error was redundant AND untestable
    # (both paths raise SystemExit, so a test could not tell them apart).

    # Resolve + gate the cell BEFORE touching AWS config: the suite manifest
    # defines the whole cell (Arena runtime contract, fine-tune dataset, embodiment
    # wiring) and every check here is offline, so a bad suite or a missing policy
    # config should not require resolvable credentials to report.
    spec = resolve(args.family, _SIMULATOR)          # pure: no AWS
    suite_name = args.suite or spec.default_suite
    suite = resolve_suite(suite_name)
    try:
        gate_suite(suite, allow_experimental=args.allow_experimental)
    except ArenaConfigError as e:
        raise SystemExit(f"FATAL: {e}") from e
    eval_sim_config = build_eval_sim_config(suite, args)
    knobs = json.loads(eval_sim_config)

    cfg = load_config()
    base = f"{cfg.account_id}.dkr.ecr.{cfg.region}.amazonaws.com"
    train_image = f"{base}/{args.train_image or spec.train_image_repo}"
    if not args.checkpoint_s3:
        train_image = resolve_image_digest(train_image)
    if args.eval_image:
        eval_image = f"{base}/{args.eval_image}"
    else:
        eval_tag = spec.eval_image_repo.rsplit(":", 1)[1]
        eval_image = f"{load_arena_repository(cfg)}:{eval_tag}"

    print("=" * 60)
    print(f"  ARENA CELL: {args.family} x isaac_lab_arena")
    print(f"  Train image: {train_image}")
    print(f"  Eval  image: {eval_image}")

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
    print(f"  Connector={spec.arena_connector} UseGrootServer={spec.use_groot_server}")
    print(f"  Suite={suite.name} ({suite.status}) task={knobs['task_name']}")
    print(f"  Arena: --embodiment {knobs['arena_embodiment']} "
          f"--object {knobs['object']} --num_episodes {args.eval_trials}")
    print(f"  GR00T: version={args.gr00t_version} "
          f"embodiment_tag={knobs['embodiment_tag']}")
    print(f"  policy_config={knobs['policy_config_yaml']}")
    print(f"  TrainSteps={args.train_steps} EvalTrials={args.eval_trials} "
          f"tasks={args.eval_task_ids} seed={args.eval_seed} "
          f"threshold={args.threshold}")
    print("=" * 60)

    print(f"\n[1/4] Staging + uploading {args.family} sourcedir...")
    staged_dir = stage(args.family)
    source_uri = upload_directory(cfg, staged_dir)
    print(f"  staged: {staged_dir}")
    print(f"  source: {source_uri}")

    # 1a fix: embed validator.py into validate_entry.py so Validate loads it.
    staged_validate = stage_validate_code()
    validate_uri = upload_code(cfg, staged_validate)
    print(f"  validate_code (validator embedded): {validate_uri}")

    print("\n[2/4] Deploying pipeline...")
    from sagemaker.workflow.pipeline_context import PipelineSession
    session = PipelineSession(boto_session=boto3.Session(region_name=cfg.region))
    # 1c fix: consume the content-addressed validate code URI.
    pipeline = build_pipeline(cfg, session=session, validate_code_uri=validate_uri,
                              checkpoint_s3_uri=args.checkpoint_s3)
    result = upsert_versioned(pipeline, cfg.role_arn)
    print(f"  Pipeline: {result['PipelineArn']}")

    print("\n[3/4] Ensuring Model Package Group...")
    sm = boto3.client("sagemaker", region_name=cfg.region)
    ensure_model_package_group(sm, spec.registry_group)

    print("\n[4/4] Starting execution (NOT blocking)...")
    _train_instance = select_instance(spec, "train", args.train_instance,
                                     args.acknowledge_instance_override)
    _eval_instance = select_instance(spec, "eval", args.eval_instance,
                                    args.acknowledge_instance_override)
    _train_volume = select_volume_gb(spec, "train", args.volume_size)
    _eval_volume = select_volume_gb(spec, "eval", args.eval_volume_size)
    # See run_libero.py: a 300 GB request was rejected by an instance with 250 GB of local NVMe.
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
        "ModelFamily": args.family,
        "ModelPackageGroupName": spec.registry_group,
        "Suite": suite.name,
        "TrainSuite": suite.name,
        "TrainImageUri": train_image,
        # Digest-form, resolved from ECR: the attestation must name the BYTES that ran, and a tag
        # cannot. Validate REFUSES a non-digest reference, so this is required, not decorative.
        "EvalImageUri": _checked_eval_image,
        "DatasetS3Uri": DATASET_FROM_SUITE_MANIFEST,
        # cycle-17 R2 (found by a real launch, not by any test): sending "" here made
        # StartPipelineExecution reject the whole call -- the parameter declares
        # minLength 1, and its pipeline default is the "__FROM_SUITE_MANIFEST__"
        # sentinel. An empty override is not "no override", it is an invalid one, so
        # the override is OMITTED when the suite carries no revision and the
        # pipeline default applies. Set below, after the dict.
        "TrainInstanceType": _train_instance,
        "EvalInstanceType": _eval_instance,
        "TrainSourceDirUri": source_uri,
        "EvalSourceDirUri": source_uri,
        "TrainSteps": str(args.train_steps),
        "TrainSaveSteps": str(args.save_steps),
        "EvalDoseSteps": args.eval_dose_steps,
        "VolumeSizeInGB": str(_train_volume),
        "EvalVolumeSizeInGB": str(_eval_volume),
        "EvalSeed": str(args.eval_seed),
        "EvalTrials": str(args.eval_trials),
        "EvalTaskIds": args.eval_task_ids,
        "SuccessThreshold": str(args.threshold),
        "UseGrootServer": spec.use_groot_server,
        "ArenaConnector": spec.arena_connector,
        "Gr00tVersion": args.gr00t_version,
        # The Arena knobs, suite-manifest-resolved, as one JSON blob.
        "EvalSimConfig": eval_sim_config,
        "ExpectedEmbodimentTag": knobs["embodiment_tag"],
        "ExpectedArenaEmbodiment": knobs["arena_embodiment"],
        "ExpectedArenaObject": knobs["object"],
        "ExpectedPolicyConfig": knobs["policy_config_yaml"],
    }
    _revision = (suite.dataset.revision if suite.dataset else None)
    if _revision:
        params["DatasetRevision"] = _revision
    _SM_ENV_LIMIT = 256
    for _k in ("ExpectedPolicyConfig", "ExpectedArenaEmbodiment",
               "ExpectedEmbodimentTag", "ExpectedArenaObject"):
        if len(params.get(_k, "")) > _SM_ENV_LIMIT:
            parser.error(f"{_k} value is {len(params[_k])} chars, exceeds "
                         f"SageMaker Processing env limit ({_SM_ENV_LIMIT})")

    if args.checkpoint_s3:
        params["CheckpointS3Uri"] = args.checkpoint_s3

    # Filter to the parameters the (variant-selected) pipeline actually declares.
    # Eval-only drops the FineTune-only params; passing an undeclared parameter
    # fails StartPipelineExecution.
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
    print(f"\n  Execution ARN: {execution_arn}")
    print(f"  Params: {json.dumps(params, indent=2)}")
    print("\n  Submitted. Monitor SimEval log tail (never trust status).")


if __name__ == "__main__":
    main()
