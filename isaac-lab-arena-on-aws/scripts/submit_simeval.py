#!/usr/bin/env python3
"""submit_simeval.py -- run ONLY the Arena SimEval step against an EXISTING checkpoint.

Debug/iteration accelerator: the full pipeline (run_arena.py) re-runs a ~25-min FineTune
every time. When you already have a valid FineTune output in S3 (model.tar.gz) and only
need to iterate on the Arena SimEval (gr00t-venv build, policy_runner config, the
dump_arena_policy_cfg read, #9 --policy_config_yaml_path, etc.), this submits a standalone
SageMaker training job that mounts that checkpoint and runs eval_entry.py -- skipping FineTune.

It mirrors the SimEval TrainingStep in src/vla_pipeline/pipeline.py (image = Arena eval image,
checkpoint mounted at /opt/ml/input/data/model, baked eval_entry.py driven by env vars). It
does NOT run Validate/Gate/Register -- use run_arena.py for a full validated cell. DEV-ONLY.

Account, roles and storage resolve through vla_pipeline.config.load_config().
Deploy Foundation and Arena first: discovery reads their required SSM parameters.
Use AWS_PROFILE for credentials and VLA_REGION=us-east-1 for deployment selection.

Like run_arena.py, the Arena runtime knobs are resolved from the SUITE MANIFEST
(`config/suites/<suite>.yaml`); the corresponding flags are OVERRIDES, unset
meaning "use the manifest". This script previously carried its own defaults, which
were mutually incoherent (`--suite libero_spatial` alongside `--task-name
gr1_open_microwave` on an Arena-only script) and disagreed with run_arena.py's
`--eval-seed`.

Example:
  VLA_REGION=us-east-1 python scripts/submit_simeval.py \
    --eval-image "$LOCAL_ARENA_EVAL_IMAGE" \
    --suite arena_gr1_fridge --gr00t-version n16 \
    --checkpoint-s3 "$CHECKPOINT_S3_URI" \
    --eval-source-dir "$EVAL_SOURCE_DIR_URI"

The README's standalone-evaluation section shows how to obtain these three URIs.
Omitting --instance selects the model manifest's evaluation instance.
"""
import argparse
import os
import re
import sys
import time
import uuid

import boto3

# Self-contained path bootstrap so `python scripts/submit_simeval.py` works
# without requiring PYTHONPATH=src to be set by the caller.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from vla_pipeline.arena import ArenaConfigError, gate_suite, resolve_runtime  # noqa: E402
from vla_pipeline.config import load_config  # noqa: E402
from vla_pipeline.registry import (  # noqa: E402
    require_image_capability,
    resolve_image_digest,
    check_storage_fits,
    resolve,
    resolve_suite,
    select_instance,
    select_volume_gb,
    suites_for_simulator,
)

_SIMULATOR = "isaac_arena"


def build_environment(args, suite, knobs: dict) -> dict:
    """The SimEval job's Environment. Extracted from main() so it is TESTABLE.

    While this dict was an inline literal inside create_training_job(...), the only
    way a test could check a key was present was to grep the script's source -- and a
    key reintroduced as a COMMENT satisfied that grep while the job ran without it
    (that is exactly how the EVAL_GR00T_VERSION bug survived a round of review).
    """
    env = {
        "EVAL_MODEL_FAMILY": args.model_family,
        # Routes eval_entry to the n16 native-GR1 server. WITHOUT this the job
        # resolves the n16 embodiment tag (GR1) from the manifest but starts the
        # n17 server, which passes --embodiment-tag GR1 to gr00t@376ba890 whose
        # enum has no bare GR1 -> dead job after the venv build.
        # pipeline.py sets the same key for the pipeline path.
        "EVAL_GR00T_VERSION": args.gr00t_version,
        "EVAL_SUITE": suite.name,
        "EVAL_SEED": args.eval_seed,
        "EVAL_TRIALS": args.eval_trials,
        "EVAL_TASK_IDS": args.eval_task_ids,
        "EVAL_MODEL_SOURCE_URI": args.checkpoint_s3,
        "USE_GROOT_SERVER": args.use_groot_server,
        "SM_HP_USE_GROOT_SERVER": args.use_groot_server,
        "ARENA_CONNECTOR": args.arena_connector,
        "HF_SECRET_NAME": args.hf_secret,
        # Suite-manifest-resolved (with this run's overrides applied). Set as
        # the discrete keys rather than EVAL_SIM_CONFIG because this path talks
        # to the baked eval entry directly; the entry's blob shim is a no-op
        # when no blob is present, so the discrete values stand.
        "EVAL_POLICY_CONFIG_YAML": knobs["policy_config_yaml"],
        "SM_HP_EMBODIMENT_TAG": knobs["embodiment_tag"],
        "SM_HP_TASK_NAME": knobs["task_name"],
        "EVAL_ARENA_EMBODIMENT": knobs["arena_embodiment"],
        "EVAL_OBJECT": knobs["object"],
        "EVAL_POSCTRL_N16": args.posctrl_n16,
        "MUJOCO_GL": "egl",
    }
    if args.posctrl_repo:
        env["N16_POSCTRL_CKPT_REPO"] = args.posctrl_repo
    # The evaluator REQUIRES an explicit 40-hex revision for the positive control: it used to publish
    # resolved_commit="main", a moving tag that pins nothing, so N16_POSCTRL_CKPT_REV now defaults to
    # "" and the child refuses an empty value. This submitter never forwarded it, which made the
    # positive control unusable from here the moment that fix landed -- a regression in the one path
    # that was left unconverted.
    if args.posctrl_revision:
        env["N16_POSCTRL_CKPT_REV"] = args.posctrl_revision
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-image", required=True, help="Arena connector image URI, preferably pinned by digest")
    p.add_argument("--checkpoint-s3", required=True,
                   help="S3 URI of the FineTune output model.tar.gz (its PREFIX is mounted at /opt/ml/input/data/model)")
    # No hardware defaults here. This submitter declared ml.g6e.8xlarge and 60 GB -- a third set of
    # figures, agreeing with neither the family manifest nor the other launchers -- and submitted them
    # straight into ResourceConfig without ever resolving the family's requirements. Absence now
    # resolves from the manifest like every other path.
    p.add_argument("--instance", default=None,
                   help="Override the evaluation instance type. Defaults to the family manifest's "
                        "eval_instance.")
    p.add_argument("--volume-size", type=int, default=None,
                   help="Override the evaluation volume. Defaults to the family manifest's "
                        "eval_volume_gb.")
    p.add_argument("--acknowledge-instance-override", action="store_true",
                   help="State deliberately that an instance differing from the manifest is "
                        "intended; without it a differing --instance is refused.")
    p.add_argument("--arena-connector", default="groot", choices=["groot"])
    p.add_argument("--use-groot-server", default="true")
    p.add_argument("--model-family", default="gr00t")
    p.add_argument("--suite", default="arena_gr1_fridge",
                   help="Arena suite (config/suites/*.yaml with simulator: "
                        f"isaac_arena). Supported: "
                        f"{', '.join(suites_for_simulator(_SIMULATOR))}. "
                        "Drives the Arena runtime knobs below. (Previously defaulted "
                        "to libero_spatial on this Arena-only script.)")
    p.add_argument("--allow-experimental", action="store_true",
                   help="permit a suite whose manifest declares status: experimental")
    p.add_argument("--gr00t-version", default="n17", choices=["n16", "n17"],
                   help="selects which family_overrides entry supplies the "
                        "embodiment tag (n16 = native GR1 head)")
    # A single episode is not a measurement: the rate can only be 0.0 or 1.0, and a sample run must still produce a rate that means something. Three is the floor for any episode count.
    p.add_argument("--eval-trials", default="3")
    # 100 is the Arena eval protocol seed, matching run_arena.py, eval_entry.py and
    # the pipeline parameter. This defaulted to 1000 (the LIBERO value).
    p.add_argument("--eval-seed", default="100")
    p.add_argument("--eval-task-ids", default="all")
    p.add_argument("--eval-source-dir", required=True,
                   help="SageMaker sourcedir tarball S3 URI for sagemaker_submit_directory. "
                        "The baked-code path ignores its contents, but the object MUST exist "
                        "(SageMaker downloads it); point at any valid sourcedir tarball in your "
                        "account, e.g. one produced by run_arena.py's upload step.")
    p.add_argument("--hf-secret", default=None,
                   help="HF secret name (default: from config or vla-pipeline/hf-token)")
    # --- Arena runtime OVERRIDES (default: resolved from the suite manifest) ---
    # `--task-name` is deliberately absent, as on run_arena.py: the task IS the
    # suite, and an independent task flag is how a run ends up evaluating something
    # the checkpoint was not trained for.
    ov = p.add_argument_group(
        "Arena overrides (default: resolved from the suite manifest)")
    ov.add_argument("--policy-config-yaml", default=None,
                    help="Container path to the Arena GR00T closed-loop config YAML "
                         "(EVAL_POLICY_CONFIG_YAML); overrides arena.policy_config. "
                         "REQUIRED for a valid run -- it must match the served "
                         "checkpoint's embodiment. There is no auto-discovery: the "
                         "eval entry rejects it, because Arena's packaged example "
                         "configs are embodiment placeholders.")
    ov.add_argument("--embodiment-tag", default=None,
                    help="GR00T server --embodiment-tag (SM_HP_EMBODIMENT_TAG); MUST "
                         "match the served checkpoint's experiment_cfg/metadata.json "
                         "tag. Default: the suite's family_overrides value.")
    ov.add_argument("--arena-embodiment", default=None,
                    help="Arena --embodiment twin flag (e.g. gr1_joint); pass 'NONE' "
                         "to omit it (EVAL_ARENA_EMBODIMENT).")
    ov.add_argument("--object", dest="object", default=None,
                    help="Arena per-task --object selector; pass 'NONE' to omit it "
                         "(EVAL_OBJECT).")
    p.add_argument("--posctrl-revision", dest="posctrl_revision", default="",
                   help="The IMMUTABLE 40-hex commit of the positive-control checkpoint. Required "
                        "when --posctrl-n16 is true: the evaluator refuses a moving tag, because a "
                        "report naming 'main' pins nothing and cannot be reproduced.")
    p.add_argument("--posctrl-repo", dest="posctrl_repo", default="",
                   help="Override the HF repo served by the N1.6 posctrl path (e.g. "
                        "nvidia/GR00T-N1.6-3B for a raw-base P4 anchor). Empty = NVIDIA tuned default.")
    p.add_argument("--posctrl-n16", dest="posctrl_n16", default="false",
                   help="If 'true', run the N1.6 native-GR1 positive control (EVAL_POSCTRL_N16): "
                        "ignores the mounted checkpoint, uses the baked N1.6 venv, downloads the NVIDIA "
                        "GN1.6 checkpoint, serves it under embodiment GR1.")
    args = p.parse_args()

    # Refused HERE, not in the container. The evaluator fails closed on an empty revision, but it does
    # so after a GPU node has been provisioned and the image pulled -- and the operator who omitted the
    # flag learns about it from a log rather than from the command they just typed.
    if str(args.posctrl_n16).strip().lower() == "true":
        if not re.fullmatch(r"[0-9a-f]{40}", args.posctrl_revision.strip()):
            raise SystemExit(
                f"--posctrl-n16 true requires --posctrl-revision <40 hex>, got "
                f"{args.posctrl_revision!r}. The positive control published resolved_commit='main' "
                f"once, a moving tag that pins nothing; an explicit immutable commit is what makes "
                f"the control reproducible.")

    try:
        _trials = int(args.eval_trials)
    except ValueError:
        sys.exit(f"FATAL: --eval-trials must be an integer, got {args.eval_trials!r}")
    if _trials < 1:
        sys.exit("FATAL: --eval-trials must be a positive integer (it is the number "
                 "of complete episodes to evaluate)")
    args.eval_trials = str(_trials)
    if args.eval_task_ids not in ("all", "[0]"):
        sys.exit(f"FATAL: --eval-task-ids must be 'all' or '[0]' for Arena single-task "
                 f"suites, got {args.eval_task_ids!r}")

    # The N1.6 positive control serves an N1.6 checkpoint under embodiment GR1, so
    # it only makes sense on the n16 route. Left at the n17 default it would resolve
    # the n17 embodiment tag while eval_entry starts the n16 server -- disagreeing
    # with itself. Fail at submit rather than burn the job.
    if args.posctrl_n16.strip().lower() == "true" and args.gr00t_version != "n16":
        sys.exit(
            f"FATAL: --posctrl-n16 true serves an N1.6 checkpoint under embodiment "
            f"GR1, but --gr00t-version is {args.gr00t_version!r}. Pass "
            f"--gr00t-version n16 so the resolved embodiment tag and the eval "
            f"server route agree.")

    # Resolve the Arena runtime from the suite manifest, via the SAME
    # implementation run_arena.py uses (vla_pipeline.arena), so the two launchers
    # cannot drift apart. Fails at submit on an unrunnable suite.
    suite = resolve_suite(args.suite)
    try:
        gate_suite(suite, allow_experimental=args.allow_experimental)
        knobs = resolve_runtime(
            suite, args.model_family, args.gr00t_version,
            embodiment_tag=args.embodiment_tag,
            policy_config=args.policy_config_yaml,
            arena_embodiment=args.arena_embodiment,
            obj=args.object,
        )
    except ArenaConfigError as e:
        sys.exit(f"FATAL: {e}")

    # Account-portable resolution -- no hardcoded account/role/bucket. Fail-loud
    # (ConfigError names any unresolved field) rather than assume an account.
    # Validate the checkpoint URI shape BEFORE anything that needs AWS: this is pure
    # argument checking and must not depend on SSM/credentials being resolvable.
    #
    # The evaluator opens the canonical `model.tar.gz` from the mounted channel, while
    # the recorded source identity (EVAL_MODEL_SOURCE_URI) is whatever URI was passed
    # here. Mounting the PARENT PREFIX therefore let the evaluated bytes differ from
    # the bytes the run claims to have evaluated: point this at `.../other.tar.gz` in a
    # prefix that also holds a `model.tar.gz` and the sibling gets evaluated under the
    # other object's identity. The documented and only supported form is the canonical
    # object, so require it rather than silently reinterpreting anything else.
    if not args.checkpoint_s3.startswith("s3://"):
        sys.exit(f"--checkpoint-s3 must be an s3:// URI, got {args.checkpoint_s3!r}")
    if not args.checkpoint_s3.endswith("/model.tar.gz"):
        sys.exit(
            f"--checkpoint-s3 must name the canonical checkpoint OBJECT "
            f"(.../model.tar.gz), got {args.checkpoint_s3!r}. A prefix or a "
            f"differently-named archive is not supported: the evaluator opens "
            f"model.tar.gz from the mount, so any other input would evaluate one "
            f"object while recording another as the source identity.")
    _bucket_key = args.checkpoint_s3[len("s3://"):]
    if "/" not in _bucket_key:
        sys.exit(f"--checkpoint-s3 has no key component: {args.checkpoint_s3!r}")
    _ckpt_bucket, _ckpt_key = _bucket_key.split("/", 1)

    # Resolve hardware before AWS calls so an unapproved override fails before submission.
    _spec = resolve(args.model_family, _SIMULATOR)
    _eval_instance = select_instance(_spec, "eval", args.instance,
                                    args.acknowledge_instance_override)
    _eval_volume = select_volume_gb(_spec, "eval", args.volume_size)
    # See run_libero.py: local-NVMe instances cap VolumeSizeInGB at their local total.
    check_storage_fits(_eval_instance, _eval_volume)

    # The FOURTH emit site for the digest rule, and it was missed. run_libero, run_arena and run_matrix
    # all submit a digest so the attestation can name the BYTES that ran; this submitter passed
    # args.eval_image straight through, so a positive control -- the traceability baseline -- could run
    # on a mutable tag. Same one-of-N pattern as the three before it.
    _checked_eval_image = resolve_image_digest(args.eval_image)
    print(f"  Eval image (digest): {_checked_eval_image}")
    # The FIFTH site for the capability gate, found by an L0 call-site enumeration in one second --
    # after I had claimed this class closed across "all three launchers". This submitter is a fourth
    # launcher, and it submits the same evaluator images.
    # Reuses the spec resolved above; a second resolve() of the same pair is a second answer waiting
    # to disagree with the first.
    if _spec.required_image_capability:
        require_image_capability(_checked_eval_image, _spec.required_image_capability)

    cfg = load_config()
    # C1: the JOB EXECUTION identity must be the component workload role, not the shared
    # Foundation role. Standalone SimEval creates a training job directly, so using the
    # Foundation role gave evaluator code the broad shared identity and bypassed the
    # trust boundary entirely -- the pipeline path already uses workload_role_arn, so
    # the two paths granted different privileges for the same container.
    region, role, bucket = cfg.region, cfg.workload_role_arn, cfg.bucket
    if args.hf_secret is None:
        args.hf_secret = getattr(cfg, "hf_secret_name", "vla-pipeline/hf-token")
    # Baked entrypoint (docker_entrypoint_multi.sh) ignores the sourcedir, but the SM training
    # toolkit still expects a valid sagemaker_program/submit_directory that EXISTS in S3.
    # Supplied by the caller via --eval-source-dir (required).
    eval_source_dir = args.eval_source_dir

    # Mount the EXACT checkpoint object, not its parent prefix. The URI shape was
    # already validated above; what remains needs an S3 client.
    #
    # SageMaker has no single-object input type; an S3Prefix equal to the full key
    # selects that object, but ALSO any key that merely starts with it (a stray
    # `model.tar.gz.sha256` beside it would be pulled in too). Verify the object
    # exists and that it is the only key under that prefix, so the mounted channel is
    # exactly the artifact whose identity gets recorded.
    _s3 = boto3.client("s3", region_name=region)
    try:
        _s3.head_object(Bucket=_ckpt_bucket, Key=_ckpt_key)
    except Exception as exc:
        sys.exit(f"--checkpoint-s3 {args.checkpoint_s3!r} cannot be read: {exc}")
    _listed = _s3.list_objects_v2(Bucket=_ckpt_bucket, Prefix=_ckpt_key).get(
        "Contents", [])
    _extra = sorted(o["Key"] for o in _listed if o["Key"] != _ckpt_key)
    if _extra:
        sys.exit(
            f"--checkpoint-s3 {args.checkpoint_s3!r} is not the only object under that "
            f"key prefix; these would also be mounted: {_extra}. Remove them or move "
            f"the checkpoint so the mounted channel is exactly the recorded artifact.")
    ckpt_prefix = args.checkpoint_s3

    # Unique per LAUNCH, and it names the instance type it asked for.
    #
    # This was `f"simeval-only-{connector}-{int(time.time())}"`, which has one-second resolution:
    # two launches inside the same second requested the SAME name and SageMaker refused the second
    # with `ResourceInUse`. That error reads as "the resource is taken", so a client-side name
    # collision was indistinguishable from the service being unable to satisfy the request.
    #
    # The instance type is in the name so a caller reading a job list can tell what each job asked
    # for without describing every one of them.
    _type_tag = re.sub(r"[^a-z0-9]", "", _eval_instance.replace("ml.", "").replace("xlarge", "xl"))
    job = f"simeval-only-{args.arena_connector}-{_type_tag}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    sm = boto3.client("sagemaker", region_name=region)
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": _checked_eval_image,
                                "TrainingInputMode": "File"},
        RoleArn=role,
        InputDataConfig=[{
            "ChannelName": "model",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": ckpt_prefix, "S3DataDistributionType": "FullyReplicated"}},
            "CompressionType": "None",
        }],
        OutputDataConfig={"S3OutputPath": cfg.s3_uri("eval"), "CompressionType": "GZIP"},
        ResourceConfig={"InstanceType": _eval_instance, "InstanceCount": 1,
                        "VolumeSizeInGB": _eval_volume},
        StoppingCondition={"MaxRuntimeInSeconds": 86400},
        HyperParameters={
            "sagemaker_submit_directory": eval_source_dir,
            "sagemaker_program": "eval_entry.py",
            "USE_GROOT_SERVER": args.use_groot_server,
        },
        Environment=build_environment(args, suite, knobs),
    )
    print(f"Submitted SimEval-only job: {job}")
    print(f"  image={args.eval_image}")
    print(f"  checkpoint={args.checkpoint_s3} (mounted object {ckpt_prefix})")
    print(f"  instance={_eval_instance}  connector={args.arena_connector}")
    print(f"  suite={suite.name} task={knobs['task_name']} "
          f"embodiment_tag={knobs['embodiment_tag']} "
          f"--embodiment {knobs['arena_embodiment']} "
          f"--object {knobs['object']} num_episodes={args.eval_trials}")
    print(f"  policy_config={knobs['policy_config_yaml']}")
    print(f"  region={region}  bucket={bucket}")
    print(f"Status: aws sagemaker describe-training-job --region {region} "
          f"--training-job-name {job}")


if __name__ == "__main__":
    main()
