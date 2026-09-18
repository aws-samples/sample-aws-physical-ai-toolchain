"""SageMaker entry point: OpenVLA-OFT LIBERO evaluation (models/ adapter).

Contract: models/contract.md rev6. Extends the proven v1 eval recipe
(eval/eval_entry.py, the reference pins, 5 verified cloud runs) with the
contract machinery: reads checkpoint_manifest.json from the mounted model
channel, applies its input_config VERBATIM (env overrides of manifest
fields are fatal), recomputes the weights digest over the extracted tree
(the mounted-bytes half of the content binding), records the source
artifact's S3 identity via HeadObject, emits a schema-v2 report, and
SELF-VALIDATES with the shared gate validator before declaring success.

Config via environment (set by the pipeline):
  EVAL_SUITE          libero_spatial | libero_object | libero_goal | libero_10
  EVAL_TRIALS         trials per task
  EVAL_SEED           evaluation seed
  EVAL_CHECKPOINT     local dir (pipeline mode) or HF repo id
  EVAL_CKPT_REV       HF revision pin ("" for pipeline-local checkpoints)
  EVAL_MODEL_SOURCE_URI  s3://... of the FineTune artifact (pipeline mode;
                      the mounted channel does not expose object metadata)

Fail-fast: no fallbacks that mask failure.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capped_reader import DEFAULT_MAX_ARCHIVE_BYTES, capped_tar_open  # noqa: E402
from digest import weights_digest  # noqa: E402  (shipped in sourcedir)
from source_identity import from_archive as si_from_archive  # noqa: E402
from source_identity import from_snapshot as si_from_snapshot  # noqa: E402
from validator import validate_report  # noqa: E402

OUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
# PIPELINE CHANGE: metrics.json + small evidence written to SM_MODEL_DIR so the
# Validate step consumes it via step-property reference (ModelArtifacts).
# Videos/large evidence stay in SM_OUTPUT_DATA_DIR.
METRICS_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
WORK = "/opt/ml/code"
OFT_DIR = os.path.join(WORK, "openvla-oft")
LIBERO_DIR = os.path.join(WORK, "LIBERO")

OFT_REPO = "https://github.com/moojink/openvla-oft.git"
OFT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
MUJOCO_PIN = "mujoco==3.3.1"
TFMD_PIN = "tensorflow-metadata==1.17.3"
FLASH_ATTN = "flash-attn==2.5.5"
SUITE_TASK_COUNT = 10  # full LIBERO spatial suite
_EVAL_TASK_IDS_RAW = os.environ.get("EVAL_TASK_IDS", "all")
if _EVAL_TASK_IDS_RAW and _EVAL_TASK_IDS_RAW != "all":
    raise SystemExit(
        "EVAL_TASK_IDS task subsetting is not supported by the OpenVLA/LIBERO "
        "evaluator (full suite only). Remove EVAL_TASK_IDS or set it to 'all'.")

VALID_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}


def _fatal(msg: str) -> None:
    print(f"[eval] FATAL: {msg}", flush=True)
    sys.exit(1)


def _require_int(name: str, value: str, lo: int, hi: int) -> str:
    # Strict parse before ANY use (BLOCKER fix: parameters are caller-
    # controlled strings; never let raw text reach subprocess construction).
    try:
        n = int(value)
    except ValueError:
        _fatal(f"{name} not an integer: {value!r}")
    if not (lo <= n <= hi):
        _fatal(f"{name}={n} outside [{lo}, {hi}]")
    return str(n)


def _require_env(name: str, supplied_by: str) -> str:
    """Read a required setting, or refuse in a way an operator can act on.

    `os.environ[name]` raises a bare KeyError naming the variable and nothing else -- not what it
    is for, and not what is supposed to set it. Since these are read at MODULE level, that
    KeyError is the entire output of a job that died before doing any work, which leaves an
    operator with a variable name and no next step.
    """
    value = os.environ.get(name, "")
    if not value:
        _fatal(f"{name} is unset or empty. It is supplied by {supplied_by}. This evaluator "
               f"cannot select what to evaluate without it, so it refuses rather than guessing.")
    return value


CKPT = _require_env("EVAL_CHECKPOINT",
                    "the launcher as a hyperparameter (SageMaker maps it into the environment)")
CKPT_REV = os.environ.get("EVAL_CKPT_REV", "")
SUITE = _require_env("EVAL_SUITE", "the suite manifest, via the launcher")
if SUITE not in VALID_SUITES:
    _fatal(f"EVAL_SUITE {SUITE!r} not in {sorted(VALID_SUITES)}")
TRIALS = _require_int("EVAL_TRIALS", os.environ.get("EVAL_TRIALS", "5"), 1, 50)
SEED = _require_int("EVAL_SEED", os.environ.get("EVAL_SEED", "7"), 0, 2**31 - 1)
SOURCE_URI = os.environ.get("EVAL_MODEL_SOURCE_URI", "")
import re as _re  # noqa: E402  (intentional: placed after the env-var setup above)


# I4: the GPU evaluators extracted the checkpoint archive with NO resource bound. Path safety is
# not resource safety: a well-formed archive with no traversal and no links can still carry
# millions of members or expand past the volume, and this extraction happens on paid accelerator
# capacity BEFORE Validate ever sees the run -- so the cost is incurred exactly where it hurts
# most. Validate's safe_extract enforces the same two limits.
#
# Self-contained on purpose, matching safe_extract's reason: these entry scripts are delivered to
# bare containers (Arena from a baked image, LIBERO via the staged sourcedir) and cannot rely on
# importing vla_pipeline.common. A shared module would need a new ECR image for the Arena path.
#
# Enforced while reading HEADERS, never after tar.getmembers(): materialising the member list is
# itself the unbounded operation the member limit exists to prevent.
_MAX_ARCHIVE_MEMBERS = int(os.environ.get("MAX_ARCHIVE_MEMBERS", "200000"))
_MAX_ARCHIVE_BYTES = int(os.environ.get("MAX_ARCHIVE_BYTES", str(64 * 1024 ** 3)))


def _bound_archive(tar, label="checkpoint"):
    """Refuse an archive whose declared size or member count exceeds the run's budget."""
    members = 0
    declared = 0
    for member in tar:
        members += 1
        if members > _MAX_ARCHIVE_MEMBERS:
            print(f"FATAL: {label} archive declares more than {_MAX_ARCHIVE_MEMBERS} members; "
                  f"refusing to extract (archive resource guard). Enforced while reading "
                  f"headers, so the rest of the archive was never read.", flush=True)
            sys.exit(1)
        if member.isreg():
            declared += member.size
            if declared > _MAX_ARCHIVE_BYTES:
                print(f"FATAL: {label} archive declares more than {_MAX_ARCHIVE_BYTES} expanded "
                      f"bytes, which exceeds what the job volume holds; refusing to extract "
                      f"(archive resource guard).", flush=True)
                sys.exit(1)
    print(f"[archive-guard] {label}: {members} members, {declared} declared bytes -- within "
          f"limits ({_MAX_ARCHIVE_MEMBERS} members, {_MAX_ARCHIVE_BYTES} bytes)", flush=True)
    # The caller extracts from the same handle; rewind so iteration did not consume it.
    tar.fileobj.seek(0)
    tar.offset = 0
    tar.members = []
    tar._loaded = False

if CKPT_REV and not _re.fullmatch(r"[0-9a-f]{40}", CKPT_REV):
    _fatal(f"EVAL_CKPT_REV not a 40-hex revision: {CKPT_REV!r}")


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


# cycle-16 I3: one deadline for the WHOLE job, fixed at import. Each caller gets what REMAINS.
# Every earlier version returned 0.95 * budget on every call, so with a 100-second budget and 60
# already spent the next command was handed another 95 -- the headroom the comment promised did not
# exist, and N subprocesses each received the entire allowance.
_JOB_BUDGET_S = float(os.environ.get("VLA_MAX_RUNTIME_SECONDS", "28800"))
# Scaled below the budget so a timeout fires BEFORE SageMaker terminates the job: TimeoutExpired names
# the command, an external kill does not. Anchored at import, which is the earliest point in the
# entrypoint and therefore includes installation and download time in the accounting.
_DEADLINE = time.monotonic() + _JOB_BUDGET_S * 0.95
_MIN_SUBPROCESS_TIMEOUT_S = 30.0


def _subprocess_timeout() -> float:
    """Derive a subprocess timeout from the SAME budget SageMaker enforces via max_run.

    Deliberately NOT a fresh constant: a second number that disagrees with the declared budget is the
    defect cycle-8 I12 reported, where max_runtime_seconds was declared and max_run appeared nowhere.
    The pipeline passes VLA_MAX_RUNTIME_SECONDS from params["max_runtime_seconds"].

    Scaled BELOW the budget on purpose. At 100% SageMaker would kill the job first and the failure
    would surface as an opaque termination; timing out a little earlier raises TimeoutExpired naming
    the command, which is diagnosable. The fallback matches defaults.json (28800) so a container run
    outside the pipeline still has a bound rather than none.
    """
    remaining = _DEADLINE - time.monotonic()
    # A floor rather than a non-positive timeout: subprocess rejects <= 0, and a budget already spent
    # should fail the very next command loudly instead of raising a ValueError about the timeout.
    return max(_MIN_SUBPROCESS_TIMEOUT_S, remaining)


def run(cmd: list[str], **kw) -> None:
    log("$ " + " ".join(cmd))
    # A call with no timeout can hang for the whole budget and produce nothing; the step then
    # looks like a long job rather than a stuck one.
    kw.setdefault("timeout", _subprocess_timeout())
    subprocess.run(cmd, check=True, **kw)


def pip(*args: str) -> None:
    # pip retry/backoff to absorb transient PyPI 502s within the job (the top
    # recurring cause of spurious ladder failures) rather than re-running eval.
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "--retries", "10", "--timeout", "60", *args])


def load_family_schemas() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "defaults.json")) as fh:
        d = json.load(fh)
    return {d["family"]: {"input_config_schema": d["input_config_schema"],
                          "provenance_keys": d["provenance_keys"]}}


def head_object_identity(s3_uri: str) -> dict:
    import boto3
    bucket, _, key = s3_uri.replace("s3://", "").partition("/")
    resp = boto3.client("s3").head_object(
        Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    version_id = resp.get("VersionId")
    if not version_id:
        log("FATAL: source artifact has no VersionId -- pipeline bucket must "
            "have versioning enabled (contract: Artifact identity binding)")
        sys.exit(1)
    return {
        "s3_uri": s3_uri,
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "etag": resp["ETag"].strip('"'),
    }


def _patch_unnorm_key(src: str) -> str:
    """Make upstream apply the manifest's unnorm_key VERBATIM.

    Two edits are required, and patching only the first (which is what this used to do)
    left the contract unmet. Pinned upstream `check_unnorm_key` reads:

        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, ...

    The first edit makes the initial assignment honour cfg.unnorm_key. But the fallback
    then still fires: a manifest key ABSENT from norm_stats whose `_no_noops` variant
    exists was silently replaced, so the recorded input configuration described a
    different normalization from the one actually applied to actions.

    The second edit guards the substitution on there being no explicit key. Upstream
    behaviour is preserved when none is supplied; when the manifest supplies one,
    upstream's own assert fails loud on a genuinely missing key instead of substituting.

    Raises RuntimeError if either target is absent -- upstream having changed means the
    verbatim guarantee cannot be made, which must stop the run rather than proceed.
    """
    assignment = "    unnorm_key = cfg.task_suite_name"
    if assignment not in src:
        raise RuntimeError(
            "expected unnorm_key assignment not found in check_unnorm_key; upstream "
            "changed. The manifest's unnorm_key cannot be applied, so refusing to run.")
    src = src.replace(
        assignment,
        "    unnorm_key = cfg.unnorm_key if cfg.unnorm_key else cfg.task_suite_name",
        1)
    fallback = '        unnorm_key = f"{unnorm_key}_no_noops"'
    if fallback not in src:
        raise RuntimeError(
            "expected _no_noops fallback not found in check_unnorm_key; upstream "
            "changed. The manifest's unnorm_key cannot be guaranteed verbatim, so "
            "refusing to run.")
    return src.replace(
        fallback,
        "        if not cfg.unnorm_key:\n"
        '            unnorm_key = f"{unnorm_key}_no_noops"',
        1)


_OPENVLA_GUARDS = '''

def _vla_strict_load_audit(info, checkpoint):
    """Reject a model whose weights did not all come from the checkpoint.

    Upstream loads with AutoModelForVision2Seq.from_pretrained and never requests or checks
    loading diagnostics. Transformers initializes missing tensors, warns, and returns the
    model, so an incomplete checkpoint yields a running model containing randomly initialized
    weights -- and the run is still reported as a checkpoint evaluation. Rejecting logged
    episode exceptions does not cover this: successful construction is not an exception.
    """
    problems = {}
    # I7: an ABSENT field was read as empty, so a load reporting no diagnostics at all passed
    # as clean. Absence means they could not be read, not that there were no problems.
    missing_fields = []
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if isinstance(info, dict):
            present, values = key in info, info.get(key)
        else:
            present, values = hasattr(info, key), getattr(info, key, None)
        if not present or values is None:
            missing_fields.append(key)
            continue
        if values:
            problems[key] = list(values)
    if missing_fields:
        raise RuntimeError(
            f"loading diagnostics for {checkpoint} are incomplete: {missing_fields} absent. "
            f"An unreadable diagnostic is not a clean load, so the weights cannot be shown to "
            f"have come from the checkpoint.")
    if problems:
        summary = "; ".join(
            f"{key}={values[:8]}{'...' if len(values) > 8 else ''} ({len(values)} total)"
            for key, values in problems.items())
        raise RuntimeError(
            f"strict load audit FAILED for {checkpoint}: {summary}. Missing tensors are "
            f"randomly initialized and the model still runs, so this would have evaluated a "
            f"partly random policy and reported it as a checkpoint policy.")
    print(f"[wrapper] strict load audit PASSED for {checkpoint}: no missing, unexpected or "
          f"mismatched tensors", flush=True)




# The processor classes a real OpenVLA checkpoint legitimately names, verified against the
# published openvla/openvla-7b and the pinned OFT finetune output: both save an auto_map
# pointing at processing_prismatic. These are the SAME classes get_vla registers from the
# image, so with trust_remote_code=False the registry supplies them and the checkpoint's copy
# is not imported. Any other value would be checkpoint-supplied code.
#
# A blanket ban on auto_map rejected the component's own trainer output. The fixture that was
# supposed to catch that carried no processor configuration at all, so it could not.
_TRUSTED_AUTO_MAP = {
    "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
    "AutoProcessor": "processing_prismatic.PrismaticProcessor",
}

# Python modules a real checkpoint carries. The first two are compared against the evaluation
# code; processing_prismatic is present but NOT imported, because trust_remote_code is False.
_EXPECTED_CHECKPOINT_MODULES = (
    "modeling_prismatic.py",
    "configuration_prismatic.py",
    "processing_prismatic.py",
)

def _reject_checkpoint_processor_code(checkpoint):
    """Refuse a checkpoint that supplies its own processor implementation.

    AutoProcessor.from_pretrained(..., trust_remote_code=True) imports and EXECUTES the class
    named in the checkpoint's processor or preprocessor config auto_map, with the evaluation
    job's credentials. The model-code guards cover modeling_prismatic.py and
    configuration_prismatic.py only, so a checkpoint could carry those two unchanged, along
    with valid weights, and still ship arbitrary processor code.

    The trusted processor class is registered from the image, so a checkpoint has no
    legitimate reason to name one. Any auto_map in these configs is refused.
    """
    import glob as _glob

    for name in ("processor_config.json", "preprocessor_config.json",
                 "tokenizer_config.json"):
        path = os.path.join(checkpoint, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                config = json.load(handle)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"{name} in the checkpoint is unreadable ({exc}); refusing to construct a "
                f"processor from a configuration that cannot be inspected.")
        auto_map = config.get("auto_map") if isinstance(config, dict) else None
        if auto_map:
            if not isinstance(auto_map, dict):
                raise RuntimeError(
                    f"checkpoint {name} has a non-object auto_map ({auto_map!r}); refusing to "
                    f"interpret it.")
            unexpected = {key: value for key, value in auto_map.items()
                          if _TRUSTED_AUTO_MAP.get(key) != value}
            if unexpected:
                raise RuntimeError(
                    f"checkpoint {name} names processor classes this evaluator does not "
                    f"trust: {unexpected!r}. Allowed: {_TRUSTED_AUTO_MAP!r}. An entry outside "
                    f"that set would import and execute code supplied by the checkpoint using "
                    f"this job's credentials. Refusing to load.")
    # A checkpoint-local python module is only reachable through an auto_map, but its presence
    # alongside the two audited model files is worth naming rather than ignoring.
    extra = sorted(
        os.path.basename(path) for path in _glob.glob(os.path.join(checkpoint, "*.py"))
        if os.path.basename(path) not in _EXPECTED_CHECKPOINT_MODULES)
    if extra:
        raise RuntimeError(
            f"checkpoint contains unaudited python modules {extra}. Only "
            f"modeling_prismatic.py and configuration_prismatic.py are compared against the "
            f"evaluation code; anything else could be imported without being checked.")
    print("[wrapper] checkpoint supplies no processor code; using the registered class",
          flush=True)

def _refuse_model_code_substitution(curr_filepath, checkpoint_filepath):
    """Refuse to REPLACE checkpoint model code with the current checkout's copy.

    Upstream check_model_logic_mismatch() copies the current repository's
    modeling_prismatic.py / configuration_prismatic.py over the checkpoint's when they
    differ, and update_auto_map() rewrites config.json -- both immediately before loading.
    The wrapper hashes the checkpoint BEFORE evaluation, so after substitution the reported
    digest no longer describes the tree that was loaded, and a supplied checkpoint can carry
    different model logic from the logic that earned its score. The backup files upstream
    leaves behind also add members the digest never saw.

    Rather than silently substituting, this fails and names the file. A checkpoint whose model
    code already matches the evaluation code is untouched and its digest stays meaningful.
    """
    raise RuntimeError(
        f"checkpoint model code differs from the evaluation code: upstream would replace "
        f"{checkpoint_filepath} with {curr_filepath} immediately before loading. The digest "
        f"recorded for this checkpoint would then describe a tree that was never evaluated, "
        f"and the score would belong to model logic the checkpoint does not contain. "
        f"Refusing to substitute; supply a checkpoint whose model code matches.")
'''


def _patch_openvla_load_guards(src: str) -> str:
    """Install the strict load audit and refuse model-code substitution.

    Raises RuntimeError if any anchor is absent, since upstream having moved means an
    incomplete or substituted model could again be evaluated and reported as a checkpoint.
    """
    edits = [
        # Critical 4: request and check loading diagnostics at the real constructor. The
        # keyword must follow the positional checkpoint argument.
        (
            "    vla = AutoModelForVision2Seq.from_pretrained(\n"
            "        cfg.pretrained_checkpoint,\n",
            "    vla, _loading_info = AutoModelForVision2Seq.from_pretrained(\n"
            "        cfg.pretrained_checkpoint,\n"
            "        output_loading_info=True,\n",
        ),
        # Critical 5 (cont.): update_auto_map() rewrites the checkpoint's config.json and
        # creates a timestamped backup beside it -- unconditionally, even when the mapping
        # already matches. Both change the tree after its digest was taken: the rewrite
        # changes config.json's bytes and the backup adds a member the digest never saw.
        # Replaced with read-only validation.
        (
            '    # Create timestamped backup\n'
            '    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\n'
            '    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")\n'
            '    shutil.copy2(config_path, backup_path)\n'
            '    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")\n'
            '\n'
            '    # Read and update the config\n'
            '    with open(config_path, "r") as f:\n'
            '        config = json.load(f)\n'
            '\n'
            '    config["auto_map"] = {\n'
            '        "AutoConfig": "configuration_prismatic.OpenVLAConfig",\n'
            '        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",\n'
            '    }\n'
            '\n'
            '    # Write back the updated config\n'
            '    with open(config_path, "w") as f:\n'
            '        json.dump(config, f, indent=2)\n',
            '    # READ-ONLY validation: this checkpoint was digested before evaluation, so\n'
            '    # rewriting its config.json or dropping a backup file beside it would mean the\n'
            '    # recorded digest no longer describes what was loaded.\n'
            '    with open(config_path, "r") as f:\n'
            '        config = json.load(f)\n'
            '\n'
            '    _expected_auto_map = {\n'
            '        "AutoConfig": "configuration_prismatic.OpenVLAConfig",\n'
            '        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",\n'
            '    }\n'
            '    _actual_auto_map = config.get("auto_map")\n'
            '    if _actual_auto_map != _expected_auto_map:\n'
            '        raise RuntimeError(\n'
            '            "checkpoint config.json auto_map does not match the evaluation code: "\n'
            '            f"found {_actual_auto_map!r}, expected {_expected_auto_map!r}. Upstream "\n'
            '            "would rewrite it here and leave a timestamped backup, both of which "\n'
            '            "change the checkpoint after its digest was taken -- so the recorded "\n'
            '            "digest would no longer describe what was evaluated. Refusing to "\n'
            '            "mutate; supply a checkpoint whose auto_map already matches.")\n'
            '    print("auto_map already matches the evaluation code; checkpoint left "\n'
            '          "unmodified")\n',
        ),
        # Upstream's trailing narration claims the config was updated. Nothing is updated
        # now, and a log saying otherwise would mislead an operator reading a failure.
        (
            '    print(f"Updated config.json at: {os.path.abspath(config_path)}")\n'
            '    print("Changes made:")\n'
            "    print('  - Set AutoConfig to \"configuration_prismatic.OpenVLAConfig\"')\n"
            "    print('  - Set AutoModelForVision2Seq to "
            "\"modeling_prismatic.OpenVLAForActionPrediction\"')\n",
            "",
        ),
        # Critical (cycle 5) 1: get_processor loads with trust_remote_code=True, so a class
        # named in the checkpoint's processor/preprocessor config auto_map is IMPORTED AND
        # EXECUTED with SimEval's credentials. The model-code guards above cover only
        # modeling_prismatic.py and configuration_prismatic.py, so a checkpoint could carry
        # those two files unchanged and valid weights while supplying an additional processor
        # implementation. get_vla already registers the trusted PrismaticProcessor, so
        # trust_remote_code served only to let a checkpoint override it.
        (
            "    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, "
            "trust_remote_code=True)\n",
            "    _reject_checkpoint_processor_code(cfg.pretrained_checkpoint)\n"
            "    # trust_remote_code=False: the trusted processor class is registered from the\n"
            "    # image by get_vla, so resolution goes through the registry rather than\n"
            "    # importing code out of the checkpoint.\n"
            "    return AutoProcessor.from_pretrained(\n"
            "        cfg.pretrained_checkpoint, trust_remote_code=False)\n",
        ),
        # Critical 5: both copy sites in check_model_logic_mismatch.
        (
            "            shutil.copy2(curr_filepath, checkpoint_filepath)\n",
            "            _refuse_model_code_substitution(curr_filepath, checkpoint_filepath)\n",
        ),
        (
            "    else:\n"
            "        # If file doesn't exist in checkpoint directory, copy it\n"
            "        shutil.copy2(curr_filepath, checkpoint_filepath)\n",
            "    else:\n"
            "        # If file doesn't exist in checkpoint directory, upstream copies it in.\n"
            "        # A checkpoint MISSING its model code is not a checkpoint we can attest.\n"
            "        _refuse_model_code_substitution(curr_filepath, checkpoint_filepath)\n",
        ),
    ]
    for index, (target, replacement) in enumerate(edits, start=1):
        if src.count(target) != 1:
            raise RuntimeError(
                f"openvla load-guard anchor {index} occurs {src.count(target)} times in the "
                f"pinned openvla_utils.py (expected exactly 1); upstream changed. An "
                f"incomplete or substituted model could again be evaluated, so refusing "
                f"to run.")
        src = src.replace(target, replacement, 1)
    # The audit call goes immediately after construction, before the model is used.
    anchor = "        trust_remote_code=True,\n    )\n"
    if src.count(anchor) != 1:
        raise RuntimeError(
            f"openvla load-guard audit anchor occurs {src.count(anchor)} times (expected 1); "
            f"upstream changed. Refusing to run without the load audit.")
    src = src.replace(
        anchor,
        anchor + "    _vla_strict_load_audit(_loading_info, cfg.pretrained_checkpoint)\n", 1)
    return src.rstrip("\n") + "\n" + _OPENVLA_GUARDS


def _select_run_log(logs_dir: str, run_identity: str) -> str:
    """Return the one evaluation log belonging to THIS run.

    The directory was created empty for this invocation and the run identity is embedded in
    the filename by upstream's --run_id_note, so exactly one log must be present and it must
    carry that identity. Modification time is deliberately not consulted: newest-wins was
    the defect, and an unrelated log with a later timestamp must not be able to win.

    Raises ValueError naming the reason; the caller fails loud.
    """
    names = sorted(name for name in os.listdir(logs_dir)
                   if name.startswith("EVAL-") and name.endswith(".txt"))
    if not names:
        raise ValueError(
            f"no EVAL-*.txt log produced in {logs_dir}, so this run has no evaluation "
            f"output to parse.")
    if len(names) != 1:
        raise ValueError(
            f"{len(names)} EVAL-*.txt logs in the per-run directory {logs_dir}: {names}. "
            f"Exactly one is required so the parsed result cannot be attributed to the "
            f"wrong rollout.")
    if run_identity not in names[0]:
        raise ValueError(
            f"log {names[0]} does not carry this run's identity {run_identity!r}. Upstream "
            f"appends --<run_id_note> to the run id, so a log without it was not produced "
            f"by this invocation.")
    return os.path.join(logs_dir, names[0])


def _run_identity() -> str:
    """A token identifying THIS invocation, for binding its evaluation log to it.

    Prefers the SageMaker training job name, which is unique per job and lets an operator
    tie a log back to the execution that produced it. Falls back to the process start time
    and pid, which is unique per invocation within a container but carries no external
    meaning -- so the source is logged rather than left ambiguous.

    Restricted to characters safe in a filename, because upstream embeds this in the log
    file name via --run_id_note.
    """
    raw = os.environ.get("TRAINING_JOB_NAME", "").strip()
    source = "TRAINING_JOB_NAME"
    if not raw:
        try:
            env = json.loads(os.environ.get("SM_TRAINING_ENV", "{}"))
            raw = str(env.get("job_name", "")).strip()
            source = "SM_TRAINING_ENV.job_name"
        except (ValueError, TypeError):
            raw = ""
    if not raw:
        raw = f"local-{int(time.time())}-{os.getpid()}"
        source = "process clock and pid (no SageMaker job identity present)"
    token = re.sub(r"[^A-Za-z0-9._-]", "-", raw)[:96]
    log(f"run identity for log attribution: {token} (from {source})")
    return token


# The source identity of what this invocation loaded, CONSTRUCTED once by whichever branch resolved
# the checkpoint. Previously two seedable dicts, which is what let a fixture make the HF path
# unexecutable: seeding _SOURCE_ARCHIVE with a plain dict sent every test down the archive branch,
# so the snapshot branch that carried I6 never ran and the suite stayed green. A ResolvedCheckpoint
# cannot be partially seeded -- constructing one requires a real directory and passes every field
# check -- so the branch has to actually execute to produce a report.
_RESOLVED = None


def main() -> None:
    # PyPI 502 resilience for EVERY pip in this job (incl. build-isolation
    # subprocess pips that ignore an explicit --retries flag).
    os.environ["PIP_RETRIES"] = "10"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "60"
    pipeline_mode = CKPT.startswith("/")
    log(f"checkpoint={CKPT} rev={CKPT_REV or 'local'} suite={SUITE} "
        f"trials={TRIALS} seed={SEED} pipeline_mode={pipeline_mode}")
    # check=False does NOT make this safe. It suppresses a nonzero EXIT CODE; it does nothing
    # about the BINARY BEING ABSENT, which raises FileNotFoundError from Popen before any exit
    # code exists. Those are different failures and only one of them is handled by check=False.
    #
    # This is a diagnostic -- it logs which accelerators are present and nothing depends on its
    # output -- so its absence must never abort the step. It did: on a machine without the NVIDIA
    # tools the whole run died here, several minutes of setup in, with a traceback that points at
    # a logging call.
    try:
        subprocess.run(["nvidia-smi", "-L"], check=False)
    except (FileNotFoundError, OSError) as _exc:
        log(f"nvidia-smi unavailable ({_exc.__class__.__name__}: {_exc}); continuing, because this call only logs which accelerators are present and nothing reads its output.")

    # ---- [1/5] System libs
    os.chmod("/tmp", 0o1777)

    # --- Idempotence: if the prebuilt image already has the env, skip install ---
    baked_marker = "/opt/vla/.baked_env"
    if os.path.exists(baked_marker):
        with open(baked_marker) as f:
            marker = f.read().strip()
        log(f"BAKED ENV DETECTED: {marker} -- skipping install")
        # I7: existence of the marker used to be sufficient, and its CONTENTS were only logged.
        # The report records fixed OFT_COMMIT and LIBERO_COMMIT regardless, so an older or custom
        # image could execute while the report attributed the result to the pinned
        # implementation -- the provenance claim would be false without anything detecting it.
        # The GR00T trainer already gates on its stamp this way (train_entry.py); this is the
        # same check applied to the evaluator that lacked it.
        # C2 (cycle 15): this demanded the stamp name LIBERO_COMMIT too, which the image never
        # writes -- docker/openvla/Dockerfile:48 stamps only OFT_COMMIT and does not install LIBERO
        # at all. So EVERY baked run exited 1 here: the check required the image to attest to
        # something it deliberately does not own.
        #
        # LIBERO's provenance does not need the stamp. It is established more strongly further down
        # by `git -C LIBERO_DIR checkout LIBERO_COMMIT` at runtime, which pins the actual working
        # tree rather than asserting a label about it. The stamp's job is only to prove the BAKED
        # layers match what this evaluator reports, so it may only require what the image bakes.
        missing_pins = [name for name, commit in (("OFT_COMMIT", OFT_COMMIT),)
                        if commit not in marker]
        if missing_pins:
            log(f"FATAL: baked image stamp {marker!r} does not name {missing_pins}. This image "
                f"was not built from the pinned source this evaluator reports "
                f"(OFT_COMMIT={OFT_COMMIT}), so the run would attribute its result to code it did "
                f"not execute. Refusing to evaluate.")
            sys.exit(1)
        log(f"baked stamp verified against OFT_COMMIT={OFT_COMMIT[:12]}; "
            f"LIBERO revision is verified separately below")
        # Override paths to where the baked image installed everything
        global WORK, OFT_DIR, LIBERO_DIR
        WORK = "/opt/vla"
        OFT_DIR = os.path.join(WORK, "openvla-oft")
        LIBERO_DIR = os.path.join(WORK, "LIBERO")

        # LIBERO may not be baked -- install if missing
        try:
            _libero_mod = __import__("libero")
            # cycle-17 I1: the import result was DISCARDED and a DIRECTORY was checked instead, so the
            # two objects were never connected. A decoy `libero` package earlier on sys.path satisfied
            # the import while the git check inspected an unrelated pristine checkout at LIBERO_DIR --
            # and the run was attributed to code it never executed.
            #
            # Bind them: the file the interpreter actually loaded must live under the tree whose
            # revision is verified below. realpath on both sides, so a symlink cannot straddle them.
            _mod_file = getattr(_libero_mod, "__file__", None)
            if not _mod_file:
                log(f"FATAL: the imported `libero` package reports no __file__, so the code that will "
                    f"run cannot be located. This evaluator reports LIBERO_COMMIT={LIBERO_COMMIT}; an "
                    f"unlocatable implementation cannot be attributed to it.")
                sys.exit(1)
            _mod_real = os.path.realpath(_mod_file)
            _dir_real = os.path.realpath(LIBERO_DIR)
            if os.path.commonpath([_mod_real, _dir_real]) != _dir_real:
                log(f"FATAL: `libero` imports from {_mod_real}, which is NOT under the verified tree "
                    f"{_dir_real}. Something else on sys.path is shadowing the pinned checkout, so the "
                    f"benchmark definitions that would run are not the ones this result would be "
                    f"attributed to. Refusing to evaluate.")
                sys.exit(1)
            # cycle-16 I4: this used to accept an importable LIBERO without checking anything, while
            # the report still recorded the LIBERO_COMMIT constant. My own C2 rationale claimed LIBERO
            # was "pinned by runtime checkout" -- but that checkout is SKIPPED on exactly this path,
            # so on an image with LIBERO preinstalled the provenance claim was unverified.
            #
            # Verify the working tree instead of trusting the import. A revision that cannot be read is
            # refused, not assumed: an unverifiable pin is the same evidence gap as a wrong one.
            _rev = subprocess.run(["git", "-C", LIBERO_DIR, "rev-parse", "HEAD"],
                                  capture_output=True, text=True,
                                  timeout=_subprocess_timeout())
            _head = _rev.stdout.strip()
            if _rev.returncode != 0 or not _head:
                log(f"FATAL: LIBERO is importable at {LIBERO_DIR} but its revision cannot be read "
                    f"({_rev.stderr.strip()[:200]}). This evaluator reports "
                    f"LIBERO_COMMIT={LIBERO_COMMIT}, so an unverifiable tree would attribute the "
                    f"result to code that was never confirmed. Refusing to evaluate.")
                sys.exit(1)
            if _head != LIBERO_COMMIT:
                log(f"FATAL: LIBERO at {LIBERO_DIR} is at {_head}, but this evaluator reports "
                    f"LIBERO_COMMIT={LIBERO_COMMIT}. The benchmark definitions differ from the ones "
                    f"the result would be attributed to. Refusing to evaluate.")
                sys.exit(1)
            # HEAD equality attests the REF, not the CONTENTS: a modified tracked file at the pinned
            # commit passes a rev-parse check while running different code.
            _status = subprocess.run(["git", "-C", LIBERO_DIR, "status", "--porcelain"],
                                     capture_output=True, text=True,
                                     timeout=_subprocess_timeout())
            _dirty = [ln for ln in _status.stdout.splitlines()
                      if ln.strip() and not ln.startswith("?? ")]
            if _status.returncode != 0 or _dirty:
                log(f"FATAL: LIBERO at {LIBERO_DIR} is at the pinned commit {_head[:12]} but its "
                    f"tracked files are MODIFIED: {_dirty[:10]}. HEAD identifies the ref, not the "
                    f"contents, so the code that would run differs from the commit this result would "
                    f"be attributed to. Refusing to evaluate.")
                sys.exit(1)
            log(f"LIBERO verified: imported from {_mod_real}, clean tree at {_head[:12]}")
        except ImportError:
            log("LIBERO not baked -- installing at runtime")
            run(["apt-get", "update", "-qq"])
            run(["apt-get", "install", "-y", "-qq", "--no-install-recommends",
                 "libegl1", "libgles2", "libglib2.0-0", "libsm6", "libxext6",
                 "libxrender1", "libosmesa6", "ffmpeg"])
            run(["git", "clone", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", LIBERO_DIR])
            run(["git", "-C", LIBERO_DIR, "checkout", LIBERO_COMMIT])
            pip("-e", LIBERO_DIR, "--config-settings", "editable_mode=compat")
            pip("-r", os.path.join(OFT_DIR, "experiments/robot/libero/libero_requirements.txt"))
            pip("--force-reinstall", "numpy<2")
            pip(MUJOCO_PIN)
            pip(TFMD_PIN)
    else:
        log("No baked env -- installing from scratch (runtime install path)")
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "--no-install-recommends",
             "libegl1", "libgles2", "libglib2.0-0", "libsm6", "libxext6",
             "libxrender1", "libosmesa6", "ffmpeg", "git"])

        # ---- [2/5] Pinned repos
        run(["git", "clone", OFT_REPO, OFT_DIR])
        run(["git", "-C", OFT_DIR, "checkout", OFT_COMMIT])
        run(["git", "clone", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", LIBERO_DIR])
        run(["git", "-C", LIBERO_DIR, "checkout", LIBERO_COMMIT])

        # ---- [3/5] Python deps (the reference order and pins, v1-proven)
        pip("-e", OFT_DIR)
        pip(TFMD_PIN)
        pip("packaging", "ninja")
        pip(FLASH_ATTN, "--no-build-isolation")
        pip("-e", LIBERO_DIR, "--config-settings", "editable_mode=compat")
        pip("-r", os.path.join(OFT_DIR, "experiments/robot/libero/libero_requirements.txt"))
        pip("--force-reinstall", "numpy<2")
        pip(MUJOCO_PIN)
        pip(TFMD_PIN)

    libero_pkg = os.path.join(LIBERO_DIR, "libero", "libero")
    cfg_dir = os.path.expanduser("~/.libero")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.yaml"), "w") as fh:
        fh.write(
            f"assets: {libero_pkg}/assets\n"
            f"bddl_files: {libero_pkg}/bddl_files\n"
            f"benchmark_root: {libero_pkg}\n"
            f"datasets: {libero_pkg}/../datasets\n"
            f"init_states: {libero_pkg}/init_files\n"
        )

    verify = (
        "from libero.libero import benchmark; import prismatic; "
        "import tensorflow as tf, torch, numpy, mujoco, importlib.metadata as m; "
        "assert tuple(int(p) for p in mujoco.__version__.split('.')[:2]) < (3,10), mujoco.__version__; "
        "assert m.version('tensorflow-metadata')=='1.17.3', m.version('tensorflow-metadata'); "
        "assert numpy.__version__.startswith('1.'), numpy.__version__; "
        "print('verify OK:', tf.__version__, torch.__version__, numpy.__version__, mujoco.__version__)"
    )
    run([sys.executable, "-c", verify])

    # ---- [4/5] Materialize the checkpoint + manifest + digest + identity
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")

    model_artifact_identity = None
    if pipeline_mode:
        tar_path = os.path.join(CKPT, "model.tar.gz")
        if not os.path.isfile(tar_path):
            # Phrased to match the other three producers: one grep diagnoses a failed job whichever
            # family produced it. OpenVLA already handled this case correctly -- with the INVERTED
            # guard, which is the stronger form -- it just said so differently.
            log(f"FATAL: no model.tar.gz in the mounted channel {CKPT!r}. Pipeline mode needs the RAW "
                f"archive: its pre-extraction measurement is what Validate compares against its own, "
                f"and an already-extracted tree cannot supply one.")
            sys.exit(1)
        if not SOURCE_URI:
            log("FATAL: pipeline mode requires EVAL_MODEL_SOURCE_URI "
                "(mounted channel exposes no object metadata)")
            sys.exit(1)
        model_artifact_identity = head_object_identity(SOURCE_URI)
        log(f"source artifact identity: {model_artifact_identity}")
        import tarfile
        extract_dir = os.path.join(WORK, "model")
        os.makedirs(extract_dir, exist_ok=True)
        # schema 3: measure the archive BEFORE extraction. A tree digest excludes
        # manifests, logs and hidden paths, so two different archives can share
        # one; and HeadObject reports the object current when HEAD runs, not the
        # object this job already downloaded. This is the only value that can be
        # compared with Validate's independent measurement.
        from digest import measure_archive as _measure_archive
        _ARCHIVE_SHA, _ARCHIVE_SIZE = _measure_archive(str(tar_path))
        log(f"measured source archive: sha256={_ARCHIVE_SHA} "
            f"size={_ARCHIVE_SIZE}")
        with capped_tar_open(tar_path, "r:*",
                                     max_bytes=DEFAULT_MAX_ARCHIVE_BYTES) as tf:
            _bound_archive(tf)
            try:
                tf.extractall(extract_dir, filter="data")
            except TypeError:
                base = os.path.abspath(extract_dir)
                for m in tf.getmembers():
                    p = os.path.abspath(os.path.join(base, m.name))
                    inside = os.path.commonpath([base, p]) == base
                    if not inside or not (m.isreg() or m.isdir()):
                        log(f"FATAL: unsafe tar member {m.name}")
                        sys.exit(1)
                tf.extractall(extract_dir)
        ckpt_arg = extract_dir
        # Constructed AFTER extraction, from the measurement taken BEFORE it: the tree digest is
        # measured over what inference will read, while the archive bytes could only be measured
        # while they still existed. One object now carries both, so the report cannot name an
        # archive and a tree belonging to different things.
        global _RESOLVED
        _RESOLVED = si_from_archive(load_root=extract_dir, archive_path=str(tar_path),
                                    checkpoint=CKPT, sha256=_ARCHIVE_SHA,
                                    size_bytes=_ARCHIVE_SIZE)
    else:
        if not CKPT_REV:
            log("FATAL: HF mode requires EVAL_CKPT_REV (revision enforcement)")
            sys.exit(1)
        ckpt_arg = os.path.join(WORK, "ckpt", CKPT_REV)
        run([sys.executable, "-c",
             "import sys; from huggingface_hub import snapshot_download; "
             "print(snapshot_download(sys.argv[1], revision=sys.argv[2], "
             "local_dir=sys.argv[3]))",
             CKPT, CKPT_REV, ckpt_arg])
        # The source identity for HF mode, CONSTRUCTED rather than assembled. The previous code
        # put weights_digest()'s "sha256:<hex>" straight into tree_sha256, which the validator
        # requires to be bare 64-hex (I6) -- so every HF run spent its whole inference budget and
        # then failed to publish. from_snapshot measures the tree itself and emits the field in the
        # form the schema wants, so neither the convention nor the measured directory can drift.
        _RESOLVED = si_from_snapshot(load_root=ckpt_arg, repo_id=CKPT, resolved_commit=CKPT_REV)
        log(f"measured source snapshot: repo={CKPT} commit={CKPT_REV} "
            f"tree={_RESOLVED.report_source_fields()['source_snapshot']['tree_sha256']}")

    manifest_path = os.path.join(ckpt_arg, "checkpoint_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    elif pipeline_mode:
        log("FATAL: checkpoint_manifest.json missing from pipeline checkpoint -- "
            "the FineTune step must write it (contract rev6); refusing to fabricate")
        sys.exit(1)
    else:
        # HF mode: SYNTHESIZE the manifest for a published checkpoint
        # (contract: published-checkpoint mode; the three nulls travel together).
        # Published OFT checkpoints use 2 images + proprio (upstream defaults);
        # unnorm_key = the suite's no_noops stats key, matching upstream's
        # resolution for these checkpoints.
        log("HF mode: synthesizing checkpoint manifest for published checkpoint")
        pre_digest = weights_digest(ckpt_arg)
        manifest = {
            "manifest_version": 1,
            "model_family": "openvla",
            "base_checkpoint": CKPT,
            "base_revision": CKPT_REV,
            "train_seed": None,
            "input_config": {
                "num_images_in_input": 2,
                "use_proprio": True,
                "unnorm_key": f"{SUITE}_no_noops",
            },
            "train_recipe": {"repo": OFT_REPO, "commit": OFT_COMMIT,
                             "max_steps": None},
            "weights_digest": pre_digest,
            "dataset_manifest": None,
        }
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    # R5 (a real run): this consumed manifest["input_config"]["num_images_in_input"] directly, so a
    # manifest missing the key died with a bare `KeyError: 'num_images_in_input'` two minutes into a
    # paid job -- after 8 GB had downloaded and been measured -- naming neither what was expected nor
    # where. The schema that describes the key is declared in defaults.json and WAS being checked,
    # but only inside validate_report at the very END of the run (validator.py:486). Checking here
    # moves the same check to the moment the manifest is first trusted, so the failure is named
    # before any inference budget is spent.
    # R5b: the wrong-family checkpoint is the likeliest operator error and produced the least
    # useful failure -- a gr00t checkpoint handed to this evaluator died on KeyError for a field
    # only openvla manifests carry, two minutes into a paid job. The Arena evaluator has always
    # checked this (isaac_arena/gr00t/eval_entry.py:1090); three of four did not.
    _declared_family = manifest.get("model_family")
    if _declared_family != 'openvla':
        log(f"FATAL: this checkpoint was produced for model_family={_declared_family!r} but "
            f"is being evaluated as 'openvla'. Its manifest describes a different family's "
            f"input_config, so every field read from here would be wrong or missing.")
        sys.exit(1)
    from validator import validate_manifest as _validate_manifest
    _validate_manifest(manifest, load_family_schemas())

    # Contract (H2 fix): input_config comes ONLY from the manifest, applied
    # VERBATIM -- including unnorm_key. Any env override attempt is fatal.
    for var in ("EVAL_NUM_IMAGES", "EVAL_USE_PROPRIO", "EVAL_UNNORM_KEY"):
        if os.environ.get(var, ""):
            log(f"FATAL: {var} attempts to override a manifest-owned input_config field")
            sys.exit(1)
    input_config = manifest["input_config"]
    num_images = str(input_config["num_images_in_input"])
    use_proprio = str(input_config["use_proprio"])
    unnorm_key = input_config["unnorm_key"]

    # Content binding: digest over the tree the evaluator will load.
    recomputed = weights_digest(ckpt_arg)
    log(f"weights_digest_recomputed_by_eval: {recomputed}")
    if recomputed != manifest["weights_digest"]:
        log(f"FATAL: digest mismatch -- mounted bytes are not the produced bytes\n"
            f"  manifest: {manifest['weights_digest']}\n  recomputed: {recomputed}")
        sys.exit(1)

    # ---- [5/5] Run the upstream eval
    # C6: give THIS invocation its own empty log directory and an explicit run identity.
    # The wrapper used to take the newest EVAL-* file from a shared directory, matching no
    # run identifier, so an older log with a later mtime could win even when a new log
    # existed -- and its internally consistent numbers would satisfy Validate because the
    # wrapper supplied the current run's labels around them. Upstream builds the log name
    # as EVAL-<suite>-<family>-<timestamp>[--<run_id_note>].txt, so binding the note lets
    # the file be identified rather than guessed.
    run_identity = _run_identity()
    logs_dir = os.path.join(OFT_DIR, "experiments", "logs", run_identity)
    if os.path.exists(logs_dir):
        # A directory that already exists cannot be proven empty of foreign logs.
        log(f"FATAL: per-run log directory {logs_dir} already exists; refusing to reuse a "
            f"directory whose contents cannot be attributed to this run")
        sys.exit(1)
    os.makedirs(logs_dir)
    eval_cmd = [
        sys.executable, "experiments/robot/libero/run_libero_eval.py",
        "--pretrained_checkpoint", ckpt_arg,
        "--task_suite_name", SUITE,
        "--num_trials_per_task", TRIALS,
        "--seed", SEED,
        "--center_crop", "True",
        "--use_wandb", "False",
        "--num_images_in_input", num_images,
        "--use_proprio", use_proprio,
        "--local_log_dir", logs_dir,
        "--run_id_note", run_identity,
    ]
    # H2: the manifest's unnorm_key is applied VERBATIM via upstream's
    # --unnorm_key after patching check_unnorm_key to honor it (disclosed
    # modification; upstream otherwise overwrites it with the suite name and
    # applies a _no_noops fallback -- the manifest value pins the resolution).
    # Criticals 4 and 5: openvla_utils.get_vla() loads without checking loading
    # diagnostics, and rewrites the checkpoint's config.json and model code from the current
    # checkout immediately before loading -- so the digest taken above would describe a tree
    # that was never evaluated. Patch both before the evaluation subprocess starts.
    utils_py = os.path.join(OFT_DIR, "experiments/robot/openvla_utils.py")
    with open(utils_py) as _fh:
        utils_src = _fh.read()
    if "_vla_strict_load_audit" not in utils_src:
        try:
            utils_src = _patch_openvla_load_guards(utils_src)
        except RuntimeError as exc:
            log(f"FATAL: {exc}")
            sys.exit(1)
        with open(utils_py, "w") as _fh:
            _fh.write(utils_src)
        log("PATCHED openvla_utils: strict load audit installed; model-code "
            "substitution now refused")
    else:
        log("openvla_utils already patched for load guards")
    eval_py = os.path.join(OFT_DIR, "experiments/robot/libero/run_libero_eval.py")
    with open(eval_py) as _fh:
        src = _fh.read()
    try:
        src = _patch_unnorm_key(src)
    except RuntimeError as exc:
        log(f"FATAL: {exc}")
        sys.exit(1)
    with open(eval_py, "w") as _fh:
        _fh.write(src)
    log(f"PATCHED check_unnorm_key (assignment + _no_noops fallback); "
        f"applying manifest unnorm_key={unnorm_key} verbatim")
    eval_cmd += ["--unnorm_key", unnorm_key]
    # Trace boundary: run_libero_eval loads the policy onto a CUDA device, so on a CPU L3 job
    # this chunk OPENS and never closes -- read_trace reports it as the death point, which is
    # the GPU boundary this rung exists to locate.
    log(f">>> libero_eval run_libero_eval on {SUITE} (model load binds a GPU device); ckpt={CKPT}")
    log("$ " + " ".join(eval_cmd))
    # M4: the eval rollout is the longest-running work in this job and must honour the SAME
    # runtime budget every other subprocess gets via run(). It was launched with a bare
    # subprocess.run and no timeout, so a stalled rollout held paid accelerator capacity until
    # SageMaker terminated the job -- an opaque external kill rather than a diagnosable timeout.
    # A watchdog kills the whole process group (start_new_session so killpg reaches children)
    # once the deadline passes, matching the molmoact2 trainer and the Arena evaluator.
    timed_out = {"v": False}
    proc = subprocess.Popen(eval_cmd, cwd=OFT_DIR, env=env, start_new_session=True)

    def _kill_eval() -> None:
        timed_out["v"] = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    hang_timer = threading.Timer(_subprocess_timeout(), _kill_eval)
    hang_timer.start()
    try:
        returncode = proc.wait()
    finally:
        hang_timer.cancel()
    if timed_out["v"]:
        log("FATAL: eval exceeded the runtime budget and was killed")
        sys.exit(124)
    if returncode != 0:
        log(f"FATAL: eval exited {returncode}")
        sys.exit(returncode)

    # ---- Parse the evaluator's own log, identified rather than guessed (C6).
    # The directory was created empty for this invocation and the run identity is embedded
    # in the filename, so exactly one log must be present and it must carry that identity.
    # Selecting by mtime is not used at all: an unrelated log with a later timestamp must
    # not be able to win.
    try:
        chosen = _select_run_log(logs_dir, run_identity)
    except ValueError as exc:
        log(f"FATAL: {exc}")
        sys.exit(1)
    log(f"parsing evaluation log {os.path.basename(chosen)} (identity {run_identity})")
    text = open(chosen).read()

    error_episodes = re.findall(r"Episode error:", text)
    if error_episodes:
        log(f"FATAL: {len(error_episodes)} inference error(s) detected in eval log. "
            "Upstream run_episode() caught exceptions and counted them as completed "
            "episodes with success=False. This is not a legitimate evaluation result -- "
            "it means the model failed to run, not that it ran and failed the task.")
        sys.exit(1)
    total = re.findall(r"Current total success rate:\s+([0-9.]+)", text)
    episodes_seen = re.findall(r"# episodes completed so far:\s+(\d+)", text)
    per_task_rates = re.findall(r"Current task success rate:\s+([0-9.]+)", text)
    if not total or not episodes_seen:
        log("FATAL: success rate / episode count not found in eval log")
        sys.exit(1)
    success_rate = float(total[-1])
    reported_episodes = int(episodes_seen[-1])
    if len(per_task_rates) != SUITE_TASK_COUNT:
        log(f"FATAL: expected {SUITE_TASK_COUNT} per-task rates, got {len(per_task_rates)}")
        sys.exit(1)

    trials = int(TRIALS)
    task_ids = list(range(SUITE_TASK_COUNT))
    per_task = [
        {"task_id": i, "task": f"{SUITE}_{i}", "episodes": trials,
         "success_rate": float(per_task_rates[i])}
        for i in task_ids
    ]

    if _RESOLVED is None:
        log("FATAL: no source identity was constructed, so this report cannot name what it "
            "evaluated. Both the archive and the HF branch must build a ResolvedCheckpoint.")
        sys.exit(1)

    metrics = {
        "schema_version": 3,
        # checkpoint, checkpoint_revision and the EXACTLY-ONE source variant all come from the
        # single resolved object. Assembling them separately is what let a report carry a coherent
        # snapshot beside a checkpoint naming something else (I4), and put a prefixed digest into a
        # bare-hex field (I6). The no-op ternary that used to sit on "checkpoint" -- identical in
        # both branches -- is gone with it.
        **_RESOLVED.report_identity_fields(),
        "policy_type": "checkpoint",
        "model_family": "openvla",
        "checkpoint_manifest": manifest,
        "weights_digest_recomputed_by_eval": recomputed,
        "model_artifact_identity": model_artifact_identity,
        "suite": SUITE,
        "task_ids": task_ids,
        "num_trials_per_task": trials,
        "eval_seed": int(SEED),
        "train_seed": manifest["train_seed"],
        "success_rate": success_rate,
        "episodes": reported_episodes,
        "episodes_reported_by_evaluator": reported_episodes,
        "per_task": per_task,
        "provenance": {
            "recipe_repo": OFT_REPO,
            "recipe_commit": OFT_COMMIT,
            "mujoco_gl": env["MUJOCO_GL"],
            "libero_commit": LIBERO_COMMIT,
        },
    }

    # SELF-VALIDATION (preflight, not the gate: the Lambda re-validates with
    # pipeline-owned expectations)
    validate_report(
        metrics,
        expected_suite=SUITE,
        expected_trials=trials,
        expected_eval_seed=int(SEED),
        expected_task_ids=task_ids,
        family_schemas=load_family_schemas(),
        pipeline_mode=pipeline_mode,
        expected_budget_type="fixed_trials",
    )
    log("schema-v2 self-validation PASSED (shared gate validator)")

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    freeze_proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if freeze_proc.returncode != 0 or not freeze_proc.stdout.strip():
        log("FATAL: pip freeze failed -- provenance manifest cannot be empty")
        sys.exit(1)
    with open(os.path.join(OUT_DIR, "runtime_manifest.txt"), "w") as fh:
        fh.write(f"# oft_commit={OFT_COMMIT}\n# libero_commit={LIBERO_COMMIT}\n")
        fh.write(freeze_proc.stdout)

    # A6: Removed dead EVAL_METRICS_UPLOAD_PREFIX path -- RegisterModel doesn't
    # reference standalone metrics. All metrics flow through the pipeline's
    # PropertyFile mechanism (SimEval → Validate → Gate).

    with open(os.path.join(OUT_DIR, "eval_log.txt"), "w") as fh:
        fh.write(text)
    rollouts = os.path.join(OFT_DIR, "rollouts")
    if os.path.isdir(rollouts):
        import shutil
        vids = []
        for root, _dirs, files in os.walk(rollouts):
            vids += [os.path.join(root, f) for f in files if f.endswith(".mp4")]
        by_task: dict[str, str] = {}
        for v in sorted(vids):
            m = re.search(r"--task=(.+)\.mp4$", os.path.basename(v))
            by_task.setdefault(m.group(1) if m else os.path.basename(v), v)
        for v in by_task.values():
            shutil.copy(v, OUT_DIR)
        log(f"copied {len(by_task)} videos ({len(vids)} total)")

    log(f"DONE success_rate={success_rate}")


if __name__ == "__main__":
    main()
