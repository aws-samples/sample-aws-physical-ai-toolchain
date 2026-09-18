#!/usr/bin/env python3
"""Isaac Lab Arena evaluation entry point for SageMaker Training Jobs.

Runs GR00T policy evaluation using Isaac Lab Arena's policy_runner.py
in headless mode. The GR00T model server runs in-process (single container).

SageMaker contract:
  - Input: model checkpoint from FineTune step (mounted at /opt/ml/input/data/model/)
           OR downloaded from S3 via SM_CHANNEL_MODEL
  - Output: metrics.json written to /opt/ml/model/ (becomes model.tar.gz artifact)
  - Env: OMNI_KIT_ACCEPT_EULA=YES, ACCEPT_EULA=Y (set in container)

Topology: Single container runs both GR00T inference server AND Arena eval client.
This mirrors the CI pattern from isaaclab_arena_gr00t/docker/ci_gr00t_train_and_serve.sh
"""
from __future__ import annotations

import ast
import glob
import json
import os
import tempfile
import re
import shutil  # noqa: F401 -- existing tests monkeypatch module.shutil.which
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path


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

# ---------------------------------------------------------------------------
# Isaac Sim environment setup
# SageMaker's training toolkit runs this script with system python.
# We use /isaac-sim/python.sh as a subprocess for Arena commands so that
# Isaac Sim's LD_LIBRARY_PATH, PYTHONPATH, and LD_PRELOAD are correct.
# ---------------------------------------------------------------------------
ISAAC_PYTHON = "/isaac-sim/python.sh"

# ---------------------------------------------------------------------------
# SageMaker paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
INPUT_MODEL = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")

# the 6 per-run Arena eval knobs are collapsed into ONE
# EVAL_SIM_CONFIG JSON blob (was EvalEmbodimentTag/EvalTaskName/
# EvalPolicyConfigYaml/EvalArenaEmbodiment/EvalObject/EvalNumSteps). Seed the
# discrete SM_HP_*/EVAL_* keys the scattered readers below still use, so the
# collapse is behavior-preserving. setdefault: an explicitly-set discrete env
# still wins (e.g. a manual submit_simeval run). use_groot_server + arena_connector
# are intentionally NOT here -- docker_entrypoint_multi.sh routes on them before
# this Python runs, so they stay discrete env vars (registry-resolved).
_EVAL_SIM_CONFIG_TO_ENV = {
    "embodiment_tag": "SM_HP_EMBODIMENT_TAG",
    "task_name": "SM_HP_TASK_NAME",
    "policy_config_yaml": "EVAL_POLICY_CONFIG_YAML",
    "arena_embodiment": "EVAL_ARENA_EMBODIMENT",
    "object": "EVAL_OBJECT",
    # No budget key: the episode count travels as EVAL_TRIALS, not inside this blob,
    # so there is exactly one place a sample size can be set. A blob still carrying
    # the retired "num_steps" key now fails loud via the unknown-key check below,
    # which is what we want -- a stale control plane must not run silently.
}


def _seed_env_from_eval_sim_config():
    raw = os.environ.get("EVAL_SIM_CONFIG", "").strip()
    if not raw:
        return
    try:
        blob = json.loads(raw)
    except ValueError as e:  # never mask a malformed blob -- hard fail
        raise ValueError(f"EVAL_SIM_CONFIG is not valid JSON: {e!r}") from e
    if not isinstance(blob, dict):
        raise ValueError(
            f"EVAL_SIM_CONFIG must be a JSON object, got {type(blob).__name__}")
    unknown = set(blob) - set(_EVAL_SIM_CONFIG_TO_ENV)
    if unknown:  # fail loud on a typo'd/unknown key rather than silently ignore it
        raise ValueError(
            f"EVAL_SIM_CONFIG has unknown key(s) {sorted(unknown)}; "
            f"expected {sorted(_EVAL_SIM_CONFIG_TO_ENV)}")
    for key, env_key in _EVAL_SIM_CONFIG_TO_ENV.items():
        if blob.get(key) is not None:
            # Blob is AUTHORITATIVE (run_arena is the source of truth): overwrite
            # so a baked image ENV default can never shadow the requested value.
            # submit_simeval (discrete env, no blob) is unaffected -- the blob is
            # absent there, so this shim no-ops and its discrete env stands.
            os.environ[env_key] = str(blob[key])


_seed_env_from_eval_sim_config()

# --- Episode budget -------------------------------------------------------------
# EVAL_TRIALS (k) is the number of COMPLETE episodes to evaluate, passed straight
# through to policy_runner as `--num_episodes k`. It used to be pinned to 1 while a
# separate step budget was transported independently; nothing reconciled the two, so
# a 280-step budget against a 500-step max_episode_length silently produced ZERO
# episodes (an episode completes only on termination or truncation).
#
# Episode mode removes that arithmetic: Arena stops after k boundaries, so the
# evaluated sample size is what was requested. The step budget is no longer part of
# the registrable path -- for a step-budgeted diagnostic, invoke policy_runner
# directly rather than reintroducing a second, unreconciled budget knob here.
EVAL_TRIALS = int(os.environ.get("EVAL_TRIALS", "1"))
if EVAL_TRIALS < 1:
    raise ValueError(
        f"EVAL_TRIALS={EVAL_TRIALS} but at least one complete episode must be "
        "evaluated. A count of zero cannot produce a measurement.")
EVAL_SEED = int(os.environ.get("EVAL_SEED", os.environ.get("SM_HP_EVAL_SEED", "100")))
_DECLARED_POLICY_CONFIG = os.environ.get("EVAL_POLICY_CONFIG_YAML", "")
# I5: the patched config the runner is handed, recorded at the launch site so the report
# can name the bytes that were executed rather than the template they came from.
_EXECUTED_POLICY_CONFIG = ""
# Normalize the GR00T version ONCE, here, and use this constant everywhere.
#
# It used to be re-read raw at each site while only the launcher branch applied
# .strip().lower(). With EVAL_GR00T_VERSION="N16" that disagreed with itself: the early
# embodiment-tag guard compared raw ("N16" != "n16") and was skipped, the launcher
# normalized and started the N1.6 server (which hardcodes --embodiment-tag GR1), and the
# reporter also compared raw and therefore recorded the REQUESTED tag instead of GR1. The
# report then claimed an embodiment the server never served -- a provenance lie that the
# digest chain cannot detect. Validating here also moves an unsupported value from a
# mid-run failure to an immediate one.
EVAL_GR00T_VERSION = os.environ.get("EVAL_GR00T_VERSION", "n17").strip().lower()
if EVAL_GR00T_VERSION not in ("n16", "n17"):
    raise RuntimeError(
        f"unsupported EVAL_GR00T_VERSION: "
        f"{os.environ.get('EVAL_GR00T_VERSION')!r} (normalized to "
        f"{EVAL_GR00T_VERSION!r}); expected 'n16' or 'n17'")
# Exactly one environment. Arena's rollout counts EVERY environment that ends in the
# same vectorized step, so with num_envs=2 and k=3 it can complete four episodes --
# the evaluated sample would then exceed what was requested and per-task equality
# would be unenforceable. Sequential episodes in one runner invocation keep k exact.
NUM_ENVS = int(os.environ.get("SM_HP_NUM_ENVS", "1"))
if NUM_ENVS != 1:
    raise ValueError(
        f"SM_HP_NUM_ENVS={NUM_ENVS} but Arena evaluation requires exactly one "
        "environment: the rollout completes every environment that terminates in the "
        "same vectorized step, so k episodes cannot be requested exactly with a "
        "vectorized environment. Raise EVAL_TRIALS for a larger sample instead.")
# No default: the task must come from the suite manifest (via EVAL_SIM_CONFIG) or
# an explicit SM_HP_TASK_NAME. A cross-cell literal here (this was
# "cube_goal_pose") means an absent or task_name-less blob silently rolls out the
# WRONG task for any suite -- the exact pre-suite-manifest bug, reintroduced inside
# the image where no launcher check can see it.
# NOTE: baked into the connector image -- needs an image rebuild to take effect.
TASK_NAME = os.environ.get("SM_HP_TASK_NAME", "").strip()
if not TASK_NAME:
    raise RuntimeError(
        "SM_HP_TASK_NAME is unset. The Arena task is declared by the suite "
        "manifest (config/suites/<suite>.yaml arena.task) and delivered via "
        "EVAL_SIM_CONFIG; refusing to fall back to a hardcoded task, which would "
        "roll out a task the checkpoint was not trained for and still report a "
        "success_rate.")
SERVER_PORT = int(os.environ.get("SM_HP_SERVER_PORT", "5555"))

# Isaac Lab Arena workspace
ARENA_WORKSPACE = "/workspace"

# ---------------------------------------------------------------------------
# Isolated GR00T server venvs (baked at image build time -- R1)
#
# The connector Dockerfile clones Isaac-GR00T at the pinned commits, strips the
# training-only deepspeed dep, and runs `uv sync --python 3.12` into dedicated
# venvs under /opt/gr00t-{n17,n16}/. Runtime REQUIRES these venvs and fails
# immediately if they are absent (never installs at runtime).
# ---------------------------------------------------------------------------
GR00T_COMMIT = "376ba890cff8c9de64d71d982772a9c36185fdd7"  # == steps/gr00t/defaults.json repo_commit
GR00T_DIR = os.environ.get("GR00T_N17_DIR", "/opt/gr00t-n17")
GR00T_VENV = f"{GR00T_DIR}/.venv/bin/python"
GR00T_SERVER_LOG = "/tmp/gr00t_server.log"

GR00T_N16_COMMIT = "5dc80c4afd726b34faad1d8f7e007a13b34e4c88"  # github n1.6.1-release
GR00T_N16_DIR = os.environ.get("GR00T_N16_DIR", "/opt/gr00t-n16")
GR00T_N16_VENV = f"{GR00T_N16_DIR}/.venv/bin/python"
# Component-owned wrapper that binds the evaluation seed to the SERVER process's RNG
# before delegating to the pinned server. Baked into the image beside this entrypoint
# (the container entrypoint ignores SageMaker's source_dir), so it is addressed by its
# absolute in-image path rather than relative to this file.
SEEDED_SERVER_ENTRY = os.environ.get(
    "GR00T_SEEDED_SERVER_ENTRY", "/workspace/gr00t_seeded_server.py")
GR00T_N16_SERVER_LOG = "/tmp/gr00t_server_n16.log"
N16_POSCTRL_CKPT_REPO = os.environ.get(
    "N16_POSCTRL_CKPT_REPO", "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task")
# The REQUESTED revision. Defaulted to "main", which is a MOVING reference: the report published it
# as resolved_commit, so the attestation claimed to pin bytes while naming a branch that can point
# somewhere else tomorrow. There is no safe default -- a diagnostic that cannot say which weights it
# scored is not evidence -- so absence is fatal at the point of use rather than silently "main".
N16_POSCTRL_CKPT_REV = os.environ.get("N16_POSCTRL_CKPT_REV", "")
N16_POSCTRL_CKPT_DIR = "/tmp/gn16_posctrl_ckpt"
#: Where the child records the commit the Hub actually resolved. A file, not stdout, because stdout is
#: interleaved with upstream logging -- the same idiom the backbone precache uses.
N16_POSCTRL_RESOLVED_FILE = os.path.join(N16_POSCTRL_CKPT_DIR, ".resolved_commit")
# Isaac Sim sets these to its own py3.10 site-packages; the py3.12 gr00t venv must
# NOT inherit them or it loads the wrong stdlib/site-packages -> ABI crash.
_ISAAC_ENV_KEYS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")


def log(msg: str):
    print(f"[isaac-arena-eval] {msg}", flush=True)

# The source identity of what this invocation loaded, CONSTRUCTED once by the resolver. Two
# seedable dicts whose emptiness DECIDED the mode is what let a fixture render one branch
# unexecutable. _ARCHIVE_MEASUREMENT holds the pre-extraction measurement, which can only be
# taken while the archive is still being read.
_RESOLVED = None
_ARCHIVE_MEASUREMENT = None
# The EXTERNAL backbone this invocation actually loaded. Starts EMPTY on purpose.
#
# This was initialised to {"repo_id": "nvidia/Cosmos-Reason2-2B"} unconditionally, while the N1.6 path
# calls ensure_gr00t_venv_n16() and never precache_cosmos() at all. So every N1.6 report -- and every
# positive-control report, which is always N1.6 -- published a backbone identity naming a model the run
# had never loaded. The field is allowlisted-but-never-read by the validator and copied verbatim into
# the attestation, so nothing caught it: the receipt simply asserted the wrong backbone.
#
# A report with NO backbone identity states a gap. A report naming the wrong one states a falsehood,
# and a falsehood in an attestation is worse than a gap, because a consumer cannot tell it is wrong.
_BACKBONE_IDENTITY: dict = {}


def extract_checkpoint(input_path: str, dest: str) -> str:
    """Measure and extract model.tar.gz to a checkpoint directory.

    The archive is MEASURED BEFORE EXTRACTION. A tree digest cannot show that SimEval and
    Validate read the same bytes: it excludes manifests, logs and hidden paths, so two
    different archives can share one. HeadObject cannot either -- it reports the object
    current when HEAD runs, not the object this job already downloaded.
    """
    from capped_reader import DEFAULT_MAX_ARCHIVE_BYTES, capped_tar_open  # noqa: E402
    from digest import measure_archive

    log(f">>> checkpoint_extract from {input_path}")
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    tarball = Path(input_path) / "model.tar.gz"
    if tarball.exists():
        sha256, size_bytes = measure_archive(str(tarball))
        global _ARCHIVE_MEASUREMENT
        _ARCHIVE_MEASUREMENT = (str(tarball), sha256, size_bytes)
        log(f"measured source archive: sha256={sha256} size={size_bytes}")
        log(f"Extracting {tarball} to {dest}")
        with capped_tar_open(tarball, "r:gz",
                                     max_bytes=DEFAULT_MAX_ARCHIVE_BYTES) as tar:
            # Tar path-traversal / link-escape guard (matches the LIBERO
            # eval idiom): prefer the stdlib "data" filter (>=3.12); on older
            # runtimes fall back to explicit per-member validation (in-bounds +
            # regular-file/dir only). Fail loud on any unsafe member.
            _bound_archive(tar)
            try:
                tar.extractall(dest, filter="data")
            except TypeError:
                base = os.path.abspath(dest)
                for m in tar.getmembers():
                    p = os.path.abspath(os.path.join(base, m.name))
                    if os.path.commonpath([base, p]) != base or not (
                            m.isreg() or m.isdir()):
                        raise RuntimeError(
                            f"unsafe tar member {m.name!r} in {tarball} "
                            f"(path-traversal/link-escape guard)")
                tar.extractall(dest)
    else:
        # An already-extracted channel leaves no archive to measure, so the run could not state
        # which bytes it evaluated. That was a silent fallback; it is now fatal, because the
        # alternative is emitting weaker evidence under a field that claims to be a measurement.
        log(f"FATAL: no model.tar.gz in {input_path}. The archive is what S3 versions and "
            f"ETags refer to and what Validate independently measures, so without it this run "
            f"cannot show that it evaluated the same bytes Validate promotes.")
        sys.exit(1)

    if _ARCHIVE_MEASUREMENT is not None:
        # Built over the tree inference reads, from the measurement taken before extraction.
        # Imported function-locally like every other shared module here: this file is delivered to
        # a bare container and runs from /workspace, so it cannot import vla_pipeline.common.
        from source_identity import from_archive as si_from_archive
        global _RESOLVED
        _archive_path, _sha, _size = _ARCHIVE_MEASUREMENT
        _RESOLVED = si_from_archive(load_root=dest, archive_path=_archive_path,
                                    checkpoint=INPUT_MODEL, sha256=_sha, size_bytes=_size)
    _n_files = sum(len(f) for _, _, f in os.walk(dest))
    log(f"<<< checkpoint_extract OK: {_n_files} files under {dest}, "
        f"sha256={_ARCHIVE_MEASUREMENT[1] if _ARCHIVE_MEASUREMENT else 'unmeasured'}")
    return dest


def resolve_hf_token() -> str:
    """Resolve the HF token (env or Secrets Manager) for the gated Cosmos backbone.

    Mirrors the proven LIBERO GR00T eval (steps/gr00t/eval_entry.py). GR00T N1.7
    loads the gated nvidia/Cosmos-Reason2-2B backbone, which 401s without a token.
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        secret_name = os.environ.get("HF_SECRET_NAME", "")
        if secret_name:
            import boto3 as _boto3
            region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            client = _boto3.client("secretsmanager", region_name=region)
            hf_token = client.get_secret_value(SecretId=secret_name)["SecretString"].strip()
            os.environ["HF_TOKEN"] = hf_token
            log(f"Resolved HF token from secret '{secret_name}'")
    if not hf_token:
        log("WARNING: No HF_TOKEN — gated Cosmos-Reason2-2B access will 401")
    return hf_token


# Evidence path per served checkpoint, recorded by whichever launch actually started
# the server. The reporter must not recompute it: it does not know the port used.
_LAUNCH_EVIDENCE: dict = {}


def _is_positive_control(results: dict | None = None) -> bool:
    """The ONE reader of positive-control mode. R4#I1.

    Four sites decided this independently, and two of them disagreed in a way that made the mode
    unusable: the publication guard tested ``results["policy_type"] == "positive_control"`` while
    the branch that handles the mode tested the environment flag. run_arena_eval overwrites
    policy_type to "checkpoint" for any remotely served policy -- which the positive control always
    is -- so the guard rejected every positive-control run before its own branch could label it.

    The environment flag is authoritative because no caller rewrites it. The policy_type label is
    still honoured so a report already built by the branch below reads as positive control.
    """
    if os.environ.get("EVAL_POSCTRL_N16", "").lower() == "true":
        return True
    return bool(results) and results.get("policy_type") == "positive_control"


def _server_evidence_path(checkpoint_path: str, port: int) -> str:
    """A distinct evidence path per server launch.

    I10: a single reusable /tmp filename could be inherited across launches within one job.
    The dose-curve path starts a server per checkpoint, so the LAST server's evidence would
    otherwise be read as every checkpoint's -- and a launch that wrote nothing would silently
    inherit a previous one's proof.

    Derived from the checkpoint and port so the reader can compute the same path for the
    checkpoint it is reporting, rather than being handed one out of band.
    """
    import hashlib

    token = hashlib.sha256(
        f"{os.path.realpath(checkpoint_path)}|{port}".encode("utf-8")).hexdigest()[:16]
    override = os.environ.get("GR00T_SERVER_SEED_EVIDENCE_DIR", "/tmp")
    return os.path.join(override, f"gr00t_server_seed_evidence_{token}.json")


def _build_server_env(version: str) -> dict:
    """Env for the isolated gr00t venv (py3.12). Inherit os.environ but STRIP Isaac
    Sim's PYTHONPATH/PYTHONHOME and filter /isaac-sim//omni from LD_LIBRARY_PATH,
    else the venv loads Isaac Sim's py3.10 site-packages/libs -> ABI crash."""
    env = os.environ.copy()
    # The RESOLVED seed, forwarded explicitly. The copy above only carries EVAL_SEED when the
    # caller happened to set that exact name -- but this evaluator also accepts SM_HP_EVAL_SEED
    # and falls back to a default, and the server requires EVAL_SEED to be present. So a launcher
    # using the alias, or relying on the default, produced a server that refused to start while
    # the evaluator had a perfectly good seed in hand.
    env["EVAL_SEED"] = str(EVAL_SEED)
    for k in _ISAAC_ENV_KEYS:
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    ld = env.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(os.pathsep) if p and "/isaac-sim/" not in p and "/omni/" not in p]
    # cuDNN library precedence for the N1.7 policy server (same root cause + fix as
    # train_entry.py + the LIBERO eval): prepend the gr00t venv's nvidia/*/lib so its
    # bundled nvidia-cudnn-cu12 (>=9.5, has cudnnGetLibConfig) wins over any older
    # system cuDNN on the loader path. The GR00T-N1.7 policy invokes the cuDNN-graph
    # API at model load; without this the server aborts ("libcudnn_graph.so.9:
    # undefined symbol: cudnnGetLibConfig"). Applied ONLY on the n17 path -- the n16
    # native-GR1 server (EVAL_GR00T_VERSION=n16) never calls the cuDNN-graph API, so
    # it stays byte-for-byte inert.
    _added_nvidia = False
    if version == "n17":
        _nvidia_libs = sorted(glob.glob(os.path.join(
            GR00T_DIR, ".venv", "lib", "python*", "site-packages", "nvidia", "*", "lib")))
        if not _nvidia_libs:
            log("[cudnn-fix] FATAL: no venv nvidia/*/lib dirs under "
                f"{GR00T_DIR}/.venv -- the N1.7 Arena server needs cuDNN>=9.3 "
                "(cudnnGetLibConfig); the uv venv is malformed, aborting")
            sys.exit(1)
        parts = _nvidia_libs + parts
        _added_nvidia = True
        log(f"[cudnn-fix] prepended {len(_nvidia_libs)} venv nvidia lib dir(s) to "
            "LD_LIBRARY_PATH (venv cuDNN precedence over system cuDNN 9.1)")
    # Re-set LD_LIBRARY_PATH whenever the original was non-empty (to STRIP the
    # isaac-sim/omni entries -- the primary purpose of this function) or when we
    # prepended venv libs. Leave it untouched when there was nothing, so we never
    # inject an empty "" entry (= cwd on the loader search path).
    if ld or _added_nvidia:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    tok = env.get("HF_TOKEN", "")
    if tok:
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    return env


def ensure_gr00t_venv() -> str:
    """Require the N1.7 GR00T venv baked into the connector image at build time.

    The Dockerfile clones Isaac-GR00T@GR00T_COMMIT, strips deepspeed, and runs
    `uv sync --python 3.12` into GR00T_DIR/.venv/. This function validates the
    venv exists and fails immediately if it does not -- it never installs anything
    at runtime."""
    if not Path(GR00T_VENV).exists():
        raise RuntimeError(
            f"FATAL: baked N1.7 GR00T venv not found at {GR00T_VENV}. "
            f"The connector image must be rebuilt with the R1 Dockerfile that "
            f"bakes GR00T venvs at build time. Runtime dependency installation "
            f"is no longer supported.")
    log(f"N1.7 GR00T venv present: {GR00T_VENV}")
    return GR00T_VENV


def precache_cosmos(hf_token: str):
    """Pre-cache the gated nvidia/Cosmos-Reason2-2B backbone from the GR00T VENV.

    GR00T's internal AutoModel.from_pretrained('nvidia/Cosmos-Reason2-2B') does NOT
    forward HF_TOKEN to nested downloads (LIBERO 401 root cause). Pre-caching under
    GR00T_VENV (the same interpreter that runs the GR00T server) puts the weights on
    disk so server startup needs no HTTP auth. Fatal on failure (no silent skip).
    """
    if not hf_token:
        raise RuntimeError("cannot pre-cache Cosmos-Reason2-2B without an HF token")
    log(">>> backbone_precache nvidia/Cosmos-Reason2-2B (gated, gr00t venv)")
    script = (
        "import os, sys\n"
        "token = os.environ.get('HF_TOKEN', '')\n"
        "if not token:\n"
        "    print('FATAL: HF_TOKEN empty', file=sys.stderr); sys.exit(1)\n"
        "from huggingface_hub import login, snapshot_download\n"
        "login(token=token)\n"
        # cycle-16 C1: this downloaded the VLM backbone with NO revision, and the processor
        # guard checks only the repository NAME. A change to that external snapshot can
        # change preprocessing while the checkpoint archive, its digest, the evaluator image
        # and the sourcedir all stay identical -- and nothing in the receipt could tell the
        # two runs apart.
        #
        # snapshot_download caches to .../snapshots/<commit>/, so the RESOLVED commit is
        # recoverable from the returned path. Recorded rather than pinned to a hardcoded
        # hash: a pin invented here could not be verified, and recording what actually
        # arrived is the identity the evidence chain was missing.
        "_p = snapshot_download('nvidia/Cosmos-Reason2-2B', token=token)\n"
        "_c = os.path.basename(os.path.realpath(_p))\n"
        "import json as _j\n"
        "open(os.environ['BACKBONE_COMMIT_FILE'], 'w').write("
        "    _j.dumps({'commit': _c, 'snapshot_path': os.path.realpath(_p)}))\n"
        "print(f'backbone resolved commit: {_c} at {_p}')\n"
    )
    env = _build_server_env("n17")
    env["HF_TOKEN"] = hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    # cycle-16 C1: the resolved backbone commit is the external input that was missing from
    # the evidence. Written by the child to a file rather than scraped from stdout, which is
    # interleaved with upstream logging.
    _bb_commit_file = os.path.join(tempfile.mkdtemp(prefix="backbone-"), "commit")
    env["BACKBONE_COMMIT_FILE"] = _bb_commit_file
    rc = subprocess.run([GR00T_VENV, "-c", script], env=env).returncode
    if os.path.isfile(_bb_commit_file):
        # repo_id is set HERE, by the path that actually downloaded it, rather than assumed at
        # module import by every path including the ones that never touch Cosmos.
        _BACKBONE_IDENTITY["repo_id"] = "nvidia/Cosmos-Reason2-2B"
        # Written as JSON by the child. Tolerates the older bare-commit form so a
        # partially-updated image cannot turn a working precache into a crash.
        _raw = open(_bb_commit_file).read().strip()
        try:
            _bb = json.loads(_raw)
        except ValueError:
            _bb = {"commit": _raw, "snapshot_path": ""}
        _BACKBONE_IDENTITY["resolved_commit"] = _bb.get("commit", "")
        _BACKBONE_SNAPSHOT_PATH = _bb.get("snapshot_path", "")
        log(f"backbone identity: nvidia/Cosmos-Reason2-2B @ "
            f"{_BACKBONE_IDENTITY['resolved_commit']}")
    else:
        # Absent is recorded as absent. A guessed value here would be worse than the gap.
        log("WARNING: backbone commit file not written; identity unavailable")
    if rc != 0:
        raise RuntimeError("FATAL: failed to pre-cache nvidia/Cosmos-Reason2-2B")
    # Measure ONLY the snapshot this download returned. Totalling HF_HOME was wrong: that
    # directory holds every repository the job ever fetched plus the blob store each snapshot
    # symlinks into, so unrelated downloads inflated the figure and a short backbone could read
    # as a full one -- defeating the reason the size is reported at all, which is to make a
    # throttled partial download visible.
    _bb_root = locals().get("_BACKBONE_SNAPSHOT_PATH") or ""
    _bb_bytes = 0
    _bb_files = 0
    if _bb_root and os.path.isdir(_bb_root):
        for _root, _dirs, _files in os.walk(_bb_root):
            for _f in _files:
                _fp = os.path.join(_root, _f)
                try:
                    # follow the symlink into the blob store: the snapshot tree is links, so
                    # lstat would report a few bytes per file and every download would look empty
                    _bb_bytes += os.stat(_fp).st_size
                    _bb_files += 1
                except OSError:
                    pass
        log(f"<<< backbone_precache OK: {_bb_files} file(s), {_bb_bytes / 1e9:.2f} GB in the "
            f"snapshot at {_bb_root}, commit="
            f"{_BACKBONE_IDENTITY.get('resolved_commit') or 'unavailable'}")
    else:
        # No path means the child did not report one. Say that rather than substituting a
        # number measured somewhere else, which would be a figure about the wrong directory.
        log(f"<<< backbone_precache OK: snapshot size UNMEASURED (the child reported no path), "
            f"commit={_BACKBONE_IDENTITY.get('resolved_commit') or 'unavailable'}")


def start_groot_server(checkpoint_path: str, port: int) -> subprocess.Popen:
    """Start the GR00T policy server from the ISOLATED gr00t venv (NOT Isaac Sim
    python). Uses --model-path + --embodiment-tag + --use-sim-policy-wrapper (proven
    LIBERO invocation). Env is stripped of Isaac Sim PYTHONPATH/PYTHONHOME + isaac-sim
    LD_LIBRARY_PATH via _build_server_env(). Logs to a file (not PIPE) to avoid a
    buffer-fill deadlock on the verbose server."""
    embodiment_tag = os.environ.get("SM_HP_EMBODIMENT_TAG",
                                    os.environ.get("EMBODIMENT_TAG", "new_embodiment"))
    # ROOT CAUSE FIX: --use-sim-policy-wrapper is a
    # LIBERO-era adapter at gr00t@376ba890 that expects UNPREFIXED obs keys
    # (ego_view, left_arm) and rebuilds them. The Arena client (Gr00tRemoteClosedloopPolicy)
    # sends ALREADY-PREFIXED keys (video.ego_view, state.left_arm) -> the wrapper never
    # finds "ego_view" -> model validation fails "Video key 'video.ego_view' must be in
    # observation". Fix: drop the wrapper (server reads prefixed keys directly, as NVIDIA's
    # Arena recipe does).
    #
    # CORRECTION (I7): --modality-config-path does NOT configure checkpoint inference.
    # This comment previously claimed it made "the server's modality transform match the
    # checkpoint/client". It does not: both pinned servers obtain the modality
    # configuration from the CHECKPOINT'S OWN PROCESSOR, and N1.7 handles
    # --modality-config-path only in its replay-policy branch, not the checkpoint branch.
    # The flag is therefore INERT on this path. It is still passed so the launch stays
    # byte-identical to the validated one, but nothing about the observation contract
    # depends on it -- do not debug modality mismatches by changing it.
    #
    # This also invalidates the porting plan that used to sit here. It said the suite
    # manifest's family_overrides.*.modality_config "does NOT reach here" and prescribed
    # plumbing it through EVAL_SIM_CONFIG plus COPYing all three configs. That would not
    # work: since the checkpoint processor supplies the modality configuration, no value
    # passed on this flag can override it. Changing the consumed modality contract means
    # changing the checkpoint's processor configuration, not this argument.
    modality_cfg = os.environ.get("MODALITY_CONFIG_PATH",
                                  "/workspace/arena_gr1_data_config.py")
    cmd = [
        # Run the pinned server THROUGH the component's seeding wrapper. EVAL_SEED
        # reached this process's environment but neither pinned server read it, so the
        # policy's inference noise (torch.randn in the action head) was unbound while
        # Arena's client-side seeding made the scene sequence look reproducible.
        GR00T_VENV, SEEDED_SERVER_ENTRY,
        "--model-path", checkpoint_path,
        "--embodiment-tag", embodiment_tag,
        "--modality-config-path", modality_cfg,
        "--port", str(port),
    ]
    # A real run died here: an N1.6 checkpoint, the n17 server, and a 300-second timeout before
    # transformers reported an architecture it did not recognise. Both facts were knowable first --
    # the checkpoint states its architecture in config.json and this function knows which venv it is
    # about to launch. Checked in BOTH launchers; fixing only this one would leave the n16 path with
    # the identical defect.
    # Imported HERE, not at module scope: this file loads its flat /workspace modules lazily
    # inside functions (as `from digest import measure_archive` does) because they are only on
    # sys.path inside the baked image. A module-level import breaks test collection.
    from checkpoint_compat import require_loadable
    require_loadable(checkpoint_path, "n17", log=log)
    server_env = _build_server_env("n17")
    # I10: a distinct evidence file per launch, cleared first so a launch that
    # writes nothing cannot inherit a previous server's proof.
    _evidence_path = _server_evidence_path(checkpoint_path, port)
    # I3: record the path THIS launch used. Dose points run on SERVER_PORT + idx, so
    # recomputing the path in write_metrics from SERVER_PORT looked up a file that
    # belonged to a different launch, or none at all.
    _LAUNCH_EVIDENCE[os.path.realpath(checkpoint_path)] = _evidence_path
    if os.path.exists(_evidence_path):
        os.remove(_evidence_path)
    server_env["GR00T_SERVER_SEED_EVIDENCE"] = _evidence_path
    log(f">>> policy_server launching (isolated venv): {' '.join(cmd)}")
    log(f"  LD_LIBRARY_PATH(head)={server_env.get('LD_LIBRARY_PATH', '')[:160]}")
    log_file = open(GR00T_SERVER_LOG, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=server_env,
    )
    proc._log_file = log_file
    log(f"GR00T server PID={proc.pid}, log={GR00T_SERVER_LOG}")
    return proc


def verify_server_alive(proc) -> None:
    """Require that the server this process launched is STILL RUNNING after readiness passed.

    Readiness proves something is listening. It does not prove the child is that something, and
    the full version of that check -- matching the listening socket's inode against the child's
    open descriptors -- needs another forty lines for a state the deployment does not reach,
    because each job gets its own network namespace and no other server is in it.

    So this checks the part that IS reachable and costs three lines: a child that has already
    exited cannot be serving the rollouts, and continuing would attribute a score to a process
    that is gone. If jobs ever share a namespace, the stronger check becomes worth its size --
    that reasoning is recorded rather than implemented.
    """
    if proc.poll() is not None:
        raise RuntimeError(
            f"policy server exited (rc={proc.returncode}) even though readiness passed. Something "
            f"else is answering on the port; refusing to attribute a score to a checkpoint this "
            f"process is no longer serving.")


def wait_for_server(host: str, port: int, timeout: float = 300.0) -> bool:
    """Wait for the GR00T server to become ready."""
    log(f"Waiting for GR00T server at {host}:{port} (timeout={timeout}s)")

    # Use the Arena utility if available
    wait_script = Path(ARENA_WORKSPACE) / "isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py"
    if wait_script.exists():
        result = subprocess.run(
            [ISAAC_PYTHON, str(wait_script),
             "--host", host, "--port", str(port),
             "--timeout-sec", str(timeout),
             "--poll-interval-sec", "10"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log(f"<<< policy_server OK: ready at {host}:{port} (readiness probe)")
            return True
        log(f"<<< policy_server FAIL: readiness probe rejected {host}:{port}: {result.stderr[-500:]}")
        return False

    # Fallback: simple TCP poll
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                log(f"<<< policy_server OK: ready at {host}:{port} (TCP fallback)")
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(10)
    log(f"<<< policy_server FAIL: {host}:{port} never accepted a connection within {timeout}s")
    return False


def _patch_policy_config_model_path(cfg_path: str, model_path: str) -> str:
    """Point the Arena closed-loop config's `model_path` at the served checkpoint.

    Arena's stock gr1_manip closed-loop config bakes a PLACEHOLDER model_path (an
    example tutorial path absent from our container -> Gr00tClosedloopPolicyConfig
    __post_init__ AssertionError "model_path does not exist"). Even for the REMOTE
    policy, the client loads model METADATA (experiment_cfg/modality/norm stats)
    locally from model_path while inference goes to the server, so it must point at
    the SAME checkpoint the n16 server serves (--model-path checkpoint_path). The
    hand-built v11 base carried a corrected config; since we build the base from
    STOCK Arena, rewrite the field here and pass a patched copy. Dependency-free
    line rewrite (system python3 may lack pyyaml)."""
    import re as _re
    import tempfile as _tf
    try:
        with open(cfg_path) as f:
            lines = f.readlines()
    except Exception as e:  # noqa: BLE001 -- fail loud: the placeholder WILL fail in Arena
        raise RuntimeError(
            f"cannot read policy config {cfg_path!r} to patch model_path ({e!r}); the "
            "stock Arena config carries a placeholder that fails at runtime -- refusing "
            "to proceed with an unpatched config.") from e
    out, n_match = [], 0
    for ln in lines:
        m = _re.match(r"^(\s*)model_path\s*:", ln)
        if m:
            n_match += 1
            if n_match == 1:
                out.append(f"{m.group(1)}model_path: {model_path}\n")  # preserve indentation
                continue
        out.append(ln)
    if n_match == 0:
        raise RuntimeError(
            f"policy config {cfg_path!r} has no 'model_path:' key to patch -- unexpected "
            "Arena closed-loop config schema; refusing to append a possibly-ignored key.")
    if n_match > 1:
        # Patching only the first used to be a WARNING. But which duplicate the YAML
        # loader honours is not something this line-rewriter can know: if a later key
        # wins, the served checkpoint is the untouched PLACEHOLDER and the evaluation
        # silently measures a different model than the one being registered. An
        # ambiguous config is a hard error.
        raise RuntimeError(
            f"policy config {cfg_path!r} declares 'model_path' {n_match} times. Which "
            f"one the YAML loader honours cannot be determined here, so the evaluated "
            f"checkpoint would be ambiguous -- and if a later duplicate wins it is the "
            f"unpatched placeholder. Refusing to proceed; de-duplicate the config.")
    fd, dst = _tf.mkstemp(prefix="policy_config_patched_", suffix=".yaml")  # unique per call
    os.close(fd)
    with open(dst, "w") as f:
        f.writelines(out)
    log(f"Patched policy config model_path -> {model_path} (wrote {dst})")
    global _EXECUTED_POLICY_CONFIG
    _EXECUTED_POLICY_CONFIG = dst
    return dst


def required_policy_config() -> str:
    path = os.environ.get("EVAL_POLICY_CONFIG_YAML", "").strip()
    if not path or path.upper() == "AUTO":
        raise ValueError("EVAL_POLICY_CONFIG_YAML must name an explicit policy configuration")
    try:
        content = Path(path).read_text()
    except OSError as exc:
        raise ValueError(f"EVAL_POLICY_CONFIG_YAML cannot be read: {path!r}") from exc
    if not content.strip():
        raise ValueError("EVAL_POLICY_CONFIG_YAML is empty")
    return path


def requested_task_ids() -> list[int]:
    raw = os.environ.get("EVAL_TASK_IDS", "all").strip()
    if raw == "all":
        return [0]
    try:
        task_ids = json.loads(raw)
    except ValueError as exc:
        raise ValueError("EVAL_TASK_IDS must be 'all' or [0]") from exc
    if (not isinstance(task_ids, list) or len(task_ids) != 1
            or type(task_ids[0]) is not int or task_ids[0] != 0):
        raise ValueError("EVAL_TASK_IDS must be 'all' or [0]; Arena runs one task per invocation")
    return task_ids


def run_arena_eval(
    task_name: str,
    num_envs: int,
    num_episodes: int,
    policy_type: str = "gr00t_remote",
    server_host: str = "localhost",
    server_port: int = 5555,
    checkpoint_path: str = None,
) -> dict:
    """Run Isaac Lab Arena evaluation and return results.

    `num_episodes` has NO default: a budget is a required, explicit input. The
    retired step budget defaulted to 280 in three separate places, which is how a
    value below the environment's 500-step episode length reached production.
    """
    requested_task_ids()
    is_remote = policy_type == "gr00t_remote"
    policy_cfg_yaml = None
    if is_remote:
        policy_type = (
            "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy."
            "Gr00tRemoteClosedloopPolicy"
        )
        policy_cfg_yaml = required_policy_config()
        # Point the (stock Arena) closed-loop config's placeholder model_path at the
        # checkpoint THIS server is serving -- the remote client loads model metadata
        # (experiment_cfg/modality/norm) locally from model_path even though inference is
        # remote. Applied here so EVERY remote path gets it (n16 pipeline, n17, posctrl,
        # and each dose-curve checkpoint), each with its own patched file.
        if checkpoint_path:
            policy_cfg_yaml = _patch_policy_config_model_path(policy_cfg_yaml, checkpoint_path)
    # policy_runner.py uses argparse.parse_intermixed_args() with `task` as a
    # choices-positional followed by further positionals. Required optionals
    # (esp. --policy_config_yaml_path) MUST precede the positional task, else
    # they are mis-bound and argparse reports them "required" even when passed
    # (a prior SimEval failed when the positional task preceded the optionals).
    # Correct order (matches Arena's contract): all main optionals -> positional
    # task -> per-task subparser flags (--embodiment) last.
    cmd = [
        ISAAC_PYTHON,
        f"{ARENA_WORKSPACE}/isaaclab_arena/evaluation/policy_runner.py",
        "--policy_type", policy_type,
        "--num_episodes", str(num_episodes),
        "--num_envs", str(num_envs),
        "--seed", str(EVAL_SEED),
    ]

    # Remote GR00T-protocol policy needs the server connection args + (policy-config requirement)
    # the policy config YAML + cameras enabled (GR00T N1.7 requires camera obs).
    if is_remote:
        cmd.extend(["--remote_host", server_host, "--remote_port", str(server_port)])
        # policy_cfg_yaml is guaranteed non-empty above (fail-loud on AUTO/unset).
        cmd.extend(["--policy_config_yaml_path", policy_cfg_yaml])
        cmd.append("--enable_cameras")

    # Positional task AFTER all main optionals (parse_intermixed_args ordering).
    cmd.append(task_name)

    # Scene/embodiment flags apply to ALL policy types (remote gr00t AND the
    # zero_action negative control) -- they set up the TASK SCENE, not the policy.
    # Hoisted out of the is_remote guard so the zero_action control runs on the SAME
    # GR1 scene as the real eval (else it would omit --embodiment gr1_joint). Env-gated
    # with NONE sentinels, so a no-op when unset (LIBERO / single-embodiment tasks).
    _arena_obj = os.environ.get("EVAL_OBJECT", "").strip()
    if _arena_obj.upper() == "NONE":
        _arena_obj = ""
    if _arena_obj:
        cmd.extend(["--object", _arena_obj])
        log(f"Arena object: --object {_arena_obj}")

    _arena_emb = os.environ.get("EVAL_ARENA_EMBODIMENT", "").strip()
    if _arena_emb.upper() == "NONE":
        _arena_emb = ""
    if _arena_emb:
        cmd.extend(["--embodiment", _arena_emb])
        log(f"Arena embodiment: --embodiment {_arena_emb}")

    log(f">>> arena_eval {task_name}, {num_episodes} episodes: {' '.join(cmd)}")
    # Stream policy_runner output LIVE (merged stdout+stderr, flushed per line) so a
    # hang or slow rollout is visible in CloudWatch in real time. Previously used
    # subprocess.run(capture_output=True), which buffered ALL output and the 1h
    # timeout discarded it -> the job hung invisibly. A watchdog thread enforces the
    # hard timeout even when the child emits nothing (readline would block forever).
    import shutil as _shutil
    import threading as _threading
    # Force the child (Isaac Sim C++ + Python policy_runner) to flush line-by-line.
    # bufsize=1 only line-buffers the PARENT's pipe reads; the CHILD, seeing a pipe
    # (non-TTY), block-buffers stdout -> readline() below sees NOTHING until 4KB
    # fills or the process exits, which reproduced v30's blind hang. stdbuf sets
    # libc line-buffering (covers Isaac Sim's C++ side); PYTHONUNBUFFERED covers the
    # Python layers. Fall back to the bare cmd if stdbuf is unavailable in the image.
    _stream_env = dict(os.environ)
    _stream_env["PYTHONUNBUFFERED"] = "1"
    _stream_cmd = (["stdbuf", "-oL", "-eL"] + cmd) if _shutil.which("stdbuf") else cmd
    proc = subprocess.Popen(_stream_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=_stream_env,
                            start_new_session=True)
    _timed_out = {"v": False}

    def _watchdog():
        _timed_out["v"] = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # The watchdog is a HANG detector, not a second budget knob. It must not be
    # derived from the sampling budget (it previously scaled with num_steps, which no
    # longer exists), and expiry must FAIL the evaluation -- never publish whatever
    # was printed before the kill as a completed result.
    #
    # In episode mode the simulation bound is conditional: each episode ends within
    # the environment's max_episode_length, but only if a termination or truncation
    # flag actually fires. If it never does, the rollout loop would run forever, so a
    # finite wall-clock ceiling is the backstop. Allowance per episode is deliberately
    # generous (observed ~6.5 policy steps/s, ~500 steps per episode ~= 77s) plus a
    # startup/teardown buffer for Isaac Sim and the GR00T server.
    _timeout_s = int(os.environ.get("SM_HP_EVAL_TIMEOUT_S", "0") or 0)
    if _timeout_s <= 0:
        _timeout_s = max(3600, num_episodes * 900 + 1800)
    _timer = _threading.Timer(_timeout_s, _watchdog)
    _timer.start()
    _lines = []
    try:
        for _line in iter(proc.stdout.readline, ""):
            _line = _line.rstrip("\n")
            _lines.append(_line)
            print(f"[policy_runner] {_line}", flush=True)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
        _timer.cancel()
    result_stdout = "\n".join(_lines)

    if _timed_out["v"]:
        log(f"<<< arena_eval FAIL: TIMEOUT after {_timeout_s}s -- policy_runner killed. Tail:")
        log(result_stdout[-3000:])
        raise RuntimeError(f"Arena eval timed out after {_timeout_s}s")
    if rc != 0:
        log(f"<<< arena_eval FAIL: policy_runner exited rc={rc}. Tail:")
        log(result_stdout[-3000:])
        raise RuntimeError(f"Arena eval failed with return code {rc}")

    # Parse and VALIDATE before claiming success: a zero exit code only establishes
    # that policy_runner exited cleanly, not that it produced a usable measurement.
    # The observed count is checked against the requested budget because the reported
    # number comes from the recorder dataset, not the rollout loop's counter.
    results = parse_arena_output(result_stdout, expected_episodes=num_episodes)
    log(f"<<< arena_eval OK: {results['episodes']} episodes, "
        f"success_rate={results.get('success_rate')}, task={task_name}")
    log(f"Output: {result_stdout[-1000:]}")
    results["policy_type"] = "checkpoint" if is_remote else policy_type
    results["task_name"] = task_name
    return results


def run_zero_action_eval(
    task_name: str,
    num_envs: int,
    num_episodes: int,
) -> dict:
    """Faithful zero-action NEGATIVE CONTROL (zero-action).

    Runs policy_runner `--policy_type zero_action` on the SAME task/scene as the real
    eval by delegating to run_arena_eval (is_remote=False → no gr00t server, no
    --policy_config_yaml_path/--enable_cameras; the hoisted scene block still supplies
    the env-gated --embodiment/--object so the GR1 scene loads identically). It STREAMS
    output and PARSES the real `[Rank] Metrics:` line via parse_arena_output.

    Previously this returned a HARDCODED success_rate=0.0 (a fabricated null, ignoring
    result.stdout) -- fixed so the negative control reports the ACTUAL measured rate.
    A near-0 result confirms the metric's zero point; anything high => the scene/reset
    physics opens the door without a policy (harness bug)."""
    return run_arena_eval(
        task_name=task_name,
        num_envs=num_envs,
        num_episodes=num_episodes,
        policy_type="zero_action",
    )


# Arena's policy_runner emits a CUMULATIVE metrics record at every episode boundary
# and once more after the rollout loop, so a *sequence* of Metrics lines is expected
# and the final count legitimately repeats (upstream policy_runner.py:104-111,234-235).
# Take the LAST such record (the cumulative total).
# ANCHORED AT LINE START. The previous pattern anchored only the END, so the comment
# above claiming a rank-qualified anchor was false: any line CONTAINING the prefix
# matched, and a probe accepted
#   debug anticipated [Rank 0/1] Metrics: {"num_episodes": 1, "success_rate": 1.0}
# as a completed successful evaluation. Upstream emits the record at the start of its
# own line, so requiring that costs nothing and removes the spoofing surface.
_RANK_METRICS_RE = re.compile(
    r"^\[Rank\s+\d+\s*/\s*\d+\]\s*Metrics:\s*(\{.*\})\s*$")

# `ast.literal_eval` cannot parse bare `nan`/`inf`, which Arena emits for a metric
# averaged over zero samples. Substitute ONLY bare tokens in value position: the
# lookarounds keep identifiers and quoted strings intact, so `inference_latency`
# and `'nan'` survive (a blind str.replace corrupts both).
_NONFINITE_RE = re.compile(
    r"""(?<![A-Za-z0-9_.'"])-?(?:nan|inf(?:inity)?)(?![A-Za-z0-9_.'"])""",
    re.IGNORECASE,
)

# Required, result-bearing fields. Everything else Arena reports (per-subtask rates,
# subtask_success_rate, ...) is auxiliary: preserved verbatim under its ORIGINAL name
# so a non-finite auxiliary value is recorded explicitly instead of vanishing.
_REQUIRED_RATES = ("success_rate",)


def parse_arena_output(stdout: str, expected_episodes: int | None = None) -> dict:
    """Parse Arena's final cumulative metrics record.

    Fails loud on an incomplete rollout. `num_episodes == 0` means the step budget
    never produced an episode boundary, so every rate was averaged over an empty
    sample -- that is a configuration error, not a result, and it must NOT reach
    Validate as a report. When `expected_episodes` is given the observed count must
    equal it: the printed count comes from the recorder dataset rather than the loop
    counter, so it is checked rather than assumed.
    """
    summary = None
    for line in stdout.splitlines():
        m = _RANK_METRICS_RE.search(line.strip())
        if m:
            summary = m.group(1)
    if summary is None:
        raise RuntimeError(
            "Arena output has no rank-qualified '[Rank i/n] Metrics: {...}' record -- "
            "the rollout did not reach the metrics stage (or the output was truncated)")
    try:
        metrics = ast.literal_eval(_NONFINITE_RE.sub("None", summary))
    except (ValueError, SyntaxError) as exc:
        raise RuntimeError(
            f"Arena final Metrics summary is malformed: {summary!r}") from exc
    if not isinstance(metrics, dict):
        raise RuntimeError("Arena Metrics summary must be a dictionary")

    episodes = metrics.get("num_episodes")
    if type(episodes) is not int or episodes < 1:
        raise RuntimeError(
            f"Arena reported num_episodes={episodes!r}: a completed evaluation requires "
            "at least one finished episode. An episode completes only on termination or "
            "truncation, so a step budget below the environment's max_episode_length "
            "yields zero episodes and every rate is averaged over an empty sample. "
            "Request episodes directly (--num_episodes) or raise the step budget above "
            "k * max_episode_length. HARD FAIL (do NOT record an empty evaluation).")
    if expected_episodes is not None and episodes != expected_episodes:
        raise RuntimeError(
            f"Arena completed {episodes} episodes but {expected_episodes} were requested. "
            "The evaluated sample must match the requested budget exactly; a differing "
            "count means the rollout stopped early or overshot (e.g. num_envs > 1 ends "
            "several episodes in one vectorized step). HARD FAIL.")

    results = {"eval_backend": "isaac_lab_arena", "episodes": episodes}
    for field in _REQUIRED_RATES:
        value = metrics.get(field)
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise RuntimeError(f"Arena {field} must be a finite number in [0, 1]")
        results[field] = value
    # Auxiliary metrics are diagnostic, never gated: keep them under their real names
    # (Arena suffixes per-subtask rates, e.g. revolute_joint_moved_rate_subtask_1, so
    # an unsuffixed lookup silently matches nothing). `None` here is an explicit
    # "reported but not finite", not a fabricated zero.
    aux = {k: v for k, v in metrics.items()
           if k != "num_episodes" and k not in _REQUIRED_RATES}
    if aux:
        results["aux_metrics"] = aux
    return results


def _policy_config_digest(path=None):
    """Digest the CONTENT of the policy config the run consumed.

    Recording only the path proved which file was selected, not what it contained: the
    same path can carry a different protocol between runs, so a comparison over labels
    and paths cannot detect that the evaluation ran a different configuration than the
    one requested.

    Returns None when the file cannot be read, and says so, rather than substituting a
    value that would look like evidence.
    """
    import hashlib

    # I5: this always hashed the DECLARED (template) config, while the runner is handed a
    # PATCHED copy with model_path rewritten. So the report described bytes the
    # evaluation did not consume -- the same class of defect as reporting module globals
    # while executing checkpoint-overridden locals. The caller now names the file that
    # was actually passed; the template digest is kept separately so the rewrite itself
    # remains auditable.
    path = path if path is not None else _DECLARED_POLICY_CONFIG
    if not path:
        log("WARNING: no declared policy config, so no config digest is recorded")
        return None
    try:
        with open(path, "rb") as handle:
            return "sha256:" + hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        log(f"WARNING: cannot read policy config {path!r} to digest it ({exc}); "
            f"recording null rather than a value that would imply verification")
        return None


def _seed_scope_for_report(checkpoint_path=None, port=None):
    """What the eval seed actually bound, read from evidence rather than asserted.

    Arena's runner seeds the CLIENT process and the environment (python/numpy/torch plus
    env.seed), which covers scene and initial-state randomness. The policy server is a
    SEPARATE process whose inference noise (torch.randn in the action head) was
    previously unbound, so the same seed reproduced the scene sequence while the policy
    sampled different actions.

    The seeding wrapper publishes what it applied; this records that rather than claiming
    a scope this process cannot observe. If the evidence is absent the policy RNG is
    recorded as UNVERIFIED -- silently implying it was bound is the overstatement this
    finding is about.
    """
    # R4#I2b: the per-launch RECORD is authoritative. Nothing in this process ever writes
    # GR00T_SERVER_SEED_EVIDENCE into os.environ -- both launchers set it only on the CHILD's
    # server_env -- so a value visible HERE is inherited from outside the run and cannot
    # describe the launch being reported. Two launchers record (start_groot_server for N1.7,
    # start_groot_server_n16 for the N1.6 positive control), so letting an ambient value win
    # attributes one arm's seeding proof to the other.
    recorded = (_LAUNCH_EVIDENCE.get(os.path.realpath(checkpoint_path))
                if checkpoint_path is not None else None)
    if recorded is None and checkpoint_path is not None and port is not None:
        # No launch recorded for this checkpoint: fall back to the derived path so a caller
        # that bypassed the launcher still looks somewhere principled rather than at a shared
        # default. It will fail validation if nothing wrote there, which is correct.
        recorded = _server_evidence_path(checkpoint_path, port)
    # The env var is consulted ONLY when no launch was recorded AND none could be derived --
    # the bypass case, where a baked layout or a test genuinely is the only available source.
    evidence_path = recorded or os.environ.get(
        "GR00T_SERVER_SEED_EVIDENCE", "/tmp/gr00t_server_seed_evidence.json")
    server_seeding = None
    unverified_reason = None
    try:
        with open(evidence_path) as handle:
            candidate = json.load(handle)
    except (OSError, ValueError) as exc:
        candidate = None
        unverified_reason = f"no readable evidence at {evidence_path} ({exc})"
    if candidate is not None:
        # I10: any nonempty JSON object used to count as evidence, and the reader asked for
        # none of the things the evidence exists to certify. A stale file from an earlier
        # launch, or one recording a different seed, was indistinguishable from proof.
        problems = []
        if not isinstance(candidate, dict):
            problems.append(f"evidence is {type(candidate).__name__}, not an object")
        else:
            if candidate.get("evidence_schema") != "gr00t_server_evidence_v1":
                problems.append(
                    f"schema is {candidate.get('evidence_schema')!r}, expected "
                    f"'gr00t_server_evidence_v1'")
            if candidate.get("seed") != int(EVAL_SEED):
                problems.append(
                    f"records seed {candidate.get('seed')!r} but this run requested "
                    f"{int(EVAL_SEED)}")
            for stage in ("process_start", "inference_ready"):
                if not candidate.get(stage):
                    problems.append(f"reseed stage {stage!r} not recorded as applied")
            if candidate.get("strict_load_audit") != "installed":
                problems.append("strict weight-load audit was not installed")
            audits = candidate.get("strict_load_audits")
            if not isinstance(audits, list) or not audits:
                problems.append("no completed weight-load audit is recorded")
        if problems:
            unverified_reason = "; ".join(problems)
        else:
            server_seeding = candidate
    if server_seeding is None:
        log(f"WARNING: server seed evidence not usable ({unverified_reason}); recording "
            f"the policy RNG as unverified rather than claiming it was bound")
    return {
        "client_and_env": True,
        "server_policy_rng": server_seeding is not None,
        "server_seeding_evidence": server_seeding,
        # Why it is unverified, so a reader is not left guessing whether evidence was
        # absent or present-but-wrong.
        "server_policy_rng_unverified_reason": unverified_reason,
        # Episodes within a run are NOT independently replayable: the server is seeded
        # once per run and episode resets do not reseed (both pinned Gr00tPolicy.reset()
        # implementations return {}, so a reset-time seed would not apply).
        "per_episode_replayable": False,
    }


def _s3_head_identity(model_source_uri: str):
    """model_artifact_identity via head_object on the FineTune S3 artifact.

    Schema-v2 requires bucket/key/version_id/etag. FAILS HARD (raises) if the URI
    is missing/malformed or the object/VersionId/ETag cannot be obtained -- never
    fabricates a value ("null") and never returns None to mask the failure. A
    required identity that can't be resolved is a real error, surfaced HERE.
    """
    if not model_source_uri:
        raise RuntimeError(
            "EVAL_MODEL_SOURCE_URI is empty -- cannot build model_artifact_identity "
            "(required in pipeline mode). HARD FAIL (do NOT fabricate/skip).")
    from urllib.parse import urlparse

    import boto3
    u = urlparse(model_source_uri)
    bucket, key = u.netloc, u.path.lstrip("/")
    if u.scheme != "s3" or not bucket or not key:
        raise RuntimeError(f"EVAL_MODEL_SOURCE_URI is not a valid s3:// URI: {model_source_uri!r}")
    head = boto3.client("s3").head_object(Bucket=bucket, Key=key)  # raises if missing
    version_id = head.get("VersionId")
    if not version_id:
        raise RuntimeError(
            f"s3://{bucket}/{key} has no VersionId -- the model-artifact bucket MUST be "
            "versioned so model_artifact_identity is provable. Enable bucket versioning; "
            "do NOT fabricate a version_id.")
    # An UNVERSIONED bucket does not omit VersionId -- it returns the literal string
    # "null". The falsy check above therefore passes it straight through, and the
    # docstring's promise never to record "null" was defeated by S3 supplying that
    # value itself rather than by this function fabricating it. Reject it explicitly:
    # "null" selects no particular generation, so it proves nothing about the bytes.
    if version_id.strip().lower() in {"null", "none"}:
        raise RuntimeError(
            f"s3://{bucket}/{key} reported VersionId={version_id!r}, which is the marker "
            f"S3 returns for an object in an UNVERSIONED bucket. It cannot be used to "
            f"retrieve one specific generation of the artifact, so it is not provable "
            f"identity. Enable bucket versioning; do NOT record it as a version.")
    etag = (head.get("ETag") or "").strip('"')
    if not etag:
        raise RuntimeError(f"s3://{bucket}/{key} head_object returned no ETag.")
    return {"s3_uri": model_source_uri, "bucket": bucket, "key": key,
            # schema 3: checksum_sha256 is gone. S3 returns it only when the object was
            # uploaded with a checksum algorithm, so it was always None here -- a key
            # implying a guarantee the identity never carried. source_archive is the
            # content evidence.
            "version_id": version_id, "etag": etag}


def verify_checkpoint(checkpoint_root: str) -> tuple[dict, str]:
    from digest import weights_digest
    from validator import validate_manifest

    with open(Path(__file__).with_name("defaults.json")) as source:
        defaults = json.load(source)
    with open(Path(checkpoint_root) / "checkpoint_manifest.json") as source:
        manifest = json.load(source)
    expected_family = os.environ.get("EVAL_MODEL_FAMILY", "gr00t")
    if manifest["model_family"] != expected_family:
        raise ValueError("checkpoint model_family differs from EVAL_MODEL_FAMILY")
    validate_manifest(manifest, {defaults["family"]: {
        "input_config_schema": defaults["input_config_schema"],
        "provenance_keys": defaults["provenance_keys"],
    }})
    recomputed = weights_digest(checkpoint_root)
    if recomputed != manifest["weights_digest"]:
        raise ValueError("checkpoint digest differs from manifest")
    return manifest, recomputed


def realizable_successes(success_rate, episodes):
    """The whole number of successful episodes a rate implies, or None if impossible.

    Success is a BINARY per-episode outcome, so a rate must be a whole number of successful
    episodes over the episode count. Three episodes at 0.5 is one and a half successful
    episodes: not a measurement.

    Shared so the canonical report and the dose sidecar cannot disagree. The canonical path had
    this check and the dose path did not, so a dose point recording three episodes at 0.5 was
    marked ok=True (I4). Fixing one of two paths and leaving the other has been the recurring
    shape of defects in this component.
    """
    if type(episodes) is not int or episodes < 1:
        return None
    if type(success_rate) not in (int, float):
        return None
    if not 0 <= success_rate <= 1:
        return None
    exact = float(success_rate) * episodes
    nearest = round(exact)
    if abs(exact - nearest) > 1e-6 or not 0 <= nearest <= episodes:
        return None
    return nearest


def write_metrics(results: dict, output_dir: str, checkpoint_root: str,
                  posctrl_commit: str | None = None):
    """Write a schema-v2-COMPLETE metrics.json (validator TOP_KEYS).

    Reads checkpoint_manifest.json from the mounted FineTune checkpoint,
    recomputes the weights digest (digest.py excludes the manifest + logs so it
    matches train's digest), resolves the model artifact identity via head_object,
    and emits provenance. Also copies the manifest into the eval output dir so the
    Validate step's digest cross-check can find it. NO non-schema extras.
    """
    from digest import weights_digest  # baked into the Arena image next to eval_entry

    model_family = os.environ.get("EVAL_MODEL_FAMILY", "gr00t")
    suite = os.environ.get("EVAL_SUITE", "arena_gr1")
    eval_seed = int(os.environ.get("EVAL_SEED", str(EVAL_SEED)))
    trials = EVAL_TRIALS
    model_source_uri = os.environ.get("EVAL_MODEL_SOURCE_URI", "")

    task_ids = requested_task_ids()
    task_name = results["task_name"]
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("Arena results must identify the executed task")

    episodes = int(results.get("episodes", 0) or 0)
    # success_rate is result-bearing: parse_arena_output guarantees it (or raises)
    # and the zero_action path sets it explicitly. If it is somehow absent, fail
    # loud rather than record a fabricated 0.0 (episodes==0 is separately gated).
    if "success_rate" not in results:
        raise RuntimeError(
            "write_metrics: results has no 'success_rate' -- the eval produced no "
            "score. HARD FAIL (do NOT record a fabricated 0.0).")
    # schema 3: refuse to publish without a measured archive. Emitting an empty object would
    # satisfy the key set while carrying no evidence, which is worse than failing -- the field
    # exists precisely so a consumer need not wonder whether it means anything.
    # Scoped to archive mode. This demanded an archive measurement for EVERY report including the
    # positive control, which evaluates a published HF snapshot and has no mounted tarball -- so a
    # diagnostic that deliberately ignores the mounted checkpoint still required one.
    #
    # R4#I1: this guard tested results["policy_type"], but run_arena_eval OVERWRITES that field --
    # `results["policy_type"] = "checkpoint" if is_remote else policy_type` -- and the positive
    # control is served remotely, so it always arrives here labelled "checkpoint". The only place
    # "positive_control" is ever written is INSIDE the branch below, which this guard prevented it
    # from reaching. The result: the positive control could never publish, while the branch that
    # knows how to build its snapshot report sat unreachable ten lines further down.
    #
    # Key the guard on the SAME condition as the branch -- the mode flag, which no caller rewrites.
    # _is_positive_control() is the single reader, so the two cannot drift apart again.
    if _RESOLVED is None and not _is_positive_control(results):
        raise RuntimeError(
            "no source identity was constructed, so this report cannot state which bytes were "
            "evaluated. The archive is measured before extraction; reaching publication without it "
            "means extraction was skipped or the channel held no model.tar.gz.")
    success_rate = float(results["success_rate"])
    # Success is a BINARY per-episode outcome, so the rate must be a whole number of
    # successful episodes over the episode count. The producer checked aggregate ranges but
    # not realizability, so it could publish e.g. 3 episodes at 0.5 -- one and a half
    # successful episodes. The shared validator rejects that, but Arena publishes WITHOUT
    # calling it, so a standalone SimEval completed and published invalid metrics that only
    # the pipeline's Validate step would have caught.
    _successes = realizable_successes(success_rate, episodes)
    if _successes is None:
        raise RuntimeError(
            f"write_metrics: success_rate={success_rate!r} is not realizable from "
            f"{episodes} episode(s) -- it implies {success_rate * episodes} successful "
            f"episodes, which is not a whole number in [0, {episodes}]. Success is binary "
            f"per episode. HARD FAIL (do NOT publish an impossible score).")
    per_task = [{
        "task_id": task_ids[0],
        "task": task_name,
        "episodes": episodes,
        "success_rate": success_rate,
        # The authoritative count, rather than leaving consumers to reconstruct it from a
        # floating-point rate.
        "successes": _successes,
    }]
    episodes_reported = episodes

    # --- POSITIVE-CONTROL PATH (EVAL_POSCTRL_N16): the served checkpoint is NVIDIA's
    # PUBLISHED HF snapshot, NOT a FineTune output -- there is no checkpoint_manifest.json
    # and no trust-chain digest to verify against our recipe. Emit a positive-
    # control report (provenance = the HF repo, no fabricated manifest) and return,
    # BYPASSING the FineTune-manifest requirement below. This branch fires ONLY for
    # EVAL_POSCTRL_N16=true, so the real fine-tuned path keeps the full trust chain intact.
    if _is_positive_control(results):
        posctrl_repo = N16_POSCTRL_CKPT_REPO
        # The commit the Hub RESOLVED, threaded from the download that fetched these bytes -- not the
        # requested reference, which was "main" by default and published as though it pinned them.
        if not (isinstance(posctrl_commit, str) and re.fullmatch(r"[0-9a-f]{40}", posctrl_commit)):
            raise RuntimeError(
                f"FATAL: write_metrics needs the RESOLVED positive-control commit, got "
                f"{posctrl_commit!r}. Publishing the requested reference instead is how a moving "
                f"branch name was attested as the identity of the evaluated weights.")
        posctrl_rev = posctrl_commit
        # NO try/except. A digest that cannot be computed is a MISSING measurement, and publishing
        # tree_sha256: None recorded the absence as though it were a value -- the validator requires
        # 64-hex, so this produced a report that had already spent its whole budget and could never
        # be published. Failing here says which bytes could not be measured, and why.
        # C1 (review pass-3): this hand-built its own source_snapshot and put weights_digest()'s
        # PREFIXED "sha256:<hex>" into tree_sha256, which the validator requires to be BARE 64-hex --
        # so every positive control would have spent its full inference budget and then failed to
        # publish. That is cycle-17's I6 defect recurring, in the ONE path left unconverted when the
        # shared factory was introduced. It also set checkpoint=<local path> while the validator binds
        # repo_id == report["checkpoint"], so the two disagreed by construction.
        #
        # Built by the shared factory now, like every other producer. The factory measures the tree
        # itself and owns the digest form -- report_source_fields() applies bare_hex() to tree_sha256 --
        # so neither mistake is expressible here.
        from source_identity import from_snapshot
        _posctrl_source = from_snapshot(load_root=checkpoint_root,
                                        repo_id=posctrl_repo,
                                        resolved_commit=posctrl_rev)
        # The PREFIXED form, which is what weights_digest_recomputed_by_eval carries.
        posctrl_digest = _posctrl_source.tree_digest
        metrics = {
            "schema_version": 3,
            # cycle-16 C1: the external VLM backbone determines preprocessing, and it was
            # downloaded with no revision -- so it could change while the checkpoint, its
            # digest, the image and the sourcedir all stayed identical. Recorded under
            # aux_metrics because no gate reads it, but it travels inside the report and is
            # therefore covered by the same schema and digest chain.
            "backbone_identity": dict(_BACKBONE_IDENTITY),
            # The archive this job actually read, measured before extraction. Validate
            # measures its own copy and requires both to agree, which is what shows the
            # two read the same bytes.
            # cycle-15 I6 + cycle-14 I4, together. This report used to carry the MOUNTED archive,
            # which did not supply the inference weights -- the served HF snapshot did. I first
            # added weights_served_by alongside it because the validator required an archive; now
            # that a snapshot variant exists, the archive can be dropped entirely and the report
            # names ONLY what actually served the policy.
            # Every identity field from the one object -- checkpoint, checkpoint_revision and the
            # source variant together, so they cannot disagree.
            **_posctrl_source.report_identity_fields(),
            "policy_type": "positive_control",
            "model_family": model_family,
            "checkpoint_manifest": None,          # published reference, not a FineTune output
            "positive_control": True,
            "positive_control_repo": posctrl_repo,
            "weights_digest_recomputed_by_eval": posctrl_digest,
            "model_artifact_identity": None,
            "suite": suite,
            "task_ids": task_ids,
            "num_trials_per_task": trials,
            "eval_seed": eval_seed,
            "train_seed": None,                   # not our FineTune -> no seed
            "success_rate": success_rate,
            "episodes": episodes,
            "episodes_reported_by_evaluator": episodes_reported,
            "per_task": per_task,
            "provenance": {
                "recipe_repo": posctrl_repo,
                "recipe_commit": posctrl_rev,
                "mujoco_gl": os.environ.get("MUJOCO_GL", "egl"),
                "repo_commit": posctrl_rev,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "note": ("N1.6 native-GR1 positive control -- NVIDIA published checkpoint served "
                         "through our harness; NOT a FineTune output (no manifest/digest chain)."),
            },
        }
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        temporary = output_path / "metrics.json.tmp"
        with open(temporary, "w") as f:
            json.dump(metrics, f, indent=2, allow_nan=False)
        os.replace(temporary, output_path / "metrics.json")
        log(f"Wrote schema-v3 POSITIVE-CONTROL metrics to {output_path / 'metrics.json'}")
        log(f"  success_rate={success_rate} episodes={episodes} posctrl_repo={posctrl_repo} "
            f"weights_digest={posctrl_digest}")
        return metrics

    # --- checkpoint_manifest from the EXTRACTED FineTune checkpoint (REQUIRED) ---
    # No fabrication/degrade: a checkpoint with no manifest is a broken artifact.
    # SageMaker does NOT auto-extract the "model" channel tarball, so the raw
    # channel (INPUT_MODEL) holds only model.tar.gz. The eval extracts it to
    # `checkpoint_root` (/tmp/checkpoint) -- resolve the manifest there, with an
    # os.walk fallback (mirrors the proven steps/gr00t/eval_entry.py resolver,
    # which resolves against the EXTRACTED model_path, not the channel). Still
    # HARD-FAIL if genuinely absent; never fabricate.
    manifest_path = os.path.join(checkpoint_root, "checkpoint_manifest.json")
    if not os.path.exists(manifest_path):
        found = None
        for _root, _dirs, _files in os.walk(checkpoint_root):
            if "checkpoint_manifest.json" in _files:
                found = os.path.join(_root, "checkpoint_manifest.json")
                break
        manifest_path = found
    if not manifest_path or not os.path.exists(manifest_path):
        raise RuntimeError(
            f"checkpoint_manifest.json not found under {checkpoint_root} -- the FineTune "
            "artifact is incomplete; cannot build a schema-v2 report. HARD FAIL "
            "(do NOT emit a null/partial manifest to pass validation).")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # --- weights digest recomputed by eval (must equal manifest.weights_digest) ---
    # Recompute over the tree that CONTAINS the manifest (the resolved checkpoint
    # root), matching train_entry's weights_digest(MODEL_DIR). No swallowing: a
    # digest failure is fatal (the trust chain depends on it).
    manifest_tree = os.path.dirname(manifest_path)
    recomputed = weights_digest(manifest_tree)
    # The comment above promised this was fatal, but the comparison only ever happened
    # in a log line at the END of this function -- so a mismatch was RECORDED in the
    # report and published anyway. The checkpoint is verified once when it is resolved;
    # this is the second, post-rollout recompute, and it is the one that would catch a
    # checkpoint that changed underneath a running evaluation. Compare it here, before
    # anything is written, so the producer cannot publish evidence it already knows is
    # invalid and merely hope the downstream Validate step rejects it.
    _declared_digest = manifest["weights_digest"]
    if recomputed != _declared_digest:
        raise RuntimeError(
            f"weights digest mismatch after evaluation: recomputed={recomputed} but the "
            f"checkpoint manifest declares {_declared_digest}. The evaluated bytes are "
            f"not the bytes the manifest describes, so this report cannot be attributed "
            f"to that checkpoint. HARD FAIL (do NOT publish a report with a known "
            f"digest mismatch).")

    train_seed = manifest["train_seed"]      # required key (may be null per contract)
    recipe = manifest["train_recipe"]        # required
    recipe_commit = recipe["commit"]         # required 40-hex -- NEVER fabricate
    recipe_repo = recipe["repo"]             # required nonempty -- NEVER "unknown"

    metrics = {
        "schema_version": 3,
        # cycle-16 C1: the SECOND report site in this file. The first got the field and this one
        # did not, because my edit anchored on a specific indentation -- the same one-of-N slip
        # this review keeps finding. Both sites publish a report, so both need the identity.
        "backbone_identity": dict(_BACKBONE_IDENTITY),
        # The archive this job read, measured before extraction (schema 3). Both report
        # sites in this file must carry it -- adding it to one produced a report the
        # shared validator rejected, which is how this second site was found.
        # checkpoint, checkpoint_revision and the single source variant all derive from the one
        # resolved object, so they cannot describe different things.
        **_RESOLVED.report_identity_fields(),
        "policy_type": results["policy_type"],
        "model_family": model_family,
        "checkpoint_manifest": manifest,
        "weights_digest_recomputed_by_eval": recomputed,
        "model_artifact_identity": _s3_head_identity(model_source_uri),
        "suite": suite,
        "task_ids": task_ids,
        "num_trials_per_task": trials,
        "eval_seed": eval_seed,
        "train_seed": train_seed,
        "success_rate": success_rate,
        "episodes": episodes,
        "episodes_reported_by_evaluator": episodes_reported,
        "per_task": per_task,
        "provenance": {
            "recipe_repo": recipe_repo,
            "recipe_commit": recipe_commit,
            "mujoco_gl": os.environ.get("MUJOCO_GL", "egl"),
            "repo_commit": recipe_commit,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "effective_eval_config": {
            # The budget protocol is part of the evidence: `fixed_trials` means the
            # sample size was requested as complete episodes and must equal what was
            # asked for, which is what Validate enforces per task.
            "budget_type": "fixed_trials",
            "num_episodes": EVAL_TRIALS,
            "num_envs": NUM_ENVS,
            # N1.6 hardcodes --embodiment-tag GR1 at the server, so report what was
            # SERVED, not what was requested. Uses the normalized constant so an
            # unnormalized value cannot make this disagree with the launcher.
            "embodiment_tag": ("GR1" if EVAL_GR00T_VERSION == "n16"
                               else os.environ.get("SM_HP_EMBODIMENT_TAG", "")),
            "arena_embodiment": os.environ.get("EVAL_ARENA_EMBODIMENT", ""),
            # NOTE: this is the value PASSED, and "NONE" means "omit the flag" rather
            # than "no object" -- the pinned environment then selects its own default.
            # The consumed object is therefore NOT identified by this field. Recorded as
            # the request, with the consumed value left explicitly uncaptured below
            # rather than implied.
            "arena_object": os.environ.get("EVAL_OBJECT", ""),
            "policy_config": _DECLARED_POLICY_CONFIG,
            # The CONTENT of that config, not just its path. A path proves which file was
            # selected, not what it contained: the same path can carry a different
            # protocol (chunk length, joint ordering, camera handling) between runs, and
            # Validate compared only the path and a few labels.
            # I5: the digest of the config the runner ACTUALLY received. _EXECUTED_POLICY_CONFIG
        # is set at the launch site; when it is unset this falls back to the declared file
        # and the template digest below makes the two distinguishable rather than
        # silently identical.
        "policy_config_digest": _policy_config_digest(_EXECUTED_POLICY_CONFIG or None),
        "policy_config_template_digest": _policy_config_digest(),
            "gr00t_version": EVAL_GR00T_VERSION,
            # Where the observation/modality contract actually comes from. The
            # --modality-config-path flag is inert on the checkpoint path (I7): both
            # pinned servers read the modality configuration from the checkpoint's own
            # processor. Recording the source prevents the report from implying that the
            # flag's value describes the contract that was used.
            "modality_config_source": "checkpoint_processor",
            "modality_config_flag_inert": True,
            # Values that are only observable inside Arena at runtime and are NOT yet
            # captured. Listed explicitly so this evidence is not read as a complete
            # protocol fingerprint: physics timestep, control decimation, the resolved
            # scene object and assets, camera configuration, the language string actually
            # sent (the runner selects an override or task description, with the YAML
            # language only as a fallback), and the executed action chunk length.
            "uncaptured_protocol_values": [
                "physics_timestep", "control_decimation", "resolved_scene_object",
                "camera_configuration", "language_sent", "executed_chunk_length",
            ],
        },
    }

    # Auxiliary metrics the simulator reported alongside the gated score (per-subtask
    # rates, subtask success vectors, ...). parse_arena_output preserves them under
    # their real names; they were then dropped here, so the run could not be diagnosed
    # afterwards even though the values had been read. Non-gated evidence: recorded, but
    # nothing reads them for a pass/fail decision, and a value the simulator reported as
    # non-finite stays null rather than becoming a fabricated number.
    if results.get("aux_metrics"):
        metrics["aux_metrics"] = results["aux_metrics"]

    # What the eval seed actually bound. Arena's runner seeds the CLIENT process and the
    # environment, which covers scene and initial-state randomness; the policy server is
    # a separate process whose inference noise (torch.randn in the action head) was
    # previously unbound, so the seed reproduced the scene sequence but not the actions.
    # The seeding wrapper publishes what it applied; record it rather than asserting a
    # scope this code cannot observe. A run whose server evidence is absent must say so
    # instead of implying the policy RNG was bound.
    # Scoped to THIS checkpoint's server launch, so the dose-curve path cannot report
    # the last server's evidence as every checkpoint's.
    metrics["seed_scope"] = _seed_scope_for_report(checkpoint_root, SERVER_PORT)

    # I2: backbone_identity was copied from _BACKBONE_IDENTITY, which the PRECACHE step populates. The
    # server's own audit records the commit inference actually resolved, and nothing read it -- so a
    # precache commit A with a serving commit B published A. A false attribution is worse than an absent
    # one, and this is the exact "declared value that governs nothing" shape the audit was added to fix.
    #
    # Scoped to the N1.7 checkpoint path: N1.6 serves natively without the Cosmos backbone, so demanding
    # a Cosmos audit there would refuse a valid run.
    if metrics.get("policy_type") == "checkpoint" and EVAL_GR00T_VERSION == "n17":
        from gr00t_seeded_server import consumed_backbone_identity
        metrics["backbone_identity"] = consumed_backbone_identity(
            metrics["seed_scope"], checkpoint_root, "nvidia/Cosmos-Reason2-2B")
        log(f"backbone identity from the LOAD: {metrics['backbone_identity']}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Publish atomically: write beside the target then rename, so a reader (or a job
    # that dies mid-write) can never observe a truncated report as if it were complete.
    # Run the SHARED report validator before publishing, so the producer cannot emit a
    # report its own consumer considers impossible. Arena previously published without
    # calling it, so a standalone SimEval completed successfully while writing metrics that
    # Validate would reject -- the invalid evidence existed on disk and only the pipeline
    # path caught it.
    #
    # Scoped to REGISTRABLE reports. validate_report is the registration contract and
    # requires policy_type == "checkpoint", while zero_action and positive_control are
    # legitimate diagnostic modes that never claim registrability -- running the full
    # contract against them would reject valid diagnostics. The realizability check above is
    # unconditional, because an impossible score is impossible in any mode.
    if metrics.get("policy_type") == "checkpoint":
        try:
            from validator import validate_report
        except ImportError as _exc:
            raise RuntimeError(
                f"cannot import the shared report validator ({_exc}), so this report "
                f"cannot be checked against the contract it must satisfy. HARD FAIL "
                f"rather than publish unvalidated evidence.") from _exc
        with open(Path(__file__).with_name("defaults.json")) as _fh:
            _defaults = json.load(_fh)
        validate_report(
            metrics,
            expected_suite=suite,
            expected_trials=EVAL_TRIALS,
            expected_eval_seed=EVAL_SEED,
            expected_task_ids=task_ids,
            family_schemas={_defaults["family"]: {
                "input_config_schema": _defaults["input_config_schema"],
                "provenance_keys": _defaults.get("provenance_keys", {}),
            }},
            pipeline_mode=bool(manifest),
            expected_budget_type="fixed_trials",
        )
        log("shared schema-v3 validation PASSED at the producer")
    else:
        log(f"policy_type={metrics.get('policy_type')!r} is a diagnostic mode, so the "
            f"registration contract does not apply; realizability was still enforced")

    if _EXECUTED_POLICY_CONFIG:
        # Preserve the final checkpoint's patched YAML beside its metrics. This
        # is the evaluator's recorded configuration, not an independent scene audit.
        import hashlib
        config_bytes = Path(_EXECUTED_POLICY_CONFIG).read_bytes()
        config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        if config_digest != metrics["effective_eval_config"]["policy_config_digest"]:
            raise RuntimeError("Policy config changed while publishing evaluation output")
        config_output = output_path / "policy_config.yaml"
        config_temporary = output_path / "policy_config.yaml.tmp"
        config_temporary.write_bytes(config_bytes)
        config_temporary.replace(config_output)
        log(f"Saved effective policy config: {config_output} ({config_digest})")

    _final = output_path / "metrics.json"
    _tmp = output_path / "metrics.json.tmp"
    with open(_tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(_tmp, _final)

    # Copy the manifest into the eval output so Validate's digest cross-check finds it.
    if manifest is not None:
        with open(output_path / "checkpoint_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    log(f"Wrote schema-v3 metrics to {output_path / 'metrics.json'}")
    log(f"  success_rate={success_rate} episodes={episodes} task_ids={task_ids}")
    log(f"  weights_digest_recomputed_by_eval={recomputed}")
    log(f"  digest_matches_manifest={bool(manifest) and recomputed == manifest.get('weights_digest')}")

    return metrics


# === DOSE-CURVE EVAL LOOP -- active ONLY when .dose_checkpoints/ is present in the model artifact ===
# Absent .dose_checkpoints/ -> single-point eval, BYTE-IDENTICAL to the pre-dose path (the
# `if is_curve:` guard means no dose_curve.json is written). Submit-layer doc:
# src/vla_pipeline/dose_curve.py. NOTE: splitting main() into an importable
# run_single_eval(checkpoint_path) + a thin dose wrapper is a SEPARATE follow-on PR
# -- do NOT do it here; it would touch the frozen
# sagemaker_program=eval_entry.py invocation contract.
def _final_dose_steps(extract_root: str) -> int | None:
    with open(os.path.join(extract_root, "checkpoint_manifest.json")) as source:
        recipe = json.load(source).get("train_recipe") or {}
    step = recipe.get("max_steps")
    if step is None:
        return None
    if type(step) is not int or step < 1:
        raise RuntimeError(f"checkpoint train_recipe.max_steps must be a positive integer: {step!r}")
    return step


def _enumerate_dose_points(extract_root: str) -> list:
    """Single-run dose curve: return [(dose_steps:int, checkpoint_dir:str), ...] =
    each MODEL_DIR/.dose_checkpoints/checkpoint-<step>/ intermediate PLUS the canonical
    root (max dose, path == extract_root). Empty/absent .dose_checkpoints/ -> just the
    root (== today's single eval, byte-unchanged). EVAL_DOSE_STEPS (comma list, or 'all')
    optionally restricts the INTERMEDIATE subset; the canonical root is ALWAYS included.
    Sorted ascending by dose (unknown/-1 doses sort last)."""
    final_dose = _final_dose_steps(extract_root)
    if final_dose is None:
        return [(-1, extract_root)]
    available = {final_dose: extract_root}
    dose_dir = Path(extract_root) / ".dose_checkpoints"
    if dose_dir.exists():
        for checkpoint in sorted(dose_dir.iterdir()):
            if not checkpoint.name.startswith("checkpoint-"):
                continue
            suffix = checkpoint.name.removeprefix("checkpoint-")
            if not suffix.isdecimal() or int(suffix) < 1 or not checkpoint.is_dir():
                raise RuntimeError(f"invalid saved checkpoint: {checkpoint}")
            step = int(suffix)
            if step > final_dose or _final_dose_steps(str(checkpoint)) != step:
                raise RuntimeError(f"saved checkpoint step disagrees with its manifest: {checkpoint}")
            if step != final_dose:
                if step in available:
                    raise RuntimeError(f"duplicate saved checkpoint step: {step}")
                available[step] = str(checkpoint)

    selection = os.environ.get("EVAL_DOSE_STEPS", "all").strip()
    if selection.lower() == "all":
        selected = set(available)
    else:
        tokens = [token.strip() for token in selection.split(",")]
        if any(not token.isdecimal() or int(token) < 1 for token in tokens):
            raise RuntimeError(f"EVAL_DOSE_STEPS must be 'all' or positive integers: {selection!r}")
        selected = {int(token) for token in tokens}
        if len(selected) != len(tokens):
            raise RuntimeError(f"EVAL_DOSE_STEPS contains duplicate steps: {selection!r}")
        missing = selected - available.keys()
        if missing:
            raise RuntimeError(f"EVAL_DOSE_STEPS requests unavailable checkpoints: {sorted(missing)}")
    selected.add(final_dose)
    return [(step, available[step]) for step in sorted(selected)]


def _run_server_and_eval(checkpoint_path: str, port: int, version: str):
    launcher = start_groot_server_n16 if version == "n16" else start_groot_server
    server_log = GR00T_N16_SERVER_LOG if version == "n16" else GR00T_SERVER_LOG
    server_proc = None
    try:
        server_proc = launcher(checkpoint_path, port)
        if not wait_for_server("localhost", port):
            try:
                tail = Path(server_log).read_text()[-3000:]
                log(f"GR00T server log tail:\n{tail}")
            except OSError as exc:
                log(f"Could not read {server_log}: {exc}")
            raise RuntimeError("GR00T server failed to start within timeout")
        # Readiness said something answered. This says it is OURS. Without it a compatible
        # server already on the port serves the whole evaluation and its actions are credited
        # to the checkpoint launched here.
        verify_server_alive(server_proc)
        return run_arena_eval(
            task_name=TASK_NAME, num_envs=NUM_ENVS, num_episodes=EVAL_TRIALS,
            policy_type="gr00t_remote", server_host="localhost", server_port=port,
            checkpoint_path=checkpoint_path,
        )
    finally:
        if server_proc and server_proc.poll() is None:
            log("Stopping GR00T server...")
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if server_proc and server_proc.stdout and not server_proc.stdout.closed:
            server_proc.stdout.close()
        if server_proc and hasattr(server_proc, "_log_file"):
            try:
                server_proc._log_file.close()
            except Exception:
                pass


def ensure_gr00t_venv_n16() -> str:
    """Require the N1.6 GR00T venv baked into the connector image at build time.

    The Dockerfile clones Isaac-GR00T@GR00T_N16_COMMIT (n1.6.1-release), strips
    deepspeed, and runs `uv sync --python 3.12` into GR00T_N16_DIR/.venv/. This
    function validates the venv exists and fails immediately if it does not."""
    if not Path(GR00T_N16_VENV).exists():
        raise RuntimeError(
            f"FATAL: baked N1.6 GR00T venv not found at {GR00T_N16_VENV}. "
            f"The connector image must be rebuilt with the R1 Dockerfile that "
            f"bakes GR00T venvs at build time. Runtime dependency installation "
            f"is no longer supported.")
    log(f"N1.6 GR00T venv present: {GR00T_N16_VENV}")
    return GR00T_N16_VENV


def _read_posctrl_identity() -> str:
    """Bind a completed download record to this repository and requested revision."""
    record = json.loads(Path(N16_POSCTRL_RESOLVED_FILE).read_text())
    if not isinstance(record, dict):
        raise ValueError("positive-control identity must be a JSON object")
    if (record.get("repo") != N16_POSCTRL_CKPT_REPO
            or record.get("requested_rev") != N16_POSCTRL_CKPT_REV):
        raise ValueError("positive-control identity does not match the requested repo/revision")
    commit = record.get("resolved_sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("positive-control identity requires a resolved 40-hex commit")
    return commit


def download_n16_posctrl_ckpt() -> tuple[str, str]:
    """Download the pinned positive-control repo, validate its identity and GR1 layout.

    Returns the checkpoint directory and resolved commit.
    """
    if not N16_POSCTRL_CKPT_REV:
        raise RuntimeError(
            "FATAL: N16_POSCTRL_CKPT_REV is unset. The positive control must state WHICH revision of "
            f"{N16_POSCTRL_CKPT_REPO} it scored; the previous default of 'main' is a moving reference "
            "that was published as resolved_commit, so the report claimed to pin bytes it could not "
            "identify. Pass a tag or commit explicitly.")

    # A NON-EMPTY DIRECTORY IS NOT CACHE EVIDENCE. It says only that something was written there, not
    # that it is this repo at this revision -- a partial download, or a different revision from an
    # earlier run, satisfied it equally. The cache is trusted only when a recorded resolved commit is
    # present; otherwise it is re-downloaded.
    # The record must identify WHICH repo at WHICH requested revision produced these bytes. A
    # 40-hex commit alone does not: a cache written for revision A satisfied a later request for
    # revision B, so the run served A while the report named B -- a wrong attestation from a
    # cache hit. Repo, requested revision and resolved commit are bound together and all three
    # must match before the cache is trusted.
    _cached = None
    if Path(N16_POSCTRL_RESOLVED_FILE).is_file():
        try:
            _cached = _read_posctrl_identity()
        except (OSError, ValueError) as exc:
            log(f"discarding positive-control cache identity: {exc}")
    if _cached:
        log(f"n16 positive-control checkpoint present and identified: {N16_POSCTRL_CKPT_REPO}"
            f"@{N16_POSCTRL_CKPT_REV} -> {_cached}")
        resolved_commit = _cached
    else:
        # The child ALSO resolves the requested revision to its commit sha and writes it beside the
        # download. Resolving in the child is what makes it evidence: it is the same process, and the
        # same API call, that fetched the bytes. Resolving here in the parent would attest a lookup
        # rather than the download.
        _script = (
            "import sys, pathlib, json\n"
            "from huggingface_hub import HfApi, snapshot_download\n"
            "repo, rev, dest, out = sys.argv[1:5]\n"
            "sha = HfApi().model_info(repo, revision=rev).sha\n"
            "p = snapshot_download(repo, revision=sha, local_dir=dest)\n"
            "pathlib.Path(out).write_text(json.dumps("
            "    {'repo': repo, 'requested_rev': rev, 'resolved_sha': sha}))\n"
            "print('downloaded to', p, 'at', sha)\n")
        dl = subprocess.run(
            [GR00T_N16_VENV, "-c", _script,
             N16_POSCTRL_CKPT_REPO, N16_POSCTRL_CKPT_REV, N16_POSCTRL_CKPT_DIR,
             N16_POSCTRL_RESOLVED_FILE],
            capture_output=True, text=True, env=_build_server_env("n16"))
        log(f"n16 ckpt download rc={dl.returncode}: {(dl.stdout or dl.stderr or '')[-400:]}")
        if dl.returncode != 0:
            raise RuntimeError(f"FATAL: failed to download {N16_POSCTRL_CKPT_REPO}")
        if not Path(N16_POSCTRL_RESOLVED_FILE).is_file():
            raise RuntimeError(
                f"FATAL: the download did not record a resolved commit at "
                f"{N16_POSCTRL_RESOLVED_FILE}. Without it the report cannot say which bytes it "
                f"scored, and the requested revision may have been a moving reference.")
        try:
            resolved_commit = _read_posctrl_identity()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"FATAL: invalid positive-control download identity: {exc}") from exc
    # The GN1.6 repo NESTS the checkpoint under a task subdir (ranch_bottle_into_fridge/),
    # and N1.6 checkpoints have NO experiment_cfg/metadata.json (that is N1.7-only; N1.6's
    # experiment_cfg holds conf.yaml/dataset_statistics.json/...). Locate the real checkpoint
    # root = the dir containing config.json (architectures Gr00tN1d6) + embodiment_id.json,
    # and assert the gr1 embodiment via embodiment_id.json.
    ckpt_root = None
    for root, _dirs, files in os.walk(N16_POSCTRL_CKPT_DIR):
        if "config.json" in files and "embodiment_id.json" in files:
            ckpt_root = root
            break
    if ckpt_root is None:
        listing = [str(p) for p in Path(N16_POSCTRL_CKPT_DIR).rglob("*")][:60]
        raise RuntimeError(f"FATAL: no checkpoint root (config.json+embodiment_id.json) under "
                           f"{N16_POSCTRL_CKPT_DIR} (incomplete download?). Tree: {listing}")
    emb_txt = (Path(ckpt_root) / "embodiment_id.json").read_text()
    if "gr1" not in emb_txt.lower():
        raise RuntimeError(f"FATAL: {ckpt_root}/embodiment_id.json does not reference 'gr1'; wrong checkpoint")
    log(f"n16 positive-control checkpoint root: {ckpt_root} @ {resolved_commit} "
        f"(embodiment_id.json references gr1)")
    return ckpt_root, resolved_commit


def start_groot_server_n16(checkpoint_path: str, port: int) -> subprocess.Popen:
    """Launch the N1.6 GR00T server from the N1.6 venv. NATIVE GR1: NO modality-config
    (Gr00tPolicy resolves it from the checkpoint), NO Cosmos, NO --use-sim-policy-wrapper,
    strict on (its own guard). The embodiment tag is HARDCODED to the N1.6 tyro enum NAME
    'GR1' (uppercase) -- N1.6's run_gr00t_server tyro accepts EmbodimentTag member NAMES
    (GR1, UNITREE_G1, ...), NOT the lowercase values our N1.7 server uses (new_embodiment).
    It MUST NOT read SM_HP_EMBODIMENT_TAG (the N1.7 forced-recipe path sets it to
    'new_embodiment'; serving the gr1 checkpoint under that loads the wrong head)."""
    cmd = [
        # Same seeding wrapper as the N1.7 path: the N1.6 server likewise never read
        # EVAL_SEED, and its action head samples inference noise through torch.randn too.
        GR00T_N16_VENV, SEEDED_SERVER_ENTRY,
        "--model-path", checkpoint_path,
        "--embodiment-tag", "GR1",
        "--port", str(port),
    ]
    from checkpoint_compat import require_loadable
    require_loadable(checkpoint_path, "n16", log=log)
    server_env = _build_server_env("n16")
    # I10: a distinct evidence file per launch, cleared first so a launch that
    # writes nothing cannot inherit a previous server's proof.
    _evidence_path = _server_evidence_path(checkpoint_path, port)
    # I3: record the path THIS launch used. Dose points run on SERVER_PORT + idx, so
    # recomputing the path in write_metrics from SERVER_PORT looked up a file that
    # belonged to a different launch, or none at all.
    _LAUNCH_EVIDENCE[os.path.realpath(checkpoint_path)] = _evidence_path
    if os.path.exists(_evidence_path):
        os.remove(_evidence_path)
    server_env["GR00T_SERVER_SEED_EVIDENCE"] = _evidence_path
    log(f"Starting N1.6 GR00T server (isolated venv): {' '.join(cmd)}")
    log_file = open(GR00T_N16_SERVER_LOG, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=server_env)
    proc._log_file = log_file
    log(f"N1.6 GR00T server PID={proc.pid}, log={GR00T_N16_SERVER_LOG}")
    return proc


def main():
    requested_task_ids()
    _version = EVAL_GR00T_VERSION
    _requested_tag = os.environ.get("SM_HP_EMBODIMENT_TAG", "")
    if _version == "n16" and _requested_tag and _requested_tag != "GR1":
        log(f"FATAL: N1.6 server hardcodes --embodiment-tag GR1 but the "
            f"requested tag is {_requested_tag!r}. This would be reported as "
            f"applied but ignored by the server. Reject at runtime to prevent "
            f"a false validation pass.")
        sys.exit(1)
    if (_is_positive_control()
            or os.environ.get("SM_HP_USE_GROOT_SERVER", "false").lower() == "true"):
        required_policy_config()
    # Execution trace. Plain prints, deliberately: `=== INIT` / `>>>` / `<<<` /
    # `=== EXECUTION_SUCCESS` are greppable breadcrumbs that let a human localise a failure
    # in CloudWatch without a debugger. A `>>>` with no matching `<<<` means the run died
    # INSIDE that block. Each `<<<` states WHAT IT PRODUCED, not merely that it passed --
    # which is the only thing that catches an operation returning a PARTIAL result rather
    # than raising.
    log(f"=== INIT: starting Arena eval of {TASK_NAME}, {EVAL_TRIALS} episodes, "
        f"seed={EVAL_SEED}, gr00t={EVAL_GR00T_VERSION}")
    log(f"  EVAL_TRIALS={EVAL_TRIALS} (episodes; budget_type=fixed_trials)")
    log(f"  NUM_ENVS={NUM_ENVS}")
    log(f"  TASK_NAME={TASK_NAME}")
    log(f"  EVAL_SEED={EVAL_SEED}")
    log(f"  MODEL_DIR={MODEL_DIR}")
    log(f"  INPUT_MODEL={INPUT_MODEL}")

    # Accept EULA
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["ACCEPT_EULA"] = "Y"

    # POSITIVE CONTROL (reference policy): serve NVIDIA's published N1.6 GR1 checkpoint through OUR
    # Arena harness to score a known-good policy HIGH on the screened task. Native GR1 (no
    # new_embodiment shim, no Cosmos). The mounted checkpoint is IGNORED; the N1.6 checkpoint is
    # downloaded from HF.
    #
    # Selected BEFORE extraction. This branch used to sit after it, so a diagnostic that ignores the
    # mounted checkpoint still extracted and measured one -- and once extraction began CONSTRUCTING a
    # ResolvedCheckpoint from that archive, a positive control could be REJECTED outright by the
    # identity of a checkpoint it does not use. Doing the work you have decided to ignore is a latent
    # failure waiting for the ignored path to grow a constraint, which is exactly what happened.
    if _is_positive_control():
        log("=== N1.6 POSITIVE-CONTROL mode (native GR1 server) ===")
        ensure_gr00t_venv_n16()
        ckpt, posctrl_commit = download_n16_posctrl_ckpt()
        port = SERVER_PORT
        server_proc = start_groot_server_n16(ckpt, port)
        try:
            if not wait_for_server("localhost", port):
                try:
                    log(f"N1.6 server log tail:\n{Path(GR00T_N16_SERVER_LOG).read_text()[-3000:]}")
                except Exception as _e:
                    log(f"(could not read {GR00T_N16_SERVER_LOG}: {_e!r})")
                raise RuntimeError("N1.6 GR00T server failed to start within timeout")
            verify_server_alive(server_proc)
            results = run_arena_eval(
                task_name=TASK_NAME, num_envs=NUM_ENVS, num_episodes=EVAL_TRIALS,
                policy_type="gr00t_remote", server_host="localhost", server_port=port,
                checkpoint_path=ckpt,
            )
        finally:
            if server_proc and server_proc.poll() is None:
                log("Stopping N1.6 GR00T server...")
                server_proc.send_signal(signal.SIGTERM)
                try:
                    server_proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
            if server_proc and hasattr(server_proc, "_log_file"):
                try:
                    server_proc._log_file.close()
                except Exception:
                    pass
        metrics = write_metrics(results, MODEL_DIR, ckpt, posctrl_commit=posctrl_commit)
        log("=" * 60)
        log(f"DONE (N1.6 positive control) success_rate={metrics['success_rate']}")
        log("=" * 60)
        return

    # Step 1: Extract checkpoint. Reached only in ordinary archive mode: the positive control above
    # returns before this point, because it evaluates a downloaded snapshot and has no mounted archive.
    checkpoint_dir = "/tmp/checkpoint"
    checkpoint_path = extract_checkpoint(INPUT_MODEL, checkpoint_dir)
    log(f"Checkpoint at: {checkpoint_path}")

    # Step 2: Determine eval mode
    use_groot_server = os.environ.get("SM_HP_USE_GROOT_SERVER", "false").lower() == "true"

    if use_groot_server:
        version = EVAL_GR00T_VERSION   # normalized + validated at module import
        dose_points = _enumerate_dose_points(checkpoint_path)
        for _, point in dose_points:
            verify_checkpoint(point)
        if version == "n16":
            ensure_gr00t_venv_n16()
        else:
            ensure_gr00t_venv()
            hf_token = resolve_hf_token()
            precache_cosmos(hf_token)
        is_curve = len(dose_points) > 1
        log(f"dose points to eval ({len(dose_points)}, curve={is_curve}): "
            f"{[(s, os.path.basename(p.rstrip('/')) or 'root') for s, p in dose_points]}")

        curve = []
        results = None  # canonical (final/root) results -> metrics.json
        for idx, (step, cp) in enumerate(dose_points):
            is_canonical = (cp == checkpoint_path)
            port = SERVER_PORT + idx  # fresh port per checkpoint -> avoid TCP TIME_WAIT reuse (EADDRINUSE)
            log(f"--- dose_steps={step} ckpt={cp} canonical={is_canonical} port={port} ---")
            r = _run_server_and_eval(cp, port, version)
            # `ok` used to be hardcoded True, which made it meaningless: a point that
            # produced no usable measurement was still stamped successful, and
            # dose_curve.json gates nothing, so nothing downstream could tell a broken
            # point from a valid one. Derive it from the evidence instead. Reaching here
            # already implies the parser accepted the point (it raises on a zero/short
            # episode count or an out-of-range rate), so this is a second, explicit
            # check: if the invariant ever weakens, the flag says so rather than lying.
            _eps, _sr = r.get("episodes"), r.get("success_rate")
            # I4: realizability was checked for the canonical report but not here, so a dose
            # point recording three episodes at 0.5 -- one and a half successful episodes --
            # was marked ok=True. Same shared helper, so the two cannot disagree.
            _point_successes = realizable_successes(_sr, _eps)
            _ok = (type(_eps) is int and _eps == EVAL_TRIALS
                   and _point_successes is not None)
            _point = {
                "dose_steps": step,
                "success_rate": _sr,
                "episodes": _eps,
                # Arena suffixes per-subtask metrics, so the unsuffixed lookup that used
                # to be here always yielded None even when the simulator reported a
                # value. Read the preserved auxiliary metrics under their real names.
                "revolute_joint_moved_rate_subtask_1": (
                    r.get("aux_metrics", {}).get("revolute_joint_moved_rate_subtask_1")),
                # The authoritative count, so a consumer need not reconstruct it from a
                # floating-point rate. None when the point is not realizable.
                "successes": _point_successes,
                "canonical": is_canonical,
                "ok": _ok,
            }
            if not _ok:
                _point["not_ok_reason"] = (
                    f"episodes={_eps!r} (requested {EVAL_TRIALS}) success_rate={_sr!r}"
                    + ("" if _point_successes is not None
                       else "; rate is not realizable from that episode count"))
            curve.append(_point)
            if is_canonical:
                results = r
        if results is None:
            raise RuntimeError(
                "canonical (final/max-dose) checkpoint eval produced no results -- HARD FAIL")

        # Step 3: canonical metrics.json (schema UNCHANGED) from the final/root checkpoint.
        metrics = write_metrics(results, MODEL_DIR, checkpoint_path)

        # Sidecar dose_curve.json -- ONLY for a real multi-point curve, so single-checkpoint
        # runs (sample/showcase) stay byte-identical. Passes NO gate (sidecar data only).
        if is_curve:
            curve_doc = {
                "schema": "dose_curve/v1",
                "task": TASK_NAME,
                "num_episodes": EVAL_TRIALS,
                "budget_type": "fixed_trials",
                "eval_seed": int(os.environ.get("EVAL_SEED", str(EVAL_SEED))),
                "suite": os.environ.get("EVAL_SUITE", "arena_gr1"),
                "canonical_metrics": "metrics.json",
                "points": sorted(curve, key=lambda p: p["dose_steps"]),
            }
            with open(Path(MODEL_DIR) / "dose_curve.json", "w") as fh:
                json.dump(curve_doc, fh, indent=2)
            log(f"wrote dose_curve.json: {len(curve)} points, "
                f"{sum(1 for p in curve if p['ok'])} ok")
    else:
        # POC mode: zero-action eval (validates Isaac Sim runs on SageMaker). Unchanged.
        log("Running in POC mode (zero_action) - validates Isaac Sim on SageMaker")
        results = run_zero_action_eval(
            task_name=TASK_NAME,
            num_envs=NUM_ENVS,
            num_episodes=EVAL_TRIALS,
        )
        metrics = write_metrics(results, MODEL_DIR, checkpoint_path)

    log(f"=== EXECUTION_SUCCESS: all code executed -- success_rate={metrics['success_rate']}, "
        f"episodes={metrics.get('episodes')}, "
        f"tasks={sorted({row['task'] for row in metrics.get('per_task', [])})}, "
        f"metrics written to {MODEL_DIR}/metrics.json")


if __name__ == "__main__":
    main()
