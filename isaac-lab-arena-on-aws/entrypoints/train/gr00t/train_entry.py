"""SageMaker entry point: GR00T N1.6/N1.7 fine-tuning for LIBERO and Arena.

Fine-tune the version selected by GR00T_VERSION using NVIDIA's pinned
examples/finetune.sh recipe. defaults.json supplies the upstream revision and
base checkpoint; the suite maps select the dataset and embodiment.
Write checkpoint weights, input configuration and training lineage to
SM_MODEL_DIR for downstream evaluation and validation. Gated model downloads
require Hugging Face authentication.

Config via environment (set by the launcher):
  TRAIN_SUITE       selects a dataset from SUITE_DATASETS (LIBERO or Arena)
  TRAIN_MAX_STEPS   fine-tune steps (default 4000; launcher supplies its budget)
  TRAIN_SEED        only 42 is accepted, matching the pinned trainer's fixed seed;
                    this wrapper does not pass a seed argument
  TRAIN_NUM_GPUS    override GPU count (default SM_NUM_GPUS from resource config)
  TRAIN_BATCH_SIZE  global batch size (default 64; scaled for fewer GPUs)

Installation or training failure exits nonzero.
"""
from __future__ import annotations

import glob
import json
import os
import re
import threading
import signal
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digest import weights_digest  # noqa: E402  (shipped in sourcedir)
from training_lineage import write_training_lineage

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
WORK = "/opt/ml/code"
_BAKED_MARKER = os.path.isfile("/opt/vla/.baked_env")
_BAKED = _BAKED_MARKER
if _BAKED:
    WORK = "/opt/vla"
GR00T_DIR = os.path.join(WORK, os.environ.get("GROOT_SUBDIR", "gr00t" if _BAKED else "Isaac-GR00T"))
UV_BIN = "uv"
_defaults_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "defaults.json")
with open(_defaults_path) as _fh:
    _D = json.load(_fh)
GR00T_REPO = _D["repo"]
# --- GR00T version selector (n16 matches the NVIDIA reference on N1.6+GR1) ------
# GR00T_VERSION in {n17 (default, UNTOUCHED), n16}. n16 -> native GR00T-N1.6-3B @
# 5dc80c4 (n1.6.1-release), EmbodimentTag.GR1. Same finetune.sh CLI interface
# (verified against both pinned sources), so only base/commit/revision + the
# embodiment/modality-config differ; the train command construction is unchanged.
GR00T_VERSION = os.environ.get("GR00T_VERSION", "n17").strip().lower()
if GR00T_VERSION not in ("n16", "n17"):
    print(f"[gr00t-train] FATAL: GR00T_VERSION={GR00T_VERSION!r} not in "
          "{'n16','n17'}", flush=True)
    sys.exit(1)
if GR00T_VERSION == "n16":
    GR00T_COMMIT = _D["repo_commit_n16"]
    BASE_MODEL = _D["base_checkpoint_n16"]
    BASE_REVISION = _D["base_checkpoint_hf_revision_n16"]
    if _BAKED:
        _baked_content = ""
        try:
            with open("/opt/vla/.baked_env") as _bf:
                _baked_content = _bf.read().strip()
        except OSError:
            pass
        if GR00T_COMMIT not in _baked_content:
            _BAKED = False
            WORK = "/opt/ml/code"
            GR00T_DIR = os.path.join(WORK, "Isaac-GR00T")
            print(f"[gr00t-train] N1.6 selected but baked env is {_baked_content} "
                  f"(need {GR00T_COMMIT}) -- runtime install", flush=True)
else:
    GR00T_COMMIT = _D["repo_commit"]
    BASE_MODEL = _D["base_checkpoint"]              # 3B foundation we fine-tune
    BASE_REVISION = _D["base_checkpoint_hf_revision"]  # its OWN pinned sha (NOT the LIBERO ckpt's)
# Isolated FFmpeg-7 prefix for torchcodec 0.8.0 (supports FFmpeg 4-7). A dedicated
# conda prefix avoids the DLC base env's stale ffmpeg-5 dependency graph (which
# made libtorchcodec_core5.so fail to load). ffmpeg 7 -> libavcodec.so.61 matches
# libtorchcodec_core7.so (tried first).
# Compute after version selection, which can switch away from the baked workspace.
FFMPEG_PREFIX = os.path.join(WORK, "ffmpeg7")
EMBODIMENT_TAG = _D["embodiment_tag"]
PYTHON_PIN = "3.12"
N_ACTION_STEPS = "8"
MAX_EPISODE_STEPS = "720"

# TODO(port): these two maps duplicate config/suites/*.yaml. They live here
# because this file ships in the SageMaker source_dir and runs where config/ is
# absent; tests/test_suites.py CI-pins them to the manifests. Rewire to a staged
# suites.json alongside a real training run. See README.md#notes-and-limitations.
SUITE_DATASETS = {
    "libero_spatial": "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object": "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
    "arena_gr1": "nvidia/Arena-GR1-Manipulation-Task",
    "arena_gr1_fridge": "nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task",
    "arena_g1": "nvidia/Arena-G1-Loco-Manipulation-Task",
}

# Per-suite dataset-delivery config. LIBERO datasets live at the repo root and
# REQUIRE NVIDIA's examples/LIBERO/modality.json copied into meta/. The Arena GR1
# dataset ships pre-converted GR00T-LeRobot under a `lerobot/` subfolder at a
# pinned revision and carries its OWN meta/modality.json (GR1 arms/hands/waist
# mapping) -- must NOT be overwritten. gr00t@376ba890 has no bundled `gr1` tag, so
# Arena GR1 fine-tunes under the custom-embodiment slot `new_embodiment`
# (+ optional MODALITY_CONFIG_PATH env for the Arena gr1_arms_only data config).
SUITE_META = {
    "libero_spatial": {"subdir": "", "revision": None, "copy_libero_modality": True, "embodiment_tag": None},
    "libero_object": {"subdir": "", "revision": None, "copy_libero_modality": True, "embodiment_tag": None},
    "arena_gr1": {"subdir": "lerobot", "revision": "arena_v0.2_lab_v3.0", "copy_libero_modality": False, "embodiment_tag": "new_embodiment"},
    "arena_gr1_fridge": {"subdir": "ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot", "revision": "arena_v0.2_lab_v3.0", "copy_libero_modality": False, "embodiment_tag": "new_embodiment"},
    "arena_g1": {"subdir": "lerobot", "revision": None, "copy_libero_modality": False, "embodiment_tag": "new_embodiment"},
}


def _require_int(name: str, value: str, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except ValueError:
        print(f"[gr00t-train] FATAL: {name} not an integer: {value!r}",
              flush=True)
        sys.exit(1)
    if not (lo <= n <= hi):
        print(f"[gr00t-train] FATAL: {name}={n} outside [{lo},{hi}]",
              flush=True)
        sys.exit(1)
    return n


SUITE = os.environ.get("TRAIN_SUITE", "libero_spatial")
if SUITE not in SUITE_DATASETS:
    print(f"[gr00t-train] FATAL: TRAIN_SUITE={SUITE!r} not in "
          f"{sorted(SUITE_DATASETS)}", flush=True)
    sys.exit(1)

# Per-suite embodiment-tag override: Arena GR1 trains under the custom-embodiment
# slot `new_embodiment` (gr00t@376ba890 has no bundled `gr1` tag); LIBERO keeps
# the defaults.json tag (LIBERO_PANDA).
if SUITE_META[SUITE].get("embodiment_tag"):
    EMBODIMENT_TAG = SUITE_META[SUITE]["embodiment_tag"]

# N1.6 has the NATIVE pretrained GR1 head. N1.6's launch_finetune tyro CLI accepts
# the EmbodimentTag enum NAME "GR1" (uppercase) -- NOT the value "gr1" (verified:
# "invalid choice: 'gr1' (choose from ...'GR1'...)"). The eval server likewise uses
# the NAME (start_groot_server_n16 hardcodes GR1). Internally tyro maps GR1 ->
# EmbodimentTag.GR1 -> .value "gr1", matching the modality config key + dataset tag.
if GR00T_VERSION == "n16" and SUITE in ("arena_gr1", "arena_gr1_fridge"):
    EMBODIMENT_TAG = "GR1"

# LIBERO consumes these protocol settings from the checkpoint. Arena uses its
# policy YAML instead; do not label an Arena checkpoint with LIBERO's limits.
INPUT_CONFIG = {"embodiment_tag": EMBODIMENT_TAG}
if SUITE.startswith("libero_"):
    INPUT_CONFIG.update(n_action_steps=N_ACTION_STEPS, max_episode_steps=MAX_EPISODE_STEPS)

# train_steps (from pipeline hyperparameter) overrides TRAIN_MAX_STEPS
_max_steps_raw = os.environ.get("TRAIN_MAX_STEPS", "4000")
MAX_STEPS = _require_int("TRAIN_MAX_STEPS", _max_steps_raw,
                         1, 1_000_000)
# TRAIN_SAVE_STEPS: checkpoint save interval. Default = MAX_STEPS (one final
# checkpoint = the historical sample/final-only behavior, byte-unchanged). For the
# CONTROLLED single-run dose curve, set it to an interval (e.g. 25) so ONE training
# trajectory saves intermediate checkpoints {SAVE_STEPS, 2*SAVE_STEPS, ...}; the eval
# loop then rolls out a chosen subset -> dose_curve.json (see steps/isaac_arena/eval_entry.py).
SAVE_STEPS = _require_int("TRAIN_SAVE_STEPS",
                          os.environ.get("TRAIN_SAVE_STEPS", str(MAX_STEPS)),
                          1, 1_000_000)
TRAIN_SEED = _require_int("TRAIN_SEED",
                          os.environ.get("TRAIN_SEED", "42"),
                          0, 2**31 - 1)
if TRAIN_SEED != 42:
    print(f"[gr00t-train] FATAL: TRAIN_SEED={TRAIN_SEED} but GR00T's finetune.sh "
          "does not accept a seed argument. Only seed=42 (the upstream default) is "
          "supported. Recording a different seed without applying it is misleading.",
          flush=True)
    sys.exit(1)
_sm_num_gpus = os.environ.get("SM_NUM_GPUS", "4")
NUM_GPUS = _require_int("TRAIN_NUM_GPUS",
                        os.environ.get("TRAIN_NUM_GPUS", _sm_num_gpus),
                        1, 8)
# For a sample run (<=10 steps), use single GPU to avoid multi-GPU coordination overhead
if int(MAX_STEPS) <= 10:
    NUM_GPUS = 1
# Single-run dose curve (SAVE_STEPS < MAX_STEPS): FORCE single-GPU so each
# checkpoint-<step>/ is a consolidated HF checkpoint. Multi-GPU training writes
# DeepSpeed ZeRO shards per step dir (experiment.py: deepspeed_config=None ONLY when
# num_gpus==1), which AutoModel.from_pretrained cannot load -> would break the
# per-checkpoint eval loop. This also fixes the world size across the whole curve
# (one run, one trajectory). Supersedes an earlier design decision (drop the
# <=10 special-case): the single-run design has NO multi-job sweep to make
# GPU-count-consistent, and the genuine requirement is loadable intermediate
# checkpoints -- which single-GPU guarantees. (main() logs the resulting num_gpus.)
# === DOSE-CURVE LAYER (1/3, train-time) -- active ONLY when SAVE_STEPS < MAX_STEPS ===
# Force single-GPU so each intermediate checkpoint-<step>/ is a standalone HF checkpoint
# (multi-GPU ZeRO shards are not loadable by from_pretrained). Submit-layer mirror /
# doc: src/vla_pipeline/dose_curve.py (DoseCurveConfig.validate_preconditions).
if SAVE_STEPS < MAX_STEPS:
    NUM_GPUS = 1
BATCH_SIZE = _require_int("TRAIN_BATCH_SIZE",
                          os.environ.get("TRAIN_BATCH_SIZE", "64"),
                          1, 1280)
DATASET_REPO = SUITE_DATASETS[SUITE]
_DATASET_S3_URI = os.environ.get("TRAIN_DATASET_S3URI", "").strip()
if _DATASET_S3_URI and _DATASET_S3_URI != "__LIBERO_DEFAULT__":
    print(f"[gr00t-train] FATAL: GR00T does not support BYO S3 datasets. "
          f"TRAIN_DATASET_S3URI={_DATASET_S3_URI!r} was provided but GR00T always "
          f"trains on the HuggingFace dataset from the suite manifest "
          f"({DATASET_REPO}). Remove DatasetS3Uri or use the default sentinel.",
          flush=True)
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[gr00t-train] {msg}", flush=True)


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
    kw.setdefault("check", True)
    # A call with no timeout can hang for the whole budget and produce nothing; the step then looks
    # like a long job rather than a stuck one.
    kw.setdefault("timeout", _subprocess_timeout())
    subprocess.run(cmd, **kw)


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    # PyPI 502 resilience for uv (default 3 retries in ~4s is too short for a
    # transient files.pythonhosted.org 502 on a big CUDA wheel -- observed).
    env.setdefault("UV_HTTP_TIMEOUT", "120")
    env.setdefault("UV_HTTP_RETRIES", "10")
    # torchcodec (in the uv .venv) dynamically loads FFmpeg's libav* shared libs.
    # Point LD_LIBRARY_PATH at our ISOLATED ffmpeg-7 prefix (built in main) so the
    # venv binds libavcodec.so.61 etc -- NOT the DLC's system FFmpeg 8 (which
    # torchcodec 0.8.0 cannot use) nor the base env's stale ffmpeg 5. torchrun
    # workers inherit this env (normal fork/exec; nothing sanitizes it).
    _ff_lib = os.path.join(FFMPEG_PREFIX, "lib")
    _prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = _ff_lib + (os.pathsep + _prev if _prev else "")
    return env


def uv_sync_resilient(sync_args: list, env: dict, cwd: str) -> None:
    """Run `uv sync` with process-level retry + backoff on top of UV_HTTP_RETRIES.
    uv sync is resumable (reuses the cache), so a retry only re-fetches what a
    transient PyPI 502 dropped. Fail-closed after the last attempt."""
    delay = 30
    for attempt in range(1, 6):
        out = subprocess.run([UV_BIN, *sync_args], cwd=cwd, env=env)
        if out.returncode == 0:
            if attempt > 1:
                log(f"uv sync OK on attempt {attempt}")
            return
        if attempt < 5:
            log(f"uv sync failed (rc={out.returncode}); transient PyPI blip? "
                f"backing off {delay}s (attempt {attempt}/5)")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    log("FATAL: uv sync failed after 5 attempts")
    sys.exit(1)


def main() -> None:
    # PyPI 502 resilience for EVERY pip in this job (incl. build-isolation
    # subprocess pips that ignore an explicit --retries flag).
    os.environ["PIP_RETRIES"] = "10"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "60"
    log(f"suite={SUITE} max_steps={MAX_STEPS} seed={TRAIN_SEED} "
        f"num_gpus={NUM_GPUS} batch={BATCH_SIZE} dataset={DATASET_REPO}")
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

    hf_home = os.path.join(WORK, "hf-cache")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        # resolve from Secrets Manager if HF_SECRET_NAME is set
        secret_name = os.environ.get("HF_SECRET_NAME", "")
        if secret_name:
            import boto3
            region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            client = boto3.client("secretsmanager", region_name=region)
            hf_token = client.get_secret_value(SecretId=secret_name)["SecretString"].strip()
            os.environ["HF_TOKEN"] = hf_token
            log(f"Resolved HF token from secret '{secret_name}'")
    if not hf_token:
        log("FATAL: HF_TOKEN required (gated Cosmos-Reason2-2B backbone)")
        sys.exit(1)

    # ---- [1/7] System libs (same as eval_entry: EGL + build tooling)
    os.chmod("/tmp", 0o1777)

    # --- Idempotence: if prebuilt image has the env, skip sections 1-3 ---
    # Use the module-level _BAKED which accounts for version mismatches
    # (e.g. baked N1.7 but N1.6 requested -> _BAKED=False at import time).
    _is_baked = _BAKED
    if _is_baked:
        with open("/opt/vla/.baked_env") as f:
            log(f"BAKED ENV DETECTED: {f.read().strip()} -- skipping install (sections 1-3)")
    else:
        log("No baked env (or version mismatch) -- full runtime install")

    if not _is_baked:
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "--no-install-recommends",
         "git", "git-lfs", "curl", "ca-certificates", "libegl1", "libgles2",
         "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
         "build-essential", "cmake"])

        conda_bin = os.path.join(os.path.dirname(sys.executable), "conda")
        run([conda_bin, "create", "-y", "--override-channels", "-c", "conda-forge",
             "-p", FFMPEG_PREFIX, "ffmpeg=7.*"])
        _avcodec61 = os.path.join(FFMPEG_PREFIX, "lib", "libavcodec.so.61")
        if not os.path.isfile(_avcodec61):
            log(f"FATAL: FFmpeg 7 ABI missing after conda solve: {_avcodec61}")
            sys.exit(1)
        run([os.path.join(FFMPEG_PREFIX, "bin", "ffmpeg"), "-version"])

    env = clean_env()

    if not _is_baked:
        # ---- [2/7] uv (pinned) + pinned GR00T clone (LFS wheel materialized)
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--retries", "10", "--timeout", "60", "uv==0.12.1"])
        global UV_BIN
        UV_BIN = os.path.join(os.path.dirname(sys.executable), "uv")
        run([UV_BIN, "--version"])
        run(["git", "lfs", "install", "--skip-repo"])
        clone_env = dict(env, GIT_LFS_SKIP_SMUDGE="1")
        run(["git", "clone", GR00T_REPO, GR00T_DIR], env=clone_env)
        run(["git", "-C", GR00T_DIR, "checkout", GR00T_COMMIT], env=clone_env)
        head = subprocess.run(["git", "-C", GR00T_DIR, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
        if head != GR00T_COMMIT:
            log(f"FATAL: repo commit mismatch: {head}")
            sys.exit(1)
        run(["git", "-C", GR00T_DIR, "lfs", "pull",
             "--include", "scripts/deployment/**/wheels/*.whl"], env=env)

        # ---- [3/7] Server venv: gr00t package at pinned python 3.12
        uv_sync_resilient(["sync", "--python", PYTHON_PIN], env=env, cwd=GR00T_DIR)
    else:
        if not os.path.isdir(GR00T_DIR):
            log(f"FATAL: baked marker present but {GR00T_DIR} missing")
            sys.exit(1)
        UV_BIN = "uv"

    # Smoke-test torchcodec's NATIVE loader with the same interpreter + env the
    # training will use -- fail NOW (before the ~30min dataset+base download) if
    # the ffmpeg-7 linkage is wrong, instead of deep in the DataLoader.
    _venv_py = os.path.join(GR00T_DIR, ".venv", "bin", "python")
    run([_venv_py, "-c",
         "import torchcodec; from torchcodec.decoders import VideoDecoder; "
         "print('torchcodec native loader OK:', torchcodec.__version__)"],
        cwd=GR00T_DIR, env=env)

    # ---- [4/7] Download the LIBERO LeRobot dataset from HuggingFace.
    # HF throttles at 1000 API requests / 5 min; a bare snapshot_download of this
    # many-file dataset can trip a 429 (observed on the first spike). Mitigate:
    # (a) cap concurrency (max_workers=4) so we issue far fewer parallel requests,
    # (b) bounded retry with exponential backoff INSIDE the download process, and
    # (c) resume across attempts (snapshot_download is resumable, so a retry only
    # fetches what's missing). Authenticated (HF_TOKEN in env) also raises limits.
    _subdir = SUITE_META[SUITE].get("subdir", "") or ""
    _env_rev = os.environ.get("TRAIN_DATASET_REVISION", "").strip()
    if _env_rev and _env_rev != "__FROM_SUITE_MANIFEST__":
        _revision = _env_rev
    else:
        _revision = SUITE_META[SUITE].get("revision") or ""
    _dl_dst = os.path.join(WORK, "lerobot-data", DATASET_REPO)
    # Datasets that ship the LeRobot tree under a subfolder (Arena GR1 -> lerobot/)
    # download ONLY that subfolder (allow_patterns) at the pinned revision; the
    # LeRobot root finetune consumes is dst/<subdir>.
    data_root = os.path.join(_dl_dst, _subdir) if _subdir else _dl_dst
    os.makedirs(data_root, exist_ok=True)
    log(f">>> dataset_download {DATASET_REPO} subdir={_subdir!r} "
        f"rev={_revision!r} (completeness-gated)")
    # CRITICAL: on a 429, snapshot_download SWALLOWS the error and RETURNS the
    # partial local_dir as "success" ("Returning existing local_dir ... as remote
    # repo cannot be accessed"). That silently yielded an INCOMPLETE video set ->
    # torchcodec "No valid stream found" mid-training. So we do NOT trust its
    # return: we loop until the on-disk .mp4 count equals the repo's expected
    # count (from the HF file listing), retrying on 429 or incomplete. Resumable,
    # so each pass only fetches what's missing. Fail-closed if never complete.
    _dl_cmd = [_venv_py, "-c"] if _is_baked else [UV_BIN, "run", "--python", PYTHON_PIN, "python", "-c"]
    _download_identity_path = os.path.join(WORK, "dataset-download-identity.json")
    run([*_dl_cmd,
         "import sys, time, glob, os, json\n"
         "from huggingface_hub import snapshot_download, HfApi\n"
         "repo, dst, rev, sub = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]\n"
         "expected_mp4 = None\n"
         "resolved = None\n"
         "for attempt in range(1, 13):\n"
         "    try:\n"
         "        if expected_mp4 is None:\n"
         "            resolved = HfApi().repo_info(repo, repo_type='dataset', revision=(rev or None)).sha\n"
         "            files = HfApi().list_repo_files(repo, repo_type='dataset', revision=resolved)\n"
         "            expected_mp4 = sum(1 for f in files if f.endswith('.mp4') and (not sub or f.startswith(sub + '/')))\n"
         "            print('expected mp4 count:', expected_mp4, flush=True)\n"
         "        snapshot_download(repo, repo_type='dataset', local_dir=dst,\n"
         "                          max_workers=2, revision=resolved,\n"
         "                          allow_patterns=([sub + '/**'] if sub else None))\n"
         "    except Exception as e:\n"
         "        if '429' in str(e) and attempt < 12:\n"
         "            print('HF 429, backoff 90s (attempt %d/12)' % attempt, flush=True)\n"
         "            time.sleep(90); continue\n"
         "        raise\n"
         "    have = len(glob.glob(os.path.join(dst, '**', '*.mp4'), recursive=True))\n"
         "    if expected_mp4 and have >= expected_mp4:\n"
         "        with open(sys.argv[5], 'w') as identity:\n"
         "            json.dump({'repo_id': repo, 'requested_revision': rev, 'resolved_revision': resolved, 'subdirectory': sub}, identity)\n"
         "        print('dataset COMPLETE: %d/%d mp4 on attempt %d'\n"
         "              % (have, expected_mp4, attempt)); break\n"
         "    print('INCOMPLETE: %d/%d mp4, retrying after 90s (attempt %d/12)'\n"
         "          % (have, expected_mp4 or -1, attempt), flush=True)\n"
         "    time.sleep(90)\n"
         "else:\n"
         "    raise SystemExit('dataset never reached complete mp4 count')",
         DATASET_REPO, _dl_dst, _revision, _subdir, _download_identity_path],
        cwd=GR00T_DIR, env=env)
    if not os.path.isdir(data_root):
        log(f"FATAL: dataset download failed: {data_root} missing")
        sys.exit(1)

    # Count episodes from the dataset metadata (LeRobot meta/info.json)
    episode_count = None
    info_candidates = [
        os.path.join(data_root, "meta", "info.json"),
        os.path.join(data_root, "info.json"),
    ]
    for info_path in info_candidates:
        if os.path.isfile(info_path):
            with open(info_path) as fh:
                info = json.load(fh)
            episode_count = info.get("total_episodes") or info.get("num_episodes")
            if isinstance(episode_count, int) and episode_count >= 1:
                break
            episode_count = None
    if episode_count is None:
        log("counting episodes from parquet metadata fallback...")
        parquet_files = glob.glob(os.path.join(data_root, "**", "*.parquet"),
                                  recursive=True)
        episode_files = [f for f in parquet_files if "episode" in os.path.basename(f).lower()]
        if episode_files:
            episode_count = len(episode_files)
        else:
            log("FATAL: cannot determine episode_count from dataset metadata")
            sys.exit(1)
    # NVIDIA's LIBERO fine-tune REQUIRES their shipped modality.json copied into
    # the dataset's meta/ (maps LeRobot fields -> GR00T modalities). Without it
    # the loader fails (FileNotFoundError meta/modality.json or a modality-config
    # error). Copy it BEFORE the dataset digest so the digest binds what we feed.
    import shutil as _shutil
    if SUITE_META[SUITE].get("copy_libero_modality", False):
        modality_src = os.path.join(GR00T_DIR, "examples", "LIBERO", "modality.json")
        if not os.path.isfile(modality_src):
            log(f"FATAL: NVIDIA LIBERO modality.json missing at {modality_src}")
            sys.exit(1)
        modality_dst = os.path.join(data_root, "meta", "modality.json")
        os.makedirs(os.path.dirname(modality_dst), exist_ok=True)
        _shutil.copy2(modality_src, modality_dst)
        log(f"copied NVIDIA modality.json -> {modality_dst}")
    else:
        # Non-LIBERO suites (Arena GR1) ship their OWN meta/modality.json (the
        # GR1 arms/hands mapping) -- must NOT be overwritten. Assert it's present.
        own_modality = os.path.join(data_root, "meta", "modality.json")
        if not os.path.isfile(own_modality):
            log(f"FATAL: dataset's own meta/modality.json missing at {own_modality}")
            sys.exit(1)
        log(f"using dataset's own modality.json at {own_modality} (no overwrite)")

    # DECLARED versus VERIFIED. episode_count comes from the dataset's own info.json -- the
    # download vouching for its own completeness -- so it cannot detect a tree that has the
    # metadata but is missing episode data. snapshot_download can be throttled into returning a
    # PARTIAL tree without raising (see the 429 note above), and the child's completeness gate
    # counts MP4s only, so a tree with every video and no parquet passes it.
    #
    # So count the data files HERE, independently, and require them. A declared count with no
    # parquet behind it is the failure this exists to catch.
    _parquet = glob.glob(os.path.join(data_root, "**", "*.parquet"), recursive=True)
    _mp4 = glob.glob(os.path.join(data_root, "**", "*.mp4"), recursive=True)
    _empty = [p for p in _parquet if os.path.getsize(p) == 0]
    if not _parquet:
        log(f"<<< dataset_download FAIL: {data_root} declares "
            f"episode_count={episode_count} but contains NO parquet episode data. The tree has "
            f"{len(_mp4)} mp4 file(s); a video-only tree cannot train.")
        sys.exit(1)
    if _empty:
        log(f"<<< dataset_download FAIL: {len(_empty)} of {len(_parquet)} parquet file(s) are "
            f"zero bytes under {data_root}, e.g. {_empty[:3]}. A truncated download reports "
            f"success.")
        sys.exit(1)
    log(f"<<< dataset_download OK: declared_episode_count={episode_count} (from the dataset's "
        f"own info.json), verified {len(_parquet)} parquet + {len(_mp4)} mp4 file(s) present and "
        f"non-empty for {DATASET_REPO} under {data_root}")
    dataset_source = f"hf:{DATASET_REPO}"
    dataset_revision = weights_digest(data_root)
    with open(_download_identity_path) as identity_file:
        download_identity = json.load(identity_file)
    for key, expected in {
        "repo_id": DATASET_REPO, "requested_revision": _revision, "subdirectory": _subdir,
    }.items():
        if download_identity.get(key) != expected:
            raise RuntimeError(f"Downloaded dataset identity differs for {key}")
    lineage_args = dict(
        train_suite=SUITE, source_parameter=_DATASET_S3_URI,
        revision_parameter=_env_rev, repo_id=DATASET_REPO,
        requested_revision=_revision,
        resolved_revision=download_identity["resolved_revision"],
        subdirectory=_subdir, content_digest=dataset_revision,
    )

    # ---- [4b/7] Materialize the 3B FOUNDATION base locally at its pinned sha.
    # finetune.sh -> launch_finetune.py loads --base-model-path via HF and failed
    # on the bare repo id ("does not appear to have files named
    # model-00001-of-00002.safetensors") -- pass a LOCAL dir at the correct sha
    # (BASE_REVISION is now the 3B's own sha, not the LIBERO checkpoint's).
    base_local = os.path.join(WORK, "base", BASE_MODEL.split("/")[-1])
    log(f">>> base_materialize {BASE_MODEL}@{BASE_REVISION[:12]} -> {base_local}")
    # 429-hardened: this 1301-file base pull trips HF's 1000-req/5min quota
    # (observed). snapshot_download is resumable, so retries only fetch what's
    # missing. max_workers caps parallel requests; backoff waits out the window.
    # Cool-down first: the dataset pull just above (1301 files) plus this base
    # pull (1301 files) together exceed HF's 1000-req/5min quota. Wait out the
    # window before starting so the base pull begins with fresh budget.
    log("cooling down 120s before base pull (HF quota window reset)")
    time.sleep(120)
    # Retry catches ANY exception whose message mentions 429 -- snapshot_download
    # re-raises the model_info 429 as LocalEntryNotFoundError (NOT HfHubHTTPError),
    # so a type-specific except missed it on the prior run. snapshot_download is
    # resumable, so each retry only fetches the remaining files.
    _base_cmd = [_venv_py, "-c"] if _is_baked else [UV_BIN, "run", "--python", PYTHON_PIN, "python", "-c"]
    run([*_base_cmd,
         "import sys, time\n"
         "from huggingface_hub import snapshot_download\n"
         "repo, rev, dst = sys.argv[1], sys.argv[2], sys.argv[3]\n"
         "delay = 60\n"
         "for attempt in range(1, 9):\n"
         "    try:\n"
         "        snapshot_download(repo, revision=rev, local_dir=dst,\n"
         "                          max_workers=2)\n"
         "        print('base materialized on attempt', attempt); break\n"
         "    except Exception as e:\n"
         "        if '429' in str(e) and attempt < 8:\n"
         "            print('HF 429 on base pull, backoff %ds (%d/8)' % (delay, attempt),\n"
         "                  flush=True)\n"
         "            time.sleep(delay); delay = min(delay * 2, 300); continue\n"
         "        raise\n"
         "else:\n"
         "    raise SystemExit('base download failed after 8 attempts')",
         BASE_MODEL, BASE_REVISION, base_local], cwd=GR00T_DIR, env=env)
    _required = ["model-00001-of-00002.safetensors",
                 "model-00002-of-00002.safetensors",
                 "model.safetensors.index.json", "config.json"]
    _missing = [p for p in _required
                if not os.path.isfile(os.path.join(base_local, p))]
    if _missing:
        log(f"<<< base_materialize FAIL: snapshot at {base_local} missing {_missing} -- GR00T "
            "source/model-release mismatch; do NOT substitute the LIBERO ckpt")
        sys.exit(1)
    _base_files = sum(len(f) for _, _, f in os.walk(base_local))
    log(f"<<< base_materialize OK: {_base_files} files at {base_local}, "
        f"{BASE_MODEL}@{BASE_REVISION[:12]}, all {len(_required)} required present")

    # ---- [5/7] Run NVIDIA's fine-tune via examples/finetune.sh
    # FLAG: The Python entrypoint behind finetune.sh is not spike-confirmed.
    # Using the documented shell interface with env vars.
    # DISK FIX (a multi-checkpoint run exhausted the overlay partition at the 3rd checkpoint, ~step 750):
    # a multi-checkpoint dose-curve run keeps N checkpoint dirs under train_out.
    # DF-PROBE VERIFIED: on SageMaker NVMe
    # instances (g5/g6e) the instance-store NVMe is mounted at /tmp, /opt/ml/output,
    # /opt/ml/model (229GB on g5.xlarge, ~1.9TB on g6e.16xl). /opt/ml ITSELF and any
    # arbitrary subdir (e.g. /opt/ml/train-out) are on the SMALL container OVERLAY
    # (120GB total, ~71GB free) -- as is WORK=/opt/vla. Two prior runs ENOSPC'd at
    # checkpoint 3 (~70GB) because train_out was on that overlay. VolumeSizeInGB is
    # IGNORED on NVMe instances (fixed instance-store size). FIX: write train_out to
    # /tmp (NVMe scratch, NOT auto-uploaded); staging copies the final + weights-only
    # intermediates into MODEL_DIR (/opt/ml/model, also NVMe) for the eval loop.
    _scratch_base = "/tmp" if os.path.isdir("/tmp") else WORK
    train_out = os.path.join(_scratch_base, "train-out")
    os.makedirs(train_out, exist_ok=True)

    finetune_sh = os.path.join(GR00T_DIR, "examples", "finetune.sh")
    if not os.path.isfile(finetune_sh):
        log(f"FATAL: finetune.sh not found at {finetune_sh}")
        sys.exit(1)

    # SPIKE-CONFIRMED (finetune.sh @ pinned commit 376ba890): the four REQUIRED
    # args (base-model-path, dataset-path, embodiment-tag, output-dir) are CLI
    # FLAGS, not env vars -- passing OUTPUT_DIR/DATA_PATH as env was silently
    # ignored ("Missing required argument: DATASET_PATH"). Tunables (max/num-gpus/
    # batch/save-steps) ARE env vars the script converts to --flags. WANDB via
    # USE_WANDB. NOTE: NVIDIA's finetune.sh + launch_finetune.py expose NO seed
    # argument -- TRAIN_SEED is NOT honorable here; we record this as a
    # reproducibility caveat rather than fake a seed we cannot set.
    train_env = dict(env)
    train_env["MAX_STEPS"] = str(MAX_STEPS)
    train_env["NUM_GPUS"] = str(NUM_GPUS)
    train_env["GLOBAL_BATCH_SIZE"] = str(BATCH_SIZE)
    train_env["SAVE_STEPS"] = str(SAVE_STEPS)   # =MAX_STEPS (one final ckpt) unless a dose-curve interval is set
    train_env["USE_WANDB"] = "0"
    train_env["HF_TOKEN"] = hf_token
    train_env["WANDB_MODE"] = "offline"
    # ACTIVATE the uv venv for finetune.sh: the script calls bare `torchrun`/
    # `python`, which otherwise resolve to the DLC's conda 3.10 (where the gr00t
    # deps -- tyro etc -- are NOT installed -> "ModuleNotFoundError: No module
    # named 'tyro'"). uv sync installed everything into GR00T_DIR/.venv (3.12);
    # put it first on PATH and set VIRTUAL_ENV so the subprocess uses it.
    venv_dir = os.path.join(GR00T_DIR, ".venv")
    venv_bin = os.path.join(venv_dir, "bin")
    if not os.path.isfile(os.path.join(venv_bin, "torchrun")):
        log(f"FATAL: torchrun not in the uv venv at {venv_bin} -- uv sync did not "
            "install the gr00t launcher deps; refusing to run against conda 3.10")
        sys.exit(1)
    train_env["VIRTUAL_ENV"] = venv_dir
    train_env["PATH"] = venv_bin + os.pathsep + train_env.get("PATH", "")

    # cuDNN library precedence (required for the n17 path; harmless for n16).
    # The DLC base ships system cuDNN 9.1 in /lib/x86_64-linux-gnu, which LACKS
    # cudnnGetLibConfig (added in cuDNN 9.3). The GR00T-N1.7 backbone (Qwen3VL +
    # Flash-Attn-2 + DiT) invokes the cuDNN-graph API on the first training step;
    # with the system lib winning the loader search, finetune aborts with
    # "libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig" (SIGABRT -6).
    # N1.6's backbone never exercises that path, so the identical base works for
    # n16. The uv venv's torch bundles a newer nvidia-cudnn-cu12 (>=9.5, has the
    # symbol); prepend the venv's nvidia/*/lib dirs so they precede the DLC system
    # path. torchrun workers inherit this env (fork/exec; nothing sanitizes it).
    # Applied ONLY on the n17 path -- the proven n16 headline path is left
    # byte-for-byte unchanged (N1.6 never calls the cuDNN-graph API), so this is
    # strictly inert for n16.
    if GR00T_VERSION == "n17":
        _nvidia_libs = sorted(glob.glob(os.path.join(
            venv_dir, "lib", "python*", "site-packages", "nvidia", "*", "lib")))
        if not _nvidia_libs:
            log("[cudnn-fix] FATAL: no venv nvidia/*/lib dirs under "
                f"{venv_dir} -- the n17 finetune needs cuDNN>=9.3 (cudnnGetLibConfig); "
                "the uv venv is malformed, aborting")
            sys.exit(1)
        _prev_ld = train_env.get("LD_LIBRARY_PATH", "")
        train_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            _nvidia_libs + ([_prev_ld] if _prev_ld else []))
        log(f"[cudnn-fix] prepended {len(_nvidia_libs)} venv nvidia lib dir(s) to "
            "LD_LIBRARY_PATH (venv cuDNN precedence over DLC system 9.1)")

    train_cmd = [
        "bash", finetune_sh,
        "--base-model-path", base_local,   # LOCAL dir at the pinned 3B sha
        "--dataset-path", data_root,
        "--embodiment-tag", EMBODIMENT_TAG,
        "--output-dir", train_out,
        "--experiment-name", f"gr00t-{SUITE}-s{MAX_STEPS}",
        "--state-dropout-prob", "0.2",
    ]
    # Custom-embodiment suites (Arena GR1 -> new_embodiment) require a modality
    # config module that registers the embodiment's ModalityConfig into gr00t's
    # MODALITY_CONFIGS at import (launch_finetune.py imports --modality-config-path).
    # Ship the vendored module next to this file; env MODALITY_CONFIG_PATH overrides.
    _modality_cfg = os.environ.get("MODALITY_CONFIG_PATH", "").strip()
    if not _modality_cfg and SUITE in ("arena_gr1", "arena_gr1_fridge"):
        # n16 registers the GR1 spec under EmbodimentTag.GR1 (native head); n17
        # registers the identical spec under NEW_EMBODIMENT (376ba890 has no GR1).
        _gr1_cfg = ("arena_gr1_data_config_n16.py" if GR00T_VERSION == "n16"
                    else "arena_gr1_data_config.py")
        _modality_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     _gr1_cfg)
    if not _modality_cfg and SUITE == "arena_g1":
        _modality_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "arena_g1_data_config.py")
    if _modality_cfg:
        if not os.path.isfile(_modality_cfg):
            log(f"FATAL: modality config module missing at {_modality_cfg}")
            sys.exit(1)
        train_cmd += ["--modality-config-path", _modality_cfg]
        log(f"using --modality-config-path {_modality_cfg}")
    # Multi-checkpoint dose-curve run (SAVE_STEPS < MAX_STEPS): KEEP all intermediate
    # checkpoints. finetune.sh's LAUNCH_CMD hardcodes --save_total_limit 5 (HF Trainer
    # would prune to the 5 newest -> earliest ladder points lost). finetune.sh forwards a
    # trailing `-- <EXTRA_ARGS>` block verbatim to launch_finetune.py (tyro); save_total_limit
    # IS a FinetuneConfig field, so tyro accepts it and argparse last-wins overrides the 5.
    # Guarded on the multi-checkpoint case so the proven single-save (sample/final-only) path
    # is byte-unchanged. MUST be appended LAST (trailing EXTRA_ARGS block).
    # === DOSE-CURVE LAYER (2/3, train-time) -- active ONLY when SAVE_STEPS < MAX_STEPS ===
    # Keep ALL intermediate checkpoints as weights-only (--save_only_model drops
    # optimizer/scheduler/RNG). Submit-layer doc: src/vla_pipeline/dose_curve.py.
    if SAVE_STEPS < MAX_STEPS and GR00T_VERSION == "n16":
        log("FATAL: dose-curve mode (SAVE_STEPS < MAX_STEPS) is not supported for "
            "N1.6 -- the pinned N1.6 FinetuneConfig lacks --save_only_model. "
            "Use --save-steps >= --train-steps for N1.6 (single final checkpoint).")
        sys.exit(1)
    if SAVE_STEPS < MAX_STEPS:
        # --save_only_model (bare bool flag; verified first-class FinetuneConfig field @376ba890
        # -> transformers.TrainingArguments.save_only_model) drops optimizer+scheduler+RNG state
        # from EVERY checkpoint-N/ (~2/3 of an AdamW-fp32 checkpoint) at WRITE time. This is the
        # disk-fix root cause: save_total_limit=999999 kept full checkpoints (incl. optimizer)
        # live in the train dir, exhausting it at step 750. Intermediate ckpts are EVAL-ONLY
        # (per-checkpoint eval loop, never resumed), so weights-only stays fully eval-loadable.
        # Safe because resume_from_checkpoint defaults False (setting both is rejected upstream).
        train_cmd += ["--", "--save_total_limit=999999", "--save_only_model"]
        log(f"multi-checkpoint run (SAVE_STEPS={SAVE_STEPS} < MAX_STEPS={MAX_STEPS}): "
            "appended EXTRA_ARGS '-- --save_total_limit=999999 --save_only_model' to keep all "
            "intermediate checkpoints as WEIGHTS-ONLY (drops optimizer/scheduler/RNG ~2/3 disk; "
            "eval-loadable; single '=' int token + bare bool flag per tyro syntax)")
    log(f"CAVEAT: TRAIN_SEED={TRAIN_SEED} is RECORDED but NOT applied -- NVIDIA "
        "finetune.sh/launch_finetune.py have no seed arg; GR00T run-to-run "
        "variance is uncontrolled (documented limitation).")
    log(f">>> finetune {MAX_STEPS} steps on {SUITE}, base={BASE_MODEL}")
    log("$ " + " ".join(train_cmd))
    train_log = "/tmp/gr00t_train.log"
    with open(train_log, "w") as lf:
        proc = subprocess.Popen(train_cmd, cwd=GR00T_DIR, env=train_env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
            # cycle-16 I2: the watchdog calls os.killpg, which only reaches the whole
            # process tree if the child LEADS its own group. Without this the killpg
            # raised, the except fell back to proc.kill(), and only the direct child
            # died -- the actual training/eval grandchildren kept running while the
            # watchdog appeared to have worked. The arena evaluator this was ported
            # from already had it; the four ports did not.
            start_new_session=True)
        assert proc.stdout is not None
        # I12: the hang lives in the stdout loop below, not in wait() -- by the time
        # wait() runs the child has already closed stdout. A killed child must FAIL the
        # step: continuing with whatever was printed would manufacture a truncated
        # result, which is worse than hanging. Ported from the arena evaluator, which
        # was the only one of five streaming loops that had this.
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
        _hang_timer = threading.Timer(_subprocess_timeout(), _watchdog)
        # Daemon so a surviving timer cannot hold the interpreter open for the remainder of the
        # budget after everything else has finished. This is a backstop, not the fix: a daemon
        # timer still leaves the expensive child running, which is what the finally block below
        # is for.
        _hang_timer.daemon = True
        _hang_timer.start()

        # try/finally because the streaming loop can raise for reasons that have nothing to do
        # with training: a decode error on a malformed line, a full disk on lf.write, a broken
        # console on print. Without this, such an exception skipped BOTH the timer cancel and
        # the process kill -- so a GPU kept billing until the watchdog fired at the end of the
        # runtime budget, for a job that had already failed.
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
            rc = proc.wait()
        finally:
            _hang_timer.cancel()
            if proc.poll() is None:
                # Reached only on an abnormal exit from the block above. Kill the whole group
                # (the child leads its own session, see start_new_session) and reap it, so no
                # grandchild outlives the failure.
                log("<<< finetune FAIL: streaming ended abnormally; killing the training "
                    "process group")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=30)
                except Exception:
                    pass
        if _timed_out["v"]:
            log("<<< finetune FAIL: exceeded the runtime budget and was killed")
            sys.exit(124)
    if rc != 0:
        log(f"<<< finetune FAIL: finetune.sh exited {rc}")
        sys.exit(rc)
    # The step count the run actually TOOK, and it GOVERNS rather than merely being logged. A
    # fine-tune that stops early still produces a real checkpoint, so exit 0 does not establish
    # that the requested dose was applied -- and the dose is what the registered provenance
    # claims. Logging a discrepancy while publishing anyway is how a shorter run gets attested
    # as the full one.
    _steps_seen = None
    try:
        for _l in reversed(open(train_log, encoding="utf-8", errors="replace").read()
                           .splitlines()):
            _m = re.search(r"'?global_step'?[\"\']?\s*[:=]\s*(\d+)", _l)
            if _m:
                _steps_seen = int(_m.group(1))
                break
    except OSError:
        pass
    if _steps_seen is None:
        # This transformers/GR00T build logs {'loss','grad_norm','learning_rate'} with NO
        # 'global_step' key, so the regex above finds nothing on a genuinely complete run. Derive
        # the observed dose from two signals the log DOES carry: the HF checkpoint dir name
        # ("checkpoint-<global_step>", HF's own naming, so it attests the dose actually saved) and
        # the tqdm progress ("<n>/<MAX_STEPS>"). Take the max. Without this the attestation
        # fail-closes on a fully-trained checkpoint whose log simply never prints global_step.
        try:
            _txt = open(train_log, encoding="utf-8", errors="replace").read()
            _cands = [int(x) for x in re.findall(r"checkpoint-(\d+)", _txt)]
            _cands += [int(x) for x in re.findall(r"(\d+)/" + str(MAX_STEPS) + r"\b", _txt)]
            if _cands:
                _steps_seen = max(_cands)
        except OSError:
            pass
    if _steps_seen is None or _steps_seen != MAX_STEPS:
        log(f"<<< finetune FAIL: the completed training dose is not established. "
            f"requested={MAX_STEPS}, observed="
            f"{_steps_seen if _steps_seen is not None else 'unparsed from ' + train_log}. "
            f"Refusing to stage a checkpoint whose dose cannot be attested.")
        sys.exit(1)
    log(f"<<< finetune OK: requested {MAX_STEPS} steps, log confirms global_step={_steps_seen}, "
        f"train log at {train_log}")
    # Carried to the manifest so the recorded dose is the OBSERVED one, not the requested one.
    OBSERVED_MAX_STEPS = _steps_seen

    # ---- [6/7] Locate the CONSOLIDATED HF checkpoint, stage to SM_MODEL_DIR.
    # SPIKE-CONFIRMED (Isaac-GR00T @ 376ba890, experiment.py:370): with DeepSpeed
    # ZeRO-2 (default), the per-step dir checkpoint-<step>/ holds only the native
    # DeepSpeed engine files (mp_rank_00_model_states.pt + bf16_zero_pp_rank_*
    # optim shards) -- AutoModel.from_pretrained CANNOT read those. The eval loads
    # via Gr00tPolicy -> AutoModel.from_pretrained, which needs config.json +
    # model.safetensors. trainer.save_model() writes THAT consolidated HF
    # checkpoint (config.json + model.safetensors + experiment_cfg/ + processor/)
    # to the OUTPUT_DIR ROOT (the experiment dir), NOT into checkpoint-<step>/.
    # So we stage the experiment root, not the step dir.
    def _has_hf_weights(d):
        return (os.path.isfile(os.path.join(d, "model.safetensors"))
                or os.path.isfile(os.path.join(d, "model.safetensors.index.json")))
    exp_root = os.path.join(train_out, f"gr00t-{SUITE}-s{MAX_STEPS}")
    if not (os.path.isdir(exp_root) and _has_hf_weights(exp_root)):
        # Fallback: the single experiment subdir under train_out that has weights.
        subdirs = [os.path.join(train_out, d) for d in os.listdir(train_out)
                   if os.path.isdir(os.path.join(train_out, d))]
        exp_root = next((d for d in subdirs if _has_hf_weights(d)), exp_root)
    final_ckpt = exp_root
    if not _has_hf_weights(final_ckpt):
        run(["ls", "-laR", train_out])
        log(f"FATAL: no consolidated safetensors at experiment root {final_ckpt} "
            "-- trainer.save_model() output not found. Tree above.")
        sys.exit(1)
    # SPIKE-CONFIRMED real layout: save_model() writes SHARDED weights
    # (model-0000N-of-*.safetensors + index) + config.json + processor/ +
    # experiment_cfg/ to the root, but statistics.json + processor_config.json
    # land in checkpoint-<step>/ (not root). Backfill those from the step dir so
    # the staged tree is a COMPLETE loadable Gr00tPolicy checkpoint.
    step_dir = next((os.path.join(final_ckpt, d)
                     for d in os.listdir(final_ckpt)
                     if d.startswith("checkpoint-")), None)
    for _need in ("statistics.json", "processor_config.json", "embodiment_id.json"):
        if not os.path.isfile(os.path.join(final_ckpt, _need)) and step_dir:
            _src = os.path.join(step_dir, _need)
            if os.path.isfile(_src):
                shutil.copy2(_src, os.path.join(final_ckpt, _need))
                log(f"backfilled {_need} from {step_dir}")

    log(f"final (consolidated HF) checkpoint dir: {final_ckpt}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Stage the CONSOLIDATED HF checkpoint from the experiment root:
    #   config.json + model.safetensors     (trainer.save_model() output)
    #   processor_config.json OR processor/  (AutoProcessor.from_pretrained)
    #   statistics.json                      (Gr00tPolicy norm stats)
    #   experiment_cfg/                      (provenance)
    # EXCLUDE the intermediate checkpoint-<step>/ dirs (DeepSpeed ZeRO engine
    # shards -- AutoModel can't read them) and any *_optim_states.pt, so we don't
    # ship ~20GB of optimizer state (spike-confirmed w/ the Isaac-GR00T source).
    for item in os.listdir(final_ckpt):
        if item.startswith("checkpoint-") or item.endswith("optim_states.pt"):
            log(f"skipping non-eval artifact: {item}")
            continue
        src = os.path.join(final_ckpt, item)
        dst = os.path.join(MODEL_DIR, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Assert the checkpoint is a LOADABLE HF checkpoint before digesting --
    # fail-closed. SPIKE-CONFIRMED required set (Gr00tPolicy -> AutoModel /
    # AutoProcessor.from_pretrained): config.json + weights + processor + stats.
    def _here(*p):
        return os.path.join(MODEL_DIR, *p)
    _has_weights = (os.path.isfile(_here("model.safetensors"))
                    or os.path.isfile(_here("model.safetensors.index.json")))
    _has_processor = (os.path.isfile(_here("processor_config.json"))
                      or os.path.isfile(_here("processor", "processor_config.json")))
    _missing_ck = []
    if not os.path.isfile(_here("config.json")):
        _missing_ck.append("config.json")
    if not _has_weights:
        _missing_ck.append("model.safetensors[.index.json]")
    if not _has_processor:
        _missing_ck.append("processor_config.json (root or processor/)")
    if not os.path.isfile(_here("statistics.json")):
        _missing_ck.append("statistics.json")
    if _missing_ck:
        run(["ls", "-laR", MODEL_DIR])
        log(f"FATAL: incomplete GR00T checkpoint in {MODEL_DIR}: missing "
            f"{_missing_ck} -- eval's AutoModel/AutoProcessor.from_pretrained "
            "would fail. Inspect the tree above.")
        sys.exit(1)

    # ---- [6b/7] Stage intermediate checkpoints for the single-run dose curve.
    # Only when SAVE_STEPS < MAX_STEPS (a dose-curve run). Under SINGLE-GPU training
    # (deepspeed_config=None iff num_gpus==1) each finetune checkpoint-<step>/ is a
    # standalone HF checkpoint (consolidated model.safetensors); multi-GPU writes
    # unloadable DeepSpeed ZeRO shards, so those are warned+skipped (sidecar only).
    # Stage under a HIDDEN dir MODEL_DIR/.dose_checkpoints/checkpoint-<step>/ -- hidden
    # so weights_digest EXCLUDES them (digest.py drops any dot-leading path component),
    # keeping the canonical digest / Validate / Register BYTE-IDENTICAL to a final-only
    # artifact. They still ride in model.tar.gz for the eval loop's per-checkpoint
    # rollout (-> dose_curve.json sidecar). Checkpoint-invariant aux
    # (statistics.json/processor/experiment_cfg/...) is backfilled from the staged
    # canonical root; bulky optimizer/scheduler/rng state is dropped (eval only needs
    # from_pretrained weights). A non-loadable intermediate is NEVER fatal -- only the
    # canonical final gates the run.
    # === DOSE-CURVE LAYER (3/3, train-time) -- active ONLY when SAVE_STEPS < MAX_STEPS ===
    # Stage intermediates into a HIDDEN MODEL_DIR/.dose_checkpoints/. The dot-prefix is
    # LOAD-BEARING: weights_digest (digest.py) excludes dot-leading paths, so the canonical
    # digest stays BYTE-IDENTICAL to a final-only artifact. Never rename off the dot-prefix.
    if SAVE_STEPS < MAX_STEPS:
        dose_root = os.path.join(MODEL_DIR, ".dose_checkpoints")
        os.makedirs(dose_root, exist_ok=True)
        _shared_files = ("statistics.json", "processor_config.json", "embodiment_id.json")
        _shared_dirs = ("processor", "experiment_cfg")
        _drop_prefixes = ("optimizer", "scheduler", "rng_state", "trainer_state")
        staged_doses = []
        for _ck in sorted(d for d in os.listdir(final_ckpt) if d.startswith("checkpoint-")):
            _src = os.path.join(final_ckpt, _ck)
            if not os.path.isdir(_src):
                continue
            _dst = os.path.join(dose_root, _ck)
            os.makedirs(_dst, exist_ok=True)
            for _f in os.listdir(_src):
                if _f.endswith("optim_states.pt") or any(_f.startswith(p) for p in _drop_prefixes):
                    continue
                _fp = os.path.join(_src, _f)
                if os.path.isdir(_fp):
                    shutil.copytree(_fp, os.path.join(_dst, _f), dirs_exist_ok=True)
                else:
                    shutil.copy2(_fp, os.path.join(_dst, _f))
            for _f in _shared_files:
                _rt = os.path.join(MODEL_DIR, _f)
                if os.path.isfile(_rt) and not os.path.isfile(os.path.join(_dst, _f)):
                    shutil.copy2(_rt, os.path.join(_dst, _f))
            for _d in _shared_dirs:
                _rt = os.path.join(MODEL_DIR, _d)
                if os.path.isdir(_rt) and not os.path.isdir(os.path.join(_dst, _d)):
                    shutil.copytree(_rt, os.path.join(_dst, _d), dirs_exist_ok=True)
            _ok = (os.path.isfile(os.path.join(_dst, "model.safetensors"))
                   or os.path.isfile(os.path.join(_dst, "model.safetensors.index.json")))
            if not _ok:
                log(f"WARNING: intermediate {_ck} has no loadable safetensors "
                    "(multi-GPU ZeRO shard?); dropping from the dose-curve set")
                shutil.rmtree(_dst, ignore_errors=True)
                continue
            _step = int(_ck.rsplit("-", 1)[-1])
            write_training_lineage(_dst, **lineage_args)
            _dose_digest = weights_digest(_dst)
            _dose_manifest = {
                "manifest_version": 1,
                "model_family": "gr00t",
                "base_checkpoint": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "train_seed": TRAIN_SEED,
                "input_config": dict(INPUT_CONFIG),
                "train_recipe": {
                    "repo": GR00T_REPO,
                    "commit": GR00T_COMMIT,
                    "max_steps": _step,
                },
                "weights_digest": _dose_digest,
                "dataset_manifest": {
                    "source": dataset_source,
                    "revision": dataset_revision,
                    "episode_count": episode_count,
                },
            }
            with open(os.path.join(_dst, "checkpoint_manifest.json"), "w") as _mf:
                json.dump(_dose_manifest, _mf, indent=2)
            staged_doses.append(_ck)
        log(f"staged {len(staged_doses)} intermediate dose checkpoints -> "
            f"{dose_root} (hidden; excluded from weights_digest): {staged_doses}")

    # ---- [7/7] Compute digest and write checkpoint_manifest.json
    # Stage train_log.txt BEFORE the digest: it is a .txt -> INCLUDED by the
    # contract digest, so writing it after would make the shipped tree differ
    # from the recorded digest -> eval "digest mismatch" (mirrors OpenVLA's
    # runtime_manifest.txt ordering; same fix as the molmoact2 adapter).
    with open(train_log) as s, open(
            os.path.join(MODEL_DIR, "train_log.txt"), "w") as d:
        d.write(s.read())

    # This sidecar is part of the normative weights digest. Keeping it outside
    # checkpoint_manifest.json preserves the v1 schema in existing Arena images.
    write_training_lineage(MODEL_DIR, **lineage_args)
    digest = weights_digest(MODEL_DIR)
    log(f"weights_digest: {digest}")

    manifest = {
        "manifest_version": 1,
        "model_family": "gr00t",
        "base_checkpoint": BASE_MODEL,
        "base_revision": BASE_REVISION,
        # train_seed = the REQUESTED/configured seed. IMPORTANT CAVEAT: NVIDIA's
        # finetune.sh + launch_finetune.py expose NO seed argument
        # (spike-confirmed @ 376ba890), so this seed is NOT actually applied to
        # the RNG -- GR00T run-to-run variance is NOT controlled. The contract
        # schema has no free-form field for this note; it is documented in code,
        # in the shipped train_log.txt, and in the ladder spec. We do NOT null
        # train_seed (that would signal a published checkpoint per the rev6
        # contract, which this fine-tune is not).
        "train_seed": TRAIN_SEED,
        "input_config": dict(INPUT_CONFIG),
        "train_recipe": {
            "repo": GR00T_REPO,
            "commit": GR00T_COMMIT,
            # The dose the run REACHED, read from the trainer's own log and already required to
            # equal MAX_STEPS above. Recording the requested value here would let a shorter run
            # be attested as the full one, which is the whole point of the gate.
            "max_steps": OBSERVED_MAX_STEPS,
            # NOTE: the REQUESTED dose (MAX_STEPS) is intentionally NOT recorded here. It was
            # write-only provenance that no consumer reads, and the shipped Arena eval validator
            # is baked into an immutable-tagged image whose RECIPE key allowlist does not include
            # it -- emitting it made SimEval's verify_checkpoint reject an otherwise-valid
            # manifest ("train_recipe: unknown keys ['max_steps_requested']"). Gate integrity is
            # on max_steps (the OBSERVED dose) and is unaffected. To reinstate the field, add it
            # to the validator allowlist AND rebuild the Arena eval image in the same change.
        },
        "weights_digest": digest,
        "dataset_manifest": {
            "source": dataset_source,
            "revision": dataset_revision,
            "episode_count": episode_count,
        },
    }
    with open(os.path.join(MODEL_DIR, "checkpoint_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    run(["ls", "-la", MODEL_DIR])
    log("DONE: checkpoint + manifest staged for S3 upload")


if __name__ == "__main__":
    main()
