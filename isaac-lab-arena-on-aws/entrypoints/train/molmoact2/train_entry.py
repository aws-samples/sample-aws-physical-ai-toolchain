"""SageMaker entry point: MolmoAct2 LoRA fine-tune on LIBERO (models/ adapter).

Task 5 K2. Ports the allenai/lerobot molmoact2-policy fine-tune recipe
(docs/source/molmoact2.mdx, verified 2026-08-14 -- see
models/molmoact2/TRAIN_RECIPE.md) into a SageMaker training-job entry that
produces a merged checkpoint + contract manifest (models/contract.md) in
SM_MODEL_DIR for the pipeline's SimEval + gate.

Recipe facts (root-caused in the fork; cited in TRAIN_RECIPE.md):
  - LeRobot trainer: `accelerate launch -m lerobot.scripts.lerobot_train`
  - Policy in allenai/lerobot FORK branch molmoact2-policy (NOT upstream)
  - LoRA-VLM fine-tune (train_mode_vlm=lora, config default rank 64/alpha 16)
    fits ONE g6e.12xlarge L40S (20-41 GiB per the doc's memory table); full FT
    would need multi-GPU -- we use LoRA, so single-GPU.
  - bf16 model loading, num_flow_timesteps=8, gradient_checkpointing=true
  - Dataset: allenai/MolmoAct2-LIBERO-Dataset (LeRobot format, NOT OpenVLA RLDS)
  - Python MUST be 3.12 (fork target; same as eval_entry)

Config via environment (set by the launcher / pipeline):
  TRAIN_MAX_STEPS   fine-tune steps (default 10000; recipe reference)
  TRAIN_BATCH_SIZE  per-device batch (default 8 -> 20.2 GiB, safe on 48GB)
  TRAIN_SEED        seed (default 42)
  TRAIN_DATASET_S3URI  sentinel __LIBERO_DEFAULT__ = pull the HF LeRobot
                       dataset; else an s3:// URI to a LeRobot-layout dataset
  TRAIN_BASE_CKPT / TRAIN_BASE_REV  override the base checkpoint + revision
                       (must be given together; revision enforced)

Fail-fast per repo rules: any install/train failure exits nonzero. No
fallback that masks a failure.
"""
from __future__ import annotations

import json
import os
import threading
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digest import weights_digest  # noqa: E402  (shipped in sourcedir)

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
# Default WORK; overridden below if baked env detected
WORK = "/opt/ml/code"
if os.path.isfile("/opt/vla/.baked_env"):
    WORK = "/opt/vla"
    print("[train] BAKED ENV DETECTED -- using /opt/vla as WORK", flush=True)
LEROBOT_DIR = os.path.join(WORK, "lerobot")
# Keep large intermediate checkpoints on SageMaker scratch, outside the baked environment.
# The local runner also mounts /tmp on its selected scratch filesystem.
TRAIN_OUT = "/tmp/molmoact2-train-out"
UV_BIN = os.path.join(os.path.dirname(sys.executable), "uv")

# Pins (identical fork to eval_entry.py -- one source of truth for the port).
LEROBOT_REPO = "https://github.com/allenai/lerobot.git"
LEROBOT_BRANCH = "molmoact2-policy"
LEROBOT_COMMIT = "a4f15bf347dee7eb8a8c5f4a70a37f476b091113"
PYTHON_PIN = "3.12"
DATASET_REPO = "allenai/MolmoAct2-LIBERO-Dataset"
# Discrete-action ("FAST") tokenizer the processor's molmoact2_pack_inputs step
# resolves when action_mode includes "discrete"/"both". A SEPARATE repo from the
# base checkpoint; must be staged locally for offline eval. Revision is resolved
# + pinned at train time (the fork exposes no field to carry a tokenizer rev, so
# the staged local dir IS the pin; the manifest records its sha).
FAST_TOKENIZER_REPO = "allenai/MolmoAct2-FAST-Tokenizer"
IMAGE_KEYS = '["observation.images.image","observation.images.wrist_image"]'
CAMERA_MAP = '{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'

_HEX40 = __import__("re").compile(r"^[0-9a-f]{40}$")
_DATASET_SENTINEL = "__LIBERO_DEFAULT__"



def _resolve_norm_stats(base_dir, log, fail):
    """I1: the loader takes the normalization filename FROM base/config.json.

    Pinned MolmoAct2 reads config.json, takes norm_stats_filename, and loads statistics from the
    resulting path (lerobot @ a4f15bf3, processor_molmoact2.py:144-152). Checking that the
    CONVENTIONAL base/norm_stats.json exists proved nothing about the file actually used: a
    checkpoint could redirect to different statistics and still pass, because the conventional
    file sat there satisfying the check.

    Line 152 is a pathlib join, so an absolute value REPLACES the base and reads from outside
    the checkpoint entirely. Both that and a `..` escape are refused here rather than left to
    fail obscurely at load time.
    """
    import json as _json
    import os as _os
    config_path = _os.path.join(base_dir, "config.json")
    name = "norm_stats.json"
    try:
        with open(config_path) as _fh:
            configured = _json.load(_fh).get("norm_stats_filename")
    except (OSError, ValueError) as exc:
        fail(f"base/config.json at {config_path} is unreadable ({exc}); the normalization "
             f"filename it carries decides which statistics the loader uses.")
        return None
    if configured is not None:
        if not isinstance(configured, str) or not configured:
            fail(f"base/config.json norm_stats_filename is {configured!r}, which is not a "
                 f"usable filename.")
            return None
        name = configured
    if _os.path.isabs(name):
        fail(f"base/config.json norm_stats_filename is the ABSOLUTE path {name!r}. The loader "
             f"joins it with pathlib, so an absolute value replaces the checkpoint directory "
             f"and reads statistics from outside the checkpoint.")
        return None
    resolved = _os.path.realpath(_os.path.join(base_dir, name))
    base_real = _os.path.realpath(base_dir)
    if resolved != base_real and not resolved.startswith(base_real + _os.sep):
        fail(f"base/config.json norm_stats_filename {name!r} resolves outside base/ "
             f"({resolved}). Statistics outside the checkpoint are not covered by its digest.")
        return None
    if not _os.path.isfile(resolved):
        fail(f"the normalization statistics the loader will read are missing: {resolved} "
             f"(named by norm_stats_filename={name!r} in base/config.json)")
        return None
    log(f"[preflight] normalization statistics resolved from base/config.json: {name}")
    return resolved

def _require_int(name: str, value: str, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except ValueError:
        print(f"[molmoact2-train] FATAL: {name} not an integer: {value!r}",
              flush=True)
        sys.exit(1)
    if not (lo <= n <= hi):
        print(f"[molmoact2-train] FATAL: {name}={n} outside [{lo},{hi}]",
              flush=True)
        sys.exit(1)
    return n


_max_steps_raw = os.environ.get("TRAIN_MAX_STEPS", "10000")
MAX_STEPS = str(_require_int("TRAIN_MAX_STEPS", _max_steps_raw, 1, 1_000_000))
BATCH = str(_require_int("TRAIN_BATCH_SIZE",
                         os.environ.get("TRAIN_BATCH_SIZE", "8"), 1, 512))
TRAIN_SEED = str(_require_int("TRAIN_SEED",
                              os.environ.get("TRAIN_SEED", "42"), 0, 2**31 - 1))
# Base checkpoint: default to the released LIBERO checkpoint; overridable, but
# repo+rev must be supplied together (revision enforcement, mirrors eval).
BASE_CKPT = os.environ.get("TRAIN_BASE_CKPT", "allenai/MolmoAct2-LIBERO")
BASE_REV = os.environ.get("TRAIN_BASE_REV",
                          "0d24a92bd1faf321ef497c3bbd5681af97c65aa2")
if bool(os.environ.get("TRAIN_BASE_CKPT")) ^ bool(os.environ.get("TRAIN_BASE_REV")):
    print("[molmoact2-train] FATAL: TRAIN_BASE_CKPT and TRAIN_BASE_REV must be "
          "overridden together", flush=True)
    sys.exit(1)
if not _HEX40.fullmatch(BASE_REV):
    print(f"[molmoact2-train] FATAL: TRAIN_BASE_REV not 40-hex: {BASE_REV!r}",
          flush=True)
    sys.exit(1)

DATASET_S3URI = (os.environ.get("TRAIN_DATASET_S3URI", _DATASET_SENTINEL)
                 or _DATASET_SENTINEL)
BYO_DATASET = DATASET_S3URI != _DATASET_SENTINEL
if BYO_DATASET and not DATASET_S3URI.startswith("s3://"):
    print(f"[molmoact2-train] FATAL: TRAIN_DATASET_S3URI must be s3:// or the "
          f"sentinel, got {DATASET_S3URI!r}", flush=True)
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[molmoact2-train] {msg}", flush=True)


def log_storage(phase: str) -> None:
    """Record actual filesystems/headroom without hiding the original training error."""
    import shutil

    for path in ("/", WORK, "/tmp", MODEL_DIR):
        existing = os.path.abspath(path)
        while not os.path.exists(existing):
            existing = os.path.dirname(existing)
        try:
            usage = shutil.disk_usage(existing)
            log("[storage] " + json.dumps({
                "phase": phase, "path": path, "measured_path": existing,
                "device": os.stat(existing).st_dev,
                "total_bytes": usage.total, "free_bytes": usage.free,
            }))
        except OSError as exc:
            log(f"[storage] {phase} {path}: unavailable ({exc})")


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
    the command, which is diagnosable. The pipeline passes its selected budget.
    Standalone invocations use a 28800-second fallback rather than running unbounded.
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


def uv(*args: str, **kw) -> None:
    # Strip the DLC's py3.10 PYTHONPATH so uv's py3.12 venv is self-contained
    # (SRE module mismatch otherwise -- proven in the eval spike).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # PyPI 502 resilience (uv default is 3 retries in ~4s -- too aggressive for
    # a transient blip). Harmless if an older uv ignores UV_HTTP_RETRIES.
    env.setdefault("UV_HTTP_TIMEOUT", "120")
    env.setdefault("UV_HTTP_RETRIES", "10")
    kw.setdefault("env", env)
    run([UV_BIN, *args], cwd=LEROBOT_DIR, **kw)


def uv_sync_resilient(*sync_args: str) -> None:
    """`uv sync` with process-level retry + backoff on top of UV_HTTP_RETRIES.
    uv sync is resumable, so a retry only re-fetches what a transient PyPI 502
    dropped (observed on a big CUDA wheel). Fail-closed after the last attempt."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("UV_HTTP_TIMEOUT", "120")
    env.setdefault("UV_HTTP_RETRIES", "10")
    delay = 30
    for attempt in range(1, 6):
        out = subprocess.run([UV_BIN, "sync", *sync_args], cwd=LEROBOT_DIR,
                             env=env)
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


# The rule by which training episodes are chosen, recorded verbatim in the manifest so
# the selection can be reproduced and audited rather than inferred from the code.
EPISODE_SELECTION_RULE = "prefix: range(min(max(5, max_steps // 2), 100, episode_count))"


def _select_episodes(max_steps: int, episode_count: int) -> list[int]:
    """Choose the episodes training will consume.

    A PREFIX of the dataset, capped at 100, floored at 5, and scaled by the requested
    step count -- decoding all 1693 video episodes would write ~1TB to disk. The cap is
    a real constraint, but it means the artifact's dataset provenance must record what
    was CONSUMED: recording only the dataset's episode_count made a run over the first
    five episodes indistinguishable from a run over all of them, and at smaller doses
    changing the step count silently changes the training data too.

    Kept as a pure function so the recorded provenance and the argv passed to training
    cannot drift apart, and so the rule is testable without standing up a training job.
    """
    if episode_count < 1:
        raise ValueError(f"episode_count must be >= 1, got {episode_count}")
    return list(range(min(max(5, max_steps // 2), 100, episode_count)))


def main() -> None:
    # PyPI 502 resilience for EVERY pip in this job (incl. build-isolation
    # subprocess pips that ignore an explicit --retries flag).
    os.environ["PIP_RETRIES"] = "10"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "60"
    log(f"max_steps={MAX_STEPS} batch={BATCH} seed={TRAIN_SEED} "
        f"base={BASE_CKPT}@{BASE_REV[:12]} byo={BYO_DATASET}")
    log(f"intermediate checkpoints: {TRAIN_OUT}; final model: {MODEL_DIR}")
    log_storage("before_downloads")
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

    # ---- [1/6] System libs (LeRobot dataset video decode + build tooling)
    os.chmod("/tmp", 0o1777)
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "--no-install-recommends",
         "git", "curl", "ca-certificates", "libegl1", "libgles2",
         "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "ffmpeg",
         "build-essential", "cmake"])

    # ---- [2/6] uv (pinned) + pinned fork clone (commit-verified)
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "--retries", "10", "--timeout", "60", "uv==0.12.1"])
    run([UV_BIN, "--version"])
    if os.path.isdir(LEROBOT_DIR):
        log(f"lerobot dir exists at {LEROBOT_DIR} -- skipping clone, verifying commit")
        run(["git", "-C", LEROBOT_DIR, "checkout", LEROBOT_COMMIT])
    else:
        run(["git", "clone", "--branch", LEROBOT_BRANCH, LEROBOT_REPO, LEROBOT_DIR])
        run(["git", "-C", LEROBOT_DIR, "checkout", LEROBOT_COMMIT])
    head = subprocess.run(["git", "-C", LEROBOT_DIR, "rev-parse", "HEAD"],
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != LEROBOT_COMMIT:
        log(f"FATAL: fork commit mismatch: {head}")
        sys.exit(1)

    # ---- [3/6] uv sync at the pinned interpreter (fork source build)
    # The `dataset` extra provides HuggingFace `datasets`, which the LeRobot
    # TRAINER requires to load the LeRobot-format dataset (the eval path does
    # not need it, so eval_entry omits it). Missing here caused: ImportError
    # "'datasets' is required ... install 'lerobot[dataset]'" on first K2 run.
    uv_sync_resilient("--locked", "--python", PYTHON_PIN,
                      "--extra", "molmoact2", "--extra", "dataset")
    uv("run", "--active", "python", "-c",
       "import sys; v='%d.%d' % sys.version_info[:2]; "
       f"assert v == '{PYTHON_PIN}', v; print('venv python', v)")

    # ---- [4/6] Dataset: default HF LeRobot pull, or BYO S3 sync.
    data_root = os.path.join(WORK, "lerobot-data", DATASET_REPO)
    os.makedirs(data_root, exist_ok=True)
    if BYO_DATASET:
        # A LeRobot-layout dataset. (The RLDS validator in data/ is OpenVLA-
        # specific; a LeRobot validator is a documented follow-on -- see
        # README.md, Bring your own dataset. We still record identity below.)
        log(f"BYO dataset: syncing {DATASET_S3URI} -> {data_root}")
        run(["aws", "s3", "sync", DATASET_S3URI, data_root])
        dataset_source = DATASET_S3URI
        dataset_revision = weights_digest(data_root)  # content digest = its identity
    else:
        # Resolve the requested HF revision, then download that exact snapshot.
        # Hugging Face retains its normal local download cache.
        log(f"downloading LeRobot dataset {DATASET_REPO} ...")
        _ds_sha_path = "/tmp/molmoact2_ds_sha.txt"
        _hf_rev = os.environ.get("TRAIN_DATASET_REVISION", "").strip()
        if _hf_rev == "__FROM_SUITE_MANIFEST__":
            _hf_rev = ""
        _resolve_code = (
            "import sys; from huggingface_hub import HfApi\n"
            "rev = sys.argv[2] or None\n"
            "sha = HfApi().dataset_info(sys.argv[1], revision=rev).sha\n"
            "open(sys.argv[3],'w').write(sha)")
        uv("run", "--active", "python", "-c", _resolve_code,
           DATASET_REPO, _hf_rev, _ds_sha_path)
        with open(_ds_sha_path) as _fh:
            _resolved_sha = _fh.read().strip()
        log(f"resolved dataset revision: {_hf_rev or '(default)'} -> {_resolved_sha}")
        _dl_code = (
            "import sys; from huggingface_hub import snapshot_download\n"
            "snapshot_download(sys.argv[1], repo_type='dataset', "
            "local_dir=sys.argv[2], revision=sys.argv[3])")
        uv("run", "--active", "python", "-c", _dl_code,
           DATASET_REPO, data_root, _resolved_sha)
        dataset_source = f"hf:{DATASET_REPO}"
        dataset_revision = _resolved_sha

    # Episode count from the LeRobot dataset metadata (meta/info.json:
    # total_episodes). Contract requires a positive integer -- fail-closed if the
    # metadata is absent/malformed rather than guessing.
    _info_path = os.path.join(data_root, "meta", "info.json")
    if not os.path.isfile(_info_path):
        log(f"FATAL: dataset meta/info.json missing at {_info_path}")
        sys.exit(1)
    with open(_info_path) as _fh:
        _info = json.load(_fh)
    episode_count = _info.get("total_episodes")
    if not isinstance(episode_count, int) or episode_count < 1:
        log(f"FATAL: dataset total_episodes invalid: {episode_count!r}")
        sys.exit(1)
    log(f"dataset episode_count={episode_count} revision={dataset_revision}")
    log_storage("after_dataset_download")

    # Which episodes training will actually see. This is a PREFIX of the dataset, and
    # the size depends on the requested step count, so it must be recorded as provenance
    # rather than left implicit: the manifest used to carry only the full episode_count.
    selected_episodes = _select_episodes(int(MAX_STEPS), episode_count)
    log(f"dataset episodes selected: {len(selected_episodes)} of {episode_count} "
        f"({EPISODE_SELECTION_RULE})")

    # ---- [5/6] LoRA fine-tune via the fork's trainer (single GPU; accelerate
    # num_processes auto-detects). Preserve early checkpoints (save_freq small)
    # so Task 7 has a genuinely-undertrained negative-branch checkpoint.
    # Do NOT pre-create TRAIN_OUT: the LeRobot trainer refuses an existing
    # output_dir when resume=False ("Output directory ... already exists")
    # -- pre-creating it made EVERY K2 run fail. Let the trainer own it.
    save_freq = str(max(1, int(MAX_STEPS) // 5))  # save at ~20% intervals; at least 1
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    train_cmd = [
        UV_BIN, "run", "--active", "accelerate", "launch",
        "--num_processes=1", "--mixed_precision=bf16",
        "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.root={data_root}",
        "--dataset.video_backend=pyav",
        "--dataset.image_transforms.enable=true",
        # Limit episodes to prevent decoding ALL 1693 video episodes to disk.
        # Scale: min(max(5, steps//2), 100) -- caps at 100 episodes always.
        # 100 episodes × 280 frames = 28,000 training samples (enough for 2000 steps at batch=8).
        # Without limit: 1693 episodes decode ~1TB+ to disk, filling any EBS volume.
        # The selected count and rule are recorded in the manifest (see
        # episodes_selected / episode_selection): the artifact used to report the FULL
        # dataset episode_count, so its training provenance did not describe the data
        # actually used, and at smaller doses changing the step count silently changes
        # the dataset too.
        *([f"--dataset.episodes=[{','.join(str(i) for i in selected_episodes)}]"]),
        "--policy.type=molmoact2",
        f"--policy.checkpoint_path={BASE_CKPT}",
        f"--policy.checkpoint_revision={BASE_REV}",
        "--policy.device=cuda",
        "--policy.action_mode=both",
        "--policy.train_mode_vlm=lora",          # LoRA (not fft): fits 1 L40S
        "--policy.chunk_size=10",
        "--policy.n_action_steps=10",
        "--policy.setup_type=single franka robotic arm in libero",
        "--policy.control_mode=delta end-effector pose",
        f"--policy.image_keys={IMAGE_KEYS}",
        "--policy.model_dtype=bfloat16",
        "--policy.num_flow_timesteps=8",
        "--policy.gradient_checkpointing=true",
        "--policy.freeze_embedding=true",
        "--policy.normalize_gripper=false",
        "--policy.enable_knowledge_insulation=false",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--job_name=molmoact2-k2",
        f"--output_dir={TRAIN_OUT}",
        f"--steps={MAX_STEPS}",
        f"--batch_size={BATCH}",
        f"--seed={TRAIN_SEED}",
        "--num_workers=4",
        "--log_freq=20",
        "--eval_freq=-1",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
    ]
    # Trace boundary: the last line reachable without an accelerator. accelerate launch loads
    # the model onto a CUDA device, so on a CPU L3 job this chunk OPENS and never closes --
    # read_trace reports it as the death point, which is the GPU boundary this rung locates.
    log(">>> finetune accelerate launch lerobot_train (model load binds a GPU device)")
    log("$ " + " ".join(train_cmd))
    train_log = "/tmp/molmoact2_train.log"
    with open(train_log, "w") as lf:
        proc = subprocess.Popen(train_cmd, cwd=LEROBOT_DIR, env=env,
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
        _hang_timer.daemon = True
        _hang_timer.start()
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
            rc = proc.wait()
        finally:
            _hang_timer.cancel()
            if proc.poll() is None:
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
            proc.stdout.close()
        if _timed_out["v"]:
            log("FATAL: lerobot_train exceeded the runtime budget and was killed")
            sys.exit(124)
    if rc != 0:
        log_storage("training_failed")
        log(f"FATAL: lerobot_train exited {rc}")
        sys.exit(rc)
    log_storage("after_training")

    # ---- [6/6] Locate the final checkpoint LeRobot saved, stage it +
    # checkpoint_manifest.json to SM_MODEL_DIR for the gate.
    ckpts_root = os.path.join(TRAIN_OUT, "checkpoints")
    if not os.path.isdir(ckpts_root):
        log(f"FATAL: no checkpoints dir at {ckpts_root}")
        sys.exit(1)
    # LeRobot (verified in src/lerobot/utils/constants.py @ pinned commit):
    # checkpoints/<zero-padded-step>/pretrained_model/, plus a 'last' symlink.
    # Prefer 'last'; else the max numbered step. Zero-padded names ('010000')
    # still pass isdigit() and sort correctly by int.
    steps = sorted(
        (d for d in os.listdir(ckpts_root)
         if d.isdigit() and os.path.isdir(os.path.join(ckpts_root, d))),
        key=int)
    if not steps:
        log(f"FATAL: no numbered checkpoints under {ckpts_root}: "
            f"{os.listdir(ckpts_root)}")
        sys.exit(1)
    last_link = os.path.join(ckpts_root, "last")
    step_dir = (last_link if os.path.isdir(last_link)
                else os.path.join(ckpts_root, steps[-1]))
    final_ckpt = os.path.join(step_dir, "pretrained_model")
    if not os.path.isdir(final_ckpt):
        final_ckpt = step_dir  # defensive: some versions nest differently
    log(f"final checkpoint: {final_ckpt} (early kept for negative branch: "
        f"step {steps[0]})")

    # Option C -- SELF-CONTAINED base+adapter tree. The fork does NOT merge LoRA;
    # its eval re-instantiates the base architecture, strict-loads the BASE
    # weights (from config.checkpoint_path), re-wraps with PEFT, then loads our
    # LoRA adapter (strict=False). So the shipped checkpoint must carry BOTH the
    # adapter AND the pinned base, and the digest must cover the whole tree.
    #   MODEL_DIR/policy/   LoRA output (config.json, model.safetensors, ...)
    #   MODEL_DIR/base/     pinned base snapshot (weights + config + norm_stats.json)
    # At eval, --policy.path=<root>/policy re-wraps PEFT (train_mode_vlm=lora
    # persists in policy/config.json); the base is resolved from an ABSOLUTE
    # --policy.checkpoint_path=<root>/base CLI override (a relative path would
    # resolve against the eval CWD, not the config dir; a baked absolute path is
    # wrong because the eval mount root is arbitrary -- verified in
    # _resolve_checkpoint_location: Path(checkpoint_path).expanduser().exists()).
    import shutil
    policy_dir = os.path.join(MODEL_DIR, "policy")
    base_dir = os.path.join(MODEL_DIR, "base")
    os.makedirs(policy_dir, exist_ok=True)

    # (a) LoRA output -> MODEL_DIR/policy/
    for item in os.listdir(final_ckpt):
        src = os.path.join(final_ckpt, item)
        dst = os.path.join(policy_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    policy_cfg_path = os.path.join(policy_dir, "config.json")
    if not os.path.isfile(policy_cfg_path):
        log(f"FATAL: LoRA policy config.json missing at {policy_cfg_path}")
        sys.exit(1)
    if not os.path.isfile(os.path.join(policy_dir, "model.safetensors")):
        log("FATAL: LoRA policy model.safetensors missing")
        sys.exit(1)

    # (b) Base snapshot -> MODEL_DIR/base/. Materialized real files (no symlinks:
    # the contract digest FORBIDS symlinks). Resolve the immutable 40-hex commit
    # the tag/rev pinned. Fail-closed if base or its norm_stats.json is missing.
    log(f"materializing base {BASE_CKPT}@{BASE_REV[:12]} -> {base_dir}")
    resolved_sha_path = "/tmp/molmoact2_base_sha.txt"
    uv("run", "--active", "python", "-c",
       "import sys\n"
       "from huggingface_hub import snapshot_download, HfApi\n"
       "snapshot_download(sys.argv[1], revision=sys.argv[2],\n"
       "    local_dir=sys.argv[3], local_dir_use_symlinks=False)\n"
       "info = HfApi().model_info(sys.argv[1], revision=sys.argv[2])\n"
       "open(sys.argv[4], 'w').write(info.sha)\n"
       "print('materialized base sha', info.sha)",
       BASE_CKPT, BASE_REV, base_dir, resolved_sha_path)
    if not os.path.isdir(base_dir):
        log(f"FATAL: base snapshot missing at {base_dir}")
        sys.exit(1)
    if not os.path.isfile(os.path.join(base_dir, "config.json")):
        log(f"FATAL: base/ missing config.json at {base_dir}")
        sys.exit(1)

    def _fail_norm(msg):
        log(f"FATAL: {msg}")
        sys.exit(1)
    # I1: the TRAIN side had the same hardcoded check as eval. Fixing one of two paths has been
    # a recurring defect here, so both resolve the configured filename.
    _resolve_norm_stats(base_dir, log, _fail_norm)
    # Contract: no symlinks anywhere in the digested tree. hf's local_dir may
    # leave a hidden .cache/ (excluded by the digest); a stray symlink is fatal.
    with open(resolved_sha_path) as _fh:
        base_snapshot_resolved_sha = _fh.read().strip()
    if not _HEX40.fullmatch(base_snapshot_resolved_sha):
        log(f"FATAL: resolved base sha not 40-hex: {base_snapshot_resolved_sha!r}")
        sys.exit(1)

    # Stage the FAST (discrete-action) tokenizer -> MODEL_DIR/fast_tokenizer/. The
    # processor's molmoact2_pack_inputs step resolves this SEPARATE repo at eval
    # (action_mode=both); offline mode blocks the HF fetch, so it must be local.
    # Resolve + pin its sha here (the fork has no field to carry a tokenizer rev).
    fast_dir = os.path.join(MODEL_DIR, "fast_tokenizer")
    fast_sha_path = "/tmp/molmoact2_fast_sha.txt"
    log(f"materializing FAST tokenizer {FAST_TOKENIZER_REPO} -> {fast_dir}")
    # Resolve the sha FIRST, then download AT that sha, then record it. This used to
    # call snapshot_download() with no revision and afterwards call model_info()
    # independently: two separate observations of the repository head, so the recorded
    # sha did not necessarily identify the bytes that were downloaded. Pinning the
    # download to the resolved sha makes the record describe the snapshot by
    # construction rather than by assumption.
    uv("run", "--active", "python", "-c",
       "import sys\n"
       "from huggingface_hub import snapshot_download, HfApi\n"
       "sha = HfApi().model_info(sys.argv[1]).sha\n"
       "if not sha:\n"
       "    raise SystemExit('FATAL: could not resolve a sha for ' + sys.argv[1])\n"
       "snapshot_download(sys.argv[1], revision=sha, local_dir=sys.argv[2],\n"
       "    local_dir_use_symlinks=False)\n"
       "open(sys.argv[3], 'w').write(sha)\n"
       "print('materialized fast-tokenizer at pinned sha', sha)",
       FAST_TOKENIZER_REPO, fast_dir, fast_sha_path)
    if not os.path.isfile(os.path.join(fast_dir, "tokenizer.json")):
        log(f"FATAL: fast_tokenizer/ missing tokenizer.json at {fast_dir} -- the "
            "processor's discrete-action step cannot load it offline")
        sys.exit(1)
    with open(fast_sha_path) as _fh:
        fast_tokenizer_resolved_sha = _fh.read().strip()
    if not _HEX40.fullmatch(fast_tokenizer_resolved_sha):
        log(f"FATAL: resolved fast-tokenizer sha not 40-hex: "
            f"{fast_tokenizer_resolved_sha!r}")
        sys.exit(1)

    # (c) Rewrite policy/config.json: checkpoint_path is a PLACEHOLDER (the eval
    # injects the real absolute <root>/base as a CLI override); pin the resolved
    # sha in checkpoint_revision (NOT embedded in checkpoint_path).
    with open(policy_cfg_path) as _fh:
        policy_cfg = json.load(_fh)
    policy_cfg["checkpoint_path"] = "PLACEHOLDER_SET_AT_EVAL"
    policy_cfg["checkpoint_revision"] = base_snapshot_resolved_sha
    with open(policy_cfg_path, "w") as _fh:
        json.dump(policy_cfg, _fh, indent=4)

    # (d) Record the ACTUAL PEFT config the fork applies (defaults, not assumed):
    # rank/alpha/dropout/bias from MolmoAct2Config; target-module regex from the
    # fork's _lora_target_modules (modeling_molmoact2.py).
    lora_probe = "/tmp/molmoact2_lora.json"
    uv("run", "--active", "python", "-c",
       "import sys, json\n"
       "from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config\n"
       "c = MolmoAct2Config()\n"
       "leaves = 'w1|w2|w3|wq|wk|wv|wo|att_proj|attn_out|ff_proj|ff_out|patch_embedding'\n"
       "tm = r'model\\.(transformer|vision_backbone)\\.(?:.*\\.)?(' + leaves + r')$'\n"
       "json.dump({'lora_rank': c.lora_rank, 'lora_alpha': c.lora_alpha,\n"
       "           'lora_dropout': c.lora_dropout, 'lora_bias': c.lora_bias,\n"
       "           'lora_target_modules': tm}, open(sys.argv[1], 'w'))",
       lora_probe)
    with open(lora_probe) as _fh:
        lora_cfg = json.load(_fh)

    # Stage the training log BEFORE the digest so the recorded weights_digest
    # covers the FULL shipped tree (train_log.txt is a .txt -> INCLUDED by the
    # contract digest; writing it after made shipped bytes differ from the
    # recorded digest). Mirrors OpenVLA's proven runtime_manifest.txt ordering.
    with open(train_log) as s, open(
            os.path.join(MODEL_DIR, "train_log.txt"), "w") as d:
        d.write(s.read())

    # Digest LAST over the ENTIRE MODEL_DIR (base/ + policy/ + log), binding
    # base+adapter+norm+config. checkpoint_manifest.json is excluded by the
    # digest itself (models/common/digest.py EXCLUDED_BASENAMES).
    digest = weights_digest(MODEL_DIR)
    log(f"weights_digest (base+policy tree): {digest}")
    manifest = {
        "manifest_version": 1,
        "model_family": "molmoact2",
        "base_checkpoint": BASE_CKPT,
        "base_revision": BASE_REV,
        "base_snapshot_resolved_sha": base_snapshot_resolved_sha,
        "fast_tokenizer_resolved_sha": fast_tokenizer_resolved_sha,
        "train_seed": int(TRAIN_SEED),
        "input_config": {
            "norm_tag": "libero",
            # EVAL runs float32 (official LIBERO recipe); the bf16 TRAINING dtype
            # is recorded in train_recipe, not here.
            "model_dtype": "float32",
            "inference_action_mode": "continuous",
            "camera_name_mapping": CAMERA_MAP,
        },
        "train_recipe": {
            "repo": LEROBOT_REPO, "commit": LEROBOT_COMMIT,
            "max_steps": int(MAX_STEPS), "train_mode_vlm": "lora",
            "batch_size": int(BATCH),
            "train_model_dtype": "bfloat16",
            "lora_rank": lora_cfg["lora_rank"],
            "lora_alpha": lora_cfg["lora_alpha"],
            "lora_dropout": lora_cfg["lora_dropout"],
            "lora_bias": lora_cfg["lora_bias"],
            "lora_target_modules": lora_cfg["lora_target_modules"],
        },
        "checkpoint_layout": {"policy": "policy", "base": "base",
                              "fast_tokenizer": "fast_tokenizer"},
        "weights_digest": digest,
        # Contract DATASET_KEYS = {source, revision, episode_count} exactly --
        # no 'byo' (unknown key -> validate fails); revision non-null (pinned sha
        # or BYO content digest); episode_count a positive int from meta/info.json.
        "dataset_manifest": {
            "source": dataset_source,
            "revision": dataset_revision,
            "episode_count": episode_count,
            # What training ACTUALLY consumed. episode_count alone described the
            # dataset, not the data used, so the provenance did not distinguish a run
            # over 1693 episodes from one over the first 5.
            "episodes_selected": len(selected_episodes),
            "episode_selection": EPISODE_SELECTION_RULE,
        },
    }
    with open(os.path.join(MODEL_DIR, "checkpoint_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    run(["ls", "-la", MODEL_DIR])
    log("DONE: self-contained base+adapter checkpoint + manifest staged")


if __name__ == "__main__":
    main()
