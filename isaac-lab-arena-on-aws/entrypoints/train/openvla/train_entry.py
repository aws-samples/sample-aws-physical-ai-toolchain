"""SageMaker entry point: LoRA fine-tune OpenVLA on LIBERO demos (models/ adapter).

Contract: models/contract.md rev6. Extends the proven v1 train recipe
(train/train_entry.py, 4 pipeline-verified runs) with the contract's
checkpoint manifest: real dataset identity, an actually-applied train seed,
and the normative weights digest computed over SM_MODEL_DIR BEFORE upload --
the content half of the gate's mounted-bytes == produced-bytes binding.

Config via environment (set by the pipeline):
  TRAIN_SUITE       libero_spatial (maps to dataset libero_spatial_no_noops)
  TRAIN_MAX_STEPS   e.g. 4000
  TRAIN_BATCH_SIZE  per-GPU batch size
  TRAIN_GRAD_ACCUM  gradient accumulation steps
  TRAIN_LORA_RANK   default 32 (OFT paper)
  TRAIN_SEED        int; seeds random/numpy/torch in-process before finetune.py
                    (upstream FinetuneConfig has no seed field -- verified; the
                    shim seeds the process instead, which is the scope it can actually claim:
                    dataloader shuffling and LoRA init draw from these RNGs)

Fail-fast: no fallbacks that mask failure.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digest import weights_digest  # noqa: E402  (shipped in sourcedir)
from rlds_validator import validate_rlds_dataset  # noqa: E402  (shipped in sourcedir)

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
WORK = "/opt/ml/code"
OFT_DIR = os.path.join(WORK, "openvla-oft")
RLDS_DIR = os.path.join(WORK, "rlds")

OFT_REPO = "https://github.com/moojink/openvla-oft.git"
OFT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
BASE_CHECKPOINT = "openvla/openvla-7b"
DATASET_REPO = "openvla/modified_libero_rlds"
TFMD_PIN = "tensorflow-metadata==1.17.3"
FLASH_ATTN = "flash-attn==2.5.5"

VALID_SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
HEX40 = __import__("re").compile(r"^[0-9a-f]{40}$")


def _require_suite(value: str) -> str:
    if value not in VALID_SUITES:
        print(f"[train] FATAL: TRAIN_SUITE {value!r} not in {sorted(VALID_SUITES)}",
              flush=True)
        sys.exit(1)
    return value


def _require_int(name: str, value: str, lo: int, hi: int) -> int:
    # BLOCKER fix (whole-system review): parameters are strings a
    # StartPipelineExecution caller controls -- parse strictly BEFORE any
    # use; never let raw parameter text reach generated code.
    try:
        n = int(value)
    except ValueError:
        print(f"[train] FATAL: {name} not an integer: {value!r}", flush=True)
        sys.exit(1)
    if not (lo <= n <= hi):
        print(f"[train] FATAL: {name}={n} outside [{lo}, {hi}]", flush=True)
        sys.exit(1)
    return n


def _require_rev(name: str, value: str) -> str:
    if value and not HEX40.fullmatch(value):
        print(f"[train] FATAL: {name} not a 40-hex revision: {value!r}", flush=True)
        sys.exit(1)
    return value


SUITE = _require_suite(os.environ.get("TRAIN_SUITE", "libero_spatial"))
DATASET = f"{SUITE}_no_noops"
# train_steps (from pipeline hyperparameter) overrides TRAIN_MAX_STEPS
_max_steps_raw = os.environ.get("TRAIN_MAX_STEPS", "4000")
MAX_STEPS = str(_require_int("TRAIN_MAX_STEPS", _max_steps_raw, 1, 1_000_000))
BATCH = str(_require_int("TRAIN_BATCH_SIZE", os.environ.get("TRAIN_BATCH_SIZE", "1"), 1, 512))
GRAD_ACCUM = str(_require_int("TRAIN_GRAD_ACCUM", os.environ.get("TRAIN_GRAD_ACCUM", "4"), 1, 512))
LORA_RANK = str(_require_int("TRAIN_LORA_RANK", os.environ.get("TRAIN_LORA_RANK", "32"), 1, 1024))
TRAIN_SEED = str(_require_int("TRAIN_SEED", os.environ.get("TRAIN_SEED", "42"), 0, 2**31 - 1))
_raw_rev = os.environ.get("TRAIN_DATASET_REV", "") or os.environ.get("TRAIN_DATASET_REVISION", "")
if _raw_rev == "__FROM_SUITE_MANIFEST__":
    _raw_rev = ""
DATASET_REV_PIN = _require_rev("TRAIN_DATASET_REV", _raw_rev)
BASE_REV_PIN = _require_rev("TRAIN_BASE_REV", os.environ.get("TRAIN_BASE_REV", ""))
# Bring-your-own-data (Task 3): sentinel = pull the default LIBERO RLDS from
# HF; any other value is an s3:// URI to the caller's RLDS dataset. Empty
# string is treated as the sentinel (defensive: an unset param must not be
# mistaken for a real URI).
_DATASET_SENTINEL = "__LIBERO_DEFAULT__"
DATASET_S3URI = os.environ.get("TRAIN_DATASET_S3URI", _DATASET_SENTINEL) or _DATASET_SENTINEL
BYO_DATASET = DATASET_S3URI != _DATASET_SENTINEL
if BYO_DATASET and not DATASET_S3URI.startswith("s3://"):
    print(f"[train] FATAL: TRAIN_DATASET_S3URI must be an s3:// URI or the "
          f"sentinel {_DATASET_SENTINEL!r}, got {DATASET_S3URI!r}", flush=True)
    sys.exit(1)

INPUT_CONFIG = {
    "num_images_in_input": 1,
    "use_proprio": False,
    "unnorm_key": DATASET,
}


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


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
    # Absorb transient PyPI 502s WITHIN the job (pip's own retry/backoff)
    # instead of failing the whole hours-long training and re-running it. The
    # 'too many 502 error responses' flakes on files.pythonhosted.org were the
    # top recurring cause of spurious training-job failures on this ladder.
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
         "--retries", "10", "--timeout", "60", *args])


def rlds_episode_count(dataset_dir: str) -> int:
    """Episode count from the RLDS dataset_info.json (fatal if unreadable)."""
    info_path = os.path.join(dataset_dir, "1.0.0", "dataset_info.json")
    if not os.path.isfile(info_path):
        candidates = [os.path.join(r, "dataset_info.json")
                      for r, _d, fs in os.walk(dataset_dir)
                      if "dataset_info.json" in fs]
        if not candidates:
            log(f"FATAL: no dataset_info.json under {dataset_dir}")
        info_path = candidates[0]
    with open(info_path) as fh:
        info = json.load(fh)
    total = 0
    for split in info.get("splits", []):
        lengths = split.get("shardLengths", [])
        total += sum(int(x) for x in lengths)
    if total < 1:
        log(f"FATAL: could not derive episode count from {info_path}")
        sys.exit(1)
    return total


def hf_revision(repo_id: str, repo_type: str = "model") -> str:
    """Resolved HEAD revision actually used by this job (recorded, not pinned)."""
    from huggingface_hub import HfApi
    api = HfApi()
    if repo_type == "dataset":
        sha = api.dataset_info(repo_id).sha
    else:
        sha = api.model_info(repo_id).sha
    if not sha:
        log(f"FATAL: could not resolve revision for {repo_id}")
        sys.exit(1)
    return sha


def main() -> None:
    # PyPI 502 resilience for EVERY pip in this job -- including build-isolation
    # subprocess pips (e.g. flash-attn/setuptools build envs) that don't inherit
    # our explicit --retries flag. These env vars are read by all pip processes.
    os.environ["PIP_RETRIES"] = "10"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "60"
    n_gpu = int(os.environ.get("SM_NUM_GPUS", "1"))
    # For a sample run (<=10 steps), use single GPU to leave memory for LoRA merge
    if int(MAX_STEPS) <= 10:
        n_gpu = 1
    log(f"suite={SUITE} dataset={DATASET} max_steps={MAX_STEPS} "
        f"batch={BATCH} seed={TRAIN_SEED} gpus={n_gpu}")
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

    os.chmod("/tmp", 0o1777)

    # --- Idempotence: if the prebuilt image already has the env, skip install ---
    baked_marker = "/opt/vla/.baked_env"
    if os.path.exists(baked_marker):
        with open(baked_marker) as f:
            marker = f.read().strip()
        log(f"BAKED ENV DETECTED: {marker} -- skipping install")
        # Override WORK to where the baked image installed everything
        global WORK, OFT_DIR, RLDS_DIR
        WORK = "/opt/vla"
        OFT_DIR = os.path.join(WORK, "openvla-oft")
        RLDS_DIR = os.path.join(WORK, "rlds")
    else:
        log("No baked env -- installing from scratch (runtime install path)")
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", "git", "git-lfs"])

        run(["git", "clone", OFT_REPO, OFT_DIR])
        run(["git", "-C", OFT_DIR, "checkout", OFT_COMMIT])

        pip("-e", OFT_DIR)
        pip(TFMD_PIN)
        pip("packaging", "ninja")
        pip(FLASH_ATTN, "--no-build-isolation")
        pip("--force-reinstall", "numpy<2")
        pip(TFMD_PIN)

    # H3 fix (whole-system review): resolve revisions FIRST, then download AT
    # those exact revisions -- the manifest's identity is cryptographically the
    # trained-on identity, not a separately observed HEAD. Env pins override
    # HEAD resolution for full reproducibility.
    base_revision = BASE_REV_PIN or hf_revision(BASE_CHECKPOINT)
    dataset_dir = os.path.join(RLDS_DIR, DATASET)

    if BYO_DATASET:
        # Bring-your-own: sync the caller's RLDS from S3, then VALIDATE it
        # fail-closed before spending GPU time. dataset_identity is ATTESTED
        # by this job (URI + version) -- the gate verifies the WEIGHTS are what
        # training produced, NOT that training consumed the claimed data (that
        # boundary is stated in data/BYO_DATASET.md).
        os.makedirs(dataset_dir, exist_ok=True)
        log(f"BYO dataset: syncing {DATASET_S3URI} -> {dataset_dir}")
        run(["aws", "s3", "sync", DATASET_S3URI, dataset_dir])
        if not any(os.scandir(dataset_dir)):
            log(f"FATAL: BYO dataset sync yielded empty dir: {dataset_dir}")
            sys.exit(1)
    if BYO_DATASET:
        episode_count = validate_rlds_dataset(dataset_dir)
        dataset_source = DATASET_S3URI
        dataset_revision = weights_digest(dataset_dir)
        log(f"BYO dataset OK: {episode_count} episodes, digest "
            f"{dataset_revision[:12]}")
    else:
        # H3 fix: resolve revision FIRST, download AT that exact revision so
        # the manifest identity is the trained-on identity.
        dataset_revision = DATASET_REV_PIN or hf_revision(
            DATASET_REPO, repo_type="dataset")
        log(f"pinned: dataset {DATASET_REPO}@{dataset_revision[:12]}, "
            f"base {BASE_CHECKPOINT}@{base_revision[:12]}")
        log(f"downloading {DATASET} at {dataset_revision[:12]} ...")
        run([sys.executable, "-c",
             "import sys; from huggingface_hub import snapshot_download; "
             "snapshot_download(sys.argv[1], repo_type='dataset', "
             "revision=sys.argv[2], local_dir=sys.argv[3], "
             "allow_patterns=[sys.argv[4] + '/*'])",
             DATASET_REPO, dataset_revision, RLDS_DIR, DATASET])
        dataset_source = f"hf:{DATASET_REPO}:{DATASET}"
        episode_count = None  # derived below via rlds_episode_count
    run(["ls", "-la", dataset_dir])

    base_dir = os.path.join(WORK, "base", base_revision)
    log(f"materializing base checkpoint at {base_revision[:12]} ...")
    run([sys.executable, "-c",
         "import sys; from huggingface_hub import snapshot_download; "
         "snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])",
         BASE_CHECKPOINT, base_revision, base_dir])

    if episode_count is None:  # default-HF path; BYO already counted+validated
        episode_count = rlds_episode_count(dataset_dir)
    log(f"dataset episodes={episode_count}")

    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONHASHSEED"] = TRAIN_SEED
    env["TRAIN_SEED"] = TRAIN_SEED  # validated int-string for the shim

    # Per-rank HF module cache (v1 race fix) + REAL seeding (contract
    # train_seed: upstream FinetuneConfig exposes no seed field, so the shim
    # seeds every RNG the trainer draws from, in-process, before it runs).
    # No parameter text is interpolated into generated source (BLOCKER fix):
    # the shim reads TRAIN_SEED from env, which this process has already
    # validated as an integer; OFT_DIR is a repo constant, not a parameter.
    shim = os.path.join(WORK, "finetune_rank_shim.py")
    with open(shim, "w") as fh:
        fh.write(
            "import os, random, runpy, sys\n"
            "seed = int(os.environ['TRAIN_SEED'])\n"
            "rank = int(os.environ.get('LOCAL_RANK', '0'))\n"
            "random.seed(seed + rank)\n"
            "import numpy as np\n"
            "np.random.seed(seed + rank)\n"
            "import torch\n"
            "torch.manual_seed(seed + rank)\n"
            "torch.cuda.manual_seed_all(seed + rank)\n"
            "os.environ['HF_MODULES_CACHE'] = f'/tmp/hf_modules_rank_{rank}'\n"
            "sys.argv = ['finetune.py'] + sys.argv[1:]\n"
            f"runpy.run_path('{OFT_DIR}/vla-scripts/finetune.py', run_name='__main__')\n"
        )

    # A directory belonging to THIS invocation. It was `WORK/runs`, shared and fixed, and the
    # export below then selected the newest entry anywhere beneath it -- so any retained
    # directory from an earlier or concurrent run could win, and this invocation's manifest
    # would be written around another run's weights. The digest then faithfully binds
    # misattributed bytes, and `meta.eval_only` cannot reveal it: that field records which
    # pipeline graph ran, not whose training produced the weights.
    run_root = tempfile.mkdtemp(prefix="openvla-run-", dir=WORK)
    train_cmd = [
        "torchrun", "--standalone", "--nnodes", "1", "--nproc-per-node", str(n_gpu),
        shim,
        "--vla_path", base_dir,
        "--data_root_dir", RLDS_DIR,
        "--dataset_name", DATASET,
        "--run_root_dir", run_root,
        "--use_l1_regression", "True",
        "--use_diffusion", "False",
        "--use_film", "False",
        "--num_images_in_input", str(INPUT_CONFIG["num_images_in_input"]),
        "--use_proprio", str(INPUT_CONFIG["use_proprio"]),
        "--batch_size", BATCH,
        "--grad_accumulation_steps", GRAD_ACCUM,
        "--learning_rate", "5e-4",
        "--num_steps_before_decay", str(int(int(MAX_STEPS) * 0.7)),
        "--max_steps", MAX_STEPS,
        "--save_freq", MAX_STEPS,
        "--save_latest_checkpoint_only", "True",
        "--image_aug", "True",
        "--lora_rank", LORA_RANK,
        "--shuffle_buffer_size", "10000",
        "--merge_lora_during_training", "True",
        "--run_id_override", "poc",
    ]
    # Trace boundary: the last line reachable without an accelerator. torchrun binds a CUDA
    # device, so on a CPU L3 job this chunk OPENS and never closes -- read_trace reports it as
    # the death point, which is exactly the GPU boundary this rung exists to locate.
    log(f">>> finetune {MAX_STEPS} steps, dataset={DATASET} (torchrun binds a GPU device)")
    log("$ " + " ".join(train_cmd))
    # M4: the finetune is the longest-running work in this job and must honour the SAME
    # runtime budget every other subprocess gets via run(). It was launched with a bare
    # subprocess.run and no timeout, so a stalled trainer held paid accelerator capacity until
    # SageMaker terminated the job -- an opaque external kill rather than a diagnosable timeout.
    # A watchdog kills the whole process group (torchrun spawns children; start_new_session so
    # killpg reaches them) once the deadline passes, matching the pattern the molmoact2 trainer
    # and the Arena evaluator already use.
    timed_out = {"v": False}
    proc = subprocess.Popen(train_cmd, cwd=OFT_DIR, env=env, start_new_session=True)

    def _kill_finetune() -> None:
        timed_out["v"] = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    hang_timer = threading.Timer(_subprocess_timeout(), _kill_finetune)
    hang_timer.start()
    try:
        returncode = proc.wait()
    finally:
        hang_timer.cancel()
    if timed_out["v"]:
        log("FATAL: finetune exceeded the runtime budget and was killed")
        sys.exit(124)
    if returncode != 0:
        log(f"FATAL: finetune exited {returncode}")
        sys.exit(returncode)

    # Selection is confined to the directory this invocation created, and must be
    # UNAMBIGUOUS. Newest-by-mtime was the defect: modification time cannot establish
    # ownership, so a newer neighbour outranked the run that was actually requested. The
    # isdir filter matters too -- the old enumeration accepted plain files, and a stray file
    # selected as the checkpoint fails later with NotADirectoryError rather than here.
    run_dirs = [os.path.join(run_root, d) for d in sorted(os.listdir(run_root))
                if os.path.isdir(os.path.join(run_root, d))]
    if len(run_dirs) != 1:
        log(f"FATAL: expected exactly one run directory under {run_root}, found "
            f"{len(run_dirs)}: {[os.path.basename(d) for d in run_dirs]}. Refusing to guess "
            f"which one this invocation trained.")
        sys.exit(1)
    ckpt_dir = run_dirs[0]
    log(f"checkpoint dir: {ckpt_dir}")
    run(["ls", "-la", ckpt_dir])

    # The destination must be empty. A pre-existing file here would survive the copy below and
    # be digested and registered as part of this checkpoint.
    _existing = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []
    if _existing:
        log(f"FATAL: {MODEL_DIR} is not empty before staging: {sorted(_existing)[:10]}. "
            f"Anything already there would be digested as part of this checkpoint.")
        sys.exit(1)

    import shutil
    for entry in os.listdir(ckpt_dir):
        src = os.path.join(ckpt_dir, entry)
        if entry == "lora_adapter":
            continue
        dst = os.path.join(MODEL_DIR, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)

    freeze_proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if freeze_proc.returncode != 0 or not freeze_proc.stdout.strip():
        log("FATAL: pip freeze failed -- provenance manifest cannot be empty")
        sys.exit(1)
    with open(os.path.join(MODEL_DIR, "runtime_manifest.txt"), "w") as fh:
        fh.write(f"# oft_commit={OFT_COMMIT}\n"
                 f"# dataset={DATASET_REPO}:{DATASET}@{dataset_revision}\n"
                 f"# train_seed={TRAIN_SEED}\n")
        fh.write(freeze_proc.stdout)

    # CONTRACT: digest over everything staged (runtime_manifest.txt included --
    # it is content identity), THEN the manifest (excluded from the digest by
    # name). This is the "digest of the bytes it ACTUALLY WROTE" half of the
    # gate's content binding.
    digest = weights_digest(MODEL_DIR)
    log(f"weights_digest: {digest}")
    manifest = {
        "manifest_version": 1,
        "model_family": "openvla",
        "base_checkpoint": BASE_CHECKPOINT,
        "base_revision": base_revision,
        "train_seed": int(TRAIN_SEED),
        "input_config": INPUT_CONFIG,
        "train_recipe": {"repo": OFT_REPO, "commit": OFT_COMMIT,
                         "max_steps": int(MAX_STEPS)},
        "weights_digest": digest,
        "dataset_manifest": {
            # source is hf:<repo>:<name> for the default pull, or the s3://
            # URI for a bring-your-own dataset (the s3:// prefix already marks
            # BYO -- a separate `byo` key violated the shared validator's
            # DATASET_KEYS {source,revision,episode_count} and rejected every
            # pipeline-mode eval that read this manifest). revision is the HF
            # dataset SHA (default) or a content digest over the synced tree.
            "source": dataset_source,
            "revision": dataset_revision,
            "episode_count": episode_count,
        },
    }
    with open(os.path.join(MODEL_DIR, "checkpoint_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    run(["ls", "-la", MODEL_DIR])
    log("DONE: merged checkpoint + manifest staged for S3 upload")


if __name__ == "__main__":
    main()
