"""SageMaker entry point: GR00T N1.7 LIBERO evaluation.

Run NVIDIA's GPU policy server and the LIBERO rollout client in separate
Python environments inside one container, communicating over localhost.
Evaluate selected tasks against the same server and aggregate the per-episode
boolean results printed by rollout_policy.py. Task IDs and environment names
come from LIBERO's benchmark registry.

Write schema-v3 metrics.json and validate it in-job with the shared report
validator. The later Validate processing step additionally checks artifact
identity and publishes evidence. Installation, launch and parsing failures
exit nonzero.

defaults.json pins the upstream code and base checkpoint. Runtime prerequisites
include authentication for gated Hugging Face backbones, git-lfs for NVIDIA's
wheel dependency, and an FFmpeg binary with libx264 for rollout video.

Config via environment (set by the launcher):
  EVAL_SUITE      selected LIBERO suite (default libero_spatial)
  EVAL_TASK_IDS   selected task IDs, or "all" for the complete suite
  EVAL_TRIALS     episodes per task (EVAL_EPISODES accepted as an alias)
  EVAL_SEED       rollout seed (default 1000)
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
import threading
import signal
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capped_reader import DEFAULT_MAX_ARCHIVE_BYTES, capped_tar_open  # noqa: E402
from digest import weights_digest  # noqa: E402  (shipped in sourcedir)
from source_identity import from_archive as si_from_archive  # noqa: E402
from source_identity import from_snapshot as si_from_snapshot  # noqa: E402
from validator import validate_report  # noqa: E402


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

OUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
# PIPELINE CHANGE: metrics.json + small evidence to SM_MODEL_DIR for step-property binding.
METRICS_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
WORK = "/opt/ml/code"
_BAKED = os.path.isfile("/opt/vla/.baked_env")
if _BAKED:
    WORK = "/opt/vla"
# The baked image clones GR00T to ${WORK}/gr00t, but this evaluator always looked in
# ${WORK}/Isaac-GR00T -- the name of its own RUNTIME clone. In a baked image that path
# does not exist, so the baked environment was never used and the evaluator reinstalled
# everything over the network on an expensive GPU node. The trainer in the same image
# already resolves this correctly; mirror it rather than keeping two answers.
GR00T_DIR = os.path.join(
    WORK, os.environ.get("GROOT_SUBDIR", "gr00t" if _BAKED else "Isaac-GR00T"))
UV_BIN = "uv"

# Pins from spike FINDINGS + defaults.json (single source of truth).
# Read from THIS file's directory because the sourcedir builder stages defaults.json flat
# alongside the entrypoint; the canonical copy lives with the family's train entry. A bare
# FileNotFoundError here names a path and nothing else, which is unactionable: it looks like a
# missing config file when it is actually a STAGING gap, and the two have different fixes.
_defaults_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "defaults.json")
if not os.path.isfile(_defaults_path):
    raise RuntimeError(
        f"defaults.json is absent from this entrypoint's own directory ({_defaults_path}). "
        f"It carries the pinned repo commit and checkpoint revision, and the sourcedir builder "
        f"is what stages it flat next to this file at submit time; the canonical copy lives with "
        f"the family's train entry. Its absence means the sourcedir was not built, or was built "
        f"without this family's files -- not that a setting is missing. Running from a bare "
        f"checkout will always hit this, because the flat layout only exists after staging.")
with open(_defaults_path) as _fh:
    _D = json.load(_fh)
GR00T_REPO = _D["repo"]
GR00T_COMMIT = _D["repo_commit"]
HF_REPO = _D["checkpoint"]
HF_REVISION = _D["checkpoint_hf_revision"]  # immutable Hub commit (pinned)
CKPT_SUBDIR = _D["checkpoint_subdir"]
EMBODIMENT_TAG = _D["embodiment_tag"]
SERVER_PORT = _D["server_port"]
# The seeded server wrapper, delivered by stage() into this sourcedir (flattened to
# root, so it sits beside this file). Unlike Arena, this image does not bake it.
# Overridable for the baked case; absence is fatal rather than a silent fallback to
# an unseeded, unaudited server.
SEEDED_SERVER_ENTRY = os.environ.get(
    "GR00T_SEEDED_SERVER_ENTRY",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "gr00t_seeded_server.py"))
PYTHON_PIN = "3.12"
_gr00t_version = os.environ.get("EVAL_GR00T_VERSION", "n17").strip().lower()
if _gr00t_version != "n17":
    print(f"[gr00t-libero-eval] FATAL: GR00T version {_gr00t_version!r} is not "
          "supported for LIBERO evaluation. This evaluator uses the N1.7 GR00T code "
          "and model type (Gr00tN1d7). N1.6 checkpoints (Gr00tN1d6) require a "
          "different model loader. Use --gr00t-version n17 or Arena for N1.6.",
          flush=True)
    sys.exit(1)

FAMILY_SCHEMAS = {
    _D["family"]: {
        "input_config_schema": _D["input_config_schema"],
        "provenance_keys": _D["provenance_keys"],
    },
}

SUITE = os.environ.get("EVAL_SUITE", "libero_spatial")
TASK_IDS = os.environ.get("EVAL_TASK_IDS", "")
# Contract interface names: EVAL_TRIALS canonical, EVAL_EPISODES deprecated
# alias accepted ONLY when EVAL_TRIALS unset; a conflict is fatal.
_trials = os.environ.get("EVAL_TRIALS", "")
_episodes_alias = os.environ.get("EVAL_EPISODES", "")
if _trials and _episodes_alias and _trials != _episodes_alias:
    print(f"[gr00t-eval] FATAL: EVAL_TRIALS={_trials} conflicts with "
          f"EVAL_EPISODES={_episodes_alias}", flush=True)
    sys.exit(1)
EPISODES = _trials or _episodes_alias or "20"
SEED = os.environ.get("EVAL_SEED", "1000")
MAX_EPISODE_STEPS = os.environ.get("EVAL_MAX_EPISODE_STEPS", "720")
N_ACTION_STEPS = os.environ.get("EVAL_N_ACTION_STEPS", "8")
# Published-checkpoint override (contract: both-or-neither, revision enforced).
_ckpt_override = os.environ.get("EVAL_CHECKPOINT", "")
_rev_override = os.environ.get("EVAL_CKPT_REV", "")
# PIPELINE / FINE-TUNED mode (dose ladder): when EVAL_MODEL_SOURCE_URI is set,
# a fine-tuned model.tar.gz is mounted at the "model" channel and evaluated in
# place -- no HF download. EVAL_CHECKPOINT is then a LOCAL path (starts with /)
# and revision is null (the artifact identity + weights digest are the binding).
SOURCE_URI = os.environ.get("EVAL_MODEL_SOURCE_URI", "")
PIPELINE_MODE = _ckpt_override.startswith("/") if _ckpt_override else False
if PIPELINE_MODE:
    if not SOURCE_URI:
        print("[gr00t-eval] FATAL: local EVAL_CHECKPOINT requires "
              "EVAL_MODEL_SOURCE_URI (artifact identity binding)", flush=True)
        sys.exit(1)
    HF_REPO = _ckpt_override
    CKPT_SUBDIR = ""
elif _ckpt_override or _rev_override:
    if not (_ckpt_override and _rev_override):
        print("[gr00t-eval] FATAL: EVAL_CHECKPOINT and EVAL_CKPT_REV must be "
              "overridden together", flush=True)
        sys.exit(1)
    HF_REPO = _ckpt_override
    # A comment elsewhere in this file claimed "the revision is already validated as a 40-hex commit
    # at import". That was true only for OpenVLA (openvla/eval_entry.py:132); this override path had
    # no such check, so a branch or tag name reached the download and the report published a moving
    # reference as though it pinned bytes. Checked here, before anything is fetched.
    if not re.fullmatch(r"[0-9a-f]{40}", _rev_override or ""):
        print(f"[gr00t-eval] FATAL: EVAL_CKPT_REV must be a resolved 40-hex commit, got "
              f"{_rev_override!r} -- a branch or tag identifies different bytes over time",
              flush=True)
        sys.exit(1)
    HF_REVISION = _rev_override


def log(msg: str) -> None:
    print(f"[gr00t-eval] {msg}", flush=True)


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
    # A call with no timeout can hang for the whole budget and produce nothing; the step then
    # looks like a long job rather than a stuck one.
    kw.setdefault("timeout", _subprocess_timeout())
    subprocess.run(cmd, **kw)


def clean_env() -> dict:
    # Strip the DLC's py3.10 PYTHONPATH -- it poisons uv-managed py3.12
    # ("SRE module mismatch", proven in prior spikes).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("UV_HTTP_TIMEOUT", "120")
    env.setdefault("UV_HTTP_RETRIES", "10")
    # Belt-and-suspenders HF token propagation: older huggingface_hub versions
    # read HUGGING_FACE_HUB_TOKEN; newer ones read HF_TOKEN. Set both.
    # Also write a token file that huggingface_hub.login() would create.
    hf_token = env.get("HF_TOKEN", "")
    if hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        # Write token file for huggingface_hub cache-based auth
        hf_home = env.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        token_path = os.path.join(hf_home, "token")
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w") as f:
            f.write(hf_token)
        # Also write to absolute default path in case uv subprocess has different HF_HOME
        root_token_path = "/root/.cache/huggingface/token"
        os.makedirs(os.path.dirname(root_token_path), exist_ok=True)
        with open(root_token_path, "w") as f:
            f.write(hf_token)
    return env


def uv_sync_resilient(sync_args: list, env: dict, cwd: str) -> None:
    """`uv sync` with process-level retry + backoff (transient PyPI 502 on a big
    wheel killed a molmoact2 eval even with UV_HTTP_RETRIES). Resumable; fail-
    closed after the last attempt."""
    delay = 30
    for attempt in range(1, 9):
        out = subprocess.run([UV_BIN, *sync_args], cwd=cwd, env=env)
        if out.returncode == 0:
            if attempt > 1:
                log(f"uv sync OK on attempt {attempt}")
            return
        if attempt < 8:
            log(f"uv sync failed (rc={out.returncode}); PyPI blip? backoff "
                f"{delay}s ({attempt}/8)")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    log("FATAL: uv sync failed after 8 attempts")
    sys.exit(1)


def _dump(path: str, label: str) -> None:
    try:
        with open(path) as fh:
            text = fh.read()
        log(f"---- {label} (last 8000 chars) ----")
        print(text[-8000:], flush=True)
        log(f"---- end {label} ----")
    except OSError as e:
        log(f"could not read {label}: {e}")


def wait_for_port(port: int, timeout_s: int, proc: subprocess.Popen,
                  server_log_path: str) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            _dump(server_log_path, "server log")
            log(f"FATAL: server exited early rc={proc.returncode}")
            sys.exit(1)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                log(f"server port {port} open")
                return
        except OSError:
            time.sleep(5)
    _dump(server_log_path, "server log")
    log(f"FATAL: server port {port} not open after {timeout_s}s")
    sys.exit(1)


# The one structured record upstream emits per rollout:
#   results:  ('<env_name>', [True, False, ...], {...})
# Anchored on the `results:` label AND the quoted environment name, so a bare
# bracketed boolean list appearing anywhere else in the output cannot match.
def _patch_success_telemetry(src: str) -> str:
    """Make UNOBSERVED success telemetry invalidate the episode, at the right boundary.

    Pinned upstream accepts success from THREE places, in this order:

      1. per step, top-level ``env_infos["success"][env_idx]``
      2. per step, ``env_infos["final_info"][env_idx]["success"]``
      3. at episode end, ``env_infos["final_info"][env_idx]["success"]`` again

    and when (1) is absent it did ``current_successes[env_idx] = False``. Two problems with
    that: a missing metric signal became an ordinary zero score, which at the deliberately
    supported zero threshold can register; and because it ASSIGNS rather than accumulates,
    one step without the top-level key ERASED a success already recorded for that episode.

    An earlier version of this patch raised in that else-branch. That was wrong: it fired
    before (2) and (3) could contribute, so a run reporting success only at episode end --
    the normal terminal-telemetry pattern -- crashed instead of scoring. The absence
    decision belongs at the EPISODE boundary, once every source has had its chance.

    So this patch:
      * stops the else-branch clobbering accumulated success (it becomes a no-op),
      * records whether ANY authoritative source was observed for each env,
      * raises at the episode boundary if none was, and
      * resets that tracker with the other per-episode trackers.

    "The task failed" and "the evaluator never observed success telemetry" are then
    distinguishable, and a legitimate all-false outcome still records normally.

    Raises RuntimeError if any anchor is absent: upstream having moved means the guarantee
    cannot be made, which must stop the run rather than proceed.
    """
    edits = [
        # 1. A per-env tracker beside the successes tracker.
        (
            "    current_successes = [False] * n_envs\n",
            "    current_successes = [False] * n_envs\n"
            "    # Whether an authoritative success signal was OBSERVED this episode.\n"
            "    _telemetry_seen = [False] * n_envs\n",
        ),
        # 2. Top-level source: record the observation, and stop the else clobbering.
        (
            "                    current_successes[env_idx] |= bool(env_success)\n"
            "                else:\n"
            "                    current_successes[env_idx] = False\n",
            "                    current_successes[env_idx] |= bool(env_success)\n"
            "                    _telemetry_seen[env_idx] = True\n"
            "                else:\n"
            "                    # No top-level signal on THIS step. Do not clobber an\n"
            "                    # accumulated success, and do not decide absence here:\n"
            "                    # final_info may still supply it below or at episode end.\n"
            "                    pass\n",
        ),
        # 3. Per-step final_info source: record the observation.
        (
            "                    current_successes[env_idx] |= bool(env_success)\n"
            "                current_rewards[env_idx] += rewards[env_idx]\n",
            "                    current_successes[env_idx] |= bool(env_success)\n"
            "                    _telemetry_seen[env_idx] = True\n"
            "                current_rewards[env_idx] += rewards[env_idx]\n",
        ),
        # 4. Episode-end final_info source: record the observation.
        (
            "                        current_successes[env_idx] |= any(\n"
            "                            env_infos[\"final_info\"][env_idx][\"success\"]\n"
            "                        )\n",
            "                        current_successes[env_idx] |= any(\n"
            "                            env_infos[\"final_info\"][env_idx][\"success\"]\n"
            "                        )\n"
            "                        _telemetry_seen[env_idx] = True\n",
        ),
        # 5. The boundary decision, before the outcome is recorded.
        (
            "                    episode_successes.append(current_successes[env_idx])\n",
            "                    if not _telemetry_seen[env_idx]:\n"
            "                        raise RuntimeError(\n"
            "                            \"no success telemetry was observed for the episode \"\n"
            "                            f\"completed on env {env_idx}. An episode whose \"\n"
            "                            \"success was never reported has not been measured; \"\n"
            "                            \"refusing to record it as an unsuccessful episode.\")\n"
            "                    episode_successes.append(current_successes[env_idx])\n",
        ),
        # 6. Reset the tracker with the others.
        (
            "                    current_successes[env_idx] = False\n"
            "                    # only update completed_episodes if valid\n",
            "                    current_successes[env_idx] = False\n"
            "                    _telemetry_seen[env_idx] = False\n"
            "                    # only update completed_episodes if valid\n",
        ),
    ]
    for index, (target, replacement) in enumerate(edits, start=1):
        if src.count(target) != 1:
            raise RuntimeError(
                f"success-telemetry patch anchor {index} occurs {src.count(target)} times "
                f"in _collect_rollout_episodes (expected exactly 1); the pinned GR00T rollout policy "
                f"changed. Unobserved telemetry could again be recorded as a failed "
                f"episode, so refusing to run.")
        src = src.replace(target, replacement, 1)
    return src


_TELEMETRY_VALIDATOR = '''

def _require_boolean_success(value, where):
    """Require success telemetry to BE a boolean measurement before it is coerced.

    Upstream converts whatever it finds with bool()/any(), so an invalid value silently
    became an ordinary outcome and no later check could recover the original. Probes against
    the pinned upstream showed [[]] recorded a completed failed episode, an array containing
    NaN recorded a success, and the integer 2 recorded a success -- none of which is a
    measurement of whether the task was achieved.

    Accepts a boolean scalar, or a non-empty array/sequence whose dtype is boolean, and
    returns the reduced boolean. Everything else raises, naming the value and the site.
    """
    import numpy as _np

    if isinstance(value, (bool, _np.bool_)):
        return bool(value)
    if isinstance(value, (list, tuple, _np.ndarray)):
        array = _np.asarray(value)
        if array.size == 0:
            raise ValueError(
                f"success telemetry at {where} is empty ({value!r}), so no episode outcome "
                "was reported. An absent measurement must not become a failed episode.")
        if array.dtype != _np.bool_:
            raise ValueError(
                f"success telemetry at {where} has dtype {array.dtype} ({value!r}), not "
                "boolean. Numeric or object values are coerced by bool()/any() into an "
                "outcome they do not express -- NaN becomes success and 2 becomes success.")
        return bool(array.any())
    raise ValueError(
        f"success telemetry at {where} is {type(value).__name__} ({value!r}), not a boolean "
        "or a boolean array. Refusing to coerce it into an episode outcome.")
'''


def _patch_telemetry_shape(src: str) -> str:
    """Validate success telemetry at the OBSERVATION boundary, before any coercion.

    The telemetry patch marks a source observed, but did so after upstream's
    bool()/any() conversion, so a malformed value was already an ordinary outcome by then.
    This inserts a shape and type check at each raw read, before conversion.

    Raises RuntimeError if any anchor is absent, since upstream having moved means malformed
    telemetry could again pass as a measurement.
    """
    edits = [
        # Top-level per-step source.
        (
            '                    env_success = env_infos["success"][env_idx]\n',
            '                    env_success = _require_boolean_success(\n'
            '                        env_infos["success"][env_idx], "info[\'success\']")\n',
        ),
        # Per-step final_info source.
        (
            '                    env_success = env_infos["final_info"][env_idx]["success"]\n',
            '                    env_success = _require_boolean_success(\n'
            '                        env_infos["final_info"][env_idx]["success"],\n'
            '                        "final_info[\'success\']")\n',
        ),
        # Episode-end final_info source.
        (
            '                        current_successes[env_idx] |= any(\n'
            '                            env_infos["final_info"][env_idx]["success"]\n'
            '                        )\n',
            '                        current_successes[env_idx] |= _require_boolean_success(\n'
            '                            env_infos["final_info"][env_idx]["success"],\n'
            '                            "final_info[\'success\'] at episode end")\n',
        ),
    ]
    for index, (target, replacement) in enumerate(edits, start=1):
        if src.count(target) != 1:
            raise RuntimeError(
                f"telemetry-shape patch anchor {index} occurs {src.count(target)} times in "
                f"the pinned GR00T rollout policy (expected exactly 1); upstream changed. "
                f"Malformed telemetry could again be coerced into an outcome, so refusing "
                f"to run.")
        src = src.replace(target, replacement, 1)
    # C2 (cycle 12): this APPENDED the validator to the end of the file. The pinned upstream
    # already carries its own `if __name__ == "__main__":` block, so a helper appended after it
    # is defined only after the script has finished running -- the patched calls raised
    # NameError: name '_require_boolean_success' is not defined, on the NORMAL path, the moment
    # success telemetry was read. The guard could never have fired; it would only ever have
    # crashed the evaluation it was added to protect.
    #
    # Inserted before the main block instead. Anchored on the guard clause and required to occur
    # exactly once, so an upstream change that moves or duplicates it fails loudly here rather
    # than silently restoring the append-after-main ordering.
    main_guard = '\nif __name__ == "__main__":'
    count = src.count(main_guard)
    if count != 1:
        raise RuntimeError(
            f"pinned GR00T rollout policy has {count} top-level main guards (expected 1); "
            f"the telemetry validator must be defined BEFORE the script runs, and the "
            f"insertion point is no longer unambiguous. Refusing to run.")
    head, tail = src.split(main_guard, 1)
    return (head.rstrip("\n") + "\n\n" + _TELEMETRY_VALIDATOR.strip("\n")
            + "\n" + main_guard + tail)


_RESULT_JSON_ENV = "GR00T_RESULT_JSON"


def _patch_result_file(src: str) -> str:
    """Make the rollout write a structured result FILE instead of relying on stdout.

    stdout is a shared, lossy, untrusted channel: it interleaves with every library that
    logs, it can be truncated mid-line by a flush, and anything may print text that looks
    like a result. Scraping it for the score means arbitrary log output can influence the
    reported outcome, and no digest can detect that -- the checkpoint is genuine and only
    the outcome is fabricated by parsing.

    The rollout already computes the authoritative tuple `(env_name, episode_successes,
    episode_infos)`. This writes it, as JSON, to the path named by GR00T_RESULT_JSON,
    atomically via a temp file and os.replace so a partially written file can never be
    read as complete. The wrapper then reads THAT and never parses stdout for the score.

    Raises RuntimeError if the anchor is absent, since upstream having moved means the
    result file would silently not be produced.
    """
    anchor = '    print("results: ", results)\n'
    if src.count(anchor) != 1:
        raise RuntimeError(
            f"result-file patch anchor occurs {src.count(anchor)} times in the pinned "
            f"GR00T rollout policy (expected exactly 1); upstream changed. The structured "
            f"result file would not be written, so refusing to run.")
    injected = (
        '    _result_path = os.environ.get("GR00T_RESULT_JSON")\n'
        '    if not _result_path:\n'
        '        raise RuntimeError(\n'
        '            "GR00T_RESULT_JSON is not set, so the authoritative result file "\n'
        '            "cannot be written and the wrapper would have nothing to read. "\n'
        '            "Refusing to finish a rollout whose outcome cannot be recorded.")\n'
        '    _env_name, _successes, _ = results\n'
        '    _payload = {\n'
        '        "schema": "gr00t_rollout_result_v1",\n'
        '        "env_name": _env_name,\n'
        '        "episode_successes": [bool(_s) for _s in _successes],\n'
        '        "episodes": len(_successes),\n'
        '        "successes": int(sum(bool(_s) for _s in _successes)),\n'
        '    }\n'
        '    _tmp_path = _result_path + ".partial"\n'
        '    with open(_tmp_path, "w") as _rf:\n'
        '        json.dump(_payload, _rf)\n'
        '        _rf.flush()\n'
        '        os.fsync(_rf.fileno())\n'
        '    os.replace(_tmp_path, _result_path)\n'
        '    print("wrote structured result file:", _result_path)\n'
    )
    src = src.replace(anchor, injected + anchor, 1)
    # The injected code needs both modules; upstream may not import either.
    for module in ("json", "os"):
        if not re.search(rf"^import {module}$", src, re.MULTILINE):
            src = f"import {module}\n" + src
    return src


def _read_result_file(path: str, expected_env: str, expected_episodes: int):
    """Read the per-episode success vector from the rollout's structured result file.

    Replaces stdout scraping. The previous regex was neither anchored nor a parser for the
    complete record: `results:` matched as a substring, so `debug anticipated results:
    ('env', [True, True]` was accepted, and the pattern stopped at the closing bracket
    without requiring the rest of the tuple, so a line truncated by a log flush parsed as a
    complete result.

    Validates identity and shape, and cross-checks the recorded count against the vector so
    a payload that disagrees with itself cannot pass. Raises ValueError naming the reason.
    """
    if not os.path.exists(path):
        raise ValueError(
            f"the rollout produced no structured result file at {path}. Its outcome is "
            f"therefore unknown, and stdout is not an acceptable substitute -- any log "
            f"line can imitate a result record. Refusing to infer successes.")
    with open(path) as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"the structured result file at {path} is not valid JSON ({exc}). A "
                f"truncated or corrupt result must not be interpreted.") from exc
    if payload.get("schema") != "gr00t_rollout_result_v1":
        raise ValueError(
            f"result file schema is {payload.get('schema')!r}, expected "
            f"'gr00t_rollout_result_v1'. Refusing to read an unrecognised result format.")
    env = payload.get("env_name")
    if env != expected_env:
        raise ValueError(
            f"result file is for environment {env!r} but this rollout requested "
            f"{expected_env!r}. The success vector describes a different task.")
    vector = payload.get("episode_successes")
    if not isinstance(vector, list) or not vector:
        raise ValueError(
            f"result file for {env!r} has no episode_successes list; there is no outcome "
            f"to record.")
    if not all(isinstance(flag, bool) for flag in vector):
        raise ValueError(
            f"result file for {env!r} has non-boolean episode outcomes ({vector!r}). An "
            f"outcome that is not a boolean has not been measured.")
    if len(vector) != expected_episodes:
        raise ValueError(
            f"result file for {env!r} reports {len(vector)} episode outcomes but "
            f"{expected_episodes} episodes were requested.")
    # The file carries a count as well as a vector; disagreement means the producer is
    # inconsistent and neither value can be trusted.
    if payload.get("episodes") != len(vector):
        raise ValueError(
            f"result file for {env!r} records episodes={payload.get('episodes')!r} but "
            f"carries {len(vector)} outcomes; the payload disagrees with itself.")
    if payload.get("successes") != sum(vector):
        raise ValueError(
            f"result file for {env!r} records successes={payload.get('successes')!r} but "
            f"its vector sums to {sum(vector)}; the payload disagrees with itself.")
    return vector


def enumerate_tasks(libero_py: str, env: dict) -> list[tuple[int, str]]:
    """Return [(task_id, env_name), ...] from LIBERO's OWN registry, in the
    canonical task_id order the sim uses -- not a hardcoded list. This is the
    same iteration register_libero_envs() does (get_task(task_id).name)."""
    probe = (
        "import builtins, json, sys\n"
        "builtins.input = lambda *a, **k: 'N'\n"
        "from libero.libero import benchmark\n"
        "suite = benchmark.get_benchmark_dict()[sys.argv[1]]()\n"
        "n = suite.n_tasks\n"
        "out = [[i, 'libero_sim/' + suite.get_task(i).name] "
        "for i in range(n)]\n"
        "print('TASKLIST_JSON=' + json.dumps(out))\n"
    )
    res = subprocess.run([libero_py, "-c", probe, SUITE], cwd=GR00T_DIR, env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        log(f"FATAL: could not enumerate {SUITE} tasks:\n{res.stderr[-1500:]}")
        sys.exit(1)
    line = next((ln for ln in res.stdout.splitlines()
                 if ln.startswith("TASKLIST_JSON=")), None)
    if not line:
        log(f"FATAL: task enumeration produced no TASKLIST_JSON:\n{res.stdout[-1500:]}")
        sys.exit(1)
    tasks = json.loads(line[len("TASKLIST_JSON="):])
    if not tasks:
        log(f"FATAL: {SUITE} enumerated to zero tasks")
        sys.exit(1)
    return [(int(t[0]), str(t[1])) for t in tasks]


def checkpoint_input_config(manifest: dict) -> dict[str, str]:
    config = manifest["input_config"]
    settings = {}
    for key, variable in (
        ("embodiment_tag", "EVAL_EMBODIMENT_TAG"),
        ("n_action_steps", "EVAL_N_ACTION_STEPS"),
        ("max_episode_steps", "EVAL_MAX_EPISODE_STEPS"),
    ):
        value = config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"checkpoint input_config.{key} must be a nonempty string")
        if key != "embodiment_tag" and (not value.isdecimal() or int(value) < 1):
            raise ValueError(f"checkpoint input_config.{key} must be a positive integer string")
        if variable in os.environ and os.environ[variable] != value:
            raise ValueError(f"{variable} conflicts with checkpoint input_config.{key}")
        settings[key] = value
    return settings


def _seed_scope_for_report():
    """Report what the seeding wrapper actually applied, validated rather than assumed.

    I7: this file reported the bare string "client_and_env". The server now starts through the
    seeding wrapper, so the same structured evidence Arena reports exists here -- but only if
    it passes the same checks. Any nonempty JSON object used to count as evidence elsewhere,
    and a stale file from an earlier launch was indistinguishable from proof.
    """
    evidence_path = os.environ.get(
        "GR00T_SERVER_SEED_EVIDENCE", "/tmp/gr00t_server_seed_evidence.json")
    unverified_reason = None
    candidate = None
    try:
        with open(evidence_path) as handle:
            candidate = json.load(handle)
    except (OSError, ValueError) as exc:
        unverified_reason = f"no readable evidence at {evidence_path} ({exc})"

    server_seeding = None
    if candidate is not None:
        problems = []
        if not isinstance(candidate, dict):
            problems.append(f"evidence is {type(candidate).__name__}, not an object")
        else:
            if candidate.get("evidence_schema") != "gr00t_server_evidence_v1":
                problems.append(
                    f"schema is {candidate.get('evidence_schema')!r}, expected "
                    f"'gr00t_server_evidence_v1'")
            if str(candidate.get("seed")) != str(SEED):
                problems.append(
                    f"records seed {candidate.get('seed')!r} but this run requested {SEED}")
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
        log(f"WARNING: server seed evidence not usable ({unverified_reason}); recording the "
            f"policy RNG as unverified rather than claiming it was bound")
    return {
        "client_and_env": True,
        "server_policy_rng": server_seeding is not None,
        "server_seeding_evidence": server_seeding,
        "server_policy_rng_unverified_reason": unverified_reason,
        # The server is seeded once per run; episode resets do not reseed.
        "per_episode_replayable": False,
    }


# The source identity of what this invocation loaded, CONSTRUCTED once by whichever mode resolved
# the checkpoint. Previously two seedable dicts whose emptiness DECIDED the mode, so seeding one in a
# fixture made the other branch unexecutable and hid I6 behind a green suite.
_RESOLVED = None
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


def main() -> None:
    embodiment_tag = EMBODIMENT_TAG
    n_action_steps = N_ACTION_STEPS
    max_episode_steps = MAX_EPISODE_STEPS
    # PyPI 502 resilience for EVERY pip in this job (incl. build-isolation
    # subprocess pips that ignore an explicit --retries flag).
    os.environ["PIP_RETRIES"] = "10"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "60"
    log(f"suite={SUITE} task_ids={TASK_IDS or 'ALL'} "
        f"episodes={EPISODES}/task seed={SEED}")
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

    # resolve HF token from Secrets Manager at runtime
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        secret_name = os.environ.get("HF_SECRET_NAME", "")
        if secret_name:
            import boto3 as _boto3
            _region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
            _client = _boto3.client("secretsmanager", region_name=_region)
            hf_token = _client.get_secret_value(SecretId=secret_name)["SecretString"].strip()
            os.environ["HF_TOKEN"] = hf_token
            log(f"Resolved HF token from secret '{secret_name}'")
    if not hf_token:
        log("WARNING: No HF_TOKEN -- gated model access will fail")

    # ---- [1/7] System libs (EGL render + build tooling for LIBERO deps)
    os.chmod("/tmp", 0o1777)
    # LIBERO's robomimic dep `egl-probe` (v1.0.2) ships a CMakeLists with an
    # ancient `cmake_minimum_required` (<3.5). The pip build-isolation CMake is
    # now 4.x, which REMOVED compatibility with those old policy versions, so
    # `cmake ..` errors out ("Compatibility with CMake < 3.5 has been removed")
    # -> no Makefile -> `make` fails -> setup_libero.sh exits 1. CMake honors this
    # env var as the documented escape hatch; set it before any build subprocess
    # (clean_env() snapshots os.environ, so setup_libero.sh + its uv/pip build
    # inherit it).
    os.environ["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
    # I17: a baked image must not repeat the installation it already contains. This block
    # ran unconditionally, so an expensive GPU job depended on apt, pip, a clone, a uv
    # sync and the LIBERO setup script all succeeding over the network -- and a successful
    # image build was therefore not evidence that the evaluation runtime worked. The
    # locals these sections derive (env, UV_BIN, libero_py) are still computed either way;
    # only the network operations are skipped.
    if not _BAKED:
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "--no-install-recommends",
             "git", "git-lfs", "curl", "ca-certificates", "libegl1", "libgles2",
             "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "ffmpeg",
             "build-essential", "cmake"])
    else:
        log("BAKED ENV DETECTED -- skipping system libs (section 1)")

    env = clean_env()

    # ---- [2/7] uv (pinned) + pinned Isaac-GR00T clone (LFS wheel materialized)
    global UV_BIN
    if not _BAKED:
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "--retries", "10", "--timeout", "60", "uv==0.12.1"])
        UV_BIN = os.path.join(os.path.dirname(sys.executable), "uv")
        run([UV_BIN, "--version"])
        run(["git", "lfs", "install", "--skip-repo"])
        clone_env = dict(env, GIT_LFS_SKIP_SMUDGE="1")
        run(["git", "clone", GR00T_REPO, GR00T_DIR], env=clone_env)
        run(["git", "-C", GR00T_DIR, "checkout", GR00T_COMMIT], env=clone_env)
    else:
        UV_BIN = os.path.join(os.path.dirname(sys.executable), "uv")
        if not os.path.isdir(GR00T_DIR):
            log(f"FATAL: baked marker present but {GR00T_DIR} is missing. The image does "
                f"not contain the environment its marker claims; refusing to fall back to "
                f"a runtime install on an evaluation node.")
            sys.exit(1)
        log(f"BAKED ENV DETECTED -- using {GR00T_DIR} (sections 2-4 skipped)")
    # The commit check runs on BOTH paths: a baked image must be the pinned commit too.
    head = subprocess.run(["git", "-C", GR00T_DIR, "rev-parse", "HEAD"],
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != GR00T_COMMIT:
        log(f"FATAL: repo commit mismatch: {head}")
        sys.exit(1)
    # I2: make absent success telemetry fatal in the pinned rollout loop. Applied on BOTH
    # paths -- a baked image carries the same unpatched upstream file, and the patch is
    # idempotent because the target text is gone once applied.
    _rollout_py = os.path.join(GR00T_DIR, "gr00t/eval/rollout_policy.py")
    with open(_rollout_py) as _rf:
        _rollout_src = _rf.read()
    _rollout_dirty = False
    if "no success telemetry was observed" not in _rollout_src:
        try:
            _rollout_src = _patch_success_telemetry(_rollout_src)
        except RuntimeError as exc:
            log(f"FATAL: {exc}")
            sys.exit(1)
        _rollout_dirty = True
        log("PATCHED _collect_rollout_episodes: absent success telemetry is now fatal")
    else:
        log("_collect_rollout_episodes already patched for success telemetry")
    # C4: validate telemetry SHAPE at the observation boundary, before upstream's
    # bool()/any() conversion turns a malformed value into an ordinary outcome. Must run
    # BEFORE the telemetry patch's own edits are relied upon, and is idempotent by content.
    if "_require_boolean_success" not in _rollout_src:
        try:
            _rollout_src = _patch_telemetry_shape(_rollout_src)
        except RuntimeError as exc:
            log(f"FATAL: {exc}")
            sys.exit(1)
        _rollout_dirty = True
        log("PATCHED rollout: success telemetry is validated before coercion")
    else:
        log("rollout already patched to validate telemetry shape")
    # I5: make the rollout write its authoritative outcome to a structured FILE, so the
    # score never comes from scraping stdout. Same both-paths, idempotent treatment.
    if "gr00t_rollout_result_v1" not in _rollout_src:
        try:
            _rollout_src = _patch_result_file(_rollout_src)
        except RuntimeError as exc:
            log(f"FATAL: {exc}")
            sys.exit(1)
        _rollout_dirty = True
        log("PATCHED rollout: writes a structured result file")
    else:
        log("rollout already patched to write a structured result file")
    if _rollout_dirty:
        with open(_rollout_py, "w") as _rf:
            _rf.write(_rollout_src)
    if not _BAKED:
        run(["git", "-C", GR00T_DIR, "lfs", "pull",
             "--include", "scripts/deployment/**/wheels/*.whl"], env=env)
        wheel = os.path.join(GR00T_DIR, "scripts/deployment/dgpu/wheels/"
                             "torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl")
        if os.path.getsize(wheel) < 100_000:
            log(f"FATAL: LFS wheel still a pointer file: {wheel}")
            sys.exit(1)

        # ---- [3/7] Server venv: gr00t package at pinned python 3.12
        uv_sync_resilient(["sync", "--python", "3.12"], env=env, cwd=GR00T_DIR)

        # ---- [4/7] Client venv: LIBERO sim env via NVIDIA's own setup script
        run(["bash", "gr00t/eval/sim/LIBERO/setup_libero.sh"],
            cwd=GR00T_DIR, env=env)
    libero_py = os.path.join(
        GR00T_DIR, "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python")
    if not os.path.exists(libero_py):
        if _BAKED:
            # The marker asserts capabilities. If the LIBERO client interpreter is absent,
            # the IMAGE is wrong -- say so, rather than reporting an opaque missing-file
            # error or silently reinstalling on a GPU node. The root `uv sync` does not
            # create this venv; upstream's setup_libero.sh does, and the Dockerfile now
            # runs and verifies it before writing the marker.
            log(f"FATAL: baked image claims the gr00t environment but the LIBERO CLIENT "
                f"interpreter is missing at {libero_py}. The image did not run "
                f"setup_libero.sh, so its marker overstates what it contains. Rebuild the "
                f"gr00t image; do not fall back to installing the client at evaluation "
                f"time.")
            _baked_marker = "/opt/vla/.baked_env"
            if os.path.isfile(_baked_marker):
                with open(_baked_marker) as _bf:
                    log(f"       marker says: {_bf.read().strip()!r} "
                        f"(expected it to include 'libero-client-verified')")
            sys.exit(1)
        log(f"FATAL: LIBERO venv python missing: {libero_py}")
        sys.exit(1)

    # ---- ffmpeg resolution (spike root-cause): symlink imageio-ffmpeg's
    # libx264-capable binary onto PATH, then PROBE it (fail loud if absent).
    ff_bin_dir = os.path.join(WORK, "ffbin")
    os.makedirs(ff_bin_dir, exist_ok=True)
    imageio_ff = subprocess.run(
        [libero_py, "-c",
         "import imageio_ffmpeg,sys; sys.stdout.write("
         "imageio_ffmpeg.get_ffmpeg_exe())"],
        capture_output=True, text=True, env=env)
    if imageio_ff.returncode != 0 or not imageio_ff.stdout.strip():
        log("FATAL: could not resolve imageio-ffmpeg binary: "
            f"{imageio_ff.stderr[-300:]}")
        sys.exit(1)
    ff_path = imageio_ff.stdout.strip()
    link = os.path.join(ff_bin_dir, "ffmpeg")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(ff_path, link)
    env["PATH"] = ff_bin_dir + ":" + env.get("PATH", "")
    enc = subprocess.run([link, "-hide_banner", "-encoders"],
                         capture_output=True, text=True, env=env)
    if "libx264" not in enc.stdout:
        log("FATAL: resolved ffmpeg lacks libx264 -- video wrapper's -crf "
            f"would fail. encoders head:\n{enc.stdout[:500]}")
        sys.exit(1)
    log("ffmpeg libx264 present -- video recording will honor -crf")

    # ---- [5/7] Checkpoint resolution: PIPELINE mode (mounted fine-tuned ckpt)
    # or PUBLISHED mode (HF download with subfolder).
    model_artifact_identity = None
    if PIPELINE_MODE:
        model_path = os.environ.get("SM_CHANNEL_MODEL",
                                    "/opt/ml/input/data/model")
        if not os.path.isdir(model_path):
            log(f"FATAL: mounted model channel missing: {model_path!r}")
            sys.exit(1)
        # SageMaker mounts the training output as a RAW model.tar.gz -- it does
        # NOT auto-extract. Extract it so the checkpoint (and its manifest,
        # which lives INSIDE the tarball) is visible (same proven path as
        # OpenVLA/MolmoAct2 evals).
        _tar = os.path.join(model_path, "model.tar.gz")
        if os.path.isfile(_tar):
            import tarfile
            _ex = os.path.join(WORK, "model")
            os.makedirs(_ex, exist_ok=True)
            log(f"extracting mounted {_tar} -> {_ex}")
            # schema 3: measure the archive BEFORE extraction. A tree digest excludes
            # manifests, logs and hidden paths, so two different archives can share
            # one; and HeadObject reports the object current when HEAD runs, not the
            # object this job already downloaded. This is the only value that can be
            # compared with Validate's independent measurement.
            from digest import measure_archive as _measure_archive
            _ARCHIVE_SHA, _ARCHIVE_SIZE = _measure_archive(str(_tar))
            log(f"measured source archive: sha256={_ARCHIVE_SHA} "
                f"size={_ARCHIVE_SIZE}")
            with capped_tar_open(_tar, "r:*",
                                             max_bytes=DEFAULT_MAX_ARCHIVE_BYTES) as tf:
                _bound_archive(tf)
                try:
                    tf.extractall(_ex, filter="data")
                except TypeError:
                    _base = os.path.abspath(_ex)
                    for m in tf.getmembers():
                        p = os.path.abspath(os.path.join(_base, m.name))
                        if os.path.commonpath([_base, p]) != _base or not (
                                m.isreg() or m.isdir()):
                            log(f"FATAL: unsafe tar member {m.name}")
                            sys.exit(1)
                    tf.extractall(_ex)
            model_path = _ex
        else:
            # An already-extracted channel is FATAL, not a fall-through. This branch did not exist,
            # so _ARCHIVE_SHA / _ARCHIVE_SIZE were never bound and the failure surfaced later as a
            # bare NameError -- naming a variable rather than the condition. Pipeline mode REQUIRES a
            # measurable archive: the measurement taken before extraction is the only value Validate
            # can compare against its own independent one, so a checkpoint that arrives already
            # unpacked cannot be attributed to the bytes the training step produced.
            _listing = sorted(os.listdir(model_path))[:20]
            log(f"FATAL: no model.tar.gz in the mounted channel {model_path!r}. Pipeline mode needs "
                f"the RAW archive: its pre-extraction measurement is what Validate compares against "
                f"its own, and an already-extracted tree cannot supply one. Found: {_listing}")
            sys.exit(1)
        log(f"pipeline mode: evaluating mounted checkpoint at {model_path}")
        import boto3
        _b, _, _k = SOURCE_URI.replace("s3://", "").partition("/")
        _resp = boto3.client("s3").head_object(
            Bucket=_b, Key=_k, ChecksumMode="ENABLED")
        if not _resp.get("VersionId"):
            log("FATAL: source artifact has no VersionId (bucket versioning "
                "must be on for identity binding)")
            sys.exit(1)
        model_artifact_identity = {
            "s3_uri": SOURCE_URI, "bucket": _b, "key": _k,
            "version_id": _resp["VersionId"],
            "etag": _resp["ETag"].strip('"'),
        }
    else:
        ckpt_root = os.path.join(WORK, "checkpoints", "GR00T-N1.7-LIBERO")
        run([UV_BIN, "run", "--python", "3.12", "python", "-c",
             "import sys; from huggingface_hub import snapshot_download\n"
             "p = snapshot_download(sys.argv[1], revision=sys.argv[2],\n"
             "    allow_patterns=[sys.argv[3] + '/*'],\n"
             "    local_dir=sys.argv[4])\n"
             "print('checkpoint at', p)",
             HF_REPO, HF_REVISION, CKPT_SUBDIR, ckpt_root], cwd=GR00T_DIR, env=env)
        model_path = os.path.join(ckpt_root, CKPT_SUBDIR)
        if not os.path.isdir(model_path):
            log(f"FATAL: checkpoint subdir missing: {model_path}")
            sys.exit(1)

    # Content identity of the served checkpoint (contract binding).
    log("computing weights digest over the enforced snapshot ...")
    global _RESOLVED
    if not PIPELINE_MODE:
        # HF mode. Constructed here, AFTER the descent to the final model_path, so the measured tree
        # is the one inference reads. The digest comes back from the object rather than being hashed
        # twice: from_snapshot measures internally by design (a caller-supplied digest validates as a
        # format while potentially describing a different tree), so reusing its result keeps exactly
        # one traversal of a multi-gigabyte tree.
        #
        # Selected on PIPELINE_MODE, not on whether an archive dict happens to be populated. The old
        # `if not _SOURCE_ARCHIVE:` made the mode a side effect of dictionary state, which is what let
        # a fixture seed the archive and render this branch unexecutable.
        _RESOLVED = si_from_snapshot(load_root=model_path, repo_id=HF_REPO,
                                     resolved_commit=HF_REVISION)
        snapshot_digest = _RESOLVED.tree_digest
        log(f"measured source snapshot: repo={HF_REPO} commit={HF_REVISION} "
            f"tree={snapshot_digest}")
    else:
        # Pipeline mode re-roots this digest to the manifest-containing tree below (binding_digest),
        # and the archive identity is constructed there, from the pre-extraction measurement.
        snapshot_digest = weights_digest(model_path)
    log(f"weights_digest: {snapshot_digest}")

    # Pipeline mode: read and cross-check the manifest from the fine-tune step.
    # Locate it by searching the whole mount (it may sit at the checkpoint root
    # while model_path was descended to weights) -- same fix as the molmoact2
    # eval; recompute the digest over the tree that CONTAINS the manifest.
    if PIPELINE_MODE:
        # Search the RESOLVED (extracted) checkpoint tree, not the raw channel.
        mount_root = model_path
        manifest_path = os.path.join(model_path, "checkpoint_manifest.json")
        if not os.path.isfile(manifest_path):
            found = None
            for _root, _dirs, _files in os.walk(mount_root):
                if "checkpoint_manifest.json" in _files:
                    found = os.path.join(_root, "checkpoint_manifest.json")
                    break
            manifest_path = found
        if not manifest_path or not os.path.isfile(manifest_path):
            log("FATAL: checkpoint_manifest.json not found under the mounted "
                f"checkpoint {mount_root!r} -- the FineTune step must write it "
                "(contract rev6); refusing to fabricate")
            sys.exit(1)
        with open(manifest_path) as fh:
            checkpoint_manifest = json.load(fh)
        # R5: validate against the family schema at the moment the manifest is first trusted.
        # Consuming a key straight out of the manifest gave a bare KeyError two minutes into a paid
        # job, naming neither the expected field nor its source; the same schema check already ran,
        # but only inside validate_report at the END of the run (validator.py:486).
        # R5b: the wrong-family checkpoint is the likeliest operator error and produced the least
        # useful failure -- a gr00t checkpoint handed to this evaluator died on KeyError for a field
        # only openvla manifests carry, two minutes into a paid job. The Arena evaluator has always
        # checked this (isaac_arena/gr00t/eval_entry.py:1090); three of four did not.
        _declared_family = checkpoint_manifest.get("model_family")
        if _declared_family != 'gr00t':
            log(f"FATAL: this checkpoint was produced for model_family={_declared_family!r} but "
                f"is being evaluated as 'gr00t'. Its manifest describes a different family's "
                f"input_config, so every field read from here would be wrong or missing.")
            sys.exit(1)
        from validator import validate_manifest as _validate_manifest
        _validate_manifest(checkpoint_manifest, FAMILY_SCHEMAS)
        settings = checkpoint_input_config(checkpoint_manifest)
        embodiment_tag = settings["embodiment_tag"]
        # R7: the family matching is NOT enough. A gr00t checkpoint trained for Isaac Lab Arena
        # declares embodiment_tag "GR1", which this LIBERO server does not know -- so a
        # cross-SIMULATOR mismatch passed the family check above and then died inside upstream code
        # with `ValueError: Unknown embodiment tag: 'GR1'`, after the image had pulled and the
        # checkpoint had downloaded. Family is not simulator.
        #
        # The suite manifests already carry the answer: LIBERO suites declare embodiment_tag null
        # (the GR00T LIBERO path uses the base model's own tags) while Arena suites declare GR1 or
        # new_embodiment. So the mismatch is decidable here, at the same seam as the family check,
        # rather than in a stack trace from a third-party library.
        _ARENA_ONLY_TAGS = {"GR1", "new_embodiment", "g1_wbc_joint", "gr1_joint"}
        if embodiment_tag in _ARENA_ONLY_TAGS:
            log(f"FATAL: this checkpoint declares embodiment_tag={embodiment_tag!r}, which belongs to "
                f"Isaac Lab Arena, but it is being evaluated on a LIBERO suite. The GR00T server "
                f"rejects unknown embodiment tags from inside upstream code, so the mismatch would "
                f"otherwise surface as a ValueError after the image and checkpoint had been "
                f"downloaded. Evaluate this checkpoint with run_arena.py, or use a LIBERO-trained "
                f"checkpoint here.")
            sys.exit(1)
        n_action_steps = settings["n_action_steps"]
        max_episode_steps = settings["max_episode_steps"]
        manifest_tree = os.path.dirname(manifest_path)
        binding_digest = weights_digest(manifest_tree)
        if binding_digest != checkpoint_manifest["weights_digest"]:
            log(f"FATAL: digest mismatch -- mounted bytes are not the produced bytes\n"
                f"  manifest: {checkpoint_manifest['weights_digest']}\n"
                f"  recomputed over {manifest_tree}: {binding_digest}")
            sys.exit(1)
        model_path = manifest_tree
        snapshot_digest = binding_digest
        # The archive identity is built HERE, at the tree the digest was just verified against, not
        # at the extraction root: pipeline mode re-roots to the manifest-containing directory, so
        # constructing earlier would name a tree the report's own weights_digest does not describe.
        # The archive bytes measurement comes from before extraction, where it had to be taken.
        _RESOLVED = si_from_archive(load_root=manifest_tree, archive_path=_tar,
                                    checkpoint=manifest_tree,
                                    sha256=_ARCHIVE_SHA, size_bytes=_ARCHIVE_SIZE)
        log("pipeline mode: read checkpoint_manifest.json from mounted checkpoint "
            f"(base={checkpoint_manifest['base_checkpoint']}, "
            f"seed={checkpoint_manifest.get('train_seed')})")

    # ---- [5.5/7] Pre-cache gated Cosmos backbone from within the uv venv.
    # The GR00T model's internal AutoModel.from_pretrained("nvidia/Cosmos-Reason2-2B")
    # does NOT forward the HF_TOKEN to nested downloads. Pre-caching ensures
    # the model is on disk so no HTTP auth is needed at server startup.
    log("pre-caching gated backbone nvidia/Cosmos-Reason2-2B ...")
    _precache_script = (
        "import os, sys\n"
        "token = os.environ.get('HF_TOKEN', '')\n"
        "if not token:\n"
        "    print('FATAL: HF_TOKEN empty in uv subprocess', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "from huggingface_hub import login, snapshot_download\n"
        "login(token=token)\n"
        "print(f'logged in, downloading nvidia/Cosmos-Reason2-2B ...')\n"
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
        "open(os.environ['BACKBONE_COMMIT_FILE'], 'w').write(_c)\n"
        "print(f'backbone resolved commit: {_c}')\n"
    )
    # cycle-16 C1: the resolved backbone commit is the external input that was missing from
    # the evidence. Written by the child to a file rather than scraped from stdout, which is
    # interleaved with upstream logging.
    _bb_commit_file = os.path.join(tempfile.mkdtemp(prefix="backbone-"), "commit")
    env["BACKBONE_COMMIT_FILE"] = _bb_commit_file
    _precache_rc = subprocess.run(
        [UV_BIN, "run", "--no-env-file", "python", "-c", _precache_script],
        cwd=GR00T_DIR, env=env
    ).returncode
    if os.path.isfile(_bb_commit_file):
        # repo_id is set HERE, by the path that actually downloaded it, rather than assumed at
        # module import by every path including the ones that never touch Cosmos.
        _BACKBONE_IDENTITY["repo_id"] = "nvidia/Cosmos-Reason2-2B"
        _BACKBONE_IDENTITY["resolved_commit"] = open(_bb_commit_file).read().strip()
        log(f"backbone identity: nvidia/Cosmos-Reason2-2B @ "
            f"{_BACKBONE_IDENTITY['resolved_commit']}")
    else:
        # Absent is recorded as absent. A guessed value here would be worse than the gap.
        log("WARNING: backbone commit file not written; identity unavailable")
    if _precache_rc != 0:
        log("FATAL: failed to pre-cache nvidia/Cosmos-Reason2-2B")
        sys.exit(1)
    log("backbone pre-cached successfully")


    # ---- [6/7] Launch server, health-check the port.
    # cuDNN library precedence (same root cause as the FineTune step): the DLC
    # base ships system cuDNN 9.1 in /lib/x86_64-linux-gnu (no cudnnGetLibConfig,
    # added in 9.3). The GR00T-N1.7 policy invokes the cuDNN-graph API at model
    # load, so with the system lib winning the loader search the server aborts
    # ("libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig") and the client
    # then fails with zmq.error.Again. Prepend the gr00t uv venv's nvidia/*/lib
    # so its newer bundled nvidia-cudnn-cu12 (>=9.5) wins. N1.6 never exercises
    # this path, so this is inert for n16.
    _nvidia_libs = sorted(glob.glob(os.path.join(
        GR00T_DIR, ".venv", "lib", "python*", "site-packages", "nvidia", "*", "lib")))
    if _nvidia_libs:
        _prev_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            _nvidia_libs + ([_prev_ld] if _prev_ld else []))
        log(f"[cudnn-fix] prepended {len(_nvidia_libs)} venv nvidia lib dir(s) to "
            "LD_LIBRARY_PATH (venv cuDNN precedence over DLC system 9.1)")
    else:
        log("[cudnn-fix] FATAL: no venv nvidia/*/lib dirs found under "
            f"{GR00T_DIR}/.venv -- the N1.7 server needs cuDNN>=9.3 "
            "(cudnnGetLibConfig); the uv venv is malformed, aborting")
        sys.exit(1)
    server_log_path = "/tmp/gr00t_server.log"
    server_log = open(server_log_path, "w")
    server_env = {**env, "PYTHONHASHSEED": SEED, "EVAL_SEED": SEED}
    # Launch THROUGH the seeding wrapper, which binds the eval seed to the server's RNG at
    # the inference-ready boundary and installs the strict weight-load audit. This path
    # previously launched gr00t/eval/run_gr00t_server.py directly, so neither guarantee
    # applied to LIBERO while the wrapper's own comment claimed both launchers were covered.
    # The wrapper is delivered by stage() (it is not baked into this image, unlike Arena's).
    # The sibling of the Arena guard, and it was missing. This evaluator only ever checked the REQUESTED
    # version (line ~143 refuses anything but n17); it never asked the CHECKPOINT what it is. So an N1.6
    # checkpoint here reaches the N1.7 loader exactly as it did on Arena -- server starts, load fails
    # after the startup timeout, and the error names transformers rather than the mismatch. I added that
    # guard to Arena's two launchers and a test that both were covered, then left this third one
    # unguarded: the same one-of-N pattern the test was meant to prevent.
    from checkpoint_compat import require_loadable
    require_loadable(model_path, "n17", log=log)

    if not os.path.exists(SEEDED_SERVER_ENTRY):
        log(f"FATAL: seeded server wrapper not found at {SEEDED_SERVER_ENTRY}. Refusing to "
            f"launch the policy server unseeded and without the strict load audit -- a "
            f"partly random model would otherwise be evaluated and reported as a checkpoint.")
        sys.exit(1)
    # Trace boundary: the seeded policy server loads the model onto a CUDA device, so on a CPU
    # L3 job this chunk OPENS and never closes (the server exits early in wait_for_port) --
    # read_trace reports it as the death point, which is the GPU boundary this rung locates.
    log(f">>> policy_server launching seeded GR00T server (model load binds a GPU device): {model_path}")
    server = subprocess.Popen(
        [UV_BIN, "run", "--no-env-file", "python", SEEDED_SERVER_ENTRY,
         "--model-path", model_path,
         "--embodiment-tag", embodiment_tag,
         "--use-sim-policy-wrapper"],
        cwd=GR00T_DIR, env=server_env,
        stdout=server_log, stderr=subprocess.STDOUT)
    log(f"server pid={server.pid}, waiting for port {SERVER_PORT} "
        "(model load can take minutes)")
    wait_for_port(SERVER_PORT, timeout_s=1200, proc=server,
                  server_log_path=server_log_path)
    # Establish attribution before spending the episode budget. The final report
    # checks it again against the completed server evidence.
    from gr00t_seeded_server import consumed_backbone_identity
    try:
        consumed_backbone_identity(
            _seed_scope_for_report(), model_path, "nvidia/Cosmos-Reason2-2B")
    except Exception:
        server.terminate()
        raise
    # check=False handles a nonzero exit, NOT a missing binary -- that raises
    # FileNotFoundError from Popen. This only logs accelerator state, so it must
    # not abort the step.
    try:
        subprocess.run(["nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv"], check=False)
    except (FileNotFoundError, OSError) as _exc:
        log(f"nvidia-smi unavailable ({_exc.__class__.__name__}: {_exc}); continuing, because nothing reads its output.")

    # ---- [7/7] Loop the suite tasks (one rollout invocation each), parse the
    # per-episode boolean vector, aggregate into per_task records.
    all_tasks = enumerate_tasks(libero_py, env)
    if TASK_IDS and TASK_IDS != "all":
        wanted = set(json.loads(TASK_IDS))
        tasks = [(tid, name) for tid, name in all_tasks if tid in wanted]
        if len(tasks) != len(wanted):
            log(f"FATAL: requested task_ids {sorted(wanted)} not all present "
                f"in {SUITE} (have {[t[0] for t in all_tasks]})")
            server.terminate()
            sys.exit(1)
    else:
        tasks = all_tasks
    log(f"running {len(tasks)} tasks x {EPISODES} episodes = "
        f"{len(tasks) * int(EPISODES)} episodes")

    os.makedirs(OUT_DIR, exist_ok=True)
    per_task = []
    task_logs_dir = os.path.join(WORK, "task-logs")
    os.makedirs(task_logs_dir, exist_ok=True)
    try:
        for tid, env_name in tasks:
            client_cmd = [
                libero_py, "gr00t/eval/rollout_policy.py",
                "--n-episodes", str(EPISODES),
                "--policy-client-host", "127.0.0.1",
                "--policy-client-port", str(SERVER_PORT),
                "--max-episode-steps", max_episode_steps,
                "--env-name", env_name,
                "--n-action-steps", n_action_steps,
                "--n-envs", "1",
                "--seed", SEED,
            ]
            # I5: the rollout writes its authoritative outcome to THIS file; stdout is
            # no longer parsed for the score. Remove any stale file first so a previous
            # task's result can never be read as this one's.
            result_json = os.path.join(task_logs_dir, f"task_{tid}_result.json")
            if os.path.exists(result_json):
                os.remove(result_json)
            env = {**env, _RESULT_JSON_ENV: result_json}
            log(f"[task {tid}] $ " + " ".join(client_cmd))
            task_log_path = os.path.join(task_logs_dir, f"task_{tid}.log")
            buf = []
            with open(task_log_path, "w") as lf:
                proc = subprocess.Popen(client_cmd, cwd=GR00T_DIR, env=env,
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
                # I12: the hang lives in this loop, not in wait() -- by the time wait() runs
                # the child has closed stdout. A killed child must FAIL the step: continuing
                # with whatever was printed would manufacture a truncated per-task result.
                #
                # This bound is COARSE. It is the whole-step budget, so one hung task can
                # consume the run's entire allowance before firing. That is still strictly
                # better than no bound, but a per-task share would be tighter; the arena
                # evaluator derives its own from the episode count.
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
                _hang_timer.start()
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    lf.write(line)
                    buf.append(line)
                rc = proc.wait()
                _hang_timer.cancel()
                if _timed_out["v"]:
                    _dump(server_log_path, "server log")
                    log(f"FATAL: rollout for task {tid} exceeded the runtime budget, killed")
                    sys.exit(124)
            if rc != 0:
                _dump(server_log_path, "server log")
                log(f"FATAL: rollout for task {tid} ({env_name}) exited {rc}")
                sys.exit(rc)
            try:
                successes = _read_result_file(
                    result_json, env_name, int(EPISODES))
            except ValueError as exc:
                log(f"FATAL: task {tid} ({env_name}): {exc}")
                sys.exit(1)
            rate = sum(successes) / len(successes)
            per_task.append({
                "task_id": tid,
                "task": f"{SUITE}_{tid}",
                "episodes": len(successes),
                "success_rate": rate,
                # The observed count, not a rate the schema has to reverse-engineer.
                "successes": sum(successes),
            })
            log(f"[task {tid}] success_rate={rate} ({sum(successes)}/"
                f"{len(successes)})")
    finally:
        server.terminate()

    # ---- Assemble + self-validate the schema-v2 report.
    per_task.sort(key=lambda r: r["task_id"])
    explicit_task_ids = sorted(r["task_id"] for r in per_task)
    reported_episodes = sum(r["episodes"] for r in per_task)
    weighted = sum(r["success_rate"] * r["episodes"] for r in per_task)
    success_rate = weighted / reported_episodes

    if not PIPELINE_MODE:
        checkpoint_manifest = {
            "manifest_version": 1,
            "model_family": "gr00t",
            "base_checkpoint": HF_REPO,
            # The immutable Hub commit, NOT the suite subdir (CKPT_SUBDIR is
            # e.g. "libero_spatial" -- a path, not a revision).
            "base_revision": HF_REVISION,
            "train_seed": None,
            "input_config": {
                "embodiment_tag": EMBODIMENT_TAG,
                # C3 (cycle 12): these reported the MODULE GLOBALS while the command at --n-action-steps
                # / --max-episode-steps uses the LOCALS, which pipeline mode replaces with
                # checkpoint-provided values. A checkpoint declaring 4 steps and a 100-step horizon
                # therefore EXECUTED 4/100 and REPORTED 8/720, and Validate compared that report against
                # the suite expectation of 8/720 and passed it.
                "n_action_steps": n_action_steps,
                "max_episode_steps": max_episode_steps,
            },
            "train_recipe": {"repo": GR00T_REPO, "commit": GR00T_COMMIT,
                             "max_steps": None},
            "weights_digest": snapshot_digest,
            "dataset_manifest": None,
        }
    # else: checkpoint_manifest was loaded from the mounted checkpoint above
    recomputed = weights_digest(model_path)
    # Computed ONCE, above the report, so the same accepted scope both populates
    # seed_scope and establishes which backbone commit inference consumed. Reading it
    # twice risked two different answers in one report.
    _seed_scope = _seed_scope_for_report()

    metrics = {
        "schema_version": 3,
        # cycle-16 C1: the external VLM backbone determines preprocessing, and it was
        # downloaded with no revision -- so it could change while the checkpoint, its
        # digest, the image and the sourcedir all stayed identical. Recorded under
        # aux_metrics because no gate reads it, but it travels inside the report and is
        # therefore covered by the same schema and digest chain.
        "backbone_identity": dict(_BACKBONE_IDENTITY),
        # checkpoint, checkpoint_revision and the EXACTLY-ONE source variant all derive from the
        # single resolved object, so they cannot describe different things (I4) and the digest
        # cannot arrive in the wrong convention (I6).
        **_RESOLVED.report_identity_fields(),
        "policy_type": "checkpoint",
        "model_family": "gr00t",
        # checkpoint_revision comes from the spread above; re-assigning it here defeated the
        # single-source contract, because a later key in the same dict literal wins.
        "checkpoint_manifest": checkpoint_manifest,
        "weights_digest_recomputed_by_eval": recomputed,
        "model_artifact_identity": model_artifact_identity,
        "suite": SUITE,
        "task_ids": explicit_task_ids,
        "num_trials_per_task": int(EPISODES),
        "eval_seed": int(SEED),
        "train_seed": checkpoint_manifest.get("train_seed"),
        "success_rate": success_rate,
        "episodes": reported_episodes,
        "episodes_reported_by_evaluator": reported_episodes,
        "per_task": per_task,
        "provenance": {
            "recipe_repo": GR00T_REPO,
            "recipe_commit": GR00T_COMMIT,
            "mujoco_gl": env["MUJOCO_GL"],
            "repo_commit": GR00T_COMMIT,
            "python_version": PYTHON_PIN,
        },
        # I7: this was the bare string "client_and_env", which understated what the run now
        # establishes AND could not be checked. LIBERO launches its server through the seeding
        # wrapper, so the same structured evidence Arena reports is available here -- including
        # whether the inference-ready reseed happened and whether the strict weight-load audit
        # completed. Reporting the string meant a reader could not tell a seeded, audited run
        # from an unseeded one.
        "seed_scope": _seed_scope,
        # I6: state which evaluator version actually ran. Validate compared this only when
        # present, and required the effective config only for Arena, so a LIBERO report could
        # omit it and pass under any EXPECTED_FAMILY_VERSION -- a probe accepted a report
        # carrying N1.6 provenance under n17. The LIBERO suites' embodiment tags are null, so
        # the embodiment check does not independently distinguish the versions either.
        #
        # This module refuses anything but n17 at import, so the value is the version that
        # ran rather than the version that was requested.
        "effective_eval_config": {
            "gr00t_version": _gr00t_version,
            "num_episodes": reported_episodes,
            "budget_type": "fixed_trials",
            "eval_backend": "libero",
            # I6: the evaluation PROTOCOL was chosen by the checkpoint and never reported, so
            # a consumer could not tell whether two packages sharing a suite, trial count and
            # seed had actually been evaluated the same way. Episode horizon and action-chunk
            # length change the experiment. These are the values this run consumed; Validate
            # compares them against the suite's declared expectation.
            "n_action_steps": int(n_action_steps),
            "max_episode_steps": int(max_episode_steps),
        },
    }

    # I2, the LIBERO half. I hoisted the seed scope here and then bound the consumed backbone only in
    # Arena -- so this evaluator kept publishing the PRECACHE commit, which is the whole defect, in the
    # very change that fixed it elsewhere. One of N sibling paths again, in a fix for one of N sibling
    # paths. Found by enumerating the call sites myself rather than trusting my own summary.
    # model_path, because that is what the SERVER was launched with and therefore what the evidence's
    # processor_validated records. _RESOLVED.load_root is the identity's measurement root, which differs
    # in archive mode -- using it compared two different directories and refused every pipeline-mode run.
    # The same "verified the wrong object" shape one level down.
    metrics["backbone_identity"] = consumed_backbone_identity(
        _seed_scope, model_path, "nvidia/Cosmos-Reason2-2B")
    log(f"backbone identity from the LOAD: {metrics['backbone_identity']}")

    validate_report(
        metrics,
        expected_suite=SUITE,
        expected_trials=int(EPISODES),
        expected_eval_seed=int(SEED),
        expected_task_ids=explicit_task_ids,
        family_schemas=FAMILY_SCHEMAS,
        pipeline_mode=PIPELINE_MODE,
        expected_budget_type="fixed_trials",
    )
    log("schema-v3 self-validation PASSED (shared gate validator)")

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    # Evidence: server log + all per-task rollout logs.
    with open(server_log_path) as s, open(
            os.path.join(OUT_DIR, "gr00t_server.log"), "w") as d:
        d.write(s.read())
    for tid, _name in tasks:
        src = os.path.join(task_logs_dir, f"task_{tid}.log")
        if os.path.exists(src):
            with open(src) as s, open(
                    os.path.join(OUT_DIR, f"task_{tid}.log"), "w") as d:
                d.write(s.read())
    log(f"DONE suite={SUITE} success_rate={success_rate} "
        f"episodes={reported_episodes}")


if __name__ == "__main__":
    main()
