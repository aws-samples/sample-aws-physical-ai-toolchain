"""SageMaker entry point: MolmoAct2 LIBERO evaluation (models/ adapter).

Ports the reference validated MolmoAct2 recipe (aws-samples/sample-vla-simulator-on-aws,
models/molmoact2.yaml + templates/molmoact2-userdata.sh.j2, validated 2026-06-29 on
g6e.2xlarge EC2) from EC2 UserData into a SageMaker training-job entry script, then
runs the fork's lerobot-eval and writes metrics.json in this repo's metrics shape.

This entrypoint evaluates a fine-tuned, mounted checkpoint and publishes a
schema-v3 report, including its measured archive identity. It self-validates
with the same shared validator used by the pipeline's Validate step.
Published MolmoAct2 checkpoints are not accepted by the pinned LeRobot loader.
Use FineTune -> SimEval, or --checkpoint-s3 to evaluate a saved training artifact.

Recipe facts inherited from the reference repo (cited inline):
  - Model: allenai/MolmoAct2-LIBERO (5B, Apache-2.0, public/non-gated)
  - Policy lives in the allenai/lerobot FORK branch molmoact2-policy, NOT upstream
  - EVAL dtype is fp32 + use_amp=false (official LIBERO recipe; bf16 is off-recipe)
  - fp32 5B needs a single GPU >= 40GB VRAM (L40S 48GB) AND >= 64GiB host RAM
    (32GiB thrashes on ckpt load) -> ml.g6e.2xlarge
  - Python MUST be 3.12 (fork targets 3.12/3.13; 3.14 argparse rejects draccus'
    `type=str | None` and lerobot-eval crashes at CLI parse)
  - norm_tag=libero is REQUIRED (omission raises ValueError)

Config via environment (set by the launcher):
  EVAL_SUITE        libero_goal (fork's documented example suite)
  EVAL_TASK_IDS     JSON array capping suite tasks, e.g. "[0,1]" ("" = all tasks)
  EVAL_TRIALS       episodes per task (pipe-proof: 5; full recipe: 50)
  EVAL_SEED         lerobot-eval seed (the reference config: 1000)

Fail-fast policy: any install or eval failure exits nonzero.
No fallbacks that mask failure.
"""
from __future__ import annotations

import json
import os
import re
import time
import threading
import signal
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capped_reader import DEFAULT_MAX_ARCHIVE_BYTES, capped_tar_open  # noqa: E402
from digest import weights_digest  # noqa: E402  (shipped in sourcedir)
from source_identity import from_archive as si_from_archive  # noqa: E402
from validator import validate_report  # noqa: E402

OUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
# PIPELINE CHANGE: metrics.json to SM_MODEL_DIR for step-property binding.
METRICS_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
WORK = "/opt/ml/code"
if os.path.isfile("/opt/vla/.baked_env"):
    WORK = "/opt/vla"
    print("[eval] BAKED ENV DETECTED -- using /opt/vla as WORK", flush=True)
LEROBOT_DIR = os.path.join(WORK, "lerobot")
EVAL_OUT = os.path.join(WORK, "eval-out")
UV_BIN = os.path.join(os.path.dirname(sys.executable), "uv")

# Pins from the reference models/molmoact2.yaml (validated 2026-06-29)
LEROBOT_REPO = "https://github.com/allenai/lerobot.git"
LEROBOT_BRANCH = "molmoact2-policy"
LEROBOT_COMMIT = "a4f15bf347dee7eb8a8c5f4a70a37f476b091113"
PYTHON_PIN = "3.12"
CAMERA_MAP = '{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}'

# Family schemas load from defaults.json (single source; the
# inline copy risked drift against what the gate bakes at deploy time).
_defaults_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "defaults.json")
with open(_defaults_path) as _fh:
    _defaults = json.load(_fh)
FAMILY_SCHEMAS = {
    _defaults["family"]: {
        "input_config_schema": _defaults["input_config_schema"],
        "provenance_keys": _defaults["provenance_keys"],
    },
}

SUITE = os.environ.get("EVAL_SUITE", "libero_goal")
TASK_IDS = os.environ.get("EVAL_TASK_IDS", "")
# Contract interface names (EVAL_EPISODES was a private dialect).
# EVAL_TRIALS is canonical; EVAL_EPISODES accepted as a deprecated alias ONLY
# when EVAL_TRIALS is unset, and a conflict between the two is fatal.
_trials = os.environ.get("EVAL_TRIALS", "")
_episodes_alias = os.environ.get("EVAL_EPISODES", "")
if _trials and _episodes_alias and _trials != _episodes_alias:
    print(f"[molmoact2-eval] FATAL: EVAL_TRIALS={_trials} conflicts with "
          f"EVAL_EPISODES={_episodes_alias}", flush=True)
    sys.exit(1)
EPISODES = _trials or _episodes_alias or "5"
SEED = os.environ.get("EVAL_SEED", "1000")
# FineTune outputs and --checkpoint-s3 both mount the same archive contract.
HF_REPO = os.environ.get("EVAL_CHECKPOINT", "")
HF_REV = ""
SOURCE_URI = os.environ.get("EVAL_MODEL_SOURCE_URI", "")
PIPELINE_MODE = HF_REPO.startswith("/")
if not PIPELINE_MODE:
    print("[molmoact2-eval] published-checkpoint evaluation is not supported; "
          "use FineTune -> SimEval or --checkpoint-s3 with a saved training artifact", flush=True)
    sys.exit(1)
if not SOURCE_URI:
    print("[molmoact2-eval] local EVAL_CHECKPOINT requires EVAL_MODEL_SOURCE_URI "
          "for artifact identity binding", flush=True)
    sys.exit(1)


# I4: bound the checkpoint archive before extracting it on accelerator capacity.
# Inserted before the first module-level def: an earlier attempt anchored on the last
# import line and landed INSIDE a string literal, because this file embeds injected
# source that contains its own imports at column 0.
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

def log(msg: str) -> None:
    print(f"[molmoact2-eval] {msg}", flush=True)


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
    # A call with no timeout can hang for the whole budget and produce nothing; the step then
    # looks like a long job rather than a stuck one.
    kw.setdefault("timeout", _subprocess_timeout())
    subprocess.run(cmd, check=True, **kw)


def uv(*args: str, **kw) -> None:
    # The SageMaker DLC exports PYTHONPATH for its py3.10 site-packages; uv's
    # managed py3.12 inherits it and the stdlib re module loads 3.10's _sre
    # ("AssertionError: SRE module mismatch", spike run 20260804-211546).
    # Strip it for every uv invocation -- the fork's venv is self-contained.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # PyPI 502 resilience: uv's default is 3 retries in ~4s, which lost a whole
    # eval to a transient grpcio 502 (run 20260817-042040). Raise retries +
    # per-request timeout so a brief PyPI blip is absorbed in-job, not by
    # burning an orchestrator retry (which respawns a full g6e instance).
    env.setdefault("UV_HTTP_TIMEOUT", "120")
    env.setdefault("UV_HTTP_RETRIES", "10")
    kw.setdefault("env", env)
    run([UV_BIN, *args], cwd=LEROBOT_DIR, **kw)


def uv_sync_resilient(*sync_args: str) -> None:
    """`uv sync` with process-level retry + backoff on top of UV_HTTP_RETRIES.
    A transient PyPI 502 on a big wheel (pyarrow/torch/grpcio) killed whole evals
    even with UV_HTTP_RETRIES; uv sync is resumable so a retry only re-fetches the
    dropped wheel. Fail-closed after the last attempt."""
    import time as _time
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("UV_HTTP_TIMEOUT", "120")
    env.setdefault("UV_HTTP_RETRIES", "10")
    delay = 30
    for attempt in range(1, 9):
        out = subprocess.run([UV_BIN, "sync", *sync_args], cwd=LEROBOT_DIR,
                             env=env)
        if out.returncode == 0:
            if attempt > 1:
                log(f"uv sync OK on attempt {attempt}")
            return
        if attempt < 8:
            log(f"uv sync failed (rc={out.returncode}); transient PyPI blip? "
                f"backing off {delay}s (attempt {attempt}/8)")
            _time.sleep(delay)
            delay = min(delay * 2, 300)
    log("FATAL: uv sync failed after 8 attempts")
    sys.exit(1)


# Live-policy audit probe, executed in a separate CPU process inside the eval
# container. See the call site for what it does and does not establish.
_POLICY_LOAD_PROBE = r'''
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer, Linear as LoraLinear
from safetensors import safe_open

from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_policy
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
import lerobot.policies.pretrained as pretrained_api


def require(condition, message):
    if not condition:
        raise RuntimeError("MolmoAct2 policy audit: " + message)


def audit(policy_dir, base_dir, manifest, suite, task_ids):
    policy_dir = Path(policy_dir).resolve(strict=True)
    base_dir = Path(base_dir).resolve(strict=True)
    weights = policy_dir / "model.safetensors"

    cfg = PreTrainedConfig.from_pretrained(
        str(policy_dir),
        local_files_only=True,
        cli_overrides=[
            f"--checkpoint_path={base_dir}",
            "--norm_tag=libero",
            "--inference_action_mode=continuous",
            "--model_dtype=float32",
            "--use_amp=false",
            "--enable_inference_cuda_graph=true",
            "--device=cpu",
        ],
    )
    require(isinstance(cfg, MolmoAct2Config), "wrong policy config class")
    require(not cfg.use_peft, "unexpected outer/generic PEFT checkpoint layout")
    require(cfg.device == "cpu", "CPU override was not applied")
    require(cfg.model_dtype == "float32" and not cfg.use_amp,
            "dtype/AMP override failed")
    cfg.pretrained_path = policy_dir

    require(isinstance(manifest, dict), "manifest must be an object")
    recipe = manifest.get("train_recipe", {})
    require(isinstance(recipe, dict), "train_recipe must be an object")
    if "train_mode_vlm" in recipe:
        require(
            recipe["train_mode_vlm"] == cfg.train_mode_vlm,
            f"manifest mode {recipe['train_mode_vlm']!r} != "
            f"effective config mode {cfg.train_mode_vlm!r}",
        )

    # This is the environment CONFIG dataclass, not a running simulator.
    env_cfg = LiberoEnv(
        task=suite,
        task_ids=task_ids,
        camera_name_mapping={
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "wrist_image",
        },
    )

    original_load = pretrained_api.load_model_as_safetensor
    loaded_models = []

    def checked_load(model, filename, **kwargs):
        require(Path(filename).resolve() == weights,
                "unexpected policy weights file")

        # Obtain the same alias-aware diagnostics used by strict=True,
        # then make every missing/unexpected key fatal.
        kwargs["strict"] = False
        missing, unexpected = original_load(model, filename, **kwargs)
        require(
            not missing and not unexpected,
            "state-dict mismatch: " + json.dumps({
                "missing_keys": sorted(missing),
                "unexpected_keys": sorted(unexpected),
            }),
        )
        loaded_models.append(model)
        return missing, unexpected

    # Confined to this separate, single-threaded probe process.
    with patch.object(pretrained_api, "load_model_as_safetensor", checked_load):
        policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()

    require(
        len(loaded_models) == 1 and loaded_models[0] is policy,
        "factory did not use the audited policy loader exactly once",
    )

    live = policy.state_dict()
    for name, tensor in live.items():
        require(
            not tensor.is_meta and tensor.device.type == "cpu",
            f"unmaterialized/non-CPU state: {name}",
        )

    layers = [
        (name, layer)
        for name, layer in policy.named_modules()
        if isinstance(layer, LoraLayer)
    ]

    if cfg.train_mode_vlm != "lora":
        require(not layers, "live LoRA layers contradict non-LoRA config")
        return {"mode": cfg.train_mode_vlm, "full_policy_load": "ok"}

    require(isinstance(policy.model, PeftModel), "missing inner PeftModel")
    require(bool(layers), "no live LoRA layers")
    require(policy.model.active_adapters == ["default"],
            "wrong active adapter")

    statuses = policy.model.get_layer_status()
    require(len(statuses) == len(layers),
            "unexpected adapter-layer topology")
    for status in statuses:
        require(status.enabled is True,
                f"disabled adapter layer: {status.name}")
        require(status.active_adapters == ["default"],
                f"inactive adapter: {status.name}")
        require(status.available_adapters == ["default"],
                f"unexpected adapters: {status.name}")
        require(status.merged_adapters == [],
                f"merged adapter: {status.name}")

    banks = {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"}
    adapter_state = {
        name: tensor
        for name, tensor in live.items()
        if banks.intersection(name.split("."))
    }
    require(bool(adapter_state), "no live adapter tensors")

    for name, tensor in adapter_state.items():
        require(tensor.dtype == torch.float32,
                f"unexpected adapter dtype: {name}")
        require(bool(torch.isfinite(tensor).all().item()),
                f"nonfinite adapter: {name}")

    # Stored adapter tensors must match identically named LIVE tensors.
    # Full coverage, including shared aliases, was checked by load_model.
    compared = 0
    with safe_open(str(weights), framework="pt", device="cpu") as saved:
        for name in saved.keys():
            if name in adapter_state:
                value = saved.get_tensor(name).to(
                    dtype=adapter_state[name].dtype
                )
                require(torch.equal(adapter_state[name], value),
                        f"loaded value mismatch: {name}")
                compared += 1
                del value
    require(compared > 0,
            "no stored adapter tensors matched the live policy")

    delta_witness = None
    for name, layer in layers:
        require(
            isinstance(layer, LoraLinear) and not layer.lora_variant,
            f"unexpected LoRA implementation: {name}",
        )
        require(not layer.training,
                f"adapter is in training mode: {name}")

        scale = float(layer.scaling["default"])
        require(math.isfinite(scale) and scale > 0,
                f"invalid adapter scaling: {name}")

        if (
            delta_witness is None
            and torch.count_nonzero(layer.lora_B["default"].weight).item()
        ):
            with torch.no_grad():
                delta = layer.get_delta_weight("default")
                require(bool(torch.isfinite(delta).all().item()),
                        f"nonfinite LoRA delta: {name}")
                if torch.count_nonzero(delta).item():
                    delta_witness = name
                del delta

    require(
        delta_witness is not None,
        "no active adapter has a nonzero LoRA weight delta",
    )
    return {
        "mode": cfg.train_mode_vlm,
        "full_policy_load": "ok",
        "enabled_unmerged_layers": len(layers),
        "adapter_tensors_compared": compared,
        "nonzero_delta_layer": delta_witness,
    }


if __name__ == "__main__":
    ids = None if sys.argv[5] in {"", "all"} else json.loads(sys.argv[5])
    result = audit(
        sys.argv[1],
        sys.argv[2],
        json.loads(sys.argv[3]),
        sys.argv[4],
        ids,
    )
    print("MolmoAct2 live policy audit OK: "
          + json.dumps(result, sort_keys=True))
'''


# The value the pinned upstream MolmoAct2 configuration class applies when
# train_mode_vlm is omitted from a saved policy config. Reading the field with a bare
# .get() and treating a missing value as "no mode" skipped the adapter audit on runs that
# were in fact using LoRA, because the loader applies this default regardless.
UPSTREAM_DEFAULT_TRAIN_MODE_VLM = "lora"


def _resolve_train_mode(policy_dir: str, checkpoint_manifest: dict):
    """Establish the policy's training mode from the SAVED POLICY CONFIG.

    The adapter audit used to be gated on the manifest's optional
    `train_recipe.train_mode_vlm` provenance field, so deleting that one declaration
    disabled the audit entirely without changing the saved policy. The authoritative
    source is policy/config.json, because that is what the loader itself reads: it is
    why `__init__` re-wraps with PEFT.

    The manifest is a cross-check rather than the gate. If it declares a mode and the
    saved config disagrees, the provenance record and the policy being evaluated
    describe different things and neither can be trusted.

    Exits nonzero rather than returning a default: an unresolvable mode means the audit
    cannot run, and skipping the audit is the failure this guards against.
    """
    config_path = os.path.join(policy_dir, "config.json")
    if not os.path.isfile(config_path):
        log(f"FATAL: policy config missing at {config_path}; cannot establish whether "
            f"this checkpoint is a LoRA policy, so the adapter audit cannot run. "
            f"Refusing to evaluate an unaudited policy.")
        sys.exit(1)
    try:
        with open(config_path) as handle:
            saved = json.load(handle)
    except (OSError, ValueError) as exc:
        log(f"FATAL: policy config at {config_path} is unreadable ({exc}); the training "
            f"mode cannot be established.")
        sys.exit(1)
    # An OMITTED field is not an absent mode. The upstream configuration class defaults
    # train_mode_vlm to "lora" (MolmoAct2Config, configuration_molmoact2.py), so the
    # evaluator still installs adapters when the saved config says nothing. Reading this
    # with a plain .get() therefore returned None and skipped the adapter audit on a run
    # that WAS using LoRA -- the same shape of bypass as gating on the optional manifest
    # field. Fall back to the upstream default explicitly.
    saved_mode = saved.get("train_mode_vlm", UPSTREAM_DEFAULT_TRAIN_MODE_VLM)
    if saved_mode is None:
        saved_mode = UPSTREAM_DEFAULT_TRAIN_MODE_VLM
    if "train_mode_vlm" not in saved:
        log(f"policy config omits train_mode_vlm; using the upstream default "
            f"{saved_mode!r} (the loader applies that default too, so the adapter audit "
            f"must not be skipped)")
    declared = (checkpoint_manifest or {}).get("train_recipe", {}).get("train_mode_vlm")
    if declared is not None and declared != saved_mode:
        log(f"FATAL: the checkpoint manifest declares train_mode_vlm={declared!r} but "
            f"the saved policy config says {saved_mode!r}. The provenance record and "
            f"the policy being evaluated disagree.")
        sys.exit(1)
    return saved_mode


_POLICY_AUDIT = '''

def _audit_loaded_policy(policy, checkpoint_dir):
    """Verify the ACTUAL evaluation policy was fully populated from the checkpoint.

    This runs in the evaluator process, on the object that generates actions. The previous
    audit ran in a separate CPU probe process and inspected a different policy instance, so it
    could pass while the CUDA policy that produced the score had silently initialized tensors.
    The wrapper's own comment acknowledged that limitation across three review cycles.

    Compares the checkpoint's tensor names and shapes against the constructed policy's
    state_dict:

      * a checkpoint tensor absent from the policy means the file was not consumed;
      * a shape disagreement means the weights were reinterpreted;
      * a policy parameter with no checkpoint counterpart was initialized from nothing, which
        is the failure mode a digest cannot see -- the checkpoint is intact and the model is
        partly random.

    Raises RuntimeError naming the tensors. A checkpoint whose tensors all match is untouched.
    """
    import glob as _glob
    import json as _json
    import os as _os

    # C4 (cycle 10): this globbed EVERY .safetensors under the checkpoint and then required
    # every tensor in them to be a model state_dict entry. Pinned LeRobot writes PROCESSOR
    # STATE as additional .safetensors in the same directory -- pipeline.py:385-390 names them
    # "<step>_step_<i>[_<registry>].safetensors" and records each in the graph's state_file --
    # and normalization state carries keys like action.q01 (normalize_processor.py). Those are
    # not model weights and never appear in state_dict(), so the audit rejected the normal
    # output of this component's own trainer, which copies the whole pretrained_model directory
    # into policy/.
    #
    # Model files and processor state are different categories with different authorities. The
    # processor graph is the authority on which files are state, so they are excluded HERE by
    # name from the graph rather than by a filename pattern -- a pattern would silently stop
    # matching if upstream renamed them, which is how this class of bug recurs.
    state_files = set()
    for _graph in _glob.glob(_os.path.join(checkpoint_dir, "**", "*processor.json"),
                             recursive=True):
        try:
            with open(_graph) as _gh:
                _doc = _json.load(_gh)
        except (OSError, ValueError):
            continue
        for _step in (_doc.get("steps") or []):
            if isinstance(_step, dict) and _step.get("state_file"):
                state_files.add(_os.path.basename(str(_step["state_file"])))

    shard_paths = [
        path for path in sorted(_glob.glob(
            _os.path.join(checkpoint_dir, "**", "*.safetensors"), recursive=True))
        if _os.path.basename(path) not in state_files]
    if state_files:
        # print, NOT log: this function is injected source (_patch_policy_load_audit) and `log`
        # is unbound in that namespace, so a log() call here raises NameError inside the guard --
        # visible only at evaluation time on GPU capacity.
        print(f"[wrapper] excluding {len(state_files)} processor-state file(s) named by the "
              f"processor graph from the MODEL audit: {sorted(state_files)}", flush=True)
    if not shard_paths:
        raise RuntimeError(
            f"no .safetensors found under {checkpoint_dir}, so the loaded policy cannot be "
            "compared against the checkpoint it claims to come from. Refusing to evaluate a "
            "policy whose provenance cannot be checked.")
    from safetensors import safe_open

    checkpoint_tensors = {}
    for path in shard_paths:
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                checkpoint_tensors[key] = tuple(handle.get_slice(key).get_shape())

    policy_tensors = {name: tuple(tensor.shape)
                      for name, tensor in policy.state_dict().items()}

    def _suffix_index(names):
        index = {}
        for name in names:
            index.setdefault(name.split(".")[-1], set()).add(name)
        return index

    # Names are compared after stripping any single wrapper prefix, because LeRobot nests the
    # model under its own attribute while the checkpoint stores it flat. A tensor is matched if
    # some policy key ENDS WITH the checkpoint key, which is the relationship that nesting
    # produces; anything unmatched is reported rather than assumed benign.
    unmatched = []
    for key, shape in checkpoint_tensors.items():
        candidates = [name for name in policy_tensors if name == key or name.endswith("." + key)]
        if not candidates:
            unmatched.append(f"{key} (absent from the policy)")
            continue
        mismatched = [name for name in candidates if policy_tensors[name] != shape]
        if len(mismatched) == len(candidates):
            unmatched.append(
                f"{key} shape {shape} != policy {policy_tensors[candidates[0]]}")
    if unmatched:
        raise RuntimeError(
            f"strict policy load audit FAILED for {checkpoint_dir}: "
            f"{unmatched[:12]}{'...' if len(unmatched) > 12 else ''} "
            f"({len(unmatched)} of {len(checkpoint_tensors)} checkpoint tensors). The policy "
            "that generates actions does not match the checkpoint, so its score would not "
            "belong to these weights.")
    print(f"[wrapper] strict policy load audit PASSED: all {len(checkpoint_tensors)} "
          f"checkpoint tensors are present in the evaluation policy with matching shapes",
          flush=True)
'''


def _patch_policy_load_audit(src: str) -> str:
    """Audit the REAL evaluation policy after construction, before any action.

    Raises RuntimeError if the anchor is absent, since upstream having moved means the audit
    would silently not run and a partly random policy could be evaluated.
    """
    anchor = (
        "    policy = make_policy(\n"
        "        cfg=cfg.policy,\n"
        "        env_cfg=cfg.env,\n"
        "        rename_map=cfg.rename_map,\n"
        "    )\n"
        "\n"
        "    policy.eval()\n")
    if src.count(anchor) != 1:
        raise RuntimeError(
            f"policy-audit anchor occurs {src.count(anchor)} times in the pinned LeRobot eval "
            f"script (expected exactly 1); upstream changed. The real evaluation policy would "
            f"not be audited, so refusing to run.")
    src = src.replace(
        anchor,
        anchor + "    _audit_loaded_policy(policy, cfg.policy.pretrained_path)\n", 1)
    return src.rstrip("\n") + "\n" + _POLICY_AUDIT


_TELEMETRY_VALIDATOR = '''

def _require_boolean_success_vector(value, where, expected):
    """Require success telemetry to BE boolean measurements before coercion.

    Upstream converts whatever it finds with .tolist() or bool(), so an invalid value became
    an ordinary outcome and no later check could recover the original. Probes against the
    pinned upstream showed the scalar string "false" accepted as SUCCESS -- bool("false") is
    True -- and None accepted as failure, neither of which measures task achievement.

    Returns a list of `expected` booleans. Everything else raises, naming value and site.
    """
    import numpy as _np

    if isinstance(value, (bool, _np.bool_)):
        return [bool(value)] * expected
    if hasattr(value, "tolist") or isinstance(value, (list, tuple)):
        array = _np.asarray(value.tolist() if hasattr(value, "tolist") else value)
        if array.size == 0:
            raise ValueError(
                f"success telemetry at {where} is empty ({value!r}), so no episode outcome "
                "was reported. An absent measurement must not become a failed episode.")
        if array.dtype != _np.bool_:
            raise ValueError(
                f"success telemetry at {where} has dtype {array.dtype} ({value!r}), not "
                "boolean. Numeric or object values are coerced into an outcome they do not "
                "express.")
        flat = [bool(item) for item in array.reshape(-1)]
        if len(flat) != expected:
            raise ValueError(
                f"success telemetry at {where} reports {len(flat)} outcomes for "
                f"{expected} environments ({value!r}).")
        return flat
    raise ValueError(
        f"success telemetry at {where} is {type(value).__name__} ({value!r}), not a boolean "
        "or a boolean array. Refusing to coerce it into an episode outcome.")
'''


def _patch_telemetry_shape(src: str) -> str:
    """Validate success telemetry at the OBSERVATION boundary, before any coercion.

    The telemetry patch marks a source observed on field PRESENCE, so a present but malformed
    value counted as a measurement. This validates the value itself at each raw read.

    Raises RuntimeError if any anchor is absent.
    """
    edits = [
        (
            '            successes = final_info["is_success"].tolist()\n',
            '            successes = _require_boolean_success_vector(\n'
            '                final_info["is_success"], "final_info[\'is_success\']",\n'
            '                env.num_envs)\n',
        ),
        (
            '            successes = (\n'
            '                is_success.tolist() if hasattr(is_success, "tolist") '
            'else [bool(is_success)] * env.num_envs\n'
            '            )\n',
            '            successes = _require_boolean_success_vector(\n'
            '                is_success, "info[\'is_success\']", env.num_envs)\n',
        ),
    ]
    for index, (target, replacement) in enumerate(edits, start=1):
        if src.count(target) != 1:
            raise RuntimeError(
                f"telemetry-shape patch anchor {index} occurs {src.count(target)} times in "
                f"the pinned LeRobot eval script (expected exactly 1); upstream changed. "
                f"Malformed telemetry could again be coerced into an outcome, so refusing "
                f"to run.")
        src = src.replace(target, replacement, 1)
    return src.rstrip("\n") + "\n" + _TELEMETRY_VALIDATOR


def _patch_success_telemetry(src: str) -> str:
    """Make a rollout that never observed success telemetry fail, not score zero.

    Pinned upstream `rollout` reads the success signal from two supported sources and
    otherwise falls back to `[False] * env.num_envs`:

        if "final_info" in info:      successes = final_info["is_success"].tolist()
        elif "is_success" in info:    successes = info["is_success"] ...
        else:                         successes = [False] * env.num_envs

    A rollout where neither field ever appeared therefore produced internally consistent
    zero-success metrics with no observed success signal, and at the deliberately supported
    zero threshold those can be accepted. "Measured failure" and "measurement unavailable"
    were not distinguishable.

    The else-branch is NOT patched to raise, and that is deliberate. Upstream's own comment
    says `final_info` is absent when no env has finished yet, so the fallback is the normal
    path for early steps -- raising there would break every rollout on its first step. The
    same mistake was made once in the GR00T patch and caught by review.

    Instead: record whether an authoritative source was observed at any point, and require
    it once, at the rollout boundary where the result is assembled. Per-step accumulation is
    reduced with `any`, so the fallback is non-destructive and needs no change.

    Raises RuntimeError if either anchor is absent: upstream having moved means the
    guarantee cannot be made, which must stop the run rather than proceed.
    """
    edits = [
        (
            '        if "final_info" in info:\n'
            '            final_info = info["final_info"]\n',
            '        if "final_info" in info:\n'
            '            _telemetry_observed = True\n'
            '            final_info = info["final_info"]\n',
        ),
        (
            '        elif "is_success" in info:\n'
            '            is_success = info["is_success"]\n',
            '        elif "is_success" in info:\n'
            '            _telemetry_observed = True\n'
            '            is_success = info["is_success"]\n',
        ),
        (
            "    # Stack the sequence along the first dimension so that we have "
            "(batch, sequence, *) tensors.\n"
            "    ret = {\n",
            "    if not _telemetry_observed:\n"
            "        raise RuntimeError(\n"
            "            'no success telemetry was observed during this rollout: neither '\n"
            "            'the terminal is_success field nor the per-step one ever '\n"
            "            'appeared. A rollout that could not observe success has not '\n"
            "            'measured anything; refusing to report it as unsuccessful '\n"
            "            'episodes.')\n"
            "\n"
            "    # Stack the sequence along the first dimension so that we have "
            "(batch, sequence, *) tensors.\n"
            "    ret = {\n",
        ),
    ]
    # The flag must exist before the loop reads it.
    anchor = "    all_successes = []\n"
    if src.count(anchor) < 1:
        raise RuntimeError(
            "success-telemetry patch anchor (all_successes initialisation) not found in "
            "the pinned LeRobot eval script; unobserved telemetry could again be reported "
            "as failed episodes, so refusing to run.")
    src = src.replace(
        anchor,
        anchor + "    # Whether an authoritative success source was seen at any step.\n"
                 "    _telemetry_observed = False\n", 1)
    for index, (target, replacement) in enumerate(edits, start=1):
        if src.count(target) != 1:
            raise RuntimeError(
                f"success-telemetry patch anchor {index} occurs {src.count(target)} times "
                f"in the pinned LeRobot eval script (expected exactly 1); upstream "
                f"changed. Refusing to run.")
        src = src.replace(target, replacement, 1)
    return src


def _validate_saved_processor_graphs(policy_dir: str) -> None:
    """Reject a saved processor graph that would import code of the checkpoint's choosing.

    C2 (cycle 6): the pinned LeRobot pipeline resolves each configured step one of two ways.
    A step with `registry_name` is looked up in ProcessorStepRegistry -- installed, trusted
    code. A step with `class` is a dotted path resolved through importlib and then INVOKED with
    checkpoint-supplied keyword arguments. A probe reached subprocess.Popen's constructor that
    way. Disabling network access does not help: the import is local and the side effects are in
    the constructor.

    The rewrite below only looked for the one step it needed to patch, so a valid
    molmoact2_pack_inputs step could sit alongside a malicious one and be patched successfully.

    Both graphs are validated -- preprocessor and postprocessor -- because either is loaded.
    Every step must resolve through the registry; a `class` key is refused. Registry names
    themselves are NOT enumerated: the registry only contains installed code, and enumerating
    them would risk rejecting the component's own trainer output, which is how the OpenVLA
    guard went wrong.
    """
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = os.path.join(policy_dir, name)
        if not os.path.isfile(path):
            # The preprocessor is required and checked separately below; a missing
            # postprocessor simply means there is no graph to validate.
            continue
        try:
            with open(path) as handle:
                graph = json.load(handle)
        except (OSError, ValueError) as exc:
            log(f"FATAL: saved processor graph {path} is unreadable ({exc}); refusing to "
                f"evaluate with a processor pipeline that cannot be inspected.")
            sys.exit(1)
        steps = graph.get("steps") if isinstance(graph, dict) else None
        if not isinstance(steps, list):
            log(f"FATAL: saved processor graph {path} has no steps list; refusing to "
                f"interpret it.")
            sys.exit(1)
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                log(f"FATAL: step {index} in {path} is not an object; refusing to interpret it.")
                sys.exit(1)
            if "class" in step:
                log(f"FATAL: step {index} in {path} names class {step['class']!r}. That value "
                    f"is resolved with importlib and the resulting class is invoked with "
                    f"configuration from this checkpoint, so the checkpoint would choose code "
                    f"to execute with the evaluation identity. Steps must resolve through the "
                    f"processor registry, which contains only installed code.")
                sys.exit(1)
            registry_name = step.get("registry_name")
            if not (isinstance(registry_name, str) and registry_name.strip()):
                log(f"FATAL: step {index} in {path} declares neither a usable registry_name "
                    f"nor anything this evaluator will accept. A step that cannot be resolved "
                    f"from the registry must not be loaded.")
                sys.exit(1)
        log(f"validated saved processor graph {name}: "
            f"{len(steps)} registry step(s), no dynamic imports")


# The measured identity of the archive this job read (schema 3).
# Constructed once by whichever mode resolved the checkpoint. Two seedable dicts whose
# emptiness DECIDED the mode is what let a fixture render one branch unexecutable.
_RESOLVED = None


def main() -> None:
    # PyPI 502 resilience for EVERY pip in this job (incl. build-isolation
    # subprocess pips that ignore an explicit --retries flag).
    os.environ["PIP_RETRIES"] = "10"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "60"
    log(f"suite={SUITE} task_ids={TASK_IDS or 'ALL'} episodes={EPISODES}/task seed={SEED}")
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

    # HF cache on the training volume (root volume is too small for a 22GB ckpt)
    hf_home = os.path.join(WORK, "hf-cache")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home

    # ---- [1/6] System libs (EGL headless render + ffmpeg for rollout videos)
    os.chmod("/tmp", 0o1777)  # DLC quirk found in the OpenVLA smoke test
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "--no-install-recommends",
         "git", "curl", "ca-certificates", "libegl1", "libgles2",
         "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "ffmpeg"])

    # ---- [2/6] uv (PINNED via pip -- the live curl|sh installer was flagged
    # by review as unpinned mutable input) + pinned fork clone
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
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != LEROBOT_COMMIT:
        log(f"FATAL: fork commit mismatch: {head}")
        sys.exit(1)

    # I2: make a rollout that never observed success telemetry fail rather than report
    # zero successes. Applied after the commit check, guarded by a content check so a
    # re-run is safe. The patch itself refuses if any anchor is missing, because upstream
    # having moved means the guarantee cannot be made.
    _eval_py = os.path.join(
        LEROBOT_DIR, "src/lerobot/scripts/lerobot_eval.py")
    with open(_eval_py) as _fh:
        _eval_src = _fh.read()
    _eval_dirty = False
    # C4: validate telemetry SHAPE before upstream coerces it. The observation patch marks a
    # source seen on field PRESENCE, so a present but malformed value counted as a
    # measurement -- the scalar string "false" was accepted as success.
    if "_require_boolean_success_vector" not in _eval_src:
        try:
            _eval_src = _patch_telemetry_shape(_eval_src)
        except RuntimeError as exc:
            log(f"FATAL: {exc}")
            sys.exit(1)
        _eval_dirty = True
        log("PATCHED lerobot rollout: success telemetry is validated before coercion")
    else:
        log("lerobot rollout already patched to validate telemetry shape")
    # I5: audit the REAL evaluation policy in the evaluator process. The previous audit
    # ran in a separate CPU probe and inspected a different policy instance.
    if "_audit_loaded_policy" not in _eval_src:
        try:
            _eval_src = _patch_policy_load_audit(_eval_src)
        except RuntimeError as exc:
            log(f"FATAL: {exc}")
            sys.exit(1)
        _eval_dirty = True
        log("PATCHED lerobot eval: the real policy is audited after construction")
    else:
        log("lerobot eval already patched to audit the loaded policy")
    if "no success telemetry was observed" not in _eval_src:
        try:
            _eval_src = _patch_success_telemetry(_eval_src)
        except RuntimeError as exc:
            log(f"FATAL: {exc}")
            sys.exit(1)
        _eval_dirty = True
        log("PATCHED lerobot rollout: unobserved success telemetry is now fatal")
    else:
        log("lerobot rollout already patched for success telemetry")
    if _eval_dirty:
        with open(_eval_py, "w") as _fh:
            _fh.write(_eval_src)

    # ---- [3/6] uv sync at the pinned interpreter (fork source build, ~20-30 min)
    uv_sync_resilient("--locked", "--python", PYTHON_PIN,
       "--extra", "molmoact2", "--extra", "libero")
    # Fail fast if uv ignored --python (draccus needs 3.12/3.13)
    uv("run", "--active", "python", "-c",
       "import sys; v='%d.%d' % sys.version_info[:2]; "
       f"assert v == '{PYTHON_PIN}', v; print('venv python', v)")

    # ---- [4/6] Seed LIBERO config non-interactively, then verify the venv
    # (first import prompts via input() and EOFErrors headless; answering 'N'
    # writes ~/.libero/config.yaml -- the reference recipe, kept verbatim)
    uv("run", "--active", "python", "-c",
       "import builtins; builtins.input = lambda *a, **k: 'N'\n"
       "from libero.libero import benchmark\n"
       "benchmark.get_benchmark_dict()\n"
       "import os; cfg = os.path.expanduser('~/.libero/config.yaml')\n"
       "assert os.path.exists(cfg), cfg\n"
       "print('LIBERO config seeded at', cfg)")
    uv("run", "--active", "python", "-c",
       "import torch\n"
       "assert torch.cuda.is_available(), 'CUDA not visible to torch'\n"
       "from lerobot.policies.molmoact2 import MolmoAct2Policy\n"
       "from lerobot.envs.libero import LiberoEnv\n"
       "print('verify OK:', torch.__version__, MolmoAct2Policy.__name__)")

    # ---- [4b/6] Stage LIBERO scene ASSETS (ONLINE phase, before offline mode).
    # hf-libero 0.1.3 ships NO assets/ dir; it resolves scenes/*.xml via
    # get_assets_path() = package-local <pkg>/assets OR an HF download of
    # lerobot/libero-assets. Under HF_HUB_OFFLINE=1 (set later) that download is
    # blocked -> FileNotFoundError: assets/scenes/libero_tabletop_base_style.xml.
    # Fix (mirrors OpenVLA's WORKING eval): git-clone the
    # real LIBERO @ the commit OpenVLA pins (ships the complete assets/ tree) and
    # copy assets/ into the installed hf-libero package dir, so get_assets_path()
    # short-circuits to local and never touches HF. A GitHub clone is not HF, so
    # this stays offline-eval-safe. config.yaml already supplies bddl/init.
    libero_gh = os.path.join(WORK, "LIBERO-assets-src")
    libero_asset_commit = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    run(["git", "clone", "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
         libero_gh])
    run(["git", "-C", libero_gh, "checkout", libero_asset_commit])
    src_assets = os.path.join(libero_gh, "libero", "libero", "assets")
    if not os.path.isdir(os.path.join(src_assets, "scenes")):
        log(f"FATAL: cloned LIBERO missing assets/scenes at {src_assets}")
        sys.exit(1)
    pkg_dir = subprocess.run(
        [UV_BIN, "run", "--active", "python", "-c",
         "import libero.libero, os; print(os.path.dirname(libero.libero.__file__))"],
        cwd=LEROBOT_DIR, capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        check=True).stdout.strip()
    dst_assets = os.path.join(pkg_dir, "assets")
    import shutil as _shutil
    _shutil.copytree(src_assets, dst_assets, dirs_exist_ok=True)
    if not os.path.isfile(os.path.join(dst_assets, "scenes",
                                       "libero_tabletop_base_style.xml")):
        log(f"FATAL: asset staging failed -- scenes/*.xml absent under {dst_assets}")
        sys.exit(1)
    log(f"staged LIBERO assets into {dst_assets}")

    # ---- [5/6] Resolve the checkpoint, then run the eval.
    # PIPELINE/FINE-TUNED mode: the fine-tuned checkpoint is mounted at the
    # "model" channel (SM_CHANNEL_MODEL, default /opt/ml/input/data/model); use
    # it in place. Its artifact identity (S3 VersionId + checksum) is the
    # binding, recorded in model_artifact_identity below.
    if PIPELINE_MODE:
        snapshot_path = os.environ.get("SM_CHANNEL_MODEL",
                                       "/opt/ml/input/data/model")
        if not os.path.isdir(snapshot_path):
            log(f"FATAL: mounted model channel missing: {snapshot_path!r}")
            sys.exit(1)
        # SageMaker mounts the training output as a RAW model.tar.gz in the
        # channel -- it does NOT auto-extract (proven: OpenVLA's eval extracts
        # it manually; MolmoAct2 previously walked the un-extracted channel and
        # the manifest, being INSIDE the tarball, was invisible -> false
        # "manifest not found"). Extract it to a work dir, then treat that as
        # the checkpoint root. This is the same proven path OpenVLA uses.
        tar_path = os.path.join(snapshot_path, "model.tar.gz")
        if os.path.isfile(tar_path):
            import tarfile
            extract_dir = os.path.join(WORK, "model")
            os.makedirs(extract_dir, exist_ok=True)
            log(f"extracting mounted {tar_path} -> {extract_dir}")
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
                        if os.path.commonpath([base, p]) != base or not (
                                m.isreg() or m.isdir()):
                            log(f"FATAL: unsafe tar member {m.name}")
                            sys.exit(1)
                    tf.extractall(extract_dir)
            snapshot_path = extract_dir
        else:
            # An already-extracted channel is FATAL, not a fall-through. This branch did not exist, so
            # _ARCHIVE_SHA / _ARCHIVE_SIZE were never bound and the failure surfaced later as a bare
            # NameError -- naming a variable rather than the condition. Pipeline mode REQUIRES a
            # measurable archive: the pre-extraction measurement is the only value Validate can
            # compare against its own, so a checkpoint arriving already unpacked cannot be attributed
            # to the bytes the training step produced.
            _listing = sorted(os.listdir(snapshot_path))[:20]
            log(f"FATAL: no model.tar.gz in the mounted channel {snapshot_path!r}. Pipeline mode needs "
                f"the RAW archive: its pre-extraction measurement is what Validate compares against "
                f"its own, and an already-extracted tree cannot supply one. Found: {_listing}")
            sys.exit(1)
        # Option C layout: the extracted root holds checkpoint_manifest.json with
        # sibling policy/ and base/ dirs. Descend to the dir CONTAINING the
        # manifest (NOT config.json -- config.json now lives inside policy/, and
        # descending there would break the whole-tree digest binding).
        if not os.path.exists(os.path.join(snapshot_path, "checkpoint_manifest.json")):
            subdirs = [os.path.join(snapshot_path, d)
                       for d in os.listdir(snapshot_path)
                       if os.path.isdir(os.path.join(snapshot_path, d))]
            for d in subdirs:
                if os.path.exists(os.path.join(d, "checkpoint_manifest.json")):
                    snapshot_path = d
                    break
        log(f"pipeline mode: evaluating mounted checkpoint tree at {snapshot_path}")
    # Belt-and-suspenders: the digest itself will refuse any residual symlink.

    # weights_digest over the resolved checkpoint = the content identity the
    # report carries; recomputed below over the same tree the evaluator loads.
    log("computing weights digest over the checkpoint ...")
    snapshot_digest = weights_digest(snapshot_path)
    log(f"weights_digest: {snapshot_digest}")
    # Artifact identity binding (pipeline mode only): HeadObject the source
    # model.tar.gz so the report records the exact bytes evaluated.
    model_artifact_identity = None
    if PIPELINE_MODE:
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
    if PIPELINE_MODE:
        # The manifest sits at the checkpoint ROOT the FineTune wrote, but the
        # descend-to-config.json logic above may have moved snapshot_path into a
        # nested subdir -> look next to the checkpoint first, then anywhere under
        # the mount (it is a single small file). This decouples "where the
        # weights live" from "where the manifest lives" (Sol contract: the
        # FineTune writes it; we must find it, not fabricate).
        # Search the RESOLVED checkpoint tree (the extracted tarball), not the
        # raw channel -- the manifest lives inside the extracted content.
        mount_root = snapshot_path
        manifest_path = os.path.join(snapshot_path, "checkpoint_manifest.json")
        if not os.path.isfile(manifest_path):
            found = None
            for _root, _dirs, _files in os.walk(mount_root):
                if "checkpoint_manifest.json" in _files:
                    found = os.path.join(_root, "checkpoint_manifest.json")
                    break
            manifest_path = found
        if not manifest_path or not os.path.isfile(manifest_path):
            log("FATAL: checkpoint_manifest.json not found anywhere under the "
                f"mounted checkpoint {mount_root!r} -- the FineTune step must "
                "write it (contract rev6); refusing to fabricate")
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
        if _declared_family != 'molmoact2':
            log(f"FATAL: this checkpoint was produced for model_family={_declared_family!r} but "
                f"is being evaluated as 'molmoact2'. Its manifest describes a different family's "
                f"input_config, so every field read from here would be wrong or missing.")
            sys.exit(1)
        from validator import validate_manifest as _validate_manifest
        _validate_manifest(checkpoint_manifest, FAMILY_SCHEMAS)
        # Content binding: recompute the digest over the SAME tree the manifest
        # was written over (the checkpoint dir CONTAINING the manifest), which
        # is what train's weights_digest(MODEL_DIR) covered -- not necessarily
        # the descended weights-only subdir.
        manifest_tree = os.path.dirname(manifest_path)
        binding_digest = weights_digest(manifest_tree)
        if binding_digest != checkpoint_manifest["weights_digest"]:
            log(f"FATAL: digest mismatch -- mounted bytes are not the produced bytes\n"
                f"  manifest: {checkpoint_manifest['weights_digest']}\n"
                f"  recomputed over {manifest_tree}: {binding_digest}")
            sys.exit(1)
        # The binding tree IS the checkpoint root (base/ + policy/ siblings).
        snapshot_path = manifest_tree
        snapshot_digest = binding_digest
        # Built at the tree the digest was just verified against, from the pre-extraction archive
        # measurement. Constructing at the extraction root would name a tree the report's own
        # weights_digest does not describe.
        _RESOLVED = si_from_archive(load_root=manifest_tree, archive_path=str(tar_path),
                                    checkpoint=manifest_tree,
                                    sha256=_ARCHIVE_SHA, size_bytes=_ARCHIVE_SIZE)
        ic = checkpoint_manifest.get("input_config", {})
        # Each check used to be `if ic.get(key) and ic[key] != expected`, so an ABSENT or
        # EMPTY declaration skipped it entirely -- while the evaluator went on to
        # hardcode float32, continuous and its own camera map regardless. The manifest
        # could therefore claim nothing about the settings actually used. The evaluator
        # only supports one value for each, so require it to be declared and to match:
        # an empty string is not a weaker claim, it is an absent one.
        _supported = {
            "model_dtype": "float32",
            "inference_action_mode": "continuous",
            "camera_name_mapping": CAMERA_MAP,
        }
        _unsupported = {}
        _undeclared = []
        for _key, _expected in _supported.items():
            _value = ic.get(_key)
            if not (isinstance(_value, str) and _value.strip()):
                _undeclared.append(_key)
            elif _value != _expected:
                _unsupported[_key] = _value
        if _undeclared:
            log(f"FATAL: checkpoint manifest does not declare the inference settings "
                f"{_undeclared}. The evaluator applies fixed values for these, so an "
                f"undeclared or empty setting means the recorded input configuration "
                f"does not describe the evaluation that will run.")
            sys.exit(1)
        if _unsupported:
            log(f"FATAL: checkpoint manifest declares inference settings that differ "
                f"from the hardcoded evaluator defaults: {_unsupported}. "
                "MolmoAct2 evaluation currently supports float32/continuous only.")
            sys.exit(1)
        # Capture the verified digest NOW, before the processor-config patch below
        # mutates policy_preprocessor.json. This is the value the report carries as
        # weights_digest_recomputed_by_eval; recomputing after the patch would
        # (correctly) differ and fail the binding, so we bind to the pre-patch
        # verified tree -- the exact bytes we validated against the manifest.
        verified_tree_digest = binding_digest
        # Option C: resolve the two subdirs the evaluator needs. policy_dir is
        # the LoRA output (--policy.path); base_dir is the pinned base the loader
        # strict-loads (--policy.checkpoint_path, absolute, injected below).
        policy_dir = os.path.join(snapshot_path, "policy")
        base_dir = os.path.join(snapshot_path, "base")
        for _d, _label in ((policy_dir, "policy"), (base_dir, "base")):
            if not os.path.isfile(os.path.join(_d, "config.json")):
                log(f"FATAL: Option-C {_label}/ missing config.json at {_d}")
                sys.exit(1)
        if not os.path.isfile(os.path.join(policy_dir, "model.safetensors")):
            log(f"FATAL: policy/model.safetensors missing at {policy_dir}")
            sys.exit(1)
        def _fail(msg):
            log(f"FATAL: {msg}")
            sys.exit(1)
        _resolve_norm_stats(base_dir, log, _fail)
        fast_tok_dir = os.path.join(snapshot_path, "fast_tokenizer")
        if not os.path.isfile(os.path.join(fast_tok_dir, "tokenizer.json")):
            log(f"FATAL: fast_tokenizer/tokenizer.json missing at {fast_tok_dir} "
                "-- the processor's discrete-action step needs it offline")
            sys.exit(1)
        # PATCH the SAVED processor config so its molmoact2_pack_inputs step
        # resolves to LOCAL dirs, not HF. The processor pipeline is loaded from
        # policy/policy_preprocessor.json (NOT from policy.config), so the
        # --policy.checkpoint_path override does NOT reach it; the step has its
        # own checkpoint_path (base) + discrete_action_tokenizer (FAST) that
        # otherwise hit HF and fail under offline mode. _resolve_*_location both
        # short-circuit on an existing local Path, so absolute local dirs here
        # make the load fully offline. This mutates the tree, so it runs AFTER
        # the digest-binding check above (which already verified the shipped
        # bytes) -- and the eval-side report recompute is taken BEFORE this patch.
        # C2: validate BOTH saved graphs before touching either. The rewrite below only
        # looks for the step it needs, so a malicious step could ride alongside it.
        _validate_saved_processor_graphs(policy_dir)
        prep_path = os.path.join(policy_dir, "policy_preprocessor.json")
        if not os.path.isfile(prep_path):
            log(f"FATAL: policy_preprocessor.json missing at {prep_path}")
            sys.exit(1)
        with open(prep_path) as fh:
            prep = json.load(fh)
        patched = False
        for step in prep.get("steps", []):
            if step.get("registry_name") == "molmoact2_pack_inputs":
                step["config"]["checkpoint_path"] = base_dir
                step["config"]["discrete_action_tokenizer"] = fast_tok_dir
                patched = True
        if not patched:
            log("FATAL: molmoact2_pack_inputs step not found in "
                "policy_preprocessor.json -- cannot redirect processor offline")
            sys.exit(1)
        with open(prep_path, "w") as fh:
            json.dump(prep, fh, indent=2)
        log(f"patched processor -> local base+fast_tokenizer (offline): {prep_path}")
        log("pipeline mode: read checkpoint_manifest.json from mounted checkpoint "
            f"(base={checkpoint_manifest['base_checkpoint']}, "
            f"seed={checkpoint_manifest.get('train_seed')})")
    # A silent base-only policy would score the base model and corrupt the benchmark,
    # because the fork loads the LoRA model.safetensors with strict=False and mismatches
    # only WARN. That risk is addressed by the live-policy audit further down, which
    # reconstructs the policy and makes those diagnostics fatal. It replaced a
    # file-level check over the saved tensors that could not detect the case at all.
    #
    # Is this a LoRA policy? Decide from the SAVED POLICY CONFIG, which is what the
    # loader itself reads: `__init__` re-wraps with PEFT because policy/config.json
    # carries train_mode_vlm=lora. The audit used to be gated on the manifest's
    # train_recipe.train_mode_vlm, which is an OPTIONAL provenance field -- so deleting
    # that one declaration disabled the whole audit without changing the saved policy
    # at all. The manifest is now a cross-check, not the gate: if it declares a mode
    # and the saved config disagrees, one of them is describing a different checkpoint
    # and neither can be trusted.
    saved_mode = _resolve_train_mode(policy_dir, checkpoint_manifest)

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # OFFLINE: the base loads from the LOCAL base_dir and must NEVER hit HF. If it
    # tries, fail loudly (fail-closed) rather than silently pulling a wrong rev.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env.pop("HF_TOKEN", None)
    env.pop("HF_ACCESS_TOKEN", None)

    # Audit the INSTANTIATED policy, not the saved file.
    #
    # The previous audit inspected model.safetensors for `lora_` names and a nonzero
    # lora_B, and a comment called that "the robust, cheap equivalent of a runtime
    # probe". It was not. The evaluator loads through the ordinary LeRobot path, whose
    # from_pretrained defaults to strict=False and only LOGS missing/unexpected keys, so
    # a checkpoint whose tensor names do not match the instantiated policy passed the
    # file audit while non-strict loading left some or all of the policy at its
    # base/initialised state. Recomputed file digests still agreed, because the bytes
    # were genuine -- only their effect was absent.
    #
    # This reconstructs the policy through the same factory the evaluator uses, makes
    # the loader's missing/unexpected-key diagnostics FATAL (equivalent to strict
    # loading, which is supported on this path: LoRA re-wrapping does not require
    # accepting mismatched keys for checkpoints from this pinned trainer), and then
    # inspects the live PEFT layers -- enabled, selected, unmerged, values equal to the
    # stored tensors, and at least one nonzero effective LoRA delta.
    #
    # It runs on CPU in a separate process so it does not compete with the evaluation
    # for GPU memory. What it does NOT establish, and must not be read as establishing:
    # that training occurred or used the declared data; that every adapter changed; that
    # a nonzero delta changes the final action; that a delta was not previously baked
    # into the base weights; that non-adapter values are numerically correct; or that
    # the later CUDA process is byte-identical. It audits reconstruction and adapter
    # state only.
    if PIPELINE_MODE:
        probe_env = dict(env)
        probe_env["CUDA_VISIBLE_DEVICES"] = ""
        log(f"auditing instantiated MolmoAct2 policy on CPU (mode={saved_mode!r}) ...")
        uv("run", "--active", "python", "-c", _POLICY_LOAD_PROBE,
           policy_dir,
           base_dir,
           json.dumps(checkpoint_manifest),
           SUITE,
           TASK_IDS or "all",
           env=probe_env)

    os.makedirs(EVAL_OUT, exist_ok=True)
    eval_cmd = [
        UV_BIN, "run", "--active", "lerobot-eval",
        # Load the LoRA policy dir (routes MolmoAct2Policy.from_pretrained; the
        # saved config keeps train_mode_vlm=lora so __init__ re-wraps with PEFT).
        # --policy.path is mutually exclusive with --policy.type, so type is DROPPED.
        f"--policy.path={policy_dir}",
        # ABSOLUTE base override (config's checkpoint_path is a placeholder; a
        # relative path resolves against the eval CWD, not the config dir).
        f"--policy.checkpoint_path={base_dir}",
        "--policy.norm_tag=libero",
        "--policy.inference_action_mode=continuous",
        "--policy.model_dtype=float32",
        "--policy.use_amp=false",
        "--policy.enable_inference_cuda_graph=true",
        "--policy.device=cuda",
        "--env.type=libero",
        f"--env.task={SUITE}",
        f"--env.camera_name_mapping={CAMERA_MAP}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={EPISODES}",   # PER TASK (10 tasks x 20 = 200 total)
        f"--output_dir={EVAL_OUT}",
        f"--seed={SEED}",
    ]
    if TASK_IDS and TASK_IDS != "all":
        eval_cmd.append(f"--env.task_ids={TASK_IDS}")
    # Trace boundary: lerobot_eval loads the policy onto a CUDA device, so on a CPU L3 job this
    # chunk OPENS and never closes -- read_trace reports it as the death point, which is the GPU
    # boundary this rung exists to locate. (molmoact2 eval reaches here only in pipeline mode.)
    log(">>> lerobot_eval rollout via lerobot_eval (model load binds a GPU device)")
    log("$ " + " ".join(eval_cmd))
    eval_log_path = "/tmp/molmoact2_eval.log"
    with open(eval_log_path, "w") as lf:
        proc = subprocess.Popen(eval_cmd, cwd=LEROBOT_DIR, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
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
        _hang_timer.start()

        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        rc = proc.wait()
        _hang_timer.cancel()
        if _timed_out["v"]:
            log("FATAL: lerobot-eval exceeded the runtime budget and was killed")
            sys.exit(124)
    if rc != 0:
        log(f"FATAL: lerobot-eval exited {rc}")
        sys.exit(rc)

    # ---- [6/6] Parse eval_info.json -> this repo's metrics shape
    info_path = os.path.join(EVAL_OUT, "eval_info.json")
    if not os.path.isfile(info_path):
        log("FATAL: eval_info.json not produced")
        sys.exit(1)
    with open(info_path) as fh:
        info = json.load(fh)
    pc = info.get("overall", {}).get("pc_success")
    if pc is None:
        log("FATAL: overall.pc_success missing from eval_info.json")
        sys.exit(1)
    success_rate = float(pc) / 100.0

    # Episode/task accounting from the EVALUATOR'S OWN report (note
    # #2): per_task records with successes lists, plus overall.n_episodes.
    # Config-derived counts are used only as expectations to cross-check.
    reported_episodes = info.get("overall", {}).get("n_episodes")
    raw_per_task = info.get("per_task")
    if not isinstance(reported_episodes, int) or not isinstance(raw_per_task, list) or not raw_per_task:
        log("FATAL: eval_info.json lacks overall.n_episodes or per_task records")
        sys.exit(1)
    per_task = []
    for rec in raw_per_task:
        successes = rec.get("metrics", {}).get("successes")
        if not isinstance(successes, list) or not successes:
            log(f"FATAL: per_task record missing successes: {rec.get('task_id')!r}")
            sys.exit(1)
        # Contract rule: successes must be real JSON booleans -- a truthy
        # non-boolean (e.g. 1.0, "true") indicates an evaluator format change
        # and must fail loudly, not be coerced (note).
        if any(not isinstance(s, bool) for s in successes):
            log(f"FATAL: non-boolean success values for task {rec.get('task_id')!r}: {successes!r}")
            sys.exit(1)
        per_task.append({
            "task_id": rec["task_id"],
            "task": f"{rec.get('task_group', SUITE)}_{rec['task_id']}",
            "episodes": len(successes),
            "success_rate": sum(1 for s in successes if s) / len(successes),
        })
    episodes_from_tasks = sum(t["episodes"] for t in per_task)
    if episodes_from_tasks != reported_episodes:
        log(f"FATAL: per-task episode sum {episodes_from_tasks} != overall.n_episodes {reported_episodes}")
        sys.exit(1)
    expected = int(EPISODES) * (len(json.loads(TASK_IDS)) if (TASK_IDS and TASK_IDS != "all") else 10)
    if reported_episodes != expected:
        log(f"FATAL: evaluator ran {reported_episodes} episodes, config expected {expected}")
        sys.exit(1)
    mean_rate = sum(t["success_rate"] * t["episodes"] for t in per_task) / reported_episodes
    if abs(mean_rate - success_rate) > 1e-4:
        log(f"FATAL: per-task weighted mean {mean_rate} != overall {success_rate}")
        sys.exit(1)

    # Explicit task set (contract: null is fatal -- state what actually ran)
    explicit_task_ids = sorted(json.loads(TASK_IDS)) if (TASK_IDS and TASK_IDS != "all") else list(range(10))

    # Content binding the gate checks. In PIPELINE mode we bind to the digest
    # verified against the manifest BEFORE the offline processor-config patch
    # mutated the tree (recomputing now would differ because we edited
    # policy_preprocessor.json to point at local dirs).
    recomputed = verified_tree_digest

    metrics = {
        "schema_version": 3,
        # EXACTLY ONE variant, matching the validator: both would claim two sources and
        # establish neither, and an empty archive is what I4 was.
        **_RESOLVED.report_identity_fields(),
        "policy_type": "checkpoint",
        "model_family": "molmoact2",
        # checkpoint and checkpoint_revision come from the spread above. They were re-assigned HERE,
        # four lines later, which silently DEFEATED the single-source contract: the whole point of
        # report_identity_fields() is that these cannot disagree with the measured source, and a
        # later key in the same dict literal wins. Never re-assign a field the spread provides.
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
            "recipe_repo": LEROBOT_REPO,
            "recipe_commit": LEROBOT_COMMIT,
            "mujoco_gl": env["MUJOCO_GL"],
            "lerobot_branch": LEROBOT_BRANCH,
            "python_version": PYTHON_PIN,
        },
    }

    # SELF-VALIDATION with the shared gate validator: a contract violation
    # fails the eval job itself, before any pipeline step consumes the report.
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
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Evidence: full eval_info.json, eval log, resolved environment, videos
    with open(info_path) as src, open(os.path.join(OUT_DIR, "eval_info.json"), "w") as dst:
        dst.write(src.read())
    with open(eval_log_path) as src, open(os.path.join(OUT_DIR, "eval_log.txt"), "w") as dst:
        dst.write(src.read())
    # uv venvs do not bundle pip; `uv pip freeze` lists the active venv directly
    # (python -m pip freeze died here on run 221520 after a fully green eval).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    freeze = subprocess.run(
        [UV_BIN, "pip", "freeze"],
        cwd=LEROBOT_DIR, capture_output=True, text=True, env=env)
    if freeze.returncode != 0 or not freeze.stdout.strip():
        log("FATAL: pip freeze failed -- provenance manifest cannot be empty")
        sys.exit(1)
    with open(os.path.join(OUT_DIR, "runtime_manifest.txt"), "w") as fh:
        fh.write(f"# lerobot_fork={LEROBOT_REPO}@{LEROBOT_COMMIT} ({LEROBOT_BRANCH})\n")
        fh.write(f"# checkpoint={HF_REPO}@{HF_REV}\n# python={PYTHON_PIN}\n")
        fh.write(freeze.stdout)
    import shutil
    vid_count = 0
    for root, _dirs, files in os.walk(EVAL_OUT):
        for f in files:
            if f.endswith(".mp4") and vid_count < 10:
                shutil.copy(os.path.join(root, f), OUT_DIR)
                vid_count += 1
    log(f"copied {vid_count} rollout videos")
    log(f"DONE success_rate={success_rate}")


if __name__ == "__main__":
    main()
