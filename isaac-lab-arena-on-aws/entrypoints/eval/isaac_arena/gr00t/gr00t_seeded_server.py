#!/usr/bin/env python
"""Seed the GR00T inference server's RNG, then run the pinned server unchanged.

WHY THIS EXISTS
---------------
The evaluation seed reached the server process's ENVIRONMENT but nothing in the server
read it, so the policy's inference randomness was unbound:

- ``_build_server_env()`` copies ``os.environ``, so ``EVAL_SEED`` is present in the child.
- Neither pinned server (N1.6 or N1.7) reads that variable, and neither ``ServerConfig``
  exposes a seed argument.
- Arena receives ``--seed`` and its runner seeds the CLIENT process and environment
  (Python, NumPy, Torch CPU/CUDA, ``env.unwrapped.seed``). That covers scene/initial-state
  randomness, not the policy.
- Both action heads sample inference noise through ``torch.randn`` in the SERVER process.
- Arena calls the server's reset endpoint, but both pinned ``Gr00tPolicy.reset()``
  implementations return ``{}``, so passing a seed in reset options would not apply it.
- N1.7 ships ``gr00t.utils.determinism.seed_everything()`` which optionally reads
  ``GR00T_EVAL_SEED``, but the server path never calls it. Renaming the variable would
  therefore not have been enough.

So a rerun with the same ``EVAL_SEED`` reproduced the same scene sequence while the policy
sampled different action noise, and ``seed_scope`` overstated what was reproducible.

WHAT THIS DOES
--------------
Seeds Python, NumPy and Torch (CPU and every visible CUDA device) in the server process,
applies an explicit determinism policy, and reseeds once more at the INFERENCE-READY
boundary -- immediately after the policy object finishes construction and before the
server accepts requests. The second seeding is the load-bearing one: model construction
consumes a checkpoint-dependent amount of RNG, so seeding only at process start would let
different dose checkpoints begin inference from different stream positions.

It seeds ONCE per server run. Episode resets do NOT reseed, so episodes within a run are
not independently replayable; reproducing a single episode requires rerunning the whole
sequence. That is a deliberate limitation of the smallest correct change -- per-episode
replay needs a seed schedule and an explicit reset protocol the pinned servers do not have.

WHAT THIS DOES NOT ESTABLISH
----------------------------
Bit-identical results across different GPUs, driver or library versions, or with CUDA
graph capture and non-deterministic kernels in play. It binds the RNG streams this
process controls; it does not make the whole stack numerically deterministic.
"""
from __future__ import annotations

import json
import os
import re
import runpy
import sys

EVIDENCE_PATH = os.environ.get(
    "GR00T_SERVER_SEED_EVIDENCE", "/tmp/gr00t_server_seed_evidence.json")
SERVER_MODULE = "gr00t.eval.run_gr00t_server"
# Named so a reader can require this exact shape rather than accepting any nonempty
# JSON object as evidence.
EVIDENCE_SCHEMA = "gr00t_server_evidence_v1"


def _fail(message: str) -> "None":
    print(f"[seeded-server] FATAL: {message}", file=sys.stderr, flush=True)
    print(f"[seeded-server] FATAL: {message}", flush=True)
    raise SystemExit(1)


def _required_seed() -> int:
    """The evaluation seed, required. An unseeded server is the defect being fixed."""
    raw = os.environ.get("EVAL_SEED", "").strip()
    if not raw:
        _fail("EVAL_SEED is not set. The server's inference RNG would be unbound and "
              "the reported seed_scope would overstate reproducibility. Refusing to "
              "start an unseeded policy server.")
    try:
        seed = int(raw)
    except ValueError:
        _fail(f"EVAL_SEED={raw!r} is not an integer.")
    # numpy's legacy seeding accepts [0, 2**32-1]; keep the whole chain in that range so
    # every generator receives the same value rather than a silently truncated one.
    if not 0 <= seed < 2 ** 32:
        _fail(f"EVAL_SEED={seed} is outside [0, 2**32); generators would receive "
              f"different values.")
    return seed


def _seed_everything(seed: int, phase: str) -> dict:
    """Seed every RNG this process controls. Returns what was applied.

    Python and NumPy are seeded unconditionally; Torch is seeded when importable and the
    outcome is RECORDED either way. Torch's absence is not swallowed as "fine" -- main()
    requires it before starting, because the action heads sample inference noise through
    torch.randn and a server without torch cannot serve at all. Splitting it this way
    keeps the seeding behaviour exercisable outside the GPU image.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    applied = {
        "phase": phase,
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_cpu": False,
        "torch_cuda_devices": 0,
    }
    try:
        import torch
    except ImportError:
        applied["torch"] = "unavailable"
        print(f"[seeded-server] seeded {phase}: seed={seed} (torch unavailable)",
              flush=True)
        return applied
    torch.manual_seed(seed)
    applied["torch_cpu"] = True
    if torch.cuda.is_available():
        applied["torch_cuda_devices"] = torch.cuda.device_count()
        torch.cuda.manual_seed_all(seed)
    print(f"[seeded-server] seeded {phase}: seed={seed} "
          f"cuda_devices={applied['torch_cuda_devices']}", flush=True)
    return applied


def _require_torch() -> None:
    """The server cannot serve without torch, so its absence must stop the run."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        _fail(f"torch is not importable in the server venv ({exc}). The action head "
              f"samples inference noise through torch.randn, so the policy RNG cannot be "
              f"bound and the server cannot serve.")


def _apply_determinism_policy() -> dict:
    """Apply an explicit, RECORDED determinism policy.

    cuDNN benchmarking is disabled because it selects kernels by timing, which can pick a
    different algorithm run to run. Full deterministic algorithm enforcement is NOT
    turned on: it makes some required kernels raise, which would trade a real evaluation
    for a crash. What is applied is recorded so the report cannot imply more.
    """
    policy = {"cudnn_deterministic": None, "cudnn_benchmark": None,
              "torch_deterministic_algorithms": False}
    try:
        import torch
    except ImportError:
        policy["torch"] = "unavailable"
        print(f"[seeded-server] determinism policy: {policy}", flush=True)
        return policy
    backends = getattr(torch.backends, "cudnn", None)
    if backends is not None:
        backends.deterministic = True
        backends.benchmark = False
        policy["cudnn_deterministic"] = True
        policy["cudnn_benchmark"] = False
    print(f"[seeded-server] determinism policy: {policy}", flush=True)
    return policy


def _install_inference_ready_reseed(seed: int, evidence: dict) -> None:
    """Reseed immediately after the policy is constructed, before requests are served.

    Model construction consumes a checkpoint-dependent amount of RNG, so seeding only at
    process start would leave different dose checkpoints starting inference from
    different stream positions. Wrapping the policy constructor puts the reseed exactly
    at the inference-ready boundary without touching the pinned server's own code.
    """
    try:
        from gr00t.policy import gr00t_policy as policy_module
    except ImportError as exc:
        _fail(f"cannot import gr00t.policy.gr00t_policy to bind the inference RNG "
              f"({exc}). The pinned server layout changed; refusing to run with an "
              f"unbound policy RNG.")
    policy_class = getattr(policy_module, "Gr00tPolicy", None)
    if policy_class is None:
        _fail("gr00t.policy.gr00t_policy.Gr00tPolicy not found, so the inference RNG "
              "cannot be bound at the inference-ready boundary.")
    original_init = policy_class.__init__

    def seeded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        evidence["inference_ready"] = _seed_everything(seed, "inference_ready")
        _write_evidence(evidence)

    policy_class.__init__ = seeded_init
    print("[seeded-server] inference-ready reseed installed on "
          f"{policy_class.__module__}.{policy_class.__qualname__}", flush=True)


def _install_strict_load_audit(evidence: dict) -> None:
    """Reject a model whose weights did not all come from the checkpoint.

    C3: Gr00tPolicy loads through AutoModel.from_pretrained(model_dir) without requesting
    or checking loading diagnostics. The pinned Transformers initializes missing tensors,
    warns about missing keys, and returns the model -- so a checkpoint missing tensors yields
    a running model containing randomly initialized weights, and the run is still labelled
    policy_type=checkpoint.

    A digest cannot catch this. It proves an incomplete checkpoint was transferred intact; it
    says nothing about whether the running model consisted entirely of checkpoint tensors.

    Audit the actual serving process, including N1.7's direct Qwen loader.
    Its nested Cosmos load bypasses AutoModel, so auditing only AutoModel records
    the local checkpoint but misses the external backbone's consumed revision.
    """
    try:
        import transformers
    except ImportError as exc:
        _fail(f"transformers is not importable ({exc}), so the strict load audit cannot be "
              f"installed and a silently-initialized model could produce a registrable "
              f"result.")
    auto_model = getattr(transformers, "AutoModel", None)
    if auto_model is None or not hasattr(auto_model, "from_pretrained"):
        _fail("transformers.AutoModel.from_pretrained not found, so weight loading cannot "
              "be audited. Refusing to run a policy whose weights are unverified.")

    loaders = [auto_model]
    # N1.6's pinned Transformers has no Qwen3-VL class and does not use Cosmos.
    qwen_model = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if qwen_model is not None:
        loaders.append(qwen_model)
    for loader in loaders:
        _audit_model_loader(loader, evidence)
    evidence["strict_load_audit"] = "installed"
    print("[seeded-server] strict load audit installed on "
          + ", ".join(f"{loader.__name__}.from_pretrained" for loader in loaders),
          flush=True)


def _audit_model_loader(model_class, evidence: dict) -> None:
    """Preserve each loader's return contract while checking its real load result."""
    original = model_class.from_pretrained

    def audited_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        # If a caller already asks for the info, respect its contract and return the pair.
        caller_wants_info = bool(kwargs.get("output_loading_info"))
        kwargs["output_loading_info"] = True
        try:
            model, info = original(pretrained_model_name_or_path, *args, **kwargs)
        except TypeError as exc:
            _fail(f"AutoModel.from_pretrained rejected output_loading_info ({exc}), so "
                  f"loading cannot be audited. The pinned Transformers supports it; refusing "
                  f"to load weights that cannot be checked.")
        problems = {}
        # I7: a field that is ABSENT was previously read as empty, so a load reporting no
        # diagnostics at all -- because the structure changed, or because something returned a
        # partial object -- passed as clean. Absence means the diagnostics could not be read,
        # which is not the same as there being no problems. All four must be present.
        missing_fields = []
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if isinstance(info, dict):
                present = key in info
                values = info.get(key)
            else:
                present = hasattr(info, key)
                values = getattr(info, key, None)
            if not present or values is None:
                missing_fields.append(key)
                continue
            if values:
                problems[key] = list(values)
        if missing_fields:
            _fail(f"loading diagnostics for {pretrained_model_name_or_path} are incomplete: "
                  f"{missing_fields} absent. An unreadable diagnostic is not a clean load, so "
                  f"the weights cannot be shown to have come from the checkpoint.")
        if problems:
            summary = "; ".join(
                f"{key}={values[:8]}{'...' if len(values) > 8 else ''} "
                f"({len(values)} total)" for key, values in problems.items())
            _fail(f"strict load audit FAILED for {pretrained_model_name_or_path}: {summary}. "
                  f"Missing tensors are randomly initialized by Transformers and the model "
                  f"still runs, so this would have produced an evaluation of a partly random "
                  f"policy reported as a checkpoint policy.")
        # C1: WHICH backbone did inference consume? The commit was previously taken from the PRECACHE
        # SUBPROCESS -- a different process, which called snapshot_download with NO revision and had
        # its commit scraped from the returned cache path. This server runs ONLINE (only MolmoAct2
        # sets HF_HUB_OFFLINE), so its own from_pretrained can legitimately resolve to a DIFFERENT
        # commit than the precache did, and the report would name the precache's.
        #
        # Recorded HERE because this hook runs inside the serving process, at the load that actually
        # produces the weights inference uses. `_commit_hash` is set by transformers on a hub load.
        _consumed = getattr(getattr(model, "config", None), "_commit_hash", None)
        # A hub repo id is "org/name": RELATIVE, exactly one slash, no path syntax. My first version
        # tested only for a slash, which classified the absolute local path "/ckpt" as a hub id --
        # caught immediately by an existing test, which is the argument for running the whole suite
        # after touching a shared seam.
        _path_str = str(pretrained_model_name_or_path)
        _is_hub_form = bool(
            not os.path.isabs(_path_str)
            and not os.path.isdir(_path_str)
            and re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", _path_str))
        if _is_hub_form and not (isinstance(_consumed, str)
                                 and re.fullmatch(r"[0-9a-f]{40}", _consumed)):
            _fail(f"loaded {pretrained_model_name_or_path} from the hub but could not read the "
                  f"commit it resolved to (got {_consumed!r}). An external backbone whose revision "
                  f"cannot be established is unattributable: the preprocessing it determines could "
                  f"change while the checkpoint, its digest, the image and the sourcedir all stay "
                  f"identical.")
        evidence.setdefault("strict_load_audits", []).append({
            "path": str(pretrained_model_name_or_path),
            "loader": model_class.__name__,
            # The commit THIS process resolved, or an explicit realpath for a local load. Absent is
            # recorded as null rather than omitted, so a reader can tell "no hub load" from "not
            # recorded".
            "consumed_commit": _consumed if _is_hub_form else None,
            "consumed_realpath": (None if _is_hub_form
                                  else os.path.realpath(str(pretrained_model_name_or_path))),
            "missing_keys": 0, "unexpected_keys": 0,
            "mismatched_keys": 0, "error_msgs": 0,
        })
        _write_evidence(evidence)
        print(f"[seeded-server] strict load audit PASSED for "
              f"{pretrained_model_name_or_path}: no missing, unexpected or mismatched "
              f"tensors", flush=True)
        return (model, info) if caller_wants_info else model

    model_class.from_pretrained = audited_from_pretrained


# C1 (cycle 6): the VLM backbone a GR00T checkpoint's saved processor is allowed to name.
#
# The saved config carries processor_kwargs.model_name, and the pinned processor passes it
# into Qwen3VLProcessor.from_pretrained with transformers_loading_kwargs defaulting to
# trust_remote_code=True. So a checkpoint could select a local directory or hub repo whose
# image-processor code is then imported and EXECUTED with the evaluation identity. A correct
# weights digest and a clean weight-load audit say nothing about this: the tensors are
# genuine and the executed code came from somewhere else.
#
# nvidia/Cosmos-Reason2-2B is the pinned upstream default and the backbone this component
# pre-caches and gates HF_TOKEN on. Overridable for a deliberate change, but a checkpoint
# cannot choose it.
TRUSTED_VLM_BACKBONES = tuple(
    name.strip() for name in os.environ.get(
        "GR00T_TRUSTED_VLM_BACKBONES",
        # C5 (cycle 8): this listed only the N1.7 backbone, so the guard rejected valid N1.6
        # checkpoints -- including this component's OWN trainer output, since native N1.6 Arena
        # runs through this same wrapper and train_entry copies the saved processor files into
        # the model artifact. Verified at N1.6's own pinned commit
        # 5dc80c4afd726b34faad1d8f7e007a13b34e4c88 (NOT the N1.7 commit, where no n1d6 module
        # exists at all): gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py:124 defaults
        # model_name to nvidia/Eagle-Block2A-2B-v2, and :49 asserts that exact string.
        #
        # Both entries are upstream DEFAULTS for their pinned version, not checkpoint choices:
        # the point of the guard is that a checkpoint cannot select a backbone, and it still
        # cannot -- it can only match one of the two versions this component supports.
        "nvidia/Cosmos-Reason2-2B,nvidia/Eagle-Block2A-2B-v2").split(",")
    if name.strip())
# Processor classes a real GR00T checkpoint names. Anything else would be checkpoint code.
# C5: N1.6 saves Gr00tN1d6Processor (processing_gr00t_n1d6.py:108) and Gr00tN1d6DataCollator
# (:56). Omitting them rejected the documented native N1.6 Arena workflow.
TRUSTED_PROCESSOR_CLASSES = ("Gr00tN1d7Processor", "Gr00tN1d7DataCollator",
                             "Gr00tN1d6Processor", "Gr00tN1d6DataCollator")


def _validate_saved_processor(checkpoint_dir: str) -> None:
    """Reject a checkpoint whose saved processor would execute code of its choosing.

    Runs BEFORE any policy is constructed, in the real server process. Validates the saved
    processor configuration rather than the weights, because the weights are not the problem.
    """
    # C1: this searched only ROOT-level processor_config.json / preprocessor_config.json and
    # returned clean when neither existed, on the assumption that absence means upstream
    # defaults apply. That assumption is false. At the pinned commit, upstream resolves:
    #
    #   processor_dir = model_dir / "processor"
    #       if (model_dir / "processor").is_dir()
    #       and not (model_dir / "processor_config.json").exists()
    #       else model_dir
    #
    # (gr00t/policy/gr00t_policy.py:118-124). So a checkpoint carrying config.json,
    # model.safetensors and processor/processor_config.json passed this guard entirely and then
    # had its nested processor loaded -- including a processor_kwargs.model_name selecting a
    # local, checkpoint-supplied backbone, which reaches from_pretrained with trust_remote_code
    # enabled. A valid archive digest and clean weight diagnostics do not prevent that code
    # from executing under the evaluation identity.
    #
    # Note the second mismatch this also fixes: upstream's fallback keys ONLY on
    # processor_config.json, so a checkpoint with a root preprocessor_config.json AND a
    # processor/ directory had the ROOT file validated while upstream loaded the NESTED one.
    # The guard must resolve the same directory upstream will, not a directory of its own
    # choosing.
    nested = os.path.join(checkpoint_dir, "processor")
    root_processor_config = os.path.join(checkpoint_dir, "processor_config.json")
    if os.path.isdir(nested) and not os.path.exists(root_processor_config):
        resolved_dir = nested
    else:
        resolved_dir = checkpoint_dir
    config_path = None
    for name in ("processor_config.json", "preprocessor_config.json"):
        candidate = os.path.join(resolved_dir, name)
        if os.path.exists(candidate):
            config_path = candidate
            break
    if config_path is None:
        # Nothing saved in the directory UPSTREAM RESOLVES means its own defaults apply, which
        # are the trusted ones. Absence elsewhere in the tree is not equivalent.
        print(f"[seeded-server] checkpoint saves no processor config in {resolved_dir}; "
              f"upstream defaults apply", flush=True)
        return

    try:
        with open(config_path) as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        _fail(f"saved processor config {config_path} is unreadable ({exc}); refusing to "
              f"construct a policy from a configuration that cannot be inspected.")
    if not isinstance(config, dict):
        _fail(f"saved processor config {config_path} is not an object; refusing to interpret it.")

    processor_class = config.get("processor_class")
    if processor_class is not None and processor_class not in TRUSTED_PROCESSOR_CLASSES:
        _fail(f"saved processor names class {processor_class!r}, which this evaluator does not "
              f"trust. Allowed: {list(TRUSTED_PROCESSOR_CLASSES)}.")

    auto_map = config.get("auto_map")
    if auto_map:
        _fail(f"saved processor declares auto_map={auto_map!r}, which would import processor "
              f"code from the checkpoint. The processor class comes from the image.")

    kwargs = config.get("processor_kwargs")
    if kwargs is not None:
        if not isinstance(kwargs, dict):
            _fail(f"processor_kwargs in {config_path} is not an object; refusing to interpret it.")
        backbone = kwargs.get("model_name")
        if backbone is not None and backbone not in TRUSTED_VLM_BACKBONES:
            _fail(f"saved processor selects VLM backbone {backbone!r}, which is not in the "
                  f"trusted set {list(TRUSTED_VLM_BACKBONES)}. That value is passed to "
                  f"from_pretrained with trust_remote_code enabled, so a checkpoint choosing "
                  f"it selects code to execute with this job's identity.")
        loading = kwargs.get("transformers_loading_kwargs")
        if isinstance(loading, dict) and loading.get("trust_remote_code") is False:
            # Stricter than upstream's default; allowed.
            pass
        elif loading is not None and not isinstance(loading, dict):
            _fail(f"transformers_loading_kwargs in {config_path} is not an object.")
    print(f"[seeded-server] saved processor config validated: class={processor_class!r} "
          f"backbone={(kwargs or {}).get('model_name')!r}", flush=True)


def _write_evidence(evidence: dict) -> None:
    """Publish what was actually applied, for the run's recorded protocol evidence.

    A write failure is FATAL. This file is the only record that the seed was bound and the
    strict load audit ran, so a run whose evidence could not be written cannot substantiate
    either guarantee -- and it was previously a warning, meaning the guards could be installed
    while nothing provable came out of them.

    Written atomically so a partially written file can never be read as complete.
    """
    evidence["evidence_schema"] = EVIDENCE_SCHEMA
    # WHICH process wrote this. One line, and it is the part worth having: a reader comparing it
    # against the pid it launched can tell that the evidence came from its own server rather than
    # from a leftover file. The stronger listener-ownership check is deliberately not implemented
    # -- see verify_server_alive in the evaluator for why the deployment does not reach that state.
    evidence["server_pid"] = os.getpid()
    temp_path = f"{EVIDENCE_PATH}.partial"
    try:
        with open(temp_path, "w") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, EVIDENCE_PATH)
    except OSError as exc:
        _fail(f"could not write seed evidence to {EVIDENCE_PATH}: {exc}. That file is the "
              f"only record that the seed was bound and the weight-load audit ran, so the "
              f"run cannot substantiate either guarantee. Refusing to evaluate.")


def _model_path_from_argv(argv: list[str]) -> "str | None":
    """The checkpoint the server was told to serve.

    Read from the arguments rather than an environment variable, so it is the same value the
    server itself will load.
    """
    for index, token in enumerate(argv):
        if token == "--model-path" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--model-path="):
            return token.split("=", 1)[1]
    return None


def main(argv: list[str]) -> None:
    seed = _required_seed()
    _require_torch()
    evidence = {
        "seed": seed,
        "seed_source": "EVAL_SEED",
        "server_module": SERVER_MODULE,
        "reseeds_per_episode": False,
        "scope": "server_process_rng",
    }
    evidence["process_start"] = _seed_everything(seed, "process_start")
    evidence["determinism_policy"] = _apply_determinism_policy()
    # C3: installed BEFORE the policy is constructed, so the very first weight load is
    # audited. Constructing first and checking after would already have a model built
    # from randomly initialized tensors.
    # C1: validate the SAVED PROCESSOR before any policy exists. The weight-load audit
    # cannot cover this -- the tensors are genuine and the risk is which code the saved
    # processor selects for execution.
    checkpoint_dir = _model_path_from_argv(argv)
    if checkpoint_dir is None:
        _fail("no --model-path in the server arguments, so the saved processor cannot "
              "be validated. Refusing to construct a policy whose processor "
              "configuration was never inspected.")
    _validate_saved_processor(checkpoint_dir)
    evidence["processor_validated"] = checkpoint_dir
    _install_strict_load_audit(evidence)
    _install_inference_ready_reseed(seed, evidence)
    _write_evidence(evidence)

    # Delegate to the pinned server untouched: same module, same arguments.
    sys.argv = [SERVER_MODULE] + list(argv)
    print(f"[seeded-server] running {SERVER_MODULE} argv={sys.argv[1:]}", flush=True)
    runpy.run_module(SERVER_MODULE, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])


class BackboneAttributionError(RuntimeError):
    """The consumed backbone identity cannot be established from the server's own evidence."""


def consumed_backbone_identity(seed_scope, checkpoint_root: str, repo_id: str) -> dict:
    """The backbone commit INFERENCE loaded, taken from the server's audit -- not from the precache.

    This exists because the previous fix was incomplete in the way it was meant to prevent. The audit
    above records `consumed_commit`, the commit the serving process actually resolved, and NOTHING read
    it: all three report sites copied a module global populated by the PRECACHE step. So precache commit
    A with serving commit B published backbone_identity=A -- a false attribution, which is worse than an
    absent one, and exactly the "declared value that governs nothing" shape the audit was added to close.

    Refuses rather than guesses. An unattributable backbone must fail loudly: it determines the
    preprocessing, so it can change while the checkpoint, its digest, the image and the sourcedir all
    stay identical.
    """
    if not isinstance(seed_scope, dict):
        raise BackboneAttributionError(
            f"seed_scope is {type(seed_scope).__name__}, not a dict, so the server's evidence cannot be "
            f"read and the consumed backbone cannot be established")
    evidence = seed_scope.get("server_seeding_evidence")
    if not isinstance(evidence, dict):
        raise BackboneAttributionError(
            "the accepted seed scope carries no server_seeding_evidence, so nothing records which "
            "backbone this run loaded")

    validated = evidence.get("processor_validated")
    if not validated or os.path.realpath(str(validated)) != os.path.realpath(checkpoint_root):
        raise BackboneAttributionError(
            f"the evidence validates a processor at {validated!r} but this report describes "
            f"{checkpoint_root!r}. Evidence from a different launch cannot attribute this one -- the "
            f"defect that made a dose point read another point's proof.")

    audits = evidence.get("strict_load_audits") or []
    commits = {a.get("consumed_commit") for a in audits
               if isinstance(a, dict) and str(a.get("path")) == repo_id}
    commits.discard(None)
    if not commits:
        raise BackboneAttributionError(
            f"no strict-load audit records a hub load of {repo_id!r}, so the commit inference consumed "
            f"is unknown. Paths audited: {sorted(str(a.get('path')) for a in audits if isinstance(a, dict))}")
    if len(commits) > 1:
        raise BackboneAttributionError(
            f"{repo_id!r} was loaded at MORE THAN ONE commit in a single run ({sorted(commits)}); the "
            f"report cannot name one of them as the backbone that produced the result")

    commit = commits.pop()
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise BackboneAttributionError(
            f"the recorded consumed commit for {repo_id!r} is {commit!r}, not an immutable 40-hex "
            f"commit, so it pins nothing")
    return {"repo_id": repo_id, "resolved_commit": commit}
