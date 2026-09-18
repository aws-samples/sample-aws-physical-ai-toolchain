"""I11: the evaluation seed must bind the SERVER process's RNG.

EVAL_SEED reached the server process's environment, but neither pinned server read it and
neither ServerConfig exposed a seed argument. Arena's runner seeds the CLIENT process and
environment, which covers scene and initial-state randomness -- while both action heads
sample inference noise through torch.randn in the SERVER process. A rerun with the same
seed therefore reproduced the same scene sequence while the policy sampled different
action noise, and the reported seed_scope overstated what was reproducible.

Passing a seed through the reset endpoint would not have worked either: both pinned
Gr00tPolicy.reset() implementations return {}. N1.7 ships a seed_everything() that
optionally reads GR00T_EVAL_SEED, but the server path never calls it, so renaming the
variable would not have been enough.

The wrapper's seeding logic is exercised here directly. What needs the container is only
whether the pinned server layout matches the constructor it wraps; that check is a hard
failure at startup rather than a silent skip.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import sys
import types

import numpy as np
import pytest

_WRAPPER = (pathlib.Path(__file__).resolve().parents[1]
            / "entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py")


_WRAPPER_PATH = (pathlib.Path(__file__).resolve().parents[1]
                 / "entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py")


def _wrapper():
    spec = importlib.util.spec_from_file_location("gr00t_seeded_server_test", _WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ARENA_ENTRY_PATH = (pathlib.Path(__file__).resolve().parents[1]
                     / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py")


def _arena_entry():
    """Load the Arena eval entry to reach its module-level helpers."""
    import os as _os
    _os.environ.setdefault("SM_HP_TASK_NAME", "fixture_task")
    entry = (pathlib.Path(__file__).resolve().parents[1]
             / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py")
    spec = importlib.util.spec_from_file_location("arena_seed_scope_test", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_policy_config_digest_covers_the_file_content(monkeypatch, tmp_path):
    """I5: recording the PATH proved which file was selected, not what it contained.

    The same path can carry a different protocol (chunk length, joint ordering, camera
    handling) between runs, so a comparison over paths and labels cannot detect that the
    evaluation ran a different configuration than the one requested.
    """
    import hashlib

    entry = _arena_entry()
    config = tmp_path / "policy.yaml"
    config.write_text("chunk_length: 16\n")
    monkeypatch.setattr(entry, "_DECLARED_POLICY_CONFIG", str(config))
    expected = "sha256:" + hashlib.sha256(config.read_bytes()).hexdigest()
    assert entry._policy_config_digest() == expected
    # Changing the CONTENT at the same path must change the digest.
    config.write_text("chunk_length: 8\n")
    assert entry._policy_config_digest() != expected


def test_policy_config_digest_is_null_when_unreadable(monkeypatch, tmp_path):
    """An unreadable config yields no digest rather than a value that looks verified."""
    entry = _arena_entry()
    monkeypatch.setattr(entry, "_DECLARED_POLICY_CONFIG", str(tmp_path / "absent.yaml"))
    assert entry._policy_config_digest() is None


def test_policy_config_digest_is_null_when_undeclared(monkeypatch):
    entry = _arena_entry()
    monkeypatch.setattr(entry, "_DECLARED_POLICY_CONFIG", "")
    assert entry._policy_config_digest() is None


def test_the_report_records_an_unverified_policy_rng_when_evidence_is_absent(
    monkeypatch, tmp_path
):
    """With no server evidence, the report must NOT imply the policy RNG was bound.

    Arena's runner seeds the client and environment; the policy server is a separate
    process whose inference noise was unbound. Claiming a scope this process cannot
    observe is the overstatement the finding is about.
    """
    entry = _arena_entry()
    monkeypatch.setenv("GR00T_SERVER_SEED_EVIDENCE", str(tmp_path / "absent.json"))
    scope = entry._seed_scope_for_report()
    assert scope["client_and_env"] is True
    assert scope["server_policy_rng"] is False
    assert scope["server_seeding_evidence"] is None
    assert scope["per_episode_replayable"] is False


def _complete_evidence(seed=100):
    """Evidence in the shape the wrapper actually writes.

    The previous fixture carried only four descriptive fields and no schema, stages or audit
    records -- exactly the "any nonempty JSON object counts as evidence" case, so it proved
    the reader accepted something rather than that it accepted PROOF.
    """
    return {
        "evidence_schema": "gr00t_server_evidence_v1",
        "seed": seed,
        "seed_source": "EVAL_SEED",
        "reseeds_per_episode": False,
        "scope": "server_process_rng",
        "process_start": {"applied": ["random", "numpy", "torch"]},
        "inference_ready": {"applied": ["random", "numpy", "torch"]},
        "strict_load_audit": "installed",
        "strict_load_audits": [{"path": "/ckpt", "missing_keys": 0, "unexpected_keys": 0,
                                "mismatched_keys": 0, "error_msgs": 0}],
    }


def test_the_report_records_the_server_seeding_when_evidence_is_present(
    monkeypatch, tmp_path
):
    entry = _arena_entry()
    evidence = _complete_evidence()
    path = tmp_path / "seed_evidence.json"
    path.write_text(json.dumps(evidence))
    monkeypatch.setenv("GR00T_SERVER_SEED_EVIDENCE", str(path))
    scope = entry._seed_scope_for_report()
    assert scope["server_policy_rng"] is True
    assert scope["server_seeding_evidence"] == evidence
    assert scope["server_policy_rng_unverified_reason"] is None
    # Even with the server seeded, episodes are not independently replayable.
    assert scope["per_episode_replayable"] is False


@pytest.mark.parametrize("mutate,expected", [
    (lambda e: {**e, "evidence_schema": "something_else"}, "schema is"),
    (lambda e: {k: v for k, v in e.items() if k != "evidence_schema"}, "schema is"),
    (lambda e: {**e, "seed": 999}, "records seed"),
    (lambda e: {k: v for k, v in e.items() if k != "process_start"}, "process_start"),
    (lambda e: {k: v for k, v in e.items() if k != "inference_ready"}, "inference_ready"),
    (lambda e: {**e, "strict_load_audit": None}, "audit was not installed"),
    (lambda e: {**e, "strict_load_audits": []}, "no completed weight-load audit"),
])
def test_incomplete_evidence_is_not_accepted_as_proof(monkeypatch, tmp_path,
                                                      mutate, expected):
    """I10: the reader required none of the things the evidence exists to certify.

    A stale file from an earlier launch, or one recording a different seed, was
    indistinguishable from proof that this run's seed was bound and its weights audited.
    """
    entry = _arena_entry()
    path = tmp_path / "seed_evidence.json"
    path.write_text(json.dumps(mutate(_complete_evidence())))
    monkeypatch.setenv("GR00T_SERVER_SEED_EVIDENCE", str(path))
    scope = entry._seed_scope_for_report()
    assert scope["server_policy_rng"] is False
    assert scope["server_seeding_evidence"] is None
    assert expected in scope["server_policy_rng_unverified_reason"]


def test_a_non_object_evidence_file_is_rejected(monkeypatch, tmp_path):
    entry = _arena_entry()
    path = tmp_path / "seed_evidence.json"
    path.write_text(json.dumps(["not", "an", "object"]))
    monkeypatch.setenv("GR00T_SERVER_SEED_EVIDENCE", str(path))
    scope = entry._seed_scope_for_report()
    assert scope["server_policy_rng"] is False
    assert "not an object" in scope["server_policy_rng_unverified_reason"]


def test_an_evidence_write_failure_is_fatal():
    """The evidence is the only record the guards ran, so a warning was not enough."""
    source = _WRAPPER_PATH.read_text()
    assert "could not write seed evidence" in source
    write = source.index("def _write_evidence")
    body = source[write:write + 1400]
    assert "_fail(" in body, "a write failure must be fatal, not a warning"
    assert "WARNING" not in body
    assert "os.replace(" in body, "evidence must be published atomically"


def test_a_missing_seed_refuses_to_start(monkeypatch):
    """An unseeded policy server is the defect; it must not be a silent default."""
    mod = _wrapper()
    monkeypatch.delenv("EVAL_SEED", raising=False)
    with pytest.raises(SystemExit):
        mod._required_seed()


@pytest.mark.parametrize("bad", ["", "   ", "abc", "1.5", "-1", str(2 ** 32)])
def test_an_unusable_seed_refuses_to_start(monkeypatch, bad):
    """Out-of-range values would reach different generators as different numbers."""
    mod = _wrapper()
    monkeypatch.setenv("EVAL_SEED", bad)
    with pytest.raises(SystemExit):
        mod._required_seed()


@pytest.mark.parametrize("value", ["0", "100", "1000", str(2 ** 32 - 1)])
def test_a_usable_seed_is_accepted(monkeypatch, value):
    mod = _wrapper()
    monkeypatch.setenv("EVAL_SEED", value)
    assert mod._required_seed() == int(value)


def test_seeding_makes_the_python_and_numpy_streams_reproducible():
    """The actual property being bought: same seed, same stream."""
    mod = _wrapper()
    mod._seed_everything(1234, "test")
    first = (random.random(), np.random.rand())
    mod._seed_everything(1234, "test")
    second = (random.random(), np.random.rand())
    assert first == second


def test_different_seeds_give_different_streams():
    """Control: otherwise the test above would pass with seeding broken."""
    mod = _wrapper()
    mod._seed_everything(1234, "test")
    first = (random.random(), np.random.rand())
    mod._seed_everything(4321, "test")
    assert (random.random(), np.random.rand()) != first


def test_seeding_reports_what_it_applied():
    mod = _wrapper()
    applied = mod._seed_everything(7, "process_start")
    assert applied["seed"] == 7
    assert applied["phase"] == "process_start"
    # Host RNGs are seeded unconditionally.
    assert applied["python_random"] is True
    assert applied["numpy"] is True
    # Torch is recorded either way rather than assumed. It is absent from this offline
    # venv; the server refuses to start without it (test_the_server_requires_torch).
    try:
        import torch  # noqa: F401
    except ImportError:
        assert applied["torch"] == "unavailable"
        assert applied["torch_cpu"] is False
    else:
        assert applied["torch_cpu"] is True
    assert isinstance(applied["torch_cuda_devices"], int)


def test_the_server_requires_torch(monkeypatch):
    """torch's absence must stop the run, not be recorded and ignored.

    The action head samples inference noise through torch.randn, so a server without
    torch cannot bind the policy RNG and cannot serve.
    """
    mod = _wrapper()
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_torch)
    with pytest.raises(SystemExit):
        mod._require_torch()


def test_the_determinism_policy_is_recorded_not_assumed():
    """Full deterministic-algorithm enforcement is NOT applied; the record must say so."""
    mod = _wrapper()
    policy = mod._apply_determinism_policy()
    assert policy["torch_deterministic_algorithms"] is False
    assert "cudnn_deterministic" in policy
    assert "cudnn_benchmark" in policy


def test_the_evidence_records_the_scope_and_reset_behaviour(monkeypatch, tmp_path):
    """seed_scope must not overstate: episode resets do not reseed."""
    mod = _wrapper()
    evidence_path = tmp_path / "seed_evidence.json"
    monkeypatch.setattr(mod, "EVIDENCE_PATH", str(evidence_path))
    mod._write_evidence({
        "seed": 100, "seed_source": "EVAL_SEED",
        "reseeds_per_episode": False, "scope": "server_process_rng",
    })
    written = json.loads(evidence_path.read_text())
    assert written["reseeds_per_episode"] is False
    assert written["scope"] == "server_process_rng"
    assert written["seed_source"] == "EVAL_SEED"


def test_a_changed_server_layout_fails_loudly(monkeypatch):
    """If the policy class cannot be found, the RNG cannot be bound -- so stop.

    Proceeding would run with an unbound policy RNG while the report claimed a seed.
    """
    mod = _wrapper()
    monkeypatch.setitem(sys.modules, "gr00t", None)
    with pytest.raises(SystemExit):
        mod._install_inference_ready_reseed(100, {})


def test_the_reseed_happens_after_construction_not_before():
    """Model construction consumes checkpoint-dependent RNG.

    Seeding only at process start would leave different dose checkpoints beginning
    inference from different stream positions, so the reseed must wrap the constructor's
    RETURN, not precede it.
    """
    mod = _wrapper()
    order = []

    class FakePolicy:
        def __init__(self):
            order.append("construct")

    fake_module = type(sys)("gr00t.policy.gr00t_policy")
    fake_module.Gr00tPolicy = FakePolicy
    package = type(sys)("gr00t.policy")
    package.gr00t_policy = fake_module
    root = type(sys)("gr00t")
    root.policy = package
    sys.modules.update({"gr00t": root, "gr00t.policy": package,
                        "gr00t.policy.gr00t_policy": fake_module})
    try:
        evidence: dict = {}
        mod._seed_everything = lambda seed, phase: order.append(f"seed:{phase}") or {}
        mod._write_evidence = lambda _evidence: None
        mod._install_inference_ready_reseed(100, evidence)
        FakePolicy()
        assert order == ["construct", "seed:inference_ready"], order
    finally:
        for name in ("gr00t.policy.gr00t_policy", "gr00t.policy", "gr00t"):
            sys.modules.pop(name, None)


def _fake_transformers(info):
    """Stands in for transformers, asserting the audit requests diagnostics."""
    module = types.ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            assert kwargs.get("output_loading_info") is True, (
                "the audit must REQUEST loading diagnostics; upstream does not")
            return ("MODEL", info)

    module.AutoModel = AutoModel
    return module


def _loading_info(**overrides):
    info = {"missing_keys": [], "unexpected_keys": [],
            "mismatched_keys": [], "error_msgs": []}
    info.update(overrides)
    return info


def test_a_clean_load_passes_and_returns_only_the_model(monkeypatch):
    """The audit must be transparent to upstream, which expects a bare model."""
    module = _wrapper()
    fake = _fake_transformers(_loading_info())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    evidence = {}
    module._install_strict_load_audit(evidence)
    assert fake.AutoModel.from_pretrained("/ckpt") == "MODEL"
    assert evidence["strict_load_audit"] == "installed"
    assert evidence["strict_load_audits"][0]["missing_keys"] == 0


@pytest.mark.parametrize("problem,value", [
    ("missing_keys", ["backbone.layer.0.weight"]),
    ("unexpected_keys", ["stale.weight"]),
    ("mismatched_keys", [("w", (1,), (2,))]),
    ("error_msgs", ["size mismatch"]),
])
def test_an_incomplete_load_is_refused(monkeypatch, problem, value):
    """C3: Transformers initializes missing tensors, warns, and returns the model.

    A digest proves an incomplete checkpoint transferred intact; it says nothing about
    whether the running model consisted entirely of checkpoint tensors. Such a run was still
    labelled policy_type=checkpoint.
    """
    module = _wrapper()
    fake = _fake_transformers(_loading_info(**{problem: value}))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    module._install_strict_load_audit({})
    with pytest.raises(SystemExit):
        fake.AutoModel.from_pretrained("/ckpt")


def test_a_caller_that_asks_for_the_info_still_gets_it(monkeypatch):
    """The wrapper must not break the documented from_pretrained contract."""
    module = _wrapper()
    fake = _fake_transformers(_loading_info())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    module._install_strict_load_audit({})
    model, info = fake.AutoModel.from_pretrained("/ckpt", output_loading_info=True)
    assert model == "MODEL" and info["missing_keys"] == []


def test_the_audit_is_installed_before_the_policy_is_constructed():
    """Checking after construction would already have a partly random model built."""
    source = _WRAPPER_PATH.read_text()
    assert source.index("_install_strict_load_audit(evidence)") < source.index(
        "_install_inference_ready_reseed(seed, evidence)")


def test_the_wrapper_is_delivered_into_the_image():
    """The build uses an explicit allowlist; an omitted file silently never ships."""
    allowlist = (pathlib.Path(__file__).resolve().parents[1]
                 / "scripts/build_arena_connector.py").read_text()
    assert "gr00t_seeded_server.py" in allowlist


def test_each_server_launch_gets_its_own_evidence_path():
    """I10: one reusable /tmp filename could be inherited across launches in a job.

    The dose-curve path starts a server per checkpoint, so the LAST server's evidence would
    otherwise be read as every checkpoint's.
    """
    entry = _arena_entry()
    first = entry._server_evidence_path("/ckpt/checkpoint-100", 5555)
    second = entry._server_evidence_path("/ckpt/checkpoint-200", 5555)
    assert first != second


def test_the_evidence_path_is_stable_for_the_same_launch():
    """The reader recomputes it, so it must be deterministic rather than random."""
    entry = _arena_entry()
    assert (entry._server_evidence_path("/ckpt/a", 5555)
            == entry._server_evidence_path("/ckpt/a", 5555))


def test_a_different_port_is_a_different_launch():
    entry = _arena_entry()
    assert (entry._server_evidence_path("/ckpt/a", 5555)
            != entry._server_evidence_path("/ckpt/a", 5556))


def test_the_launchers_clear_stale_evidence_before_starting():
    """A launch that writes nothing must not inherit a previous server's proof."""
    source = _ARENA_ENTRY_PATH.read_text()
    assert source.count('server_env["GR00T_SERVER_SEED_EVIDENCE"] = _evidence_path') == 2, (
        "both the n17 and n16 launchers must set a per-launch evidence path")
    assert source.count("os.remove(_evidence_path)") == 2
    # And the removal must precede the launch, not follow it.
    first_remove = source.index("os.remove(_evidence_path)")
    assert first_remove < source.index("subprocess.Popen(")


def test_the_report_reads_the_evidence_for_the_checkpoint_it_reports():
    source = _ARENA_ENTRY_PATH.read_text()
    assert "_seed_scope_for_report(checkpoint_root, SERVER_PORT)" in source


@pytest.mark.parametrize("absent", [
    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_absent_diagnostics_are_not_a_clean_load(monkeypatch, absent):
    """I7: an absent diagnostic field was read as empty, so a load reporting NO diagnostics
    passed as clean. That is exactly what a changed upstream structure looks like."""
    module = _wrapper()
    info = _loading_info()
    del info[absent]
    fake = _fake_transformers(info)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    module._install_strict_load_audit({})
    with pytest.raises(SystemExit):
        fake.AutoModel.from_pretrained("/ckpt")


def test_a_none_valued_diagnostic_is_also_incomplete(monkeypatch):
    module = _wrapper()
    fake = _fake_transformers(_loading_info(missing_keys=None))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    module._install_strict_load_audit({})
    with pytest.raises(SystemExit):
        fake.AutoModel.from_pretrained("/ckpt")


def test_the_libero_report_carries_the_structured_seed_evidence():
    """I7: LIBERO reported the bare string "client_and_env", which could not be checked."""
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "entrypoints/eval/libero/gr00t/eval_entry.py").read_text()
    # Was: assert the literal '"seed_scope": _seed_scope_for_report()'. The call is now HOISTED
    # above the report so one accepted scope both populates seed_scope and establishes which
    # backbone commit inference consumed -- reading it twice risked two answers in one report.
    # Bind the property: the scope is computed and reaches the report.
    assert "_seed_scope_for_report()" in source, (
        "the report never computes a seed scope, so a run whose policy RNG was not\n         demonstrably bound would register indistinguishably from one where it was")
    assert '"seed_scope"' in source, "the report omits seed_scope entirely"
    assert '"seed_scope": "client_and_env"' not in source


def test_the_libero_reader_validates_the_same_way_arena_does():
    """Two readers of one evidence format must not disagree about what counts as proof."""
    libero = (pathlib.Path(__file__).resolve().parents[1]
              / "entrypoints/eval/libero/gr00t/eval_entry.py").read_text()
    for requirement in ("gr00t_server_evidence_v1", "inference_ready",
                        "strict_load_audit", "strict_load_audits",
                        "server_policy_rng_unverified_reason"):
        assert requirement in libero, f"the LIBERO reader does not check {requirement}"


def test_the_reporter_uses_the_path_the_launch_recorded(monkeypatch, tmp_path):
    """Cycle-6 I3: dose points run on SERVER_PORT + idx.

    Recomputing the evidence path in write_metrics from SERVER_PORT looked up a file belonging
    to a different launch, or none at all -- so a multi-point dose run reported the canonical
    checkpoint's seeding using the wrong port's evidence.
    """
    entry = _arena_entry()
    monkeypatch.delenv("GR00T_SERVER_SEED_EVIDENCE", raising=False)
    checkpoint = tmp_path / "checkpoint-200"
    checkpoint.mkdir()
    # A launch on a NON-default port records where it wrote.
    dose_path = tmp_path / "dose_evidence.json"
    dose_path.write_text(json.dumps(_complete_evidence(seed=100)))
    entry._LAUNCH_EVIDENCE[str(checkpoint.resolve())] = str(dose_path)

    scope = entry._seed_scope_for_report(str(checkpoint), entry.SERVER_PORT)
    assert scope["server_policy_rng"] is True, (
        "the recorded launch path must be used, not one derived from the default port")


def test_an_unrecorded_checkpoint_does_not_borrow_another_launchs_evidence(
        monkeypatch, tmp_path):
    """Falling back to a shared default is how one launch's proof became another's."""
    entry = _arena_entry()
    monkeypatch.delenv("GR00T_SERVER_SEED_EVIDENCE", raising=False)
    entry._LAUNCH_EVIDENCE.clear()
    other = tmp_path / "other_evidence.json"
    other.write_text(json.dumps(_complete_evidence()))
    entry._LAUNCH_EVIDENCE[str((tmp_path / "checkpoint-100").resolve())] = str(other)

    unrecorded = tmp_path / "checkpoint-999"
    unrecorded.mkdir()
    scope = entry._seed_scope_for_report(str(unrecorded), entry.SERVER_PORT)
    assert scope["server_policy_rng"] is False
    assert scope["server_seeding_evidence"] is None


def test_both_launchers_record_the_evidence_path():
    source = _ARENA_ENTRY_PATH.read_text()
    assert source.count("_LAUNCH_EVIDENCE[os.path.realpath(checkpoint_path)]") == 2, (
        "both the n17 and n16 launchers must record where they wrote")


# --- Cycle-6 C1: checkpoint-selected processor code in the GR00T server ------------------

_REAL_GR00T_PROCESSOR = {
    "processor_class": "Gr00tN1d7Processor",
    "processor_kwargs": {
        "model_name": "nvidia/Cosmos-Reason2-2B",
        "image_crop_size": 224,
        "transformers_loading_kwargs": {"trust_remote_code": True},
    },
}


def _processor_checkpoint(tmp_path, config=None, name="processor_config.json"):
    directory = tmp_path / "ckpt"
    directory.mkdir(exist_ok=True)
    if config is not None:
        (directory / name).write_text(json.dumps(config))
    return str(directory)


def test_a_real_saved_processor_is_accepted(tmp_path):
    """Shaped from the pinned upstream save path, including its default backbone.

    Rejecting this would fail the component's own training output, which is the mistake the
    OpenVLA guard made before being corrected.
    """
    _wrapper()._validate_saved_processor(
        _processor_checkpoint(tmp_path, _REAL_GR00T_PROCESSOR))


def test_a_checkpoint_with_no_saved_processor_is_accepted(tmp_path):
    """Absence means upstream applies its own defaults, which are the trusted ones."""
    _wrapper()._validate_saved_processor(_processor_checkpoint(tmp_path))


@pytest.mark.parametrize("backbone", [
    "/opt/ml/input/data/model/evil", "attacker/vlm", "./local-dir"])
def test_a_checkpoint_selected_backbone_is_refused(tmp_path, backbone):
    """C1: model_name reaches Qwen3VLProcessor.from_pretrained with trust_remote_code enabled.

    A correct weights digest and a clean weight-load audit say nothing about this: the tensors
    are genuine and the executed code came from wherever this value points.
    """
    config = {**_REAL_GR00T_PROCESSOR,
              "processor_kwargs": {**_REAL_GR00T_PROCESSOR["processor_kwargs"],
                                   "model_name": backbone}}
    with pytest.raises(SystemExit):
        _wrapper()._validate_saved_processor(_processor_checkpoint(tmp_path, config))


def test_an_untrusted_processor_class_is_refused(tmp_path):
    config = {**_REAL_GR00T_PROCESSOR, "processor_class": "EvilProcessor"}
    with pytest.raises(SystemExit):
        _wrapper()._validate_saved_processor(_processor_checkpoint(tmp_path, config))


def test_an_auto_map_in_the_saved_processor_is_refused(tmp_path):
    config = {**_REAL_GR00T_PROCESSOR, "auto_map": {"AutoProcessor": "evil.Thing"}}
    with pytest.raises(SystemExit):
        _wrapper()._validate_saved_processor(_processor_checkpoint(tmp_path, config))


def test_an_unreadable_saved_processor_is_refused(tmp_path):
    directory = tmp_path / "ckpt"
    directory.mkdir()
    (directory / "processor_config.json").write_text("{not json")
    with pytest.raises(SystemExit):
        _wrapper()._validate_saved_processor(str(directory))


def test_the_guard_runs_before_the_policy_is_constructed():
    """Validating after construction would be too late: the code has already run."""
    source = _WRAPPER_PATH.read_text()
    assert source.index("_validate_saved_processor(checkpoint_dir)") < source.index(
        "_install_inference_ready_reseed(seed, evidence)")


def test_a_missing_model_path_is_fatal():
    """Without it the processor cannot be inspected, so the policy must not be built."""
    source = _WRAPPER_PATH.read_text()
    assert "no --model-path in the server arguments" in source


@pytest.mark.parametrize("argv,expected", [
    (["--model-path", "/ckpt", "--port", "5555"], "/ckpt"),
    (["--model-path=/ckpt2"], "/ckpt2"),
    (["--port", "5555"], None),
])
def test_the_checkpoint_is_read_from_the_server_arguments(argv, expected):
    """The same value the server itself will load, not a separate environment variable."""
    assert _wrapper()._model_path_from_argv(argv) == expected


# --- C1: the guard must resolve the directory UPSTREAM loads -------------------------------

_PINNED_RESOLVER = (pathlib.Path(__file__).parent
                    / "data/gr00t_policy_processor_dir_pinned.py.txt")


def _probe(tmp_path, layout):
    """Run the committed guard against a checkpoint layout; True if accepted."""
    for rel, body in layout.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body))
    module = _wrapper()
    try:
        module._validate_saved_processor(str(tmp_path))
        return True
    except SystemExit:
        return False


_HOSTILE = {"processor_kwargs": {"model_name": "/untrusted/local-backbone"}}


def test_a_nested_processor_directory_cannot_bypass_the_guard(tmp_path):
    """C1: the guard searched only root-level names and returned clean on absence.

    Upstream falls back to processor/ when the root lacks processor_config.json, so a
    checkpoint carrying only a nested processor config passed the guard entirely and then had
    that processor loaded -- selecting a local backbone that reaches from_pretrained with
    trust_remote_code enabled.
    """
    assert _probe(tmp_path, {"processor/processor_config.json": _HOSTILE}) is False


def test_a_root_preprocessor_does_not_shadow_the_nested_config(tmp_path):
    """Upstream's fallback keys ONLY on processor_config.json.

    So a root preprocessor_config.json plus a processor/ directory had the ROOT file validated
    while upstream loaded the NESTED one -- the guard inspected a file upstream never reads.
    """
    assert _probe(tmp_path, {
        "preprocessor_config.json": {"processor_class": None},
        "processor/processor_config.json": _HOSTILE,
    }) is False


def test_legitimate_layouts_are_still_accepted(tmp_path):
    """Constraining the mechanism must not reject the component's own trainer output."""
    trusted = {"processor_kwargs": {"model_name": "nvidia/Cosmos-Reason2-2B"}}
    assert _probe(tmp_path / "a", {"processor_config.json": trusted}) is True
    assert _probe(tmp_path / "b", {"processor/processor_config.json": trusted}) is True
    assert _probe(tmp_path / "c", {"config.json": {}}) is True


def test_the_guard_reproduces_the_pinned_upstream_resolution():
    """Pinned upstream is vendored so a version bump that changes the rule breaks this."""
    pinned = _PINNED_RESOLVER.read_text()
    for token in ('model_dir / "processor"', '(model_dir / "processor").is_dir()',
                  'not (model_dir / "processor_config.json").exists()'):
        assert token in pinned, f"vendored resolver no longer contains {token}"
    source = _WRAPPER.read_text()
    # The guard must key on the same two conditions, in the same combination.
    assert 'os.path.isdir(nested) and not os.path.exists(root_processor_config)' in source


# --- C5: both supported GR00T versions are the component's own output ----------------------

def _probe_root(tmp_path, config):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "processor_config.json").write_text(json.dumps(config))
    module = _wrapper()
    try:
        module._validate_saved_processor(str(tmp_path))
        return True
    except SystemExit:
        return False


def test_the_n16_save_shape_is_accepted(tmp_path):
    """C5: the allowlists carried only N1.7, so the guard rejected valid N1.6 checkpoints.

    Native N1.6 Arena runs through this same wrapper and train_entry copies the saved processor
    files into the model artifact, so this rejected the component's OWN trainer output.
    Verified at N1.6's pinned commit 5dc80c4a, where processing_gr00t_n1d6.py:124 defaults
    model_name to nvidia/Eagle-Block2A-2B-v2 and :108 defines Gr00tN1d6Processor.
    """
    assert _probe_root(tmp_path, {
        "processor_class": "Gr00tN1d6Processor",
        "processor_kwargs": {"model_name": "nvidia/Eagle-Block2A-2B-v2"},
    }) is True


def test_the_n17_save_shape_is_still_accepted(tmp_path):
    assert _probe_root(tmp_path, {
        "processor_class": "Gr00tN1d7Processor",
        "processor_kwargs": {"model_name": "nvidia/Cosmos-Reason2-2B"},
    }) is True


def test_widening_for_n16_did_not_admit_checkpoint_chosen_code(tmp_path):
    """Accepting a second upstream default must not accept a checkpoint's own selection."""
    assert _probe_root(tmp_path / "a", {
        "processor_class": "Gr00tN1d6Processor",
        "processor_kwargs": {"model_name": "/untrusted/backbone"}}) is False
    assert _probe_root(tmp_path / "b", {"processor_class": "EvilProcessor"}) is False
    assert _probe_root(tmp_path / "c", {
        "processor_class": "Gr00tN1d6Processor",
        "auto_map": {"AutoProcessor": "evil.Proc"}}) is False


def test_both_versions_processor_classes_are_trusted():
    module = _wrapper()
    for name in ("Gr00tN1d7Processor", "Gr00tN1d7DataCollator",
                 "Gr00tN1d6Processor", "Gr00tN1d6DataCollator"):
        assert name in module.TRUSTED_PROCESSOR_CLASSES
