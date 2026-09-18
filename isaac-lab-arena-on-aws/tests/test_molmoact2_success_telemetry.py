"""I2: a MolmoAct2 rollout that never observed success telemetry must not score zero.

Pinned upstream `rollout` reads the success signal from two supported sources and otherwise
falls back to `[False] * env.num_envs`. A rollout where neither field ever appeared produced
internally consistent zero-success metrics with no observed success signal, and at the
deliberately supported zero threshold those can be accepted. "Measured failure" and
"measurement unavailable" were not distinguishable.

The else-branch is deliberately NOT patched to raise: upstream's own comment says
`final_info` is absent when no env has finished yet, so the fallback is the normal path for
early steps. The same mistake was made in the GR00T patch (I3) and caught by review. The
decision belongs at the rollout boundary.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types
from unittest.mock import MagicMock

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _REPO / "entrypoints/eval/libero/molmoact2/eval_entry.py"
# Verbatim from allenai/lerobot at the pinned commit
# (a4f15bf347dee7eb8a8c5f4a70a37f476b091113), src/lerobot/scripts/lerobot_eval.py.
# Production refuses to run if its anchors are absent, so this copy going stale cannot
# silently disable the guarantee.
_PINNED = pathlib.Path(__file__).with_name("data") / "lerobot_eval_pinned.py"


def _module():
    """Load the entrypoint from a staging dir with defaults.json as its sibling.

    The image copies both into one directory; the repo keeps them apart.
    """
    for name in ("digest", "validator"):
        sys.modules.setdefault(name, MagicMock())
    os.environ.update({
        "EVAL_CHECKPOINT": "/opt/ml/input/data/model",
        "EVAL_SUITE": "libero_spatial", "EVAL_SEED": "1000",
        "EVAL_TRIALS": "3", "EVAL_TASK_IDS": "all",
        "EVAL_MODEL_SOURCE_URI": "s3://bucket/key/model.tar.gz",
    })
    staged = pathlib.Path(tempfile.mkdtemp(prefix="molmo-telemetry-"))
    shutil.copy(_ENTRY, staged / "eval_entry.py")
    shutil.copy(_REPO / "entrypoints/train/molmoact2/defaults.json",
                staged / "defaults.json")
    spec = importlib.util.spec_from_file_location(
        "molmo_telemetry_under_test", staged / "eval_entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patched_source():
    return _module()._patch_success_telemetry(_PINNED.read_text())


def _extract(patched: str, start: str, end: str, dedent: int) -> str:
    """Pull an executable block out of the patched source, dedented to module level."""
    begin = patched.index(start)
    finish = patched.index(end, begin) + len(end)
    block = patched[begin:finish]
    return "\n".join(line[dedent:] if line.startswith(" " * dedent) else line
                     for line in block.splitlines())


def _run_source_selection(info: dict) -> bool:
    """Execute the PATCHED source-selection chain and report whether telemetry was seen.

    This runs the real emitted code rather than inspecting it as text.
    """
    block = _extract(
        _patched_source(),
        '        if "final_info" in info:\n',
        "            successes = [False] * env.num_envs\n",
        8)
    namespace = {
        "info": info,
        "env": types.SimpleNamespace(num_envs=1),
        "_telemetry_observed": False,
    }
    exec(compile(block, "<selection>", "exec"), namespace)
    return namespace["_telemetry_observed"], namespace.get("successes")


def _run_boundary(telemetry_observed: bool) -> None:
    """Execute the PATCHED boundary check. Raises if it rejects."""
    block = _extract(
        _patched_source(),
        "    if not _telemetry_observed:\n",
        "episodes.')\n",
        4)
    exec(compile(block, "<boundary>", "exec"),
         {"_telemetry_observed": telemetry_observed})


class _Vector:
    """Stands in for the tensor upstream calls .tolist() on."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


def test_terminal_telemetry_counts_as_observed():
    observed, successes = _run_source_selection(
        {"final_info": {"is_success": _Vector([True])}})
    assert observed is True
    assert successes == [True]


def test_per_step_telemetry_counts_as_observed():
    observed, successes = _run_source_selection({"is_success": _Vector([False])})
    assert observed is True
    assert successes == [False]


def test_a_step_with_no_telemetry_is_not_an_observation_and_does_not_raise():
    """The fallback is the NORMAL path before any env finishes.

    Raising here would break every rollout on its first step -- the mistake the GR00T patch
    made (I3). It must yield the upstream default and leave the flag alone.
    """
    observed, successes = _run_source_selection({})
    assert observed is False
    assert successes == [False]


def test_the_boundary_rejects_a_rollout_that_never_observed_telemetry():
    """The load-bearing behaviour. If the raise is removed, THIS fails."""
    with pytest.raises(RuntimeError, match="no success telemetry was observed"):
        _run_boundary(False)


def test_the_boundary_accepts_a_rollout_that_did_observe_telemetry():
    """A legitimate all-false outcome must still be reportable."""
    _run_boundary(True)


def test_a_measured_all_false_rollout_is_accepted_end_to_end():
    """Measured failure and measurement unavailable must be distinguishable."""
    observed, successes = _run_source_selection(
        {"final_info": {"is_success": _Vector([False])}})
    assert observed is True and successes == [False]
    _run_boundary(observed)


def test_disabling_the_rejection_makes_these_tests_fail():
    """N2: the previous tests inspected patched text, so they passed with the rejection
    disabled. This asserts the suite is actually sensitive to that mutation.
    """
    module = _module()
    patched = module._patch_success_telemetry(_PINNED.read_text())
    disabled = patched.replace("        raise RuntimeError(\n", "        _ = RuntimeError(\n")
    assert disabled != patched, "the mutation must apply"
    block = _extract(disabled, "    if not _telemetry_observed:\n", "episodes.')\n", 4)
    namespace = {"_telemetry_observed": False}
    exec(compile(block, "<mutated>", "exec"), namespace)
    # With the raise removed nothing is raised -- which is precisely why the behavioural
    # test above, not a text assertion, is what protects this property.
    assert namespace["_"].args


def test_the_patched_source_is_valid_python():
    ast.parse(_patched_source())


def test_the_observation_flag_covers_both_supported_sources():
    """Upstream accepts terminal AND per-step telemetry; both must count as observed."""
    patched = _patched_source()
    assert "_telemetry_observed = False" in patched
    assert patched.count("_telemetry_observed = True") == 2


def test_the_fallback_branch_is_left_intact():
    """It is the NORMAL path before any env finishes -- raising there breaks every rollout.

    This is the mistake the GR00T patch made (I3): raising in the first branch fired before
    the other sources could contribute.
    """
    patched = _patched_source()
    assert "successes = [False] * env.num_envs" in patched
    # And no raise was injected into that branch.
    fallback = patched[patched.index("successes = [False] * env.num_envs"):]
    assert "raise RuntimeError" not in fallback.split("\n\n")[0]


def test_the_decision_happens_at_the_rollout_boundary():
    """Checked once, where the result is assembled -- not per step."""
    patched = _patched_source()
    assert "no success telemetry was observed" in patched
    assert patched.index("if not _telemetry_observed:") < patched.index(
        '        "success": torch.stack(all_successes, dim=1),')


def test_a_changed_upstream_stops_the_run():
    module = _module()
    moved = _PINNED.read_text().replace('        if "final_info" in info:\n',
                                        '        if False:\n', 1)
    with pytest.raises(RuntimeError, match="anchor 1 occurs 0 times"):
        module._patch_success_telemetry(moved)


def test_a_missing_initialisation_anchor_stops_the_run():
    module = _module()
    moved = _PINNED.read_text().replace("    all_successes = []\n", "    pass\n")
    with pytest.raises(RuntimeError, match="all_successes initialisation"):
        module._patch_success_telemetry(moved)


def test_the_patch_is_applied_after_the_commit_check_and_is_idempotent():
    source = _ENTRY.read_text()
    assert "_patch_success_telemetry(_eval_src)" in source
    assert 'if "no success telemetry was observed" not in _eval_src:' in source
    assert source.index("fork commit mismatch") < source.index(
        "_patch_success_telemetry(_eval_src)")


def test_re_patching_an_already_patched_source_is_refused():
    """Guards against double application producing two flags or two checks."""
    module = _module()
    with pytest.raises(RuntimeError):
        module._patch_success_telemetry(_patched_source())


def _shape_validator():
    """The validator the patch injects, executed rather than inspected."""
    namespace = {}
    exec(compile(_module()._TELEMETRY_VALIDATOR, "<validator>", "exec"), namespace)
    return namespace["_require_boolean_success_vector"]


@pytest.mark.parametrize("value,reason", [
    ("false", "the string is truthy, so bool() made it a SUCCESS"),
    ("", "an empty string is falsy, so bool() made it a failure"),
    (None, "None is falsy, so bool() made it a failure"),
    (2, "an integer is not a measurement of task achievement"),
    (1.0, "a float is not a boolean outcome"),
    ([], "no outcome was reported at all"),
])
def test_a_non_boolean_scalar_is_not_an_outcome(value, reason):
    """C4: telemetry was coerced before it could be validated.

    Probes against pinned upstream accepted the string "false" as SUCCESS and None as
    failure. Neither measures whether the task was achieved, and no later boolean-vector
    check can recover the original value.
    """
    with pytest.raises(ValueError):
        _shape_validator()(value, "probe", 1), reason


def test_a_numeric_vector_is_rejected():
    numpy = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="not \nboolean|not boolean"):
        _shape_validator()(numpy.array([1, 0]), "probe", 2)


def test_a_vector_containing_nan_is_rejected():
    numpy = pytest.importorskip("numpy")
    with pytest.raises(ValueError):
        _shape_validator()(numpy.array([True, numpy.nan]), "probe", 2)


def test_a_vector_of_the_wrong_width_is_rejected():
    with pytest.raises(ValueError, match="reports 1 outcomes for 2 environments"):
        _shape_validator()([True], "probe", 2)


def test_genuine_boolean_telemetry_is_accepted():
    numpy = pytest.importorskip("numpy")
    validator = _shape_validator()
    assert validator(True, "probe", 2) == [True, True]
    assert validator([False, True], "probe", 2) == [False, True]
    assert validator(numpy.array([False]), "probe", 1) == [False]


def test_the_shape_patch_runs_before_the_observation_patch():
    """A value must be validated before anything marks it observed."""
    source = _ENTRY.read_text()
    assert source.index("_patch_telemetry_shape(_eval_src)") < source.index(
        "_patch_success_telemetry(_eval_src)")


def test_the_patches_compose_in_production_order():
    """Executed, not asserted from text. These patches edit adjacent lines, so composition
    is checked rather than assumed."""
    module = _module()
    combined = module._patch_success_telemetry(
        module._patch_telemetry_shape(_PINNED.read_text()))
    ast.parse(combined)
    assert "_require_boolean_success_vector" in combined
    assert "no success telemetry was observed" in combined


def test_a_changed_upstream_stops_the_shape_patch():
    module = _module()
    moved = _PINNED.read_text().replace(
        '            successes = final_info["is_success"].tolist()\n', "", 1)
    with pytest.raises(RuntimeError, match="telemetry-shape patch anchor 1 occurs 0 times"):
        module._patch_telemetry_shape(moved)


# --- I5: the audit must run on the policy that generates actions -------------------------

def _policy_audit():
    namespace = {}
    exec(compile(_module()._POLICY_AUDIT, "<audit>", "exec"), namespace)
    return namespace["_audit_loaded_policy"]


class _Policy:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def _checkpoint(tmp_path, tensors, name="model.safetensors"):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    directory = tmp_path / "ckpt"
    directory.mkdir(exist_ok=True)
    save_file({k: torch.zeros(*shape) for k, shape in tensors.items()},
              str(directory / name))
    return str(directory)


_BASE = {"layer.weight": (4, 4), "layer.bias": (4,)}


def test_a_fully_loaded_policy_passes(tmp_path):
    torch = pytest.importorskip("torch")
    state = {k: torch.zeros(*shape) for k, shape in _BASE.items()}
    _policy_audit()(_Policy(state), _checkpoint(tmp_path, _BASE))


def test_a_policy_nesting_the_model_under_a_wrapper_passes(tmp_path):
    """LeRobot nests the model under its own attribute while the checkpoint stores it flat."""
    torch = pytest.importorskip("torch")
    state = {f"model.{k}": torch.zeros(*shape) for k, shape in _BASE.items()}
    _policy_audit()(_Policy(state), _checkpoint(tmp_path, _BASE))


def test_a_checkpoint_tensor_absent_from_the_policy_is_refused(tmp_path):
    """I5: the audited CPU probe was a DIFFERENT instance from the CUDA policy that scored.

    A tensor present in the checkpoint but missing from the policy means the file was not
    consumed -- the failure a digest cannot see, because the checkpoint is intact and the
    model is partly random.
    """
    torch = pytest.importorskip("torch")
    state = {"layer.weight": torch.zeros(4, 4)}
    with pytest.raises(RuntimeError, match="strict policy load audit FAILED"):
        _policy_audit()(_Policy(state), _checkpoint(tmp_path, _BASE))


def test_a_shape_disagreement_is_refused(tmp_path):
    torch = pytest.importorskip("torch")
    state = {"layer.weight": torch.zeros(8, 8), "layer.bias": torch.zeros(4)}
    with pytest.raises(RuntimeError, match="shape"):
        _policy_audit()(_Policy(state), _checkpoint(tmp_path, _BASE))


def test_a_checkpoint_without_tensors_is_refused(tmp_path):
    """Provenance that cannot be checked is not provenance."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="no .safetensors found"):
        _policy_audit()(_Policy({}), str(empty))


def test_the_audit_runs_in_the_evaluator_process_after_construction():
    patched = _module()._patch_policy_load_audit(_PINNED.read_text())
    assert "_audit_loaded_policy(policy, cfg.policy.pretrained_path)" in patched
    assert patched.index("policy = make_policy(") < patched.index("_audit_loaded_policy(policy")


def test_a_changed_upstream_stops_the_audit_patch():
    module = _module()
    moved = _PINNED.read_text().replace("    policy = make_policy(\n", "", 1)
    with pytest.raises(RuntimeError, match="policy-audit anchor occurs 0 times"):
        module._patch_policy_load_audit(moved)


def test_all_three_lerobot_patches_compose():
    """They edit the same file; composition is checked rather than assumed."""
    module = _module()
    combined = module._patch_success_telemetry(
        module._patch_telemetry_shape(
            module._patch_policy_load_audit(_PINNED.read_text())))
    ast.parse(combined)
    for marker in ("_audit_loaded_policy", "_require_boolean_success_vector",
                   "no success telemetry was observed"):
        assert marker in combined


# --- Cycle-6 C2: checkpoint-chosen classes in the saved processor graphs -----------------

_REAL_GRAPH = {"steps": [
    {"registry_name": "molmoact2_pack_inputs",
     "config": {"checkpoint_path": "base", "discrete_action_tokenizer": "fast_tokenizer"}},
    {"registry_name": "normalize_step", "config": {}},
]}


def _policy_dir(tmp_path, preprocessor, postprocessor=None):
    directory = tmp_path / "policy"
    directory.mkdir(exist_ok=True)
    (directory / "policy_preprocessor.json").write_text(json.dumps(preprocessor))
    if postprocessor is not None:
        (directory / "policy_postprocessor.json").write_text(json.dumps(postprocessor))
    return str(directory)


def test_a_registry_only_graph_is_accepted(tmp_path):
    """Registry entries resolve to INSTALLED code, so they are the trusted path.

    Registry names are deliberately not enumerated: the registry contains only installed code,
    and enumerating them would risk rejecting the component's own trainer output.
    """
    _module()._validate_saved_processor_graphs(_policy_dir(tmp_path, _REAL_GRAPH))


def test_a_valid_postprocessor_is_also_accepted(tmp_path):
    _module()._validate_saved_processor_graphs(
        _policy_dir(tmp_path, _REAL_GRAPH, {"steps": [{"registry_name": "unnormalize"}]}))


def test_a_dotted_class_step_is_refused(tmp_path):
    """C2: `class` is resolved with importlib and INVOKED with checkpoint-supplied kwargs.

    A probe reached subprocess.Popen's constructor that way. Disabling network access does not
    help: the import is local and the side effects are in the constructor.
    """
    graph = {"steps": [{"class": "subprocess.Popen", "config": {"args": ["id"]}}]}
    with pytest.raises(SystemExit):
        _module()._validate_saved_processor_graphs(_policy_dir(tmp_path, graph))


def test_a_hostile_step_alongside_a_valid_one_is_refused(tmp_path):
    """The rewrite only looked for the step it needed, so a malicious one could ride along."""
    graph = {"steps": _REAL_GRAPH["steps"] + [{"class": "subprocess.Popen"}]}
    with pytest.raises(SystemExit):
        _module()._validate_saved_processor_graphs(_policy_dir(tmp_path, graph))


def test_a_hostile_step_in_the_postprocessor_only_is_refused(tmp_path):
    """Both graphs are loaded, so validating one would leave the other open."""
    with pytest.raises(SystemExit):
        _module()._validate_saved_processor_graphs(
            _policy_dir(tmp_path, _REAL_GRAPH, {"steps": [{"class": "os.system"}]}))


def test_a_step_resolving_to_nothing_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        _module()._validate_saved_processor_graphs(
            _policy_dir(tmp_path, {"steps": [{"config": {}}]}))


def test_an_unreadable_or_misshapen_graph_is_refused(tmp_path):
    module = _module()
    with pytest.raises(SystemExit):
        module._validate_saved_processor_graphs(
            _policy_dir(tmp_path, {"steps": "not a list"}))
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "policy_preprocessor.json").write_text("{not json")
    with pytest.raises(SystemExit):
        module._validate_saved_processor_graphs(str(directory))


def test_the_graphs_are_validated_before_the_rewrite():
    """Rewriting first would mean patching a graph that was never checked."""
    source = _ENTRY.read_text()
    assert source.index("_validate_saved_processor_graphs(policy_dir)") < source.index(
        'prep_path = os.path.join(policy_dir, "policy_preprocessor.json")')
