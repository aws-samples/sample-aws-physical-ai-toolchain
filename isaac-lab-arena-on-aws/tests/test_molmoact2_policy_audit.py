"""C2: the adapter audit must check the INSTANTIATED policy, not the saved file.

The previous audit inspected model.safetensors for `lora_` names and a nonzero lora_B,
and a comment called that "the robust, cheap equivalent of a runtime probe". It was not:
the evaluator loads through the ordinary LeRobot path, whose from_pretrained defaults to
strict=False and only LOGS missing/unexpected keys, so a checkpoint whose tensor names do
not match the instantiated policy passed the file audit while non-strict loading left
some or all of the policy at its base state. Recomputed file digests still agreed,
because the bytes were genuine and only their effect was absent.

The probe itself needs torch, PEFT and LeRobot and can only execute inside the eval
container, so these tests check its structure and wiring. The container checks that would
exercise its behaviour are listed in test_the_container_verification_plan_is_recorded.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ENTRY = (pathlib.Path(__file__).resolve().parents[1]
          / "entrypoints/eval/libero/molmoact2/eval_entry.py")


def _probe_source() -> str:
    """The embedded probe, extracted from the production module."""
    tree = ast.parse(_ENTRY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_POLICY_LOAD_PROBE" for t in node.targets):
            return node.value.value
    raise AssertionError("_POLICY_LOAD_PROBE not found")


def test_the_probe_is_valid_python():
    """A syntax error would only surface at evaluation time, inside the container."""
    ast.parse(_probe_source())


def test_the_probe_constructs_the_policy_through_the_evaluator_factory():
    """Auditing a policy the evaluator would not build proves nothing about the run."""
    probe = _probe_source()
    assert "from lerobot.policies.factory import make_policy" in probe
    assert "make_policy(cfg=cfg, env_cfg=env_cfg" in probe
    assert "PreTrainedConfig.from_pretrained" in probe


def test_the_probe_makes_key_mismatches_fatal():
    """The whole point: non-strict loading must no longer be tolerated silently."""
    probe = _probe_source()
    assert "not missing and not unexpected" in probe
    assert "state-dict mismatch" in probe
    # The diagnostics must come from the loader the factory actually calls.
    assert "load_model_as_safetensor" in probe


def test_the_probe_inspects_live_adapter_state_not_the_file():
    probe = _probe_source()
    for expected in (
        "named_modules()",          # live modules
        "policy.state_dict()",      # live tensors
        "active_adapters",          # adapters selected
        "get_layer_status()",       # enabled / merged status
        "merged_adapters",          # unmerged
        "get_delta_weight",         # effective delta, not just nonzero lora_B
    ):
        assert expected in probe, expected


def test_the_probe_requires_a_nonzero_effective_delta():
    """A nonzero lora_B alone was the old, weaker claim."""
    probe = _probe_source()
    assert "no active adapter has a nonzero LoRA weight delta" in probe


def test_the_probe_runs_on_cpu_without_competing_for_the_gpu():
    src = _ENTRY.read_text()
    assert 'probe_env["CUDA_VISIBLE_DEVICES"] = ""' in src
    assert "--device=cpu" in _probe_source()


def test_the_old_file_only_audit_is_gone():
    """The superseded audit must not remain as a second, weaker gate."""
    src = _ENTRY.read_text()
    assert "ALL lora_B are zero" not in src
    assert "robust, cheap equivalent of a runtime probe" not in src


def test_the_audit_runs_before_the_evaluation_is_launched():
    src = _ENTRY.read_text()
    assert src.index("_POLICY_LOAD_PROBE,") < src.index("eval_cmd = [")


def test_the_audit_does_not_overclaim():
    """The original defect was a comment asserting more than the check established."""
    src = _ENTRY.read_text()
    call_site = src[src.index("Audit the INSTANTIATED policy"):]
    for caveat in ("does NOT establish", "reconstruction and adapter"):
        assert caveat in call_site, caveat


@pytest.mark.parametrize("check", [
    "a real produced checkpoint passes",
    "a renamed LoRA key fails despite nonzero B",
    "a missing action-expert or base key fails",
    "omitted manifest and saved-mode fields cannot bypass the audit",
    "manifest/saved-mode disagreement fails",
    "nonzero B with zero A fails the effective-delta assertion",
])
def test_the_container_verification_plan_is_recorded(check):
    """These require the container; record them so they are not quietly skipped.

    The probe's runtime behaviour is NOT verified by this suite. Two of the six are
    already covered offline (the bypass and disagreement cases, in
    test_molmoact2_startup.py); the rest need a real checkpoint in the eval image.
    """
    plan = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    assert plan.is_file(), f"missing {plan}"
    section = plan.read_text().split("### MolmoAct2 policy-audit diagnostics\n", 1)[1]
    section = section.split("\n## ", 1)[0]
    assert check in section, check
