"""A report must not name a backbone the run never loaded.

`_BACKBONE_IDENTITY` was initialised at module import to
`{"repo_id": "nvidia/Cosmos-Reason2-2B"}` in both GR00T evaluators. The N1.6 path calls
`ensure_gr00t_venv_n16()` and never `precache_cosmos()`, so it loads no Cosmos at all -- yet every
N1.6 report, and every positive-control report (which is always N1.6), published that identity.

Nothing caught it. `backbone_identity` was allowlisted in TOP_OPTIONAL_KEYS, read by no check, and
copied verbatim into the attestation. So the receipt asserted a backbone the run had never touched,
and validation passed.

A report with NO backbone identity states a gap. One naming the wrong backbone states a falsehood,
and a falsehood in an attestation is worse than a gap because a consumer cannot tell it is wrong.
"""
import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GR00T_EVALUATORS = {
    "arena gr00t": "entrypoints/eval/isaac_arena/gr00t/eval_entry.py",
    "libero gr00t": "entrypoints/eval/libero/gr00t/eval_entry.py",
}


@pytest.mark.parametrize("label,relative", sorted(_GR00T_EVALUATORS.items()))
def test_the_backbone_identity_starts_empty(label, relative):
    """Resolved from the AST: the module-level assignment must be an EMPTY dict.

    A default here is a claim made before anything was loaded, by every path including the ones that
    never load it.
    """
    tree = ast.parse((_ROOT / relative).read_text())
    found = []
    for node in tree.body:                      # module level ONLY
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_BACKBONE_IDENTITY":
            found.append(node.value)
        elif isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_BACKBONE_IDENTITY" for t in node.targets):
            found.append(node.value)
    assert len(found) == 1, f"{label}: expected exactly one module-level _BACKBONE_IDENTITY, got {len(found)}"
    value = found[0]
    assert isinstance(value, ast.Dict) and not value.keys, (
        f"{label} initialises _BACKBONE_IDENTITY with content at module import. Any path that does "
        f"not load that backbone -- N1.6 calls ensure_gr00t_venv_n16() and never precache_cosmos() -- "
        f"then publishes an identity for a model it never touched")


@pytest.mark.parametrize("label,relative", sorted(_GR00T_EVALUATORS.items()))
def test_the_repo_id_is_set_where_it_is_actually_loaded(label, relative):
    """Guards the guard: emptying the dict is only correct if something populates it on the real path."""
    source = (_ROOT / relative).read_text()
    assert '_BACKBONE_IDENTITY["repo_id"]' in source, (
        f"{label} never assigns repo_id, so the identity would now be empty on EVERY path including "
        f"the one that does load a backbone -- that trades a false claim for a missing one")


def test_the_validator_rejects_a_half_stated_backbone():
    """A repo id without a commit is a MOVING reference that reads as though it pinned something."""
    from vla_pipeline.common.validator import ReportInvalid, validate_report

    from test_v2_validator import EXPECT, good_report

    # Absent is fine: not every family loads an external backbone.
    r = good_report()
    r.pop("backbone_identity", None)
    validate_report(r, **EXPECT)

    # Empty is fine: an explicit "this run loaded none".
    r = good_report()
    r["backbone_identity"] = {}
    validate_report(r, **EXPECT)

    # repo_id alone -- the exact shape the module-level default produced before a commit was resolved.
    r = good_report()
    r["backbone_identity"] = {"repo_id": "nvidia/Cosmos-Reason2-2B"}
    with pytest.raises(ReportInvalid, match="exactly"):
        validate_report(r, **EXPECT)

    # A tag instead of a resolved commit.
    r = good_report()
    r["backbone_identity"] = {"repo_id": "nvidia/Cosmos-Reason2-2B", "resolved_commit": "main"}
    with pytest.raises(ReportInvalid, match="40-character hex"):
        validate_report(r, **EXPECT)

    # Coherent.
    r = good_report()
    r["backbone_identity"] = {"repo_id": "nvidia/Cosmos-Reason2-2B", "resolved_commit": "a" * 40}
    validate_report(r, **EXPECT)


def _server_module():
    """The seeded-server module, loaded with heavyweight third-party imports stubbed."""
    import importlib.util
    import sys
    import types

    for name in ("torch", "transformers", "zmq", "gr00t"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                stub = types.ModuleType(name)
                stub.__path__ = []
                sys.modules[name] = stub
    path = (pathlib.Path(__file__).resolve().parents[1]
            / "entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py")
    spec = importlib.util.spec_from_file_location("seeded_server_probe", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:                      # pragma: no cover - partial import is enough
        pass
    return module


_REPO = "nvidia/Cosmos-Reason2-2B"


def _scope(checkpoint_root, audits, validated=None):
    return {"server_seeding_evidence": {
        "processor_validated": validated or checkpoint_root,
        "strict_load_audits": audits}}


def test_the_report_names_the_commit_INFERENCE_loaded_not_the_precached_one(tmp_path):
    """The whole point, and the defect the previous fix left open.

    The server audit recorded `consumed_commit` -- the commit the SERVING process resolved -- and nothing
    read it. All three report sites copied a module global populated by the PRECACHE step, so a precache
    commit A with a serving commit B published backbone_identity=A. A false attribution is worse than an
    absent one, because a consumer cannot tell it is wrong.
    """
    module = _server_module()
    serving = "d" * 40
    identity = module.consumed_backbone_identity(
        _scope(str(tmp_path), [{"path": _REPO, "consumed_commit": serving}]), str(tmp_path), _REPO)
    assert identity == {"repo_id": _REPO, "resolved_commit": serving}


def test_evidence_from_a_different_launch_cannot_attribute_this_one(tmp_path):
    """Dose points run on separate ports; one point reading another's proof was a real defect."""
    module = _server_module()
    other = tmp_path / "another-checkpoint"
    other.mkdir()
    with pytest.raises(module.BackboneAttributionError, match="different launch"):
        module.consumed_backbone_identity(
            _scope(str(tmp_path), [{"path": _REPO, "consumed_commit": "c" * 40}],
                   validated=str(other)),
            str(tmp_path), _REPO)


def test_two_commits_in_one_run_cannot_both_be_the_backbone(tmp_path):
    module = _server_module()
    with pytest.raises(module.BackboneAttributionError, match="MORE THAN ONE"):
        module.consumed_backbone_identity(
            _scope(str(tmp_path), [{"path": _REPO, "consumed_commit": "c" * 40},
                                   {"path": _REPO, "consumed_commit": "e" * 40}]),
            str(tmp_path), _REPO)


def test_a_moving_tag_is_refused_because_it_pins_nothing(tmp_path):
    module = _server_module()
    with pytest.raises(module.BackboneAttributionError, match="pins nothing"):
        module.consumed_backbone_identity(
            _scope(str(tmp_path), [{"path": _REPO, "consumed_commit": "main"}]),
            str(tmp_path), _REPO)


def test_absent_evidence_is_refused_rather_than_defaulted(tmp_path):
    """An unattributable backbone must FAIL. It determines preprocessing, so it can change while the
    checkpoint, its digest, the image and the sourcedir all stay identical."""
    module = _server_module()
    for scope, pattern in ((None, "not a dict"),
                           ({}, "no server_seeding_evidence"),
                           (_scope(str(tmp_path), [{"path": "other/model",
                                                    "consumed_commit": "c" * 40}]),
                            "no strict-load audit")):
        with pytest.raises(module.BackboneAttributionError, match=pattern):
            module.consumed_backbone_identity(scope, str(tmp_path), _REPO)


def test_the_arena_evaluator_actually_BINDS_the_consumed_identity():
    """The wiring, which no test reached.

    The helper being correct is worthless if the report still publishes the precache value -- which was
    the entire defect: `consumed_commit` was recorded by the server and read by nobody. Removing the
    binding left every other test green, because the remaining assertion only required each report site
    to MENTION backbone_identity, and the precache assignment mentions it.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"
    tree = ast.parse(path.read_text())
    calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "consumed_backbone_identity"]
    assert calls, (
        "the Arena evaluator never calls consumed_backbone_identity, so backbone_identity keeps the "
        "PRECACHE commit. Precache commit A with serving commit B then publishes A -- a false "
        "attribution, which a consumer cannot detect")

    # And it must run AFTER the report is assembled, or the precache value survives.
    assignments = [n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript)
                           and isinstance(t.slice, ast.Constant)
                           and t.slice.value == "backbone_identity" for t in n.targets)]
    assert assignments, "nothing re-binds metrics['backbone_identity'] after the report is built"
    assert max(calls) >= min(assignments), (
        "the consumed identity is computed before the report re-binding that should use it")
