"""A report dict must not re-assign a field that `**_RESOLVED.report_identity_fields()` provides.

A later key in a dict literal WINS over an earlier spread. So this:

    metrics = {
        **_RESOLVED.report_identity_fields(),   # checkpoint, checkpoint_revision, source variant
        "checkpoint": HF_REPO,                  # <- silently overrides the measured value
        "checkpoint_revision": (None if PIPELINE_MODE else HF_REV),
    }

looks like it derives identity from one object and does not. Two of four producers shipped exactly
that: molmoact2 re-assigned BOTH fields four lines after the spread, and libero gr00t re-assigned the
revision. A reviewer caught it after I had claimed all four producers derived their identity from a
single object -- the claim was false and the spread was cosmetic in those two.

Resolved with the AST rather than by reading, because the two statements sit four lines apart in a
250-line dict literal and the defect is invisible at a glance. This checks the PROPERTY -- no key in
the same dict may shadow the spread -- so it also catches a NEW producer, or a new identity field
added to report_identity_fields() later.
"""
import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_PRODUCERS = {
    "libero openvla": "entrypoints/eval/libero/openvla/eval_entry.py",
    "libero gr00t": "entrypoints/eval/libero/gr00t/eval_entry.py",
    "libero molmoact2": "entrypoints/eval/libero/molmoact2/eval_entry.py",
    "arena gr00t": "entrypoints/eval/isaac_arena/gr00t/eval_entry.py",
}


def _fields_the_spread_provides() -> set:
    """Derived from the source of truth, not hardcoded, so a new field is covered automatically."""
    from vla_pipeline.common.source_identity import ARCHIVE, SNAPSHOT, ResolvedCheckpoint

    # Read the keys report_identity_fields() writes straight out of its AST: constructing a real
    # instance needs a filesystem, and a hardcoded list here would be a second declaration that
    # drifts from the first.
    src = pathlib.Path(ResolvedCheckpoint.__module__.replace(".", "/"))
    module = (_ROOT / "src/vla_pipeline/common/source_identity.py").read_text()
    tree = ast.parse(module)
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("report_identity_fields",
                                                               "report_source_fields"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for k in sub.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            keys.add(k.value)
    assert {"checkpoint", "checkpoint_revision"} <= keys, (
        f"the derivation found {sorted(keys)}; if report_identity_fields() no longer emits the "
        f"identity fields this test is checking the wrong thing")
    assert ARCHIVE and SNAPSHOT      # imported to assert the module is the one under test
    return keys


@pytest.mark.parametrize("label,relative", sorted(_PRODUCERS.items()))
def test_no_report_dict_shadows_the_identity_spread(label, relative):
    provided = _fields_the_spread_provides()
    tree = ast.parse((_ROOT / relative).read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        # A spread appears as a None key in ast.Dict. Only consider dicts that actually spread
        # report_identity_fields().
        spreads_identity = any(
            k is None and isinstance(v, ast.Call)
            and getattr(v.func, "attr", "") == "report_identity_fields"
            for k, v in zip(node.keys, node.values))
        if not spreads_identity:
            continue

        shadowed = sorted({
            k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value in provided})
        assert not shadowed, (
            f"{label} spreads report_identity_fields() and then re-assigns {shadowed} in the same "
            f"dict literal at line {node.lineno}. A later key WINS, so the measured identity is "
            f"silently discarded and the single-source contract is cosmetic")


def test_the_check_can_actually_see_a_spread():
    """Guards the guard: if the spread is never recognised, every assertion above is vacuous."""
    found = 0
    for relative in _PRODUCERS.values():
        tree = ast.parse((_ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and any(
                    k is None and isinstance(v, ast.Call)
                    and getattr(v.func, "attr", "") == "report_identity_fields"
                    for k, v in zip(node.keys, node.values)):
                found += 1
    assert found >= len(_PRODUCERS), (
        f"expected at least one identity spread per producer, recognised {found} across "
        f"{len(_PRODUCERS)} files -- the AST match is wrong and the checks above prove nothing")
