"""cycle-17 R2: the launchers' parameter payload must satisfy the pipeline's own declared constraints.

Found by a REAL launch against a provisioned account, not by any test. StartPipelineExecution refused
the call outright:

    ValidationException: [format attribute "int64" not supported,
                          string "" is too short (length: 0, required minimum: 1)]

The suite was 1263 green and the pipeline could not start, because nothing validated the payload
against the contract. This is the offline equivalent of that API call: it earns its place because it
FAILS on the pre-fix launcher and PASSES after, without touching AWS.
"""
import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LAUNCHERS = {"libero": "scripts/run_libero.py", "arena": "scripts/run_arena.py"}


@pytest.mark.parametrize("label,relative", sorted(_LAUNCHERS.items()))
def test_no_launcher_sends_an_unconditional_empty_string_override(label, relative):
    """An empty override is not "no override" -- it is an INVALID one.

    SageMaker parameters here declare minLength 1, and DatasetRevision's pipeline default is the
    __FROM_SUITE_MANIFEST__ sentinel. Sending "" replaces a working default with a rejected value, so
    an absent value must OMIT the key rather than blank it.
    """
    source = (_ROOT / relative).read_text()
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            # `X or ""` / `... if cond else ""` in a params dict blanks the pipeline default
            blanks = [n for n in ast.walk(value)
                      if isinstance(n, ast.Constant) and n.value == ""]
            if blanks and key.value[:1].isupper():
                offenders.append(key.value)
    assert not offenders, (
        f"{label} can send an empty string for {sorted(set(offenders))}; SageMaker rejects the whole "
        f"StartPipelineExecution call, so the key must be omitted when the value is absent")


@pytest.mark.parametrize("label,relative", sorted(_LAUNCHERS.items()))
def test_the_revision_override_is_conditional(label, relative):
    """The specific repair, asserted by mechanism: set only when a revision exists."""
    source = (_ROOT / relative).read_text()
    assert 'params["DatasetRevision"] = _revision' in source, (
        f"{label} no longer sets DatasetRevision conditionally")
    guard = source[:source.index('params["DatasetRevision"] = _revision')]
    assert guard.rstrip().endswith("if _revision:"), (
        f"{label} sets DatasetRevision without guarding on a non-empty revision")
