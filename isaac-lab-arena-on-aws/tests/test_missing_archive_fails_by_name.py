"""A missing archive must fail by NAME, never as an unbound variable.

Every pipeline-mode producer measures its archive behind an existence check and reads the result
later. Two of them had NO handling for the absent case, so an already-extracted channel fell straight
through and the failure surfaced as:

    NameError: name '_ARCHIVE_SHA' is not defined

which names a variable rather than the condition, well after the point where the channel could have
been diagnosed. The refactor that moved these measurements from module globals into locals is what
exposed it -- as globals they were merely empty, which was its own defect.

The condition is real: pipeline mode REQUIRES a measurable archive, because the pre-extraction
measurement is the only value Validate can compare against its own independent one. A checkpoint
arriving already unpacked cannot be attributed to the bytes the training step produced.

WHAT THIS CHECKS, and why it is shaped this way. Proving "no path reaches the read with the variable
unbound" is a dataflow question the AST does not answer cheaply, and my first three attempts at a
clever matcher each matched the wrong set -- one skipped two files silently, which is worse than a
crude check because a skip reads as a pass. So this asserts the property that actually distinguishes
the fixed code from the broken code: a producer that measures an archive must ALSO name the
missing-archive case, and that handler must terminate.
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

#: Every producer says this when the mounted channel holds no archive. A shared phrase is deliberate:
#: it is one grep for an operator diagnosing a failed job, and it makes the check below uniform.
_MISSING_ARCHIVE_PHRASE = "no model.tar.gz in"


@pytest.mark.parametrize("label,relative", sorted(_PRODUCERS.items()))
def test_every_producer_names_the_missing_archive_case(label, relative):
    source = (_ROOT / relative).read_text()
    if "measure_archive" not in source:
        pytest.skip(f"{label} does not measure an archive")

    assert _MISSING_ARCHIVE_PHRASE in source, (
        f"{label} measures an archive but never names the case where there is none. An "
        f"already-extracted channel then falls through to read measurements that were never taken, "
        f"and the failure appears as a NameError naming a variable instead of the condition")


@pytest.mark.parametrize("label,relative", sorted(_PRODUCERS.items()))
def test_the_missing_archive_handler_terminates(label, relative):
    """Naming it is not enough -- the handler must stop, or execution reads the unbound value anyway."""
    source = (_ROOT / relative).read_text()
    if _MISSING_ARCHIVE_PHRASE not in source:
        pytest.skip(f"{label} does not measure an archive")

    tree = ast.parse(source)
    handlers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for branch in (node.body, node.orelse):
                if not branch:
                    continue
                if _MISSING_ARCHIVE_PHRASE in ast.dump(ast.Module(body=branch, type_ignores=[])):
                    handlers.append(branch)
    assert handlers, f"{label}: the phrase is present but not inside an if/else branch"

    for branch in handlers:
        terminates = any(
            isinstance(sub, (ast.Raise, ast.Return))
            or (isinstance(sub, ast.Call) and getattr(sub.func, "attr", "") == "exit")
            for node in branch for sub in ast.walk(node))
        assert terminates, (
            f"{label}'s missing-archive handler at line {branch[0].lineno} does not exit, raise or "
            f"return, so execution continues into code reading measurements never taken")


def test_the_phrase_is_present_in_more_than_one_producer():
    """Guards the guard: if only one file used the phrase, the parametrized checks would be nearly
    vacuous and a regression in another producer would skip rather than fail."""
    have = [label for label, rel in _PRODUCERS.items()
            if _MISSING_ARCHIVE_PHRASE in (_ROOT / rel).read_text()]
    assert len(have) >= 3, (
        f"only {have} name the missing-archive case; the others either do not measure an archive or "
        f"have regressed to falling through")
