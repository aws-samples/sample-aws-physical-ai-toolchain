"""A positive control must not extract the mounted checkpoint it deliberately ignores.

The positive-control path serves NVIDIA's published N1.6 checkpoint downloaded from HuggingFace. The
mounted archive plays no part in it -- that is the entire point of the mode.

But the branch sat AFTER `extract_checkpoint(INPUT_MODEL, ...)` in main(), so every positive control
extracted and measured an archive it would never read. That was merely wasteful until extraction began
CONSTRUCTING a ResolvedCheckpoint from it: at that point a positive control could be REJECTED by the
identity of a checkpoint it does not use, because construction validates the archive measurement
against the file. A reviewer traced the ordering and blocked on it.

The general lesson, which is why this test asserts ORDER rather than the specific failure: doing work
you have decided to ignore is a latent failure waiting for the ignored path to grow a constraint.
"""
import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ARENA = _ROOT / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"


def _main_function() -> ast.FunctionDef:
    tree = ast.parse(_ARENA.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("main() not found in the Arena evaluator")


def _posctrl_branch(main: ast.FunctionDef) -> ast.If:
    """The top-level `if EVAL_POSCTRL_N16 == "true":` that owns the diagnostic path.

    Identified by its body containing the N1.6 download, not by position -- so the test still finds it
    if surrounding code moves.
    """
    candidates = []
    for node in main.body:
        if not isinstance(node, ast.If):
            continue
        if any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "download_n16_posctrl_ckpt"
               for c in ast.walk(node)):
            candidates.append(node)
    assert len(candidates) == 1, (
        f"expected exactly one positive-control branch in main(), found {len(candidates)}")
    return candidates[0]


def _extract_calls(scope) -> list:
    return [n for n in ast.walk(scope)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "extract_checkpoint"]


def test_the_positive_control_branch_precedes_extraction():
    main = _main_function()
    branch = _posctrl_branch(main)

    extracts = _extract_calls(main)
    assert extracts, "main() never extracts a checkpoint; this test is checking the wrong thing"

    for call in extracts:
        assert branch.lineno < call.lineno, (
            f"extract_checkpoint is called at line {call.lineno}, at or before the positive-control "
            f"branch at line {branch.lineno}. A diagnostic that ignores the mounted checkpoint would "
            f"still extract and MEASURE one -- and since extraction now constructs a "
            f"ResolvedCheckpoint, the positive control can be rejected by the identity of a "
            f"checkpoint it does not use")


def test_the_positive_control_branch_does_not_itself_extract():
    """Moving the branch earlier is worthless if the branch extracts on its own."""
    branch = _posctrl_branch(_main_function())
    inside = _extract_calls(branch)
    assert not inside, (
        f"the positive-control branch calls extract_checkpoint at line(s) "
        f"{[c.lineno for c in inside]}; it evaluates a downloaded snapshot and has no mounted archive")


def test_the_positive_control_branch_returns_rather_than_falling_through():
    """If it fell through, the ordering fix would be undone by the archive path running anyway."""
    branch = _posctrl_branch(_main_function())
    assert any(isinstance(n, ast.Return) for n in ast.walk(branch)), (
        "the positive-control branch does not return, so execution continues into the ordinary "
        "archive path and extracts the checkpoint the branch exists to ignore")
