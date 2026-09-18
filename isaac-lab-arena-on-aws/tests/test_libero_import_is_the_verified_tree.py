"""The verified LIBERO tree must be the one the interpreter actually imported.

Cycle-17 I1. The evaluator did:

    __import__("libero")                       # result DISCARDED
    git -C LIBERO_DIR rev-parse HEAD           # checks a DIRECTORY

Two objects, never connected. A decoy `libero` package earlier on `sys.path` satisfied the import
while the git check inspected an unrelated pristine checkout at LIBERO_DIR -- and the run was
attributed to code it never executed. A reviewer executed exactly that scenario and reported
"LIBERO already available, revision verified" while the only git command addressed the other
directory.

Second gap in the same check: HEAD equality attests the REF, not the CONTENTS. A modified tracked
file at the pinned commit passes a rev-parse check while running different code.

These tests exercise the real binding logic against real directories rather than asserting source
text, because the defect was that two correct-looking checks described different things -- which any
substring assertion would also have missed.
"""
import ast
import os
import pathlib
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EVALUATOR = _ROOT / "entrypoints/eval/libero/openvla/eval_entry.py"


def _binding_holds(module_file: str, libero_dir: str) -> bool:
    """The evaluator's containment rule, applied to real paths.

    Mirrors the production expression deliberately: the point is to prove the RULE rejects a decoy,
    and the production site is asserted separately below to be using this shape.
    """
    mod_real = os.path.realpath(module_file)
    dir_real = os.path.realpath(libero_dir)
    return os.path.commonpath([mod_real, dir_real]) == dir_real


def test_a_decoy_package_outside_the_verified_tree_is_refused(tmp_path):
    """The reviewer's scenario: pristine checkout at LIBERO_DIR, foreign package winning the import."""
    verified = tmp_path / "pinned" / "LIBERO"
    (verified / "libero").mkdir(parents=True)
    (verified / "libero" / "__init__.py").write_text("# the pinned implementation\n")

    decoy = tmp_path / "somewhere_else" / "libero"
    decoy.mkdir(parents=True)
    decoy_init = decoy / "__init__.py"
    decoy_init.write_text("# a DIFFERENT implementation that won sys.path\n")

    assert not _binding_holds(str(decoy_init), str(verified)), (
        "a package outside the verified tree was accepted; the git check would then attest a "
        "revision for code that never ran")
    assert _binding_holds(str(verified / "libero" / "__init__.py"), str(verified)), (
        "the pinned implementation itself was rejected, which would fail every legitimate run")


def test_a_symlink_cannot_straddle_the_boundary(tmp_path):
    """realpath on BOTH sides, or a symlink inside the tree points at foreign code and passes."""
    verified = tmp_path / "LIBERO"
    (verified / "libero").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    real_init = outside / "__init__.py"
    real_init.write_text("# foreign\n")

    link = verified / "libero" / "__init__.py"
    try:
        link.symlink_to(real_init)
    except (OSError, NotImplementedError):      # pragma: no cover - platform dependent
        pytest.skip("symlinks unavailable")

    assert not _binding_holds(str(link), str(verified)), (
        "a symlink from inside the verified tree to foreign code was accepted; realpath must be "
        "applied to the module file, not just to the directory")


def test_a_dirty_tree_at_the_pinned_commit_is_detected(tmp_path):
    """HEAD equality attests the ref, not the contents."""
    repo = tmp_path / "LIBERO"
    (repo / "libero").mkdir(parents=True)
    tracked = repo / "libero" / "benchmark.py"
    tracked.write_text("ORIGINAL = 1\n")

    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-q", "-m", "pinned"]):
        subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True)

    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head, "fixture repo has no HEAD"

    # Modify a TRACKED file. HEAD is unchanged -- a rev-parse check still passes.
    tracked.write_text("ORIGINAL = 999   # behaviour changed\n")
    still = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert still == head, "the premise of this test is that HEAD does NOT move"

    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    dirty = [ln for ln in status.splitlines() if ln.strip() and not ln.startswith("?? ")]
    assert dirty, (
        "a modified tracked file produced no porcelain output, so the clean-tree check cannot "
        "distinguish the pinned code from modified code at the same commit")


def test_untracked_files_do_not_make_the_tree_dirty(tmp_path):
    """Generated caches are normal; failing on them would break every legitimate run."""
    repo = tmp_path / "LIBERO"
    repo.mkdir()
    (repo / "keep.py").write_text("x = 1\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "p"]):
        subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True)

    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "keep.cpython-312.pyc").write_bytes(b"\x00")

    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout
    dirty = [ln for ln in status.splitlines() if ln.strip() and not ln.startswith("?? ")]
    assert not dirty, f"untracked files were treated as modifications: {dirty}"


def test_the_evaluator_binds_the_import_to_the_tree():
    """The production site must use the module's own file, not just check a directory."""
    # AST, not substring: my first version asserted the variable NAME appeared somewhere, which stayed
    # true when the ASSIGNMENT was deleted because later lines still referenced it. What matters is
    # that the __import__ result is BOUND rather than discarded as a bare expression statement.
    tree = ast.parse(_EVALUATOR.read_text())
    bound, discarded = [], []
    for node in ast.walk(tree):
        is_import_call = (isinstance(node, ast.Call)
                          and getattr(node.func, "id", "") == "__import__"
                          and node.args and isinstance(node.args[0], ast.Constant)
                          and node.args[0].value == "libero")
        if not is_import_call:
            continue
        parent_is_bare = any(
            isinstance(p, ast.Expr) and p.value is node for p in ast.walk(tree))
        (discarded if parent_is_bare else bound).append(node.lineno)
    assert bound, (
        f"__import__(\"libero\") is not bound to a name (bare at line(s) {discarded}), so the "
        f"imported implementation is never connected to the tree whose revision is verified")
    assert not discarded, (
        f"__import__(\"libero\") is discarded at line(s) {discarded}")

    code = "\n".join(line.split("#")[0] for line in _EVALUATOR.read_text().splitlines())
    assert "commonpath" in code, "no containment check binds the imported file to the verified tree"
    assert "status" in code and "porcelain" in code, (
        "no clean-tree check: HEAD equality attests the ref, not the contents")
