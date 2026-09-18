"""The LIBERO GR00T evaluator must launch its policy server through the seeded wrapper.

It did not. `entrypoints/eval/libero/gr00t/eval_entry.py` launched
`gr00t/eval/run_gr00t_server.py` directly, so neither the seed binding (I11) nor the strict
weight-load audit (C3) applied to the LIBERO path -- while the wrapper's own comment and a
commit message claimed both launchers were covered.

Nothing caught it because the existing assertion covered Arena's two servers (n16 and n17),
not Arena versus LIBERO. That is the gap these tests close.
"""
from __future__ import annotations

import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LIBERO_ENTRY = _REPO_ROOT / "entrypoints/eval/libero/gr00t/eval_entry.py"
_ARENA_ENTRY = _REPO_ROOT / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"
_WRAPPER = _REPO_ROOT / "entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py"


def test_the_libero_entry_does_not_launch_the_server_directly():
    """The precise defect."""
    source = _LIBERO_ENTRY.read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        assert '"gr00t/eval/run_gr00t_server.py"' not in stripped, (
            f"the server must be launched through the wrapper, not directly: {stripped}")


def test_the_libero_entry_launches_through_the_wrapper():
    source = _LIBERO_ENTRY.read_text()
    assert "SEEDED_SERVER_ENTRY" in source
    assert '"python", SEEDED_SERVER_ENTRY,' in source


def test_the_libero_entry_passes_the_eval_seed_to_the_server():
    """The wrapper requires EVAL_SEED and refuses to start without it."""
    source = _LIBERO_ENTRY.read_text()
    assert '"EVAL_SEED": SEED' in source


def test_a_missing_wrapper_is_fatal_on_the_libero_path():
    """Falling back to a direct launch would silently drop both guarantees."""
    source = _LIBERO_ENTRY.read_text()
    assert "if not os.path.exists(SEEDED_SERVER_ENTRY):" in source
    assert "Refusing to " in source
    guard = source.index("if not os.path.exists(SEEDED_SERVER_ENTRY):")
    launch = source.index('"python", SEEDED_SERVER_ENTRY,')
    assert guard < launch, "the existence check must precede the launch"


def test_both_evaluators_reference_the_same_wrapper_filename():
    """Two wrappers would drift; the guarantees must come from one implementation."""
    for entry in (_LIBERO_ENTRY, _ARENA_ENTRY):
        assert "gr00t_seeded_server.py" in entry.read_text()
    assert _WRAPPER.is_file()


def test_the_wrapper_is_staged_into_the_gr00t_sourcedir():
    """Arena bakes it into its image; LIBERO has no such image, so staging delivers it."""
    staging = (_REPO_ROOT / "src/vla_pipeline/common/sourcedir.py").read_text()
    assert "gr00t_seeded_server.py" in staging
    assert 'if family == "gr00t":' in staging


def test_staging_fails_loudly_when_the_wrapper_is_absent():
    """An omitted file must not yield a sourcedir that runs unseeded."""
    staging = (_REPO_ROOT / "src/vla_pipeline/common/sourcedir.py").read_text()
    index = staging.index("gr00t_seeded_server.py")
    following = staging[index:index + 900]
    assert "FileNotFoundError" in following
    assert "HARD FAIL" in following


def test_the_staged_wrapper_does_not_collide_with_another_basename():
    """stage() hard-fails on duplicate basenames, so adding a file must not introduce one."""
    from vla_pipeline.common.sourcedir import _staging_manifest

    names = [path.name for path in _staging_manifest(_REPO_ROOT, "gr00t")]
    assert len(names) == len(set(names)), f"duplicate basenames in the manifest: {names}"
    assert "gr00t_seeded_server.py" in names


def test_the_wrapper_is_not_staged_for_families_that_do_not_use_it():
    from vla_pipeline.common.sourcedir import _staging_manifest

    for family in ("openvla", "molmoact2"):
        names = [path.name for path in _staging_manifest(_REPO_ROOT, family)]
        assert "gr00t_seeded_server.py" not in names, (
            f"{family} does not run the GR00T server; staging it would be misleading")
