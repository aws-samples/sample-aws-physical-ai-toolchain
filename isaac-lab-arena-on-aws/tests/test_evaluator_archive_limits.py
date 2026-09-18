"""I4: the GPU evaluators extracted the checkpoint archive with no resource bound.

Path safety is not resource safety. A well-formed archive with no traversal and no links can still
carry millions of members or expand past the volume, and this extraction happens on paid
accelerator capacity BEFORE Validate ever sees the run -- so the cost lands exactly where it hurts
most. Validate's safe_extract already enforced these two limits; the evaluators did not.
"""
from __future__ import annotations

import ast
import io
import os
import pathlib
import tarfile
import tempfile

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EVALUATORS = {
    "arena_gr00t": _ROOT / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py",
    "libero_gr00t": _ROOT / "entrypoints/eval/libero/gr00t/eval_entry.py",
    "libero_molmoact2": _ROOT / "entrypoints/eval/libero/molmoact2/eval_entry.py",
    "libero_openvla": _ROOT / "entrypoints/eval/libero/openvla/eval_entry.py",
}


@pytest.mark.parametrize("name", sorted(_EVALUATORS))
def test_every_evaluator_bounds_its_archive(name):
    """All four, not three: fixing a subset has been a recurring defect in this component."""
    source = _EVALUATORS[name].read_text()
    assert "_bound_archive" in source, f"{name} extracts without a resource bound"


@pytest.mark.parametrize("name", sorted(_EVALUATORS))
def test_the_bound_precedes_every_extraction(name):
    """A bound applied after extractall would be decoration."""
    source = _EVALUATORS[name].read_text()
    assert source.index("_bound_archive(") < source.index(".extractall(")


@pytest.mark.parametrize("name", sorted(_EVALUATORS))
def test_the_bound_never_materialises_the_member_list(name):
    """getmembers() is itself the unbounded operation the member limit exists to prevent."""
    source = _EVALUATORS[name].read_text()
    fn = next(n for n in ast.parse(source).body
              if isinstance(n, ast.FunctionDef) and n.name == "_bound_archive")
    body = ast.get_source_segment(source, fn)
    code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    assert "getmembers" not in code
    assert "for member in tar:" in code


def _helper(name):
    source = _EVALUATORS[name].read_text()
    fn = next(n for n in ast.parse(source).body
              if isinstance(n, ast.FunctionDef) and n.name == "_bound_archive")
    ns = {"os": os, "sys": __import__("sys"),
          "_MAX_ARCHIVE_MEMBERS": 200000, "_MAX_ARCHIVE_BYTES": 64 * 1024 ** 3}
    exec(ast.get_source_segment(source, fn), ns)
    return ns


def _archive(count, size=8):
    path = os.path.join(tempfile.mkdtemp(), "a.tar")
    with tarfile.open(path, "w") as handle:
        for i in range(count):
            info = tarfile.TarInfo(f"f{i}")
            info.size = size
            handle.addfile(info, io.BytesIO(b"\0" * size))
    return path


def test_a_legitimate_archive_still_extracts_after_the_bound():
    """The bound iterates the handle, so it must rewind or extraction yields nothing.

    This is the failure mode a source-only test would miss entirely.
    """
    ns = _helper("libero_openvla")
    with tarfile.open(_archive(5)) as handle:
        ns["_bound_archive"](handle, "test")
        dest = tempfile.mkdtemp()
        handle.extractall(dest)
        assert len(os.listdir(dest)) == 5


def test_too_many_members_is_refused():
    ns = _helper("libero_openvla")
    ns["_MAX_ARCHIVE_MEMBERS"] = 5
    with tarfile.open(_archive(6)) as handle:
        with pytest.raises(SystemExit):
            ns["_bound_archive"](handle, "test")


def test_too_many_declared_bytes_is_refused():
    ns = _helper("libero_openvla")
    ns["_MAX_ARCHIVE_BYTES"] = 20
    with tarfile.open(_archive(5)) as handle:
        with pytest.raises(SystemExit):
            ns["_bound_archive"](handle, "test")


def test_the_byte_limit_fits_the_job_volume():
    """512 GiB against a 100 GB volume could never fire before the disk filled."""
    for name, path in sorted(_EVALUATORS.items()):
        source = path.read_text()
        assert "64 * 1024 ** 3" in source, f"{name} has no volume-sized byte limit"
        assert "512 * 1024 ** 3" not in source
