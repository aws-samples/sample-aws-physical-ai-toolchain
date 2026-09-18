"""Publication hardening: adversarial tests for the tar path-traversal guard.

`validate_entry.safe_extract` must:
  * extract a well-formed archive byte-identically (behavior-preserving), and
  * fail closed (SystemExit) on any member that escapes the destination:
    absolute paths, ``..`` traversal, or symlink/hardlink/device members.

The guard is 3.10-3.12 safe (does not rely on the 3.12 ``filter="data"`` kwarg).
`validate_entry.py` is delivered to a bare ProcessingStep container via
stage_validate_code(), so the helper is self-contained (no vla_pipeline import);
this test imports it directly from steps/.
"""
from __future__ import annotations

import io
import os
import sys
import tarfile

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_REPO_ROOT, "entrypoints") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "entrypoints"))

import validate_entry  # noqa: E402


def _tar_with(members, tmp_path):
    """Build a .tar.gz at tmp_path/eviltar; `members` is a list of
    (name, data-bytes-or-TarInfo-mutator) tuples."""
    p = os.path.join(str(tmp_path), "archive.tar.gz")
    with tarfile.open(p, "w:gz") as tar:
        for name, payload in members:
            if isinstance(payload, bytes):
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            else:
                # payload is a callable that returns a fully-formed TarInfo
                info = payload(name)
                tar.addfile(info)
    return p


def test_valid_archive_extracts_identically(tmp_path):
    src = _tar_with(
        [("a.txt", b"hello"), ("sub/b.txt", b"world")], tmp_path
    )
    dest = os.path.join(str(tmp_path), "out")
    os.makedirs(dest)
    with tarfile.open(src, "r:gz") as tar:
        validate_entry.safe_extract(tar, dest)
    assert open(os.path.join(dest, "a.txt")).read() == "hello"
    assert open(os.path.join(dest, "sub", "b.txt")).read() == "world"


def test_parent_traversal_member_rejected(tmp_path):
    src = _tar_with([("../escape.txt", b"pwned")], tmp_path)
    dest = os.path.join(str(tmp_path), "out")
    os.makedirs(dest)
    with tarfile.open(src, "r:gz") as tar:
        with pytest.raises(SystemExit):
            validate_entry.safe_extract(tar, dest)
    assert not os.path.exists(os.path.join(str(tmp_path), "escape.txt"))


def test_absolute_path_member_rejected(tmp_path):
    src = _tar_with([("/tmp/vla_evil_abs.txt", b"pwned")], tmp_path)
    dest = os.path.join(str(tmp_path), "out")
    os.makedirs(dest)
    with tarfile.open(src, "r:gz") as tar:
        with pytest.raises(SystemExit):
            validate_entry.safe_extract(tar, dest)


def test_symlink_member_rejected(tmp_path):
    def _symlink(name):
        info = tarfile.TarInfo(name=name)
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        return info

    src = _tar_with([("link", _symlink)], tmp_path)
    dest = os.path.join(str(tmp_path), "out")
    os.makedirs(dest)
    with tarfile.open(src, "r:gz") as tar:
        with pytest.raises(SystemExit):
            validate_entry.safe_extract(tar, dest)
