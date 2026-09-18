"""Unit tests for the normative weights digest (models/common/digest.py)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vla_pipeline", "common"))
from digest import DigestError, weights_digest  # noqa: E402


def _mk(root, rel, content=b"x"):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def test_deterministic_and_order_independent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, order in ((a, ["w1.bin", "w2.bin"]), (b, ["w2.bin", "w1.bin"])):
        os.makedirs(root)
        for name in order:
            _mk(str(root), name, content=name.encode())
    assert weights_digest(str(a)) == weights_digest(str(b))


def test_content_change_changes_digest(tmp_path):
    _mk(str(tmp_path), "w.bin", b"one")
    d1 = weights_digest(str(tmp_path))
    _mk(str(tmp_path), "w.bin", b"two")
    assert weights_digest(str(tmp_path)) != d1


def test_rename_changes_digest(tmp_path):
    _mk(str(tmp_path), "w.bin", b"same")
    d1 = weights_digest(str(tmp_path))
    os.rename(tmp_path / "w.bin", tmp_path / "v.bin")
    assert weights_digest(str(tmp_path)) != d1


def test_framing_prevents_boundary_shift(tmp_path):
    # (path="ab", content="c") vs (path="a", content="bc") must differ --
    # the length framing is what guarantees this.
    a, b = tmp_path / "a", tmp_path / "b"
    os.makedirs(a)
    os.makedirs(b)
    _mk(str(a), "ab", b"c")
    _mk(str(b), "a", b"bc")
    assert weights_digest(str(a)) != weights_digest(str(b))


def test_exclusions(tmp_path):
    _mk(str(tmp_path), "w.bin", b"w")
    d1 = weights_digest(str(tmp_path))
    _mk(str(tmp_path), "checkpoint_manifest.json", b"{}")
    _mk(str(tmp_path), "train.log", b"noise")
    _mk(str(tmp_path), "data.lock", b"noise")
    _mk(str(tmp_path), ".hidden/secret.bin", b"noise")
    assert weights_digest(str(tmp_path)) == d1


def test_config_files_included(tmp_path):
    _mk(str(tmp_path), "w.bin", b"w")
    d1 = weights_digest(str(tmp_path))
    _mk(str(tmp_path), "config.json", b"{}")
    assert weights_digest(str(tmp_path)) != d1


def test_symlink_fatal(tmp_path):
    target = _mk(str(tmp_path), "w.bin", b"w")
    os.symlink(target, tmp_path / "link.bin")
    with pytest.raises(DigestError, match="symlink"):
        weights_digest(str(tmp_path))


def test_empty_fatal(tmp_path):
    with pytest.raises(DigestError, match="no digestable files"):
        weights_digest(str(tmp_path))


def test_subdirectories_walked(tmp_path):
    _mk(str(tmp_path), "sub/dir/w.bin", b"w")
    d = weights_digest(str(tmp_path))
    assert d.startswith("sha256:") and len(d) == 7 + 64


def test_directory_symlink_fatal(tmp_path):
    real = tmp_path / "real"
    os.makedirs(real)
    _mk(str(real), "w.bin", b"w")
    _mk(str(tmp_path), "w0.bin", b"x")
    os.symlink(real, tmp_path / "linkdir")
    with pytest.raises(DigestError, match="symlink"):
        weights_digest(str(tmp_path))


def test_hidden_double_dot_name_excluded_not_escape(tmp_path):
    # "..weights" is a legal hidden basename: excluded by the dot rule,
    # NOT an escape error.
    _mk(str(tmp_path), "w.bin", b"w")
    d1 = weights_digest(str(tmp_path))
    _mk(str(tmp_path), "..weights", b"noise")
    assert weights_digest(str(tmp_path)) == d1


def test_empty_after_exclusions_fatal(tmp_path):
    _mk(str(tmp_path), "only.log", b"noise")
    with pytest.raises(DigestError, match="no digestable files"):
        weights_digest(str(tmp_path))


def test_known_answer_framing(tmp_path):
    # Known-answer test proving exact big-endian u64 framing.
    import hashlib
    import struct
    _mk(str(tmp_path), "a.bin", b"hello")
    expect = hashlib.sha256(
        struct.pack(">Q", 5) + b"a.bin" + struct.pack(">Q", 5) + b"hello"
    ).hexdigest()
    assert weights_digest(str(tmp_path)) == "sha256:" + expect


def test_chunk_boundary(tmp_path):
    # File larger than the 1 MiB read chunk hashes correctly.
    import hashlib
    import struct
    payload = os.urandom(1024 * 1024 + 17)
    _mk(str(tmp_path), "big.bin", payload)
    expect = hashlib.sha256(
        struct.pack(">Q", 7) + b"big.bin"
        + struct.pack(">Q", len(payload)) + payload
    ).hexdigest()
    assert weights_digest(str(tmp_path)) == "sha256:" + expect


@pytest.mark.parametrize("failed_directory", [".", "sub", ".hidden"])
@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_directory_traversal_error_fatal(tmp_path, monkeypatch, failed_directory, error_type):
    _mk(str(tmp_path), "w.bin", b"readable")
    _mk(str(tmp_path), "sub/w.bin", b"nested")
    _mk(str(tmp_path), ".hidden/w.bin", b"hidden")
    failed_path = os.path.normpath(str(tmp_path / failed_directory))
    original_scandir = os.scandir
    failure = error_type(13, "injected directory read failure", failed_path)

    def scandir(path):
        if os.path.normpath(os.fspath(path)) == failed_path:
            raise failure
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    with pytest.raises(DigestError, match="cannot traverse checkpoint") as exc:
        weights_digest(str(tmp_path))
    assert exc.value.__cause__ is failure
    assert failed_path in str(exc.value)


def test_nested_log_excluded_by_basename(tmp_path):
    _mk(str(tmp_path), "w.bin", b"w")
    d1 = weights_digest(str(tmp_path))
    _mk(str(tmp_path), "sub/dir/train.log", b"noise")
    assert weights_digest(str(tmp_path)) == d1
