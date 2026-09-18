"""S6: archive extraction must bound COST, not only path safety.

The extraction guard rejected traversal and links, which stops an archive writing outside its
destination. It did not bound size or member count, so a well-formed archive with no traversal
and no links could still expand to more than the disk holds, or carry millions of tiny members
-- and extraction would be attempted before anything noticed.
"""
from __future__ import annotations

import importlib.util
import io
import pathlib
import tarfile

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _REPO_ROOT / "entrypoints/validate_entry.py"


def _module():
    spec = importlib.util.spec_from_file_location("validate_entry_limits", _ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(path: pathlib.Path, files: dict[str, bytes]) -> pathlib.Path:
    with tarfile.open(path, "w:gz") as handle:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    return path


def test_a_legitimate_checkpoint_archive_still_extracts(tmp_path):
    """The limits must not constrain any supported checkpoint."""
    module = _module()
    archive = _archive(tmp_path / "model.tar.gz",
                       {"config.json": b'{"model_type": "gr00t"}',
                        "model.safetensors": b"W" * 4096})
    destination = tmp_path / "out"
    destination.mkdir()
    with tarfile.open(archive) as handle:
        module.safe_extract(handle, str(destination))
    assert sorted(p.name for p in destination.iterdir()) == [
        "config.json", "model.safetensors"]


def test_an_archive_declaring_more_bytes_than_the_limit_is_refused(tmp_path):
    """A declared-size bomb: rejected from the header, before any bytes are written."""
    module = _module()
    archive = _archive(tmp_path / "bomb.tar.gz", {"huge.bin": b""})
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        members[0].size = module.MAX_ARCHIVE_BYTES + 1
        handle.getmembers = lambda: members
        with pytest.raises(SystemExit):
            module.safe_extract(handle, str(tmp_path / "out"))
    assert not (tmp_path / "out").exists(), "nothing may be written before the guard fires"


def test_an_archive_with_too_many_members_is_refused(tmp_path):
    """I4: this used to stub handle.getmembers().

    The guard no longer calls getmembers -- that was the defect, since materialising the member
    list is what the limit was supposed to bound -- so stubbing it left this test exercising
    nothing. It now builds a real multi-member archive and lets the guard iterate it.
    """
    module = _module()
    archive = _archive(tmp_path / "many.tar.gz",
                       {f"f{i}": b"x" for i in range(5)})
    with tarfile.open(archive) as handle:
        module.MAX_ARCHIVE_MEMBERS = 3
        with pytest.raises(SystemExit):
            module.safe_extract(handle, str(tmp_path / "out"))


def test_the_limits_are_declared_well_above_a_real_checkpoint():
    """A limit tight enough to reject a real checkpoint would be a new failure mode.

    I4: the byte floor used to be 100 GiB, which encoded the very problem the fix removes -- a
    limit at or above the job volume can never fire before the disk fills. Both concerns are
    real, so the band is now explicit: comfortably above the tens of gigabytes a real checkpoint
    occupies, and strictly below the volume that has to hold it.
    """
    module = _module()
    assert module.MAX_ARCHIVE_MEMBERS >= 100_000
    assert module.MAX_ARCHIVE_BYTES >= 32 * 1024 ** 3, "would reject a real checkpoint"
    assert module.MAX_ARCHIVE_BYTES < 100 * 1024 ** 3, "could not fire before the disk filled"


def test_the_guard_runs_before_extraction():
    """Checking after extractall would defeat the purpose."""
    source = _ENTRY.read_text()
    assert source.index("archive resource guard") < source.index("tar.extractall")


def test_path_safety_is_still_enforced(tmp_path):
    """The new limits must not have displaced the traversal guard."""
    module = _module()
    archive = tmp_path / "escape.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(archive) as handle:
        with pytest.raises(SystemExit):
            module.safe_extract(handle, str(tmp_path / "out"))


# --- I4: the limits must be enforced BEFORE the member list is materialised ------------------

def test_the_member_list_is_not_materialised_before_the_limits_apply():
    """getmembers() reads every header into memory.

    Checking MAX_ARCHIVE_MEMBERS after that call meant an archive carrying millions of members
    exhausted memory inside the very call the limit was supposed to bound.
    """
    source = _ENTRY.read_text()
    block = source[source.index("def safe_extract"):]
    block = block[:block.index("\ndef ")]
    # Strip comment lines: the code comment NAMES tar.getmembers() to explain why it was
    # removed, so matching raw text would fail on the explanation rather than on the call. This
    # is the third assertion this session to trip over its own prose.
    code = "\n".join(line for line in block.splitlines()
                     if not line.strip().startswith("#"))
    assert "tar.getmembers()" not in code, "the limits must not follow a full materialisation"
    assert "for member in tar:" in block, "headers must be read incrementally"
    # The count check must sit INSIDE the reading loop, not after it.
    loop = code[code.index("for member in tar:"):]
    assert "MAX_ARCHIVE_MEMBERS" in loop[:loop.index("\n    for member in members:")]


def test_the_expanded_size_limit_fits_the_job_volume():
    """It was 512 GiB against a 100 GB default volume, so it could never fire in time."""
    source = _ENTRY.read_text()
    assert "512 * 1024 ** 3" not in source
    assert "64 * 1024 ** 3" in source


def test_the_byte_limit_message_names_the_volume_constraint():
    source = _ENTRY.read_text()
    assert "exceeds what the job volume holds" in source
