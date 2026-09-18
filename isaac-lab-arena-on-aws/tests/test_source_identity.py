"""ResolvedCheckpoint must refuse what it previously accepted.

Every case here was executed against an earlier revision of the module and ACCEPTED. They are
input-validation defects in the one object that is supposed to make source identity unforgeable, so
an accepted bad value propagates into a published attestation.

Two classes of finding are represented:

  Format holes. The module used `re.match(r"^[0-9a-f]{64}$", value)`, and Python's `$` matches BEFORE
  a trailing newline, so a digest read with read() instead of read().strip() validated cleanly.
  `int(size_bytes)` turned True into 1. A whitespace-only repo id passed a truthiness check.

  Bypass holes, found by a reviewer after the format holes were closed. The generated constructor
  still accepted `tree_digest=...`, so the measuring factories only protected callers who chose to
  use them; and the archive sha/size were unchecked assertions, so a well-formed hash for an
  unrelated file passed. Every measured field is now `init=False` and measured inside __post_init__.
"""
import pathlib
import tarfile

import pytest

from vla_pipeline.common.digest import measure_archive, weights_digest
from vla_pipeline.common.source_identity import (
    ARCHIVE,
    SNAPSHOT,
    ResolvedCheckpoint,
    SourceIdentityError,
    bare_hex,
    from_archive,
    from_snapshot,
    prefixed,
)
from vla_pipeline.common.validator import TOP_KEYS

_COMMIT = "a" * 40
_HEX = "b" * 64


@pytest.fixture
def tree(tmp_path):
    """A real checkpoint-shaped directory, so digests are measured over actual bytes."""
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "test"}')
    (root / "model.safetensors").write_bytes(b"\x00" * 2048)
    return root


@pytest.fixture
def archive(tmp_path, tree):
    """A REAL archive of the tree, with measurements DERIVED by the real measure_archive.

    Earlier fixtures wrote two bytes and declared a 2048-byte all-"b" measurement. The constructor
    now measures the named file and refuses a disagreement, so those fixtures went red -- correctly.
    An invented positive-case measurement cannot exercise a check whose entire purpose is to compare
    a claim against the file it names.
    """
    path = tmp_path / "ckpt.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        tf.add(tree, arcname="ckpt")
    sha, size = measure_archive(str(path))
    return path, sha, size


# ---------------------------------------------------------------- format holes


def test_a_trailing_newline_is_not_a_valid_digest():
    with pytest.raises(SourceIdentityError):
        bare_hex(_HEX + "\n")
    with pytest.raises(SourceIdentityError):
        bare_hex("sha256:" + _HEX + "\n")


def test_a_trailing_newline_is_not_a_valid_commit(tree):
    with pytest.raises(SourceIdentityError):
        from_snapshot(str(tree), "org/model", _COMMIT + "\n")


def test_a_moving_tag_is_refused(tree):
    """I5: the Arena positive control published resolved_commit='main'."""
    for bad in ("main", "refs/heads/main", "v1.0", _COMMIT[:39]):
        with pytest.raises(SourceIdentityError):
            from_snapshot(str(tree), "org/model", bad)


def test_a_blank_or_padded_identifier_is_refused(tree):
    with pytest.raises(SourceIdentityError):
        from_snapshot(str(tree), "   ", _COMMIT)
    with pytest.raises(SourceIdentityError):
        from_snapshot(str(tree), " org/model ", _COMMIT)


def test_load_root_must_exist(tmp_path):
    with pytest.raises(SourceIdentityError):
        from_snapshot(str(tmp_path / "absent"), "org/model", _COMMIT)


# ---------------------------------------------------------------- bypass holes


def test_the_constructor_cannot_be_handed_a_digest(tree):
    """The reviewer's demonstration: this succeeded against a real directory whose digest differed.

    tree_digest is now init=False, so the argument does not exist and the measuring path is the only
    path. A TypeError is the correct outcome -- the value cannot be expressed, not merely rejected.
    """
    with pytest.raises(TypeError):
        ResolvedCheckpoint(load_root=str(tree), kind=SNAPSHOT,
                           tree_digest=prefixed("0" * 64),
                           checkpoint="org/model", revision=_COMMIT)


def test_the_constructor_cannot_be_handed_an_archive_measurement(tree):
    with pytest.raises(TypeError):
        ResolvedCheckpoint(load_root=str(tree), kind=ARCHIVE, checkpoint="s3://b/k",
                           revision=None, archive_sha256=_HEX, archive_size_bytes=2048)


def test_a_supplied_archive_measurement_must_match_the_named_file(tree, archive):
    """Previously an unchecked assertion: any well-formed hash for any existing path passed."""
    path, sha, size = archive
    with pytest.raises(SourceIdentityError, match="not the archive named here"):
        from_archive(str(tree), str(path), "s3://b/k", sha256=_HEX, size_bytes=size)
    with pytest.raises(SourceIdentityError, match="size disagrees"):
        from_archive(str(tree), str(path), "s3://b/k", sha256=sha, size_bytes=size + 1)


def test_a_directory_is_not_an_archive(tree):
    """os.path.exists accepted a directory; measure_archive does not."""
    with pytest.raises(Exception):
        from_archive(str(tree), str(tree), "s3://b/k", sha256=_HEX, size_bytes=2048)


def test_booleans_and_floats_are_not_archive_sizes(tree, archive):
    path, sha, _ = archive
    for bad in (True, 1.9, "2048", None):
        with pytest.raises(SourceIdentityError):
            from_archive(str(tree), str(path), "s3://b/k", sha256=sha, size_bytes=bad)


def test_an_archive_source_carries_no_upstream_revision(tree, archive):
    """The docstring said archive revisions are null; nothing enforced it."""
    path, sha, size = archive
    with pytest.raises(SourceIdentityError):
        ResolvedCheckpoint(load_root=str(tree), kind=ARCHIVE, checkpoint="s3://b/k",
                           revision=_COMMIT, archive_path=str(path),
                           expect_sha256=sha, expect_size_bytes=size)


# ---------------------------------------------------------------- measured identity


def test_the_tree_digest_is_measured_here_not_supplied(tree):
    rc = from_snapshot(str(tree), "org/model", _COMMIT)
    assert rc.tree_digest == weights_digest(str(tree))
    other = tree.parent / "other"
    other.mkdir()
    (other / "config.json").write_text('{"model_type": "other"}')
    assert from_snapshot(str(other), "org/model", _COMMIT).tree_digest != rc.tree_digest


def test_the_stored_digest_is_canonical(tree, archive):
    path, sha, size = archive
    rc = from_archive(str(tree), str(path), "s3://b/k", sha256=sha, size_bytes=size)
    assert rc.tree_digest.startswith("sha256:")
    assert rc.archive_sha256 == bare_hex(sha)
    assert rc.report_source_fields()["source_archive"]["sha256"] == bare_hex(sha)


def test_report_identity_fields_come_from_one_object(tree):
    rc = from_snapshot(str(tree), "org/model", _COMMIT)
    fields = rc.report_identity_fields()
    snap = fields["source_snapshot"]
    assert snap["repo_id"] == fields["checkpoint"], "the source must name the report's own checkpoint"
    assert snap["resolved_commit"] == fields["checkpoint_revision"]
    assert snap["tree_sha256"] == bare_hex(rc.tree_digest)
    assert "source_archive" not in fields


def test_every_required_identity_key_is_present_for_both_kinds(tree, archive):
    """The regression a reviewer BLOCKED on: an omitted key the validator REQUIRES.

    report_identity_fields() emitted checkpoint_revision only when non-None, so an archive source
    omitted it entirely -- and validator.py lists it among the required top-level keys and reads it
    directly. Every archive-mode evaluation would have completed and then failed to publish.

    Asserted against the validator's own TOP_KEYS, so a key added there is required here without
    anyone remembering to update this test.
    """
    path, sha, size = archive
    for rc in (from_archive(str(tree), str(path), "s3://b/k", sha256=sha, size_bytes=size),
               from_snapshot(str(tree), "org/model", _COMMIT)):
        fields = rc.report_identity_fields()
        missing = ({"checkpoint", "checkpoint_revision"} & set(TOP_KEYS)) - set(fields)
        assert not missing, (
            f"{rc.kind} source omits required report key(s) {sorted(missing)}; the run would "
            f"complete and then be refused at publication")
    arch = from_archive(str(tree), str(path), "s3://b/k", sha256=sha, size_bytes=size)
    assert arch.report_identity_fields()["checkpoint_revision"] is None


def test_exactly_one_variant_is_ever_emitted(tree, archive):
    path, sha, size = archive
    arch = from_archive(str(tree), str(path), "s3://b/k", sha256=sha, size_bytes=size)
    snap = from_snapshot(str(tree), "org/model", _COMMIT)
    assert set(arch.report_source_fields()) == {"source_archive"}
    assert set(snap.report_source_fields()) == {"source_snapshot"}
    assert arch.kind == ARCHIVE and snap.kind == SNAPSHOT
