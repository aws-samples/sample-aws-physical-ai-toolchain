"""The ONE place a checkpoint's source identity is constructed.

Cycle-17 I4/I5/I6. Every producer previously built its own source fields from module-level globals
(`_SOURCE_ARCHIVE`, `_SOURCE_SNAPSHOT`), which produced three defects at once:

  I6  producers put a PREFIXED digest ("sha256:<hex>", what weights_digest returns) into
      source_snapshot.tree_sha256, which the validator requires to be BARE 64-hex. Every HF
      evaluation would spend its full inference budget and then fail to publish.
  I4  the validator checked each field's FORMAT without binding fields to one another, so a report
      could name a repo, a commit and a digest belonging to three different things.
  I5  Arena's positive control published resolved_commit="main" -- a moving tag, which pins nothing.

The fix is not a format tweak. Both digest conventions are deliberate and both stay:
`weights_digest()` returns "sha256:<hex>" because it is a content-addressing token used in paths and
comparisons; `source_snapshot.tree_sha256` is bare hex because it is a hash FIELD. Conversion happens
in exactly one checked place -- `bare_hex()` below -- instead of five producers each remembering.

A ResolvedCheckpoint is INVOCATION-LOCAL and immutable: it is built once, by the resolver that chose
the source, and every reported field derives from it. There is no module global to seed, which is what
let a test fixture make the HF path unexecutable.

WHAT `frozen=True` DOES AND DOES NOT GIVE US. It freezes the attributes of this object. It does not
freeze the filesystem, does not prove that inference consumed `load_root`, and does not establish
that a supplied measurement was taken over the paths named here. That is why the tree digest is
measured BY the factories rather than accepted from a caller: a caller-supplied digest validates as
a format while potentially describing a different tree, which is the same defect class as recording a
backbone commit from the wrong process.

The archive BYTES digest is the one measurement that must ORIGINATE with the caller, because it has to
be taken before extraction consumes the bytes. That is a real constraint, but it is NOT accepted on
trust: `from_archive` refuses to construct without both the digest and the size, then RE-MEASURES the
archive and compares. A caller cannot pass a digest describing a different archive, and cannot omit
the measurement to skip the check -- omission raises rather than defaulting.

This paragraph previously said the factory "at least requires the archive to still exist so the claim
is not free-floating", which described an earlier and much weaker version. Left corrected rather than
deleted: a docstring that understates a guarantee invites a reader to add the check that is already
there, and one that overstates a guarantee is worse than none.
"""
from __future__ import annotations

import dataclasses
import os
import re

# fullmatch semantics, spelled without anchors on purpose: Python's `$` matches BEFORE a trailing
# newline, so `re.match(r"^[0-9a-f]{64}$", "<64 hex>\n")` SUCCEEDS. Every check here uses fullmatch.
_BARE_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMMIT40 = re.compile(r"[0-9a-f]{40}")

ARCHIVE = "archive"
SNAPSHOT = "snapshot"


class SourceIdentityError(ValueError):
    """A source could not be identified, or was identified inconsistently."""


def _weights_digest():
    """weights_digest, imported for whichever context this module is running in.

    Shipped contexts place digest.py flat beside this file; in-repo it is a package sibling.
    """
    try:
        from digest import weights_digest        # flat: sourcedir, Arena image
    except ImportError:
        from .digest import weights_digest       # in-repo package
    return weights_digest


def _measure_archive():
    """measure_archive, imported for whichever context this module is running in."""
    try:
        from digest import measure_archive        # flat: sourcedir, Arena image
    except ImportError:
        from .digest import measure_archive       # in-repo package
    return measure_archive


def bare_hex(digest: str) -> str:
    """Convert a digest to the bare 64-hex form the report schema requires.

    Accepts either "sha256:<hex>" (what weights_digest returns) or bare hex, and CHECKS the result
    rather than assuming. This exists because the alternative -- each producer stripping the prefix
    itself -- is what shipped a prefixed value into a bare-hex field five times over.
    """
    if not isinstance(digest, str) or not digest:
        raise SourceIdentityError(f"digest must be a non-empty string, got {digest!r}")
    value = digest.split(":", 1)[1] if digest.startswith("sha256:") else digest
    if not _BARE_HEX64.fullmatch(value):
        raise SourceIdentityError(
            f"digest is not exactly 64 lowercase hex characters after normalising: "
            f"{digest!r} -> {value!r}")
    return value


def prefixed(digest: str) -> str:
    """The inverse: the "sha256:<hex>" form weights_digest emits and comparisons use."""
    return "sha256:" + bare_hex(digest)


def _exact_int(value, what: str) -> int:
    """A genuine int. Rejects bool and float rather than coercing them.

    `int(True)` is 1 and `int(1.9)` is 1, so a plain int() call turns a wrong type into a
    plausible-looking size instead of an error.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceIdentityError(f"{what} must be an int, got {type(value).__name__} {value!r}")
    if value <= 0:
        raise SourceIdentityError(f"{what} must be positive, got {value!r}")
    return value


def _nonblank(value, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceIdentityError(f"{what} must be a non-blank string, got {value!r}")
    if value != value.strip():
        raise SourceIdentityError(f"{what} has surrounding whitespace: {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class ResolvedCheckpoint:
    """What this invocation actually loaded, and how it is identified.

    Every MEASURED field is `init=False`: the constructor cannot be handed a digest or an archive
    measurement at all. A reviewer demonstrated that the generated constructor accepted
    `tree_digest="sha256:"+"0"*64` for a real directory whose actual digest differed, so the
    measuring factories only protected callers who chose to use them. Measurement now happens inside
    __post_init__, which every construction path goes through.

    TRUST BOUNDARY. This establishes that the values describe the paths named here. It does NOT prove
    that inference read `load_root`, freeze the filesystem, or authenticate `checkpoint` as the true
    upstream identity of those bytes. Extraction and root selection stay owned by the resolver, which
    passes this object's `load_root` to inference; that trusted-resolver assumption is the boundary,
    and this object is not attestation against a hostile producer.
    """

    load_root: str          # the directory inference will read
    kind: str               # ARCHIVE or SNAPSHOT
    checkpoint: str         # the archive URI/path, or the HF repo id
    revision: str | None    # resolved 40-hex commit for SNAPSHOT; None for a local archive
    archive_path: str | None = None      # ARCHIVE only, the file the measurements describe
    #: The producer's pre-extraction measurement, checked against a fresh one taken here.
    expect_sha256: str | None = None
    expect_size_bytes: int | None = None
    #: Measured here, never supplied.
    tree_digest: str = dataclasses.field(init=False, default="")
    archive_sha256: str | None = dataclasses.field(init=False, default=None)
    archive_size_bytes: int | None = dataclasses.field(init=False, default=None)

    def __post_init__(self) -> None:
        # Cheap validation first: nothing below should read a multi-gigabyte tree to discover that
        # the kind is misspelled.
        if self.kind not in (ARCHIVE, SNAPSHOT):
            raise SourceIdentityError(f"unknown source kind {self.kind!r}")
        _nonblank(self.checkpoint, "checkpoint identifier")
        if self.kind == SNAPSHOT:
            # I5: a tag or branch moves, so it cannot identify the bytes that were read.
            if not (isinstance(self.revision, str) and _COMMIT40.fullmatch(self.revision)):
                raise SourceIdentityError(
                    f"a snapshot source must pin a RESOLVED 40-hex commit, got {self.revision!r} -- "
                    f"a tag or branch name identifies nothing")
            if self.archive_path is not None:
                raise SourceIdentityError("a snapshot source has no archive")
        else:
            if self.revision is not None:
                raise SourceIdentityError(
                    f"an archive source has no upstream revision to pin, got {self.revision!r}")
            if not self.archive_path:
                raise SourceIdentityError("an archive source must name its archive file")

        object.__setattr__(self, "tree_digest", _measured(self.load_root))

        if self.kind == ARCHIVE:
            # Measured HERE, then compared with what the producer measured before extraction. The
            # producer's values were previously accepted as unchecked assertions, so a well-formed
            # hash for an unrelated file passed. measure_archive also rejects a directory and an
            # empty file, which the old os.path.exists check did not.
            #
            # Measuring is possible because extraction does NOT consume the archive -- it is still on
            # disk afterwards. An earlier comment of mine claimed otherwise and a reviewer corrected
            # it.
            sha, size = _measure_archive()(self.archive_path)
            object.__setattr__(self, "archive_sha256", bare_hex(sha))
            object.__setattr__(self, "archive_size_bytes", _exact_int(size, "measured archive size"))
            if self.expect_sha256 is not None and bare_hex(self.expect_sha256) != self.archive_sha256:
                raise SourceIdentityError(
                    f"the archive measured before extraction is not the archive named here: "
                    f"expected {bare_hex(self.expect_sha256)}, measured {self.archive_sha256} "
                    f"over {self.archive_path!r}")
            if (self.expect_size_bytes is not None
                    and _exact_int(self.expect_size_bytes, "expected size_bytes")
                    != self.archive_size_bytes):
                raise SourceIdentityError(
                    f"archive size disagrees: expected {self.expect_size_bytes}, measured "
                    f"{self.archive_size_bytes} over {self.archive_path!r}")

    def report_identity_fields(self) -> dict:
        """EVERY identity field the report carries, from this one object.

        report_source_fields() alone left `checkpoint` and `checkpoint_revision` hand-assembled by
        each producer, so a report could carry a coherent source variant beside a `checkpoint` naming
        something else -- which is precisely the disagreement the validator now rejects. Emitting all
        of them together makes that disagreement unconstructible rather than merely detected.

        `checkpoint_revision` is ALWAYS present and is None for an archive source. Emitting it only
        when non-None omitted a REQUIRED key (validator.py:40, read directly at :499), so every
        archive-mode evaluation would have run to completion and then failed to publish -- the same
        defect as I6, reintroduced one field over. A reviewer caught it before it ran.
        """
        return {"checkpoint": self.checkpoint,
                "checkpoint_revision": self.revision,
                **self.report_source_fields()}

    def report_source_fields(self) -> dict:
        """The EXACTLY-ONE source variant this checkpoint implies."""
        if self.kind == ARCHIVE:
            return {"source_archive": {"sha256": bare_hex(self.archive_sha256),
                                       "size_bytes": _exact_int(self.archive_size_bytes,
                                                                "archive size_bytes")}}
        return {"source_snapshot": {"repo_id": self.checkpoint,
                                    "resolved_commit": self.revision,
                                    "tree_sha256": bare_hex(self.tree_digest)}}


def _measured(load_root: str) -> str:
    """The tree digest of load_root, with the directory checked first.

    Checked here rather than only in __post_init__ because the factories now measure BEFORE
    constructing, so an absent root would otherwise surface as a DigestError about a path instead of
    naming the source identity that could not be established.
    """
    if not load_root or not os.path.isdir(load_root):
        raise SourceIdentityError(
            f"load_root must be an existing directory to measure, got {load_root!r}")
    return _weights_digest()(load_root)


def from_archive(load_root: str, archive_path: str, checkpoint: str,
                 sha256: str, size_bytes: int) -> ResolvedCheckpoint:
    """Build an ARCHIVE identity, measuring both the extracted tree and the archive here.

    `sha256`/`size_bytes` are the producer's PRE-EXTRACTION measurement. They are no longer taken on
    trust: the constructor measures the named archive again and refuses any disagreement, which is
    what makes them evidence rather than an assertion. Passing them still matters -- they are measured
    at the moment the bytes arrived, before anything could have replaced the file.

    Both are REQUIRED. Accepting None would let a caller skip the comparison by omitting the claim,
    which is the check's only purpose.
    """
    if sha256 is None or size_bytes is None:
        raise SourceIdentityError(
            "from_archive requires the pre-extraction sha256 and size_bytes; omitting them would "
            "skip the comparison against the archive on disk")
    return ResolvedCheckpoint(
        load_root=load_root, kind=ARCHIVE, checkpoint=checkpoint, revision=None,
        archive_path=archive_path, expect_sha256=sha256, expect_size_bytes=size_bytes)


def from_snapshot(load_root: str, repo_id: str, resolved_commit: str) -> ResolvedCheckpoint:
    """Build a SNAPSHOT identity, measuring the tree HERE rather than trusting a caller.

    A caller-supplied tree digest validates as a format while potentially describing a different
    directory -- the same "verified the wrong object" shape as C1 and I1, one level down.
    """
    return ResolvedCheckpoint(load_root=load_root, kind=SNAPSHOT,
                              checkpoint=repo_id, revision=resolved_commit)
