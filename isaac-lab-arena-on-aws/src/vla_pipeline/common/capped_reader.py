"""A byte-capped file wrapper, so an archive cannot consume unbounded memory before it is inspected.

Cycle-14 I7. Counting members, or summing their sizes, cannot bound extraction memory: tarfile reads
a PAX/GNU extended header fully into memory inside `_proc_pax()` BEFORE yielding the TarInfo the
count would examine. Measured, one 1-byte member with a 4 MiB comment peaks around 16 MiB while the
iteration sees a single file of size 1, so the bound has to sit UNDER tarfile, on the bytes it is
allowed to read at all, not above it on what it reports.

Use the production helper, which combines a compressed-input byte budget with
bounds on decompressed extended headers before tarfile allocates their payloads:

    with capped_tar_open(path, "r:*", max_bytes=LIMIT) as tar:
        ...

CappedReader alone bounds only bytes read from its input stream. It does not
bound decompressed memory. capped_tar_open adds the extended-header guard;
callers must still validate members, expanded sizes and extraction paths.
"""
from __future__ import annotations

import contextlib
import io
import os
import tarfile

# The budget for bytes READ from an archive. Deliberately a different quantity from
# validate_entry.MAX_ARCHIVE_BYTES, which bounds DECLARED EXPANDED size -- one governs what we
# are willing to pull off disk, the other what the archive claims it unpacks to. Same env var
# and same default so an operator raising one does not silently leave the other behind.
DEFAULT_MAX_ARCHIVE_BYTES = int(os.environ.get("MAX_ARCHIVE_BYTES", str(64 * 1024 ** 3)))


class ArchiveTooLargeError(Exception):
    """The archive tried to read more bytes than the caller permitted."""


class CappedReader(io.RawIOBase):
    """Fail closed once `max_bytes` have been read.

    Deliberately NOT a truncating reader. Returning short data would hand tarfile a corrupt stream
    and let it report a confusing structural error instead of the real cause, or -- worse -- let a
    truncated archive look like a small valid one. Reaching the cap is an error and says so.
    """

    def __init__(self, raw, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._raw = raw
        self._max = int(max_bytes)
        self._count = 0

    @property
    def bytes_read(self) -> int:
        return self._count

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        """Read at most the remaining budget, and never allocate more than that.

        The request is CLAMPED BEFORE the underlying read. Accounting after the fact was the first
        version of this class and it did not work: tarfile asks for a PAX header in a single read, so
        a 4 MiB request against a 32 KiB budget materialised all 4 MiB and only then raised -- the
        error was correct and the memory was already gone. Measured peak actually went UP versus no
        wrapper at all, because the exception path retained the chunk.

        Clamping means the same request reads 32 KiB + 1 byte, notices it is over, and fails. The
        extra byte is what distinguishes "exactly at the limit" from "wants more than the limit".
        """
        remaining = self._max - self._count
        if size is None or size < 0:
            size = remaining + 1          # never unbounded; -1 would defeat the whole class
        allowed = min(size, remaining + 1)
        chunk = self._raw.read(allowed)
        self._count += len(chunk)
        if self._count > self._max:
            raise ArchiveTooLargeError(
                f"archive tried to read past the {self._max}-byte limit "
                f"(requested {size} bytes with {remaining} remaining); refusing to continue")
        return chunk

    def readinto(self, buffer) -> int:
        chunk = self.read(len(buffer))
        buffer[:len(chunk)] = chunk
        return len(chunk)

    # tarfile needs position access even in non-stream mode ("r:" calls tell() during __init__ and
    # seeks between members). Delegate both: SEEKING CONSUMES NO MEMORY, so the cap on bytes
    # actually READ is still what bounds consumption. Counting stays monotonic on purpose -- a seek
    # backwards does not refund budget, so re-reading the same region cannot be used to loop
    # underneath the limit.
    def seekable(self) -> bool:
        return self._raw.seekable()

    def tell(self) -> int:
        return self._raw.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._raw.close()


# The metadata budget, INDEPENDENT of the read budget: they bound different quantities. The read cap
# limits how much of the file we pull off disk; this limits how much a single extension header may
# allocate, which is what the compressed cap could never do (cycle-15 I2).
DEFAULT_MAX_ARCHIVE_HEADER_BYTES = int(
    os.environ.get("MAX_ARCHIVE_HEADER_BYTES", str(1024 * 1024)))

# The two whole-body metadata paths in CPython's tarfile. Checking only x/g leaves the equivalent
# allocation reachable through the GNU long-name types.
_EXTENSION_HEADER_TYPES = frozenset({
    tarfile.XHDTYPE,        # x -- PAX per-file extended header
    tarfile.XGLTYPE,        # g -- PAX global extended header
    tarfile.SOLARIS_XHDTYPE,  # X -- Solaris extended header
    tarfile.GNUTYPE_LONGNAME,  # L
    tarfile.GNUTYPE_LONGLINK,  # K
    tarfile.GNUTYPE_SPARSE,  # S -- the old sparse header; 1.0 is refused separately
})


def header_limited_tarinfo(max_header_bytes: int = DEFAULT_MAX_ARCHIVE_HEADER_BYTES):
    """A TarInfo subclass that refuses an oversized extension header BEFORE its body is read.

    This is the only placement that bounds the allocation. tarfile reads an extension header's whole
    body inside _proc_member/_proc_pax before yielding anything a caller could inspect, so a wrapper
    around the file cannot help: for `r:gz` the header is decompressed from a few KiB on disk, and
    the compressed cap never fires. Measured before this existed: a 4,237-byte archive peaked ~16 MiB.

    A class built PER OPEN rather than global mutable policy, so concurrent readers cannot affect
    each other's limit.

    _proc_member is private. It is overridden here in preference to frombuf because the whole-body
    read happens in _proc_member, and test_capped_reader.py asserts the hook still exists so an
    upgrade that renames it fails loudly rather than silently removing the bound.
    """
    if max_header_bytes <= 0:
        raise ValueError(f"max_header_bytes must be positive, got {max_header_bytes}")
    limit = int(max_header_bytes)

    class HeaderLimitedTarInfo(tarfile.TarInfo):
        def _proc_member(self, archive):
            if self.type in _EXTENSION_HEADER_TYPES:
                if self.size < 0:
                    raise ArchiveTooLargeError(
                        f"invalid extended-header size: {self.size}")
                block = tarfile.BLOCKSIZE
                padded = ((self.size + block - 1) // block) * block
                if padded > limit:
                    raise ArchiveTooLargeError(
                        f"extended header declares {self.size} bytes ({padded} padded), "
                        f"limit {limit}; refusing to continue")
            if self.type == tarfile.GNUTYPE_SPARSE:
                raise ArchiveTooLargeError(
                    "archive contains a GNU sparse member; refusing to continue")
            return super()._proc_member(archive)

        # cycle-16 I1: GNU sparse 1.0 puts its sparse MAP in the body of the FOLLOWING member, and
        # CPython parses and accumulates that map before returning the member -- so the type check
        # above never sees it. Measured on 3.10/3.11: ~6.3 MB allocated from a 108 KB archive with a
        # 1024-byte limit.
        #
        # REFUSED outright rather than size-bounded. A model checkpoint tar has no legitimate reason
        # to be sparse, so there is no valid input to preserve here, and bounding the accumulation
        # would mean re-implementing three sparse-map parsers correctly -- more surface than the
        # feature is worth. Fail closed with a message that names the format.
        def _proc_gnusparse_10(self, next, pax_headers, tarfile_):
            raise ArchiveTooLargeError(
                "archive uses GNU sparse 1.0, whose sparse map is unbounded metadata; refusing to "
                "continue (a model checkpoint archive is never legitimately sparse)")

        def _proc_gnusparse_01(self, next, pax_headers):
            raise ArchiveTooLargeError(
                "archive uses GNU sparse 0.1; refusing to continue")

        def _proc_gnusparse_00(self, next, pax_headers, buf):
            raise ArchiveTooLargeError(
                "archive uses GNU sparse 0.0; refusing to continue")

    return HeaderLimitedTarInfo


@contextlib.contextmanager
def capped_tar_open(path, mode: str = "r:gz", *, max_bytes: int,
                    max_header_bytes: int = DEFAULT_MAX_ARCHIVE_HEADER_BYTES):
    """Open `path` as a tar archive that may not read more than `max_bytes`.

    Exists so a call site becomes a one-line change:

        with capped_tar_open(path, "r:gz", max_bytes=LIMIT) as tar:

    rather than a nested `with` that re-indents the whole body. Re-indenting six call sites by hand
    is how a partially-applied edit slips through, and this file's own history has three of those.

    Opens by PATH deliberately: reading the archive into a BytesIO first would allocate exactly what
    the cap exists to prevent.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    with open(path, "rb") as raw:
        # These are already-downloaded regular files, so the extent limit is a plain size check --
        # simpler and exact, where wrapping the stream both cost reads and (for r:gz) bounded the
        # wrong quantity.
        actual = os.fstat(raw.fileno()).st_size
        if actual > max_bytes:
            raise ArchiveTooLargeError(
                f"archive is {actual} bytes, exceeding the {max_bytes}-byte limit; "
                f"refusing to open it")
        with tarfile.open(fileobj=raw, mode=mode,
                          tarinfo=header_limited_tarinfo(max_header_bytes)) as tar:
            yield tar
