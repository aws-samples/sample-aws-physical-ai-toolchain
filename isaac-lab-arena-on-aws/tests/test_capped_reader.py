"""Cycle-14 I7: the byte cap must PREVENT the allocation, not merely report it afterwards."""
import io
import pathlib
import tarfile
import tracemalloc

import pytest

from vla_pipeline.common.capped_reader import (
    ArchiveTooLargeError,
    CappedReader,
    capped_tar_open,
)

CAP = 32 * 1024


def _pax_bomb(path: pathlib.Path) -> pathlib.Path:
    """One 1-byte member carrying a 4 MiB PAX comment, written to disk.

    On disk deliberately: holding the fixture in a BytesIO would put the whole archive in memory and
    the measurement would report the test's own allocation instead of tarfile's.
    """
    with tarfile.open(path, mode="w") as tf:
        tf.format = tarfile.PAX_FORMAT
        info = tarfile.TarInfo("tiny")
        info.size = 1
        info.pax_headers = {"comment": "A" * (4 * 1024 * 1024)}
        tf.addfile(info, io.BytesIO(b"x"))
    return path


def test_a_pax_header_is_refused_without_being_allocated(tmp_path):
    """The peak must stay near the cap. A version that accounted AFTER reading raised correctly and
    still consumed 4 MiB, so raising is not sufficient evidence -- the memory is the claim."""
    bomb = _pax_bomb(tmp_path / "bomb.tar")
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    with pytest.raises(ArchiveTooLargeError) as caught:
        with open(bomb, "rb") as raw, CappedReader(raw, max_bytes=CAP) as capped:
            with tarfile.open(fileobj=capped, mode="r:") as tar:
                list(tar)
    peak = tracemalloc.get_traced_memory()[1] - baseline
    tracemalloc.stop()
    assert "refusing to continue" in str(caught.value)
    # Generous versus the 4 MiB the unclamped version held, tight versus a 4 MiB header.
    assert peak < 512 * 1024, (
        f"peak {peak} bytes means the header was allocated before the cap noticed")


def test_a_legitimate_archive_under_the_cap_still_reads(tmp_path):
    path = tmp_path / "ok.tar"
    with tarfile.open(path, mode="w") as tf:
        for name in ("a", "b"):
            info = tarfile.TarInfo(name)
            info.size = 3
            tf.addfile(info, io.BytesIO(b"abc"))
    with open(path, "rb") as raw, CappedReader(raw, max_bytes=CAP) as capped:
        with tarfile.open(fileobj=capped, mode="r:") as tar:
            got = {m.name: tar.extractfile(m).read() for m in tar}
    assert got == {"a": b"abc", "b": b"abc"}


def test_an_unbounded_read_request_cannot_bypass_the_cap(tmp_path):
    """read(-1) is the obvious hole: it would pull the whole file in before any accounting."""
    path = tmp_path / "big.bin"
    path.write_bytes(b"z" * (CAP * 4))
    with open(path, "rb") as raw, CappedReader(raw, max_bytes=CAP) as capped:
        with pytest.raises(ArchiveTooLargeError):
            capped.read(-1)


def test_seeking_backwards_does_not_refund_budget(tmp_path):
    """Counting is monotonic so re-reading a region cannot loop underneath the limit."""
    path = tmp_path / "d.bin"
    path.write_bytes(b"y" * (CAP * 2))
    quarter = CAP // 4
    with open(path, "rb") as raw, CappedReader(raw, max_bytes=CAP) as capped:
        capped.read(quarter)
        capped.seek(0)
        capped.read(quarter)          # same bytes again -- position rewound, budget did not
        assert capped.bytes_read == 2 * quarter, (
            "a backwards seek refunded budget, so re-reading could loop under the limit")
        capped.seek(0)
        with pytest.raises(ArchiveTooLargeError):
            capped.read(CAP)          # only ~CAP/2 remains, so this must fail


def test_a_nonpositive_cap_is_rejected():
    with pytest.raises(ValueError):
        CappedReader(io.BytesIO(b""), max_bytes=0)


def test_capped_tar_open_reads_a_legitimate_gzip_archive(tmp_path):
    """The one-line call form the six sites use. Covers "r:gz", which is what most of them pass."""
    path = tmp_path / "a.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("f")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"abc"))
    with capped_tar_open(path, "r:gz", max_bytes=CAP) as tar:
        assert {m.name for m in tar} == {"f"}


def test_capped_tar_open_bounds_a_streamed_payload(tmp_path):
    """A large member PAYLOAD is bounded, because payload bytes stream through the wrapper.

    Renamed from test_capped_tar_open_bounds_the_compressed_stream, which claimed more than it
    tested: it proved payload streaming is bounded and was then cited as evidence that
    decompression bombs generally are. A PAX HEADER is not streamed -- see the xfail below.
    """
    path = tmp_path / "b.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo("big")
        payload = b"\0" * (CAP * 8)      # compresses tiny, expands large
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    assert path.stat().st_size < CAP, "fixture must be small compressed, or it proves nothing"
    with pytest.raises(ArchiveTooLargeError):
        with capped_tar_open(path, "r:gz", max_bytes=64) as tar:
            for member in tar:
                tar.extractfile(member).read()


def test_capped_tar_open_does_not_leave_the_file_open(tmp_path):
    """The nested context managers must all unwind, including on the failure path."""
    path = tmp_path / "c.tar"
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo("f")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    captured = {}
    with capped_tar_open(path, "r:", max_bytes=CAP) as tar:
        captured["fileobj"] = tar.fileobj
    assert captured["fileobj"].closed, "the capped reader outlived the context manager"


def test_a_gzipped_pax_header_is_bounded(tmp_path):
    """Was ~16 MiB peak from a 4,237-byte archive; the header limit brings it to ~155 KiB.

    The bound sits on the DECLARED extension-header size, checked before tarfile reads the
    body, because that read happens before any per-member inspection could see it.
    """
    import tracemalloc
    path = tmp_path / "bomb.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        tf.format = tarfile.PAX_FORMAT
        info = tarfile.TarInfo("tiny")
        info.size = 1
        info.pax_headers = {"comment": "A" * (4 * 1024 * 1024)}
        tf.addfile(info, io.BytesIO(b"x"))
    assert path.stat().st_size < CAP, "the archive must be small COMPRESSED, or it proves nothing"
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    with pytest.raises(ArchiveTooLargeError):
        with capped_tar_open(path, "r:gz", max_bytes=CAP) as tar:
            list(tar)
    peak = tracemalloc.get_traced_memory()[1] - baseline
    tracemalloc.stop()
    assert peak < 512 * 1024, (
        f"peak {peak} bytes from a {path.stat().st_size}-byte archive: the compressed cap did not "
        f"bound the decompressed header")


def test_the_private_tarfile_hook_the_bound_relies_on_still_exists():
    """_proc_member is private. If a Python upgrade renames it, the override silently stops running
    and the bound disappears with the suite still green -- so its existence is asserted directly."""
    assert hasattr(tarfile.TarInfo, "_proc_member"), (
        "tarfile.TarInfo._proc_member is gone; header_limited_tarinfo no longer bounds anything")
    for name in ("XHDTYPE", "XGLTYPE", "SOLARIS_XHDTYPE", "GNUTYPE_LONGNAME", "GNUTYPE_LONGLINK"):
        assert hasattr(tarfile, name), f"tarfile.{name} is gone; the covered type set is incomplete"


def test_a_gnu_long_name_header_is_bounded_too(tmp_path):
    """Covering only x/g would leave the same allocation reachable through the GNU long-name types."""
    path = tmp_path / "gnu.tar"
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as tf:
        info = tarfile.TarInfo("n" * (2 * 1024 * 1024))
        info.size = 0
        tf.addfile(info)
    with pytest.raises(ArchiveTooLargeError):
        with capped_tar_open(path, "r:", max_bytes=8 * 1024 * 1024,
                             max_header_bytes=64 * 1024) as tar:
            list(tar)


def test_a_gnu_sparse_archive_is_refused(tmp_path):
    """Cycle-16 I1: GNU sparse 1.0 stores its sparse MAP in the body of the FOLLOWING member, which
    CPython parses and accumulates before returning it -- so the extension-header type check never
    sees it. Measured ~6.3 MB allocated from a 108 KB archive against a 1024-byte limit.

    Refused outright rather than bounded: a model checkpoint tar is never legitimately sparse, so
    there is no valid input to preserve, and bounding it would mean re-implementing three sparse-map
    parsers correctly.
    """
    import subprocess
    payload = tmp_path / "sparse.bin"
    with open(payload, "wb") as handle:
        handle.truncate(50_000_000)
    archive = tmp_path / "s.tar"
    result = subprocess.run(["tar", "-S", "-cf", str(archive), "-C", str(tmp_path), "sparse.bin"],
                            capture_output=True)
    if result.returncode != 0 or not archive.exists():
        pytest.skip("system tar cannot produce a sparse archive here")
    assert archive.stat().st_size < 1024 * 1024, "the fixture must be small, or it proves nothing"
    with pytest.raises(ArchiveTooLargeError, match="sparse"):
        with capped_tar_open(archive, "r:", max_bytes=8 * 1024 * 1024,
                             max_header_bytes=1024) as tar:
            list(tar)


def test_every_sparse_parser_hook_is_overridden():
    """The bound relies on private hooks. A Python upgrade that adds or renames a sparse-map parser
    would silently reopen the bypass with the suite still green."""
    from vla_pipeline.common.capped_reader import header_limited_tarinfo
    limited = header_limited_tarinfo(1024)
    for hook in ("_proc_gnusparse_00", "_proc_gnusparse_01", "_proc_gnusparse_10"):
        assert hasattr(tarfile.TarInfo, hook), f"tarfile.TarInfo.{hook} is gone"
        assert getattr(limited, hook) is not getattr(tarfile.TarInfo, hook), (
            f"{hook} is not overridden, so its unbounded map accumulation still runs")


def test_an_archive_larger_than_the_budget_is_refused_at_open(tmp_path):
    """cycle-16 S1 found this guard had NO test: a mutation removing the os.fstat extent check
    survived. capped_tar_open checks the file's size before handing it to tarfile, and only
    CappedReader.read(-1) was covered -- a different mechanism entirely.
    """
    path = tmp_path / "big.tar"
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo("payload")
        body = b"x" * (CAP * 4)
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
    assert path.stat().st_size > CAP, "the fixture must exceed the cap, or it proves nothing"
    with pytest.raises(ArchiveTooLargeError, match="exceeding"):
        with capped_tar_open(path, "r:", max_bytes=CAP) as tar:
            list(tar)


def test_the_old_gnu_sparse_typeflag_is_refused():
    """cycle-16 S1: the sparse-typeflag check also had no test of its own -- the sparse-1.0 hooks
    caught the realistic fixture first, so removing the typeflag check changed nothing observable.

    Exercised directly through _proc_member, because producing an OLD-style 'S' header from modern
    tar is impractical while the code path is still reachable from a hostile archive.
    """
    from vla_pipeline.common.capped_reader import header_limited_tarinfo
    limited = header_limited_tarinfo(1024)
    info = limited("sparse-member")
    info.type = tarfile.GNUTYPE_SPARSE
    info.size = 0
    with pytest.raises(ArchiveTooLargeError, match="sparse"):
        info._proc_member(None)
