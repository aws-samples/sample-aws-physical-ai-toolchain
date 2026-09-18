"""Probes for the archive extraction bounds.

These cover limits the ordinary suite cannot express, because the failure is in how much
memory the tarfile machinery consumes before it yields anything a per-member check could
inspect.
"""

import pytest


@pytest.mark.xfail(strict=True, reason=(
    "Documents a PERMANENT property of CPython's tarfile, not an unfixed defect: _proc_pax reads "
    "an extended header fully into memory before yielding the TarInfo, so a per-member bound can "
    "never observe it. This calls tarfile DIRECTLY, bypassing capped_tar_open, so it must keep "
    "failing -- if it ever passes, tarfile changed and header_limited_tarinfo's rationale needs "
    "rechecking. The repo-side bound is tested in test_capped_reader.py."))
def test_a_pax_header_cannot_exceed_the_advertised_memory_bound():
    """The per-member archive bound cannot bound peak memory, and the gap is enormous.

    tarfile reads a PAX/GNU extended header fully into memory inside _proc_pax() BEFORE yielding
    the TarInfo, so a bound applied per member never observes it. Measured here: ONE 1-byte member
    carrying a 4 MiB comment peaks at ~16 MiB -- 512x a 32 KiB budget -- while the iteration sees
    exactly one tiny file. Counting members, or their sizes, cannot detect this by construction.

    The fix needs the fileobj handed to tarfile.open wrapped in a byte-capped reader, plus a bound
    on individual header size, across all five sites (safe_extract and four _bound_archive callers).
    """
    import io
    import tarfile
    import tracemalloc

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.format = tarfile.PAX_FORMAT
        info = tarfile.TarInfo("tiny")
        info.size = 1
        info.pax_headers = {"comment": "A" * (4 * 1024 * 1024)}
        tf.addfile(info, io.BytesIO(b"x"))

    budget = 32 * 1024
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    with tarfile.open(fileobj=io.BytesIO(buf.getvalue()), mode="r:") as tf:
        members = [m for m in tf]
    peak = tracemalloc.get_traced_memory()[1] - baseline
    tracemalloc.stop()

    assert len(members) == 1 and members[0].size == 1, (
        "the archive must present as a single tiny file, or the probe is measuring something else")
    assert peak <= budget, (
        f"peak {peak} bytes exceeds the {budget}-byte budget from ONE 1-byte member, so the "
        f"advertised bound does not hold")


def test_no_entrypoint_opens_an_archive_without_a_cap():
    """The countermeasure to fixing one of N sibling call sites.

    Six read sites existed and each needed the same change. Asserting over a GLOB with a required
    count of zero is the only form of this check that cannot pass while a site is missed -- naming
    the six would go stale the moment a seventh appeared.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((root / "entrypoints").rglob("*.py")):
        if "__pycache__" in str(path) or path.name == "capped_reader.py":
            continue          # the wrapper module IS the capped implementation
        for match in re.finditer(r"tarfile\.open\([^)]*\)", path.read_text()):
            if '"w' not in match.group(0):
                offenders.append(f"{path.relative_to(root).as_posix()}: {match.group(0)}")
    assert not offenders, (
        "these read an archive without a byte cap, so a PAX header can exhaust memory before any "
        "per-member check runs:\n" + "\n".join(offenders))


def test_no_entrypoint_opens_an_archive_without_a_cap():
    """The countermeasure to fixing one of N sibling call sites.

    Six read sites existed and every one needed the same change. Asserting over a GLOB with a
    required count of zero is the only form of this check that cannot pass while a site is missed --
    naming the six would go stale the moment a seventh appeared. Fixing some-but-not-all sibling
    paths is the most repeated defect in this review.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((root / "entrypoints").rglob("*.py")):
        if "__pycache__" in str(path) or path.name == "capped_reader.py":
            continue          # the wrapper module IS the capped implementation
        for match in re.finditer(r"tarfile\.open\([^)]*\)", path.read_text()):
            if '"w' not in match.group(0):
                offenders.append(f"{path.relative_to(root).as_posix()}: {match.group(0)}")
    assert not offenders, (
        "these read an archive without a byte cap, so a PAX header can exhaust memory before any "
        "per-member check runs:\n" + "\n".join(offenders))
