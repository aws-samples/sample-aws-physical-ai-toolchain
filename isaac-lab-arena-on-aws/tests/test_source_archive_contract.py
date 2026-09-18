"""schema 3: the report must state which archive BYTES were evaluated.

A tree digest cannot establish that SimEval and Validate read the same archive -- it excludes
manifests, logs and hidden paths, so two different archives can produce the same tree digest.
HeadObject cannot either: it reports the object current when HEAD runs, not the object the job
had already downloaded. So a source replacement between a job's download and its HEAD was
undetectable, and the two jobs could agree while having read different bytes.

The premise this rests on was checked rather than assumed: SimEval receives the raw model.tar.gz
in its model input channel and extracts it itself, so it can hash the archive before extraction.
"""
from __future__ import annotations

import ast
import io
import pathlib
import tarfile
import tempfile

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PRODUCERS = {
    "arena gr00t": "entrypoints/eval/isaac_arena/gr00t/eval_entry.py",
    "libero gr00t": "entrypoints/eval/libero/gr00t/eval_entry.py",
    "libero molmoact2": "entrypoints/eval/libero/molmoact2/eval_entry.py",
    "libero openvla": "entrypoints/eval/libero/openvla/eval_entry.py",
}


def test_the_measurement_helper_is_deterministic_and_streams():
    from vla_pipeline.common.digest import measure_archive

    directory = pathlib.Path(tempfile.mkdtemp())
    archive = directory / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("config.json")
        payload = b'{"model_type": "gr00t"}'
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    first = measure_archive(str(archive))
    assert first == measure_archive(str(archive))
    assert len(first[0]) == 64 and first[1] > 0
    # Streamed rather than read whole: a checkpoint archive is far larger than memory.
    source = (_REPO_ROOT / "src/vla_pipeline/common/digest.py").read_text()
    assert "handle.read(1024 * 1024)" in source


def test_an_absent_or_empty_archive_cannot_be_measured():
    from vla_pipeline.common.digest import DigestError, measure_archive

    directory = pathlib.Path(tempfile.mkdtemp())
    with pytest.raises(DigestError, match="not a file"):
        measure_archive(str(directory / "absent.tar.gz"))
    empty = directory / "empty.tar.gz"
    empty.write_bytes(b"")
    with pytest.raises(DigestError, match="empty"):
        measure_archive(str(empty))


@pytest.mark.parametrize("label,relative", sorted(_PRODUCERS.items()))
def test_every_producer_measures_and_reports_the_archive(label, relative):
    """Measurement ORDER only. The assembly assertions this test used to carry are deleted.

    They matched literal implementation strings -- '"source_archive": dict(_SOURCE_ARCHIVE)' and
    'if _SOURCE_ARCHIVE' -- so they described one way of building the report rather than any property
    of it. A producer emitting a prefixed digest into a bare-hex field satisfied every one of them,
    which is why I6 survived with this test green; and they went red the moment a producer started
    CONSTRUCTING its identity instead of assembling it, even though that change is what fixed I6.

    What replaces them: tests/test_source_identity.py exercises the constructor against real trees,
    and the validator's field bindings reject a report whose parts disagree. Both observe values
    rather than source text.
    """
    source = (_REPO_ROOT / relative).read_text()
    assert "measure_archive" in source, f"{label} does not measure its archive"
    assert '"schema_version": 3,' in source, f"{label} still emits an older report version"


@pytest.mark.parametrize("label,relative", sorted(_PRODUCERS.items()))
def test_no_producer_still_emits_the_removed_identity_field(label, relative):
    source = (_REPO_ROOT / relative).read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert '"checksum_sha256"' not in stripped, (
            f"{label} still emits checksum_sha256, which S3 populates only for objects "
            f"uploaded with a checksum algorithm")


def test_the_measurement_happens_before_extraction():
    """Measuring after extraction would describe a tree, which is the thing that is not enough."""
    for label, relative in sorted(_PRODUCERS.items()):
        source = (_REPO_ROOT / relative).read_text()
        measure = source.index("measure_archive")
        extract = min(
            (source.index(token) for token in ("extractall", "tarfile.open")
             if token in source), default=None)
        assert extract is not None, f"{label} has no extraction site"
        assert measure < extract, f"{label} measures after extracting"


def test_the_arena_producer_refuses_to_publish_without_a_measurement(tmp_path, monkeypatch):
    """Behavioural: drive write_metrics with no resolved source and require it to raise.

    This previously asserted the literal error string "no source archive measurement was taken", so
    it went red when the message was reworded even though the guard was intact -- and it would have
    stayed GREEN if the guard had been deleted while the string survived in a comment. Now it calls
    the producer and observes the refusal.
    """
    import importlib.util
    import os
    import sys
    import types

    # The module refuses to import without a declared task (it will not fall back to a hardcoded
    # one), so supply the same fixture value the other Arena tests use.
    os.environ.setdefault("SM_HP_TASK_NAME", "fixture_task")
    path = _REPO_ROOT / _PRODUCERS["arena gr00t"]
    spec = importlib.util.spec_from_file_location("arena_entry_pub_guard", path)
    module = importlib.util.module_from_spec(spec)
    # write_metrics imports weights_digest from the staged sourcedir; the guard under test fires
    # after that import, so the stub must satisfy it.
    # monkeypatch.setitem, NOT sys.modules[...] = : a bare assignment leaked this stub into every
    # later test in the session, and test_source_identity.py started measuring archives with the
    # fake. The suite caught it, which is the argument for running the whole thing every time.
    stub = types.ModuleType("digest")
    stub.weights_digest = lambda path=None: "sha256:" + "0" * 64
    stub.measure_archive = lambda path: ("0" * 64, 1)
    monkeypatch.setitem(sys.modules, "digest", stub)
    for name in ("validator", "capped_reader", "source_identity"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    spec.loader.exec_module(module)

    assert module._RESOLVED is None, "no source was resolved, which is the condition under test"
    with pytest.raises(RuntimeError, match="source identity"):
        module.write_metrics({"success_rate": 1.0, "policy_type": "checkpoint",
                              "task_name": "fixture_task", "episodes": 3, "successes": 3},
                             str(tmp_path), str(tmp_path))


def test_an_already_extracted_channel_is_fatal():
    """It leaves no archive to measure, so the run cannot say which bytes it evaluated.

    This was a silent fallback that returned the directory and continued.
    """
    source = (_REPO_ROOT / _PRODUCERS["arena gr00t"]).read_text()
    assert "no model.tar.gz in" in source
    assert "using {input_path} as checkpoint dir directly" not in source


def test_both_arena_report_sites_name_exactly_one_source_identity():
    """Exactly one variant per report site, however the field is produced.

    This counted occurrences of the literal variant keys and expected one per site. Both sites now
    obtain them from the shared factory's report_identity_fields(), which emits exactly one variant by
    construction -- so counting literals found one where it wanted two, and failed against correct code.
    Assert the PROPERTY: neither site hand-writes a variant, and both spread the resolved object.
    """
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py").read_text()
    code = "\n".join(line.split("#")[0] for line in source.splitlines())

    spreads = code.count("report_identity_fields()")
    assert spreads >= 2, (
        f"only {spreads} Arena report site(s) derive identity from the resolved object; both the "
        f"checkpoint and the positive-control site must, or one can carry a variant disagreeing with "
        f"its own checkpoint")
    for hand_written in ('"tree_sha256"', '"source_archive"', '"source_snapshot"'):
        assert hand_written not in code, (
            f"an Arena report site still hand-writes {hand_written}. The factory owns the digest FORM, "
            f"and hand-assembly is how a prefixed digest reached a bare-hex field.")


def test_the_schema_requires_exactly_one_source_identity():
    """cycle-14 I4: source_archive was unconditionally required, so an HF-mode evaluation -- which
    downloads a snapshot and has no tarball to measure -- published {} and was rejected AFTER
    spending its whole inference budget.

    The requirement is a discriminated pair: both variants are individually optional in the key set,
    but exactly one must be PRESENT, so "evidence that may be absent is not evidence" still holds.

    Behavioural, because the previous version asserted the validator's literal control-flow lines
    ("if not has_archive and not has_snapshot:") and went red when that selection was rewritten to
    test key PRESENCE instead of truthiness -- a change that made the check strictly stronger. It
    would also have stayed green if the branches were deleted and the strings left in a comment.
    """
    from vla_pipeline.common.validator import (
        SOURCE_ARCHIVE_KEYS,
        SOURCE_SNAPSHOT_KEYS,
        ReportInvalid,
        validate_report,
    )
    from test_v2_validator import EXPECT, good_report, pipeline_report

    assert SOURCE_ARCHIVE_KEYS == {"sha256", "size_bytes"}
    assert SOURCE_SNAPSHOT_KEYS == {"repo_id", "resolved_commit", "tree_sha256"}

    # A coherent report of each shape is accepted in its own mode.
    validate_report(good_report(), **EXPECT)
    validate_report(pipeline_report(), **dict(EXPECT, pipeline_mode=True))

    # BOTH ABSENT: the report cannot say which bytes it read.
    r = good_report()
    del r["source_snapshot"]
    with pytest.raises(ReportInvalid, match="neither source_archive nor source_snapshot"):
        validate_report(r, **EXPECT)

    # BOTH PRESENT: two claims about one evaluation, and neither can be trusted.
    r = good_report()
    r["source_archive"] = {"sha256": "b" * 64, "size_bytes": 4096}
    with pytest.raises(ReportInvalid, match="BOTH"):
        validate_report(r, **EXPECT)

    # JUNK in the unused key is still a second claim -- truthiness selection ignored these.
    for junk in ("unavailable", {}, None, 0, []):
        r = good_report()
        r["source_archive"] = junk
        with pytest.raises(ReportInvalid):
            validate_report(r, **EXPECT)


def test_a_snapshot_must_agree_with_the_report_that_carries_it():
    """Review I1: each field was well-formed in isolation, so a report could name a repo, a commit
    and a tree digest belonging to THREE DIFFERENT THINGS and pass every check.

    A reviewer confirmed this behaviourally against both validator copies: they accepted
    independently mismatched repository, commit and tree digest in an otherwise coherent report.
    Format validation is not identity validation -- the fields have to be bound to each other.
    """
    from vla_pipeline.common.validator import ReportInvalid, validate_report

    from test_v2_validator import EXPECT, good_report

    # repo_id must be the report's own checkpoint.
    r = good_report()
    r["source_snapshot"] = dict(r["source_snapshot"], repo_id="someone-else/other-model")
    with pytest.raises(ReportInvalid, match="not the report's own checkpoint"):
        validate_report(r, **EXPECT)

    # resolved_commit must be the report's own checkpoint_revision.
    r = good_report()
    r["source_snapshot"] = dict(r["source_snapshot"], resolved_commit="f" * 40)
    with pytest.raises(ReportInvalid, match="disagrees with checkpoint_revision"):
        validate_report(r, **EXPECT)

    # tree_sha256 must be the digest the evaluator recomputed over what it loaded.
    r = good_report()
    r["source_snapshot"] = dict(r["source_snapshot"], tree_sha256="e" * 64)
    with pytest.raises(ReportInvalid, match="recomputed"):
        validate_report(r, **EXPECT)


def test_the_source_variant_must_match_the_execution_mode():
    """Either variant satisfied either mode, so a pipeline report could publish a snapshot and pass
    here while verify_checkpoint went on to require the archive -- an inconsistent contract that
    fails LATE, after the evaluation has been paid for."""
    from vla_pipeline.common.validator import ReportInvalid, validate_report

    from test_v2_validator import EXPECT, good_report, pipeline_report

    # A snapshot in pipeline mode: there IS a mounted archive, and it is what Validate cross-checks.
    r = good_report()
    r["checkpoint"] = "/opt/ml/input/data/model"
    r["checkpoint_revision"] = None
    with pytest.raises(ReportInvalid):
        validate_report(r, **dict(EXPECT, pipeline_mode=True))

    # An archive in HF mode: there is no mounted archive, so the measurement describes something else.
    r = pipeline_report()
    r["checkpoint"] = "org/model"
    with pytest.raises(ReportInvalid):
        validate_report(r, **EXPECT)


def test_a_snapshot_variant_must_pin_a_resolved_commit():
    """A branch or tag moves, so it cannot pin the bytes that were read."""
    source = (_REPO_ROOT / "src/vla_pipeline/common/validator.py").read_text()
    check = source[source.index('commit = snap["resolved_commit"]'):]
    check = check[:check.index("tree = snap")]
    assert "len(commit) == 40" in check, "resolved_commit is not checked as a full commit hash"
    assert "does not pin the bytes" in check, "the failure does not explain why a tag is refused"


# --- Validate side: the comparison that makes the measurement load-bearing ----------------

def test_validate_measures_its_own_archive_and_compares():
    """Recording a measurement proves nothing unless someone checks it.

    Every other link in the chain compares DERIVED values: the tree digest excludes manifests,
    logs and hidden paths so two archives can share one, and HeadObject describes the object
    current when HEAD runs rather than the object either job downloaded.
    """
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert "actual_sha, actual_size = measure_archive(archive_path)" in source
    assert 'reported = metrics["source_archive"]' in source
    assert "source archive mismatch" in source
    # And it must run BEFORE extraction, so a mismatched archive is never unpacked. Compared
    # against the CALL, not the def -- safe_extract is defined earlier in the file.
    assert source.index("measure_archive(archive_path)") < source.index(
        "safe_extract(archive, root)")


def test_the_mismatch_message_names_both_measurements():
    """A reader needs to know which side disagreed, not merely that something did."""
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    block = source[source.index("source archive mismatch"):]
    block = block[:block.index("print(")]
    for token in ("this job measured", "the evaluation reported",
                  "actual_sha", "actual_size",
                  "reported['sha256']", "reported['size_bytes']"):
        assert token in block, f"the mismatch message omits {token}"


def test_a_missing_archive_now_fails_with_a_named_cause():
    """It used to reach extraction and surface a bare FileNotFoundError."""
    from vla_pipeline.common.digest import DigestError, measure_archive

    directory = pathlib.Path(tempfile.mkdtemp())
    with pytest.raises(DigestError, match="checkpoint archive is not a file"):
        measure_archive(str(directory / "model.tar.gz"))


def test_the_validator_log_no_longer_claims_schema_v2():
    """Stale wording in a gate's own output is how a reader learns the wrong contract."""
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert "schema-v2" not in source
    assert "schema-v3" in source


# I5: THESE ARE TEXT-SHAPE ASSERTIONS, NOT PROTECTION.
#
# Astra disabled the version-content rejection by replacing its condition with `if False` and
# every test below still passed, because the error strings they match were untouched. They are
# useful as drift detectors -- they catch a Range= kwarg appearing, or head_object returning --
# but passing them is NOT evidence that the check rejects anything.
#
# The behavioural coverage lives in tests/test_validate_entry_gate.py, which runs staged Validate
# against a substituted version body, a changed VersionId and an absent body. One of those tests
# fails under the probe above; none of these do.
#
# --- The named S3 version must CONTAIN the evaluated bytes ---------------------------------

def test_validate_verifies_the_version_by_content_not_by_head():
    """A HEAD has no body, so it never tied the named version to any bytes.

    The old check asked whether the CURRENT object still carried the version and ETag the
    evaluation saw -- which describes the object at HEAD time. The version GET replaces "the
    current key still names this version" with "this version contains these bytes".
    """
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert "get_object(" in source
    assert "VersionId=reported_version" in source
    assert "S3 version content mismatch" in source
    # The unversioned HEAD must be gone: it is the thing that could not prove content.
    assert "head_object(" not in source


def test_the_version_body_is_streamed_whole():
    """A Range or PartNumber read would measure a slice and call it the version."""
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    block = source[source.index("VersionId=reported_version"):]
    block = block[:block.index("S3 version content mismatch")]
    assert "_body.read(1024 * 1024)" in block, "the body must be streamed"
    # Assert the KWARG forms, not the words: the surrounding comment names both deliberately.
    assert "Range=" not in block and "PartNumber=" not in block


def test_the_mismatch_message_names_the_version_and_both_measurements():
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    block = source[source.index("S3 version content mismatch"):]
    block = block[:block.index("print(")]
    for token in ("version_sha", "version_size", "reported_archive['sha256']",
                  "reported_archive['size_bytes']", "actual_version"):
        assert token in block, f"the mismatch message omits {token}"


def test_the_offline_stub_must_supply_a_body():
    """A stub without one would exercise a weaker check than production runs."""
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert "has no BodyPath" in source
    assert "would exercise a" in source


def test_the_version_check_happens_before_registration_evidence_is_emitted():
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert source.index("S3 version content mismatch") < source.index(
        "All checks passed: emit validated_metrics.json")


def test_every_external_backbone_download_records_what_it_resolved():
    """Cycle-16 C1: both N1.7 paths downloaded the VLM backbone with NO revision, and the processor
    guard checks only the repository name. That snapshot determines preprocessing, so it could change
    while the checkpoint archive, its digest, the evaluator image and the sourcedir stayed identical --
    and nothing in the receipt distinguished the two runs.

    Asserted over a glob of every file that downloads a backbone, so a third evaluator added later
    cannot reintroduce an unidentified external input.
    """
    root = _REPO_ROOT
    # Scoped to REPORT PRODUCERS. train/gr00t also downloads the backbone -- my glob found it -- but
    # a trainer writes no report, and its backbone influence is already carried by the resulting
    # checkpoint's weights digest. An evaluator's backbone influences the score directly with nothing
    # else recording it, which is the gap C1 named.
    downloaders = [p for p in sorted((root / "entrypoints/eval").rglob("*_entry.py"))
                   if "__pycache__" not in str(p) and "Cosmos" in p.read_text()]
    assert len(downloaders) >= 2, f"expected the known backbone downloaders, found {len(downloaders)}"
    for path in downloaders:
        body = path.read_text()
        assert "BACKBONE_COMMIT_FILE" in body, (
            f"{path.relative_to(root).as_posix()}: downloads the backbone without capturing the "
            f"commit it resolved, so the external input has no identity")
        assert '"backbone_identity"' in body, (
            f"{path.relative_to(root).as_posix()}: resolves the backbone commit but never reports it")
        assert "_BACKBONE_IDENTITY" in body


def test_every_report_site_in_a_backbone_evaluator_carries_the_identity():
    """One-of-N: my first edit anchored on a specific indentation and covered only one of the arena
    evaluator's two report sites. Counting both markers per file catches that; a presence check does
    not."""
    root = _REPO_ROOT
    for path in sorted((root / "entrypoints/eval").rglob("*_entry.py")):
        if "__pycache__" in str(path):
            continue
        body = path.read_text()
        if "Cosmos" not in body:
            continue
        sites = body.count('"schema_version": 3,')
        carried = body.count('"backbone_identity"')
        # >=, not ==. The Arena evaluator RE-BINDS metrics["backbone_identity"] after the report is
        # assembled, from the server audit recording what inference actually loaded -- because the field
        # previously carried the PRECACHE commit, so precache A with serving B published A. An equality
        # here failed against the fix for that false attribution. Every site must CARRY the field;
        # additional mentions are the correction.
        assert carried >= sites, (
            f"{path.relative_to(root).as_posix()}: {sites} report site(s) but only {carried} mention "
            f"backbone_identity, so some reports omit the external input's identity")
        # And the value must not come from the precache global at the point of report.
        if "consumed_backbone_identity" in body:
            # An early startup check is allowed; inspect the actual report
            # assignment rather than the first import or call in the file.
            tree = ast.parse(body)
            bindings = [
                node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "consumed_backbone_identity"
                and any(isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "backbone_identity"
                        for target in node.targets)
            ]
            report_lines = [
                node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Dict)
                and any(isinstance(key, ast.Constant) and key.value == "backbone_identity"
                        for key in node.keys)
            ]
            assert bindings and report_lines and max(bindings) > max(report_lines), (
                f"{path.relative_to(root).as_posix()}: the consumed-identity binding does not follow "
                f"the report assembly, so the precache value would survive into the published report")


# DELETED: test_a_converted_producer_actually_populates_its_snapshot.
#
# Every assertion in it matched an implementation string ("_SOURCE_SNAPSHOT.update(" in source,
# '"resolved_commit"' in source), and its skip condition was keyed to one too:
# `if "_SOURCE_SNAPSHOT" not in source: pytest.skip(...)`. When the producers stopped using that
# global, the test did not fail -- it SKIPPED, for all three, reporting nothing while appearing to
# pass. A skip condition tied to an implementation detail becomes silent vacuity the moment the
# implementation changes, which is worse than a red test.
#
# The property it was reaching for -- that an HF report carries a POPULATED snapshot with a resolved
# commit and a measured tree digest -- is now guaranteed by construction:
# ResolvedCheckpoint.__post_init__ cannot produce an instance with an unpopulated or malformed
# variant, and tests/test_source_identity.py exercises that against real directories.
