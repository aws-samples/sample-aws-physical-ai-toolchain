#!/usr/bin/env python3
"""Production validate_entry.py -- the load-bearing gate proof.

This is the CPU ProcessingStep code. It:
  1. Reads metrics.json from SimEval's ModelArtifacts (mounted at /opt/ml/processing/eval_output/)
  2. Reads pipeline-owned expectations from environment (populated from pipeline params)
  3. Runs the FULL schema-v3 validator (from vla_pipeline.common.validator)
  4. Performs digest cross-check (weights_digest_recomputed == manifest.weights_digest)
  5. ONLY if all checks pass: emits validated_metrics.json to /opt/ml/processing/output/

If ANY check fails: exits nonzero. No validated_metrics.json emitted. PropertyFile
read fails. Gate cannot evaluate. Pipeline stops.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile


def fail(msg):
    """Fail closed: print the violation and exit nonzero."""
    print(f"[validate] VALIDATION FAILED: {msg}", file=sys.stderr)
    print(f"[validate] VALIDATION FAILED: {msg}")
    sys.exit(1)


# S6: archive resource limits, applied alongside the path-safety checks below.
# Overridable so an unusually large supported checkpoint does not require a code
# change, but the DEFAULTS are what a run without configuration gets.
MAX_ARCHIVE_MEMBERS = int(os.environ.get("MAX_ARCHIVE_MEMBERS", "200000"))
# I4: this was 512 GiB, which the pipeline's default 100 GB volume (pipeline.py) can never
# hold -- so the guard could not fire before the disk filled, which is the failure it exists to
# prevent. Sized BELOW the volume instead: an archive that cannot fit is rejected from its
# headers rather than discovered part-way through extraction. Real checkpoints are tens of
# gigabytes, so this still rejects an archive that is not a checkpoint rather than bounding a
# legitimate one, and it remains overridable for an unusually large supported checkpoint.
MAX_ARCHIVE_BYTES = int(os.environ.get("MAX_ARCHIVE_BYTES", str(64 * 1024 ** 3)))


def safe_extract(tar, dest):
    """Extract every member of `tar` into `dest`, refusing any member that would
    escape `dest`.

    Rejects (fail-closed) absolute paths, ``..`` traversal, and symlink / hardlink
    / device / fifo members -- the classic tar path-traversal + link-escape
    attacks. Behavior-preserving for well-formed archives (identical output bytes);
    a malicious member aborts before any file is written.

    Self-contained on purpose: this file is delivered to a bare ProcessingStep
    container via stage_validate_code() and cannot import vla_pipeline.common.
    Works on all of Python 3.10-3.12 (does NOT rely on the 3.12 ``filter="data"``
    kwarg, which is absent/again-changing across versions).
    """
    dest_real = os.path.realpath(dest)
    prefix = dest_real + os.sep
    # S6: path safety alone does not bound COST. A well-formed archive with no traversal and no
    # links can still expand to far more than the disk holds, or carry millions of tiny members.
    #
    # I4: the limits used to be checked AFTER tar.getmembers(), which materialises the entire
    # member list -- so an archive carrying millions of members exhausted memory inside the very
    # call MAX_ARCHIVE_MEMBERS was supposed to bound. They are now enforced incrementally while
    # headers are read, so the count check fires at member N+1 and peak memory is bounded by the
    # cap rather than by the archive.
    members = []
    declared_bytes = 0
    for member in tar:
        members.append(member)
        if len(members) > MAX_ARCHIVE_MEMBERS:
            fail(f"archive declares more than {MAX_ARCHIVE_MEMBERS} members. A checkpoint does "
                 f"not contain that many files; refusing to extract (archive resource guard). "
                 f"Enforced while reading headers, so the rest of the archive was never read.")
        if member.isreg():
            declared_bytes += member.size
            if declared_bytes > MAX_ARCHIVE_BYTES:
                fail(f"archive declares more than {MAX_ARCHIVE_BYTES} expanded bytes, which "
                     f"exceeds what the job volume holds. Refusing to extract (archive "
                     f"resource guard)")
    for member in members:
        if member.issym() or member.islnk() or member.ischr() or member.isblk() or member.isfifo():
            fail(f"archive member {member.name!r} is a link/special file -- refusing "
                 f"to extract (tar link-escape guard)")
        target = os.path.realpath(os.path.join(dest_real, member.name))
        if target != dest_real and not target.startswith(prefix):
            fail(f"archive member {member.name!r} resolves outside {dest_real!r} -- "
                 f"refusing to extract (tar path-traversal guard)")
    # Members are validated (link-free, in-bounds) above -- that is the primary,
    # version-independent guard. On Python >=3.12 additionally apply the stdlib
    # "data" filter as defense-in-depth (and to satisfy the 3.14 default-filter
    # change); it is behavior-preserving here since the archive is already known
    # link-free.
    if sys.version_info >= (3, 12):
        tar.extractall(dest, filter="data")
    else:
        tar.extractall(dest)


def verify_checkpoint(metrics, checkpoint_dir):
    from capped_reader import DEFAULT_MAX_ARCHIVE_BYTES, capped_tar_open  # noqa: E402
    from digest import measure_archive, weights_digest

    archive_path = os.path.join(checkpoint_dir, "model.tar.gz")
    # schema 3: measure THIS job's archive and require it to equal what the evaluator measured.
    # Everything else in the chain compares derived values: the tree digest excludes manifests,
    # logs and hidden paths so two archives can share one, and HeadObject describes the object
    # current when HEAD runs rather than the object either job downloaded. This is the only
    # comparison that establishes the two jobs read the same bytes.
    reported = metrics["source_archive"]
    actual_sha, actual_size = measure_archive(archive_path)
    if actual_sha != reported["sha256"] or actual_size != reported["size_bytes"]:
        fail(f"source archive mismatch: this job measured sha256={actual_sha} "
             f"size={actual_size}, the evaluation reported sha256={reported['sha256']} "
             f"size={reported['size_bytes']}. The evaluated archive and the archive being "
             f"validated are not the same bytes, so the score does not describe what would "
             f"be promoted.")
    print(f"[validate] source archive matches the evaluation: sha256={actual_sha} "
          f"size={actual_size}")
    with tempfile.TemporaryDirectory(prefix="checkpoint-validation-", dir=checkpoint_dir) as root:
        with capped_tar_open(archive_path, "r:gz",
                                     max_bytes=DEFAULT_MAX_ARCHIVE_BYTES) as archive:
            safe_extract(archive, root)
        manifest_path = os.path.join(root, "checkpoint_manifest.json")
        with open(manifest_path) as source:
            manifest = json.load(source)
        if manifest != metrics["checkpoint_manifest"]:
            fail("checkpoint manifest differs from evaluation report")
        recomputed = weights_digest(root)
        if recomputed != manifest["weights_digest"]:
            fail("independent checkpoint digest differs from checkpoint manifest")
        if recomputed != metrics["weights_digest_recomputed_by_eval"]:
            fail("independent checkpoint digest differs from evaluation report")
        lineage = None
        lineage_path = os.path.join(root, "training_lineage.json")
        if os.path.isfile(lineage_path):
            with open(lineage_path) as source:
                lineage = json.loads(source.read(128 * 1024))
    return recomputed, lineage


def check_training_contract(manifest, lineage=None):
    """Compare the checkpoint's own provenance with THIS execution's training contract.

    Validate previously received only evaluation expectations, so nothing checked the
    checkpoint against what the execution actually asked FineTune to do: a checkpoint
    declaring a different training dose or a different dataset satisfied every check,
    and TrainSuite is exposed independently of Suite, so a manually started execution
    could train on one GR1 task and evaluate another while passing the embodiment
    checks.

    Returns a per-field record for the validated artifact: each entry carries a
    status (matched / recorded_only / unavailable) and, where it is not a comparison,
    the reason. Recording an uncheckable value is correct; faking a comparison across
    incompatible meanings rejected legitimate runs. The two paths
    are different provenance contracts and must be distinguished explicitly rather than
    by skipping checks: on the eval-only path the checkpoint was supplied from outside,
    so the execution's training parameters describe nothing about it.
    """
    train_path = os.environ.get("EXPECTED_TRAIN_PATH", "").strip()
    if train_path not in {"train", "eval_only"}:
        fail(f"EXPECTED_TRAIN_PATH must be 'train' or 'eval_only', got "
             f"{train_path!r}. The training-provenance contract cannot be selected "
             f"without it, and defaulting would silently skip the comparison.")
    if train_path == "eval_only":
        print("[validate] training contract: eval_only -- the checkpoint was supplied "
              "as input, so this execution's training parameters do not describe it "
              "and are NOT compared. Lineage must be established out of band.")
        return {"train_path": "eval_only", "fields": [], "matched": [],
                "verified": [], "attested": [],
                "reason": "checkpoint_training_is_outside_this_execution"}

    records = []

    def _expect(name):
        raw = os.environ.get(name, "").strip()
        if not raw:
            fail(f"{name} is empty; the training contract cannot be checked. Supply it "
                 f"or declare EXPECTED_TRAIN_PATH=eval_only.")
        return raw

    # Training dose: the manifest's recipe records what training actually ran. This one IS
    # directly comparable -- both sides are the same integer step count.
    expected_steps = _expect("EXPECTED_TRAIN_STEPS")
    actual_steps = manifest["train_recipe"]["max_steps"]
    if str(actual_steps) != expected_steps:
        fail(f"train_steps mismatch: this execution requested {expected_steps} but the "
             f"checkpoint's recipe records max_steps={actual_steps!r}. The evaluated "
             f"checkpoint was not trained under this execution's contract.")
    records.append({"field": "train_steps", "status": "matched",
                    "expected": expected_steps, "observed": str(actual_steps),
                    "basis": "checkpoint_recipe_matches_requested_steps"})

    dataset = manifest["dataset_manifest"]
    if dataset is None:
        fail("checkpoint declares no dataset_manifest, so the training dataset cannot "
             "be described at all.")

    if manifest.get("model_family") == "gr00t":
        from training_lineage import verify_training_lineage

        expected_suite = _expect("EXPECTED_TRAIN_SUITE")
        suites = json.loads(os.environ.get("VLA_SUITES_JSON", "{}"))
        if expected_suite not in suites:
            fail(f"Unknown expected training suite {expected_suite!r}")
        try:
            records.extend(verify_training_lineage(
                manifest, lineage, train_suite=expected_suite,
                source_parameter=_expect("EXPECTED_DATASET_S3URI"),
                revision_parameter=_expect("EXPECTED_DATASET_REVISION"),
                suite_spec=suites[expected_suite],
            ))
        except (ValueError, OSError, KeyError) as exc:
            fail(f"training lineage: {exc}")
        return {
            "train_path": "train", "fields": records,
            **{status: sorted(record["field"] for record in records
                              if record["status"] == status)
               for status in ("matched", "verified", "attested")},
        }

    # Dataset source and revision are NOT directly comparable with the pipeline's
    # DatasetS3Uri / DatasetRevision parameters, and comparing them literally rejected
    # manifests that real trainers produce:
    #
    #   GR00T HF          source "hf:<repo>"        revision = weights_digest(tree)
    #   OpenVLA BYO S3    source <s3 uri>           revision = weights_digest(tree)
    #   OpenVLA HF        source "hf:<repo>:<sub>"  revision = resolved 40-hex SHA
    #   MolmoAct2 cache   source <s3 cache prefix>  revision = resolved SHA
    #
    # while the pipeline passes an S3 URI (sometimes a resolution sentinel) and a revision
    # that may be a named reference. So one field carries an origin OR a locator, and the
    # other carries a content digest OR a commit. Joining them is a category error.
    #
    # Until the manifest carries separated fields (canonical source, requested reference,
    # resolved revision, content digest, subdirectory -- specified in
    # tmp/C5_dataset_contract_spec.md, a breaking manifest_version change), these are
    # RECORDED with an explicit status and reason rather than compared. Recording an
    # uncheckable value is correct; faking a comparison across incompatible meanings is not,
    # and it rejected legitimate runs.
    _source = dataset["source"]
    if not (isinstance(_source, str) and _source.strip()):
        fail(f"dataset_manifest.source is not a usable locator: {_source!r}")
    records.append({
        "field": "dataset_source", "status": "recorded_only",
        "reason": "producer_source_and_pipeline_parameter_have_different_meanings",
        "expected": os.environ.get("EXPECTED_DATASET_S3URI", ""), "observed": _source,
    })

    _revision = dataset["revision"]
    if not (isinstance(_revision, str) and _revision.strip()):
        fail(f"dataset_manifest.revision is not a usable reference: {_revision!r}")
    records.append({
        "field": "dataset_revision", "status": "recorded_only",
        "reason": "producer_revision_may_be_a_content_digest_not_a_named_reference",
        "expected": os.environ.get("EXPECTED_DATASET_REVISION", ""),
        "observed": _revision,
    })

    # EXPECTED_TRAIN_SUITE was supplied by the pipeline and never read, so an incorrect
    # value was silently accepted. It cannot be compared yet either: no producer records
    # the training suite in the manifest. Recorded so the gap is visible rather than
    # invisible.
    records.append({
        "field": "train_suite", "status": "unavailable",
        "reason": "no_producer_records_the_training_suite_in_the_manifest",
        "expected": os.environ.get("EXPECTED_TRAIN_SUITE", ""), "observed": None,
    })

    print(f"[validate] training contract: train -- "
          f"{[(r['field'], r['status']) for r in records]}")
    return {"train_path": "train", "fields": records,
            "matched": sorted(r["field"] for r in records
                              if r["status"] == "matched")}


def archive_sha256(path):
    """SHA-256 of the COMPLETE compressed archive, hashed incrementally.

    Deliberately distinct from `weights_digest`, which excludes the manifest, logs, lock
    files and hidden paths (including hidden dose checkpoints). That digest is the right
    tool for comparing checkpoint CONTENT, but it is not an exact digest of the object
    being registered -- two different archives can share a weights digest. Registration
    needs to name the bytes it points at.

    Never loads the archive into memory: Validate runs on a small instance and VLA
    checkpoints are large.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Streaming/publication sizing. Validate runs on ml.m5.large, and S3's single-PUT limit is
# 5 GiB, so a whole-archive buffer was both a memory hazard and an outright ceiling.
_CHUNK_BYTES = 8 * 1024 * 1024          # read/hash granularity
_MULTIPART_THRESHOLD = 64 * 1024 * 1024  # above this, publish in parts
_PART_BYTES = 64 * 1024 * 1024           # 10,000-part limit -> ~640 GiB ceiling


def _stream_digest_of_object(client, bucket, key):
    """SHA-256 of an existing object, streamed. Never buffers the whole body."""
    import hashlib

    digest = hashlib.sha256()
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        for chunk in iter(lambda: body.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    finally:
        body.close()
    return digest.hexdigest()


def _publish_conditionally(client, bucket, key, path, content_type, expected_digest):
    """Create an object from a FILE, conditionally, without buffering it.

    Uses a conditional create (If-None-Match) so two concurrent publications cannot
    silently overwrite one another, and so the bucket's own create-only policy is satisfied.
    Archives above the multipart threshold are sent in parts and completed conditionally --
    a single PutObject cannot carry more than 5 GiB, and reading the archive into memory to
    do it was a hazard on Validate's small instance.

    On a key that already exists, the existing content is verified by STREAMING its digest
    and comparing: an identical object is an idempotent success (a retried step), a
    different one is a hard failure. Treating "already there" as success without checking is
    how a promoted artifact could come to differ from the validated one.

    CopyObject is not used: S3's conditional-write enforcement is incompatible with it.
    """
    size = os.path.getsize(path)

    def _already_present():
        actual = _stream_digest_of_object(client, bucket, key)
        if actual != expected_digest:
            fail(f"s3://{bucket}/{key} already exists with DIFFERENT content "
                 f"(sha256={actual}, expected {expected_digest}). The protected namespace "
                 f"is create-only, so this means two runs disagree about the same "
                 f"content-addressed key. Refusing to register.")
        return "already_present_identical"

    if size <= _MULTIPART_THRESHOLD:
        try:
            with open(path, "rb") as handle:
                client.put_object(Bucket=bucket, Key=key, Body=handle,
                                  ContentType=content_type,
                                  IfNoneMatch="*")
            print(f"[validate] conditional create succeeded: PutObject IfNoneMatch=* "
                  f"s3://{bucket}/{key}", flush=True)
            return "created"
        except Exception as exc:
            if _is_precondition_failure(exc):
                return _already_present()
            fail(f"could not publish s3://{bucket}/{key}: {exc}")

    upload = client.create_multipart_upload(
        Bucket=bucket, Key=key, ContentType=content_type)
    upload_id = upload["UploadId"]
    parts = []
    try:
        with open(path, "rb") as handle:
            for number, chunk in enumerate(
                    iter(lambda: handle.read(_PART_BYTES), b""), start=1):
                result = client.upload_part(
                    Bucket=bucket, Key=key, UploadId=upload_id,
                    PartNumber=number, Body=chunk)
                parts.append({"ETag": result["ETag"], "PartNumber": number})
        client.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
            IfNoneMatch="*")
        print(f"[validate] conditional create succeeded: CompleteMultipartUpload "
              f"IfNoneMatch=* s3://{bucket}/{key}", flush=True)
        return "created_multipart"
    except Exception as exc:
        # A failed or superseded multipart upload must not linger as billable storage, and
        # must not be mistaken for a publication.
        try:
            client.abort_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id)
        except Exception as abort_exc:
            # I11: this was `pass`. The abort needs s3:AbortMultipartUpload, which the
            # validation role did not hold, so every abort failed silently and left billable
            # parts behind -- the cleanup looked implemented while never once succeeding.
            # Reported, not raised: the original publication failure below is the real error
            # and must not be masked by a cleanup problem.
            print(f"[validate] WARNING: could not abort the multipart upload for "
                  f"s3://{bucket}/{key} (upload {upload_id}): {abort_exc}. Incomplete parts "
                  f"may remain billable; the bucket lifecycle rule expires them.", flush=True)
        if _is_precondition_failure(exc):
            return _already_present()
        fail(f"could not publish s3://{bucket}/{key} in parts: {exc}")


def _ensure_conditional_write_support():
    """Require the prepackaged SDK before constructing a publishing client."""
    import boto3
    import botocore.session

    model = botocore.session.get_session().get_service_model("s3")
    for operation in ("PutObject", "CompleteMultipartUpload"):
        shape = model.operation_model(operation).input_shape
        if shape is None or "IfNoneMatch" not in shape.members:
            fail(f"Validate SDK cannot send {operation}.IfNoneMatch. Restage the "
                 "validator with stage_validate_code(), which embeds the pinned "
                 "SDK before execution. Runtime dependency installation is disabled.")
    import botocore
    print(f"[validate] packaged SDK ready: boto3={boto3.__version__} "
          f"botocore={botocore.__version__} "
          f"bundle_sha256={os.environ.get('VLA_VALIDATION_SDK_SHA256', 'absent')}",
          file=sys.stderr, flush=True)
    return boto3


def _is_precondition_failure(exc):
    """Whether S3 rejected a conditional create because the key already exists."""
    response = getattr(exc, "response", None)
    code = (response or {}).get("Error", {}).get("Code")
    return code in {"PreconditionFailed", "ConditionalRequestConflict"}


def _put_immutable(client, bucket, key, body, content_type):
    """Create a SMALL object (the attestation) that must never already exist differently.

    The artifact path uses _publish_conditionally, which streams; this stays byte-based
    because the attestation is a small JSON document.
    """
    import hashlib

    try:
        client.put_object(Bucket=bucket, Key=key, Body=body,
                          ContentType=content_type,
                          IfNoneMatch="*")
        print(f"[validate] conditional create succeeded: PutObject IfNoneMatch=* "
              f"s3://{bucket}/{key}", flush=True)
        return "created"
    except Exception as exc:  # botocore raises ClientError; keep this import-free
        code = getattr(getattr(exc, "response", {}), "get", lambda *_: {})("Error") or {}
        if code.get("Code") not in {"PreconditionFailed", "ConditionalRequestConflict"}:
            fail(f"could not publish s3://{bucket}/{key}: {exc}")
    # The key exists. Verify it holds exactly these bytes.
    try:
        existing = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        fail(f"s3://{bucket}/{key} already exists but could not be read back to confirm "
             f"it matches what this run validated: {exc}")
    if hashlib.sha256(existing).hexdigest() != hashlib.sha256(body).hexdigest():
        fail(f"s3://{bucket}/{key} already exists with DIFFERENT content. The protected "
             f"namespace is create-only, so this means two runs disagree about the same "
             f"content-addressed key. Refusing to register.")
    return "already_present_identical"


def promote_validated_artifact(archive_path, attestation):
    """Publish the validated bytes and their attestation, then return the registration URI.

    RegisterModel used to point at the FineTune artifact's plain, unversioned URI, so the
    registered package resolved to whatever occupied that key at resolution time rather
    than the bytes that passed validation. Validate recorded the source identity but never
    promoted the object, and the model-package API has no VersionId field to pin, so a
    protected never-replaced key is what gives ModelDataUrl a stable meaning.

    A failed artifact or attestation publication must prevent a successful receipt, which
    is why this runs BEFORE validated_metrics.json is written and fails closed.
    """
    import boto3

    # Offline test publication, following the same explicit-stub convention as
    # _VALIDATE_S3_HEAD_STUB above. The resulting promotion record is SELF-IDENTIFYING: it
    # carries file:// URIs and local_test_publication=True, so a report produced this way
    # cannot be mistaken downstream for a real promotion, and SageMaker would reject the
    # URI outright rather than registering something unverified.
    _local_dir = os.environ.get("_VALIDATE_PROMOTION_LOCAL_DIR", "").strip()

    bucket = os.environ.get("TRUST_BUCKET", "").strip()
    if not bucket and not _local_dir:
        fail("TRUST_BUCKET is not set, so the validated artifact cannot be promoted. "
             "Registering the unpromoted source URI would let the registered bytes "
             "differ from the validated ones; refusing to continue. Deploy the "
             "component's trust boundary (infra/trust_boundary.tf).")

    execution_id = os.environ.get("PIPELINE_EXECUTION_ID", "").strip() or "unknown"
    # The archive is NOT read into memory: it is streamed for hashing and for upload.
    # Validate runs on ml.m5.large, and a single PutObject cannot exceed 5 GiB.
    archive_digest = attestation["promoted_artifact"]["archive_sha256"]
    artifact_key = f"artifacts/v1/sha256/{archive_digest}/model.tar.gz"
    # allow_nan=False: refuse to EMIT a non-finite token rather than writing evidence that a
    # conforming parser cannot read. This is the sealed attestation, so failing here is the
    # correct outcome -- a receipt nobody can parse is worse than no receipt.
    attestation_bytes = json.dumps(attestation, indent=2, sort_keys=True,
                                   allow_nan=False).encode()
    attestation_digest = archive_sha256_of_bytes(attestation_bytes)
    attestation_key = (f"evidence/v1/{execution_id}/{attestation_digest}/"
                       f"validated_metrics.json")

    if _local_dir:
        artifact_local = os.path.join(_local_dir, artifact_key)
        os.makedirs(os.path.dirname(artifact_local), exist_ok=True)
        # Copied in chunks rather than read into memory, matching the real path.
        with open(archive_path, "rb") as source, open(artifact_local, "wb") as target:
            shutil.copyfileobj(source, target, _CHUNK_BYTES)
        attestation_local = os.path.join(_local_dir, attestation_key)
        os.makedirs(os.path.dirname(attestation_local), exist_ok=True)
        with open(attestation_local, "wb") as handle:
            handle.write(attestation_bytes)
        promotion = {
            "model_uri": f"file://{artifact_local}",
            "archive_sha256": archive_digest,
            "attestation_uri": f"file://{attestation_local}",
            "attestation_sha256": attestation_digest,
            # N1: SageMaker MetricsSource.ContentDigest requires the ALGORITHM PREFIX --
            # its service model pattern is [Ss][Hh][Aa]256:[0-9a-fA-F]{64} and bare hex is
            # rejected at the API. The raw hex above is kept because content-addressed
            # keys and digest comparisons use it; this field exists to be passed to the API.
            "attestation_content_digest": f"sha256:{attestation_digest}",
            "artifact_publication": "created",
            "attestation_publication": "created",
            "local_test_publication": True,
        }
        print(f"[validate] LOCAL TEST publication -> {promotion['model_uri']}")
        return promotion

    # Before the client exists: an upgrade after construction would leave this client bound to the
    # old botocore. The trust bucket denies an unconditional write, so this cannot be optional.
    # REBOUND from the return value: the module imported earlier in this function is the pre-upgrade
    # one, and a reload cannot change a name already bound here.
    boto3 = _ensure_conditional_write_support()
    client = boto3.client("s3")
    artifact_state = _publish_conditionally(
        client, bucket, artifact_key, archive_path, "application/gzip", archive_digest)
    attestation_state = _put_immutable(
        client, bucket, attestation_key, attestation_bytes, "application/json")

    promotion = {
        "model_uri": f"s3://{bucket}/{artifact_key}",
        "archive_sha256": archive_digest,
        "attestation_uri": f"s3://{bucket}/{attestation_key}",
        "attestation_sha256": attestation_digest,
        # N1: SageMaker MetricsSource.ContentDigest requires the ALGORITHM PREFIX --
        # its service model pattern is [Ss][Hh][Aa]256:[0-9a-fA-F]{64} and bare hex is
        # rejected at the API. The raw hex above is kept because content-addressed
        # keys and digest comparisons use it; this field exists to be passed to the API.
        "attestation_content_digest": f"sha256:{attestation_digest}",
        "artifact_publication": artifact_state,
        "attestation_publication": attestation_state,
    }
    print(f"[validate] promoted artifact -> {promotion['model_uri']} ({artifact_state})")
    print(f"[validate] attestation -> {promotion['attestation_uri']} "
          f"({attestation_state})")
    return promotion


def archive_sha256_of_bytes(body):
    import hashlib

    return hashlib.sha256(body).hexdigest()


def main():
    # C1 (cycle 15): main() extracts the SimEval archive too, and the import previously sat
    # only in verify_checkpoint() -- so every production Validate raised NameError here.
    # Function-local rather than module-level because the module is not on sys.path until the
    # staging bootstrap has run.
    from capped_reader import DEFAULT_MAX_ARCHIVE_BYTES, capped_tar_open  # noqa: E402
    eval_output_dir = os.environ.get("EVAL_OUTPUT_DIR", "/opt/ml/processing/eval_output")
    output_dir = os.environ.get("OUTPUT_DIR", "/opt/ml/processing/output")
    os.makedirs(output_dir, exist_ok=True)
    # Execution trace. Plain prints: `=== INIT` / `>>>` / `<<<` / `=== EXECUTION_SUCCESS`
    # are greppable breadcrumbs that localise a failure in CloudWatch. A `>>>` with no
    # matching `<<<` means the run died INSIDE that block. Each `<<<` states WHAT IT
    # PRODUCED, not merely that it passed.
    print(f"[validate] === INIT: starting validation, eval_output={eval_output_dir}, "
          f"output={output_dir}", flush=True)

    # --- Extract model.tar.gz if present ---
    tarball = os.path.join(eval_output_dir, "model.tar.gz")
    if os.path.exists(tarball):
        print(f"[validate] extracting {tarball}")
        with capped_tar_open(tarball, "r:gz",
                                     max_bytes=DEFAULT_MAX_ARCHIVE_BYTES) as tar:
            safe_extract(tar, eval_output_dir)

    # --- Read metrics.json ---
    metrics_path = os.path.join(eval_output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        fail(f"metrics.json not found at {metrics_path}")

    with open(metrics_path) as f:
        # cycle-16 I6: this used ordinary json.load, which ACCEPTS NaN and Infinity -- both
        # illegal in standard JSON. The shared module already ships parse_report() precisely
        # to reject them, and Validate, the one step whose output is immutable evidence, was
        # the one place not using it. A report carrying aux_metrics={"diagnostic": NaN}
        # validated cleanly and the receipt then serialised a token no conforming parser can
        # read.
        # Imported HERE, not with validate_report further down: a function-local import binds
        # when it executes, so an import below this line leaves parse_report unbound at this
        # point. cycle-15 C1 was the same mistake across functions; this is it within one.
        from validator import parse_report  # noqa: E402
        metrics = parse_report(f.read())

    print(f"[validate] Read metrics.json: success_rate={metrics.get('success_rate')}")

    # --- Read pipeline-owned expectations from environment (REQUIRED) ---
    # The gate validates the report AGAINST these. A missing expectation must
    # NEVER fall back to an invented default (trials=2 / seed=1000 /
    # suite=libero_spatial / task_ids=range(10)) -- that would let the gate BLESS
    # THE WRONG EXPERIMENT (a report validated against fabricated expectations).
    # pipeline.py always populates them from the Validate step's parameters
    # (EXPECTED_EVAL_SEED/TRIALS/TASK_IDS/SUITE); a missing one is a
    # misconfiguration, so fail loud naming the value + where it comes from.
    def _require_expectation(name):
        v = os.environ.get(name, "").strip()
        if not v:
            fail(f"{name} is not set -- the gate refuses to validate against an "
                 f"invented expectation. It is populated from the Validate step "
                 f"parameter in pipeline.py; an empty value is a misconfiguration.")
        return v

    def _require_digest_image(name):
        """An image reference that names BYTES, not a moving pointer.

        A tag is a pointer that can be republished, and an "immutable tag" prevents overwriting the
        tag -- it does not establish which image ran, because a launcher can pass an older tag and the
        receipt looks identical. Refused rather than downgraded to a warning: this component exists to
        make a score attributable, and a receipt naming a pointer cannot do that.
        """
        v = _require_expectation(name)
        if "@sha256:" not in v:
            fail(f"{name}={v!r} is not digest-form. The attestation must name the image BYTES that "
                 f"produced this evidence, not a tag: a tag can be republished, and a launcher "
                 f"passing an older tag yields an identical-looking receipt. Three Arena runs "
                 f"executed a day-old image while being attributed to current source exactly this "
                 f"way. Pass the reference as <repo>@sha256:<64 hex>.")
        ref, _, digest = v.partition("@sha256:")
        if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest.lower()):
            fail(f"{name} carries a malformed digest {digest!r}: expected 64 lowercase hex "
                 f"characters after '@sha256:'.")
        return v, f"sha256:{digest.lower()}"

    def _require_suite_specs():
        """The resolved suite table, or FAIL CLOSED.

        `json.loads(os.environ.get("VLA_SUITES_JSON", "{}"))` would fail OPEN: an
        empty table is falsy, so validate_report silently reverts the suite
        allowlist to its literal fallback AND skips both coherence gates, after
        which this step prints "PASSED". The table is embedded by
        stage_validate_code(); an absent or empty one is a misconfiguration, and a
        degraded gate must never look like a passing one.
        """
        raw = os.environ.get("VLA_SUITES_JSON", "").strip()
        if not raw or raw == "{}":
            fail("VLA_SUITES_JSON is empty -- the suite allowlist and the "
                 "suite/task + suite/embodiment coherence gates would silently "
                 "degrade to a vacuous pass. It is embedded by "
                 "stage_validate_code(); an empty value is a misconfiguration.")
        try:
            specs = json.loads(raw)
        except ValueError as e:
            fail(f"VLA_SUITES_JSON is not valid JSON: {e}")
        if not isinstance(specs, dict) or not specs:
            fail(f"VLA_SUITES_JSON must be a non-empty object, got {type(specs).__name__}")
        return specs

    expected_seed = _require_expectation("EXPECTED_EVAL_SEED")
    expected_trials = _require_expectation("EXPECTED_EVAL_TRIALS")
    expected_task_ids_str = _require_expectation("EXPECTED_EVAL_TASK_IDS")
    expected_suite = _require_expectation("EXPECTED_SUITE")
    expected_family = _require_expectation("EXPECTED_MODEL_FAMILY")
    actual_family = metrics.get("model_family")
    if actual_family != expected_family:
        fail(f"model_family mismatch: expected={expected_family!r} got={actual_family!r}")
    # `eval_backend` is NOT read from the report here. The report cannot legally carry
    # it -- the schema's strict top-level key check forbids it, so a producer that
    # emitted one would have its report rejected -- which meant this read returned ""
    # every time and validated_metrics.json advertised an empty backend on every valid
    # run. It is derived from the suite's simulator contract instead, once the resolved
    # suite specs are available below, so the recorded backend is the one the suite
    # actually declares rather than whatever a producer chose to claim.

    # --- Check 1: Schema version must be integer 3 ---
    schema_ver = metrics.get("schema_version")
    if type(schema_ver) is not int or schema_ver != 3:
        fail(f"schema_version must be integer 3, got {schema_ver!r}. "
             "Version 2 reports carry no measured archive identity.")

    # --- Check 2: Pipeline-owned expectation matching (ALL backends) ---
    # NO backend skips this check. The eval entry must write correct
    # values. expected_seed/expected_trials are REQUIRED (checked above), so this
    # is unconditional -- never silently skipped on an empty expectation.
    actual_seed = str(metrics.get("eval_seed", ""))
    if actual_seed != expected_seed:
        fail(f"eval_seed mismatch: expected={expected_seed} got={actual_seed}")
    print(f"[validate] eval_seed matches: {expected_seed}")

    actual_trials = str(metrics.get("eval_trials", metrics.get("num_trials_per_task", "")))
    if actual_trials != expected_trials:
        fail(f"eval_trials mismatch: expected={expected_trials} got={actual_trials}")
    print(f"[validate] eval_trials matches: {expected_trials}")

    policy_type = metrics.get("policy_type")
    if policy_type != "checkpoint":
        fail(f"policy_type must be 'checkpoint' for registration, got {policy_type!r}")

    # --- Check 2c: episodes must be > 0 ---
    episodes = metrics.get("episodes", 0)
    if episodes == 0:
        fail("episodes=0 means no evaluation actually ran -- cannot validate empty results")

    # --- Check 4: Weighted mean consistency ---
    # Real eval entries emit per_task records {task_id, task, episodes, success_rate}
    # (NOT successes/trials). Recompute the episode-weighted mean
    # from those and cross-check against the reported top-level success_rate. This is
    # the sabotage-detection leg (plumbing check); the old successes/trials lookup
    # always summed to 0 and silently skipped it.
    per_task = metrics.get("per_task", [])
    if per_task:
        total_episodes = sum((t.get("episodes", 0) or 0) for t in per_task)
        # Legacy fallback: only if a producer ever emits integer successes/trials.
        total_successes_int = sum((t.get("successes", 0) or 0) for t in per_task)
        total_trials_val = sum((t.get("trials", 0) or 0) for t in per_task)
        weighted_mean = None
        if total_episodes > 0:
            weighted_mean = sum((t.get("success_rate", 0.0) or 0.0) * (t.get("episodes", 0) or 0)
                                for t in per_task) / total_episodes
        elif total_trials_val > 0:
            weighted_mean = total_successes_int / total_trials_val
        else:
            print("[validate] WARNING: per_task present but has neither 'episodes' nor "
                  "'trials' -- weighted-mean consistency check SKIPPED")
        if weighted_mean is not None:
            reported_rate = metrics.get("success_rate", -1)
            if abs(weighted_mean - reported_rate) > 1e-4:
                fail(f"weighted mean of per_task ({weighted_mean:.6f}) "
                     f"!= reported success_rate ({reported_rate:.6f})")
            print(f"[validate] weighted mean consistency PASSED: {weighted_mean:.4f}")

    # --- Check 5: Full schema-v3 validator (ALL backends, mandatory) ---
    # No backend skips this. Missing validator = hard fail (not "basic checks only").
    # Arena backends are validated with arena-specific expectations.
    _validator_available = False
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from validator import validate_report  # noqa: E402
        _validator_available = True
    except ImportError as _imp_err:
        # 1b fix: FAIL CLOSED. A missing validator must NEVER silently degrade to
        # "basic checks only" -- that degrade path is exactly what made the
        # gated-registry claim aspirational. The runner embeds validator.py into
        # this file via stage_validate_code() (a base64 bootstrap writes
        # /tmp/validator.py and adds it to sys.path). If the import still fails the
        # trust chain is broken, so we refuse to validate rather than pass weakly.
        fail(f"validator.py could not be imported ({_imp_err}). The full schema-v3 "
             "validator is REQUIRED (fail-closed); the runner must embed it via "
             "stage_validate_code(). Refusing to degrade to basic checks.")

    if _validator_available:
        try:
            # Parse expected_task_ids. "all"/"" = the canonical full task set for
            # the suite (pipeline-owned expectation, NOT derived from the report).
            # Mapping "all"->[] is wrong under strict validation: the validator
            # requires report.task_ids == sorted(expected), and [] forces
            # episodes==0 (impossible). Expand to the real suite task list.
            # the canonical suite->task_ids map now comes from the
            # registry, injected as VLA_SUITE_CANONICAL_TASK_IDS_JSON by
            # stage_validate_code() -- no longer a hardcoded dict here.
            suite_task_ids = json.loads(os.environ["VLA_SUITE_CANONICAL_TASK_IDS_JSON"])
            if expected_suite not in suite_task_ids:
                fail(f"suite {expected_suite!r} has no canonical task_ids")
            canonical_ids = suite_task_ids[expected_suite]
            if (not isinstance(canonical_ids, list) or not canonical_ids
                    or any(type(task_id) is not int or task_id < 0 for task_id in canonical_ids)
                    or len(set(canonical_ids)) != len(canonical_ids)):
                fail(f"invalid canonical task IDs for suite {expected_suite!r}")
            if expected_task_ids_str == "all":
                task_ids = canonical_ids
            else:
                try:
                    task_ids = json.loads(expected_task_ids_str)
                except ValueError as exc:
                    fail(f"invalid task selection: {exc}")
                if (not isinstance(task_ids, list) or not task_ids
                        or any(type(task_id) is not int for task_id in task_ids)
                        or len(set(task_ids)) != len(task_ids)
                        or not set(task_ids) <= set(canonical_ids)):
                    fail(f"invalid task selection for suite {expected_suite!r}: {task_ids!r}")

            suite_specs = _require_suite_specs()
            validate_report(
                metrics,
                expected_suite=expected_suite,
                expected_trials=int(expected_trials),
                expected_eval_seed=int(expected_seed),
                expected_task_ids=task_ids,
                # Per-family schemas embedded by stage_validate_code
                # from steps/<family>/defaults.json (validator rejects unknown
                # families / validates manifest.input_config + provenance).
                family_schemas=json.loads(os.environ.get("VLA_FAMILY_SCHEMAS_JSON", "{}")),
                # Resolved suite manifests (registry.suites_json()), embedded by
                # stage_validate_code. Supplies the authoritative suite allowlist
                # and the suite/task + suite/embodiment coherence gates.
                #
                # FAIL CLOSED on an empty table: `{}` is falsy, so the allowlist
                # would silently revert to validator.VALID_SUITES and BOTH coherence
                # gates would be skipped -- after which this step prints "PASSED".
                # That is the same silent-degradation the missing-validator and
                # missing-EXPECTED_* paths already refuse (see Check 5 / above).
                suite_specs=suite_specs,
                # Pipeline-owned family version selector (n16/n17) -- picks WHICH
                # family_overrides entry the embodiment gate compares against.
                expected_family_version=_require_expectation(
                    "EXPECTED_FAMILY_VERSION"),
                pipeline_mode=True,
                # Arena now requests a fixed number of COMPLETE episodes
                # (--num_episodes), so the shared validator's per-task episode
                # equality is both enforceable and required. Arena used to select
                # "steps" here, which DISABLED that check -- the one gate that would
                # have caught an evaluated sample differing from the requested one.
                # This is wrapper-owned policy: never derive the enforcement mode
                # from the report being validated.
                expected_budget_type="fixed_trials",
            )
            suite_spec = suite_specs.get(expected_suite, {})
            # Derive the evaluation backend from the suite's own simulator declaration.
            # The registry requires every suite manifest to name a simulator, so an
            # absent value means the resolved specs are not the ones this check assumes
            # and must not be papered over with an empty string.
            eval_backend = suite_spec.get("simulator")
            if not (isinstance(eval_backend, str) and eval_backend.strip()):
                fail(f"suite {expected_suite!r} declares no simulator in the resolved "
                     f"suite specs, so the evaluation backend cannot be established "
                     f"(got {eval_backend!r}). Refusing to record an empty backend.")
            eval_backend = eval_backend.strip()
            arena_contract = suite_spec.get("arena")
            if arena_contract and arena_contract.get("task"):
                for record in metrics["per_task"]:
                    expected_name = arena_contract["task"]
                    if record["task"] != expected_name:
                        fail(f"suite/task mismatch: expected={expected_name!r} got={record['task']!r}")
            eec = metrics.get("effective_eval_config")
            _efv = os.environ.get("EXPECTED_FAMILY_VERSION", "").strip()
            _is_arena = suite_specs.get(expected_suite, {}).get("simulator") == "isaac_arena"
            if _is_arena and not isinstance(eec, dict):
                fail("effective_eval_config is required for Arena reports and must be a dict")
            if _is_arena and isinstance(eec, dict):
                _arena_required = ("budget_type", "num_episodes", "num_envs",
                                   "gr00t_version", "arena_embodiment",
                                   "policy_config", "embodiment_tag", "arena_object")
                _arena_missing = [k for k in _arena_required if k not in eec]
                if _arena_missing:
                    fail(f"effective_eval_config missing required Arena fields: {_arena_missing}")
                if eec["budget_type"] != "fixed_trials":
                    fail(f"Arena effective_eval_config.budget_type must be "
                         f"'fixed_trials', got {eec['budget_type']!r}")
                # Exact int checks: `type(...) is not int` also rejects True and 1.0.
                reported_episodes = eec["num_episodes"]
                if type(reported_episodes) is not int or reported_episodes < 1:
                    fail(f"effective_eval_config.num_episodes must be a positive "
                         f"integer, got {reported_episodes!r} "
                         f"({type(reported_episodes).__name__})")
                if reported_episodes != int(expected_trials):
                    fail(f"effective_eval_config.num_episodes={reported_episodes} "
                         f"differs from EXPECTED_EVAL_TRIALS={int(expected_trials)}")
                if type(eec["num_envs"]) is not int or eec["num_envs"] != 1:
                    fail(f"effective_eval_config.num_envs must be integer 1, got "
                         f"{eec['num_envs']!r} -- Arena counts every environment that "
                         f"ends in the same vectorized step, so a vectorized run "
                         f"cannot evaluate exactly the requested episode count")
                if "num_steps" in eec:
                    fail("effective_eval_config.num_steps is retired: the evaluated "
                         "sample size is an episode count. A report carrying both a "
                         "step budget and an episode count is a mixed old/new "
                         "declaration and is not trustworthy.")
                reported_version = eec["gr00t_version"]
                if not reported_version:
                    fail("effective_eval_config.gr00t_version must be non-empty")
                # The CONTENT of the consumed policy config, not just its path. Validate
                # compared the path and a handful of labels, which cannot detect that the
                # same path carried a different protocol (chunk length, joint ordering,
                # camera handling) between runs.
                #
                # This is self-reported evidence: Validate runs in a different image and
                # does not have the eval container's config file, so it CANNOT recompute
                # the digest. Requiring it to be present and well formed stops a producer
                # from omitting it, and gives an approver something to compare across
                # runs -- it does not establish that the digest matches the file.
                _cfg_digest = eec.get("policy_config_digest")
                if not (isinstance(_cfg_digest, str)
                        and re.fullmatch(r"sha256:[0-9a-f]{64}", _cfg_digest)):
                    fail(f"effective_eval_config.policy_config_digest must be a "
                         f"'sha256:<64 hex>' digest of the consumed policy config, got "
                         f"{_cfg_digest!r}. A path alone does not establish which "
                         f"protocol was run.")
                if _efv and reported_version != _efv:
                    fail(f"effective_eval_config.gr00t_version={reported_version!r} "
                         f"differs from expected {_efv!r}")
                _exp_tag = os.environ.get("EXPECTED_EMBODIMENT_TAG", "").strip()
                _exp_emb = os.environ.get("EXPECTED_ARENA_EMBODIMENT", "").strip()
                _exp_obj = os.environ.get("EXPECTED_ARENA_OBJECT", "").strip()
                _exp_pol = os.environ.get("EXPECTED_POLICY_CONFIG", "").strip()
                # No EXPECTED_NUM_STEPS: the episode count is already checked against
                # EXPECTED_EVAL_TRIALS above, which is the same pipeline parameter the
                # evaluator was given. A second budget expectation would be a second
                # source of truth for one quantity.
                _arena_exp_required = {
                    "EXPECTED_EMBODIMENT_TAG": _exp_tag,
                    "EXPECTED_ARENA_EMBODIMENT": _exp_emb,
                    "EXPECTED_ARENA_OBJECT": _exp_obj,
                    "EXPECTED_POLICY_CONFIG": _exp_pol,
                }
                _arena_exp_missing = [k for k, v in _arena_exp_required.items() if not v]
                if _arena_exp_missing:
                    fail(f"Arena execution requires expected protocol values but "
                         f"these are empty: {_arena_exp_missing}. They are populated "
                         f"from pipeline parameters; an empty value is a "
                         f"misconfiguration or a hand-start without required params.")
                def _norm(v):
                    return "" if v in ("NONE", "") else v
                _check_pairs = [
                    ("embodiment_tag", eec.get("embodiment_tag", ""), _exp_tag),
                    ("arena_embodiment", eec.get("arena_embodiment", ""), _exp_emb),
                    ("policy_config", eec.get("policy_config", ""), _exp_pol),
                ]
                _robj = _norm(eec.get("arena_object", ""))
                _eobj = _norm(_exp_obj)
                if _robj != _eobj:
                    _check_pairs.append(("arena_object", _robj, _eobj))
                for field, reported, expected in _check_pairs:
                    if str(reported) != str(expected):
                        fail(f"effective_eval_config.{field}={reported!r} differs from "
                             f"requested {expected!r}")
                print("[validate] Arena effective config matches requested expectations")
            elif isinstance(eec, dict):
                reported_version = eec.get("gr00t_version", "")
                if reported_version and _efv and reported_version != _efv:
                    fail(f"effective_eval_config.gr00t_version={reported_version!r} "
                         f"differs from expected {_efv!r}")
            # I6: the version comparison above is conditional on the field being PRESENT and
            # non-empty, and the effective config itself was required only for Arena. So a
            # LIBERO report could omit it entirely and pass under any EXPECTED_FAMILY_VERSION
            # -- a probe accepted a correctly digested N1.6 report under n17. The LIBERO
            # suites' embodiment tags are null, so the embodiment check does not independently
            # identify N1.6 versus N1.7 either.
            #
            # Whenever the execution declares an expected family version, the report must
            # state which version actually ran. Absence is not agreement.
            #
            # Scoped to GR00T: gr00t_version is a GR00T-specific field, and the pipeline sets
            # EXPECTED_FAMILY_VERSION for every family. Demanding it from a MolmoAct2 report
            # would reject valid evidence, which the suite caught immediately.
            if _efv and metrics.get("model_family") == "gr00t":
                if not isinstance(eec, dict):
                    fail(f"EXPECTED_FAMILY_VERSION={_efv!r} was requested but the report has "
                         f"no effective_eval_config, so the version that actually ran is "
                         f"unknown. A report that cannot state its evaluator version cannot "
                         f"be matched against the requested one.")
                reported_version = eec.get("gr00t_version")
                if not isinstance(reported_version, str) or not reported_version.strip():
                    fail(f"EXPECTED_FAMILY_VERSION={_efv!r} was requested but "
                         f"effective_eval_config.gr00t_version is "
                         f"{reported_version!r}. An absent version cannot be shown to match.")
                if reported_version.strip() != _efv:
                    fail(f"effective_eval_config.gr00t_version="
                         f"{reported_version.strip()!r} differs from the requested {_efv!r}")
                print(f"[validate] evaluator version confirmed as {_efv!r} from the report")

            # I6: the evaluation PROTOCOL was taken from the checkpoint's own input_config with
            # only a positive-integer check, and never reported -- so two packages sharing a
            # suite, trial count and seed could have been evaluated with different episode
            # horizons or action-chunk lengths, which is a different experiment. The suite now
            # declares what it expects and the producer records what it consumed; this compares
            # them. The expectation is pipeline-owned; the checkpoint does not get a vote.
            _protocol = (suite_specs.get(expected_suite) or {}).get("evaluation_protocol")
            # C2: `if _protocol:` was the whole gate, and the resolved table never carried the
            # field, so this branch had never executed for any suite. Absence must be decided
            # against the SIMULATOR rather than treated as "nothing to check": libero suites
            # are required to declare a protocol, so a missing one there means the table is
            # wrong, not that the suite is exempt.
            _simulator = (suite_specs.get(expected_suite) or {}).get("simulator")
            _family = metrics.get("model_family")
            # C3 (cycle 10): the declared 8 action steps / 720-step horizon is a GR00T protocol,
            # and I applied it to EVERY libero family. It is not a suite-wide truth:
            #
            #   * pinned OpenVLA-OFT uses per-suite horizons 220 / 280 / 300 / 520
            #     (run_libero_eval.py:64-67), and its real budget is max_steps + num_steps_wait
            #     (:316), so even those numbers are not the whole story;
            #   * MolmoAct2 training configures TEN action steps (train_entry.py:366-367).
            #
            # Neither producer reports effective_eval_config, so the requirement rejected them
            # outright -- and simply adding the fields with the suite constants would have
            # published a WRONG protocol rather than an absent one, which is worse. Scoped to the
            # family the declared values actually describe, matching how the effective-config
            # version requirement above is scoped.
            #
            # Extending this to OpenVLA and MolmoAct2 needs a per-family/version/suite protocol
            # table and each evaluator reporting its resolved settings. Until that exists, an
            # unscoped requirement would be enforcement of a value nobody consumes.
            if _simulator == "libero" and _family == "gr00t" and not _protocol:
                fail(f"suite {expected_suite!r} is a libero suite but the resolved suite table "
                     f"carries no evaluation_protocol. The protocol comparison would be "
                     f"skipped entirely, so episode horizon and action-chunk length would go "
                     f"unchecked.")
            if _protocol and _family == "gr00t":
                if not isinstance(eec, dict):
                    fail(f"suite {expected_suite!r} declares an evaluation protocol but the "
                         f"report has no effective_eval_config, so the protocol actually used "
                         f"is unknown.")
                for _key, _expected in sorted(_protocol.items()):
                    _actual = eec.get(_key)
                    if _actual is None:
                        fail(f"suite {expected_suite!r} declares {_key}={_expected!r} but the "
                             f"report does not record what was consumed. An unreported "
                             f"protocol setting cannot be shown to match.")
                    if _actual != _expected:
                        fail(f"evaluation protocol mismatch: {_key}={_actual!r} was consumed "
                             f"but suite {expected_suite!r} expects {_expected!r}. Episode "
                             f"horizon and action-chunk length change the experiment, so this "
                             f"is not the evaluation the suite defines.")
                print(f"[validate] evaluation protocol matches the suite: "
                      f"{ {k: _protocol[k] for k in sorted(_protocol)} }")
            print("[validate] Full schema-v3 validation PASSED")
        except Exception as e:
            if "ReportInvalid" in type(e).__name__ or "Invalid" in str(type(e)):
                fail(f"schema-v3 validation: {e}")
            # Unexpected exceptions also fail closed
            fail(f"validator raised unexpected {type(e).__name__}: {e}")

    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/opt/ml/processing/checkpoint")
    verified_digest, training_lineage = verify_checkpoint(metrics, checkpoint_dir)
    # verify_checkpoint has just proved the report's copy of the manifest equals the one
    # inside the archive, so it is safe to compare the execution's training contract
    # against that copy.
    training_contract = check_training_contract(
        metrics["checkpoint_manifest"], training_lineage)
    print(f"[validate] independent checkpoint digest verified: {verified_digest}")

    # --- S3 artifact identity verification (AR-02) ---
    expected_source_uri = _require_expectation("EXPECTED_MODEL_SOURCE_URI")
    reported_identity = metrics.get("model_artifact_identity")
    if reported_identity:
        reported_bucket = reported_identity.get("bucket", "")
        reported_key = reported_identity.get("key", "")
        try:
            from urllib.parse import urlparse

            import boto3
            u = urlparse(expected_source_uri)
            expected_bucket, expected_key = u.netloc, u.path.lstrip("/")
            if u.scheme != "s3" or not expected_bucket or not expected_key:
                fail(f"EXPECTED_MODEL_SOURCE_URI is not a valid s3:// URI: {expected_source_uri!r}")
            if reported_bucket != expected_bucket or reported_key != expected_key:
                fail(f"Model source location mismatch: expected s3://{expected_bucket}/{expected_key}, "
                     f"report claims s3://{reported_bucket}/{reported_key}")
            # schema 3: verify the named VERSION BY CONTENT, not by an unversioned HEAD.
            #
            # The old check asked "does the current object still carry the version and ETag the
            # evaluation saw". That describes the object current WHEN HEAD RUNS, and a HEAD has
            # no body, so it never tied the named version to any bytes. A version-specific GET
            # replaces "the current key still names this version" with "this version contains
            # these bytes", which is the claim registration actually depends on.
            reported_version = reported_identity.get("version_id")
            _s3_stub = os.environ.get("_VALIDATE_S3_HEAD_STUB")
            if _s3_stub:
                # Offline path: the stub stands in for the version GET. It carries the same
                # fields the real response is read for, including a Body, so a stubbed run
                # exercises the same comparisons rather than a weaker subset.
                head = json.loads(_s3_stub)
                actual_version = head.get("VersionId")
                actual_etag = (head.get("ETag") or "").strip('"')
                # BodyPath names a local file standing in for the version's bytes. A JSON
                # string cannot carry binary, and hashing an absent body would make every
                # stubbed run fail the content comparison for the wrong reason.
                body_path = head.get("BodyPath")
                if not body_path:
                    fail("_VALIDATE_S3_HEAD_STUB has no BodyPath, so the version's content "
                         "cannot be measured. A stub that omits the body would exercise a "
                         "weaker check than production.")
                with open(body_path, "rb") as _bh:
                    version_bytes = _bh.read()
            else:
                response = boto3.client("s3").get_object(
                    Bucket=expected_bucket, Key=expected_key, VersionId=reported_version)
                actual_version = response.get("VersionId")
                actual_etag = (response.get("ETag") or "").strip('"')
                # Streamed whole, with no Range or PartNumber, so the measurement covers the
                # entire version rather than a slice of it.
                _hash = hashlib.sha256()
                _count = 0
                _body = response["Body"]
                while True:
                    chunk = _body.read(1024 * 1024)
                    if not chunk:
                        break
                    _hash.update(chunk)
                    _count += len(chunk)
                version_bytes = None
                version_sha, version_size = _hash.hexdigest(), _count
            if _s3_stub:
                version_sha = hashlib.sha256(version_bytes).hexdigest()
                version_size = len(version_bytes)
            if actual_version != reported_version:
                fail(f"S3 artifact VersionId changed: eval saw {reported_version!r}, Validate "
                     f"sees {actual_version!r} -- checkpoint was overwritten between eval and "
                     f"validation")
            if actual_etag != reported_identity.get("etag"):
                fail(f"S3 artifact ETag changed: eval saw "
                     f"{reported_identity.get('etag')!r}, Validate sees {actual_etag!r}")
            reported_archive = metrics["source_archive"]
            if (version_sha != reported_archive["sha256"]
                    or version_size != reported_archive["size_bytes"]):
                fail(f"S3 version content mismatch: version {actual_version!r} contains "
                     f"sha256={version_sha} size={version_size}, but the evaluation measured "
                     f"sha256={reported_archive['sha256']} "
                     f"size={reported_archive['size_bytes']}. The named version does not "
                     f"contain the bytes that were evaluated, so registering it would attach "
                     f"the score to a different artifact.")
            print(f"[validate] S3 version verified BY CONTENT: "
                  f"s3://{expected_bucket}/{expected_key} "
                  f"VersionId={actual_version}, ETag={actual_etag}, "
                  f"sha256={version_sha}, size={version_size}")
        except SystemExit:
            raise
        except Exception as e:
            fail(f"S3 artifact identity check failed: {e}")
    elif reported_identity is None:
        print("[validate] model_artifact_identity is null (HF-mode checkpoint) -- S3 identity check skipped")

    # --- All checks passed: emit validated_metrics.json ---
    # S5: full_suite derived from the validated task SET, not from the literal selector.
    # It was `expected_task_ids_str == "all"`, so an explicit list naming every canonical
    # task -- which is full coverage -- was stamped partial, while "all" was trusted without
    # comparing anything. Both are pipeline-owned, so neither is attacker-controlled, but the
    # label was still wrong for a legitimate way of requesting the whole suite.
    #
    # canonical_ids/task_ids are bound only when the shared validator was available, so the
    # method used is RECORDED rather than assumed. The literal fallback is the old behaviour
    # and is marked as such instead of being presented as a verified comparison.
    coverage_canonical = locals().get("canonical_ids")
    coverage_selected = locals().get("task_ids")
    if isinstance(coverage_canonical, list) and isinstance(coverage_selected, list):
        full_suite = set(coverage_selected) == set(coverage_canonical)
        full_suite_basis = "task_set_equality"
    else:
        full_suite = (expected_task_ids_str == "all")
        full_suite_basis = "selector_literal"

    # zero_action / POC runs can NEVER be full_suite (caught above, but belt-and-suspenders)
    if policy_type == "zero_action":
        full_suite = False

    # Resolved BEFORE anything is assembled or promoted: a reference that cannot name the bytes it ran
    # must stop the run here, not after the evidence has been copied somewhere durable.
    _evaluator_image_uri, _evaluator_image_digest = _require_digest_image("EVALUATOR_IMAGE_URI")

    import boto3
    import botocore

    validated = {
        "success_rate": metrics["success_rate"],
        "full_suite": full_suite,
        # How coverage was determined, so a consumer is not left inferring it.
        "full_suite_basis": full_suite_basis,
        "eval_backend": eval_backend,
        # Which training-provenance contract was applied, and what it compared. Recorded
        # so a consumer can tell a checkpoint this execution trained and verified from
        # one that was supplied as input and whose lineage is established out of band.
        "training_contract": training_contract,
        "validation_runtime": {
            "boto3_version": boto3.__version__,
            "botocore_version": botocore.__version__,
            "sdk_bundle_sha256": os.environ.get("VLA_VALIDATION_SDK_SHA256"),
        },
        "model_family": metrics["model_family"],
        "suite": metrics["suite"],
        "task_ids": metrics["task_ids"],
        # Wrapper-owned, matching expected_budget_type above: the evaluated sample is
        # a fixed number of complete episodes per task.
        "budget_type": "fixed_trials",
        "eval_seed": metrics["eval_seed"],
        "eval_trials": metrics["num_trials_per_task"],
        "episodes": metrics["episodes"],
        "per_task": metrics["per_task"],
        "policy_type": policy_type,
        "weights_digest_recomputed_by_validate": verified_digest,
        "model_artifact_identity": reported_identity,
        "effective_eval_config": metrics.get("effective_eval_config"),

        # cycle-15 I1: the raw report's provenance was VALIDATED and then dropped, so two runs
        # differing only in provenance produced indistinguishable attestations -- the immutable
        # evidence linked from the model package could not tell them apart. Validating a field and
        # then discarding it is the weakest possible outcome: the check runs, and nothing it
        # established survives to the artifact a consumer actually reads.
        #
        # Same defect as I2's seed_scope above, one field over. Carried verbatim rather than
        # summarised, because a summary is a new claim.
        "provenance": metrics.get("provenance"),

        # cycle-15 I1: evaluator_image_uri alone does not identify the evaluator -- the graph passes
        # the executable sourcedir separately, so two runs sharing an image but running different
        # eval code produced identical attestations. Recorded from the PIPELINE's own environment,
        # not from the report: the producer must not name its own provenance. Absent when Validate
        # runs outside the graph, and recorded as null rather than guessed.
        "evaluator_sourcedir_uri": os.environ.get("VLA_EVAL_SOURCEDIR_URI"),

        # aux_metrics is non-gated diagnostic evidence, and dropping it meant the attestation held
        # neither the gated score's context nor the complete raw report. No gate reads it; it is
        # preserved so a registered result can be diagnosed after the fact.
        "aux_metrics": metrics.get("aux_metrics"),

        # cycle-16 C1: the external VLM backbone determines preprocessing and was
        # downloaded with no revision, so it could change while everything else stayed
        # identical. Carried into the attestation for the same reason provenance is: a
        # field validated on the report and then dropped establishes nothing a consumer
        # can read. Absent for families that use no external backbone, recorded as null
        # rather than omitted.
        "backbone_identity": metrics.get("backbone_identity"),

        # I2: the receipt carried effective_eval_config but DROPPED seed_scope, so the
        # producer's own qualification was lost exactly where it matters most. Arena
        # records what the eval seed actually bound, read from the seeding wrapper's
        # evidence rather than asserted -- and when that evidence is absent it says so.
        # Omitting it from the immutable receipt meant a run whose policy RNG was NOT
        # demonstrably bound registered indistinguishably from one where it was.
        #
        # Carried verbatim, never defaulted: a fabricated scope would be worse than an
        # absent one, since the whole point is that this process cannot observe the
        # policy server it did not run.
        "seed_scope": metrics.get("seed_scope"),
        "validation_passed": True,
        # I8: which evaluator image produced this evidence. Read from the PIPELINE, not
        # from the report -- the producer must not name its own provenance. validate_entry
        # had zero references to any image identity, so the immutable attestation certified
        # a score without recording what code produced it. An immutable tag prevents
        # overwriting a tag; it does not say which image ran, so a current launcher and
        # validator could evaluate with an older evaluator image and the receipt would look
        # identical. A score whose attestation cannot name its code is not attributable,
        # which is the property this component exists to establish.
        "evaluator_image_uri": _evaluator_image_uri,
        # I13: the DIGEST, not just the reference. A tag names a pointer; only the digest names the
        # bytes that ran. This is not hypothetical: three Arena executions ran against an
        # isaac-lab-arena image from the previous day while their results were attributed to current
        # source, because the builder that should have refreshed it was submitting to a project whose
        # role could not read the source bucket. Every receipt from those runs would have looked
        # correct. A digest-form reference is what makes that visible, so it is REQUIRED rather than
        # recorded when convenient.
        "evaluator_image_digest": _evaluator_image_digest,
    }

    # --- Promote the validated bytes, then register THOSE ---------------------------
    # The immutable attestation carries everything an approver needs to tie the registered
    # object to this evaluation, including the lineage classification: bytes verified
    # against a self-supplied manifest are NOT the same claim as training provenance
    # established by this pipeline, and promotion must not upgrade one into the other.
    archive_path = os.path.join(checkpoint_dir, "model.tar.gz")
    _archive_digest = archive_sha256(archive_path)
    attestation = {
        "attestation_version": 1,
        "validated_metrics": validated,
        "source_artifact": {
            # Where the evaluated bytes came from, as recorded by the evaluator.
            "identity": reported_identity,
            "expected_source_uri": os.environ.get("EXPECTED_MODEL_SOURCE_URI", ""),
        },
        "promoted_artifact": {
            "archive_sha256": _archive_digest,
            "size_bytes": os.path.getsize(archive_path),
            # The checkpoint-tree digest, which deliberately excludes the manifest, logs
            # and hidden paths. Kept separate from the archive digest above: it compares
            # checkpoint CONTENT and is not an exact digest of the registered object.
            "weights_digest_recomputed_by_validate": verified_digest,
        },
        "execution": {
            "pipeline_execution_id": os.environ.get("PIPELINE_EXECUTION_ID", ""),
            "gate_threshold": os.environ.get("EXPECTED_SUCCESS_THRESHOLD", ""),
        },
        # Which provenance contract applied. eval_only means the checkpoint was supplied
        # from outside: its bytes are verified, its training lineage is not.
        "lineage": training_contract,
        "protocol": {
            "requested": {
                "eval_seed": os.environ.get("EXPECTED_EVAL_SEED", ""),
                "eval_trials": os.environ.get("EXPECTED_EVAL_TRIALS", ""),
                "task_ids": os.environ.get("EXPECTED_EVAL_TASK_IDS", ""),
                "suite": expected_suite,
                "policy_config": os.environ.get("EXPECTED_POLICY_CONFIG", ""),
            },
            "consumed": metrics.get("effective_eval_config"),
            "seed_scope": metrics.get("seed_scope"),
        },
    }
    print("[validate] >>> promote_artifact: conditional create into the trust bucket",
          flush=True)
    validated["promotion"] = promote_validated_artifact(archive_path, attestation)
    _promo = validated["promotion"] or {}
    print(f"[validate] <<< promote_artifact OK: "
          f"{_promo.get('model_uri') or _promo}", flush=True)

    print("[validate] >>> write_receipt", flush=True)
    validated_path = os.path.join(output_dir, "validated_metrics.json")
    with open(validated_path, "w") as f:
        json.dump(validated, f, indent=2, allow_nan=False)
    print(f"[validate] <<< write_receipt OK: wrote {validated_path} "
          f"({os.path.getsize(validated_path)} B)", flush=True)

    print("[validate] ALL CHECKS PASSED")
    print(f"[validate] === EXECUTION_SUCCESS: all code executed -- "
          f"full_suite={full_suite}, success_rate={validated['success_rate']}, "
          f"receipt at {validated_path}", flush=True)


if __name__ == "__main__":
    main()
