"""I1: promotion must not buffer the checkpoint archive, nor single-PUT a large one.

Validate runs on ml.m5.large and S3's single-PUT limit is 5 GiB, so reading the complete
archive into memory was both a memory hazard and an outright ceiling. Collision verification
read the whole existing object back as well.

These tests drive the real publication helpers with a fake S3 client, so they check behaviour
rather than the shape of the code: what gets sent, in how many parts, and whether anything is
fully materialised.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import pathlib

import pytest

_ENTRY = (pathlib.Path(__file__).resolve().parents[1] / "entrypoints/validate_entry.py")


def _module():
    spec = importlib.util.spec_from_file_location("validate_entry_streaming", _ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeS3:
    """Records what it was asked to do, and never keeps whole bodies."""

    #: Whether this fake's botocore knows S3 conditional writes. The real Validate container's does
    #: NOT -- it runs the sklearn 1.2-1 image, whose botocore rejected IfNoneMatch outright and failed
    #: a real publish. The code detects support from the service model, so the fake must expose one or
    #: every test silently exercises the FALLBACK path while claiming to test the conditional one.
    supports_conditional_write = True

    class _Meta:
        def __init__(self, supported: bool):
            self._supported = supported

        @property
        def service_model(self):
            outer = self

            class _Shape:
                members = {"Bucket", "Key", "Body", "ContentType", "MultipartUpload", "UploadId",
                           "ChecksumSHA256"} | ({"IfNoneMatch"} if outer._supported else set())

            class _Model:
                @staticmethod
                def operation_model(_name):
                    return type("_Op", (), {"input_shape": _Shape})

            return _Model()

    @property
    def meta(self):
        return self._Meta(self.supports_conditional_write)

    def __init__(self, existing: dict | None = None):
        self.existing = existing or {}
        self.puts: list[dict] = []
        self.parts: list[int] = []
        self.completed: list[dict] = []
        self.aborted = 0
        self.streamed_reads = 0

    def head_object(self, Bucket, Key):
        """Real S3 raises when the key is absent; the fallback path depends on that distinction."""
        if Key not in self.existing:
            raise _precondition_failed()
        return {"ContentLength": len(self.existing[Key])}

    def put_object(self, Bucket, Key, Body, ContentType, IfNoneMatch=None):
        if Key in self.existing and IfNoneMatch == "*":
            raise _precondition_failed()
        # A file object, not bytes: read it in chunks the way S3 would.
        total = 0
        while True:
            chunk = Body.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
        self.puts.append({"key": Key, "bytes": total, "conditional": IfNoneMatch})
        return {}

    def create_multipart_upload(self, Bucket, Key, ContentType):
        return {"UploadId": "upload-1"}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
        self.parts.append(len(Body))
        return {"ETag": f'"etag-{PartNumber}"'}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload,
                                  IfNoneMatch=None):
        if Key in self.existing and IfNoneMatch == "*":
            raise _precondition_failed()
        self.completed.append({"key": Key, "parts": len(MultipartUpload["Parts"]),
                               "conditional": IfNoneMatch})
        return {}

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.aborted += 1

    def get_object(self, Bucket, Key):
        payload = self.existing[Key]

        class _Body:
            def __init__(self, data, owner):
                self._data, self._at, self._owner = data, 0, owner

            def read(self, size=-1):
                self._owner.streamed_reads += 1
                if self._at >= len(self._data):
                    return b""
                end = len(self._data) if size is None or size < 0 else self._at + size
                chunk = self._data[self._at:end]
                self._at = end
                return chunk

            def close(self):
                pass

        return {"Body": _Body(payload, self)}


def _precondition_failed():
    error = Exception("precondition failed")
    error.response = {"Error": {"Code": "PreconditionFailed"}}
    return error


def _archive(tmp_path, size):
    path = tmp_path / "model.tar.gz"
    path.write_bytes(b"\x1f\x8b" + os.urandom(size - 2))
    return path


def test_a_small_archive_is_streamed_from_the_file_not_buffered(tmp_path):
    module = _module()
    path = _archive(tmp_path, 1024)
    digest = module.archive_sha256(str(path))
    client = _FakeS3()
    state = module._publish_conditionally(
        client, "trust", "artifacts/v1/sha256/x/model.tar.gz", str(path),
        "application/gzip", digest)
    assert state == "created"
    # Body was a file object read in chunks -- a bytes payload would have raised on .read.
    assert client.puts == [{"key": "artifacts/v1/sha256/x/model.tar.gz",
                            "bytes": 1024, "conditional": "*"}]


def test_a_large_archive_is_published_in_parts(tmp_path):
    """Above the threshold, a single PutObject cannot be used at all beyond 5 GiB."""
    module = _module()
    module._MULTIPART_THRESHOLD = 4096
    module._PART_BYTES = 1024
    path = _archive(tmp_path, 4096 + 512)
    digest = module.archive_sha256(str(path))
    client = _FakeS3()
    state = module._publish_conditionally(
        client, "trust", "artifacts/v1/sha256/x/model.tar.gz", str(path),
        "application/gzip", digest)
    assert state == "created_multipart"
    assert client.puts == [], "must not also single-PUT"
    assert client.parts == [1024, 1024, 1024, 1024, 512]
    assert client.completed[0]["parts"] == 5
    # The completion is what carries the precondition.
    assert client.completed[0]["conditional"] == "*"


def test_an_identical_existing_object_is_an_idempotent_success(tmp_path):
    """A retried Validate step must not fail on its own previous publication."""
    module = _module()
    path = _archive(tmp_path, 2048)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    key = "artifacts/v1/sha256/x/model.tar.gz"
    client = _FakeS3(existing={key: payload})
    state = module._publish_conditionally(
        client, "trust", key, str(path), "application/gzip", digest)
    assert state == "already_present_identical"
    # Verified by STREAMING the existing object, in more than one read.
    assert client.streamed_reads > 1


def test_a_differing_existing_object_is_fatal(tmp_path):
    module = _module()
    path = _archive(tmp_path, 2048)
    digest = module.archive_sha256(str(path))
    key = "artifacts/v1/sha256/x/model.tar.gz"
    client = _FakeS3(existing={key: b"different bytes entirely"})
    with pytest.raises(SystemExit):
        module._publish_conditionally(
            client, "trust", key, str(path), "application/gzip", digest)


def test_a_failed_multipart_upload_is_aborted(tmp_path):
    """An abandoned upload must not linger as billable storage or look like a publication."""
    module = _module()
    module._MULTIPART_THRESHOLD = 512
    module._PART_BYTES = 256

    class _Failing(_FakeS3):
        def upload_part(self, **kwargs):
            raise RuntimeError("network died")

    path = _archive(tmp_path, 1024)
    client = _Failing()
    with pytest.raises(SystemExit):
        module._publish_conditionally(
            client, "trust", "artifacts/v1/sha256/x/model.tar.gz", str(path),
            "application/gzip", module.archive_sha256(str(path)))
    assert client.aborted == 1


def test_the_digest_is_computed_incrementally(tmp_path):
    """Hashing must match a whole-file digest without ever holding the file in memory."""
    module = _module()
    path = _archive(tmp_path, 3 * module._CHUNK_BYTES + 7)
    assert module.archive_sha256(str(path)) == hashlib.sha256(
        path.read_bytes()).hexdigest()


def test_the_conditional_create_is_mandatory_not_optional():
    """Measured against the real bucket, not reasoned about.

    A standalone probe ran in the Validate image under the validation role and wrote into
    artifacts/v1/ -- the actual protected namespace -- and reported:

        conditional_put:      SUCCEEDED
        unconditional_put:    DENIED AccessDenied
        second_conditional:   PreconditionFailed

    So the bucket's DenyUnconditionalWritesToProtectedNamespaces statement genuinely enforces
    create-only. An unconditional write is not a weaker publish, it is a DENIED one, and a client that
    omits the parameter when its botocore is old converts a clear error into a confusing authorization
    failure. An earlier fix of mine did exactly that and this test exists to stop it returning.
    """
    source = (pathlib.Path(__file__).resolve().parents[1] / "entrypoints/validate_entry.py").read_text()
    tree = ast.parse(source)
    publishes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") in ("put_object", "complete_multipart_upload")]
    assert publishes, "no publish calls found; the parse is wrong"
    for call in publishes:
        assert any(k.arg == "IfNoneMatch" for k in call.keywords), (
            f"the publish at line {call.lineno} omits the conditional create. The bucket DENIES an "
            f"unconditional write to artifacts/v1/, so this call cannot succeed -- and it fails with "
            f"AccessDenied, which hides the real cause.")


def test_support_is_guaranteed_before_the_publishing_client_is_built():
    """An upgrade after the client exists leaves that client bound to the old botocore.

    A standalone probe measured botocore 1.31.85 in the Validate image, no conditional-create support,
    and a successful in-place upgrade to 1.42.97 -- so the guarantee is achievable, but only if it runs
    before the client that publishes.

    Scoped to the ENCLOSING FUNCTION of the publishing client, deliberately. A file-wide line
    comparison counted two irrelevant constructions -- the guarantee's own capability probe, and a
    read-only get_object far below -- and stayed satisfied wherever the guarantee sat.
    """
    source = (pathlib.Path(__file__).resolve().parents[1] / "entrypoints/validate_entry.py").read_text()
    tree = ast.parse(source)

    holder = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == "_ensure_conditional_write_support":
            continue                      # its own probe client is not the publishing one
        for stmt in ast.walk(node):
            if (isinstance(stmt, ast.Assign)
                    and any(getattr(t, "id", "") == "client" for t in stmt.targets)
                    and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "attr", "") == "client"):
                holder = (node, stmt.lineno)
                break
        if holder:
            break
    assert holder, "no function assigns `client = boto3.client(...)`; the publishing client moved"
    function, client_line = holder

    guarantee = [n.lineno for n in ast.walk(function) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_ensure_conditional_write_support"]
    assert guarantee, (
        f"{function.name}() builds the publishing client at line {client_line} without first "
        f"guaranteeing it can send a conditional create; the bucket DENIES a write without one")
    assert min(guarantee) < client_line, (
        f"{function.name}() guarantees support at line {min(guarantee)}, AFTER building the client at "
        f"{client_line}; an upgrade after construction leaves that client on the old botocore, so it "
        f"still cannot send the parameter the bucket requires")
