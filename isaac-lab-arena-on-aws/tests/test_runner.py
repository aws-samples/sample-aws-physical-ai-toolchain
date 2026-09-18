"""Tests for runner.py -- offline, no real AWS calls.

Uses a stub S3 client to verify content-addressing logic without moto
(keeps dependencies minimal).
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest

from vla_pipeline.config import PipelineConfig  # noqa: E402
from vla_pipeline.runner import start, upload_code, upload_directory, upsert_versioned  # noqa: E402

_CFG = {
    "account_id": "123456789012",
    "region": "us-west-2",
    "role_arn": "arn:aws:iam::123456789012:role/Exec",
    "bucket": "my-bucket",
    # Component trust boundary (C4 + C5): distinct runtime identities and the
    # create-only bucket promoted artifacts are published to.
    "training_role_arn": "arn:aws:iam::123456789012:role/Training",
    "workload_role_arn": "arn:aws:iam::123456789012:role/Workload",
    "validation_role_arn": "arn:aws:iam::123456789012:role/Validation",
    "trust_bucket": "my-trust-bucket",
    "handoff_bucket": "my-handoff-bucket",
}


def _cfg():
    return PipelineConfig(**_CFG)


class _CapableBotocore:
    """Expose a service model that KNOWS the conditional-create parameter.

    The production code detects support from `client.meta.service_model` rather than a version string,
    because the real Validate container -- SageMaker's sklearn 1.2-1 image -- rejected the parameter
    outright and failed a real publish. A fake without a service model therefore silently exercises the
    FALLBACK path, so any test asserting the conditional write must opt in to capability explicitly.

    Shared by the fakes that stand in for a modern boto3.
    """

    supports_conditional_write = True

    class _Meta:
        def __init__(self, supported: bool):
            self._supported = supported

        @property
        def service_model(self):
            supported = self._supported

            class _Shape:
                members = ({"Bucket", "Key", "Body", "ContentType", "ChecksumSHA256",
                            "MultipartUpload", "UploadId"}
                           | ({"IfNoneMatch"} if supported else set()))

            class _Model:
                @staticmethod
                def operation_model(_name):
                    return type("_Op", (), {"input_shape": _Shape})

            return _Model()

    @property
    def meta(self):
        return self._Meta(self.supports_conditional_write)


class FakeS3(_CapableBotocore):
    """Minimal stub that tracks put_object calls and simulates head_object."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[dict] = []
        self.head_calls: list[str] = []

    def head_object(self, Bucket, Key):
        self.head_calls.append(Key)
        if Key not in self.objects:
            error_response = {"Error": {"Code": "404"}}
            raise self.exceptions.ClientError(error_response, "HeadObject")
        return {}

    def put_object(self, Bucket, Key, Body, ChecksumSHA256=None, IfNoneMatch=None):
        # C4: publication is conditional and checksummed, so the double must model both or
        # the tests would pass against an API the code does not actually call.
        if IfNoneMatch == "*" and Key in self.objects:
            raise self.exceptions.ClientError(
                {"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        self.put_calls.append({"Bucket": Bucket, "Key": Key,
                               "ChecksumSHA256": ChecksumSHA256,
                               "IfNoneMatch": IfNoneMatch})

    def get_object(self, Bucket, Key):
        """Published bytes are read back and hashed, so a double must serve them."""
        if Key not in self.objects:
            raise self.exceptions.ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    class exceptions:
        class ClientError(Exception):
            def __init__(self, error_response, operation_name):
                self.response = error_response
                super().__init__(f"{operation_name}: {error_response}")


def test_upsert_preserves_atomic_update_version():
    pipeline = MagicMock()
    pipeline.upsert.return_value = {"PipelineArn": "pipeline", "PipelineVersionId": 7}
    assert upsert_versioned(pipeline, "role")["PipelineVersionId"] == 7
    pipeline.update.assert_not_called()


def test_first_deployment_updates_own_graph_to_obtain_version():
    pipeline = MagicMock()
    pipeline.upsert.return_value = {"PipelineArn": "pipeline"}
    pipeline.update.return_value = {"PipelineArn": "pipeline", "PipelineVersionId": 2}
    assert upsert_versioned(pipeline, "role")["PipelineVersionId"] == 2
    pipeline.update.assert_called_once_with(role_arn="role")


def test_missing_update_version_fails():
    pipeline = MagicMock()
    pipeline.upsert.return_value = {"PipelineArn": "pipeline"}
    pipeline.update.return_value = {"PipelineArn": "pipeline"}
    with pytest.raises(KeyError, match="PipelineVersionId"):
        upsert_versioned(pipeline, "role")


@pytest.mark.parametrize("first_mode", ["train", "eval-only"])
def test_interleaved_deployments_start_their_own_versions(first_mode):
    pipeline = MagicMock()
    pipeline.upsert.side_effect = [
        {"PipelineArn": "pipeline", "PipelineVersionId": 7},
        {"PipelineArn": "pipeline", "PipelineVersionId": 8},
    ]
    first = upsert_versioned(pipeline, "role")
    second = upsert_versioned(pipeline, "role")
    with patch("vla_pipeline.runner.boto3.client") as client:
        client.return_value.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "execution"}
        start(_cfg(), {"Mode": first_mode},
              pipeline_version_id=first["PipelineVersionId"])
        start(_cfg(), {"Mode": "eval-only" if first_mode == "train" else "train"},
              pipeline_version_id=second["PipelineVersionId"])
    calls = client.return_value.start_pipeline_execution.call_args_list
    assert [call.kwargs["PipelineVersionId"] for call in calls] == [7, 8]
    assert calls[0].kwargs["PipelineParameters"] == [
        {"Name": "Mode", "Value": first_mode}]


def test_start_requires_explicit_version():
    with pytest.raises(TypeError, match="pipeline_version_id"):
        start(_cfg(), {})


def test_upload_code_key_contains_sha256():
    """The S3 key must contain the sha256 of the file bytes -- this is the
    content-addressing guarantee that pins exact code in the definition and
    busts the cache on code change."""
    cfg = _cfg()
    fake_s3 = FakeS3()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello world')\n")
        f.flush()
        tmp_path = f.name

    try:
        content = Path(tmp_path).read_bytes()
        expected_digest = hashlib.sha256(content).hexdigest()

        with patch("vla_pipeline.runner.boto3") as mock_boto:
            mock_boto.client.return_value = fake_s3
            uri = upload_code(cfg, tmp_path)

        # The URI must contain the sha256 digest
        assert expected_digest in uri, \
            f"URI {uri} must contain sha256 {expected_digest}"

        # C4: published to the component TRUST bucket under code/v1/<sha256>/, not the
        # shared models bucket where peer Foundation-role workers can replace it.
        assert f"/code/v1/{expected_digest}/" in uri
        assert uri.startswith("s3://my-trust-bucket/")
        assert uri.endswith(Path(tmp_path).name)

        # Must have been uploaded (first call, key didn't exist)
        assert len(fake_s3.put_calls) == 1
        assert fake_s3.put_calls[0]["Bucket"] == "my-trust-bucket"
        # Conditional creation and a checksum, so a pre-existing object cannot be
        # silently replaced and corruption in transit is detected.
        assert fake_s3.put_calls[0]["IfNoneMatch"] == "*"
        assert fake_s3.put_calls[0]["ChecksumSHA256"]
    finally:
        os.unlink(tmp_path)


def test_upload_code_skips_if_exists():
    """If the content-addressed key already exists, skip the upload."""
    cfg = _cfg()
    fake_s3 = FakeS3()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello world')\n")
        f.flush()
        tmp_path = f.name

    try:
        content = Path(tmp_path).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        # Pre-populate: key already exists
        existing_key = f"code/v1/{digest}/{Path(tmp_path).name}"
        fake_s3.objects[existing_key] = content

        with patch("vla_pipeline.runner.boto3") as mock_boto:
            mock_boto.client.return_value = fake_s3
            uri = upload_code(cfg, tmp_path)

        # Should NOT have uploaded again
        assert len(fake_s3.put_calls) == 0
        # URI still correct
        assert digest in uri
    finally:
        os.unlink(tmp_path)


def test_upload_code_different_content_different_key():
    """Different file content produces a different S3 key (cache-busting)."""
    cfg = _cfg()
    fake_s3 = FakeS3()

    files = []
    uris = []
    try:
        for i, content in enumerate(["version_1\n", "version_2\n"]):
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", prefix=f"entry{i}_",
                    delete=False) as f:
                f.write(content)
                files.append(f.name)

        for fpath in files:
            with patch("vla_pipeline.runner.boto3") as mock_boto:
                mock_boto.client.return_value = fake_s3
                uris.append(upload_code(cfg, fpath))

        # Different content -> different URIs (different sha256 in key)
        assert uris[0] != uris[1]
        # Both uploaded
        assert len(fake_s3.put_calls) == 2
    finally:
        for f in files:
            os.unlink(f)


def test_upload_code_nonexistent_file_raises():
    """Uploading a nonexistent file raises FileNotFoundError."""
    cfg = _cfg()
    try:
        upload_code(cfg, "/nonexistent/path/foo.py")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass


def test_upload_directory_produces_tarball_with_sha256():
    """upload_directory creates a tar.gz named with its sha256."""
    cfg = _cfg()
    fake_s3 = FakeS3()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small directory with files
        (Path(tmpdir) / "entry.py").write_text("print('train')\n")
        (Path(tmpdir) / "utils.py").write_text("def helper(): pass\n")

        with patch("vla_pipeline.runner.boto3") as mock_boto:
            mock_boto.client.return_value = fake_s3
            uri = upload_directory(cfg, tmpdir)

        # URI must contain a sha256 hex string (64 chars)
        parts = uri.split("/code/v1/")
        assert len(parts) == 2
        sha_and_rest = parts[1]
        sha_part = sha_and_rest.split("/")[0]
        assert len(sha_part) == 64  # sha256 hex
        assert all(c in "0123456789abcdef" for c in sha_part)

        # Must be a .tar.gz
        assert uri.endswith(".tar.gz")

        # Was uploaded
        assert len(fake_s3.put_calls) == 1


def test_directory_upload_is_stable_across_paths_and_timestamps(tmp_path):
    cfg = _cfg()
    fake_s3 = FakeS3()
    directories = [tmp_path / "first", tmp_path / "second"]
    for index, directory in enumerate(directories):
        directory.mkdir()
        entry = directory / "entry.py"
        entry.write_text("print('train')\n")
        os.utime(entry, (100 + index, 100 + index))

    with patch("vla_pipeline.runner.boto3") as mock_boto:
        mock_boto.client.return_value = fake_s3
        with patch("gzip.time.time", return_value=1000):
            first_uri = upload_directory(cfg, str(directories[0]))
        with patch("gzip.time.time", return_value=2000):
            second_uri = upload_directory(cfg, str(directories[1]))

    assert first_uri == second_uri
    assert first_uri.endswith("/sourcedir.tar.gz")
    assert len(fake_s3.put_calls) == 1
    content = next(iter(fake_s3.objects.values()))
    assert content[4:8] == b"\x00\x00\x00\x00"
    assert hashlib.sha256(content).hexdigest() in first_uri


def test_directory_upload_changes_when_source_changes(tmp_path):
    cfg = _cfg()
    fake_s3 = FakeS3()
    entry = tmp_path / "entry.py"
    entry.write_text("print('first')\n")
    with patch("vla_pipeline.runner.boto3") as mock_boto:
        mock_boto.client.return_value = fake_s3
        first_uri = upload_directory(cfg, str(tmp_path))
        entry.write_text("print('second')\n")
        second_uri = upload_directory(cfg, str(tmp_path))
    assert first_uri != second_uri
    assert len(fake_s3.put_calls) == 2


def test_stage_sourcedir_contains_shared_modules():
    """The staged sourcedir must contain digest.py, validator.py, defaults.json
    alongside the entry scripts so SageMaker imports work."""

    from vla_pipeline.common.sourcedir import stage

    staged = stage("openvla")
    contents = os.listdir(staged)

    # Entry scripts
    assert "train_entry.py" in contents, f"missing train_entry.py: {contents}"
    assert "eval_entry.py" in contents, f"missing eval_entry.py: {contents}"

    # Shared modules the entries import
    assert "digest.py" in contents, f"missing digest.py: {contents}"
    assert "validator.py" in contents, f"missing validator.py: {contents}"

    # Family config
    assert "defaults.json" in contents, f"missing defaults.json: {contents}"

    # Cleanup
    import shutil
    shutil.rmtree(staged)
