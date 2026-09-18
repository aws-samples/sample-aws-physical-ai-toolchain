"""Offline configuration values do not require Foundation discovery."""
from __future__ import annotations

import dataclasses
from unittest.mock import patch

import pytest

from vla_pipeline.config import PipelineConfig

_REQUIRED = {
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


def test_explicit_configuration_is_offline():
    with patch("vla_pipeline.config.boto3.Session", side_effect=AssertionError("AWS called")):
        cfg = PipelineConfig(**_REQUIRED)
    assert cfg.account_id == _REQUIRED["account_id"]
    assert cfg.bucket == _REQUIRED["bucket"]


def test_configuration_is_immutable():
    cfg = PipelineConfig(**_REQUIRED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.bucket = "another-bucket"


@pytest.mark.parametrize("missing", list(_REQUIRED))
def test_explicit_configuration_requires_all_resource_fields(missing):
    values = {key: value for key, value in _REQUIRED.items() if key != missing}
    with pytest.raises(TypeError, match=missing):
        PipelineConfig(**values)


def test_s3_uri_namespacing():
    cfg = PipelineConfig(**_REQUIRED)
    assert cfg.s3_uri("train", "job1") == "s3://my-bucket/vla-pipeline/train/job1"
    assert cfg.s3_uri() == "s3://my-bucket/vla-pipeline"
    assert cfg.s3_uri("/train/", "", "/job1/") == cfg.s3_uri("train", "job1")
