"""N1: the registration ContentDigest must satisfy the SageMaker service contract.

`MetricsSource.ContentDigest` is constrained by the service model to

    [Ss][Hh][Aa]256:[0-9a-fA-F]{64}

so bare hexadecimal is rejected at the API. The producer emitted bare hex and the verifier
and its fixture repeated the same mistake, so their agreement did not establish that the
registration argument was valid -- it only established that two wrong values matched.

These tests read the pattern from the INSTALLED service model rather than hardcoding it, so
they check the real contract the API will apply.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _service_model_pattern() -> str:
    """The ContentDigest pattern from botocore's own SageMaker service model."""
    botocore = pytest.importorskip("botocore")
    directory = pathlib.Path(botocore.__file__).parent / "data/sagemaker/2017-07-24"
    plain = directory / "service-2.json"
    if plain.exists():
        model = json.loads(plain.read_text())
    else:
        model = json.loads(gzip.decompress((directory / "service-2.json.gz").read_bytes()))
    return model["shapes"]["ContentDigest"]["pattern"]


def test_bare_hex_does_not_satisfy_the_service_contract():
    """The precise defect: the previous value could never have registered."""
    pattern = re.compile(_service_model_pattern())
    assert not pattern.fullmatch("c" * 64)
    assert pattern.fullmatch("sha256:" + "c" * 64)


def test_the_producer_emits_a_digest_the_api_accepts():
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert source.count('"attestation_content_digest": f"sha256:{attestation_digest}"') == 2, (
        "both the local and S3 promotion paths must emit the API-formatted digest")
    pattern = re.compile(_service_model_pattern())
    # Build the value the same way the producer does and check it against the contract.
    assert pattern.fullmatch(f"sha256:{'a' * 64}")


def test_the_pipeline_registers_the_prefixed_field_not_the_raw_hex():
    source = (_REPO_ROOT / "src/vla_pipeline/pipeline.py").read_text()
    assert 'json_path="promotion.attestation_content_digest"' in source
    assert 'json_path="promotion.attestation_sha256"' not in source, (
        "the raw hex must not be passed to the API")


def test_the_raw_hex_is_still_available_for_content_addressing():
    """The prefixed form is for the API only; keys and comparisons need raw hex."""
    source = (_REPO_ROOT / "entrypoints/validate_entry.py").read_text()
    assert '"attestation_sha256": attestation_digest' in source


def test_the_verifier_checks_the_argument_against_the_contract():
    source = (_REPO_ROOT / "scripts/verify_evaluation_cell.py").read_text()
    assert "_SAGEMAKER_CONTENT_DIGEST" in source
    # And the verifier's own pattern must be the service model's pattern.
    match = re.search(r'_SAGEMAKER_CONTENT_DIGEST = re\.compile\(r"([^"]+)"\)', source)
    assert match, "the verifier must declare the pattern it enforces"
    assert match.group(1) == _service_model_pattern(), (
        "the verifier's pattern has drifted from the SageMaker service model")
