"""A volume larger than an instance's LOCAL storage is rejected before submission.

From a real run:

    Invalid VolumeSizeInGB: 300 GB. The requested instance type ml.g6e.xlarge includes local
    instance storage with a fixed total size of 250 GB. Please reduce VolumeSizeInGB in the
    request or choose a different instance type.

It is a CAP, not a prohibition: ml.g5.12xlarge and ml.g6e.12xlarge both have 3800 GB of local storage
and accept 300 GB fine. My first reading was "omit VolumeSizeInGB for local-storage instances", which
would have been wrong -- and is recorded here because the corrected rule is not obvious from the error.

Checked against the EC2 API rather than a hardcoded table: a table would be a second declaration of a
fact AWS publishes, and it would rot silently as instance families are added.
"""
import pytest

from vla_pipeline.registry import RegistryError, check_storage_fits


class _FakeEC2:
    """describe_instance_types with the REAL figures for the types this component uses."""

    LOCAL_GB = {
        "g6e.xlarge": 250,        # rejected a 300 GB request in production
        "g6e.12xlarge": 3800,
        "g5.12xlarge": 3800,
        "g5.2xlarge": 236,
        "m5.large": None,         # EBS-only: no local-storage cap
    }

    def __init__(self):
        self.calls = []

    def describe_instance_types(self, InstanceTypes):
        self.calls.append(InstanceTypes)
        bare = InstanceTypes[0]
        if bare not in self.LOCAL_GB:
            return {"InstanceTypes": []}
        local = self.LOCAL_GB[bare]
        info = {"InstanceStorageInfo": {"TotalSizeInGB": local}} if local else {}
        return {"InstanceTypes": [info]}


class _BrokenEC2:
    def describe_instance_types(self, InstanceTypes):
        raise RuntimeError("AuthFailure: AWS was not able to validate the provided access credentials")


def test_the_production_rejection_is_caught_before_submission():
    """The exact case: 300 GB onto ml.g6e.xlarge, which has 250 GB of local NVMe."""
    with pytest.raises(RegistryError, match="local instance storage"):
        check_storage_fits("ml.g6e.xlarge", 300, ec2_client=_FakeEC2())


def test_a_volume_within_the_local_total_is_accepted():
    """A CAP, not a prohibition -- the same 300 GB is fine on an instance with 3800 GB."""
    ec2 = _FakeEC2()
    check_storage_fits("ml.g6e.12xlarge", 300, ec2_client=ec2)
    check_storage_fits("ml.g5.12xlarge", 300, ec2_client=ec2)
    check_storage_fits("ml.g6e.xlarge", 250, ec2_client=ec2)      # exactly at the limit
    assert ec2.calls, "the check never queried EC2, so it cannot have verified anything"


def test_an_ebs_only_instance_has_no_cap():
    check_storage_fits("ml.m5.large", 5000, ec2_client=_FakeEC2())


def test_the_ml_prefix_is_stripped_for_the_ec2_api():
    """SageMaker types carry `ml.`; the EC2 API does not accept it."""
    ec2 = _FakeEC2()
    check_storage_fits("ml.g6e.12xlarge", 100, ec2_client=ec2)
    assert ec2.calls[0] == ["g6e.12xlarge"], (
        f"queried EC2 with {ec2.calls[0]}; the ml. prefix must be stripped or every lookup returns "
        f"nothing and the check silently verifies nothing")


def test_an_unknown_instance_type_is_reported_not_assumed():
    with pytest.raises(RegistryError, match="does not recognise"):
        check_storage_fits("ml.g99.imaginary", 100, ec2_client=_FakeEC2())


def test_an_unreachable_ec2_is_reported_as_UNVERIFIED_and_does_not_block(capsys):
    """The asymmetry that justifies degrading here but not in the identity checks.

    This runs in unit tests without credentials and on hosts whose role lacks
    ec2:DescribeInstanceTypes. Hard-failing would block every submission on what is an EARLY WARNING:
    CreateTrainingJob still enforces the real limit, so not knowing costs a fast, free rejection
    rather than a wrong result. But it must SAY SO -- silence would read as a pass.
    """
    check_storage_fits("ml.g6e.xlarge", 300, ec2_client=_BrokenEC2())
    err = capsys.readouterr().err
    assert "UNVERIFIED" in err, (
        "an unreachable EC2 produced no warning, so a caller cannot distinguish 'checked and fits' "
        "from 'never checked'")
    assert "ml.g6e.xlarge" in err and "300" in err


def test_a_malformed_volume_is_refused_before_any_api_call():
    ec2 = _FakeEC2()
    for bad in (0, -1, True, 1.5, "300", None):
        with pytest.raises(RegistryError):
            check_storage_fits("ml.g6e.12xlarge", bad, ec2_client=ec2)
    assert not ec2.calls, "a malformed volume must fail before spending an API call"
