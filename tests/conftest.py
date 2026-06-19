"""Shared pytest fixtures: make AWS-touching scripts runnable offline.

The edge/cosmos/RL launchers resolve the account via sts.get_caller_identity even
in --dry-run. We monkeypatch boto3.client so tests need no real AWS credentials.
"""
import os
import sys
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
# Make `import training...` work and let scripts find the repo.
sys.path.insert(0, str(REPO))

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("PROJECT_NAME", "physical-ai")
os.environ.setdefault("ENVIRONMENT", "dev")

FAKE_ACCOUNT = "111122223333"


class _FakeSTS:
    def get_caller_identity(self):
        return {"Account": FAKE_ACCOUNT}


class _FakeSSM:
    def get_parameter(self, Name):  # noqa: N803 (boto kwarg)
        return {"Parameter": {"Value": "ami-deadbeef"}}


@pytest.fixture
def fake_boto3(monkeypatch):
    """Patch boto3.client so no real AWS calls happen. Returns the account id."""
    import boto3

    def _client(service, *a, **k):
        if service == "sts":
            return _FakeSTS()
        if service == "ssm":
            return _FakeSSM()
        # Any other client should NOT be invoked during a --dry-run; fail loudly.
        raise AssertionError(
            f"unexpected boto3.client('{service}') during dry-run — script made a real AWS call"
        )

    monkeypatch.setattr(boto3, "client", _client)
    return FAKE_ACCOUNT


def run_script(path, args, monkeypatch=None):
    """Import a script module fresh and invoke its main() with argv."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_t_{pathlib.Path(path).stem}", REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
