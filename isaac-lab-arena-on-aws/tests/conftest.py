"""Pytest configuration for the offline test suite.

The pipeline tests build a SageMaker ``PipelineSession`` to serialize the
pipeline definition offline (``pipeline.definition()``); this never calls AWS.
However, ``sagemaker.Session`` still requires a resolvable region at
construction time. In a clean CI container there is no ``~/.aws/config`` and no
``AWS_REGION``/``AWS_DEFAULT_REGION``, so the region resolves to ``None`` and
sagemaker raises "Must setup local AWS configuration with a region supported by
SageMaker." before any assertion runs.

Set a dummy region (and placeholder credentials) when unset so the offline
suite is hermetic and runs anywhere without real AWS configuration. ``setdefault``
never overrides a real value, so local runs with configured AWS are unaffected.
No test in this suite makes a live AWS call.
"""
import os

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
