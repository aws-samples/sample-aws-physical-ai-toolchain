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


class _FakeS3:
    """Fake S3 client for checkpoint download tests (additive for eval tests)."""
    def download_file(self, Bucket, Key, Filename):  # noqa: N803 (boto kwargs)
        # Stub: pretend download succeeded by creating an empty file
        import pathlib
        pathlib.Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(Filename).touch()


class _FakeClientError(Exception):
    """Stand-in for botocore ClientError on fake clients."""


class _FakeResourceNotFound(Exception):
    """Stand-in for the SageMaker ResourceNotFound modeled exception."""


class _FakeSageMakerExceptions:
    ClientError = _FakeClientError
    ResourceNotFound = _FakeResourceNotFound


class _FakeSageMaker:
    """Fake SageMaker client for CLI tests."""
    def __init__(self):
        self.calls = []
        self.exceptions = _FakeSageMakerExceptions()

    def create_training_job(self, **kwargs):
        self.calls.append(("create_training_job", kwargs))
        return {}

    def describe_training_job(self, TrainingJobName):  # noqa: N803
        self.calls.append(("describe_training_job", {"TrainingJobName": TrainingJobName}))
        return {
            "TrainingJobName": TrainingJobName,
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
            "CreationTime": "2024-01-01T00:00:00Z",
            "TrainingStartTime": "2024-01-01T00:05:00Z",
            "TrainingEndTime": "2024-01-01T01:00:00Z",
            "ResourceConfig": {
                "InstanceType": "ml.g5.xlarge",
                "InstanceCount": 1,
            },
            "OutputDataConfig": {
                "S3OutputPath": "s3://test-bucket/output/",
            },
        }

    # --- SageMaker Pipeline (pai groot launch / runs) ---

    def create_model_package_group(self, **kwargs):
        self.calls.append(("create_model_package_group", kwargs))
        return {}

    def describe_pipeline(self, PipelineName):  # noqa: N803
        self.calls.append(("describe_pipeline", {"PipelineName": PipelineName}))
        return {"PipelineName": PipelineName}

    def create_pipeline(self, **kwargs):
        self.calls.append(("create_pipeline", kwargs))
        return {}

    def update_pipeline(self, **kwargs):
        self.calls.append(("update_pipeline", kwargs))
        return {}

    def start_pipeline_execution(self, **kwargs):
        self.calls.append(("start_pipeline_execution", kwargs))
        return {"PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:111122223333:pipeline/groot-finetune-pipeline/execution/abc123"}

    def list_pipeline_executions(self, **kwargs):
        self.calls.append(("list_pipeline_executions", kwargs))
        import datetime as _dt
        return {
            "PipelineExecutionSummaries": [
                {
                    "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:111122223333:pipeline/groot-finetune-pipeline/execution/abc123",
                    "PipelineExecutionStatus": "Executing",
                    "CreationTime": _dt.datetime(2026, 6, 30, 12, 0, 0),
                }
            ]
        }

    # --- Real-time endpoint (pai groot deploy / delete) ---

    def create_model(self, **kwargs):
        self.calls.append(("create_model", kwargs))
        return {}

    def create_endpoint_config(self, **kwargs):
        self.calls.append(("create_endpoint_config", kwargs))
        return {}

    def create_endpoint(self, **kwargs):
        self.calls.append(("create_endpoint", kwargs))
        return {}

    def describe_endpoint(self, EndpointName):  # noqa: N803
        self.calls.append(("describe_endpoint", {"EndpointName": EndpointName}))
        return {"EndpointName": EndpointName, "EndpointConfigName": f"{EndpointName}-config"}

    def describe_endpoint_config(self, EndpointConfigName):  # noqa: N803
        self.calls.append(("describe_endpoint_config", {"EndpointConfigName": EndpointConfigName}))
        return {"ProductionVariants": [{"ModelName": "groot-ur3-model"}]}

    def delete_endpoint(self, EndpointName):  # noqa: N803
        self.calls.append(("delete_endpoint", {"EndpointName": EndpointName}))
        return {}

    def delete_endpoint_config(self, EndpointConfigName):  # noqa: N803
        self.calls.append(("delete_endpoint_config", {"EndpointConfigName": EndpointConfigName}))
        return {}

    def delete_model(self, ModelName):  # noqa: N803
        self.calls.append(("delete_model", {"ModelName": ModelName}))
        return {}


class _FakeSageMakerRuntime:
    """Fake sagemaker-runtime client for `pai groot invoke`."""
    def __init__(self):
        self.calls = []

    def invoke_endpoint(self, **kwargs):
        self.calls.append(("invoke_endpoint", kwargs))
        import io
        body = b'{"actions": [[0,0,0,0,0,0,0]], "action_dim": 7}'
        return {"Body": io.BytesIO(body)}


class _FakeBatch:
    """Fake AWS Batch client for CLI tests."""
    def __init__(self):
        self.calls = []

    def submit_job(self, **kwargs):
        self.calls.append(("submit_job", kwargs))
        return {"jobId": "test-job-id"}

    def describe_job_definitions(self, **kwargs):
        self.calls.append(("describe_job_definitions", kwargs))
        # Deployed job def is fixed at 2 nodes (matches the CDK construct default).
        return {
            "jobDefinitions": [
                {
                    "jobDefinitionName": kwargs.get("jobDefinitionName"),
                    "status": "ACTIVE",
                    "nodeProperties": {"numNodes": 2},
                }
            ]
        }

    def describe_jobs(self, jobs):
        self.calls.append(("describe_jobs", {"jobs": jobs}))
        return {
            "jobs": [
                {
                    "jobName": jobs[0],
                    "status": "SUCCEEDED",
                    "jobQueue": "physical-ai-dev-rl-queue",
                    "jobDefinition": "physical-ai-dev-rl-mnp",
                    "createdAt": 1704067200000,
                    "startedAt": 1704067300000,
                    "stoppedAt": 1704070800000,
                    "nodeProperties": {"numNodes": 2},
                }
            ]
        }


class _FakeCloudFormation:
    """Fake CloudFormation client for CLI tests."""
    def __init__(self):
        self.calls = []

    def describe_stacks(self, StackName):  # noqa: N803
        self.calls.append(("describe_stacks", {"StackName": StackName}))
        if "Foundation" in StackName:
            return {
                "Stacks": [
                    {
                        "StackName": StackName,
                        "StackStatus": "CREATE_COMPLETE",
                        "Outputs": [
                            {"OutputKey": "DatasetsBucketName", "OutputValue": "test-bucket-123"},
                            {"OutputKey": "SageMakerRoleArn", "OutputValue": f"arn:aws:iam::{FAKE_ACCOUNT}:role/physical-ai-dev-sagemaker-role"},
                            {"OutputKey": "GrootTrainingRepoUri", "OutputValue": f"{FAKE_ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/physical-ai/groot-training"},
                            {"OutputKey": "ECR", "OutputValue": f"{FAKE_ACCOUNT}.dkr.ecr.us-west-2.amazonaws.com/physical-ai/isaac-lab"},
                        ],
                    }
                ]
            }
        elif "Workstation" in StackName:
            return {
                "Stacks": [
                    {
                        "StackName": StackName,
                        "StackStatus": "CREATE_COMPLETE",
                        "Outputs": [
                            {"OutputKey": "WorkstationInstanceId", "OutputValue": "i-test12345"},
                            {"OutputKey": "SetPassword", "OutputValue": "aws ssm send-command --instance-ids i-test12345 --document-name AWS-RunShellScript --parameters 'commands=[\"echo ubuntu:YOUR_PASSWORD | chpasswd\"]'"},
                        ],
                    }
                ]
            }
        else:
            # Stack not found
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "ValidationError", "Message": "Stack does not exist"}},
                "describe_stacks",
            )


class _FakeEC2:
    """Fake EC2 client for CLI tests."""
    def __init__(self):
        self.calls = []

    def describe_instances(self, **kwargs):
        self.calls.append(("describe_instances", kwargs))
        instance_id = kwargs.get("InstanceIds", ["i-test12345"])[0]
        return {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": instance_id,
                            "State": {"Name": "running"},
                            "InstanceType": "g6e.4xlarge",
                            "PublicIpAddress": "203.0.113.42",
                        }
                    ]
                }
            ]
        }

    def start_instances(self, InstanceIds):  # noqa: N803
        self.calls.append(("start_instances", {"InstanceIds": InstanceIds}))
        return {}

    def stop_instances(self, InstanceIds):  # noqa: N803
        self.calls.append(("stop_instances", {"InstanceIds": InstanceIds}))
        return {}

    def get_waiter(self, waiter_name):
        """Return a fake waiter that does nothing."""
        class _FakeWaiter:
            def wait(self, **kwargs):
                pass
        return _FakeWaiter()


@pytest.fixture
def fake_boto3(monkeypatch):
    """Patch boto3.client so no real AWS calls happen. Returns the account id."""
    import boto3

    # Create persistent client instances so tests can inspect .calls
    clients = {
        "sts": _FakeSTS(),
        "ssm": _FakeSSM(),
        "s3": _FakeS3(),
        "sagemaker": _FakeSageMaker(),
        "sagemaker-runtime": _FakeSageMakerRuntime(),
        "batch": _FakeBatch(),
        "cloudformation": _FakeCloudFormation(),
        "ec2": _FakeEC2(),
    }

    def _client(service, *a, **k):
        client = clients.get(service)
        if client:
            return client
        # Any other client should NOT be invoked during a --dry-run; fail loudly.
        raise AssertionError(
            f"unexpected boto3.client('{service}') during dry-run — script made a real AWS call"
        )

    monkeypatch.setattr(boto3, "client", _client)
    # Return the account AND the clients dict so tests can inspect calls
    return FAKE_ACCOUNT, clients


def run_script(path, args, monkeypatch=None):
    """Import a script module fresh and invoke its main() with argv."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_t_{pathlib.Path(path).stem}", REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
