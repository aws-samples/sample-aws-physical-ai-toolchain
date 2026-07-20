"""CloudFormation stack output reader — wraps boto3 describe-stacks."""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from pai.config import resolve_region
from pai.helpers import friendly_boto_error


def stack_outputs(stack_name: str, region: str | None = None) -> dict:
    """Return CloudFormation stack outputs as a dict (OutputKey -> OutputValue).

    Args:
        stack_name: Stack name
        region: AWS region (defaults to resolve_region())

    Returns:
        Dict of outputs. Empty dict if stack not found.
    """
    region = region or resolve_region()
    try:
        cfn = boto3.client("cloudformation", region_name=region)
        resp = cfn.describe_stacks(StackName=stack_name)
        stacks = resp.get("Stacks", [])
        if not stacks:
            return {}
        outputs = stacks[0].get("Outputs", [])
        return {o["OutputKey"]: o["OutputValue"] for o in outputs}
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ValidationError":
            # Stack does not exist
            return {}
        friendly_boto_error(e)
    except Exception as e:
        friendly_boto_error(e)
    return {}


def stack_status(stack_name: str, region: str | None = None) -> str | None:
    """Return CloudFormation stack status (e.g., CREATE_COMPLETE) or None if absent.

    Args:
        stack_name: Stack name
        region: AWS region (defaults to resolve_region())

    Returns:
        Stack status string or None
    """
    region = region or resolve_region()
    try:
        cfn = boto3.client("cloudformation", region_name=region)
        resp = cfn.describe_stacks(StackName=stack_name)
        stacks = resp.get("Stacks", [])
        if not stacks:
            return None
        return stacks[0].get("StackStatus")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ValidationError":
            return None
        friendly_boto_error(e)
    except Exception as e:
        friendly_boto_error(e)
    return None


def bucket(stack_name: str = "PhysicalAi-dev-Foundation", region: str | None = None) -> str:
    """Return the datasets bucket name.

    Tries SSM parameter first (set by Terraform/CFN bootstrap), falls back to
    CloudFormation stack outputs for backward compatibility.
    """
    region = region or resolve_region()
    # Try SSM first (new path: Terraform / CFN bootstrap)
    try:
        ssm = boto3.client("ssm", region_name=region)
        resp = ssm.get_parameter(Name="/physical-ai/datasets-bucket")
        return resp["Parameter"]["Value"]
    except Exception:
        pass
    # Fall back to CFN stack outputs (legacy CDK path)
    outputs = stack_outputs(stack_name, region)
    return outputs.get("DatasetsBucketName", "")


def role_arn(stack_name: str = "PhysicalAi-dev-Foundation", region: str | None = None) -> str:
    """Return the SageMaker execution role ARN.

    Tries SSM parameter first, falls back to CloudFormation stack outputs.
    """
    region = region or resolve_region()
    try:
        ssm = boto3.client("ssm", region_name=region)
        resp = ssm.get_parameter(Name="/physical-ai/sagemaker-role-arn")
        return resp["Parameter"]["Value"]
    except Exception:
        pass
    outputs = stack_outputs(stack_name, region)
    return outputs.get("SageMakerRoleArn", "")


def ecr_uri(stack_name: str = "PhysicalAi-dev-Foundation", region: str | None = None, repo: str = "groot-training") -> str:
    """Return the ECR repository URI.

    Tries SSM parameter first, falls back to CloudFormation stack outputs.
    """
    region = region or resolve_region()
    try:
        ssm = boto3.client("ssm", region_name=region)
        resp = ssm.get_parameter(Name=f"/physical-ai/ecr/{repo}")
        return resp["Parameter"]["Value"]
    except Exception:
        pass
    outputs = stack_outputs(stack_name, region)
    return outputs.get("GrootTrainingRepoUri", "")
