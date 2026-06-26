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
    """Return the DatasetsBucketName output from the Foundation stack.

    Args:
        stack_name: Foundation stack name
        region: AWS region (defaults to resolve_region())

    Returns:
        Bucket name (empty string if not found)
    """
    outputs = stack_outputs(stack_name, region)
    return outputs.get("DatasetsBucketName", "")


def role_arn(stack_name: str = "PhysicalAi-dev-Foundation", region: str | None = None) -> str:
    """Return the SageMakerRoleArn output from the Foundation stack.

    Args:
        stack_name: Foundation stack name
        region: AWS region (defaults to resolve_region())

    Returns:
        Role ARN (empty string if not found)
    """
    outputs = stack_outputs(stack_name, region)
    return outputs.get("SageMakerRoleArn", "")


def ecr_uri(stack_name: str = "PhysicalAi-dev-Foundation", region: str | None = None) -> str:
    """Return the ECR repository URI output from the Foundation stack.

    The key is "ECR" in the Foundation stack (maps to the isaac-lab or groot-training
    repo depending on context). Callers construct the full image URI by appending
    :tag to this base.

    Args:
        stack_name: Foundation stack name
        region: AWS region (defaults to resolve_region())

    Returns:
        ECR repository URI (empty string if not found)
    """
    outputs = stack_outputs(stack_name, region)
    return outputs.get("ECR", "")
