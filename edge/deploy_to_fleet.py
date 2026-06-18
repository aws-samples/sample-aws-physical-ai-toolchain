"""
Deploy the UR3 inference (and optionally telemetry) Greengrass components to the
robot fleet (an IoT thing group) and track rollout status.

Account/region/thing-group are resolved from the caller / env — nothing hardcoded.

Usage:
    # Preview the deployment (no AWS writes):
    python edge/deploy_to_fleet.py --dry-run

    # Deploy to the default fleet thing group:
    python edge/deploy_to_fleet.py --inference-version 1.0.0

    # Check status of the latest deployment:
    python edge/deploy_to_fleet.py --status <DEPLOYMENT_ID>
"""

import argparse
import os
import sys

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")


def _account() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def _thing_group() -> str:
    return os.environ.get("THING_GROUP", f"{PROJECT_NAME}-{ENVIRONMENT}-robots")


def _target_arn(thing_group: str) -> str:
    return f"arn:aws:iot:{REGION}:{_account()}:thinggroup/{thing_group}"


def deploy(inference_version: str, telemetry_version: str | None, dry_run: bool):
    thing_group = _thing_group()
    target = _target_arn(thing_group)
    components = {
        f"com.physicalai.{ENVIRONMENT}.inference": {"componentVersion": inference_version},
    }
    if telemetry_version:
        components[f"com.physicalai.{ENVIRONMENT}.telemetry"] = {"componentVersion": telemetry_version}

    print(f"{'='*60}")
    print(f"  Greengrass fleet deployment")
    print(f"  Target:     {target}")
    print(f"  Components: {components}")
    print(f"  Region:     {REGION}")
    print(f"{'='*60}")

    if dry_run:
        print("\n[dry-run] Would call greengrassv2.create_deployment with the above. No AWS writes.")
        return

    gg = boto3.client("greengrassv2", region_name=REGION)
    resp = gg.create_deployment(
        targetArn=target,
        deploymentName=f"{PROJECT_NAME}-{ENVIRONMENT}-fleet",
        components=components,
    )
    print(f"  Deployment created: {resp['deploymentId']}")
    print(f"  Track with: python edge/deploy_to_fleet.py --status {resp['deploymentId']}")


def status(deployment_id: str):
    gg = boto3.client("greengrassv2", region_name=REGION)
    d = gg.get_deployment(deploymentId=deployment_id)
    print(f"  Deployment: {deployment_id}")
    print(f"  Status:     {d.get('deploymentStatus')}")
    print(f"  Target:     {d.get('targetArn')}")


def main():
    parser = argparse.ArgumentParser(description="Deploy Greengrass components to the robot fleet")
    parser.add_argument("--inference-version", default="1.0.0")
    parser.add_argument("--telemetry-version", default=None,
                        help="Also deploy the telemetry component at this version")
    parser.add_argument("--status", default=None, help="Look up a deployment by id and exit")
    parser.add_argument("--dry-run", action="store_true", help="Preview; make no AWS writes")
    args = parser.parse_args()

    if args.status:
        status(args.status)
        return
    deploy(args.inference_version, args.telemetry_version, args.dry_run)


if __name__ == "__main__":
    main()
