"""
Create / publish the UR3 inference Greengrass component version.

The CDK EdgeStack (cdk/lib/edge-stack.ts) defines the component recipe inline and
publishes it at deploy time. This script is the *imperative* alternative for
iterating on the component without a full `cdk deploy`: it uploads the model
artifact to S3 and publishes a new component version via the Greengrass v2 API.

Account/region/bucket are resolved from the caller — nothing hardcoded.

Usage:
    # Show exactly what would be published, no AWS writes:
    python edge/create_component.py --model ./model_exported/policy.trt --dry-run

    # Actually publish version 1.0.1 of the inference component:
    python edge/create_component.py --model ./model_exported/policy.trt \
        --component-version 1.0.1
"""

import argparse
import json
import os
import sys

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")


def _account() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def _models_bucket() -> str:
    return os.environ.get("MODELS_BUCKET", f"{PROJECT_NAME}-{ENVIRONMENT}-models-{_account()}")


def _inference_image() -> str:
    return f"{_account()}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT_NAME}/inference:latest"


def build_recipe(component_name: str, version: str, models_bucket: str, image_uri: str) -> dict:
    """Build the Greengrass v2 component recipe (mirrors cdk/lib/edge-stack.ts)."""
    run_script = (
        "docker run --rm --runtime nvidia --network host "
        "-v {artifacts:path}/model:/model "
        f"{image_uri} "
        "--model-path /model/policy.trt "
        "--ros-topic /ur3/joint_commands"
    )
    manifest_platform = lambda arch: {
        "Platform": {"os": "linux", "architecture": arch},
        "Lifecycle": {"run": {"Script": run_script}},
        "Artifacts": [{"Uri": f"s3://{models_bucket}/latest/policy.trt", "Unarchive": "NONE"}],
    }
    return {
        "RecipeFormatVersion": "2020-01-25",
        "ComponentName": component_name,
        "ComponentVersion": version,
        "ComponentDescription": "TensorRT inference node for UR3 pick-and-place policy",
        "ComponentPublisher": "PhysicalAI",
        "ComponentDependencies": {
            "aws.greengrass.DockerApplicationManager": {"VersionRequirement": ">=2.0.0"},
            "aws.greengrass.TokenExchangeService": {"VersionRequirement": ">=2.0.0"},
        },
        "Manifests": [manifest_platform("aarch64"), manifest_platform("x86_64")],
        "ComponentConfiguration": {"DefaultConfiguration": {"containerUri": image_uri}},
    }


def main():
    parser = argparse.ArgumentParser(description="Publish the UR3 inference Greengrass component")
    parser.add_argument("--model", required=True, help="Local path to the exported policy.trt")
    parser.add_argument("--component-name", default=None,
                        help="Defaults to com.physicalai.<env>.inference")
    parser.add_argument("--component-version", default="1.0.0")
    parser.add_argument("--dry-run", action="store_true", help="Print actions; make no AWS writes")
    args = parser.parse_args()

    component_name = args.component_name or f"com.physicalai.{ENVIRONMENT}.inference"
    models_bucket = _models_bucket()
    image_uri = _inference_image()
    model_key = "latest/policy.trt"
    recipe = build_recipe(component_name, args.component_version, models_bucket, image_uri)

    print(f"{'='*60}")
    print(f"  Publish Greengrass component")
    print(f"  Component: {component_name} v{args.component_version}")
    print(f"  Image:     {image_uri}")
    print(f"  Model:     {args.model} -> s3://{models_bucket}/{model_key}")
    print(f"  Region:    {REGION}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[dry-run] Would upload the model artifact and publish this recipe:\n")
        print(json.dumps(recipe, indent=2))
        print("\n[dry-run] No AWS calls made.")
        return

    if not os.path.isfile(args.model):
        print(f"ERROR: model file not found: {args.model}")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=REGION)
    print(f"  Uploading model -> s3://{models_bucket}/{model_key}")
    s3.upload_file(args.model, models_bucket, model_key)

    gg = boto3.client("greengrassv2", region_name=REGION)
    resp = gg.create_component_version(inlineRecipe=json.dumps(recipe).encode())
    print(f"  Published: {resp['arn']}  (status: {resp['componentState']})")


if __name__ == "__main__":
    main()
