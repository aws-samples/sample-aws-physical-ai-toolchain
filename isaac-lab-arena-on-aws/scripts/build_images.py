#!/usr/bin/env python3
"""Build and push VLA family LIBERO train/eval Docker images via AWS CodeBuild.

Uses the component's Terraform-managed CodeBuild project and ECR repositories,
discovered through SSM. Deploy infra/ first. Builds run sequentially: this command
uploads source, streams logs and waits for each build to finish, failing on error.

Usage:
    PYTHONPATH=src python scripts/build_images.py --family openvla --tag review-01
    PYTHONPATH=src python scripts/build_images.py --family all --tag review-01
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile

import boto3

sys.path.insert(0, "src")
from vla_pipeline.config import load_config, resolve_codebuild_project

FAMILIES = ["openvla", "molmoact2", "gr00t"]
# NO hardcoded project name. This defaulted to "vla-image-build" and never discovered the deployed
# value; in an account where another deployment already owns a project by that name, every build
# submitted its source to THIS component's trust bucket and then triggered a project whose role
# cannot read it -- failing at DOWNLOAD_SOURCE with a 403 naming a role Terraform does not manage.
# Discovered from the same SSM contract that publishes the roles and the prefix.


def create_source_zip(family: str) -> bytes:
    """Create a zip of the Docker context + buildspec for CodeBuild."""
    buf = io.BytesIO()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add the buildspec
        buildspec_path = os.path.join(repo_root, "docker", "buildspec.yml")
        zf.write(buildspec_path, "buildspec.yml")

        # Add the family's Dockerfile
        docker_dir = os.path.join(repo_root, "docker", family)
        for entry in os.listdir(docker_dir):
            full_path = os.path.join(docker_dir, entry)
            if os.path.isfile(full_path):
                zf.write(full_path, f"docker/{family}/{entry}")

    return buf.getvalue()


def upload_source(s3, bucket: str, family: str, source_bytes: bytes) -> str:
    """Upload the source zip to S3 for CodeBuild. Returns the S3 key."""
    # I1: this uploaded to the SHARED models bucket while build_arena_connector.py was migrated
    # to the trust bucket's code/v1/* namespace and the CodeBuild role's read was narrowed to
    # that namespace. The same CodeBuild project serves BOTH builders (infra/main.tf), so a fresh
    # deployment could no longer download this source at all -- only broad pre-existing account
    # permissions would conceal it. Two builders, one migrated: the same shape of defect the
    # migration was fixing.
    #
    # Full digest, not 12 characters: the key asserts the content, so it should assert all of it.
    digest = hashlib.sha256(source_bytes).hexdigest()
    key = f"code/v1/{digest}/{family}-source.zip"
    s3.put_object(Bucket=bucket, Key=key, Body=source_bytes)
    # Read back and compare: a same-key replacement between write and build is the attack this
    # namespace exists to prevent, so the publisher confirms what is actually stored.
    published = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if hashlib.sha256(published).hexdigest() != digest:
        raise SystemExit(
            f"FATAL: build source at s3://{bucket}/{key} does not match what was uploaded; "
            f"refusing to start a build from unverified source.")
    print(f"  Source uploaded: s3://{bucket}/{key}", flush=True)
    return key


def start_and_wait(cb, family: str, ecr_uri: str, tag: str, bucket: str,
                   source_key: str, region: str, project_name: str) -> bool:
    """Start a build on the discovered CodeBuild project and stream
    logs until complete. Overrides the S3 source + buildspec per build."""
    resp = cb.start_build(
        projectName=project_name,
        sourceTypeOverride="S3",
        sourceLocationOverride=f"{bucket}/{source_key}",
        buildspecOverride="buildspec.yml",
        environmentVariablesOverride=[
            {"name": "FAMILY", "value": family, "type": "PLAINTEXT"},
            {"name": "ECR_REPO", "value": ecr_uri, "type": "PLAINTEXT"},
            {"name": "IMAGE_TAG", "value": tag, "type": "PLAINTEXT"},
            {"name": "AWS_REGION", "value": region, "type": "PLAINTEXT"},
        ],
    )
    build_id = resp["build"]["id"]
    print(f"  Build started: {build_id}", flush=True)

    while True:
        builds = cb.batch_get_builds(ids=[build_id])
        build = builds["builds"][0]
        status = build["buildStatus"]
        phase = build.get("currentPhase", "?")

        if status == "IN_PROGRESS":
            print(f"    [{phase}] ...", flush=True)
            time.sleep(15)
        elif status == "SUCCEEDED":
            print(f"\n  BUILD SUCCEEDED: {ecr_uri}:{tag}")
            return True
        else:
            print(f"\n  BUILD FAILED: status={status}")
            print(f"    Phases: {json.dumps(build.get('phases', []), default=str)[:500]}")
            return False


def build_family(cfg, family: str, tag: str, project_name: str) -> str:
    """Submit one family's image build to the shared CodeBuild project."""
    print(f"\n{'='*60}\n  Building: {family}\n{'='*60}")

    region = cfg.region
    cb = boto3.client("codebuild", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    # ECR repo is Terraform-managed (infra/). Reference it by its canonical URI;
    # the vla-image-build role is scoped to push to these repos.
    ecr_uri = f"{cfg.account_id}.dkr.ecr.{region}.amazonaws.com/vla/{family}"

    source_bytes = create_source_zip(family)
    # I1: the trust bucket, matching build_arena_connector.py and the narrowed read grant.
    source_key = upload_source(s3, cfg.trust_bucket, family, source_bytes)

    success = start_and_wait(
        cb, family, ecr_uri, tag, cfg.trust_bucket, source_key, region, project_name)
    if not success:
        print(f"\n  FAILED: {family} image build failed")
        sys.exit(1)
    return f"{ecr_uri}:{tag}"


def main():
    parser = argparse.ArgumentParser(description="Build VLA Docker images via the component's CodeBuild project")
    parser.add_argument("--family", required=True,
                        help="Family to build (openvla|molmoact2|gr00t|all)")
    parser.add_argument("--project-name", default=None,
                        help="Override the CodeBuild project name. Defaults to the value the "
                             "component's infrastructure published to SSM.")
    parser.add_argument("--tag", required=True,
                        help="Image tag (REQUIRED). ECR repos are tag-immutable, so "
                             "pick a fresh tag per build (e.g. a version or short commit sha); "
                             "reusing an existing tag fails the push.")
    args = parser.parse_args()

    cfg = load_config()

    if args.family != "all" and args.family not in FAMILIES:
        print(f"Unknown family: {args.family}. Options: {FAMILIES}")
        sys.exit(1)
    families = FAMILIES if args.family == "all" else [args.family]

    # Resolved ONCE, before any upload, so a misconfigured builder fails before bytes are written to
    # the trust bucket rather than after a project we do not own rejects them.
    project = resolve_codebuild_project(cfg, args.project_name)
    print(f"  CodeBuild project: {project}")

    results = {}
    for family in families:
        results[family] = build_family(cfg, family, args.tag, project)

    print(f"\n{'='*60}\n  BUILD SUMMARY\n{'='*60}")
    for family, uri in results.items():
        print(f"  {family}: {uri}")


if __name__ == "__main__":
    main()
