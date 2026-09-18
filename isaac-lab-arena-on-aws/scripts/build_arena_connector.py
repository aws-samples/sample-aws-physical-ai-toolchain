#!/usr/bin/env python3
"""build_arena_connector.py — Build the GR00T Isaac Arena eval image via CodeBuild.

Builds the baked GR00T Arena *connector* image using the component's
CodeBuild project (privileged docker-in-docker). Non-blocking: submits and prints
the build id + a poll command. The connector buildspec's docker context is the
toolchain root.

TWO IMAGES, TWO REPOS
---------------------
  base       Isaac Sim + Isaac Lab Arena. Build it first using the base recipe
             in README.md, "Build the container images".
  connector  base + the GR00T server deps + the baked eval entry. Built here,
             pushed to the pair manifest's `eval_image` repo.

The base is resolved from the manifest and verified to exist in ECR before a
build is submitted. Both build recipes are included in this repository.

Only GR00T × Arena is supported: OpenVLA/MolmoAct2 × Arena are embodiment-blocked
(7-DoF Franka policies cannot emit the 26-dim GR1 action) and excluded from the
component — see the README compatibility matrix.

Usage:
    # builds the tag config/pairs/gr00t--isaac_arena.yaml declares
    PYTHONPATH=src python scripts/build_arena_connector.py --connector gr00t
    PYTHONPATH=src python scripts/build_arena_connector.py --connector gr00t --tag v2 --wait
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import time
import zipfile
from dataclasses import replace
from typing import NamedTuple

import boto3

sys.path.insert(0, "src")
from vla_pipeline.config import (load_arena_repository, load_config,
                                 resolve_codebuild_project)
from vla_pipeline.registry import resolve

COMPONENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(COMPONENT_ROOT)
COMPONENT = os.path.basename(COMPONENT_ROOT)
#: The pair whose eval image this script builds. Arena ships gr00t only.
SIMULATOR = "isaac_arena"

BUILDSPEC = {
    "gr00t": "containers/isaac-lab-arena/buildspec.yml",
}


def _required_zip_members(connector: str) -> list:
    """Files the connector's Dockerfile COPYs (repo-relative) that MUST be in the
    source zip. Missing -> fail loud (an empty/partial context surfaces as a
    confusing 'buildspec not found' or Docker COPY error in CodeBuild)."""
    return [
        BUILDSPEC[connector],
        "containers/isaac-lab-arena/Dockerfile",
        f"{COMPONENT}/entrypoints/eval/isaac_arena/{connector}/eval_entry.py",
        f"{COMPONENT}/entrypoints/eval/isaac_arena/_shared/digest.py",
        f"{COMPONENT}/entrypoints/eval/isaac_arena/_shared/docker_entrypoint_multi.sh",
        f"{COMPONENT}/entrypoints/eval/isaac_arena/gr00t/_verify_gr00t.py",
        f"{COMPONENT}/entrypoints/eval/isaac_arena/gr00t/gr00t_n17_n16_action_adapter.py",
        f"{COMPONENT}/entrypoints/eval/isaac_arena/gr00t/gr00t_n17_n16_action_adapter.pth",
        # The seeding server wrapper. Omitting a file here means it silently never
        # reaches CodeBuild, so the image would launch the pinned server directly and the
        # policy RNG would be unbound again with no visible failure.
        f"{COMPONENT}/entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py",
        f"{COMPONENT}/entrypoints/train/gr00t/arena_gr1_data_config.py",
        f"{COMPONENT}/entrypoints/train/gr00t/defaults.json",
        f"{COMPONENT}/src/vla_pipeline/common/validator.py",
        f"{COMPONENT}/src/vla_pipeline/common/capped_reader.py",
        f"{COMPONENT}/src/vla_pipeline/common/source_identity.py",
        f"{COMPONENT}/src/vla_pipeline/common/checkpoint_compat.py",
    ]


def create_source_zip(connector: str) -> bytes:
    """Zip the repo-root `entrypoints/` tree (the Docker build context -- the Arena
    Dockerfile COPYs from entrypoints/train/... AND entrypoints/eval/isaac_arena/...).
    The zip root maps to the CodeBuild context root, so repo-relative COPY paths
    resolve. Fail loud if a required build input is missing rather than submit a
    partial context."""
    buf = io.BytesIO()
    members = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in _required_zip_members(connector):
            full = os.path.join(REPO_ROOT, rel)
            if not os.path.isfile(full):
                raise FileNotFoundError(f"required connector build input missing: {full}")
            zf.write(full, rel)
            members.add(rel)
    missing = [m for m in _required_zip_members(connector) if m not in members]
    if missing:
        raise RuntimeError(
            f"create_source_zip({connector!r}): required build inputs missing from "
            f"the source zip: {missing}. Refusing to submit a partial context.")
    return buf.getvalue()


def _split_repo_tag(repo_tag: str, field: str) -> tuple:
    if ":" not in repo_tag:
        raise SystemExit(
            f"FATAL: {field}={repo_tag!r} must be 'repo:tag'.")
    repo, tag = repo_tag.rsplit(":", 1)
    if not repo or not tag:
        raise SystemExit(f"FATAL: {field}={repo_tag!r} must be 'repo:tag'.")
    return repo, tag


class BuildTarget(NamedTuple):
    """Everything the build needs, resolved from the pair manifest + overrides."""
    repo: str               # e.g. "vla/isaac-arena-gr00t"
    tag: str                # what gets built AND what SimEval runs
    ecr_repo: str           # "<registry>/<repo>"
    base_repo_tag: str      # as declared/overridden, for the ECR existence check
    base_image_uri: str     # the Dockerfile's ARENA_BASE_IMAGE


def is_full_registry_uri(repo_tag: str) -> bool:
    """True when `repo_tag` already names a registry host.

    A hostname in the first path segment means the caller supplied a full URI, so
    prefixing our own registry would produce
    `<acct>.dkr.ecr…/<other-acct>.dkr.ecr…/…` -- an unpullable FROM that only fails
    minutes into CodeBuild, the exact failure mode the base check exists to remove.
    """
    return "." in repo_tag.split("/")[0]


def resolve_build_target(spec, connector: str, registry: str, *,
                         tag: str | None = None, base_image: str | None = None,
                         skip_base_check: bool = False) -> BuildTarget:
    """Resolve repo/tag/base for a connector build. Pure -- no AWS, no argparse.

    Extracted from ``main()`` deliberately: while this logic lived as locals inside
    ``main()`` it was not callable, so its tests asserted on the script's SOURCE
    TEXT instead of its behaviour -- and a bug reintroduced as a comment (leaving
    the asserted literal in place) passed. Behaviour has to be reachable to be
    pinned.
    """
    repo, declared_tag = _split_repo_tag(spec.eval_image_repo, "eval_image")
    base_repo_tag = base_image or spec.eval_base_image_repo
    if not base_repo_tag:
        raise SystemExit(
            f"FATAL: no Arena base image declared. Set `eval_base_image` in "
            f"config/pairs/{connector}--{SIMULATOR}.yaml or pass --base-image. "
            f"See README.md#arena-images-and-connector-contract.")

    full_uri = is_full_registry_uri(base_repo_tag)
    if full_uri and not skip_base_check:
        raise SystemExit(
            f"FATAL: --base-image {base_repo_tag!r} names an explicit registry, "
            f"which this account's ECR cannot be queried for. Re-run with "
            f"--skip-base-check to accept it unverified.")

    if repo == base_repo_tag.rsplit(":", 1)[0]:
        # The old layout. Pushing the derived image into its own base's repo means
        # the next fresh-account build has no resolvable FROM.
        raise SystemExit(
            f"FATAL: the connector target repo ({repo}) is the SAME repo as "
            f"its base image ({base_repo_tag}).\n"
            f"  Pushing the derived image there makes the repo its own base, which "
            f"is why a fresh account could not build this image.\n"
            f"  Point `eval_image` in config/pairs/{connector}--{SIMULATOR}.yaml "
            f"at a separate repo (e.g. vla/isaac-arena-{connector}).")

    return BuildTarget(
        repo=repo,
        # Default to the manifest's tag so the documented build produces exactly
        # the image the pipeline submits.
        tag=tag or declared_tag,
        ecr_repo=f"{registry}/{repo}",
        base_repo_tag=base_repo_tag,
        base_image_uri=base_repo_tag if full_uri else f"{registry}/{base_repo_tag}",
    )


def verify_base_image(ecr, base_repo_tag: str, region: str) -> None:
    """Fail fast if the Arena base image is absent from ECR.

    Build the base before submitting the connector so a missing prerequisite
    is reported here rather than during CodeBuild's Docker FROM pull.
    """
    repo, tag = _split_repo_tag(base_repo_tag, "eval_base_image")
    try:
        ecr.describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])
    except ecr.exceptions.RepositoryNotFoundException:
        raise SystemExit(
            f"FATAL: the Arena base ECR repository {repo!r} does not exist in "
            f"{region}.\n"
            f"  Follow README.md, 'Build the container images': deploy the component\n"
            f"  ECR resources and run its Arena base build before the connector.\n"
            f"  Required base tag: {tag!r}; recipe: "
            f"entrypoints/eval/isaac_arena/base/buildspec_base.yml.")
    except ecr.exceptions.ImageNotFoundException:
        raise SystemExit(
            f"FATAL: the Arena base image {base_repo_tag!r} is not present in ECR "
            f"({region}).\n"
            f"  Run the Arena base build in README.md, 'Build the container images',\n"
            f"  and wait for it to succeed before building the connector.\n"
            f"  If you intentionally built a different tag, select it with --base-image.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connector", required=True, choices=["gr00t"])
    parser.add_argument("--project-name", default=None,
                        help="CodeBuild project name; default resolves from SSM (see "
                             "config.resolve_codebuild_project). NOT hardcoded: a stale default sent "
                             "builds to a CloudFormation-owned project that 403s on our source.")
    # Default to the tag the pair manifest declares, so the documented build
    # produces exactly the image the pipeline submits. A literal default here
    # (previously "v2") silently built a tag nothing would ever run.
    parser.add_argument("--tag", default=None,
                        help="image tag to build (default: the tag in the pair "
                             "manifest's eval_image, i.e. what SimEval will run)")
    parser.add_argument("--base-image", default=None,
                        help="override the Arena base image 'repo:tag' (default: "
                             "the pair manifest's eval_base_image)")
    parser.add_argument("--skip-base-check", action="store_true",
                        help="skip the ECR existence check on the base image "
                             "(e.g. the base lives in another account/registry)")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--source-bucket", default=None,
                        help="S3 bucket for the CodeBuild source zip. Defaults to the "
                              "resolved trust bucket (cfg.trust_bucket). Override when the live "
                             "CodeBuild service role can only read a different bucket "
                             "(e.g. a pre-existing Foundation role scoped to another bucket).")
    args = parser.parse_args()

    cfg = load_config()
    project_name = resolve_codebuild_project(cfg, args.project_name)
    buildspec_path = BUILDSPEC[args.connector]
    if not os.path.isfile(os.path.join(REPO_ROOT, buildspec_path)):
        print(f"FATAL: buildspec not found: {buildspec_path}")
        sys.exit(1)

    spec = resolve(args.connector, SIMULATOR)
    registry = f"{cfg.account_id}.dkr.ecr.{cfg.region}.amazonaws.com"
    repository = load_arena_repository(cfg)
    if not repository.startswith(registry + "/"):
        raise ValueError("Foundation Arena repository must belong to the configured account and region")
    declared_tag = spec.eval_image_repo.rsplit(":", 1)[1]
    spec = replace(spec, eval_image_repo=f"{repository[len(registry) + 1:]}:{declared_tag}")
    target = resolve_build_target(spec, args.connector, registry,
                                  tag=args.tag, base_image=args.base_image,
                                  skip_base_check=args.skip_base_check)
    tag = target.tag
    ecr_repo, base_image_uri = target.ecr_repo, target.base_image_uri
    base_repo_tag = target.base_repo_tag

    ecr = boto3.client("ecr", region_name=cfg.region)
    if args.skip_base_check:
        print(f"(--skip-base-check) not verifying base {base_image_uri}")
    else:
        verify_base_image(ecr, base_repo_tag, cfg.region)
        print(f"Base image OK: {base_image_uri}")

    print(f"Building {ecr_repo}:{tag}  (buildspec={buildspec_path})")

    s3 = boto3.client("s3", region_name=cfg.region)
    cb = boto3.client("codebuild", region_name=cfg.region)

    source_bytes = create_source_zip(args.connector)
    # C3 (cycle 8): this published the zip into the SHARED Foundation models bucket, which the
    # Foundation role can PutObject and DeleteObject across and which peer components submit
    # jobs under. A peer could replace the source between this upload and the build starting,
    # and the replacement would then execute under the CodeBuild role, which can push to this
    # component's ECR repositories.
    #
    # The trust bucket's code/v1/* namespace is exactly the right home: publication belongs to
    # the deploying identity and no runtime role may write there. Keyed by the FULL digest
    # rather than a 12-char prefix -- the key asserts the content, so it should assert all of it.
    digest = hashlib.sha256(source_bytes).hexdigest()
    source_bucket = args.source_bucket or cfg.trust_bucket
    key = f"code/v1/{digest}/arena-{args.connector}-source.zip"
    s3.put_object(Bucket=source_bucket, Key=key, Body=source_bytes)
    # Read the published object back and compare. A same-key replacement between the write and
    # the build is the attack this addresses, so the publisher confirms what is actually there
    # rather than assuming its own PUT is what the build will fetch.
    published = s3.get_object(Bucket=source_bucket, Key=key)["Body"].read()
    published_digest = hashlib.sha256(published).hexdigest()
    if published_digest != digest:
        raise SystemExit(
            f"FATAL: build source at s3://{source_bucket}/{key} hashes to "
            f"{published_digest}, but {digest} was uploaded. Something replaced it between the "
            f"write and this read; refusing to start a build from unverified source.")
    print(f"Source uploaded and verified: s3://{source_bucket}/{key} "
          f"({len(source_bytes)} bytes, sha256={digest})")

    resp = cb.start_build(
        projectName=project_name,
        sourceTypeOverride="S3",
        sourceLocationOverride=f"{source_bucket}/{key}",
        buildspecOverride=buildspec_path,
        # The connector pulls the ~26GB base, installs into Isaac Sim's python, and
        # pushes -- the project's 60-min/LARGE default is too small. Override both.
        timeoutInMinutesOverride=120,
        computeTypeOverride="BUILD_GENERAL1_2XLARGE",
        environmentVariablesOverride=[
            {"name": "ECR_REPO_URI", "value": ecr_repo, "type": "PLAINTEXT"},
            {"name": "IMAGE_TAG", "value": tag, "type": "PLAINTEXT"},
            {"name": "AWS_DEFAULT_REGION", "value": cfg.region, "type": "PLAINTEXT"},
            # Full base URI -> the Dockerfile's ARENA_BASE_IMAGE build arg.
            {"name": "ARENA_BASE_IMAGE", "value": base_image_uri, "type": "PLAINTEXT"},
        ],
    )
    build_id = resp["build"]["id"]
    print(f"Build started: {build_id}", flush=True)
    print(f"Poll: aws codebuild batch-get-builds --ids {build_id} --region {cfg.region} "
          "--query 'builds[0].[buildStatus,currentPhase]' --output text")

    if not args.wait:
        print("(submitted, not waiting)")
        return

    while True:
        b = cb.batch_get_builds(ids=[build_id])["builds"][0]
        status = b["buildStatus"]
        if status == "IN_PROGRESS":
            print(f"  [{b.get('currentPhase', '?')}] ...", flush=True)
            time.sleep(30)
        elif status == "SUCCEEDED":
            print(f"BUILD SUCCEEDED: {ecr_repo}:{tag}")
            return
        else:
            print(f"BUILD FAILED: status={status} phase={b.get('currentPhase')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
