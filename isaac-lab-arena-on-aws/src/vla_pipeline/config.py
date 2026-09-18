"""Discover pipeline resources from the deployed Terraform Foundation."""
from __future__ import annotations

import dataclasses
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

_FOUNDATION_PARAMETERS = {
    "role_arn": "sagemaker-role-arn",
    "bucket": "models-bucket",
    "hf_secret_name": "isaac-lab-arena/hf-secret-name",
}
# The component's own resources, published by infra/trust_boundary.tf. Read separately
# from the Foundation contract because they are OURS: the Foundation role stays the
# orchestration identity, while these are the per-stage runtime identities and the
# create-only bucket that promoted artifacts and attestations are published to.
_COMPONENT_PARAMETERS = {
    # FineTune and SimEval have SEPARATE identities: a training worker must not be able
    # to write, replace or manufacture the evaluation evidence Validate judges.
    "training_role_arn": "training-role-arn",
    "workload_role_arn": "workload-role-arn",
    "validation_role_arn": "validation-role-arn",
    "trust_bucket": "trust-bucket",
    # C3: raw SimEval evidence and the gate's receipt used to land in the SHARED Foundation
    # models bucket, where the Foundation role holds PutObject and DeleteObject across
    # everything and peer components submit jobs. A peer could replace a completed SimEval
    # archive before Validate read it, keeping the genuine checkpoint identity and substituting
    # the score. Discovered, never defaulted: falling back to the shared bucket would silently
    # reopen exactly that path.
    "handoff_bucket": "handoff-bucket",
    # I11: this file defaulted to "vla-pipeline" independently while every IAM policy in
    # trust_boundary.tf interpolates var.pipeline_prefix. A legitimate non-default Terraform
    # setting deployed policies that DENY the normal launchers' outputs, and the variable's
    # note that the values "must match" is documentation, not enforcement. Discovered like
    # the roles so the launchers write where the deployed policies allow.
    "prefix": "pipeline-prefix",
}


class ConfigError(RuntimeError):
    pass


def require_supported_region(region: str | None) -> None:
    """Reject unsupported deployment regions before discovering or creating resources."""
    if region != "us-east-1":
        raise ConfigError(
            f"This sample supports only us-east-1; selected region: {region!r}. "
            "Set AWS_DEFAULT_REGION=us-east-1 and VLA_REGION=us-east-1."
        )


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    account_id: str
    region: str
    role_arn: str
    bucket: str
    # Component-owned runtime identities and trust storage (C4 + C5). Required: the
    # loader fails closed rather than defaulting them to role_arn, because a fallback
    # would restore the single-identity boundary they replace.
    training_role_arn: str
    workload_role_arn: str
    validation_role_arn: str
    trust_bucket: str
    #: C3: component-owned storage for the SimEval -> Validate handoff.
    handoff_bucket: str
    pipeline_name: str = "vla-model-evaluation"
    prefix: str = "vla-pipeline"
    foundation_project: str = "physical-ai"
    hf_secret_name: str = "vla-pipeline/hf-token"

    def s3_uri(self, *parts: str) -> str:
        tail = "/".join(p.strip("/") for p in parts if p)
        return f"s3://{self.bucket}/{self.prefix}/{tail}" if tail else \
            f"s3://{self.bucket}/{self.prefix}"

    def handoff_uri(self, *parts: str) -> str:
        """C3: component-owned storage for evaluation evidence and the gate's receipt.

        Deliberately NOT prefixed with self.prefix. The namespaces are named in bucket-policy
        statements that restrict who may publish them, and a configurable prefix in the path
        would let a deployment write outside the namespace its own policy protects.
        """
        tail = "/".join(p.strip("/") for p in parts if p)
        return f"s3://{self.handoff_bucket}/{tail}" if tail else \
            f"s3://{self.handoff_bucket}"


def load_arena_repository(cfg: PipelineConfig) -> str:
    path = f"/{cfg.foundation_project}/ecr/isaac-lab-arena"
    try:
        value = boto3.client("ssm", region_name=cfg.region).get_parameter(
            Name=path)["Parameter"]["Value"]
    except (BotoCoreError, ClientError) as exc:
        raise ConfigError(f"Cannot read Foundation parameter {path}: {exc}") from exc
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Foundation parameter {path} is empty or invalid.")
    return value


def load_config(*, region: str | None = None,
                project_name: str | None = None, boto_session=None) -> PipelineConfig:
    project_name = project_name or os.environ.get("VLA_FOUNDATION_PROJECT", "physical-ai")
    session = boto_session or boto3.Session(region_name=region or os.environ.get("VLA_REGION"))
    require_supported_region(session.region_name)
    try:
        account_id = session.client("sts").get_caller_identity()["Account"]
    except (BotoCoreError, ClientError) as exc:
        raise ConfigError(f"Cannot resolve AWS account identity: {exc}") from exc

    ssm = session.client("ssm")
    resources = {}
    for field, suffix in _FOUNDATION_PARAMETERS.items():
        path = f"/{project_name}/{suffix}"
        try:
            value = ssm.get_parameter(Name=path)["Parameter"]["Value"]
        except (BotoCoreError, ClientError) as exc:
            raise ConfigError(
                f"Cannot read Foundation parameter {path} in {session.region_name}: {exc}"
            ) from exc
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Foundation parameter {path} is empty or invalid.")
        resources[field] = value

    # The component's own trust boundary (review findings C4 + C5): separate runtime roles
    # so workers stop holding the Foundation role's Put/DeleteObject over the bucket that
    # also holds the validation code, plus the bucket where Validate publishes promoted
    # artifacts and attestations.
    #
    # These FAIL CLOSED rather than falling back to the Foundation role. A fallback would
    # silently restore the boundary this replaced: every stage sharing one identity that
    # can overwrite the validator judging it and the bytes after validation.
    component = {}
    for field, suffix in _COMPONENT_PARAMETERS.items():
        path = f"/{project_name}/component/{suffix}"
        try:
            value = ssm.get_parameter(Name=path)["Parameter"]["Value"]
        except (BotoCoreError, ClientError) as exc:
            raise ConfigError(
                f"Cannot read component parameter {path} in {session.region_name}: "
                f"{exc}. The gated-registration trust boundary must be deployed "
                f"(infra/trust_boundary.tf) -- falling back to the Foundation role would "
                f"let a worker overwrite the validator that judges it and the model bytes "
                f"after validation, which is what this boundary exists to prevent."
            ) from exc
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"Component parameter {path} is empty or invalid.")
        component[field] = value

    return PipelineConfig(
        account_id=account_id,
        region=session.region_name,
        role_arn=resources["role_arn"],
        bucket=resources["bucket"],
        training_role_arn=component["training_role_arn"],
        workload_role_arn=component["workload_role_arn"],
        validation_role_arn=component["validation_role_arn"],
        trust_bucket=component["trust_bucket"],
        handoff_bucket=component["handoff_bucket"],
        prefix=component["prefix"],
        foundation_project=project_name,
        hf_secret_name=resources["hf_secret_name"],
        pipeline_name=os.environ.get("VLA_PIPELINE_NAME") or "vla-model-evaluation",
    )


CODEBUILD_PROJECT_SSM_SUFFIX = "codebuild-project"


def resolve_codebuild_project(cfg, override: str | None = None) -> str:
    """The DEPLOYED builder's name: an explicit override, else SSM. Never a hardcoded default.

    Lives here because BOTH image builders need it. It was originally added to build_images.py alone,
    and its sibling build_arena_connector.py kept `PROJECT = "vla-image-build"` hardcoded -- so the
    Arena connector went on submitting builds to a CloudFormation-owned project of that name whose
    role cannot read this component's source bucket. Those builds fail at DOWNLOAD_SOURCE with a 403,
    and the Arena image silently stayed days out of date while runs were attributed to current code.

    One implementation, imported twice, so a fix cannot land in one caller and miss the other.
    """
    if override:
        return override
    import boto3
    path = f"/{cfg.foundation_project}/component/{CODEBUILD_PROJECT_SSM_SUFFIX}"
    try:
        return boto3.client("ssm", region_name=cfg.region).get_parameter(
            Name=path)["Parameter"]["Value"]
    except Exception as exc:
        raise SystemExit(
            f"FATAL: cannot read {path} in {cfg.region}: {exc}\n"
            f"The component infrastructure publishes its CodeBuild project name there "
            f"(infra/trust_boundary.tf). Deploy infra/, or pass --project-name explicitly to target "
            f"a project another deployment owns. Refusing to guess: guessing is what submitted "
            f"builds to a project whose role could not read our source bucket.") from exc
