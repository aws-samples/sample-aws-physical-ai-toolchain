#!/usr/bin/env python3
"""Verify an evaluation-cell execution -- seven structural checks.

Usage: PYTHONPATH=src python scripts/verify_evaluation_cell.py <execution-arn>

Exits 0 only on 7/7 PASS. Prints evidence for STATUS.md.
"""
from __future__ import annotations

import json
import math
import re
import sys
from urllib.parse import urlparse

import boto3

sys.path.insert(0, "src")
from vla_pipeline.config import load_config


# Verbatim from the SageMaker service model shape ContentDigest (botocore
# data/sagemaker/2017-07-24/service-2.json). Bare hex does NOT match.
_SAGEMAKER_CONTENT_DIGEST = re.compile(r"[Ss][Hh][Aa]256:[0-9a-fA-F]{64}")


def check(name, condition, evidence=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if evidence:
        for line in evidence.strip().split("\n")[:5]:
            print(f"        {line}")
    return condition


def verify_execution(sm, s3, arn):
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    steps = [step for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=arn
    ) for step in page["PipelineExecutionSteps"]]
    by_name = {step["StepName"]: step for step in steps}
    params = {item["Name"]: item["Value"] for page in sm.get_paginator(
        "list_pipeline_parameters_for_execution"
    ).paginate(PipelineExecutionArn=arn) for item in page["PipelineParameters"]}
    required = {"SimEval", "Validate", "SuccessGate"}
    results = [check(
        "C1: Execution and required steps succeeded",
        execution["PipelineExecutionStatus"] == "Succeeded"
        and required <= by_name.keys()
        and all(step["StepStatus"] == "Succeeded" for step in steps),
    )]
    if not results[0]:
        return False

    eval_only = "CheckpointS3Uri" in params
    if eval_only:
        checkpoint = params["CheckpointS3Uri"]
        mode_valid = "FineTune" not in by_name
    else:
        training = by_name["FineTune"]
        job_arn = training["Metadata"]["TrainingJob"]["Arn"]
        job = sm.describe_training_job(TrainingJobName=job_arn.rsplit("/", 1)[1])
        checkpoint = job["ModelArtifacts"]["S3ModelArtifacts"]
        mode_valid = job["TrainingJobStatus"] == "Completed"
    results.append(check("C2: Execution checkpoint resolved", mode_valid, checkpoint))

    evaluation = by_name["SimEval"]
    job_arn = evaluation["Metadata"]["TrainingJob"]["Arn"]
    job = sm.describe_training_job(TrainingJobName=job_arn.rsplit("/", 1)[1])
    model_channels = [channel for channel in job["InputDataConfig"]
                      if channel["ChannelName"] == "model"]
    results.append(check(
        "C3: SimEval consumed the execution checkpoint URI",
        job["TrainingJobStatus"] == "Completed" and len(model_channels) == 1
        and model_channels[0]["DataSource"]["S3DataSource"]["S3Uri"] == checkpoint,
    ))

    processing_arn = by_name["Validate"]["Metadata"]["ProcessingJob"]["Arn"]
    processing = sm.describe_processing_job(
        ProcessingJobName=processing_arn.rsplit("/", 1)[1])
    eval_inputs = [item["S3Input"] for item in processing["ProcessingInputs"]
                   if "S3Input" in item and item["S3Input"]["LocalPath"]
                   == "/opt/ml/processing/eval_output"]
    if (len(eval_inputs) != 1
            or eval_inputs[0]["S3Uri"] != job["ModelArtifacts"]["S3ModelArtifacts"]):
        raise ValueError("Validate must consume this execution's SimEval output")
    checkpoint_inputs = [item["S3Input"] for item in processing["ProcessingInputs"]
                         if "S3Input" in item and item["S3Input"]["LocalPath"]
                         == "/opt/ml/processing/checkpoint"]
    if len(checkpoint_inputs) != 1 or checkpoint_inputs[0]["S3Uri"] != checkpoint:
        raise ValueError("Validate must consume this execution's checkpoint")
    outputs = [output for output in processing["ProcessingOutputConfig"]["Outputs"]
               if output["OutputName"] == "validated"]
    if len(outputs) != 1:
        raise ValueError("Validate must have exactly one validated output")
    output_uri = outputs[0]["S3Output"]["S3Uri"]
    location = urlparse(output_uri)
    execution_id = arn.rsplit("/", 1)[1]
    if (location.scheme != "s3" or not location.netloc or location.query or location.fragment
            or location.path.rstrip("/").rsplit("/", 1)[-1] != execution_id):
        raise ValueError(f"Validate output is not execution-scoped: {output_uri!r}")
    key = location.path.lstrip("/").rstrip("/") + "/validated_metrics.json"
    response = s3.get_object(Bucket=location.netloc, Key=key)
    body = response["Body"]
    try:
        validated = json.loads(body.read())
    finally:
        body.close()
    rate = validated["success_rate"]
    episodes = validated["episodes"]
    digest = validated["weights_digest_recomputed_by_validate"]
    results.append(check(
        "C4: This execution produced successful validation evidence",
        processing["ProcessingJobStatus"] == "Completed"
        and validated["validation_passed"] is True
        and validated["policy_type"] == "checkpoint"
        and isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
        and type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 1
        and type(episodes) is int and episodes > 0,
        f"s3://{location.netloc}/{key}",
    ))
    threshold = float(params["SuccessThreshold"])
    outcome = by_name["SuccessGate"]["Metadata"]["Condition"]["Outcome"]
    results.append(check(
        "C5: SuccessGate passed the execution threshold",
        results[-1] and math.isfinite(threshold) and 0 <= threshold <= 1
        and rate >= threshold and outcome == "True",
        f"Outcome={outcome}, threshold={threshold}",
    ))

    registrations = [step for step in steps if "RegisterModel" in step.get("Metadata", {})]
    if len(registrations) != 1:
        results.append(check("C6: Exactly one package registered by this execution", False))
    else:
        package_arn = registrations[0]["Metadata"]["RegisterModel"]["Arn"]
        package = sm.describe_model_package(ModelPackageName=package_arn)
        metadata = package.get("CustomerMetadataProperties", {})
        containers = package["InferenceSpecification"]["Containers"]
        mode_matches = (
            metadata.get("eval_only") == "true"
            and metadata.get("input_checkpoint_uri") == checkpoint
        ) if eval_only else "eval_only" not in metadata
        # The registered URL must be the PROMOTED artifact from this execution's Validate
        # receipt, not the source checkpoint. This check previously compared it against
        # `checkpoint`, which was correct only while registration pointed at the raw,
        # unversioned source -- the defect C5 fixed. On the eval-only path the source URI
        # is still carried as provenance in CustomerMetadataProperties, checked above.
        promotion = validated.get("promotion") or {}
        promoted_uri = promotion.get("model_uri")
        results.append(check(
            "C6: This execution registered its checkpoint as PendingManualApproval",
            package["ModelPackageArn"] == package_arn
            and package["ModelPackageGroupName"] == params["ModelPackageGroupName"]
            and package["ModelPackageStatus"] == "Completed"
            and package["ModelApprovalStatus"] == "PendingManualApproval"
            and len(containers) == 1
            and isinstance(promoted_uri, str) and promoted_uri.startswith("s3://")
            and containers[0]["ModelDataUrl"] == promoted_uri
            and mode_matches,
            package_arn,
        ))
        # "6/6 verification" did not establish that the package pointed at the validation
        # evidence this script inspected: ModelMetrics was never checked, so a package with
        # missing metrics, or another execution's, passed. Require the linkage explicitly
        # and require the digest to match the attestation this execution published.
        metrics_source = ((package.get("ModelMetrics") or {})
                          .get("ModelQuality", {}).get("Statistics", {}))
        results.append(check(
            "C7: The package points at THIS execution's validation evidence",
            isinstance(metrics_source.get("S3Uri"), str)
            and metrics_source["S3Uri"] == promotion.get("attestation_uri")
            # N1: must be the API-formatted digest, and it must actually agree with
            # the raw hex rather than merely being present.
            and metrics_source.get("ContentDigest") == promotion.get(
                "attestation_content_digest")
            and metrics_source.get("ContentDigest") == "sha256:{}".format(
                promotion.get("attestation_sha256"))
            and bool(_SAGEMAKER_CONTENT_DIGEST.fullmatch(
                metrics_source.get("ContentDigest") or "")),
            f"ModelMetrics={metrics_source.get('S3Uri')!r} "
            f"expected={promotion.get('attestation_uri')!r}",
        ))
    print(f"RESULT: {sum(results)}/{len(results)} checks passed")
    print("Checks establish execution linkage, not independent checkpoint-content verification.")
    return all(results)


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_evaluation_cell.py <execution-arn>")
        sys.exit(1)

    arn = sys.argv[1]
    cfg = load_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    s3 = boto3.client("s3", region_name=cfg.region)

    print(f"\nVerifying: {arn}\n{'='*60}")

    sys.exit(0 if verify_execution(sm, s3, arn) else 1)


if __name__ == "__main__":
    main()
