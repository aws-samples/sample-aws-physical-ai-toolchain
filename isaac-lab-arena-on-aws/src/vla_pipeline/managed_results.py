"""Verify managed job/output linkage and downloaded validation evidence.

This independently reads the service records and S3 evidence. Validate remains
the actor that recomputes checkpoint contents; this client does not download and
rehash a multi-gigabyte checkpoint or claim to have reconstructed its training.
"""
from __future__ import annotations

import hashlib
import json
import math
from urllib.parse import urlparse

from .backend import script
from .deployment import aws_session, discover
from .operations import timestamp, write_json

VERIFICATION_VERSION = 2


def check_environment(job, params, fields, label):
    environment = job.get("Environment", {})
    for variable, parameter in fields.items():
        if parameter in params:
            require(environment.get(variable) == str(params[parameter]),
                    f"{label} {variable} differs from requested {parameter}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def object_identity(s3, uri, *, bucket=None, version=None):
    location = urlparse(uri)
    require(location.scheme == "s3" and location.netloc and location.path.strip("/")
            and not location.query and not location.fragment, f"Invalid output URI: {uri}")
    require(bucket is None or location.netloc == bucket, "Output is outside its configured bucket")
    head = s3.head_object(Bucket=location.netloc, Key=location.path.lstrip("/"),
                          **({"VersionId": version} if version else {}))
    require(head.get("VersionId") not in (None, "", "null") and head["ContentLength"] > 0,
            "Output must be nonempty and versioned")
    return {"uri": uri, "bucket": location.netloc, "key": location.path.lstrip("/"),
            "version_id": head["VersionId"], "etag": head["ETag"], "bytes": head["ContentLength"]}


def read_json(s3, identity):
    require(identity["bytes"] <= 8 * 1024 * 1024, "Evidence JSON exceeds 8 MiB")
    stream = s3.get_object(Bucket=identity["bucket"], Key=identity["key"],
                           VersionId=identity["version_id"])["Body"]
    try:
        body = stream.read(8 * 1024 * 1024 + 1)
    finally:
        stream.close()
    require(len(body) == identity["bytes"], "Evidence download length differs from S3 metadata")
    return json.loads(body), hashlib.sha256(body).hexdigest()


def verify(record, store):
    from .execution import check_managed_steps
    from .registry import resolve_suite

    session = aws_session(record.get("profile"), record["region"])
    require(session.client("sts").get_caller_identity()["Account"] == record["account_id"],
            "Verification credentials belong to another account")
    cfg = discover(session, record["project"])
    sm, s3 = session.client("sagemaker"), session.client("s3")
    arn, request = record["execution_arn"], record["request"]
    execution = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    rows = [row for page in sm.get_paginator("list_pipeline_execution_steps").paginate(
        PipelineExecutionArn=arn) for row in page["PipelineExecutionSteps"]]
    passed, reason = check_managed_steps(execution, rows, request["steps"])
    require(passed, reason or "Requested managed steps did not succeed")
    actual = {item["Name"]: item["Value"] for page in sm.get_paginator(
        "list_pipeline_parameters_for_execution").paginate(PipelineExecutionArn=arn)
        for item in page["PipelineParameters"]}
    require(all(actual.get(key) == str(value) for key, value in request["parameters"].items()),
            "Executed parameters differ from the CLI request")
    params = request["parameters"]
    by_name = {row["StepName"]: row for row in rows}
    outputs, jobs = {}, {}
    output_buckets = {"FineTune": cfg.bucket, "SimEval": cfg.handoff_bucket}
    for step, image_key, type_key, volume_key in (
        ("FineTune", "TrainImageUri", "TrainInstanceType", "VolumeSizeInGB"),
        ("SimEval", "EvalImageUri", "EvalInstanceType", "EvalVolumeSizeInGB"),
    ):
        if step not in request["steps"]:
            continue
        job_arn = by_name[step]["Metadata"]["TrainingJob"]["Arn"]
        name = job_arn.rsplit("/", 1)[1]
        job = sm.describe_training_job(TrainingJobName=name)
        require(job["TrainingJobStatus"] == "Completed", f"{step} job did not complete")
        require(job["AlgorithmSpecification"]["TrainingImage"] == params[image_key],
                f"{step} used a different image")
        require(job["ResourceConfig"]["InstanceType"] == params[type_key]
                and job["ResourceConfig"]["VolumeSizeInGB"] == params[volume_key],
                f"{step} resources differ from the request")
        require(job["StoppingCondition"]["MaxRuntimeInSeconds"] == params["MaxRuntimeSeconds"],
                f"{step} runtime budget differs from the request")
        check_environment(job, params, {
            "VLA_MAX_RUNTIME_SECONDS": "MaxRuntimeSeconds",
            **({
                "TRAIN_MODEL_FAMILY": "ModelFamily", "TRAIN_MAX_STEPS": "TrainSteps",
                "TRAIN_SAVE_STEPS": "TrainSaveSteps", "TRAIN_SUITE": "TrainSuite",
                "TRAIN_DATASET_S3URI": "DatasetS3Uri", "TRAIN_DATASET_REVISION": "DatasetRevision",
                "GR00T_VERSION": "Gr00tVersion",
            } if step == "FineTune" else {
                "EVAL_MODEL_FAMILY": "ModelFamily", "EVAL_SUITE": "Suite",
                "EVAL_SEED": "EvalSeed", "EVAL_TRIALS": "EvalTrials",
                "EVAL_TASK_IDS": "EvalTaskIds", "EVAL_DOSE_STEPS": "EvalDoseSteps",
                "EVAL_GR00T_VERSION": "Gr00tVersion", "EVAL_SIM_CONFIG": "EvalSimConfig",
                "USE_GROOT_SERVER": "UseGrootServer", "ARENA_CONNECTOR": "ArenaConnector",
            }),
        }, step)
        outputs[step] = {**object_identity(
            s3, job["ModelArtifacts"]["S3ModelArtifacts"], bucket=output_buckets[step]), "job_name": name}
        jobs[step] = job
    directory = store.path("runs", record["id"]).parent
    write_json(directory / "managed-jobs.json", jobs)
    write_json(directory / "job-outputs.json", outputs)
    manifest = {"development_bucket": cfg.bucket,
                "checkpoint_identity": request.get("checkpoint_identity")}
    adapted_rows = json.loads(json.dumps(rows, default=str))
    for row in adapted_rows:
        if row["StepName"] in outputs:
            row["Metadata"]["TrainingJob"]["Arn"] = outputs[row["StepName"]]["job_name"]
    proof, report = script("local/local_outputs.py").verify_outputs(
        directory, s3, manifest, params, adapted_rows, expected_output_buckets=output_buckets)
    result = {"independently_verified": True, "verification_status": "passed",
              "verification_version": VERIFICATION_VERSION,
              "verification_scope": proof["scope"], "outputs": outputs,
              "source_commit": record["source_commit"], "execution_arn": arn,
              "verified_at": timestamp(), "requested_steps": request["steps"]}
    if report:
        result.update(episodes=report["episodes"], success_rate=report["success_rate"])
        checkpoint = request.get("checkpoint_s3") or outputs["FineTune"]["uri"]
        channels = [item for item in jobs["SimEval"]["InputDataConfig"] if item["ChannelName"] == "model"]
        require(len(channels) == 1 and
                channels[0]["DataSource"]["S3DataSource"]["S3Uri"] == checkpoint,
                "SimEval did not consume this execution's selected checkpoint")
    if "Validate" in request["steps"]:
        processing = sm.describe_processing_job(
            ProcessingJobName=by_name["Validate"]["Metadata"]["ProcessingJob"]["Arn"].rsplit("/", 1)[1])
        require(processing["ProcessingJobStatus"] == "Completed", "Validate job did not complete")
        check_environment(processing, params, {
            "EXPECTED_EVAL_SEED": "EvalSeed", "EXPECTED_EVAL_TRIALS": "EvalTrials",
            "EXPECTED_EVAL_TASK_IDS": "EvalTaskIds", "EXPECTED_MODEL_FAMILY": "ModelFamily",
            "EXPECTED_SUITE": "Suite", "EVALUATOR_IMAGE_URI": "EvalImageUri",
            "EXPECTED_FAMILY_VERSION": "Gr00tVersion",
            "EXPECTED_EMBODIMENT_TAG": "ExpectedEmbodimentTag",
            "EXPECTED_ARENA_EMBODIMENT": "ExpectedArenaEmbodiment",
            "EXPECTED_ARENA_OBJECT": "ExpectedArenaObject",
            "EXPECTED_POLICY_CONFIG": "ExpectedPolicyConfig",
            "EXPECTED_TRAIN_STEPS": "TrainSteps", "EXPECTED_TRAIN_SUITE": "TrainSuite",
            "EXPECTED_DATASET_S3URI": "DatasetS3Uri", "EXPECTED_DATASET_REVISION": "DatasetRevision",
            "EXPECTED_SUCCESS_THRESHOLD": "SuccessThreshold",
        }, "Validate")
        require(processing.get("Environment", {}).get("EXPECTED_TRAIN_PATH") ==
                ("train" if "FineTune" in request["steps"] else "eval_only"),
                "Validate training/checkpoint-only mode differs from the request")
        require(processing["Environment"].get("EXPECTED_MODEL_SOURCE_URI") == checkpoint,
                "Validate expected source differs from the selected checkpoint")
        inputs = {item["S3Input"]["LocalPath"]: item["S3Input"]["S3Uri"]
                  for item in processing["ProcessingInputs"] if "S3Input" in item}
        require(inputs.get("/opt/ml/processing/checkpoint") == checkpoint
                and inputs.get("/opt/ml/processing/eval_output") == outputs["SimEval"]["uri"],
                "Validate input linkage differs from this execution")
        destinations = [item["S3Output"]["S3Uri"] for item in processing[
            "ProcessingOutputConfig"]["Outputs"] if item["OutputName"] == "validated"]
        require(len(destinations) == 1 and destinations[0].rstrip("/").endswith("/" + arn.rsplit("/", 1)[1]),
                "Validate output must be scoped to this execution")
        receipt_uri = destinations[0].rstrip("/") + "/validated_metrics.json"
        receipt_identity = object_identity(s3, receipt_uri, bucket=cfg.handoff_bucket)
        receipt, receipt_sha = read_json(s3, receipt_identity)
        require(receipt.get("validation_passed") is True and receipt.get("policy_type") == "checkpoint",
                "Validate did not publish successful real-checkpoint evidence")
        for field, parameter in (("model_family", "ModelFamily"), ("suite", "Suite"),
                                 ("eval_seed", "EvalSeed"), ("eval_trials", "EvalTrials")):
            require(receipt.get(field) == params[parameter],
                    f"Validate receipt {field} differs from requested {parameter}")
        suite = resolve_suite(params["Suite"])
        require(receipt.get("task_ids") == list(suite.canonical_task_ids)
                and receipt.get("episodes") == params["EvalTrials"] * len(suite.canonical_task_ids),
                "Validate did not report the requested episodes/tasks")
        require(receipt.get("success_rate") == report["success_rate"]
                and receipt.get("episodes") == report["episodes"],
                "Validate and simulation report different results")
        lineage = script("local/local_profiles.py").check_training_contract(receipt, params)
        if "FineTune" in request["steps"]:
            comparisons = [item for item in lineage["fields"] if item["field"] == "train_steps"]
            require(len(comparisons) == 1
                    and comparisons[0]["expected"] == comparisons[0]["observed"] == str(params["TrainSteps"]),
                    "Training lineage does not contain the requested training-step count")
        identity = receipt["model_artifact_identity"]
        expected = request.get("checkpoint_identity") or outputs["FineTune"]
        source = urlparse(checkpoint)
        require((identity["bucket"], identity["key"], identity["version_id"], identity["etag"].strip('"')) ==
                (source.netloc, source.path.lstrip("/"), expected["version_id"], expected["etag"].strip('"')),
                "Validate's checkpoint identity differs from the selected checkpoint")
        checkpoint_identity = object_identity(s3, checkpoint, version=identity["version_id"])
        require(checkpoint_identity["etag"].strip('"') == identity["etag"].strip('"'),
                "S3 checkpoint version differs from validation evidence")
        runtime = receipt["validation_runtime"]
        from .validation_sdk import sdk_bundle
        sdk_digest = hashlib.sha256(sdk_bundle()).hexdigest()
        require(runtime.get("boto3_version") == runtime.get("botocore_version") == "1.42.97"
                and runtime.get("sdk_bundle_sha256") == sdk_digest, "Validate SDK identity differs")
        promotion = receipt["promotion"]
        require(not promotion.get("local_test_publication"), "Managed receipt contains a publication stub")
        require(promotion.get("artifact_publication") in {"created_multipart", "already_present_identical"}
                and promotion.get("attestation_publication") in {"created", "already_present_identical"},
                "Unexpected conditional-publication evidence")
        attestation_identity = object_identity(s3, promotion["attestation_uri"], bucket=cfg.trust_bucket)
        attestation, digest = read_json(s3, attestation_identity)
        require(promotion["attestation_content_digest"] == "sha256:" + digest
                and promotion["attestation_sha256"] == digest, "Downloaded attestation digest mismatch")
        require(attestation["lineage"] == lineage
                and attestation["validated_metrics"]["validation_runtime"] == runtime
                and attestation["execution"]["pipeline_execution_id"] == arn.rsplit("/", 1)[1],
                "Attestation is not bound to this validation/execution")
        require(attestation.get("attestation_version") == 1
                and attestation["validated_metrics"] ==
                {key: value for key, value in receipt.items() if key != "promotion"},
                "Attestation does not contain the complete validated receipt")
        require(attestation["source_artifact"]["identity"] == identity
                and attestation["source_artifact"]["expected_source_uri"] == checkpoint,
                "Attestation identifies a different source checkpoint")
        promoted = object_identity(s3, promotion["model_uri"], bucket=cfg.trust_bucket)
        artifact = attestation["promoted_artifact"]
        require(promoted["bytes"] == checkpoint_identity["bytes"] == artifact["size_bytes"]
                and artifact["archive_sha256"] == promotion["archive_sha256"]
                and artifact["weights_digest_recomputed_by_validate"] ==
                receipt["weights_digest_recomputed_by_validate"]
                and promoted["key"] == f"artifacts/v1/sha256/{promotion['archive_sha256']}/model.tar.gz",
                "Promoted artifact linkage differs")
        if "SuccessGate" in request["steps"]:
            rate = receipt["success_rate"]
            require(type(rate) in (int, float) and math.isfinite(rate)
                    and 1 >= rate >= params["SuccessThreshold"] >= 0, "Success threshold was not met")
        if "RegisterModel" in request["steps"]:
            package_arn = by_name["RegisterModel-RegisterModel"]["Metadata"]["RegisterModel"]["Arn"]
            package = sm.describe_model_package(ModelPackageName=package_arn)
            containers = package["InferenceSpecification"]["Containers"]
            metadata = package.get("CustomerMetadataProperties", {})
            metrics = package.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {})
            require(package["ModelPackageStatus"] == "Completed"
                    and package["ModelApprovalStatus"] == "PendingManualApproval"
                    and package["ModelPackageGroupName"] == params["ModelPackageGroupName"]
                    and len(containers) == 1 and containers[0]["ModelDataUrl"] == promotion["model_uri"]
                    and containers[0]["Image"] == params[
                        "TrainImageUri" if "FineTune" in request["steps"] else "EvalImageUri"],
                    "Registered package differs from this execution's promoted model")
            require(metrics.get("S3Uri") == promotion["attestation_uri"]
                    and metrics.get("ContentDigest") == "sha256:" + digest,
                    "Registered metrics do not identify the downloaded attestation")
            require((metadata.get("eval_only") == "true" and metadata.get("input_checkpoint_uri") == checkpoint)
                    if request.get("checkpoint_s3") else "eval_only" not in metadata,
                    "Registered training/evaluation-only mode differs")
            result["model_package_arn"] = package_arn
        proof.update(receipt=receipt, receipt_identity=receipt_identity, receipt_sha256=receipt_sha,
                     attestation=attestation, attestation_identity=attestation_identity,
                     promoted_identity=promoted, checkpoint_identity=checkpoint_identity)
        result.update(receipt_uri=receipt_uri, attestation_sha256=digest,
                      verification_scope="Requested managed jobs, versioned evaluation evidence, "
                      "Validate publication and requested registration. Checkpoint-content digest "
                      "and training lineage retain Validate's stated verification/attestation scope.")
    proof["output_verification_scope"] = proof["scope"]
    proof["scope"] = result["verification_scope"]
    write_json(directory / "managed-evidence.json", proof)
    write_json(directory / "independent-verification.json", result)
    return result
