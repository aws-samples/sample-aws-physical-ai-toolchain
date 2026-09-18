import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/verify_evaluation_cell.py"
_SPEC = importlib.util.spec_from_file_location("verify_evaluation_cell", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def execution():
    arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/fixture/execution/current"
    checkpoint = "s3://fixture/train/model.tar.gz"
    package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/arena-fixture/1"
    steps = [
        {"StepName": "FineTune", "StepStatus": "Succeeded",
         "Metadata": {"TrainingJob": {"Arn": "training-job/train"}}},
        {"StepName": "SimEval", "StepStatus": "Succeeded",
         "Metadata": {"TrainingJob": {"Arn": "training-job/eval"}}},
        {"StepName": "Validate", "StepStatus": "Succeeded",
         "Metadata": {"ProcessingJob": {"Arn": "processing-job/validate"}}},
        {"StepName": "SuccessGate", "StepStatus": "Succeeded",
         "Metadata": {"Condition": {"Outcome": "True"}}},
        {"StepName": "RegisterModel-RegisterModel", "StepStatus": "Succeeded",
         "Metadata": {"RegisterModel": {"Arn": package_arn}}},
    ]
    params = [{"Name": "SuccessThreshold", "Value": "0.5"},
              {"Name": "ModelPackageGroupName", "Value": "arena-fixture"}]
    sm = Mock()
    sm.describe_pipeline_execution.return_value = {"PipelineExecutionStatus": "Succeeded"}
    paginators = {
        "list_pipeline_execution_steps": Mock(),
        "list_pipeline_parameters_for_execution": Mock(),
    }
    paginators["list_pipeline_execution_steps"].paginate.side_effect = lambda **kw: [
        {"PipelineExecutionSteps": steps[:2]}, {"PipelineExecutionSteps": steps[2:]},
    ]
    paginators["list_pipeline_parameters_for_execution"].paginate.side_effect = lambda **kw: [
        {"PipelineParameters": params[:1]}, {"PipelineParameters": params[1:]},
    ]
    sm.get_paginator.side_effect = paginators.__getitem__
    jobs = {
        "train": {"TrainingJobStatus": "Completed",
                  "ModelArtifacts": {"S3ModelArtifacts": checkpoint}},
        "eval": {"TrainingJobStatus": "Completed",
                 "ModelArtifacts": {"S3ModelArtifacts": "s3://fixture/eval/model.tar.gz"},
                 "InputDataConfig": [
            {"ChannelName": "model", "DataSource": {"S3DataSource": {"S3Uri": checkpoint}}},
        ]},
    }
    sm.describe_training_job.side_effect = lambda TrainingJobName: jobs[TrainingJobName]
    sm.describe_processing_job.return_value = {
        "ProcessingJobStatus": "Completed",
        "ProcessingInputs": [{"S3Input": {
            "LocalPath": "/opt/ml/processing/eval_output",
            "S3Uri": "s3://fixture/eval/model.tar.gz",
        }}, {"S3Input": {
            "LocalPath": "/opt/ml/processing/checkpoint",
            "S3Uri": checkpoint,
        }}],
        "ProcessingOutputConfig": {"Outputs": [
            {"OutputName": "validated", "S3Output": {"S3Uri": "s3://fixture/validated/current"}},
        ]},
    }
    promoted_uri = "s3://fixture-trust/artifacts/v1/sha256/" + "b" * 64 + "/model.tar.gz"
    attestation_uri = ("s3://fixture-trust/evidence/v1/exec-1/" + "c" * 64
                       + "/validated_metrics.json")
    sm.describe_model_package.return_value = {
        "ModelPackageArn": package_arn, "ModelPackageGroupName": "arena-fixture",
        "ModelPackageStatus": "Completed", "ModelApprovalStatus": "PendingManualApproval",
        # C5: registration points at the PROMOTED artifact, not the source checkpoint.
        "InferenceSpecification": {"Containers": [{"ModelDataUrl": promoted_uri}]},
        # evidence linkage (I16/C2): the package must point at THIS execution's evidence. The fixture had no
        # ModelMetrics at all and still expected successful verification, so "6/6 checks
        # passed" said nothing about the linkage.
        "ModelMetrics": {"ModelQuality": {"Statistics": {
            "ContentType": "application/json",
            "S3Uri": attestation_uri,
            # N1: the SageMaker service model requires the sha256: prefix.
            # This fixture previously carried bare hex, matching the verifier's
            # own mistake, so their agreement proved nothing about whether the
            # registration argument was valid.
            "ContentDigest": "sha256:" + "c" * 64,
        }}},
    }
    s3 = Mock()
    report = {
        "validation_passed": True, "success_rate": 0.75, "episodes": 4,
        "policy_type": "checkpoint",
        "weights_digest_recomputed_by_validate": "sha256:" + "a" * 64,
        "promotion": {
            "model_uri": promoted_uri,
            "archive_sha256": "b" * 64,
            "attestation_uri": attestation_uri,
            "attestation_sha256": "c" * 64,
            "attestation_content_digest": "sha256:" + "c" * 64,
        },
    }
    s3.get_object.side_effect = lambda **kw: {"Body": io.BytesIO(json.dumps(report).encode())}
    return sm, s3, arn, steps, params, report


def test_current_execution_passes(execution):
    sm, s3, arn, _, _, _ = execution
    assert _MODULE.verify_execution(sm, s3, arn)
    s3.get_object.assert_called_once_with(
        Bucket="fixture", Key="validated/current/validated_metrics.json")
    sm.describe_model_package.assert_called_once_with(
        ModelPackageName="arn:aws:sagemaker:us-west-2:123456789012:model-package/arena-fixture/1")
    sm.list_model_packages.assert_not_called()
    s3.list_objects_v2.assert_not_called()


def test_eval_only_uses_runtime_checkpoint(execution):
    sm, s3, arn, steps, params, _ = execution
    steps.pop(0)
    params.append({"Name": "CheckpointS3Uri", "Value": "s3://fixture/train/model.tar.gz"})
    sm.describe_model_package.return_value["CustomerMetadataProperties"] = {
        "eval_only": "true", "input_checkpoint_uri": "s3://fixture/train/model.tar.gz",
    }
    assert _MODULE.verify_execution(sm, s3, arn)
    assert all(call.kwargs["TrainingJobName"] != "train"
               for call in sm.describe_training_job.call_args_list)


def test_unrelated_existing_package_cannot_replace_registration(execution):
    sm, s3, arn, steps, _, _ = execution
    steps.pop()
    sm.list_model_packages.return_value = {"ModelPackageSummaryList": [{"ModelPackageArn": "other"}]}
    assert not _MODULE.verify_execution(sm, s3, arn)
    sm.describe_model_package.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("ModelPackageGroupName", "unrelated"),
    ("ModelPackageArn", "unrelated"),
    ("ModelApprovalStatus", "Approved"),
    ("ModelPackageStatus", "Failed"),
])
def test_wrong_package_rejected(execution, field, value):
    sm, s3, arn, _, _, _ = execution
    sm.describe_model_package.return_value[field] = value
    assert not _MODULE.verify_execution(sm, s3, arn)


def test_wrong_checkpoint_rejected(execution):
    sm, s3, arn, _, _, _ = execution
    sm.describe_model_package.return_value["InferenceSpecification"]["Containers"][0][
        "ModelDataUrl"] = "s3://fixture/other/model.tar.gz"
    assert not _MODULE.verify_execution(sm, s3, arn)


def test_other_execution_output_rejected_before_read(execution):
    sm, s3, arn, _, _, _ = execution
    sm.describe_processing_job.return_value["ProcessingOutputConfig"]["Outputs"][0][
        "S3Output"]["S3Uri"] = "s3://fixture/validated/other"
    with pytest.raises(ValueError, match="not execution-scoped"):
        _MODULE.verify_execution(sm, s3, arn)
    s3.get_object.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("validation_passed", False), ("success_rate", float("nan")),
    ("success_rate", 0.25), ("episodes", 0), ("episodes", True),
    ("policy_type", "zero_action"), ("policy_type", "positive_control"),
    ("policy_type", None),
    ("weights_digest_recomputed_by_validate", None),
    ("weights_digest_recomputed_by_validate", ""),
    ("weights_digest_recomputed_by_validate", "sha256:" + "a" * 63),
    ("weights_digest_recomputed_by_validate", "sha256:" + "g" * 64),
])
def test_invalid_evidence_rejected(execution, field, value):
    sm, s3, arn, _, _, report = execution
    report[field] = value
    assert not _MODULE.verify_execution(sm, s3, arn)


def test_failed_execution_rejected(execution):
    sm, s3, arn, _, _, _ = execution
    sm.describe_pipeline_execution.return_value["PipelineExecutionStatus"] = "Failed"
    assert not _MODULE.verify_execution(sm, s3, arn)
    s3.get_object.assert_not_called()


@pytest.mark.parametrize("invalid_input", ["wrong_uri", "missing", "duplicate"])
def test_validation_must_consume_current_evaluation(execution, invalid_input):
    sm, s3, arn, _, _, _ = execution
    inputs = sm.describe_processing_job.return_value["ProcessingInputs"]
    if invalid_input == "wrong_uri":
        inputs[0]["S3Input"]["S3Uri"] = "s3://fixture/other/model.tar.gz"
    elif invalid_input == "missing":
        inputs.clear()
    else:
        inputs.append(inputs[0].copy())
    with pytest.raises(ValueError, match="Validate must consume this execution's SimEval output"):
        _MODULE.verify_execution(sm, s3, arn)
    s3.get_object.assert_not_called()
    sm.describe_model_package.assert_not_called()


@pytest.mark.parametrize("invalid_input", ["wrong_uri", "missing", "duplicate"])
def test_validation_must_consume_execution_checkpoint(execution, invalid_input):
    sm, s3, arn, _, _, _ = execution
    inputs = sm.describe_processing_job.return_value["ProcessingInputs"]
    checkpoint_input = inputs[1]
    if invalid_input == "wrong_uri":
        checkpoint_input["S3Input"]["S3Uri"] = "s3://fixture/other/model.tar.gz"
    elif invalid_input == "missing":
        inputs.remove(checkpoint_input)
    else:
        inputs.append(checkpoint_input.copy())
    with pytest.raises(ValueError, match="Validate must consume this execution's checkpoint"):
        _MODULE.verify_execution(sm, s3, arn)
    s3.get_object.assert_not_called()
    sm.describe_model_package.assert_not_called()


def test_evidence_read_failure_propagates(execution):
    sm, s3, arn, _, _, _ = execution
    s3.get_object.side_effect = RuntimeError("evidence unavailable")
    with pytest.raises(RuntimeError, match="evidence unavailable"):
        _MODULE.verify_execution(sm, s3, arn)
