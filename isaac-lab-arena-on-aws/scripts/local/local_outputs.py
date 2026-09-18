"""Check partial-run outputs without claiming Validate or promotion ran."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import PurePosixPath
from urllib.parse import urlparse


def verify_outputs(root, s3, manifest, parameters, rows, *, expected_output_buckets=None):
    """Re-read exact output versions and, for SimEval, the actual metrics archive.

    A training-only checkpoint gets an existence/version check. Simulation
    reports get the shared schema/protocol checks; checkpoint content validation
    and promotion remain the responsibility of the omitted Validate step.
    """
    from vla_pipeline.common.capped_reader import capped_tar_open
    from vla_pipeline.common.validator import parse_report, validate_report
    from vla_pipeline.registry import family_schemas, resolve_suite, suites_json

    outputs = json.loads((root / "job-outputs.json").read_text())
    expected = {row["StepName"]: row for row in rows
                if row["StepName"] in {"FineTune", "SimEval"}}
    if set(outputs) != set(expected):
        raise RuntimeError("Saved outputs do not match the executed GPU steps")
    heads = {}
    for name, output in outputs.items():
        bucket = (expected_output_buckets or {}).get(name, manifest["development_bucket"])
        job = expected[name]["Metadata"]["TrainingJob"]["Arn"]
        uri = urlparse(output["uri"])
        if (uri.scheme != "s3" or uri.netloc != bucket or output["job_name"] != job
                or not uri.path.endswith(f"/{job}/output/model.tar.gz")):
            raise RuntimeError(f"{name} output does not belong to the recorded job")
        if output["version_id"] in (None, "", "null"):
            raise RuntimeError(f"{name} output is unversioned")
        head = s3.head_object(Bucket=bucket, Key=uri.path.lstrip("/"),
                              VersionId=output["version_id"])
        if (head["VersionId"] != output["version_id"] or head["ETag"] != output["etag"]
                or head["ContentLength"] != output["bytes"] or head["ContentLength"] <= 0):
            raise RuntimeError(f"{name} output identity differs from its recorded version")
        heads[name] = head
    proof = {"outputs": outputs, "heads": heads,
             "scope": "Output presence/version; evaluation report schema when SimEval ran. "
                      "Checkpoint content validation and promotion were not performed."}
    if "SimEval" not in outputs:
        return proof, None
    output = outputs["SimEval"]
    print(f"Checking simulation evidence: {output['uri']} "
          f"({output['bytes'] / 1024**2:.1f} MiB; exact S3 version)", file=sys.stderr, flush=True)
    limit = 512 * 1024 * 1024
    if output["bytes"] > limit:
        raise RuntimeError("Evaluation evidence archive exceeds the 512 MiB inspection limit")
    uri = urlparse(output["uri"])
    body = s3.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"),
                         VersionId=output["version_id"])["Body"]
    digest, count = hashlib.sha256(), 0
    report = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                count += len(chunk)
                if count > limit:
                    raise RuntimeError("Evaluation evidence exceeded its download limit")
                digest.update(chunk)
                archive.write(chunk)
            archive.flush()
            if count != output["bytes"]:
                raise RuntimeError("Evaluation evidence download was incomplete")
            with capped_tar_open(archive.name, "r:gz", max_bytes=limit) as members:
                expanded = 0
                for index, member in enumerate(members):
                    expanded += member.size
                    if index >= 10000 or expanded > limit or member.size < 0:
                        raise RuntimeError("Evaluation archive exceeds its inspection limits")
                    name = PurePosixPath(member.name)
                    if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                        raise RuntimeError("Unexpected member in evaluation evidence archive")
                    if str(name) == "metrics.json":
                        if report is not None or not member.isfile() or member.size > 8 * 1024 * 1024:
                            raise RuntimeError("Expected one bounded metrics.json in evaluation output")
                        report = parse_report(members.extractfile(member).read().decode("utf-8"))
    finally:
        body.close()
    if report is None:
        raise RuntimeError("Simulation did not publish metrics.json")
    suite = resolve_suite(parameters["Suite"])
    if parameters["EvalTaskIds"] != "all":
        raise RuntimeError("This frontend expects the complete selected task set")
    validate_report(
        report, expected_suite=suite.name, expected_trials=parameters["EvalTrials"],
        expected_eval_seed=parameters["EvalSeed"], expected_task_ids=list(suite.canonical_task_ids),
        family_schemas=family_schemas(), pipeline_mode=True, suite_specs=suites_json(),
        expected_family_version=parameters["Gr00tVersion"],
    )
    if report["model_family"] != parameters["ModelFamily"]:
        raise RuntimeError("Evaluation output belongs to another model family")
    checkpoint = parameters.get("CheckpointS3Uri") or outputs["FineTune"]["uri"]
    source = urlparse(checkpoint)
    identity = report["model_artifact_identity"]
    if (identity["s3_uri"], identity["bucket"], identity["key"]) != (
        checkpoint, source.netloc, source.path.lstrip("/"),
    ):
        raise RuntimeError("Evaluation output identifies a different checkpoint")
    expected_identity = manifest.get("checkpoint_identity") or outputs["FineTune"]
    if (identity["version_id"] != expected_identity["version_id"]
            or identity["etag"].strip('"') != expected_identity["etag"].strip('"')):
        raise RuntimeError("Evaluated checkpoint differs from the selected version")
    proof.update(evaluation_archive_sha256=digest.hexdigest(),
                 evaluation_archive_bytes=count, report_schema_checked=True)
    (root / "raw-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return proof, report
