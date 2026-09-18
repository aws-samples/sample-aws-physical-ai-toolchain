#!/usr/bin/env python3
"""Require the real Validate container to reject a completed run's wrong seed."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import boto3


def check_rejection(state, log, output, expected_seed, observed_seed):
    if state["Running"] or state["ExitCode"] == 0 or state["OOMKilled"]:
        raise RuntimeError("Negative control did not exit with an ordinary validation failure")
    reason = f"eval_seed mismatch: expected={expected_seed} got={observed_seed}"
    if reason not in log:
        raise RuntimeError("Negative control failed for a reason other than the injected seed")
    if any(output.iterdir()):
        raise RuntimeError("Negative control unexpectedly emitted accepted output")


def run_control(root):
    def load(name):
        return json.loads((root / name).read_text())

    manifest, parameters = load("manifest.json"), load("parameters.json")
    verified = load("independent-verification.json")
    if verified["status"] != "Succeeded" or verified["run_id"] != manifest["run_id"]:
        raise RuntimeError("Start from this run's independently verified positive result")
    if verified["canonical_commit"] != manifest["canonical_commit"]:
        raise RuntimeError("Positive verification refers to different code")
    bucket = manifest["development_bucket"]
    s3 = boto3.Session(region_name=manifest["region"]).client("s3")
    control_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    work = root / ("negative-eval-seed-" + control_id)
    work.mkdir()
    code, data, output = (work / name for name in ("code", "eval", "output"))
    for directory in (code, data, output):
        directory.mkdir()

    def download(uri, destination):
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != bucket:
            raise RuntimeError("Negative control input is outside this run's dev bucket")
        key = parsed.path.lstrip("/")
        head = s3.head_object(Bucket=bucket, Key=key)
        version = head.get("VersionId")
        if version in (None, "", "null") or head["ContentLength"] > 64 * 1024**2:
            raise RuntimeError("Expected a small, versioned validator/evaluation input")
        value = s3.get_object(Bucket=bucket, Key=key, VersionId=version)["Body"].read()
        destination.write_bytes(value)
        return {"uri": uri, "version_id": version, "sha256": hashlib.sha256(value).hexdigest()}

    validator = download(manifest["validator_source"], code / "validate_entry.py")
    if validator["sha256"] != urlparse(manifest["validator_source"]).path.split("/")[-2]:
        raise RuntimeError("Staged validator does not match its content-addressed URI")
    step = next(row for row in load("steps.json")["PipelineExecutionSteps"]
                if row["StepName"] == "SimEval")
    job = step["Metadata"]["TrainingJob"]["Arn"].rsplit("/", 1)[-1]
    definition = next(row for row in load("pipeline-definition.redacted.json")["Steps"]
                      if row["Name"] == "SimEval")
    prefix = definition["Arguments"]["OutputDataConfig"]["S3OutputPath"].rstrip("/")
    evaluation = download(f"{prefix}/{job}/output/model.tar.gz", data / "model.tar.gz")
    environment = {
        "EXPECTED_EVAL_SEED": str(parameters["EvalSeed"] + 1),
        "EXPECTED_EVAL_TRIALS": str(parameters["EvalTrials"]),
        "EXPECTED_EVAL_TASK_IDS": parameters["EvalTaskIds"],
        "EXPECTED_SUITE": parameters["Suite"],
        "EXPECTED_MODEL_FAMILY": parameters["ModelFamily"],
    }
    name = f"vla-bad-seed-{manifest['run_id'].lower()}-{control_id.lower()}"
    command = ["docker", "run", "--name", name, "--network", "none",
               "--label", "vla.local.validation-control=" + manifest["run_id"],
               "--mount", f"type=bind,source={code},target=/opt/ml/processing/input/code,readonly",
               "--mount", f"type=bind,source={data},target=/opt/ml/processing/eval_output",
               "--mount", f"type=bind,source={output},target=/opt/ml/processing/output",
               "--entrypoint", "python"]
    for key, value in environment.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend([manifest["images"]["validate"],
                    "/opt/ml/processing/input/code/validate_entry.py"])
    try:
        process = subprocess.run(command, text=True, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "stop", "--time", "10", name],
                       timeout=30, capture_output=True, check=False)
        raise
    log = process.stdout + "\n" + process.stderr
    (work / "validate.log").write_text(log)
    actual = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
    if actual["Image"] != manifest["image_ids"]["validate"]:
        raise RuntimeError("Negative control ran a different Validate image")
    check_rejection(actual["State"], log, output,
                    parameters["EvalSeed"] + 1, parameters["EvalSeed"])
    result = {
        "status": "ExpectedFailureVerified", "run_id": manifest["run_id"],
        "positive_execution_id": verified["pipeline_execution_id"],
        "canonical_commit": manifest["canonical_commit"],
        "control_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validator": validator, "evaluation": evaluation,
        "observed_seed": parameters["EvalSeed"], "injected_expected_seed": parameters["EvalSeed"] + 1,
        "container": actual["Id"], "exit_code": actual["State"]["ExitCode"],
        "image_id": actual["Image"], "network_mode": actual["HostConfig"]["NetworkMode"],
        "accepted_output_files": [], "control_directory": str(work),
    }
    if result["network_mode"] != "none":
        raise RuntimeError("Negative control must have no network access or publication path")
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "negative-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "negative-validation.log").write_text(log)
    subprocess.run(["docker", "rm", actual["Id"]], check=True, capture_output=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(run_control(args.run_dir.resolve()), indent=2))
