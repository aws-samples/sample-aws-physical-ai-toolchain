#!/usr/bin/env python3
"""Run the sample grid from the manifest -- one command, all families.

Usage: PYTHONPATH=src python scripts/run_matrix.py --manifest config/matrix_manifest.json --deploy
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import boto3

sys.path.insert(0, "src")
from vla_pipeline.common.sourcedir import stage
from vla_pipeline.config import load_config

from vla_pipeline.pipeline import build_pipeline
from vla_pipeline.registry import (
    check_storage_fits,
    require_image_capability,
    resolve_image_digest,
    select_instance,
    select_volume_gb,
    DATASET_FROM_SUITE_MANIFEST,
    resolve,
    resolve_suite,
    suites_for_simulator,
)
from vla_pipeline.runner import stage_validate_code, upload_code, upload_directory, upsert_versioned


def gate_cell(cell: dict, index: int):
    """Resolve + gate one matrix cell, returning its ResolvedSuite. Fail loud.

    Extracted from main() so it is callable -- a test that greps main()'s source for
    a gate cannot tell a live gate from a commented-out one. config/
    matrix_manifest.json is repo-controlled input, but "the suite defines the cell"
    should hold for every entry point, and a typo or a wrong-simulator cell would
    otherwise cost the whole matrix's GPU spend.
    """
    suite = resolve_suite(cell["suite"])
    if suite.simulator != "libero":
        raise SystemExit(
            f"FATAL: matrix cell {index} suite {cell['suite']!r} runs on simulator "
            f"{suite.simulator!r}, not 'libero'. LIBERO suites: "
            f"{', '.join(suites_for_simulator('libero'))}")
    if not suite.supported:
        raise SystemExit(
            f"FATAL: matrix cell {index} suite {cell['suite']!r} is status: "
            f"experimental -- Validate would reject the report.")
    if cell.get("family") == "gr00t" and suite.dataset is None:
        raise SystemExit(
            f"FATAL: matrix cell {index} suite {cell['suite']!r} declares no "
            f"fine-tune dataset and family is gr00t (SUITE_DATASETS lookup). "
            f"OpenVLA/MolmoAct2 resolve datasets independently.")
    return suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    def _positive_int(v):
        iv = int(v)
        if iv < 1:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
        return iv

    parser.add_argument("--max-concurrent", type=_positive_int, default=3)
    deployment = parser.add_mutually_exclusive_group(required=True)
    deployment.add_argument("--deploy", action="store_true", help="Deploy/upsert pipeline first")
    deployment.add_argument("--pipeline-version-id", type=int,
                            help="Existing pipeline version to execute")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    protocol = manifest["protocol"]
    cells = manifest["cells"]

    print(f"{'='*60}")
    print(f"  SMOKE GRID: {len(cells)} cells")
    print(f"  Protocol: dose={protocol['train_steps']}, trials={protocol['eval_trials']}, "
          f"seed={protocol['eval_seed']}, tasks={protocol['eval_task_ids']}")
    print(f"{'='*60}")

    # Gate ALL cells BEFORE anything else -- before load_config, any upload, the
    # upsert, or a single submission. Gating inside the submission loop meant a bad
    # cell aborted midway, leaving earlier cells running and S3 artifacts uploaded:
    # worse than not starting. Every check is offline, so it must not require
    # resolvable AWS config either.
    gated = [gate_cell(cell, i) for i, cell in enumerate(cells)]
    trials = int(protocol.get("eval_trials", 1))
    task_ids = protocol.get("eval_task_ids", "all")
    for i, cell in enumerate(cells):
        if cell["family"] == "openvla" and trials > 50:
            raise SystemExit(
                f"FATAL: matrix cell {i} ({cell['family']}) eval_trials={trials} "
                "exceeds LIBERO's 50 initial states. Reduce eval_trials or "
                "exclude OpenVLA from the matrix.")
        if cell["family"] == "openvla" and task_ids != "all":
            raise SystemExit(
                f"FATAL: matrix cell {i} ({cell['family']}) eval_task_ids="
                f"{task_ids!r} but OpenVLA does not support task subsetting. "
                "The pinned evaluator rejects every non-'all' selection. "
                "Use eval_task_ids='all' or exclude OpenVLA.")
    print(f"  All {len(gated)} cells gated OK")

    cfg = load_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)
    base = f"{cfg.account_id}.dkr.ecr.{cfg.region}.amazonaws.com"

    # 1a/1c fix: stage+embed validator.py into validate_entry.py and upload
    # (content-addressed) BEFORE deploy so the definition's code= resolves to a
    # real object with the validator embedded.
    staged_validate = stage_validate_code()
    validate_uri = upload_code(cfg, staged_validate)
    print(f"  validate (validator embedded): {validate_uri}")

    pipeline_version_id = args.pipeline_version_id
    if args.deploy:
        print("\n[Deploy] Upserting pipeline...")
        from sagemaker.workflow.pipeline_context import PipelineSession
        session = PipelineSession(boto_session=boto3.Session(region_name=cfg.region))
        pipeline = build_pipeline(cfg, session=session, validate_code_uri=validate_uri)
        pipeline_version_id = upsert_versioned(pipeline, cfg.role_arn)["PipelineVersionId"]
        print("  Done.")

    # Stage and upload sourcedirs for each family
    family_sources = {}
    for cell in cells:
        family = cell["family"]
        if family not in family_sources:
            staged = stage(family)
            uri = upload_directory(cfg, staged)
            family_sources[family] = uri
            print(f"  {family} sourcedir: {uri}")

    # PREFLIGHT EVERY CELL'S EVALUATOR IMAGE BEFORE STARTING ANY OF THEM.
    #
    # Two defects, both real. This launcher performed no capability check at all, so the same image was
    # gated on run_libero and run_arena and ungated here -- and its digest resolution happened inside the
    # submission loop, so a bad image aborted the matrix AFTER earlier cells had already started, leaving
    # a partial run whose failure had nothing to do with the cells that did start.
    #
    # Resolved once, up front, keyed by image: nothing is submitted until every cell's image is known to
    # exist, resolvable to a digest, and to carry whatever its simulator declares.
    eval_digests = {}
    for cell in cells:
        _spec = resolve(cell["family"], "libero")
        _eval_image = f"{base}/{_spec.eval_image_repo}"
        if _eval_image in eval_digests:
            continue
        # Storage too, in the same preflight. run_matrix had NO storage check at all, so a cell whose
        # declared volume exceeds its instance's local NVMe total was submitted and rejected by
        # CreateTrainingJob after the matrix had started. Unconditional here because every matrix cell
        # trains -- unlike run_arena/run_libero, which omit FineTune in eval-only mode.
        check_storage_fits(select_instance(_spec, "train", None), select_volume_gb(_spec, "train", None))
        check_storage_fits(select_instance(_spec, "eval", None), select_volume_gb(_spec, "eval", None))
        _digest = resolve_image_digest(_eval_image)
        if _spec.required_image_capability:
            require_image_capability(_digest, _spec.required_image_capability)
        eval_digests[_eval_image] = _digest
        print(f"  {cell['family']} evaluator: {_digest.rsplit('/', 1)[-1]}")

    # (validate_entry.py already staged+uploaded above with the validator embedded)

    # Start all cells
    executions = []
    for i, cell in enumerate(cells):
        family = cell["family"]
        suite = gated[i].name   # already gated above; no second resolve
        spec = resolve(family, "libero")
        image = f"{base}/{spec.train_image_repo}"
        # Per ROLE: see run_libero.py. The pair manifest may declare a different evaluator image, and
        # gr00t x isaac_arena does.
        eval_image = f"{base}/{spec.eval_image_repo}"
        try:
            sm.describe_model_package_group(ModelPackageGroupName=spec.registry_group)
        except sm.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] != "ValidationException":
                raise
            sm.create_model_package_group(ModelPackageGroupName=spec.registry_group)

        params = [
            {"Name": "ModelFamily", "Value": family},
            {"Name": "ModelPackageGroupName", "Value": spec.registry_group},
            {"Name": "Suite", "Value": suite},
            {"Name": "TrainSuite", "Value": "unified" if family == "molmoact2" else suite},
            {"Name": "TrainImageUri", "Value": image},
            # Digest-form: the attestation must name the BYTES that ran, and a tag cannot. Taken from
            # the preflight above so no cell starts before every image has been verified.
            {"Name": "EvalImageUri", "Value": eval_digests[eval_image]},
            # "resolve the dataset from the suite manifest" -- not an S3 URI.
            {"Name": "DatasetS3Uri", "Value": DATASET_FROM_SUITE_MANIFEST},
            {"Name": "TrainInstanceType", "Value": select_instance(spec, "train", None)},
            {"Name": "EvalInstanceType", "Value": select_instance(spec, "eval", None)},
            {"Name": "TrainSourceDirUri", "Value": family_sources[family]},
            {"Name": "EvalSourceDirUri", "Value": family_sources[family]},
            {"Name": "TrainSteps", "Value": str(protocol["train_steps"])},
            {"Name": "EvalSeed", "Value": str(protocol["eval_seed"])},
            {"Name": "EvalTrials", "Value": str(protocol["eval_trials"])},
            {"Name": "EvalTaskIds", "Value": protocol["eval_task_ids"]},
            {"Name": "SuccessThreshold", "Value": str(protocol["success_threshold"])},
        ]
        # cycle-15 I7: this launcher never passed VolumeSizeInGB, so EVERY matrix cell ran on the
        # pipeline's 100 GB default while the family manifests declare 150/250/300. The families
        # needing the most disk were the ones silently under-provisioned, and it surfaces as a full
        # volume part-way through a paid run.
        #
        # R4: all three of these now come from the resolved spec rather than being re-derived or
        # inlined per launcher, which is what let the instance types drift apart -- this launcher
        # hardcoded ml.g6e.12xlarge for every cell regardless of what the family declared.
        params.append({"Name": "VolumeSizeInGB",
                       "Value": str(select_volume_gb(spec, "train", None))})
        params.append({"Name": "EvalVolumeSizeInGB",
                       "Value": str(select_volume_gb(spec, "eval", None))})

        # Respect concurrency
        while len([e for e in executions if e["status"] == "Executing"]) >= args.max_concurrent:
            time.sleep(30)
            for e in executions:
                if e["status"] == "Executing":
                    resp = sm.describe_pipeline_execution(PipelineExecutionArn=e["arn"])
                    e["status"] = resp["PipelineExecutionStatus"]

        resp = sm.start_pipeline_execution(
            PipelineName=cfg.pipeline_name, PipelineVersionId=pipeline_version_id,
            PipelineParameters=params)
        arn = resp["PipelineExecutionArn"]
        print(f"  [{i+1}/{len(cells)}] {family} x {suite}: {arn}")
        executions.append({"family": family, "suite": suite, "arn": arn, "status": "Executing"})

    # Wait for all
    print(f"\nWaiting for {len(executions)} executions...")
    while any(e["status"] == "Executing" for e in executions):
        time.sleep(60)
        for e in executions:
            if e["status"] == "Executing":
                resp = sm.describe_pipeline_execution(PipelineExecutionArn=e["arn"])
                e["status"] = resp["PipelineExecutionStatus"]
                if e["status"] != "Executing":
                    print(f"  {e['family']} x {e['suite']}: {e['status']}")

    # Summary
    print(f"\n{'='*60}")
    print("  GRID SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Family':<12} {'Suite':<16} {'Status':<12} ARN")
    print(f"  {'-'*12} {'-'*16} {'-'*12} ---")
    for e in executions:
        print(f"  {e['family']:<12} {e['suite']:<16} {e['status']:<12} {e['arn'].split('/')[-1]}")

    green = sum(1 for e in executions if e["status"] == "Succeeded")
    print(f"\n  {green}/{len(executions)} GREEN")
    sys.exit(0 if green == len(executions) else 1)


if __name__ == "__main__":
    main()
