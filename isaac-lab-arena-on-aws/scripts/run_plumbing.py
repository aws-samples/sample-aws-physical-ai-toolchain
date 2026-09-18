#!/usr/bin/env python3
"""Deploy and run the plumbing pipeline (3 executions).

Run 1: gate-true (threshold 0.5, score 0.75)
Run 2: gate-false (threshold 0.99, score 0.75)
Run 3: sabotage (weighted mean != success_rate -> Validate FAILS)

Usage:
    PYTHONPATH=src python scripts/run_plumbing.py [--run 1|2|3|all]
"""
from __future__ import annotations

import argparse
import sys

import boto3
import sagemaker

sys.path.insert(0, "src")
from vla_pipeline.config import load_config
from vla_pipeline.runner import (
    build_plumbing_pipeline,
    describe_steps,
    upsert_versioned,
    wait_for_execution,
)


def deploy_plumbing(cfg, sabotage: bool = False):
    """Deploy the plumbing pipeline and return the ARN."""
    # I9: this constructed a session with NO default bucket, so the SDK uploaded the local
    # processing script to sagemaker-<region>-<account> (SDK 2.257.6, processing.py:787-818).
    # The validation role can read source only from the Foundation models bucket
    # (trust_boundary.tf), so Validate could not fetch its own code and the rehearsal would
    # fail on an access error unrelated to what it is rehearsing.
    #
    # Production stages validator code at a content-addressed trust-bucket URI instead
    # (pipeline.py:454-461); the rehearsal does not use that mechanism, so it needs an
    # explicitly authorized code bucket rather than the SDK's default.
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=cfg.region),
        default_bucket=cfg.bucket)

    plumbing_dir = "tests/plumbing"
    pipeline = build_plumbing_pipeline(cfg, session, plumbing_dir, sabotage=sabotage)

    suffix = "-sabotage" if sabotage else ""
    pipeline.name = f"{cfg.pipeline_name}-plumbing{suffix}"

    result = upsert_versioned(pipeline, cfg.role_arn)
    arn = result["PipelineArn"]
    print(f"Deployed: {pipeline.name} -> {arn}")
    return result["PipelineVersionId"], pipeline.name


def run_execution(cfg, pipeline_name: str, threshold: float, label: str,
                  eval_seed: int = 1000, *, pipeline_version_id: int):
    """Start an execution, wait, and print results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  pipeline={pipeline_name}, threshold={threshold}, seed={eval_seed}")
    print(f"{'='*60}")

    sm = boto3.client("sagemaker", region_name=cfg.region)
    params = [
        {"Name": "SuccessThreshold", "Value": str(threshold)},
        {"Name": "EvalSeed", "Value": str(eval_seed)},
        {"Name": "EvalTrials", "Value": "3"},
    ]

    resp = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineVersionId=pipeline_version_id,
        PipelineParameters=params,
    )
    arn = resp["PipelineExecutionArn"]
    print(f"  Started: {arn}")

    # Wait
    status = wait_for_execution(cfg, arn, poll_seconds=15, timeout_seconds=1800)
    print(f"  Final status: {status}")

    # Get steps
    steps = describe_steps(cfg, arn)
    print("  Steps:")
    for s in steps:
        name = s.get("StepName", "?")
        st = s.get("StepStatus", "?")
        reason = s.get("FailureReason", "")
        cond = s.get("Metadata", {}).get("Condition", {}).get("Outcome", "")
        extra = f" (condition={cond})" if cond else ""
        extra += f" REASON: {reason}" if reason else ""
        print(f"    {name}: {st}{extra}")

    return {"arn": arn, "status": status, "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="all", help="1, 2, 3, or all")
    args = parser.parse_args()

    cfg = load_config()
    runs_to_do = [1, 2, 3] if args.run == "all" else [int(args.run)]

    results = {}

    if 1 in runs_to_do or 2 in runs_to_do:
        # Deploy normal plumbing pipeline (for runs 1 & 2)
        version, name = deploy_plumbing(cfg, sabotage=False)

        if 1 in runs_to_do:
            results[1] = run_execution(cfg, name, threshold=0.5,
                                       label="Run 1: gate-true (0.75 >= 0.5)",
                                       pipeline_version_id=version)
        if 2 in runs_to_do:
            results[2] = run_execution(cfg, name, threshold=0.99,
                                       label="Run 2: gate-false (0.75 < 0.99)",
                                       pipeline_version_id=version)

    if 3 in runs_to_do:
        # Deploy sabotage variant (separate pipeline with pre-sabotaged eval)
        version, name = deploy_plumbing(cfg, sabotage=True)
        results[3] = run_execution(cfg, name, threshold=0.5,
                                   label="Run 3: sabotage (Validate must FAIL)",
                                   pipeline_version_id=version)

    # Summary and assertions
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    failures = []
    expected_outcomes = {
        1: ("Succeeded", "gate-true: threshold 0.5, score 0.75"),
        # I8: this expected "Succeeded" because both ConditionStep branches were empty, so a
        # FALSE threshold produced a successful execution -- the rehearsal could not
        # demonstrate that the gate blocks anything, which is the one behaviour it exists to
        # rehearse. The graph now has an explicit FailStep, so a below-threshold run must
        # FAIL, and this script has to expect that or it rejects the corrected behaviour.
        2: ("Failed", "gate-false: below-threshold must FAIL via PlumbingBelowThreshold"),
        3: ("Failed", "sabotage: weighted-mean mismatch -> Validate fails"),
    }
    for run_num, r in sorted(results.items()):
        step_summary = ", ".join(
            f"{s.get('StepName', '?')}={s.get('StepStatus', '?')}"
            for s in r["steps"]
        )
        print(f"  Run {run_num}: {r['status']} -- {step_summary}")
        expected_status, desc = expected_outcomes[run_num]
        if r["status"] != expected_status:
            failures.append(f"Run {run_num} ({desc}): expected {expected_status}, "
                            f"got {r['status']}")
    if 1 in results:
        gate = next((s for s in results[1]["steps"]
                     if s.get("StepName") == "SuccessGate"), {})
        outcome = gate.get("Metadata", {}).get("Condition", {}).get("Outcome", "")
        if outcome != "True":
            failures.append(f"Run 1 gate outcome: expected True, got {outcome!r}")
    if 2 in results:
        gate = next((s for s in results[2]["steps"]
                     if s.get("StepName") == "SuccessGate"), {})
        outcome = gate.get("Metadata", {}).get("Condition", {}).get("Outcome", "")
        if outcome != "False":
            failures.append(f"Run 2 gate outcome: expected False, got {outcome!r}")
        # I8: a Failed execution is not enough -- it must fail for the RIGHT reason. Without
        # this, an unrelated failure anywhere in the graph would satisfy the expectation and
        # the rehearsal would report that the gate blocks when it may not have run at all.
        blocker = next((s for s in results[2]["steps"]
                        if s.get("StepName") == "PlumbingBelowThreshold"), None)
        if blocker is None:
            failures.append(
                "Run 2: the PlumbingBelowThreshold step did not execute, so the below-"
                "threshold path was not demonstrated -- the run failed for another reason")
        elif blocker.get("StepStatus") != "Failed":
            failures.append(
                f"Run 2 PlumbingBelowThreshold: expected Failed, got "
                f"{blocker.get('StepStatus', 'not present')!r}")

    if 3 in results:
        validate = next((s for s in results[3]["steps"]
                         if s.get("StepName") == "Validate"), {})
        if validate.get("StepStatus") != "Failed":
            failures.append(f"Run 3 Validate step: expected Failed, "
                            f"got {validate.get('StepStatus', 'not present')!r}")
        reason = validate.get("FailureReason", "")
        if "weighted mean" not in reason.lower():
            failures.append(f"Run 3 Validate failure reason does not mention "
                            f"'weighted mean': {reason[:200]!r}")
        register = [s for s in results[3]["steps"]
                    if "Register" in s.get("StepName", "")
                    and s.get("StepStatus") == "Succeeded"]
        if register:
            failures.append(f"Run 3 has Succeeded register steps: "
                            f"{[s['StepName'] for s in register]}")

    if failures:
        print(f"\n  ASSERTIONS FAILED ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("\n  ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
