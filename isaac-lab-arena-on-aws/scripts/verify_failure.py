#!/usr/bin/env python3
"""Verify a designed-failure execution -- the failure IS the pass.

Usage: PYTHONPATH=src python scripts/verify_failure.py <execution-arn> <expected-failed-step> <expected-violation-substring>

Example: verify_failure.py <arn> Validate "eval_seed mismatch"

Exits 0 only when the failure is exactly the designed one.
"""
from __future__ import annotations

import sys

import boto3

sys.path.insert(0, "src")
from vla_pipeline.config import load_config


def check(name, condition, evidence=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if evidence:
        for line in evidence.strip().split("\n")[:5]:
            print(f"        {line}")
    return condition


def main():
    if len(sys.argv) < 4:
        print("Usage: verify_failure.py <execution-arn> <failed-step> <violation-substring>")
        sys.exit(1)

    arn = sys.argv[1]
    expected_failed_step = sys.argv[2]
    expected_violation = sys.argv[3]

    cfg = load_config()
    sm = boto3.client("sagemaker", region_name=cfg.region)

    print(f"\nVerifying designed failure: {arn}")
    print(f"  Expected: {expected_failed_step} fails with '{expected_violation}'")
    print(f"{'='*60}")

    all_steps = []
    token = None
    while True:
        kwargs = {"PipelineExecutionArn": arn}
        if token:
            kwargs["NextToken"] = token
        steps_resp = sm.list_pipeline_execution_steps(**kwargs)
        all_steps.extend(steps_resp["PipelineExecutionSteps"])
        token = steps_resp.get("NextToken")
        if not token:
            break
    steps = {s["StepName"]: s for s in all_steps}

    results = []

    # 1. The execution itself reached a terminal Failed state
    exec_resp = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    exec_status = exec_resp["PipelineExecutionStatus"]
    results.append(check("Execution reached terminal Failed state",
                         exec_status == "Failed",
                         f"Actual execution status: {exec_status}"))

    # 2. The expected step Failed
    failed_step = steps.get(expected_failed_step, {})
    failed_status = failed_step.get("StepStatus")
    results.append(check(f"{expected_failed_step} status is Failed",
                         failed_status == "Failed",
                         f"Actual status: {failed_status}"))

    # 3. The failure reason contains the expected violation
    failure_reason = failed_step.get("FailureReason", "")
    results.append(check(f"Failure reason contains '{expected_violation}'",
                         expected_violation.lower() in failure_reason.lower(),
                         f"Reason: {failure_reason[:200]}"))

    # 4. Downstream steps (gate, register) show no activity
    _DOWNSTREAM_ACTIVE = {"Succeeded", "Failed", "Stopped", "Executing", "Starting"}
    gate = steps.get("SuccessGate", {})
    gate_status = gate.get("StepStatus")
    gate_active = gate_status in _DOWNSTREAM_ACTIVE
    results.append(check("SuccessGate shows no activity",
                         not gate_active,
                         f"Gate status: {gate_status or 'not present'}"))

    # 5. No model package registered -- check ALL register steps and ALL ARNs
    # regardless of step status (a failed/stopped register can still carry an ARN)
    register_steps = [s for name, s in steps.items() if "Register" in name]
    register_arns = []
    for rs in register_steps:
        package_arn = rs.get("Metadata", {}).get("RegisterModel", {}).get("Arn", "")
        if package_arn:
            register_arns.append((rs.get("StepName", "?"), rs.get("StepStatus", "?"), package_arn))
    register_active = any(s.get("StepStatus") in _DOWNSTREAM_ACTIVE for s in register_steps)
    results.append(check("No model package registered or registration attempted",
                         not register_active and not register_arns,
                         f"Register steps: {len(register_steps)}, "
                         f"active: {register_active}, ARNs: {register_arns}"))

    # 6. No downstream steps succeeded after the failed step
    failed_end = failed_step.get("EndTime")
    if failed_end:
        downstream = [s for s in all_steps
                      if s.get("StartTime") and s["StartTime"] > failed_end
                      and s.get("StepStatus") == "Succeeded"]
        results.append(check("No downstream steps succeeded after failure",
                             not downstream,
                             f"Downstream succeeded: {[s['StepName'] for s in downstream]}"))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  RESULT: {passed}/{total} (designed failure verified)")
    print(f"{'='*60}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
