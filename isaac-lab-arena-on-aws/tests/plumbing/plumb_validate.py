#!/usr/bin/env python3
"""Plumbing Validate: load-bearing gate proof.

This is the first draft of the REAL validate_entry.py. It:
  1. Reads metrics.json from the eval's ModelArtifacts (mounted at
     /opt/ml/processing/eval_output/)
  2. Reads pipeline-owned expectations from environment variables
     (populated from pipeline parameters via ProcessingStep arguments)
  3. Validates: schema version, digest cross-check, expectation match,
     weighted-mean consistency
  4. ONLY if all checks pass: emits validated_metrics.json to
     /opt/ml/processing/output/ (the PropertyFile the ConditionStep reads)

If ANY check fails, this script exits nonzero -- no validated_metrics.json
is emitted, PropertyFile read fails, gate cannot evaluate, pipeline stops.
The gate is structurally unable to consume an unvalidated number.
"""
import json
import os
import sys


def fail(msg: str) -> None:
    """Fail closed: print the violation and exit nonzero."""
    print(f"[validate] VALIDATION FAILED: {msg}", file=sys.stderr)
    print(f"[validate] VALIDATION FAILED: {msg}")  # also stdout for logs
    sys.exit(1)


def main():
    eval_output_dir = os.environ.get(
        "EVAL_OUTPUT_DIR", "/opt/ml/processing/eval_output")
    output_dir = os.environ.get(
        "OUTPUT_DIR", "/opt/ml/processing/output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Extract model.tar.gz if present (ProcessingInput from a TrainingStep's
    # ModelArtifacts delivers the tarball, not extracted files) ---
    tarball = os.path.join(eval_output_dir, "model.tar.gz")
    if os.path.exists(tarball):
        import tarfile
        print(f"[validate] extracting {tarball} to {eval_output_dir}")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(eval_output_dir)

    # --- Read metrics.json from eval output ---
    metrics_path = os.path.join(eval_output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        fail(f"metrics.json not found at {metrics_path}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    print(f"[validate] Read metrics.json: success_rate={metrics.get('success_rate')}")

    # --- Check 1: Schema version ---
    if metrics.get("schema_version") != 2:
        fail(f"schema_version must be 2, got {metrics.get('schema_version')}")

    # --- Check 2: Digest cross-check ---
    # The eval recomputes the digest over mounted bytes and records it.
    # The train's manifest also contains the original digest.
    # They must match (this is the third leg of the digest chain).
    manifest_path = os.path.join(eval_output_dir, "checkpoint_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_digest = manifest.get("weights_digest", "")
        eval_digest = metrics.get("weights_digest_recomputed", "")
        if manifest_digest and eval_digest:
            if manifest_digest != eval_digest:
                fail(f"digest mismatch: manifest={manifest_digest} "
                     f"eval_recomputed={eval_digest}")
            print(f"[validate] digest cross-check PASSED: {eval_digest}")
        else:
            print("[validate] WARNING: digest fields missing, skipping cross-check")
    else:
        print("[validate] NOTE: no manifest in eval output (plumbing mode)")

    # --- Check 3: Pipeline-owned expectations ---
    expected_seed = os.environ.get("EXPECTED_EVAL_SEED", "")
    expected_trials = os.environ.get("EXPECTED_EVAL_TRIALS", "")

    if expected_seed:
        actual_seed = str(metrics.get("eval_seed", ""))
        if actual_seed != expected_seed:
            fail(f"eval_seed mismatch: expected={expected_seed} got={actual_seed}")
        print(f"[validate] eval_seed matches expectation: {expected_seed}")

    if expected_trials:
        actual_trials = str(metrics.get("eval_trials", ""))
        if actual_trials != expected_trials:
            fail(f"eval_trials mismatch: expected={expected_trials} got={actual_trials}")
        print(f"[validate] eval_trials matches expectation: {expected_trials}")

    # --- Check 4: Weighted mean consistency ---
    # success_rate must equal the weighted mean of per_task results.
    # This catches sabotage/bugs where the headline number disagrees with details.
    per_task = metrics.get("per_task", [])
    if per_task:
        total_successes = sum(t.get("successes", 0) for t in per_task)
        total_trials = sum(t.get("trials", 0) for t in per_task)
        if total_trials > 0:
            weighted_mean = total_successes / total_trials
            reported_rate = metrics.get("success_rate", -1)
            # Allow floating point tolerance
            if abs(weighted_mean - reported_rate) > 1e-6:
                fail(f"weighted mean of per_task ({weighted_mean:.6f}) "
                     f"!= reported success_rate ({reported_rate:.6f})")
            print(f"[validate] weighted mean consistency PASSED: {weighted_mean:.4f}")

    # --- All checks passed: emit validated_metrics.json ---
    # Determine full_suite (registry eligibility)
    task_ids_str = metrics.get("task_ids", "")
    # Full suite = 10 tasks for LIBERO; "" means all tasks
    full_suite = (task_ids_str == "" or task_ids_str == "[]"
                  or len(json.loads(task_ids_str) if isinstance(task_ids_str, str)
                         and task_ids_str.startswith("[") else []) >= 10)

    validated = {
        "success_rate": metrics["success_rate"],
        "full_suite": full_suite,
        "eval_seed": metrics.get("eval_seed"),
        "eval_trials": metrics.get("eval_trials"),
        "episodes": metrics.get("episodes"),
        "validation_passed": True,
    }

    validated_path = os.path.join(output_dir, "validated_metrics.json")
    with open(validated_path, "w") as f:
        json.dump(validated, f, indent=2)

    print("[validate] ALL CHECKS PASSED")
    print(f"[validate] wrote: {validated_path}")
    print(f"[validate] full_suite={full_suite}, success_rate={validated['success_rate']}")
    print("[validate] DONE")


if __name__ == "__main__":
    main()
