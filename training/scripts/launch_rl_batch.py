"""
AWS Batch Multi-Node Parallel RL launcher.

Submits an Isaac Lab RL training job to AWS Batch using the multi-node parallel
job definition. Batch provisions the GPU instances, sets up NCCL topology via
environment variables, and mounts EFS for shared checkpoints.

This is an ALTERNATIVE to launch_rl.py (SageMaker). Use Batch when:
- You want lower per-hour cost (EC2 Spot vs SageMaker managed instances)
- You need more control over the instance lifecycle
- You're prototyping multi-node NCCL training

Account/region/queue/jobdef are resolved from the caller and stack outputs.

Usage:
    # Preview the job (no AWS writes):
    python training/scripts/launch_rl_batch.py --dry-run

    # Smoke test: 50 iterations, 2 nodes:
    python training/scripts/launch_rl_batch.py \
        --task Isaac-Velocity-Flat-Anymal-D-v0 \
        --num-envs 4096 --max-iterations 50 --num-nodes 2

    # Larger run (override job queue/def from stack outputs):
    python training/scripts/launch_rl_batch.py \
        --task Isaac-Velocity-Flat-Anymal-D-v0 \
        --num-envs 8192 --max-iterations 1500 --num-nodes 4 \
        --job-queue physical-ai-dev-rl-queue \
        --job-definition physical-ai-dev-rl-mnp

NOTE: Multi-node NCCL convergence is UNVALIDATED on hardware. This launcher
wires the Batch MNP request correctly (nodeOverrides, numNodes), but end-to-end
distributed training has not been verified on g6 instances. See the Batch stack
outputs for monitoring commands.
"""

import argparse
import json
import os
import sys
from typing import Optional

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")


def _account() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def _default_queue() -> str:
    return os.environ.get("BATCH_QUEUE", f"{PROJECT_NAME}-{ENVIRONMENT}-rl-queue")


def _default_job_def() -> str:
    return os.environ.get("BATCH_JOB_DEF", f"{PROJECT_NAME}-{ENVIRONMENT}-rl-mnp")


def _job_def_num_nodes(batch_client, job_definition: str) -> Optional[int]:
    """Return the node count baked into the active job definition, or None if it
    can't be resolved (so the caller falls back to letting Batch validate)."""
    try:
        resp = batch_client.describe_job_definitions(
            jobDefinitionName=job_definition, status="ACTIVE"
        )
        defs = resp.get("jobDefinitions", [])
        if not defs:
            return None
        return defs[0].get("nodeProperties", {}).get("numNodes")
    except Exception:
        return None


def launch(
    task: str,
    num_envs: int,
    max_iterations: int,
    framework: str,
    num_nodes: int,
    job_queue: str,
    job_definition: str,
    dry_run: bool,
):
    job_name = f"batch-rl-{int(__import__('time').time())}"

    # Node overrides: set hyperparameters as environment variables for all nodes
    node_overrides = {
        "nodePropertyOverrides": [
            {
                # Must match a node range declared in the job definition
                # (Batch rejects open-ended "0:" against an explicit "0:N-1" range).
                "targetNodes": f"0:{num_nodes - 1}",
                "containerOverrides": {
                    "environment": [
                        {"name": "TASK", "value": task},
                        {"name": "NUM_ENVS", "value": str(num_envs)},
                        {"name": "MAX_ITERATIONS", "value": str(max_iterations)},
                        {"name": "FRAMEWORK", "value": framework},
                        # PROC_PER_NODE is set by the job definition (matches GPU count)
                    ]
                },
            }
        ],
        # NOTE: no top-level "numNodes" override. The job definition declares a
        # fixed, closed node range (0:N-1), and Batch only accepts a numNodes
        # override when that range is open-ended (0: or :). Node count is therefore
        # set by the job definition; --num-nodes must match it (default: 2).
    }

    job_request = {
        "jobName": job_name,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "nodeOverrides": node_overrides,
    }

    print(f"{'='*60}")
    print(f"  AWS Batch Multi-Node Parallel RL Training")
    print(f"  Job:        {job_name}")
    print(f"  Task:       {task}")
    print(f"  Envs/iters: {num_envs} envs, {max_iterations} iterations ({framework})")
    print(f"  Nodes:      {num_nodes}")
    print(f"  Queue:      {job_queue}")
    print(f"  Job def:    {job_definition}")
    print(f"{'='*60}")

    if task.startswith("PickAndPlaceUR3") or task.startswith("PickAndPlace-UR3"):
        print(
            "\n  WARNING: the UR3 task is not yet GPU-validated and won't resolve in a "
            "training job. Use a built-in task (e.g. Isaac-Velocity-Flat-Anymal-D-v0).\n"
        )

    if num_nodes > 1:
        print(
            f"\n  NOTE: Multi-node training ({num_nodes} nodes) is wired but not yet "
            "validated on hardware.\n"
        )

    if dry_run:
        print("[dry-run] Would call batch.submit_job with:\n")
        print(json.dumps(job_request, indent=2))
        print("\n[dry-run] No AWS calls made.")
        return job_name

    batch_client = boto3.client("batch", region_name=REGION)

    # Guard: --num-nodes must match the node count baked into the job definition.
    # The job def declares a fixed, closed range (0:N-1), so Batch rejects a
    # mismatching targetNodes/numNodes override with a cryptic API error. Catch it
    # here with an actionable message before we ever call submit_job.
    expected = _job_def_num_nodes(batch_client, job_definition)
    if expected is not None and expected != num_nodes:
        print(
            f"\n  ERROR: --num-nodes {num_nodes} does not match job definition "
            f"'{job_definition}' which is deployed for {expected} node(s).\n"
            f"  Either rerun with --num-nodes {expected}, or redeploy the Batch stack "
            f"with the desired node count (cdk: batch.numNodes).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    response = batch_client.submit_job(**job_request)
    job_id = response["jobId"]

    print(f"\n  Launched! Job ID: {job_id}")
    print(f"\n  Monitor:")
    print(f"    aws batch describe-jobs --jobs {job_id} --region {REGION}")
    print(f"\n  Checkpoints persist to EFS at: /efs/models/{job_id}")
    print(
        f"  (mount the EFS filesystem to a workstation or use SSM to inspect the compute instances)"
    )
    return job_id


def main():
    p = argparse.ArgumentParser(
        description="AWS Batch Multi-Node Parallel RL launcher"
    )
    p.add_argument(
        "--task",
        default="Isaac-Velocity-Flat-Anymal-D-v0",
        help="Isaac Lab task id (built-in tasks work today; UR3 registered but not container-wired)",
    )
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument(
        "--framework", default="rsl_rl", choices=["rsl_rl", "skrl", "rl_games"]
    )
    p.add_argument(
        "--num-nodes", type=int, default=2, help="Number of Batch compute nodes"
    )
    p.add_argument(
        "--job-queue",
        default=None,
        help="Batch job queue name (default: from env or physical-ai-dev-rl-queue)",
    )
    p.add_argument(
        "--job-definition",
        default=None,
        help="Batch job definition name (default: from env or physical-ai-dev-rl-mnp)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the job; make no AWS calls",
    )
    args = p.parse_args()

    job_queue = args.job_queue or _default_queue()
    job_definition = args.job_definition or _default_job_def()

    launch(
        args.task,
        args.num_envs,
        args.max_iterations,
        args.framework,
        args.num_nodes,
        job_queue,
        job_definition,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
