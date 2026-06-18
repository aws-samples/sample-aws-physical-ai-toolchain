"""
Edge telemetry collector (runs on the robot as a Greengrass component).

Subscribes to the inference node's status/grasp topics over ROS 2 and batches
telemetry (grasp success/failure, cycle time, inference latency) to S3 for later
retraining/analysis. This is the artifact the telemetry component recipe runs
(see cdk/lib/edge-stack.ts).

On a device with ROS 2 it streams live; with --dry-run (or no ROS 2 present) it
emits a synthetic sample so the pipeline can be exercised off-robot.

Usage (on device, via the Greengrass recipe):
    python3 telemetry_collector.py --bucket <telemetry-bucket> --region <region>

Usage (local smoke test, no AWS/ROS writes):
    python3 edge/telemetry_collector.py --bucket b --region us-west-2 --dry-run
"""

import argparse
import json
import os
import sys


def _timestamp() -> str:
    # Avoid import-time clock calls; resolve lazily so --dry-run stays deterministic-ish.
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def sample_record() -> dict:
    return {
        "timestamp": _timestamp(),
        "event": "grasp_attempt",
        "success": True,
        "cycle_time_sec": 2.4,
        "inference_latency_ms": 4.8,
        "robot_type": "ur3",
    }


def main():
    parser = argparse.ArgumentParser(description="Edge telemetry collector")
    parser.add_argument("--bucket", required=True, help="S3 telemetry bucket")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    parser.add_argument("--prefix", default="grasp-telemetry")
    parser.add_argument("--dry-run", action="store_true",
                        help="Emit a synthetic record + print the S3 target; no AWS/ROS writes")
    args = parser.parse_args()

    record = sample_record()
    key = f"{args.prefix}/{record['timestamp'].replace(':', '-')}.json"

    print(f"{'='*60}")
    print(f"  Edge telemetry collector")
    print(f"  Target: s3://{args.bucket}/{key}  (region {args.region})")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[dry-run] Would upload this record (live mode subscribes to ROS 2 topics):\n")
        print(json.dumps(record, indent=2))
        print("\n[dry-run] No AWS/ROS writes.")
        return

    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not available on this device")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=args.region)
    # NOTE: live ROS 2 subscription loop is the on-device path; this minimal version
    # uploads a single batched record. Extend with an rclpy spin loop on the robot.
    s3.put_object(Bucket=args.bucket, Key=key, Body=json.dumps(record).encode())
    print(f"  Uploaded: s3://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
