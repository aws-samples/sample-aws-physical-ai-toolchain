#!/usr/bin/env python3
"""
test_cosmos_e2e.py — Cosmos Transfer 2.5 end-to-end test

Tests the full pipeline:
  1. Launch p5.48xlarge Spot (us-east-2a — cheapest)
  2. Wait for NIM health endpoint
  3. Create synthetic 100-frame test MP4 (no Isaac Lab needed)
  4. POST to /v1/infer with num_steps=5, resolution="256" (fast test)
  5. Download and verify the result MP4
  6. Terminate the instance
  7. Log all timings

Usage:
    python test_cosmos_e2e.py # full run
    python test_cosmos_e2e.py --dry-run # preview config only
    python test_cosmos_e2e.py --skip-launch i-xxxx # reuse existing instance

Account-specific facts discovered 2026-07-02:
  - NGC secret name: ngc-api-key (NOT physical-ai/ngc-api-key)
  - IAM profile: physical-ai-dev-cosmos-profile (freshly created)
  - ECR cosmos image: does not exist → pulling from NGC nvcr.io directly
  - Best Spot AZ: us-east-2a (~$13.58/hr)
"""

import argparse
import base64
import json
import os
import struct
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import boto3

# ── Config ────────────────────────────────────────────────────────────────────
# Use us-east-2 — cheapest Spot price for p5 and confirmed price history.
REGION = "us-east-2"
ACCOUNT = "YOUR_ACCOUNT_ID"
INSTANCE_TYPE = "p5.48xlarge"
SPOT_AZ = "us-east-2a" # cheapest AZ per price history
INSTANCE_PROFILE = "physical-ai-dev-cosmos-profile"
NGC_SECRET_NAME = "ngc-api-key" # actual name in Secrets Manager
NIM_PORT = 8000
BOOTSTRAP_TIMEOUT_MIN = 35 # DLAMI: NGC pull(15m) + model load(10m) + buffer
INFER_TIMEOUT_SEC = 600 # 10 min for fast test (5 steps)

# Test inference settings — fast
TEST_NUM_STEPS = 5
TEST_RESOLUTION = "256"
TEST_FRAMES = 100 # 100 frames at 5 fps = 20s clip
TEST_PROMPT = "industrial warehouse with fluorescent lighting"

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2] # notebooks/lab3/ -> repo root
USERDATA_PATH = REPO_ROOT / "scripts" / "cosmos-userdata.sh"
LOG_PATH = Path(__file__).parent / "test_cosmos_e2e.log"
OUTPUT_DIR = Path("/tmp/cosmos_test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
timings: dict = {}
log_lines: list = []

def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {level:5s} {msg}"
    print(line, flush=True)
    log_lines.append(line)

def t_start(label: str):
    timings[label] = {"start": time.time()}
    log(f"START {label}")

def t_end(label: str) -> float:
    elapsed = time.time() - timings[label]["start"]
    timings[label]["elapsed"] = elapsed
    log(f"END {label} — {elapsed:.1f}s")
    return elapsed


# ── Synthetic test video ──────────────────────────────────────────────────────
def make_test_mp4(path: Path, n_frames: int = 100, w: int = 256, h: int = 256) -> Path:
    """
    Create a synthetic MP4 with n_frames using OpenCV if available,
    otherwise fall back to a minimal valid h264 MP4 via ffmpeg.
    The video shows a moving rectangle on a gradient — simple but has motion
    which exercises Cosmos's temporal consistency.
    """
    try:
        import cv2
        import numpy as np

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 5, (w, h))
        if not writer.isOpened():
            raise RuntimeError("VideoWriter failed to open")

        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Gradient background (simulates lighting change)
            frame[:, :, 0] = int(40 + i * 0.4) # R channel ramps up
            frame[:h // 2, :, 2] = 80 # Blue ceiling
            frame[h // 2:, :, 1] = 40 # Green floor
            # Moving robot arm proxy — vertical bar moving horizontally
            x = int((w - 40) * (i / n_frames))
            cv2.rectangle(frame, (x, 20), (x + 30, h - 20), (200, 200, 200), -1)
            # Object on floor
            cv2.rectangle(frame, (w // 2 - 15, h - 50), (w // 2 + 15, h - 20), (100, 150, 200), -1)
            writer.write(frame)

        writer.release()
        log(f"Created test MP4 via OpenCV: {path} ({n_frames} frames, {path.stat().st_size // 1024}KB)")
        return path

    except ImportError:
        log("OpenCV not available, falling back to ffmpeg", "WARN")
        return _make_test_mp4_ffmpeg(path, n_frames, w, h)


def _make_test_mp4_ffmpeg(path: Path, n_frames: int, w: int, h: int) -> Path:
    """Fallback: generate test MP4 via ffmpeg lavfi."""
    import subprocess
    duration = n_frames / 5 # 5 fps
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size={w}x{h}:rate=5",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True) # nosemgrep: dangerous-subprocess-use-audit
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")
    log(f"Created test MP4 via ffmpeg: {path} ({path.stat().st_size // 1024}KB)")
    return path


# ── AWS helpers ───────────────────────────────────────────────────────────────
def get_dlami(region: str) -> str:
    """
    Resolve the latest Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04).
    This AMI ships with NVIDIA driver, CUDA, Docker, nvidia-container-toolkit,
    and nvidia-fabricmanager pre-installed. Eliminates the DKMS driver build
    that consistently fails on bare Ubuntu with AWS custom kernels.
    Owner: 898082745236 (AWS)
    """
    ec2 = boto3.client("ec2", region_name=region)
    resp = ec2.describe_images(
        Owners=["898082745236"],
        Filters=[
            {"Name": "name", "Values": ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ]
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(f"No DLAMI found in {region}")
    ami = images[0]["ImageId"]
    log(f"DLAMI in {region}: {ami} ({images[0]['Name'][:60]})")
    return ami


def launch_spot_p5(dry_run: bool = False) -> str:
    """Launch p5.48xlarge Spot in SPOT_AZ. Returns instance ID."""
    ami = "<dry-run-ami>" if dry_run else get_dlami(REGION)

    userdata = USERDATA_PATH.read_text()

    spec = {
        "ImageId": ami,
        "InstanceType": INSTANCE_TYPE,
        "Placement": {"AvailabilityZone": SPOT_AZ},
        "InstanceMarketOptions": {"MarketType": "spot"},
        "UserData": userdata,
        "IamInstanceProfile": {"Name": INSTANCE_PROFILE},
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 500, "VolumeType": "gp3", "DeleteOnTermination": True}
        }],
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": "physical-ai-cosmos-transfer-test"},
                {"Key": "Project", "Value": "physical-ai"},
                {"Key": "AutoTerminate", "Value": "true"},
            ]
        }],
    }

    log(f"Launch spec (UserData omitted): {json.dumps({k: v for k, v in spec.items() if k != 'UserData'}, indent=2)}")

    if dry_run:
        log("[dry-run] Would launch instance. No AWS calls made.")
        return "i-dryrun0000000000"

    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        resp = ec2.run_instances(MinCount=1, MaxCount=1, **spec)
    except ec2.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("InsufficientInstanceCapacity", "SpotMaxPriceTooLow"):
            log(f"Spot capacity unavailable in {SPOT_AZ}: {e}", "WARN")
            log("Trying all AZs in us-east-2...", "WARN")
            # Retry without AZ constraint to let AWS pick
            spec.pop("Placement", None)
            resp = ec2.run_instances(MinCount=1, MaxCount=1, **spec)
        else:
            raise

    iid = resp["Instances"][0]["InstanceId"]
    log(f"Launched: {iid} (AZ: {resp['Instances'][0].get('Placement', {}).get('AvailabilityZone', 'unknown')})")
    return iid


def wait_for_instance_running(instance_id: str, timeout: int = 300):
    """Wait until instance is in 'running' state."""
    ec2 = boto3.client("ec2", region_name=REGION)
    log(f"Waiting for {instance_id} to reach 'running'...")
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(
        InstanceIds=[instance_id],
        WaiterConfig={"Delay": 10, "MaxAttempts": timeout // 10}
    )
    inst = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    az = inst["Placement"]["AvailabilityZone"]
    log(f"Instance running in {az}")


def ssm_run(instance_id: str, command: str, timeout: int = 60) -> tuple[str, str, int]:
    """
    Run a shell command via SSM. Returns (stdout, stderr, exit_code).
    Retries SSM registration check up to 5 times (SSM agent can take 1-2 min).
    """
    ssm = boto3.client("ssm", region_name=REGION)

    # Wait for SSM registration
    for attempt in range(30):
        try:
            info = ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
            )["InstanceInformationList"]
            if info:
                break
        except Exception:
            pass
        if attempt == 0:
            log("Waiting for SSM agent registration...")
        time.sleep(10) # nosemgrep: arbitrary-sleep
    else:
        raise RuntimeError(f"SSM agent not registered after 5 min on {instance_id}")

    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command]},
        TimeoutSeconds=timeout,
    )
    cmd_id = resp["Command"]["CommandId"]

    waiter = ssm.get_waiter("command_executed")
    try:
        waiter.wait(
            CommandId=cmd_id,
            InstanceId=instance_id,
            WaiterConfig={"Delay": 5, "MaxAttempts": max(4, timeout // 5)}
        )
    except Exception:
        pass # Will check status in get_command_invocation

    result = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
    stdout = result.get("StandardOutputContent", "")
    stderr = result.get("StandardErrorContent", "")
    # Status: Success=0, Failed=1
    status = result.get("ResponseCode", -1)
    return stdout, stderr, status


def wait_for_nim_ready(instance_id: str, timeout_min: int = BOOTSTRAP_TIMEOUT_MIN) -> bool:
    """
    Poll /v1/health/ready via SSM until NIM reports ready.
    Uses health endpoint only — more reliable than checking bootstrap log,
    which may not reflect manual fixes applied outside the bootstrap script.
    """
    log(f"Polling NIM health (timeout={timeout_min}min)...")
    deadline = time.time() + timeout_min * 60
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            stdout, _, rc = ssm_run(
                instance_id,
                f"curl -sf http://localhost:{NIM_PORT}/v1/health/ready 2>/dev/null || echo NOT_READY",
                timeout=30
            )
            health = stdout.strip()
            log(f" Health check [{attempt}]: {health[:100]}")

            if "ready" in health.lower() and "not_ready" not in health.lower():
                log(f"NIM is READY after {attempt} attempts")
                return True

            # Log docker status every 5 checks for visibility
            if attempt % 5 == 0:
                docker_out, _, _ = ssm_run(
                    instance_id,
                    "docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null; "
                    "docker logs cosmos 2>&1 | grep -iE 'error|ready|serving|profile|starting' | tail -5",
                    timeout=30
                )
                log(f" Docker status:\n{docker_out.strip()}")

        except Exception as e:
            log(f" Health poll error (attempt {attempt}): {e}", "WARN")

        time.sleep(45) # nosemgrep: arbitrary-sleep

    log(f"NIM health timeout after {timeout_min} min", "ERROR")
    return False


def run_inference(instance_id: str, video_path: Path) -> bytes:
    """
    Send video to /v1/infer via SSM Python heredoc.
    Returns raw response bytes (JSON with base64 video).
    """
    b64_video = base64.b64encode(video_path.read_bytes()).decode()

    payload = {
        "prompt": TEST_PROMPT,
        "video": b64_video,
        "edge": {},
        "num_steps": TEST_NUM_STEPS,
        "guidance": 3,
        "resolution": TEST_RESOLUTION,
        "seed": 42,
    }

    # Write payload to a temp file on the instance, then invoke
    payload_json = json.dumps(payload)

    # Use a heredoc to write payload and run inference on the instance
    remote_cmd = f"""python3 - <<'PYEOF'
import json, urllib.request, base64, sys, os

payload = json.loads(open('/tmp/cosmos_payload.json').read())
url = 'http://localhost:{NIM_PORT}/v1/infer'

print(f"Sending request: {{len(payload['video'])}} chars b64, steps={TEST_NUM_STEPS}, res={TEST_RESOLUTION}")
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={{'Content-Type': 'application/json'}}
)
try:
    resp = urllib.request.urlopen(req, timeout={INFER_TIMEOUT_SEC})
    body = resp.read()
    print(f"Response size: {{len(body)}} bytes")
    # Save response
    with open('/tmp/cosmos_response.json', 'wb') as f:
        f.write(body)
    # Extract video
    data = json.loads(body)
    # NIM returns 'b64_video' (Cosmos Transfer NIM API) not 'video'
    video_key = 'b64_video' if 'b64_video' in data else 'video'
    if video_key in data:
        video_bytes = base64.b64decode(data[video_key])
        with open('/tmp/cosmos_output.mp4', 'wb') as f:
            f.write(video_bytes)
        print(f"Output MP4: {{len(video_bytes)}} bytes")
        print("SUCCESS")
    else:
        print(f"ERROR: no video in response. Keys: {{list(data.keys())}}")
        print(f"Response snippet: {{str(data)[:500]}}")
        sys.exit(1)
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {{e.code}}: {{body[:500]}}")
    sys.exit(1)
except Exception as e:
    print(f"Request failed: {{e}}")
    sys.exit(1)
PYEOF"""

    # Step 1: write payload file
    log(f"Writing payload to instance ({len(payload_json) // 1024}KB)...")
    # Split into chunks if large (SSM has 16KB command limit for Parameters)
    # Write via base64 to avoid quote issues
    payload_b64 = base64.b64encode(payload_json.encode()).decode()

    write_cmd = f"echo '{payload_b64}' | base64 -d > /tmp/cosmos_payload.json && echo 'payload written'"
    stdout, stderr, rc = ssm_run(instance_id, write_cmd, timeout=60)
    if "payload written" not in stdout:
        raise RuntimeError(f"Failed to write payload: {stdout} {stderr}")
    log("Payload written to instance")

    # Step 2: run inference
    log(f"Running inference (num_steps={TEST_NUM_STEPS}, resolution={TEST_RESOLUTION}, timeout={INFER_TIMEOUT_SEC}s)...")
    stdout, stderr, rc = ssm_run(instance_id, remote_cmd, timeout=INFER_TIMEOUT_SEC + 60)

    log(f"Inference stdout:\n{stdout.strip()}")
    if stderr.strip():
        log(f"Inference stderr:\n{stderr.strip()[:500]}", "WARN")

    if rc != 0 or "SUCCESS" not in stdout:
        raise RuntimeError(f"Inference failed (rc={rc}):\n{stdout}\n{stderr}")

    # Step 3: retrieve output MP4
    log("Retrieving output MP4 from instance...")
    stdout, stderr, rc = ssm_run(
        instance_id,
        "base64 /tmp/cosmos_output.mp4 2>/dev/null || echo MISSING",
        timeout=120
    )
    if stdout.strip() == "MISSING" or not stdout.strip():
        raise RuntimeError("Output MP4 not found on instance")

    output_bytes = base64.b64decode(stdout.strip())
    log(f"Retrieved output MP4: {len(output_bytes) // 1024}KB")
    return output_bytes


def terminate_instance(instance_id: str, dry_run: bool = False):
    if dry_run or instance_id.startswith("i-dryrun"):
        log("[dry-run] Would terminate instance")
        return
    ec2 = boto3.client("ec2", region_name=REGION)
    ec2.terminate_instances(InstanceIds=[instance_id])
    log(f"Terminated {instance_id}")


# ── Lessons learned recorder ───────────────────────────────────────────────────
def append_run_results(instance_id: str, success: bool, notes: list[str]):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"\n---\n",
        f"## Test Run — {ts}\n\n",
        f"**Instance:** `{instance_id}` \n",
        f"**Region/AZ:** `{REGION}` / `{SPOT_AZ}` \n",
        f"**Result:** {' SUCCESS' if success else ' FAILED'} \n\n",
        "### Timings\n\n",
        "| Step | Duration |\n|------|----------|\n",
    ]
    for label, info in timings.items():
        elapsed = info.get("elapsed", time.time() - info["start"])
        lines.append(f"| {label} | {elapsed:.1f}s |\n")

    lines.append("\n### Notes\n\n")
    for n in notes:
        lines.append(f"- {n}\n")

    with open(LOG_PATH, "a") as f:
        f.writelines(lines)
    log(f"Appended results to {LOG_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cosmos Transfer e2e test")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no AWS calls")
    parser.add_argument("--skip-launch", metavar="INSTANCE_ID",
                        help="Skip launch, use existing instance ID")
    parser.add_argument("--no-terminate", action="store_true",
                        help="Don't terminate instance after test (for debugging)")
    args = parser.parse_args()

    log("=" * 60)
    log("Cosmos Transfer 2.5 — End-to-End Test")
    log(f"Region: {REGION} AZ: {SPOT_AZ} Instance: {INSTANCE_TYPE}")
    log(f"Test: {TEST_FRAMES} frames, {TEST_NUM_STEPS} steps, res={TEST_RESOLUTION}")
    log("=" * 60)

    notes = []
    instance_id = None
    success = False

    try:
        # ── Step 1: Launch ─────────────────────────────────────────────────
        if args.skip_launch:
            instance_id = args.skip_launch
            log(f"Reusing existing instance: {instance_id}")
            notes.append(f"Reused existing instance {instance_id}")
        else:
            t_start("Launch Spot p5")
            instance_id = launch_spot_p5(dry_run=args.dry_run)
            t_end("Launch Spot p5")
            notes.append(f"Launched {instance_id} in {SPOT_AZ}")

            if not args.dry_run:
                t_start("Wait instance running")
                wait_for_instance_running(instance_id)
                t_end("Wait instance running")

        if args.dry_run:
            log("[dry-run] Stopping here — no further AWS calls")
            notes.append("Dry run — no inference attempted")
            append_run_results(instance_id, True, notes)
            return

        # ── Step 2: Wait for NIM ───────────────────────────────────────────
        t_start("Bootstrap + NIM ready")
        nim_ready = wait_for_nim_ready(instance_id)
        t_end("Bootstrap + NIM ready")

        if not nim_ready:
            notes.append("NIM did not become ready — bootstrap failed or timed out")
            notes.append("Check /var/log/cosmos-bootstrap.log on instance")
            raise RuntimeError("NIM health check failed")

        notes.append("NIM health endpoint returned ready")

        # ── Step 3: Create test video ──────────────────────────────────────
        t_start("Create test MP4")
        test_video_path = OUTPUT_DIR / "test_input.mp4"
        make_test_mp4(test_video_path, n_frames=TEST_FRAMES)
        t_end("Create test MP4")
        notes.append(f"Test MP4: {TEST_FRAMES} frames, {test_video_path.stat().st_size // 1024}KB")

        # ── Step 4: Run inference ──────────────────────────────────────────
        t_start("Inference /v1/infer")
        output_bytes = run_inference(instance_id, test_video_path)
        elapsed = t_end("Inference /v1/infer")

        output_path = OUTPUT_DIR / "test_output.mp4"
        output_path.write_bytes(output_bytes)
        log(f"Output saved: {output_path} ({len(output_bytes) // 1024}KB)")
        notes.append(f"Inference: {elapsed:.0f}s, output {len(output_bytes) // 1024}KB")

        # ── Step 5: Upload to S3 for Lab 4 consumption ────────────────────
        try:
            sts = boto3.client("sts", region_name="us-west-2")
            account = sts.get_caller_identity()["Account"]
            # SageMaker default bucket is in the home region (us-west-2), not the Cosmos region
            home_region = "us-west-2"
            s3_bucket = os.environ.get("DATASETS_BUCKET", f"sagemaker-{home_region}-{account}")
            s3_key = f"cosmos-output/test/{output_path.name}"
            s3 = boto3.client("s3", region_name=home_region)
            s3.upload_file(str(output_path), s3_bucket, s3_key)
            s3_uri = f"s3://{s3_bucket}/{s3_key}"
            log(f"Uploaded to S3: {s3_uri}")
            notes.append(f"S3: {s3_uri}")
        except Exception as e:
            log(f"S3 upload failed (non-fatal): {e}", "WARN")
            notes.append(f"S3 upload skipped: {e}")

        success = True
        log("=" * 60)
        log("TEST PASSED")
        log("=" * 60)

    except KeyboardInterrupt:
        log("Interrupted by user", "WARN")
        notes.append("Interrupted by user")

    except Exception as e:
        log(f"TEST FAILED: {e}", "ERROR")
        traceback.print_exc()
        notes.append(f"Failed: {e}")

    finally:
        # ── Step 5: Terminate ──────────────────────────────────────────────
        if instance_id and not args.no_terminate and not args.dry_run:
            if not instance_id.startswith("i-dryrun"):
                t_start("Terminate instance")
                try:
                    terminate_instance(instance_id)
                    notes.append(f"Instance {instance_id} terminated")
                except Exception as e:
                    log(f"Terminate failed: {e}", "ERROR")
                    notes.append(f"WARNING: terminate failed — {e}. Terminate manually!")
                t_end("Terminate instance")
        elif args.no_terminate:
            log(f"--no-terminate set. Instance {instance_id} still running (~${14.86:.2f}/hr)", "WARN")
            notes.append(f"Instance {instance_id} NOT terminated (--no-terminate). Terminate manually.")

        # ── Record results ─────────────────────────────────────────────────
        append_run_results(instance_id or "unknown", success, notes)

        log("\n=== TIMING SUMMARY ===")
        for label, info in timings.items():
            elapsed = info.get("elapsed", time.time() - info["start"])
            log(f" {label:40s} {elapsed:6.1f}s")
        log("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
