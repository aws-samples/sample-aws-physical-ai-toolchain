"""
Cosmos Transfer 2.5 runtime — EC2 Spot p5 + NIM /v1/infer (the validated path).

Cosmos Transfer 2.5 restyles a sim-rendered VIDEO into a photorealistic one,
guided by a prompt + a control modality (edge/depth/seg/vis). It runs as an NVIDIA
NIM container serving `POST /v1/infer` on port 8000.

It does NOT run on a SageMaker real-time endpoint: SageMaker's managed GPUs ship
NVIDIA driver 470, but Cosmos needs 580+. So this script drives an **EC2 Spot
p5.48xlarge** (8x H100) bootstrapped by `scripts/cosmos-userdata.sh`. Full runbook:
docs/cosmos-deployment-guide.md.

Account/region/image/role are resolved from the caller — nothing hardcoded.

Usage:
    # Launch the Cosmos GPU instance (Spot p5) — preview first:
    python cosmos_setup.py launch --dry-run
    python cosmos_setup.py launch

    # Check the NIM health endpoint (via SSM, no inbound port needed):
    python cosmos_setup.py status --instance-id i-xxxx

    # Restyle sim videos (POST /v1/infer). Input MP4s must be 93-480 frames:
    python cosmos_setup.py generate --instance-id i-xxxx \
        --input ./sim_videos/ --output ./cosmos_out/ --dry-run

    # Terminate when done (Spot p5 is expensive):
    python cosmos_setup.py terminate --instance-id i-xxxx
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")
INSTANCE_TYPE = os.environ.get("COSMOS_INSTANCE_TYPE", "p5.48xlarge")  # 8x H100
PORT = 8000

# Default Cosmos control modality + inference params (see the deployment runbook).
DEFAULT_CONTROL = "edge"
DEFAULT_NUM_STEPS = 35
DEFAULT_GUIDANCE = 3
DEFAULT_RESOLUTION = "480"


def _account() -> str:
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def _userdata() -> str:
    """The validated cosmos-userdata.sh bootstrap (read from the repo).

    cosmos_setup.py is at <repo>/training/scripts/, so the repo root is parents[2]
    and the bootstrap lives at <repo>/scripts/cosmos-userdata.sh.
    """
    p = Path(__file__).resolve().parents[2] / "scripts" / "cosmos-userdata.sh"
    return p.read_text()


def _ssm_run(instance_id: str, command: str, timeout: int = 60) -> str:
    """Run a shell command on the instance via SSM and return stdout."""
    ssm = boto3.client("ssm", region_name=REGION)
    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [command]},
        TimeoutSeconds=timeout,
    )
    cmd_id = resp["Command"]["CommandId"]
    waiter = ssm.get_waiter("command_executed")
    try:
        waiter.wait(CommandId=cmd_id, InstanceId=instance_id,
                    WaiterConfig={"Delay": 5, "MaxAttempts": max(2, timeout // 5)})
    except Exception:
        pass
    out = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
    return out.get("StandardOutputContent", "") + out.get("StandardErrorContent", "")


def launch(dry_run: bool):
    """Launch a Spot p5 that boots the Cosmos NIM (port 8000 /v1/infer)."""
    ec2 = boto3.client("ec2", region_name=REGION)
    ami = boto3.client("ssm", region_name=REGION).get_parameter(
        Name="/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
    )["Parameter"]["Value"] if not dry_run else "<ubuntu-22.04-ami>"

    spec = {
        "ImageId": ami,
        "InstanceType": INSTANCE_TYPE,
        "InstanceMarketOptions": {"MarketType": "spot"},
        "UserData": _userdata(),
        "IamInstanceProfile": {"Name": os.environ.get(
            "COSMOS_INSTANCE_PROFILE", f"{PROJECT_NAME}-{ENVIRONMENT}-cosmos-profile")},
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1",
                                 "Ebs": {"VolumeSize": 500, "VolumeType": "gp3"}}],
        "TagSpecifications": [{"ResourceType": "instance",
                               "Tags": [{"Key": "Name", "Value": f"{PROJECT_NAME}-cosmos-transfer"}]}],
    }

    print(f"{'='*60}")
    print(f"  Launch Cosmos Transfer 2.5 (EC2 Spot {INSTANCE_TYPE}, 8x H100)")
    print(f"  Region:  {REGION}")
    print(f"  Boots:   scripts/cosmos-userdata.sh → NIM on :{PORT} (/v1/infer)")
    print(f"  Note:    p5 Spot capacity + P5 quota required; ~$7-8/hr Spot.")
    print(f"{'='*60}")

    if dry_run:
        print("[dry-run] Would ec2.run_instances with (UserData elided):")
        print(json.dumps({k: v for k, v in spec.items() if k != "UserData"}, indent=2))
        print("[dry-run] No AWS calls made.")
        return

    resp = ec2.run_instances(MinCount=1, MaxCount=1, **spec)
    iid = resp["Instances"][0]["InstanceId"]
    print(f"  Launched: {iid}")
    print(f"  Bootstrap takes ~10-15 min (driver + image pull + model load).")
    print(f"  Check: python {sys.argv[0]} status --instance-id {iid}")


def status(instance_id: str):
    print(f"  Checking Cosmos NIM health on {instance_id} (via SSM)...")
    out = _ssm_run(instance_id, f"curl -s http://localhost:{PORT}/v1/health/ready || echo NOT_READY")
    print(f"  Health: {out.strip() or '(no response — still booting?)'}")


def generate(instance_id: str, input_dir: str, output_dir: str, control: str,
             num_steps: int, guidance: int, resolution: str, dry_run: bool):
    """Restyle each input MP4 via the NIM /v1/infer video API (run on the box via SSM)."""
    in_path, out_path = Path(input_dir), Path(output_dir)
    videos = sorted(list(in_path.glob("*.mp4")))

    print(f"{'='*60}")
    print(f"  Cosmos Transfer 2.5 — restyle videos")
    print(f"  Instance: {instance_id}  control={control}  steps={num_steps}")
    print(f"  Input:    {input_dir} ({len(videos)} mp4s)  →  Output: {output_dir}")
    print(f"  API:      POST http://localhost:{PORT}/v1/infer (93-480 frames per clip)")
    print(f"{'='*60}")

    if not videos and not dry_run:
        print("  No .mp4 files found. Render sim clips first (MP4, 93-480 frames).")
        return

    example_payload = {
        "prompt": "<style prompt, e.g. 'industrial warehouse, fluorescent lighting'>",
        "video": "<base64 MP4, 93-480 frames>",
        control: {"enabled": True},
        "num_steps": num_steps,
        "guidance": guidance,
        "resolution": resolution,
    }

    if dry_run:
        print("[dry-run] For each input MP4, would POST to /v1/infer on the instance:\n")
        print(json.dumps(example_payload, indent=2))
        print(f"\n[dry-run] {len(videos)} clip(s) would be processed. No AWS/HTTP calls made.")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    for v in videos:
        b64 = base64.b64encode(v.read_bytes()).decode()
        payload = dict(example_payload, video=b64,
                       prompt=os.environ.get("COSMOS_PROMPT",
                                             "industrial warehouse with fluorescent lighting"))
        # Run the request on the instance (NIM is bound to localhost there).
        remote = (
            f"python3 - <<'PY'\nimport json,urllib.request,base64\n"
            f"req=urllib.request.Request('http://localhost:{PORT}/v1/infer',"
            f"data=json.dumps({json.dumps(payload)}).encode(),"
            f"headers={{'Content-Type':'application/json'}})\n"
            f"r=urllib.request.urlopen(req,timeout=1200).read()\n"
            f"open('/tmp/out.mp4','wb').write(base64.b64decode(json.loads(r)['video']))\n"
            f"print('ok')\nPY"
        )
        print(f"  Restyling {v.name} (this can take minutes on CP=8)...")
        result = _ssm_run(instance_id, remote, timeout=1800)
        print(f"    {result.strip()[-200:]}")
    print(f"\n  Done. Pull results off the instance, then upload for RL training.")


def terminate(instance_id: str):
    ec2 = boto3.client("ec2", region_name=REGION)
    ec2.terminate_instances(InstanceIds=[instance_id])
    print(f"  Terminating {instance_id}. No more charges once shut down.")


def main():
    p = argparse.ArgumentParser(description="Cosmos Transfer 2.5 runtime (EC2 p5 + NIM /v1/infer)")
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("launch").add_argument("--dry-run", action="store_true")

    ps = sub.add_parser("status"); ps.add_argument("--instance-id", required=True)
    pt = sub.add_parser("terminate"); pt.add_argument("--instance-id", required=True)

    pg = sub.add_parser("generate")
    pg.add_argument("--instance-id", required=True)
    pg.add_argument("--input", default="./sim_videos/")
    pg.add_argument("--output", default="./cosmos_out/")
    pg.add_argument("--control", default=DEFAULT_CONTROL, choices=["edge", "depth", "seg", "vis"])
    pg.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    pg.add_argument("--guidance", type=int, default=DEFAULT_GUIDANCE)
    pg.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    pg.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    if args.action == "launch":
        launch(args.dry_run)
    elif args.action == "status":
        status(args.instance_id)
    elif args.action == "terminate":
        terminate(args.instance_id)
    elif args.action == "generate":
        generate(args.instance_id, args.input, args.output, args.control,
                 args.num_steps, args.guidance, args.resolution, args.dry_run)


if __name__ == "__main__":
    main()
