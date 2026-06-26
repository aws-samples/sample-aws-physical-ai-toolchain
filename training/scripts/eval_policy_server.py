"""Closed-loop policy server: loads TorchScript checkpoint, serves actions via ZMQ.

Runs on the Lab 2 workstation (g6e.4xlarge L40S). Loops forever: recv obs → forward
policy → send action. Expects a TorchScript checkpoint (scriptified via export.py);
loudly rejects raw rsl_rl .pt files (tell user to scriptify first).

SECURITY: ZMQ has no authentication. The default bind address is localhost only.
If you bind to a non-localhost address, ensure the policy server is behind a firewall
or accessed via SSH tunnel. The checkpoint must come from a trusted source (torch.jit.load
can execute code from a malicious .pt).

Usage:
    python eval_policy_server.py --checkpoint s3://bucket/model_scripted.pt
    python eval_policy_server.py --checkpoint ./model_scripted.pt --device cuda
    python eval_policy_server.py --checkpoint ./model.pt --endpoint tcp://127.0.0.1:5555
"""
import argparse
import atexit
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def resolve_checkpoint(checkpoint: str) -> str:
    """Download from S3 if s3://, else return local path. Mirrors train.py.

    Uses a unique per-process temp file to avoid TOCTOU/multi-user clobber.
    """
    if checkpoint.startswith("s3://"):
        import boto3

        s3_path = checkpoint[5:]  # strip s3://
        bucket, key = s3_path.split("/", 1)
        # Unique per-process temp file (avoid clobber)
        local_path = Path(tempfile.gettempdir()) / f"eval_checkpoint_{os.getpid()}.pt"
        print(f"  Downloading s3://{bucket}/{key} → {local_path}")
        boto3.client("s3").download_file(bucket, key, str(local_path))

        # Clean up on exit
        def cleanup():
            if local_path.exists():
                local_path.unlink()
        atexit.register(cleanup)

        return str(local_path)
    return checkpoint


def load_policy(checkpoint_path: str, device: str) -> torch.jit.ScriptModule:
    """Load TorchScript policy. Fails loudly if given a raw .pt.

    WARNING: torch.jit.load can execute code from the checkpoint. Only load checkpoints
    from trusted sources (your own training output in a private S3 bucket).
    """
    print(f"  Loading policy from {checkpoint_path}")
    print("  ⚠️  Ensure checkpoint is from a trusted/private S3 bucket (torch.jit.load can execute code)")
    try:
        policy = torch.jit.load(checkpoint_path, map_location=device)
    except RuntimeError as e:
        # torch.jit.load fails on raw rsl_rl checkpoints (state dict, not ScriptModule)
        if "PytorchStreamReader" in str(e) or "is not a zip" in str(e):
            print(
                "\n❌ ERROR: Checkpoint is NOT a TorchScript model.\n"
                "You provided a raw .pt checkpoint (likely from rl_games).\n"
                "You must scriptify it first:\n"
                "  python training/scripts/export.py --checkpoint <your-checkpoint.pt> \\\n"
                "    --output-onnx model.onnx --output-trt model.trt\n"
                "This creates a TorchScript model you can load here.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    policy.eval()
    print(f"  Policy loaded on {device}")
    return policy


def serve(checkpoint, endpoint, device):
    """Run policy server: loads TorchScript checkpoint, serves actions via ZMQ.

    Args:
        checkpoint: Path or s3:// URI to TorchScript checkpoint
        endpoint: ZMQ bind endpoint (default: tcp://127.0.0.1:5555)
        device: Inference device (cuda/cpu)
    """
    # Security warning if binding to non-localhost
    bind_host = endpoint.split("://")[1].split(":")[0] if "://" in endpoint else "127.0.0.1"
    if bind_host not in ("127.0.0.1", "localhost"):
        print("\n⚠️  WARNING: Policy server is binding to a non-localhost address.")
        print("   ZMQ has NO authentication. Ensure this server is behind a firewall")
        print("   or accessed via SSH tunnel. Anyone with network access can query your policy.\n")

    # Resolve checkpoint (download if S3)
    ckpt_path = resolve_checkpoint(checkpoint)

    # Load TorchScript policy
    policy = load_policy(ckpt_path, device)
    device_obj = torch.device(device)

    # Start ZMQ server (lazy-import pyzmq)
    try:
        import zmq
    except ImportError:
        print("❌ ERROR: pyzmq not installed. Run: pip install pyzmq>=25", file=sys.stderr)
        sys.exit(1)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(endpoint)
    print(f"  Policy server listening on {endpoint}")
    print("  Waiting for observations...\n")

    # Import protocol after zmq succeeds (so help text doesn't require pyzmq)
    from eval_protocol import decode_obs, encode_action

    try:
        while True:
            # Recv observation
            obs_bytes = socket.recv()
            obs = decode_obs(obs_bytes)

            # Forward through policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device_obj)
                action_tensor = policy(obs_tensor).squeeze(0).cpu().numpy()

            # If policy outputs a horizon (multiple steps), take only the first action
            # (v1 single-step only; N-step horizon is a forward-compat seam, not implemented)
            if action_tensor.ndim > 1:
                action_tensor = action_tensor[0]

            # Send action
            action_bytes = encode_action(action_tensor)
            socket.send(action_bytes)

    except KeyboardInterrupt:
        print("\n  Shutting down policy server.")
    finally:
        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description="Closed-loop policy server (ZMQ REP)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path or s3:// URI to TorchScript checkpoint")
    parser.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5555", help="ZMQ bind endpoint (default: localhost only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    args = parser.parse_args()

    serve(args.checkpoint, args.endpoint, args.device)


if __name__ == "__main__":
    main()
