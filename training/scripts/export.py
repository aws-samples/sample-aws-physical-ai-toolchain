"""
Policy Export: PyTorch → ONNX → TensorRT (using Isaac Lab native exporter + trtexec)

Converts a trained RL policy to optimized formats for edge deployment.

Pipeline:
  1. Export policy.pt + policy.onnx via Isaac Lab's stock play.py (bakes in obs normalizer)
  2. Optionally build TensorRT engine via `trtexec` (workstation smoke check ONLY)

IMPORTANT: TensorRT engines are NOT portable across GPU architectures or TRT versions.
The engine built on an L40S workstation CANNOT be deployed to a Jetson Orin. Ship the
.onnx file and compile on the target device.

Usage:
    # Export policy.pt + policy.onnx from checkpoint (recommended path):
    python export.py --checkpoint logs/rsl_rl/.../model_1000.pt \
                     --output-dir models/ \
                     --task Isaac-Velocity-Flat-Anymal-D-v0

    # Writes models/policy.{pt,onnx} via Isaac Lab's native exporter (with normalizer)

    # Build TRT engine on TARGET device (Jetson Orin):
    trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16

    # Workstation smoke check (NOT deployable to Jetson):
    python export.py --checkpoint logs/.../model_1000.pt \
                     --output-dir models/ \
                     --task Isaac-Velocity-Flat-Anymal-D-v0 \
                     --trtexec-smoke-check

NOTE: This uses Isaac Lab's stock play.py under the hood, which correctly composes
the observation normalizer with the policy network. Attempting to reconstruct the
normalizer from a bare checkpoint is NOT possible (it lives on the live runner object),
which is why play.py is the canonical export path.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def export_with_native_exporter(checkpoint_path: str, output_dir: str, task_name: str) -> tuple:
    """Export policy.pt and policy.onnx using Isaac Lab's stock play.py.

    This is a thin subprocess wrapper around NVIDIA's documented export path.
    play.py auto-exports to <checkpoint_dir>/exported/ with the observation
    normalizer correctly composed.

    Why subprocess instead of in-process:
    - The normalizer only exists on the live runner object (not in the checkpoint)
    - Reconstructing the runner requires building the full env + agent cfg (GPU)
    - play.py already does this correctly, is maintained by NVIDIA, and is the
      documented path - no reason to reimplement it

    Returns:
        (policy_jit_path, policy_onnx_path)
    """
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Task:       {task_name}")

    # Resolve checkpoint to absolute path and find its parent dir
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_dir = ckpt_path.parent

    # Isaac Lab's play.py writes exports to <checkpoint_dir>/exported/
    exported_dir = ckpt_dir / "exported"

    print(f"  Running Isaac Lab's play.py to export with normalizer...")
    print(f"  (This requires building the env — GPU required)")

    # Find Isaac Lab installation (assume it's in /workspace/isaaclab or parent dir structure)
    # First check common container paths, then try to find isaaclab.sh in parent dirs
    isaaclab_root = None
    for candidate in [
        Path("/workspace/isaaclab"),
        Path.cwd() / "IsaacLab",
        Path.cwd().parent / "IsaacLab",
    ]:
        if (candidate / "isaaclab.sh").exists():
            isaaclab_root = candidate
            break

    if isaaclab_root is None:
        raise RuntimeError(
            "Isaac Lab installation not found. Expected isaaclab.sh at:\n"
            "  /workspace/isaaclab (container)\n"
            "  ./IsaacLab (local)\n"
            "  ../IsaacLab (local)\n"
            "Set ISAACLAB_PATH environment variable if installed elsewhere."
        )

    # Shell out to play.py (headless, exports to <ckpt_dir>/exported/)
    cmd = [
        str(isaaclab_root / "isaaclab.sh"),
        "-p",
        "scripts/reinforcement_learning/rsl_rl/play.py",
        f"--task={task_name}",
        f"--checkpoint={ckpt_path}",
        "--num_envs=1",
        "--headless",
    ]

    print(f"  Running: {' '.join(cmd)}")

    jit_path = exported_dir / "policy.pt"
    onnx_path = exported_dir / "policy.onnx"

    # KNOWN QUIRK: Isaac Sim frequently hangs inside simulation_app.close() AFTER the
    # export is already written to disk (validated on L40S, 2026-06-30). So the exported
    # files — not the subprocess exit code — are the real completion signal: a timeout or
    # non-zero exit is only a true failure if the files are absent.
    timed_out = False
    rc = None
    stderr_tail = ""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # generous; play.py exports early then may hang on shutdown
            cwd=str(isaaclab_root),
            env={**os.environ, "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"},
        )
        rc = result.returncode
        stderr_tail = (result.stderr or "")[-1000:]
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stderr_tail = (e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))[-1000:]

    # Decide success by artifact presence, not exit code.
    if not jit_path.exists() or not onnx_path.exists():
        if timed_out:
            raise RuntimeError(
                "play.py timed out (15 min) AND no exported files were written — real failure.\n"
                f"  Expected: {jit_path}\n            {onnx_path}\n"
                f"  stderr (last 1000 chars): {stderr_tail}"
            )
        raise RuntimeError(
            f"Export incomplete. Expected files not found:\n"
            f"  {jit_path}\n  {onnx_path}\n"
            f"  play.py exit code: {rc}\n"
            f"  stderr (last 1000 chars): {stderr_tail}"
        )

    if timed_out:
        print("  ⚠️  play.py hung on shutdown AFTER export (known Isaac Sim quirk) — "
              "exported files are present, treating as success.")
    elif rc not in (0, None):
        print(f"  ⚠️  play.py exited {rc} but exported files are present — treating as success.")

    # Copy to requested output dir
    os.makedirs(output_dir, exist_ok=True)
    dest_jit = Path(output_dir) / "policy.pt"
    dest_onnx = Path(output_dir) / "policy.onnx"

    shutil.copy2(jit_path, dest_jit)
    shutil.copy2(onnx_path, dest_onnx)

    print(f"  Native export successful (with observation normalizer).")
    print(f"    JIT:  {dest_jit} ({dest_jit.stat().st_size / 1024:.1f} KB)")
    print(f"    ONNX: {dest_onnx} ({dest_onnx.stat().st_size / 1024:.1f} KB)")

    return str(dest_jit), str(dest_onnx)


def trtexec_smoke_check(onnx_path: str, output_dir: str, fp16: bool = True) -> str:
    """Build a TensorRT engine via trtexec (WORKSTATION SMOKE CHECK ONLY).

    This is NOT the deployable artifact for a Jetson Orin. TRT engines are
    hardware-specific and must be compiled on the target device.

    NOTE: the `trtexec` binary is NOT present in the isaac-lab container (only the
    TensorRT python bindings are). Validated on L40S 2026-06-30: this check skips
    cleanly there. trtexec ships with the inference/Jetson image — run the on-device
    build (the deployable path) instead of relying on this workstation smoke check.

    Returns:
        Path to the built engine, or empty string if trtexec unavailable.
    """
    print(f"\n  ⚠️  WORKSTATION SMOKE CHECK — building TRT engine on THIS GPU")
    print(f"  ⚠️  This engine is NOT portable to Jetson Orin (different arch/TRT version)")
    print(f"  ⚠️  Ship the .onnx and compile on-device with:")
    print(f"      trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16")

    trt_path = os.path.join(output_dir, "policy.trt")

    # Check trtexec availability
    try:
        result = subprocess.run(
            ["trtexec", "--help"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            print("  trtexec not found — skipping TRT smoke check")
            return ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  trtexec not found — skipping TRT smoke check")
        return ""

    # Build engine
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"  Running: {' '.join(cmd)}")
    print(f"  (This may take a few minutes...)")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        if result.returncode != 0:
            print(f"  trtexec failed with exit code {result.returncode}")
            print(f"  stderr: {result.stderr[-500:]}")  # last 500 chars
            return ""

        if not os.path.exists(trt_path):
            print(f"  trtexec completed but {trt_path} not found")
            return ""

        print(f"  Workstation TRT engine: {trt_path} ({os.path.getsize(trt_path) / 1024 / 1024:.1f} MB)")
        print(f"  ✅ Smoke check PASS — ONNX parses and builds on this GPU")
        print(f"  ⚠️  Remember: compile on the Jetson Orin before deploying")

        return trt_path

    except subprocess.TimeoutExpired:
        print("  trtexec timed out after 10 minutes")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description='Export trained policy using Isaac Lab native exporter + trtexec',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to rsl_rl checkpoint (model_*.pt)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for policy.pt / policy.onnx')
    parser.add_argument('--task', type=str, required=True,
                        help='Isaac Lab task name (e.g., Isaac-Velocity-Flat-Anymal-D-v0)')
    parser.add_argument('--trtexec-smoke-check', action='store_true',
                        help='Run trtexec to build a workstation TRT engine (smoke check ONLY)')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True,
                        help='Use FP16 precision for the TRT smoke check (default: on; --no-fp16 to disable)')
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  Policy Export Pipeline (Isaac Lab Native Exporter)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Task:       {args.task}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'='*70}")

    # Step 1: Export with Isaac Lab's native exporter
    print(f"\n[1/2] Exporting with Isaac Lab native exporter...")
    jit_path, onnx_path = export_with_native_exporter(
        args.checkpoint,
        args.output_dir,
        args.task,
    )

    # Step 2: Optional workstation TRT smoke check
    if args.trtexec_smoke_check:
        print(f"\n[2/2] Running trtexec smoke check...")
        trt_path = trtexec_smoke_check(onnx_path, args.output_dir, args.fp16)
    else:
        print(f"\n[2/2] Skipping trtexec smoke check (use --trtexec-smoke-check to enable)")
        trt_path = ""

    print(f"\n{'='*70}")
    print(f"  Export complete!")
    print(f"  TorchScript: {jit_path}")
    print(f"  ONNX:        {onnx_path}")
    if trt_path:
        print(f"  TRT (smoke): {trt_path}")
    print(f"\n  Deploy the .onnx to Jetson and compile on-device:")
    print(f"    trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
