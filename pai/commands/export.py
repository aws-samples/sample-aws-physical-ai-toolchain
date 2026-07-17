"""pai export — Export trained policies using Isaac Lab native exporter."""

import click

from pai import helpers


@click.command()
@click.option("--checkpoint", required=True, help="Path to rsl_rl checkpoint (model_*.pt)")
@click.option("--output-dir", required=True, help="Output directory for policy.pt / policy.onnx")
@click.option("--task", required=True, help="Isaac Lab task name (e.g., Isaac-Velocity-Flat-Anymal-D-v0)")
@click.option("--trtexec-smoke-check", is_flag=True, help="Run trtexec workstation smoke check (NOT for Jetson deploy)")
@click.option("--fp16", is_flag=True, default=True, help="Use FP16 precision for TRT smoke check")
@click.option("--dry-run", is_flag=True, help="Print export plan without executing")
def export(checkpoint, output_dir, task, trtexec_smoke_check, fp16, dry_run):
    """Export trained policy using Isaac Lab native exporter + trtexec.

    Exports policy.pt (TorchScript) and policy.onnx from an rsl_rl checkpoint.
    Optionally runs trtexec as a workstation smoke check (NOT the Jetson artifact).

    TensorRT engines are hardware-specific. Ship the .onnx and compile on the target device.
    """
    helpers.heading("Policy Export Pipeline (Native Exporter)")
    helpers.info(f"Checkpoint:  {checkpoint}")
    helpers.info(f"Task:        {task}")
    helpers.info(f"Output dir:  {output_dir}")
    if trtexec_smoke_check:
        helpers.info(f"TRT smoke:   Enabled (workstation GPU, NOT for Jetson)")
    else:
        helpers.info(f"TRT smoke:   Skipped (use --trtexec-smoke-check to enable)")

    if dry_run:
        helpers.info("\n[DRY RUN] Would export:")
        helpers.info(f"  {output_dir}/policy.pt")
        helpers.info(f"  {output_dir}/policy.onnx")
        if trtexec_smoke_check:
            helpers.info(f"  {output_dir}/policy.trt (workstation smoke check)")
        helpers.success("Dry run complete (no files written, no AWS calls)")
        return

    # Lazy-import (pulls torch + Isaac Lab)
    try:
        from training.scripts import export as export_module

        # Step 1: Export with native exporter
        helpers.info("\n[1/2] Exporting with Isaac Lab native exporter...")
        jit_path, onnx_path = export_module.export_with_native_exporter(
            checkpoint, output_dir, task
        )

        # Step 2: Optional trtexec smoke check
        if trtexec_smoke_check:
            helpers.info("\n[2/2] Running trtexec smoke check...")
            trt_path = export_module.trtexec_smoke_check(onnx_path, output_dir, fp16)
        else:
            helpers.info("\n[2/2] Skipping trtexec smoke check")
            trt_path = ""

        helpers.success(f"\n{'='*60}")
        helpers.success("Export complete!")
        helpers.info(f"TorchScript: {jit_path}")
        helpers.info(f"ONNX:        {onnx_path}")
        if trt_path:
            helpers.info(f"TRT (smoke): {trt_path}")
        helpers.info("\nDeploy the .onnx to Jetson and compile on-device:")
        helpers.info("  trtexec --onnx=policy.onnx --saveEngine=policy.trt --fp16")
        helpers.success(f"{'='*60}")
    except Exception as e:
        helpers.error(f"Export failed: {e}")
        raise


def register(cli: click.Group):
    """Register export command with the CLI."""
    cli.add_command(export)
