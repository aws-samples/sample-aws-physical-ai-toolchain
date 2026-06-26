"""pai export — Export trained policies to ONNX and TensorRT."""

import click

from pai import helpers


@click.command()
@click.option("--checkpoint", required=True, help="Path to trained PyTorch checkpoint (.pt)")
@click.option("--output-onnx", required=True, help="Output path for ONNX model")
@click.option("--output-trt", required=True, help="Output path for TensorRT engine")
@click.option("--target-device", default="jetson-orin", type=click.Choice(["jetson-orin", "jetson-nano", "gpu-pc"]), help="Target device for TensorRT optimization")
@click.option("--fp16", is_flag=True, default=True, help="Use FP16 precision (faster, minimal accuracy loss)")
@click.option("--benchmark", is_flag=True, help="Run inference benchmark after compilation")
def export(checkpoint, output_onnx, output_trt, target_device, fp16, benchmark):
    """Export trained policy to ONNX and TensorRT for edge deployment."""
    helpers.heading("Policy Export Pipeline")
    helpers.info(f"Checkpoint:  {checkpoint}")
    helpers.info(f"Target:      {target_device}")
    helpers.info(f"Precision:   {'FP16' if fp16 else 'FP32'}")

    # Lazy-import (pulls torch)
    try:
        from training.scripts import export as export_module

        # Step 1: PyTorch → ONNX
        helpers.info("\n[1/3] Exporting to ONNX...")
        onnx_path = export_module.export_to_onnx(checkpoint, output_onnx, {})

        # Step 2: ONNX → TensorRT
        helpers.info("\n[2/3] Compiling TensorRT engine...")
        trt_path = export_module.compile_tensorrt(onnx_path, output_trt, target_device, fp16)

        # Step 3: Benchmark (optional)
        if benchmark and trt_path:
            helpers.info("\n[3/3] Benchmarking...")
            export_module.benchmark_inference(trt_path)
        else:
            helpers.info("\n[3/3] Skipping benchmark (use --benchmark to enable)")

        helpers.success(f"\n{'='*60}")
        helpers.success("Export complete!")
        helpers.info(f"ONNX model:      {output_onnx}")
        if trt_path:
            helpers.info(f"TensorRT engine: {output_trt}")
        helpers.success(f"{'='*60}")
    except Exception as e:
        helpers.error(f"Export failed: {e}")
        raise


def register(cli: click.Group):
    """Register export command with the CLI."""
    cli.add_command(export)
