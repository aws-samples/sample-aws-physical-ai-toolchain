"""
Policy Export: PyTorch → ONNX → TensorRT

Converts a trained pick-and-place policy to an optimized TensorRT engine
for real-time inference on edge hardware (Jetson Orin or GPU PC).

Pipeline:
  1. Load PyTorch checkpoint (.pt)
  2. Export to ONNX (.onnx) — portable intermediate format
  3. Compile with TensorRT (.trt) — hardware-specific optimized engine

Usage:
    python export.py --checkpoint checkpoints/best_policy.pt \
                     --output-onnx models/policy.onnx \
                     --output-trt models/policy.trt \
                     --target-device jetson-orin
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np


def export_to_onnx(checkpoint_path: str, onnx_path: str, input_shapes: dict) -> str:
    """Export PyTorch model to ONNX format."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    model = torch.jit.load(checkpoint_path, map_location='cpu')
    model.eval()

    # Create dummy inputs matching observation space
    # Joint pos (6) + joint vel (6) + gripper (1) + object pose (7) + camera (64x64x3)
    dummy_inputs = {
        'joint_pos': torch.randn(1, 6),
        'joint_vel': torch.randn(1, 6),
        'gripper_state': torch.randn(1, 1),
        'object_pos_relative': torch.randn(1, 7),
        'wrist_camera': torch.randn(1, 3, 64, 64),  # CHW format
    }

    # Flatten to single tensor (as rl_games outputs)
    # Total: 6 + 6 + 1 + 7 + (64*64*3) = 12308
    dummy_flat = torch.cat([
        dummy_inputs['joint_pos'],
        dummy_inputs['joint_vel'],
        dummy_inputs['gripper_state'],
        dummy_inputs['object_pos_relative'],
        dummy_inputs['wrist_camera'].flatten(1),
    ], dim=-1)

    print(f"  Input shape: {dummy_flat.shape}")
    print(f"  Exporting to ONNX: {onnx_path}")

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_flat,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['observations'],
        output_names=['actions'],
        dynamic_axes={
            'observations': {0: 'batch_size'},
            'actions': {0: 'batch_size'},
        },
    )

    # Validate ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX export successful. Model size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

    return onnx_path


def compile_tensorrt(onnx_path: str, trt_path: str, target_device: str, fp16: bool = True) -> str:
    """Compile ONNX model to TensorRT engine for target hardware."""
    try:
        import tensorrt as trt
    except ImportError:
        print("  WARNING: TensorRT not available. Skipping TRT compilation.")
        print("  The ONNX model can be compiled on the target device instead.")
        return ""

    print(f"  Compiling TensorRT engine for: {target_device}")
    print(f"  FP16 precision: {fp16}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Set optimization profile (batch size 1 for real-time inference)
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # Min/opt/max batch sizes
    profile.set_shape(
        input_tensor.name,
        min=(1, input_shape[1]),
        opt=(1, input_shape[1]),
        max=(1, input_shape[1]),
    )
    config.add_optimization_profile(profile)

    # Build engine
    print("  Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    os.makedirs(os.path.dirname(trt_path), exist_ok=True)
    with open(trt_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"  TensorRT engine saved: {trt_path}")
    print(f"  Engine size: {os.path.getsize(trt_path) / 1024 / 1024:.1f} MB")

    return trt_path


def benchmark_inference(trt_path: str, num_iterations: int = 1000):
    """Benchmark inference speed of TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        print("  Skipping benchmark (TensorRT/PyCUDA not available)")
        return

    print(f"\n  Benchmarking inference ({num_iterations} iterations)...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(trt_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (1, 12308)  # Flattened observation
    output_shape = (1, 7)     # Action (6 DOF + gripper)

    h_input = np.random.randn(*input_shape).astype(np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Warmup
    for _ in range(100):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(num_iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_iterations) * 1000
    hz = num_iterations / elapsed

    print(f"  Average inference time: {avg_ms:.2f} ms")
    print(f"  Inference rate: {hz:.0f} Hz")
    print(f"  {'✅ PASS' if hz > 100 else '⚠️  SLOW'}: Target is >100 Hz for real-time control")


def main():
    parser = argparse.ArgumentParser(description='Export trained policy to TensorRT')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained PyTorch checkpoint (.pt)')
    parser.add_argument('--output-onnx', type=str, required=True,
                        help='Output path for ONNX model')
    parser.add_argument('--output-trt', type=str, required=True,
                        help='Output path for TensorRT engine')
    parser.add_argument('--target-device', type=str, default='jetson-orin',
                        choices=['jetson-orin', 'jetson-nano', 'gpu-pc'],
                        help='Target device for TensorRT optimization')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 precision (faster, minimal accuracy loss)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference benchmark after compilation')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Policy Export Pipeline")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Target: {args.target_device}")
    print(f"  Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"{'='*60}")

    # Step 1: PyTorch → ONNX
    print(f"\n[1/3] Exporting to ONNX...")
    onnx_path = export_to_onnx(args.checkpoint, args.output_onnx, {})

    # Step 2: ONNX → TensorRT
    print(f"\n[2/3] Compiling TensorRT engine...")
    trt_path = compile_tensorrt(onnx_path, args.output_trt, args.target_device, args.fp16)

    # Step 3: Benchmark (optional)
    if args.benchmark and trt_path:
        print(f"\n[3/3] Benchmarking...")
        benchmark_inference(trt_path)
    else:
        print(f"\n[3/3] Skipping benchmark (use --benchmark to enable)")

    print(f"\n{'='*60}")
    print(f"  Export complete!")
    print(f"  ONNX model: {args.output_onnx}")
    if trt_path:
        print(f"  TensorRT engine: {args.output_trt}")
    print(f"  Deploy with: osmo workflow submit -f workflows/pick-and-place.yaml")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
