#!/usr/bin/env python3
"""LLM Inference Example - Running Small LLMs Across Multiple Backends

This example demonstrates how to run a small language model (SmolLM-135M)
on different backends using PolyInfer.

Models tested:
- SmolLM-135M (HuggingFace: HuggingFaceTB/SmolLM-135M)
- TinyLlama-1.1B (HuggingFace: TinyLlama/TinyLlama-1.1B-Chat-v1.0)

Requirements:
    pip install polyinfer[nvidia]  # or [cpu], [intel], etc.
    pip install transformers optimum[onnxruntime]

Export model to ONNX (one-time):
    optimum-cli export onnx --model HuggingFaceTB/SmolLM-135M smollm-135m-onnx/

Backend Limitations:
- IREE (all devices): Not supported for LLMs with KV-cache. IREE has issues
  with dynamic shapes in past_key_values tensors. Use ONNX Runtime or OpenVINO
  for LLM inference. For non-LLM models (CNNs, vision transformers), IREE works
  well including Vulkan backend.
- OpenVINO/NPU: Not supported for LLM workloads. The NPU compiler has issues
  with dynamic shapes and certain transformer operations (rotary embeddings).
  Use OpenVINO/CPU or OpenVINO/Intel-GPU instead.
"""

import argparse
import time
from pathlib import Path

import numpy as np

# Check if we can import polyinfer
try:
    import polyinfer as pi
except ImportError:
    print("Please install polyinfer: pip install polyinfer[nvidia]")
    exit(1)


def export_model_to_onnx(model_name: str, output_dir: str) -> Path:
    """Export a HuggingFace model to ONNX format.

    Args:
        model_name: HuggingFace model name (e.g., "HuggingFaceTB/SmolLM-135M")
        output_dir: Directory to save the ONNX model

    Returns:
        Path to the exported ONNX model
    """
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError:
        print("Please install optimum: pip install optimum[onnxruntime]")
        exit(1)

    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Exporting {model_name} to ONNX...")
        model = ORTModelForCausalLM.from_pretrained(
            model_name,
            export=True,
        )
        model.save_pretrained(output_path)
        print(f"Model exported to {output_path}")
    else:
        print(f"Model already exists at {output_path}")

    # Find the decoder model (the main inference model)
    decoder_path = output_path / "decoder_model.onnx"
    if decoder_path.exists():
        return decoder_path

    # Some models use model.onnx
    model_path = output_path / "model.onnx"
    if model_path.exists():
        return model_path

    raise FileNotFoundError(f"Could not find ONNX model in {output_path}")


def get_model_inputs_info(model_path: str) -> dict:
    """Get input information from an ONNX model.

    Args:
        model_path: Path to ONNX model

    Returns:
        Dict with input names, shapes, and detected KV-cache params
    """
    try:
        import onnx
        model = onnx.load(model_path)

        inputs_info = {
            "names": [],
            "shapes": {},
            "num_heads": None,
            "head_dim": None,
        }

        for inp in model.graph.input:
            name = inp.name
            inputs_info["names"].append(name)

            # Get shape (may contain symbolic dims)
            shape = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value:
                    shape.append(d.dim_value)
                else:
                    shape.append(d.dim_param or -1)
            inputs_info["shapes"][name] = shape

            # Detect num_heads and head_dim from KV-cache inputs
            # Shape: (batch, num_heads, past_seq_len, head_dim)
            if name.startswith("past_key_values") and len(shape) >= 4:
                if isinstance(shape[1], int) and shape[1] > 0:
                    inputs_info["num_heads"] = shape[1]
                if isinstance(shape[3], int) and shape[3] > 0:
                    inputs_info["head_dim"] = shape[3]

        return inputs_info
    except ImportError:
        return {"names": ["input_ids", "attention_mask"], "shapes": {}, "num_heads": None, "head_dim": None}


def create_dummy_input(
    model_path: str,
    batch_size: int = 1,
    seq_len: int = 32,
) -> dict:
    """Create dummy input tensors for LLM inference.

    Automatically detects KV-cache requirements from the ONNX model.

    Args:
        model_path: Path to ONNX model (to detect required inputs)
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Dictionary with input tensors
    """
    info = get_model_inputs_info(model_path)
    input_names = info["names"]

    # Use a conservative vocab size (most LLMs have vocab > 30k)
    # Real vocab size: SmolLM=49152, Llama=32000, GPT-2=50257
    inputs = {
        "input_ids": np.random.randint(0, 30000, (batch_size, seq_len)).astype(np.int64),
        "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
    }

    # Check if model needs position_ids
    if "position_ids" in input_names:
        inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

    # Check if model needs KV-cache (past_key_values)
    num_heads = info.get("num_heads") or 3  # Default for SmolLM
    head_dim = info.get("head_dim") or 64    # Default for SmolLM

    for name in input_names:
        if name.startswith("past_key_values"):
            # past_key_values shape: (batch, num_heads, 0, head_dim) - empty cache
            inputs[name] = np.zeros((batch_size, num_heads, 0, head_dim), dtype=np.float32)

    return inputs


def benchmark_backend(
    model_path: str,
    backend: str,
    device: str,
    input_data: dict,
    warmup: int = 5,
    iterations: int = 20,
) -> dict | None:
    """Benchmark a model on a specific backend/device.

    Args:
        model_path: Path to ONNX model
        backend: Backend name (onnxruntime, openvino, iree)
        device: Device name (cpu, cuda, tensorrt, etc.)
        input_data: Input tensors
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results or None if backend not available
    """
    try:
        # Check if backend supports this device
        backend_obj = pi.get_backend(backend)
        if not backend_obj.supports_device(device):
            return None

        # Load model
        model = pi.load(model_path, backend=backend, device=device)

        # Prepare inputs (flatten dict to positional args)
        inputs = list(input_data.values())

        # Warmup
        for _ in range(warmup):
            try:
                _ = model(*inputs)
            except Exception:
                return None

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(*inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "backend": backend,
            "device": device,
            "backend_name": model.backend_name,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "tokens_per_sec": input_data["input_ids"].shape[1] * 1000 / np.mean(times),
        }

    except Exception as e:
        print(f"  {backend}/{device}: Error - {e}")
        return None


def run_all_backends(model_path: str, seq_len: int = 32):
    """Run benchmark on all available backends.

    Args:
        model_path: Path to ONNX model
        seq_len: Sequence length for benchmark
    """
    print(f"\n{'='*70}")
    print(f"LLM Benchmark: {Path(model_path).parent.name}")
    print(f"Sequence Length: {seq_len} tokens")
    print(f"{'='*70}\n")

    # Show available backends and devices
    print("Available backends:", pi.list_backends())
    print("Available devices:")
    for d in pi.list_devices():
        print(f"  - {d.name}: {d.backends}")
    print()

    # Create input data (automatically detects KV-cache requirements)
    input_data = create_dummy_input(model_path, batch_size=1, seq_len=seq_len)

    # Define backend/device combinations to test
    # NOTE: IREE excluded for LLMs - has issues with dynamic KV-cache shapes
    # NOTE: NPU excluded for LLMs - dynamic shape/rotary embedding issues
    test_configs = [
        # CPU backends (ONNX Runtime and OpenVINO only - IREE doesn't support KV-cache)
        ("onnxruntime", "cpu"),
        ("openvino", "cpu"),

        # NVIDIA GPU (ONNX Runtime only for LLMs)
        ("onnxruntime", "cuda"),
        ("onnxruntime", "tensorrt"),

        # Intel GPU
        ("openvino", "intel-gpu"),

        # DirectML (Windows AMD/Intel)
        ("onnxruntime", "directml"),
    ]

    results = []

    print("Running benchmarks...")
    print("-" * 70)

    for backend, device in test_configs:
        # Check if backend is available
        if backend not in pi.list_backends():
            continue

        result = benchmark_backend(
            model_path, backend, device, input_data,
            warmup=5, iterations=20
        )

        if result:
            results.append(result)
            print(f"  {result['backend_name']:<30} {result['mean_ms']:>8.2f}ms  "
                  f"({result['tokens_per_sec']:>8.1f} tok/s)")

    print("-" * 70)

    # Sort by speed and show summary
    if results:
        results.sort(key=lambda x: x["mean_ms"])

        print("\n" + "=" * 70)
        print("RESULTS (sorted by speed)")
        print("=" * 70)
        print(f"{'Backend':<30} {'Latency':>10} {'Tok/s':>10} {'Speedup':>10}")
        print("-" * 70)

        baseline = results[-1]["mean_ms"]  # Slowest as baseline

        for r in results:
            speedup = baseline / r["mean_ms"]
            print(f"{r['backend_name']:<30} {r['mean_ms']:>8.2f}ms {r['tokens_per_sec']:>9.1f} {speedup:>9.1f}x")

        print("-" * 70)
        print(f"\nFastest: {results[0]['backend_name']} ({results[0]['mean_ms']:.2f}ms)")
    else:
        print("\nNo backends available for this model.")


def export_to_mlir(model_path: str, output_dir: str):
    """Export ONNX model to MLIR for custom hardware.

    NOTE: MLIR export is primarily useful for non-LLM models (CNNs, vision
    transformers) where IREE excels. For LLMs with KV-cache, IREE has
    limitations with dynamic shapes. This function demonstrates the MLIR
    workflow - for production LLM inference, use ONNX Runtime or OpenVINO.

    Args:
        model_path: Path to ONNX model
        output_dir: Directory to save MLIR files
    """
    print(f"\n{'='*70}")
    print("Exporting to MLIR for Custom Hardware")
    print("(Note: IREE/MLIR best suited for CNNs, not LLMs with KV-cache)")
    print(f"{'='*70}\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem

    # Export to MLIR
    print("1. Exporting ONNX to MLIR...")
    mlir = pi.export_mlir(model_path, output_path / f"{model_name}.mlir", load_content=True)
    print(f"   Saved to: {mlir.path}")
    print(f"   Size: {mlir.path.stat().st_size / 1024:.1f} KB")
    print(f"   First 200 chars:\n   {mlir.content[:200]}...")

    # Compile for different targets
    # NOTE: Vulkan excluded - LLM ReduceMean ops exceed 16KB shared memory limit
    targets = [
        ("cpu", "llvm-cpu"),
        # ("vulkan", "vulkan-spirv"),  # Not supported for LLMs
    ]

    # Check if CUDA is available
    devices = pi.list_devices()
    if any(d.name == "cuda" and "iree" in d.backends for d in devices):
        targets.append(("cuda", "cuda"))

    print("\n2. Compiling for different targets...")
    for device, target_name in targets:
        try:
            vmfb_path = pi.compile_mlir(
                mlir.path,
                device=device,
                output_path=output_path / f"{model_name}_{target_name}.vmfb"
            )
            print(f"   {target_name}: {vmfb_path} ({vmfb_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"   {target_name}: Failed - {e}")

    print("\n3. Loading and testing compiled models...")
    backend = pi.get_backend("iree")
    input_data = create_dummy_input(model_path, batch_size=1, seq_len=32)
    inputs = list(input_data.values())

    for device, target_name in targets:
        vmfb_path = output_path / f"{model_name}_{target_name}.vmfb"
        if vmfb_path.exists():
            try:
                model = backend.load_vmfb(vmfb_path, device=device)
                output = model(*inputs)
                print(f"   {target_name}: Output shape = {output.shape}")
            except Exception as e:
                print(f"   {target_name}: Load failed - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Example - Run small LLMs across backends"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="HuggingFace model name or path to ONNX model"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/smollm-135m-onnx",
        help="Directory to save/load ONNX model"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for benchmark"
    )
    parser.add_argument(
        "--export-mlir",
        action="store_true",
        help="Export model to MLIR for custom hardware"
    )
    parser.add_argument(
        "--mlir-dir",
        type=str,
        default="./models/smollm-135m-mlir",
        help="Directory to save MLIR files"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export, use existing model"
    )

    args = parser.parse_args()

    # Get or export model
    model_path = Path(args.model_dir)

    if model_path.suffix == ".onnx":
        # Direct path to ONNX file
        onnx_path = model_path
    elif model_path.exists() and args.skip_export:
        # Use existing exported model
        onnx_path = model_path / "decoder_model.onnx"
        if not onnx_path.exists():
            onnx_path = model_path / "model.onnx"
    else:
        # Export from HuggingFace
        onnx_path = export_model_to_onnx(args.model, args.model_dir)

    if not onnx_path.exists():
        print(f"Error: Model not found at {onnx_path}")
        print("Run without --skip-export to download and export the model.")
        exit(1)

    print(f"Using model: {onnx_path}")

    # Run benchmarks
    run_all_backends(str(onnx_path), seq_len=args.seq_len)

    # Optionally export to MLIR
    if args.export_mlir:
        export_to_mlir(str(onnx_path), args.mlir_dir)


if __name__ == "__main__":
    main()
