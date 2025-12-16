"""PolyInfer Quickstart Example.

This example demonstrates the basic usage of PolyInfer:
1. Loading a model with auto-backend selection
2. Running inference
3. Benchmarking performance across CPU and GPU
4. Comparing all backends on all available devices

Run: python examples/quickstart.py
"""

from pathlib import Path

import numpy as np
import polyinfer as pi


def has_cuda() -> bool:
    """Check if CUDA is available via any backend."""
    for device in pi.list_devices():
        if device.device_type == "cuda":
            return True
    return False


def main():
    print("=" * 60)
    print("PolyInfer Quickstart")
    print("=" * 60)
    print()

    # Show available backends
    print("Available backends:", pi.list_backends())
    print()

    # Show available devices
    print("Available devices:")
    for device in pi.list_devices():
        print(f"  {device}")
    print()

    cuda_available = has_cuda()
    if cuda_available:
        print("CUDA is available - will benchmark on both CPU and GPU")
    else:
        print("CUDA not available - will benchmark on CPU only")
    print()

    # Download/use a simple model (ResNet18 from torchvision)
    output_dir = Path("./models/resnet18-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "resnet18.onnx"

    # Export ResNet18 to ONNX if it doesn't exist
    if model_path.exists():
        print(f"Using existing model: {model_path}")
    else:
        print("Exporting ResNet18 to ONNX...")
        import torch
        import torchvision.models as models

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            resnet,
            dummy_input,
            str(model_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"Exported: {model_path}")

    print()

    # Prepare input data
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # ===========================================
    # 1. Load with auto-backend selection (best available)
    # ===========================================
    print("1. Loading model (auto-select best backend)...")
    default_device = "cuda" if cuda_available else "cpu"
    model = pi.load(model_path, device=default_device)
    print(f"   Model: {model}")
    print(f"   Backend: {model.backend_name}")
    print(f"   Device: {model.device}")
    print(f"   Inputs: {model.input_names} -> {model.input_shapes}")
    print(f"   Outputs: {model.output_names}")
    print()

    # ===========================================
    # 2. Run inference
    # ===========================================
    print("2. Running inference...")
    output = model(input_data)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {output.shape}")

    # Get top-5 predictions
    top5_idx = np.argsort(output[0])[-5:][::-1]
    print(f"   Top-5 class indices: {top5_idx}")
    print()

    # ===========================================
    # 3. Benchmark on default device
    # ===========================================
    print(f"3. Benchmarking on {default_device.upper()} (100 iterations)...")
    results = model.benchmark(input_data, warmup=10, iterations=100)
    print(f"   Mean latency: {results['mean_ms']:.2f} ms")
    print(f"   Std dev: {results['std_ms']:.2f} ms")
    print(f"   Throughput: {results['fps']:.1f} FPS")
    print(f"   P99 latency: {results['p99_ms']:.2f} ms")
    print()

    # ===========================================
    # 4. Compare ALL backends on ALL devices
    # ===========================================
    print("4. Benchmarking all backends on all available devices...")
    print("-" * 60)

    # Collect all results
    all_results = []

    for backend_name in pi.list_backends():
        backend = pi.get_backend(backend_name)
        supported_devices = backend.supported_devices

        for device in supported_devices:
            # Skip vulkan for now (often not configured)
            if "vulkan" in device:
                continue

            # Normalize device name
            device_type = device.split(":")[0] if ":" in device else device

            # Skip CUDA devices if not available
            if device_type in ("cuda", "tensorrt") and not cuda_available:
                continue

            try:
                m = pi.load(model_path, backend=backend_name, device=device)
                result = m.benchmark(input_data, warmup=10, iterations=50)
                all_results.append({
                    "backend": backend_name,
                    "device": device,
                    "mean_ms": result["mean_ms"],
                    "fps": result["fps"],
                })
                print(f"   {backend_name:15} ({device:10}): {result['mean_ms']:8.2f} ms ({result['fps']:7.1f} FPS)")
            except Exception as e:
                print(f"   {backend_name:15} ({device:10}): Error - {str(e)[:50]}")

    print()

    # ===========================================
    # 5. Summary - sorted by speed
    # ===========================================
    if all_results:
        print("5. Summary (sorted by speed):")
        print("=" * 60)
        all_results.sort(key=lambda x: x["mean_ms"])
        fastest = all_results[0]["mean_ms"]

        for i, r in enumerate(all_results):
            speedup = r["mean_ms"] / fastest
            marker = " <-- FASTEST" if i == 0 else f" ({speedup:.1f}x slower)"
            print(f"   {r['backend']:15} ({r['device']:10}): {r['mean_ms']:8.2f} ms ({r['fps']:7.1f} FPS){marker}")

        print()

        # Show speedup from CPU to GPU if both available
        cpu_results = [r for r in all_results if r["device"] == "cpu"]
        gpu_results = [r for r in all_results if r["device"] in ("cuda", "tensorrt")]

        if cpu_results and gpu_results:
            best_cpu = min(cpu_results, key=lambda x: x["mean_ms"])
            best_gpu = min(gpu_results, key=lambda x: x["mean_ms"])
            speedup = best_cpu["mean_ms"] / best_gpu["mean_ms"]
            print(f"   GPU Speedup: {speedup:.1f}x faster than CPU")
            print(f"   Best CPU: {best_cpu['backend']} ({best_cpu['mean_ms']:.2f} ms)")
            print(f"   Best GPU: {best_gpu['backend']} on {best_gpu['device']} ({best_gpu['mean_ms']:.2f} ms)")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
