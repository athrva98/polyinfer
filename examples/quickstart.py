"""PolyInfer Quickstart Example.

This example demonstrates the basic usage of PolyInfer:
1. Loading a model with auto-backend selection
2. Running inference
3. Benchmarking performance
4. Comparing backends

Run: python examples/quickstart.py
"""

from pathlib import Path

import numpy as np
import polyinfer as pi


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

    # ===========================================
    # 1. Load with auto-backend selection
    # ===========================================
    print("1. Loading model (auto-select backend)...")
    model = pi.load(model_path, device="cpu")
    print(f"   Model: {model}")
    print(f"   Backend: {model.backend_name}")
    print(f"   Inputs: {model.input_names} -> {model.input_shapes}")
    print(f"   Outputs: {model.output_names}")
    print()

    # ===========================================
    # 2. Run inference
    # ===========================================
    print("2. Running inference...")
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    output = model(input_data)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Output shape: {output.shape}")

    # Get top-5 predictions
    top5_idx = np.argsort(output[0])[-5:][::-1]
    print(f"   Top-5 class indices: {top5_idx}")
    print()

    # ===========================================
    # 3. Benchmark performance
    # ===========================================
    print("3. Benchmarking (100 iterations)...")
    results = model.benchmark(input_data, warmup=10, iterations=100)
    print(f"   Mean latency: {results['mean_ms']:.2f} ms")
    print(f"   Std dev: {results['std_ms']:.2f} ms")
    print(f"   Throughput: {results['fps']:.1f} FPS")
    print(f"   P99 latency: {results['p99_ms']:.2f} ms")
    print()

    # ===========================================
    # 4. Compare backends
    # ===========================================
    print("4. Comparing all available backends...")
    pi.compare(model_path, input_shape=(1, 3, 224, 224), warmup=10, iterations=50)
    print()

    # ===========================================
    # 5. Explicit backend selection
    # ===========================================
    print("5. Explicit backend selection...")
    for backend_name in pi.list_backends():
        # TensorRT requires GPU
        device = "cuda" if backend_name == "tensorrt" else "cpu"
        try:
            m = pi.load(model_path, backend=backend_name, device=device)
            result = m.benchmark(input_data, warmup=5, iterations=20)
            print(f"   {backend_name} ({device}): {result['mean_ms']:.2f} ms ({result['fps']:.1f} FPS)")
        except Exception as e:
            print(f"   {backend_name} ({device}): Error - {e}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
