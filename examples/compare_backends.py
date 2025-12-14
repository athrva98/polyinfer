"""Backend Comparison Example.

This example demonstrates how to compare inference performance
across different backends and models.

Run: python examples/compare_backends.py
"""

import numpy as np
from pathlib import Path

import polyinfer as pi


def export_models():
    """Export common models to ONNX for testing."""
    models = {}

    # Try to export models using PyTorch
    try:
        import torch
        import torchvision.models as tv_models

        print("Exporting models from torchvision...")

        model_configs = [
            ("resnet18", tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT, (1, 3, 224, 224)),
            ("resnet50", tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT, (1, 3, 224, 224)),
            ("mobilenet_v2", tv_models.mobilenet_v2, tv_models.MobileNet_V2_Weights.DEFAULT, (1, 3, 224, 224)),
            ("efficientnet_b0", tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT, (1, 3, 224, 224)),
        ]

        for name, model_fn, weights, input_shape in model_configs:
            output_dir = Path(f"./models/{name}-onnx")
            output_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = output_dir / f"{name}.onnx"
            if onnx_path.exists():
                print(f"  {name}: exists")
                models[name] = {"path": str(onnx_path), "input_shape": input_shape}
                continue

            try:
                model = model_fn(weights=weights)
                model.eval()
                dummy = torch.randn(*input_shape)
                torch.onnx.export(
                    model, dummy, str(onnx_path),
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=17,
                )
                print(f"  {name}: exported")
                models[name] = {"path": str(onnx_path), "input_shape": input_shape}
            except Exception as e:
                print(f"  {name}: failed ({e})")

    except ImportError:
        print("PyTorch not installed, skipping torchvision models")

    # Try YOLOv8
    try:
        from ultralytics import YOLO

        for variant in ["yolov8n", "yolov8s"]:
            output_dir = Path(f"./models/{variant}-onnx")
            output_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = output_dir / f"{variant}.onnx"
            if onnx_path.exists():
                print(f"  {variant}: exists")
                models[variant] = {"path": str(onnx_path), "input_shape": (1, 3, 640, 640)}
            else:
                try:
                    yolo = YOLO(f"{variant}.pt")
                    yolo.export(format="onnx", imgsz=640)
                    # Move exported file to models directory
                    exported = Path(f"{variant}.onnx")
                    if exported.exists():
                        exported.rename(onnx_path)
                    print(f"  {variant}: exported")
                    models[variant] = {"path": str(onnx_path), "input_shape": (1, 3, 640, 640)}
                except Exception as e:
                    print(f"  {variant}: failed ({e})")
    except ImportError:
        print("Ultralytics not installed, skipping YOLO models")

    return models


def compare_model(name: str, onnx_path: str, input_shape: tuple, device: str = "cpu"):
    """Compare backends for a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"Input: {input_shape}")
    print(f"Device: {device}")
    print("=" * 60)

    results = pi.compare(
        onnx_path,
        input_shape=input_shape,
        device=device,
        warmup=10,
        iterations=50,
        verbose=True,
    )

    return results


def create_summary_table(all_results: dict):
    """Create a summary table of all results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # Collect all backends
    backends = set()
    for model_results in all_results.values():
        for r in model_results:
            if r["status"] == "success":
                backends.add(r["backend"])
    backends = sorted(backends)

    # Header
    header = f"{'Model':<20}"
    for b in backends:
        header += f" {b:>12}"
    print(header)
    print("-" * 80)

    # Rows
    for model_name, results in all_results.items():
        row = f"{model_name:<20}"
        result_map = {r["backend"]: r for r in results if r["status"] == "success"}

        for b in backends:
            if b in result_map:
                ms = result_map[b]["mean_ms"]
                row += f" {ms:>10.2f}ms"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print("-" * 80)

    # Find winners
    print("\nFastest backend per model:")
    for model_name, results in all_results.items():
        successful = [r for r in results if r["status"] == "success"]
        if successful:
            best = min(successful, key=lambda r: r["mean_ms"])
            print(f"  {model_name}: {best['backend']} ({best['mean_ms']:.2f}ms)")


def main():
    print("=" * 60)
    print("PolyInfer Backend Comparison")
    print("=" * 60)
    print()

    # Show available backends
    print("Available backends:", pi.list_backends())
    print()

    # Export models
    print("Preparing models...")
    models = export_models()

    if not models:
        print("\nNo models available. Please install torch/torchvision or ultralytics.")
        return

    print(f"\nModels ready: {list(models.keys())}")

    # Compare each model
    all_results = {}
    for name, config in models.items():
        try:
            results = compare_model(
                name,
                config["path"],
                config["input_shape"],
                device="cpu",
            )
            all_results[name] = results
        except Exception as e:
            print(f"Error comparing {name}: {e}")

    # Summary
    if all_results:
        create_summary_table(all_results)


if __name__ == "__main__":
    main()
