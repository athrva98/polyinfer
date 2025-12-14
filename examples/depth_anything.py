#!/usr/bin/env python3
"""Depth Anything - Monocular Depth Estimation with PolyInfer.

This example demonstrates:
1. Loading Depth Anything models for depth estimation
2. Predicting depth maps from single images
3. Real-time depth estimation from webcam
4. Benchmarking across backends

Depth Anything is a foundation model for monocular depth estimation that
predicts relative depth from a single RGB image.

Models:
- depth-anything-small: Fastest, 24.8M params
- depth-anything-base: Balanced, 97.5M params
- depth-anything-large: Best quality, 335.3M params
- depth-anything-v2-small: Improved v2, small
- depth-anything-v2-base: Improved v2, base
- depth-anything-v2-large: Improved v2, large

Backend Limitations:
- DirectML: Not supported. DINOv2 backbone uses dynamic Reshape operations
  that DirectML cannot handle.
- IREE/Vulkan: Not supported. LayerNormalization ops exceed Vulkan's 16KB
  shared memory limit.
- Recommended: Use ONNX Runtime CPU, OpenVINO CPU, or CUDA backends.

Requirements:
    pip install polyinfer[cpu]  # or [nvidia], [intel]
    pip install transformers pillow opencv-python

Usage:
    python depth_anything.py --image photo.jpg
    python depth_anything.py --webcam
    python depth_anything.py --benchmark
"""

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import polyinfer as pi
except ImportError:
    print("Please install polyinfer: pip install polyinfer[cpu]")
    exit(1)


# Depth Anything model configurations
DEPTH_MODELS = {
    "depth-anything-small": "LiheYoung/depth-anything-small-hf",
    "depth-anything-base": "LiheYoung/depth-anything-base-hf",
    "depth-anything-large": "LiheYoung/depth-anything-large-hf",
    "depth-anything-v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything-v2-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything-v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def load_image(image_path: str) -> np.ndarray:
    """Load image for depth estimation.

    Args:
        image_path: Path to image file

    Returns:
        RGB image array (H, W, 3)
    """
    try:
        from PIL import Image
    except ImportError:
        print("Please install pillow: pip install pillow")
        raise

    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def preprocess_image(image: np.ndarray, size: int = 518) -> tuple[np.ndarray, tuple[int, int]]:
    """Preprocess image for Depth Anything.

    Args:
        image: RGB image array (H, W, 3)
        size: Target size (518 for Depth Anything)

    Returns:
        Tuple of (preprocessed tensor, original size)
    """
    original_size = (image.shape[0], image.shape[1])

    try:
        import cv2
        # Resize to target size
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(image)
        pil_img = pil_img.resize((size, size), PILImage.BILINEAR)
        resized = np.array(pil_img)

    # Normalize with ImageNet stats (Depth Anything uses DINOv2 backbone)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std

    # HWC -> CHW -> NCHW
    tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    return tensor, original_size


def postprocess_depth(
    depth: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    """Postprocess depth output to original size.

    Args:
        depth: Depth prediction (1, 1, H, W) or (1, H, W)
        original_size: Original image size (H, W)

    Returns:
        Depth map at original size (H, W)
    """
    # Handle different output shapes
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]

    try:
        import cv2
        depth_resized = cv2.resize(
            depth,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    except ImportError:
        from PIL import Image
        pil_depth = Image.fromarray(depth)
        pil_depth = pil_depth.resize(
            (original_size[1], original_size[0]),
            Image.BILINEAR,
        )
        depth_resized = np.array(pil_depth)

    return depth_resized


def depth_to_colormap(depth: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Convert depth map to colored visualization.

    Args:
        depth: Depth map (H, W)
        colormap: Matplotlib colormap name

    Returns:
        RGB image (H, W, 3) as uint8
    """
    # Normalize to [0, 1]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)

    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        colored = cmap(depth_normalized)[:, :, :3]  # Remove alpha
        return (colored * 255).astype(np.uint8)
    except ImportError:
        # Fallback: grayscale
        gray = (depth_normalized * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def export_depth_anything_onnx(model_name: str, output_dir: str) -> Path:
    """Export Depth Anything model to ONNX.

    Args:
        model_name: Model name from DEPTH_MODELS
        output_dir: Directory to save ONNX file

    Returns:
        Path to ONNX model
    """
    output_path = Path(output_dir)
    onnx_path = output_path / "depth_anything.onnx"

    if onnx_path.exists():
        print(f"Model already exists: {onnx_path}")
        return onnx_path

    print(f"Exporting {model_name} to ONNX...")
    print("This may take a few minutes...")

    try:
        import torch
        from transformers import AutoModelForDepthEstimation
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers torch")
        raise

    hf_model = DEPTH_MODELS.get(model_name, model_name)

    # Load model
    print(f"Loading model from {hf_model}...")
    model = AutoModelForDepthEstimation.from_pretrained(hf_model)
    model.eval()

    output_path.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print("Exporting to ONNX...")

    class DepthWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            outputs = self.model(pixel_values)
            return outputs.predicted_depth

    wrapper = DepthWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 518, 518)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "predicted_depth": {0: "batch", 1: "height", 2: "width"},
        },
        opset_version=17,
    )

    print(f"Model exported to: {onnx_path}")
    return onnx_path


class DepthAnythingInference:
    """Depth Anything inference wrapper using PolyInfer."""

    def __init__(
        self,
        model_dir: str,
        backend: str = "onnxruntime",
        device: str = "cpu",
        input_size: int = 518,
    ):
        """Initialize Depth Anything inference.

        Args:
            model_dir: Directory containing ONNX model
            backend: Backend to use
            device: Device to use
            input_size: Model input size
        """
        model_path = Path(model_dir) / "depth_anything.onnx"

        if not model_path.exists():
            raise FileNotFoundError(f"Depth model not found: {model_path}")

        # Check for unsupported backends
        if device == "directml":
            raise ValueError(
                "DirectML is not supported for Depth Anything. "
                "The DINOv2 backbone uses dynamic Reshape operations that DirectML cannot handle. "
                "Use --device cpu or --device cuda instead."
            )
        if device == "vulkan" and backend == "iree":
            raise ValueError(
                "IREE/Vulkan is not supported for Depth Anything. "
                "LayerNormalization ops exceed Vulkan's 16KB shared memory limit. "
                "Use --device cpu or --device cuda instead."
            )

        print(f"Loading Depth Anything with {backend}/{device}...")
        self.model = pi.load(str(model_path), backend=backend, device=device)
        self.backend_name = self.model.backend_name
        self.input_size = input_size

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Predict depth from RGB image.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            Tuple of (depth_map, inference_time_ms)
        """
        tensor, original_size = preprocess_image(image, self.input_size)

        start = time.perf_counter()
        depth = self.model(tensor)
        elapsed = (time.perf_counter() - start) * 1000

        depth = postprocess_depth(depth, original_size)
        return depth, elapsed

    def predict_colored(
        self,
        image: np.ndarray,
        colormap: str = "inferno",
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Predict depth and return colored visualization.

        Args:
            image: RGB image array (H, W, 3)
            colormap: Matplotlib colormap name

        Returns:
            Tuple of (depth_map, colored_depth, inference_time_ms)
        """
        depth, elapsed = self.predict(image)
        colored = depth_to_colormap(depth, colormap)
        return depth, colored, elapsed


def run_depth_estimation(args):
    """Run depth estimation on an image."""
    model_path = Path(args.model_dir)
    if not (model_path / "depth_anything.onnx").exists():
        export_depth_anything_onnx(args.model, args.model_dir)

    depth_model = DepthAnythingInference(args.model_dir, args.backend, args.device)
    print(f"Loaded: {depth_model.backend_name}")

    # Load image
    image = load_image(args.image)
    print(f"\nImage size: {image.shape[:2]}")

    # Predict depth
    depth, colored, elapsed = depth_model.predict_colored(image, args.colormap)

    print(f"Inference time: {elapsed:.2f}ms")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Save results
    try:
        from PIL import Image
        import cv2

        # Save depth colormap
        output_path = args.output or "./models/outputs/depth_output.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colored).save(output_path)
        print(f"Saved depth visualization: {output_path}")

        # Save side-by-side comparison
        comparison_path = output_path.replace(".png", "_comparison.png")
        h, w = image.shape[:2]

        # Resize depth colored to match image
        if colored.shape[:2] != image.shape[:2]:
            colored = cv2.resize(colored, (w, h))

        comparison = np.concatenate([image, colored], axis=1)
        Image.fromarray(comparison).save(comparison_path)
        print(f"Saved comparison: {comparison_path}")

        # Save raw depth as .npy
        if args.save_raw:
            raw_path = output_path.replace(".png", "_raw.npy")
            np.save(raw_path, depth)
            print(f"Saved raw depth: {raw_path}")

    except ImportError:
        print("Install pillow to save images: pip install pillow")


def run_webcam_depth(args):
    """Run real-time depth estimation from webcam."""
    try:
        import cv2
    except ImportError:
        print("Please install opencv: pip install opencv-python")
        return

    model_path = Path(args.model_dir)
    if not (model_path / "depth_anything.onnx").exists():
        export_depth_anything_onnx(args.model, args.model_dir)

    depth_model = DepthAnythingInference(args.model_dir, args.backend, args.device)
    print(f"Loaded: {depth_model.backend_name}")

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\nControls: Q=Quit, C=Toggle colormap, S=Save frame")

    frame_times = []
    colormaps = ["inferno", "viridis", "plasma", "magma", "turbo"]
    colormap_idx = 0

    # Warmup
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for _ in range(3):
            _ = depth_model.predict(frame_rgb)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Predict depth
            depth, colored, elapsed = depth_model.predict_colored(
                frame_rgb, colormaps[colormap_idx]
            )

            # Track FPS
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_ms = np.mean(frame_times)
            fps = 1000 / avg_ms

            # Convert colored depth back to BGR for display
            colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

            # Resize to match frame
            colored_bgr = cv2.resize(colored_bgr, (frame.shape[1], frame.shape[0]))

            # Add FPS overlay
            cv2.putText(
                colored_bgr,
                f"FPS: {fps:.1f} | {depth_model.backend_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                colored_bgr,
                f"Colormap: {colormaps[colormap_idx]}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Create side-by-side view
            display = np.concatenate([frame, colored_bgr], axis=1)

            cv2.imshow("Depth Anything - PolyInfer (Press Q to quit)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                colormap_idx = (colormap_idx + 1) % len(colormaps)
            elif key == ord("s"):
                cv2.imwrite("depth_frame.png", display)
                print("Saved: depth_frame.png")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def benchmark_depth(model_name: str, model_dir: str):
    """Benchmark Depth Anything across all backends."""
    model_path = Path(model_dir)

    if not (model_path / "depth_anything.onnx").exists():
        export_depth_anything_onnx(model_name, model_dir)

    print(f"\n{'='*70}")
    print(f"Depth Anything Benchmark: {model_name}")
    print(f"{'='*70}\n")

    # Prepare dummy input
    dummy_input = np.random.randn(1, 3, 518, 518).astype(np.float32)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Available backends: {pi.list_backends()}")
    print()

    # Test configurations
    # NOTE: DirectML excluded - DINOv2 backbone uses dynamic Reshape ops
    # NOTE: Vulkan excluded - LayerNorm exceeds 16KB shared memory limit
    test_configs = [
        ("onnxruntime", "cpu"),
        ("openvino", "cpu"),
        ("iree", "cpu"),
        ("onnxruntime", "cuda"),
        ("onnxruntime", "tensorrt"),
        ("iree", "cuda"),
        # ("iree", "vulkan"),  # Not supported - LayerNorm shared memory
        ("openvino", "intel-gpu"),
        # ("onnxruntime", "directml"),  # Not supported - dynamic Reshape
    ]

    results = []

    print("Running benchmarks...")
    print("-" * 70)

    for backend, device in test_configs:
        if backend not in pi.list_backends():
            continue

        try:
            backend_obj = pi.get_backend(backend)
            if not backend_obj.supports_device(device):
                continue

            model = pi.load(
                str(model_path / "depth_anything.onnx"),
                backend=backend,
                device=device,
            )

            # Warmup
            for _ in range(5):
                _ = model(dummy_input)

            # Benchmark
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = model(dummy_input)
                times.append((time.perf_counter() - start) * 1000)

            mean_ms = np.mean(times)
            fps = 1000 / mean_ms

            results.append({
                "backend": model.backend_name,
                "mean_ms": mean_ms,
                "std_ms": np.std(times),
                "fps": fps,
            })

            print(f"  {model.backend_name:<30} {mean_ms:>8.2f}ms  ({fps:>6.1f} FPS)")

        except Exception as e:
            print(f"  {backend}/{device}: Error - {e}")

    print("-" * 70)

    if results:
        results.sort(key=lambda x: x["mean_ms"])

        print(f"\n{'='*70}")
        print("RESULTS (sorted by speed)")
        print(f"{'='*70}")
        print(f"{'Backend':<30} {'Latency':>10} {'FPS':>10} {'Speedup':>10}")
        print("-" * 70)

        baseline = results[-1]["mean_ms"]
        for r in results:
            speedup = baseline / r["mean_ms"]
            print(f"{r['backend']:<30} {r['mean_ms']:>8.2f}ms {r['fps']:>9.1f} {speedup:>9.1f}x")

        print("-" * 70)
        print(f"\nFastest: {results[0]['backend']} ({results[0]['mean_ms']:.2f}ms, {results[0]['fps']:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything - Monocular Depth Estimation with PolyInfer"
    )
    parser.add_argument(
        "--model",
        default="depth-anything-small",
        choices=list(DEPTH_MODELS.keys()),
        help="Depth Anything model variant",
    )
    parser.add_argument(
        "--model-dir",
        default="./models/depth-anything-small-onnx",
        help="Directory for ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        choices=["inferno", "viridis", "plasma", "magma", "turbo", "gray"],
        help="Colormap for depth visualization",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw depth values as .npy",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run real-time depth from webcam",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index",
    )
    parser.add_argument(
        "--backend",
        default="onnxruntime",
        help="Backend: onnxruntime, openvino, iree",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu, cuda, directml, vulkan",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark across all backends",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export model to ONNX and exit",
    )

    args = parser.parse_args()

    # Update model directory based on model name
    if args.model_dir == "./models/depth-anything-small-onnx":
        args.model_dir = f"./models/{args.model}-onnx"

    print("=" * 70)
    print(f"Depth Anything, PolyInfer ({args.model})")
    print("=" * 70)

    # Export only
    if args.export:
        export_depth_anything_onnx(args.model, args.model_dir)
        return

    # Benchmark mode
    if args.benchmark:
        benchmark_depth(args.model, args.model_dir)
        return

    # Webcam mode
    if args.webcam:
        run_webcam_depth(args)
        return

    # Image mode
    if args.image:
        run_depth_estimation(args)
        return

    # Default: show usage
    print("\nUsage examples:")
    print("  Single image:  python depth_anything.py --image photo.jpg")
    print("  Webcam:        python depth_anything.py --webcam")
    print("  Benchmark:     python depth_anything.py --benchmark")
    print("  Export:        python depth_anything.py --export")
    print("\nModel sizes:")
    for name in DEPTH_MODELS.keys():
        print(f"  --model {name}")


if __name__ == "__main__":
    main()
