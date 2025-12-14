"""ResNet Image Classification with PolyInfer.

This example demonstrates:
1. Loading a pre-trained ResNet model
2. Classifying images from file or webcam
3. Comparing inference speed across backends

Requirements:
    pip install torch torchvision opencv-python pillow requests

Run: python examples/resnet_classification.py
     python examples/resnet_classification.py --image path/to/image.jpg
     python examples/resnet_classification.py --webcam
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

import polyinfer as pi

# ImageNet class labels (abbreviated - first 20 for demo)
# Full list: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
IMAGENET_CLASSES = None  # Will be loaded dynamically


def get_imagenet_labels() -> list:
    """Download ImageNet class labels."""
    global IMAGENET_CLASSES
    if IMAGENET_CLASSES is not None:
        return IMAGENET_CLASSES

    try:
        import requests

        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url, timeout=10)
        IMAGENET_CLASSES = response.text.strip().split("\n")
    except Exception:
        # Fallback to numbered classes
        IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]

    return IMAGENET_CLASSES


def export_resnet_onnx(model_name: str = "resnet18") -> str:
    """Export ResNet model to ONNX."""
    output_dir = Path(f"./models/{model_name}-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    if onnx_path.exists():
        print(f"ONNX model exists: {onnx_path}")
        return str(onnx_path)

    print(f"Exporting {model_name} to ONNX...")
    try:
        import torch
        import torchvision.models as models

        # Load model
        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
        }
        model_fn = getattr(models, model_name)
        model = model_fn(weights=weights_map.get(model_name))
        model.eval()

        # Export
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"Exported: {onnx_path}")

    except ImportError:
        print("Please install PyTorch: pip install torch torchvision")
        raise

    return str(onnx_path)


def preprocess_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocess image for ResNet.

    ImageNet preprocessing:
    - Resize to 256, center crop to 224
    - Normalize with ImageNet mean/std
    """
    # Resize (maintain aspect ratio)
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = size + 32, int(w * (size + 32) / h)
    else:
        new_h, new_w = int(h * (size + 32) / w), size + 32
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop
    h, w = image.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    image = image[top : top + size, left : left + size]

    # BGR -> RGB
    image = image[:, :, ::-1]

    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image.astype(np.float32) / 255.0 - mean) / std

    # NHWC -> NCHW
    image = image.transpose(2, 0, 1)[np.newaxis, ...]

    return image.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def classify_image(model, image: np.ndarray, top_k: int = 5) -> list:
    """Classify an image and return top-k predictions."""
    # Preprocess
    tensor = preprocess_image(image)

    # Inference
    start = time.perf_counter()
    output = model(tensor)
    elapsed = (time.perf_counter() - start) * 1000

    # Get probabilities
    probs = softmax(output[0])

    # Get top-k
    labels = get_imagenet_labels()
    top_indices = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "class_id": int(idx),
            "class_name": labels[idx],
            "probability": float(probs[idx]),
        })

    return results, elapsed


def run_image_classification(args):
    """Classify a single image."""
    # Load model
    onnx_path = export_resnet_onnx(args.model)
    model = pi.load(onnx_path, device=args.device, backend=args.backend)
    print(f"Loaded: {model}")

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        return

    print(f"\nClassifying: {args.image}")
    print("-" * 40)

    # Classify
    results, elapsed = classify_image(model, image, top_k=5)

    print(f"Inference time: {elapsed:.2f} ms")
    print("\nTop-5 predictions:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['class_name']:30s} {r['probability']*100:5.2f}%")


def run_webcam_classification(args):
    """Real-time classification from webcam."""
    # Load model
    onnx_path = export_resnet_onnx(args.model)
    model = pi.load(onnx_path, device=args.device, backend=args.backend)
    print(f"Loaded: {model}")

    # Warmup
    dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(5):
        model(dummy)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\nControls: Q=Quit, S=Screenshot")

    frame_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Classify
            results, elapsed = classify_image(model, frame, top_k=3)

            # Track FPS
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_ms = np.mean(frame_times)
            fps = 1000 / avg_ms

            # Draw results
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | {model.backend_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            for i, r in enumerate(results):
                text = f"{r['class_name']}: {r['probability']*100:.1f}%"
                cv2.putText(
                    frame,
                    text,
                    (10, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("ResNet Classification - PolyInfer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_benchmark(args):
    """Benchmark all available backends."""
    onnx_path = export_resnet_onnx(args.model)

    print(f"\nBenchmarking {args.model} across all backends...")
    print("=" * 60)

    pi.compare(
        onnx_path,
        input_shape=(1, 3, 224, 224),
        device=args.device,
        warmup=10,
        iterations=100,
    )


def main():
    parser = argparse.ArgumentParser(description="ResNet Classification with PolyInfer")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, directml")
    parser.add_argument("--backend", default=None, help="Backend: onnxruntime, openvino, tensorrt")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all backends")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    print("=" * 60)
    print("ResNet Image Classification - PolyInfer")
    print("=" * 60)

    if args.benchmark:
        run_benchmark(args)
    elif args.webcam:
        run_webcam_classification(args)
    elif args.image:
        run_image_classification(args)
    else:
        # Default: benchmark
        run_benchmark(args)


if __name__ == "__main__":
    main()
