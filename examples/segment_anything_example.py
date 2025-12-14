#!/usr/bin/env python3
"""Segment Anything Model (SAM) with PolyInfer.

This example demonstrates:
1. Loading Meta's Segment Anything Model
2. Generating image embeddings with the encoder
3. Generating masks with point/box prompts
4. Benchmarking SAM performance across backends

SAM is a foundation model for image segmentation that can segment any object
in an image given prompts (points, boxes, or masks).

Models:
- sam-vit-base: Fastest, 91M params
- sam-vit-large: Balanced, 308M params
- sam-vit-huge: Best quality, 636M params

Requirements:
    pip install polyinfer[cpu]  # or [nvidia], [intel]
    pip install segment-anything pillow opencv-python

Model weights download:
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Usage:
    python segment_anything_example.py --image photo.jpg --point 500,300
    python segment_anything_example.py --image photo.jpg --box 100,100,400,400
    python segment_anything_example.py --benchmark
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


# SAM model configurations
SAM_MODELS = {
    "sam-vit-base": {
        "checkpoint": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "model_type": "vit_b",
        "embed_dim": 256,
        "image_size": 1024,
    },
    "sam-vit-large": {
        "checkpoint": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "model_type": "vit_l",
        "embed_dim": 256,
        "image_size": 1024,
    },
    "sam-vit-huge": {
        "checkpoint": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "model_type": "vit_h",
        "embed_dim": 256,
        "image_size": 1024,
    },
}


def load_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Load image for SAM.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (image_array in RGB, original_size)
    """
    try:
        from PIL import Image
    except ImportError:
        print("Please install pillow: pip install pillow")
        raise

    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (W, H)
    return np.array(image), (image.size[1], image.size[0])  # (H, W)


def preprocess_image(image: np.ndarray, target_size: int = 1024) -> tuple[np.ndarray, dict]:
    """Preprocess image for SAM encoder.

    Args:
        image: RGB image array (H, W, 3)
        target_size: Target longest side

    Returns:
        Tuple of (preprocessed tensor, transform info)
    """
    h, w = image.shape[:2]

    # Resize to target size (longest side)
    scale = target_size / max(h, w)
    new_h = int(h * scale + 0.5)
    new_w = int(w * scale + 0.5)

    try:
        import cv2
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        resized = np.array(pil_img)

    # Pad to square
    padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized

    # Normalize with SAM stats
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    padded = (padded - mean) / std

    # HWC -> CHW -> NCHW
    tensor = padded.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    transform_info = {
        "original_size": (h, w),
        "resized_size": (new_h, new_w),
        "scale": scale,
    }

    return tensor, transform_info


def export_sam_onnx(model_name: str, output_dir: str, checkpoint_path: str | None = None) -> tuple[Path, Path]:
    """Export SAM encoder and decoder to ONNX.

    Args:
        model_name: Model name from SAM_MODELS
        output_dir: Directory to save ONNX files
        checkpoint_path: Path to SAM checkpoint (downloads if not provided)

    Returns:
        Paths to (encoder.onnx, decoder.onnx)
    """
    output_path = Path(output_dir)
    encoder_path = output_path / "sam_encoder.onnx"
    decoder_path = output_path / "sam_decoder.onnx"

    if encoder_path.exists() and decoder_path.exists():
        print(f"Model already exists: {output_path}")
        return encoder_path, decoder_path

    print(f"Exporting {model_name} to ONNX...")
    print("This may take several minutes...")

    model_config = SAM_MODELS[model_name]

    try:
        import torch
        from segment_anything import sam_model_registry
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install segment-anything torch")
        raise

    # Find or download checkpoint
    if checkpoint_path is None:
        checkpoint_path = model_config["checkpoint"]

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Please download from: {model_config['url']}")
        print(f"wget {model_config['url']}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load SAM model
    print(f"Loading SAM from {checkpoint_path}...")
    sam = sam_model_registry[model_config["model_type"]](checkpoint=checkpoint_path)
    sam.eval()

    output_path.mkdir(parents=True, exist_ok=True)

    # Export image encoder
    print("Exporting image encoder (this takes a while)...")

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, sam):
            super().__init__()
            self.image_encoder = sam.image_encoder

        def forward(self, x):
            return self.image_encoder(x)

    encoder = EncoderWrapper(sam)
    encoder.eval()

    dummy_image = torch.randn(1, 3, 1024, 1024)
    # Use legacy exporter (dynamo=False) for better SAM compatibility
    torch.onnx.export(
        encoder,
        dummy_image,
        str(encoder_path),
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=17,
        dynamo=False,
    )

    # Export prompt encoder + mask decoder
    print("Exporting prompt encoder and mask decoder...")

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, sam):
            super().__init__()
            self.prompt_encoder = sam.prompt_encoder
            self.mask_decoder = sam.mask_decoder

        def forward(self, image_embeddings, point_coords, point_labels):
            # Encode prompts
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )

            # Decode masks
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            return low_res_masks, iou_predictions

    decoder = DecoderWrapper(sam)
    decoder.eval()

    # Dummy inputs
    dummy_embeddings = torch.randn(1, 256, 64, 64)
    dummy_point_coords = torch.randint(0, 1024, (1, 1, 2)).float()
    dummy_point_labels = torch.ones(1, 1).long()

    # Use legacy exporter (dynamo=False) for better SAM compatibility
    torch.onnx.export(
        decoder,
        (dummy_embeddings, dummy_point_coords, dummy_point_labels),
        str(decoder_path),
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        },
        opset_version=17,
        dynamo=False,
    )

    print(f"Model exported to: {output_path}")
    return encoder_path, decoder_path


def postprocess_masks(
    masks: np.ndarray,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> np.ndarray:
    """Postprocess SAM masks to original image size.

    Args:
        masks: Low-res masks from decoder (B, N, 256, 256)
        input_size: Size after preprocessing (H, W)
        original_size: Original image size (H, W)

    Returns:
        High-res masks at original size
    """
    try:
        import cv2
    except ImportError:
        print("Please install opencv: pip install opencv-python")
        raise

    # Upsample to 1024x1024
    masks_1024 = []
    for mask in masks[0]:
        upsampled = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        masks_1024.append(upsampled)
    masks_1024 = np.array(masks_1024)

    # Crop to input size
    h, w = input_size
    masks_cropped = masks_1024[:, :h, :w]

    # Resize to original size
    oh, ow = original_size
    masks_final = []
    for mask in masks_cropped:
        resized = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_LINEAR)
        masks_final.append(resized > 0)

    return np.array(masks_final)


class SAMInference:
    """SAM inference wrapper using PolyInfer."""

    def __init__(
        self,
        model_dir: str,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        """Initialize SAM inference.

        Args:
            model_dir: Directory containing ONNX models
            backend: Backend to use
            device: Device to use
        """
        model_path = Path(model_dir)
        encoder_path = model_path / "sam_encoder.onnx"
        decoder_path = model_path / "sam_decoder.onnx"

        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(f"SAM models not found in {model_dir}")

        print(f"Loading SAM with {backend}/{device}...")
        self.encoder = pi.load(str(encoder_path), backend=backend, device=device)
        self.decoder = pi.load(str(decoder_path), backend=backend, device=device)
        self.backend_name = self.encoder.backend_name

        self._image_embeddings = None
        self._transform_info = None

    def set_image(self, image: np.ndarray) -> float:
        """Encode image (run once per image).

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            Encoding time in ms
        """
        tensor, self._transform_info = preprocess_image(image)

        start = time.perf_counter()
        self._image_embeddings = self.encoder(tensor)
        elapsed = (time.perf_counter() - start) * 1000

        return elapsed

    def predict_point(
        self,
        point: tuple[int, int],
        label: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Predict masks from a point prompt.

        Args:
            point: (x, y) coordinates
            label: 1 for foreground, 0 for background

        Returns:
            Tuple of (masks, iou_scores, inference_time_ms)
        """
        if self._image_embeddings is None:
            raise RuntimeError("Call set_image() first")

        # Scale point coordinates
        scale = self._transform_info["scale"]
        point_coords = np.array([[[point[0] * scale, point[1] * scale]]], dtype=np.float32)
        point_labels = np.array([[label]], dtype=np.int64)

        start = time.perf_counter()
        masks, iou_predictions = self.decoder(
            self._image_embeddings,
            point_coords,
            point_labels,
        )
        elapsed = (time.perf_counter() - start) * 1000

        # Postprocess masks
        masks = postprocess_masks(
            masks,
            self._transform_info["resized_size"],
            self._transform_info["original_size"],
        )

        return masks, iou_predictions[0], elapsed

    def predict_box(
        self,
        box: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Predict masks from a box prompt.

        Args:
            box: (x1, y1, x2, y2) coordinates

        Returns:
            Tuple of (masks, iou_scores, inference_time_ms)
        """
        if self._image_embeddings is None:
            raise RuntimeError("Call set_image() first")

        # Scale box coordinates and convert to point format
        # SAM uses box corners as points with labels 2 (top-left) and 3 (bottom-right)
        scale = self._transform_info["scale"]
        x1, y1, x2, y2 = box
        point_coords = np.array([
            [[x1 * scale, y1 * scale], [x2 * scale, y2 * scale]]
        ], dtype=np.float32)
        point_labels = np.array([[2, 3]], dtype=np.int64)

        start = time.perf_counter()
        masks, iou_predictions = self.decoder(
            self._image_embeddings,
            point_coords,
            point_labels,
        )
        elapsed = (time.perf_counter() - start) * 1000

        # Postprocess masks
        masks = postprocess_masks(
            masks,
            self._transform_info["resized_size"],
            self._transform_info["original_size"],
        )

        return masks, iou_predictions[0], elapsed


def visualize_masks(
    image: np.ndarray,
    masks: np.ndarray,
    iou_scores: np.ndarray,
    point: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    output_path: str = "./models/outputs/sam_output.png",
):
    """Visualize segmentation masks on image.

    Args:
        image: Original RGB image
        masks: Predicted masks (N, H, W)
        iou_scores: IoU scores for each mask
        point: Optional point prompt to draw
        box: Optional box prompt to draw
        output_path: Output file path
    """
    try:
        import cv2
    except ImportError:
        print("Please install opencv: pip install opencv-python")
        raise

    # Use the mask with highest IoU
    best_idx = np.argmax(iou_scores)
    mask = masks[best_idx]

    # Create colored overlay
    overlay = image.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([30, 144, 255]) * 0.5

    # Draw prompt
    if point is not None:
        cv2.circle(overlay, point, 10, (0, 255, 0), -1)
        cv2.circle(overlay, point, 10, (255, 255, 255), 2)

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Add IoU score
    cv2.putText(
        overlay,
        f"IoU: {iou_scores[best_idx]:.3f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")


def run_segmentation(args):
    """Run segmentation on an image."""
    model_path = Path(args.model_dir)
    if not (model_path / "sam_encoder.onnx").exists():
        export_sam_onnx(args.model, args.model_dir, args.checkpoint)

    sam = SAMInference(args.model_dir, args.backend, args.device)
    print(f"Loaded: {sam.backend_name}")

    # Load image
    image, original_size = load_image(args.image)
    print(f"\nImage size: {original_size}")

    # Encode image
    encode_time = sam.set_image(image)
    print(f"Encoding time: {encode_time:.2f}ms")

    # Predict masks
    if args.point:
        point = tuple(map(int, args.point.split(",")))
        masks, iou_scores, decode_time = sam.predict_point(point)
        print(f"Decode time: {decode_time:.2f}ms")
        print(f"IoU scores: {iou_scores}")

        output_path = args.output or "./models/outputs/sam_point_output.png"
        visualize_masks(image, masks, iou_scores, point=point, output_path=output_path)

    elif args.box:
        box = tuple(map(int, args.box.split(",")))
        masks, iou_scores, decode_time = sam.predict_box(box)
        print(f"Decode time: {decode_time:.2f}ms")
        print(f"IoU scores: {iou_scores}")

        output_path = args.output or "./models/outputs/sam_box_output.png"
        visualize_masks(image, masks, iou_scores, box=box, output_path=output_path)

    else:
        # Default: center point
        h, w = original_size
        point = (w // 2, h // 2)
        print(f"Using center point: {point}")

        masks, iou_scores, decode_time = sam.predict_point(point)
        print(f"Decode time: {decode_time:.2f}ms")
        print(f"IoU scores: {iou_scores}")

        output_path = args.output or "./models/outputs/sam_output.png"
        visualize_masks(image, masks, iou_scores, point=point, output_path=output_path)


def benchmark_sam(model_name: str, model_dir: str):
    """Benchmark SAM across all backends."""
    model_path = Path(model_dir)

    if not (model_path / "sam_encoder.onnx").exists():
        print("SAM model not exported. Please run with --export first.")
        return

    print(f"\n{'='*70}")
    print(f"SAM Benchmark: {model_name}")
    print(f"{'='*70}\n")

    # Prepare dummy inputs
    dummy_image = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    dummy_embeddings = np.random.randn(1, 256, 64, 64).astype(np.float32)
    dummy_point_coords = np.random.rand(1, 1, 2).astype(np.float32) * 1024
    dummy_point_labels = np.ones((1, 1), dtype=np.int64)

    print(f"Available backends: {pi.list_backends()}")
    print()

    # Test configurations
    test_configs = [
        ("onnxruntime", "cpu"),
        ("openvino", "cpu"),
        ("iree", "cpu"),
        ("onnxruntime", "cuda"),
        ("onnxruntime", "tensorrt"),
        ("iree", "cuda"),
        ("iree", "vulkan"),
        ("openvino", "intel-gpu"),
        ("onnxruntime", "directml"),
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

            encoder = pi.load(
                str(model_path / "sam_encoder.onnx"),
                backend=backend,
                device=device,
            )
            decoder = pi.load(
                str(model_path / "sam_decoder.onnx"),
                backend=backend,
                device=device,
            )

            # Warmup
            for _ in range(2):
                _ = encoder(dummy_image)
                _ = decoder(dummy_embeddings, dummy_point_coords, dummy_point_labels)

            # Benchmark encoder (run fewer times as it's slow)
            encode_times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = encoder(dummy_image)
                encode_times.append((time.perf_counter() - start) * 1000)

            # Benchmark decoder (run more times as it's fast)
            decode_times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = decoder(dummy_embeddings, dummy_point_coords, dummy_point_labels)
                decode_times.append((time.perf_counter() - start) * 1000)

            encode_ms = np.mean(encode_times)
            decode_ms = np.mean(decode_times)

            results.append({
                "backend": encoder.backend_name,
                "encode_ms": encode_ms,
                "decode_ms": decode_ms,
            })

            print(f"  {encoder.backend_name:<25} Encode: {encode_ms:>8.1f}ms  "
                  f"Decode: {decode_ms:>6.2f}ms")

        except Exception as e:
            print(f"  {backend}/{device}: Error - {e}")

    print("-" * 70)

    if results:
        # Sort by encoder time (main bottleneck)
        results.sort(key=lambda x: x["encode_ms"])

        print(f"\n{'='*70}")
        print("RESULTS (sorted by encode time)")
        print(f"{'='*70}")
        print(f"{'Backend':<25} {'Encode':>12} {'Decode':>10} {'Speedup':>10}")
        print("-" * 70)

        baseline = results[-1]["encode_ms"]
        for r in results:
            speedup = baseline / r["encode_ms"]
            print(f"{r['backend']:<25} {r['encode_ms']:>10.1f}ms {r['decode_ms']:>8.2f}ms "
                  f"{speedup:>9.1f}x")

        print("-" * 70)
        print(f"\nFastest encode: {results[0]['backend']} ({results[0]['encode_ms']:.1f}ms)")
        print("\nNote: Encoder runs once per image, decoder runs per prompt")


def main():
    parser = argparse.ArgumentParser(
        description="Segment Anything Model (SAM) with PolyInfer"
    )
    parser.add_argument(
        "--model",
        default="sam-vit-base",
        choices=list(SAM_MODELS.keys()),
        help="SAM model variant",
    )
    parser.add_argument(
        "--model-dir",
        default="./models/sam-vit-base-onnx",
        help="Directory for ONNX model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to SAM checkpoint (.pth file)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file",
    )
    parser.add_argument(
        "--point",
        type=str,
        help="Point prompt as 'x,y'",
    )
    parser.add_argument(
        "--box",
        type=str,
        help="Box prompt as 'x1,y1,x2,y2'",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path",
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
    if args.model_dir == "./models/sam-vit-base-onnx":
        args.model_dir = f"./models/{args.model}-onnx"

    print("=" * 70)
    print(f"Segment Anything Model - PolyInfer ({args.model})")
    print("=" * 70)

    # Export only
    if args.export:
        export_sam_onnx(args.model, args.model_dir, args.checkpoint)
        return

    # Benchmark mode
    if args.benchmark:
        benchmark_sam(args.model, args.model_dir)
        return

    # Segmentation mode
    if args.image:
        run_segmentation(args)
        return

    # Default: show usage
    print("\nUsage examples:")
    print("  Point prompt: python segment_anything_example.py --image photo.jpg --point 500,300")
    print("  Box prompt:   python segment_anything_example.py --image photo.jpg --box 100,100,400,400")
    print("  Benchmark:    python segment_anything_example.py --benchmark")
    print("  Export:       python segment_anything_example.py --export --checkpoint sam_vit_b.pth")
    print("\nDownload checkpoints from:")
    for name, config in SAM_MODELS.items():
        print(f"  {name}: {config['url']}")


if __name__ == "__main__":
    main()
