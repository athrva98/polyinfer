#!/usr/bin/env python3
"""CLIP Image-Text Embeddings with PolyInfer.

This example demonstrates:
1. Loading OpenAI's CLIP model for image-text embeddings
2. Computing image and text embeddings
3. Image-text similarity search
4. Zero-shot image classification
5. Benchmarking across backends

CLIP (Contrastive Language-Image Pre-training) creates aligned embeddings
for images and text, enabling powerful search and classification capabilities.

Models:
- openai/clip-vit-base-patch32: 151M params, fastest
- openai/clip-vit-base-patch16: 151M params, better accuracy
- openai/clip-vit-large-patch14: 428M params, best accuracy

Requirements:
    pip install polyinfer[cpu]  # or [nvidia], [intel]
    pip install transformers pillow

Export model to ONNX (one-time):
    optimum-cli export onnx --model openai/clip-vit-base-patch32 clip-vit-base-patch32-onnx/

Usage:
    python clip_embeddings.py --image photo.jpg --text "a photo of a cat"
    python clip_embeddings.py --image photo.jpg --classify "cat,dog,bird,car"
    python clip_embeddings.py --benchmark
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


# CLIP model configurations
CLIP_MODELS = {
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
    "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
}


def load_image(image_path: str, size: int = 224) -> np.ndarray:
    """Load and preprocess image for CLIP.

    Args:
        image_path: Path to image file
        size: Target size (224 for base models, 336 for large)

    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    try:
        from PIL import Image
    except ImportError:
        print("Please install pillow: pip install pillow")
        raise

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize with center crop
    # First resize so smallest side is size
    w, h = image.size
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Center crop
    w, h = image.size
    left = (w - size) // 2
    top = (h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    # Convert to numpy
    img_array = np.array(image).astype(np.float32) / 255.0

    # Normalize with CLIP stats
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_array = (img_array - mean) / std

    # HWC -> CHW -> NCHW
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]

    return img_array.astype(np.float32)


def tokenize_text(texts: list[str], max_length: int = 77) -> dict:
    """Tokenize text for CLIP.

    Args:
        texts: List of text strings
        max_length: Maximum sequence length

    Returns:
        Dict with input_ids and attention_mask
    """
    try:
        from transformers import CLIPTokenizer
    except ImportError:
        print("Please install transformers: pip install transformers")
        raise

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    encoded = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )

    return {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }


def export_clip_onnx(model_name: str, output_dir: str) -> tuple[Path, Path]:
    """Export CLIP model to ONNX format.

    Args:
        model_name: Model name (e.g., "clip-vit-base-patch32")
        output_dir: Directory to save ONNX model

    Returns:
        Paths to (vision_model, text_model) ONNX files
    """
    output_path = Path(output_dir)

    vision_path = output_path / "vision_model.onnx"
    text_path = output_path / "text_model.onnx"

    if vision_path.exists() and text_path.exists():
        print(f"Model already exists: {output_path}")
        return vision_path, text_path

    print(f"Exporting {model_name} to ONNX...")
    print("This may take a few minutes...")

    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers torch")
        raise

    hf_model = CLIP_MODELS.get(model_name, model_name)

    # Load model
    model = CLIPModel.from_pretrained(hf_model)
    model.eval()

    output_path.mkdir(parents=True, exist_ok=True)

    # Export vision model
    print("Exporting vision model...")
    vision_model = model.vision_model

    class VisionWrapper(torch.nn.Module):
        def __init__(self, vision_model, projection):
            super().__init__()
            self.vision_model = vision_model
            self.projection = projection

        def forward(self, pixel_values):
            outputs = self.vision_model(pixel_values)
            pooled = outputs.pooler_output
            return self.projection(pooled)

    vision_wrapper = VisionWrapper(model.vision_model, model.visual_projection)
    vision_wrapper.eval()

    dummy_image = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        vision_wrapper,
        dummy_image,
        str(vision_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=17,
    )

    # Export text model
    print("Exporting text model...")

    class TextWrapper(torch.nn.Module):
        def __init__(self, text_model, projection):
            super().__init__()
            self.text_model = text_model
            self.projection = projection

        def forward(self, input_ids, attention_mask):
            outputs = self.text_model(input_ids, attention_mask)
            pooled = outputs.pooler_output
            return self.projection(pooled)

    text_wrapper = TextWrapper(model.text_model, model.text_projection)
    text_wrapper.eval()

    dummy_ids = torch.randint(0, 49408, (1, 77))
    dummy_mask = torch.ones(1, 77, dtype=torch.long)
    torch.onnx.export(
        text_wrapper,
        (dummy_ids, dummy_mask),
        str(text_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_embeds": {0: "batch"},
        },
        opset_version=17,
    )

    # Save processor
    processor = CLIPProcessor.from_pretrained(hf_model)
    processor.save_pretrained(output_path)

    print(f"Model exported to: {output_path}")
    return vision_path, text_path


class CLIPInference:
    """CLIP inference wrapper using PolyInfer."""

    def __init__(
        self,
        model_dir: str,
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        """Initialize CLIP inference.

        Args:
            model_dir: Directory containing ONNX models
            backend: Backend to use
            device: Device to use
        """
        model_path = Path(model_dir)
        vision_path = model_path / "vision_model.onnx"
        text_path = model_path / "text_model.onnx"

        if not vision_path.exists() or not text_path.exists():
            raise FileNotFoundError(f"CLIP models not found in {model_dir}")

        print(f"Loading CLIP models with {backend}/{device}...")
        self.vision_model = pi.load(str(vision_path), backend=backend, device=device)
        self.text_model = pi.load(str(text_path), backend=backend, device=device)
        self.backend_name = self.vision_model.backend_name

    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image to embedding.

        Args:
            image_path: Path to image file

        Returns:
            Normalized image embedding (1, 512)
        """
        pixel_values = load_image(image_path)
        embedding = self.vision_model(pixel_values)

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Normalized text embeddings (N, 512)
        """
        tokens = tokenize_text(texts)
        embedding = self.text_model(tokens["input_ids"], tokens["attention_mask"])

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding

    def similarity(self, image_path: str, texts: list[str]) -> np.ndarray:
        """Compute image-text similarity scores.

        Args:
            image_path: Path to image
            texts: List of text descriptions

        Returns:
            Similarity scores (softmax probabilities)
        """
        image_emb = self.encode_image(image_path)
        text_emb = self.encode_text(texts)

        # Cosine similarity (embeddings are already normalized)
        logits = 100.0 * (image_emb @ text_emb.T)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return probs[0]

    def zero_shot_classify(
        self,
        image_path: str,
        class_names: list[str],
        template: str = "a photo of a {}",
    ) -> list[tuple[str, float]]:
        """Zero-shot image classification.

        Args:
            image_path: Path to image
            class_names: List of class names
            template: Text template for classes

        Returns:
            List of (class_name, probability) tuples, sorted by probability
        """
        # Create text prompts
        texts = [template.format(name) for name in class_names]

        # Get similarity scores
        probs = self.similarity(image_path, texts)

        # Sort by probability
        results = list(zip(class_names, probs))
        results.sort(key=lambda x: x[1], reverse=True)

        return results


def run_similarity(args):
    """Compute image-text similarity."""
    # Ensure model exists
    model_path = Path(args.model_dir)
    if not (model_path / "vision_model.onnx").exists():
        export_clip_onnx(args.model, args.model_dir)

    clip = CLIPInference(args.model_dir, args.backend, args.device)
    print(f"Loaded: {clip.backend_name}")

    print(f"\nImage: {args.image}")
    print(f"Text: {args.text}")

    # Compute similarity
    start = time.perf_counter()
    probs = clip.similarity(args.image, [args.text])
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nSimilarity score: {probs[0]:.4f}")
    print(f"Inference time: {elapsed:.2f}ms")


def run_classification(args):
    """Zero-shot classification."""
    # Ensure model exists
    model_path = Path(args.model_dir)
    if not (model_path / "vision_model.onnx").exists():
        export_clip_onnx(args.model, args.model_dir)

    clip = CLIPInference(args.model_dir, args.backend, args.device)
    print(f"Loaded: {clip.backend_name}")

    # Parse class names
    class_names = [c.strip() for c in args.classify.split(",")]

    print(f"\nImage: {args.image}")
    print(f"Classes: {class_names}")
    print("-" * 40)

    # Classify
    start = time.perf_counter()
    results = clip.zero_shot_classify(args.image, class_names)
    elapsed = (time.perf_counter() - start) * 1000

    print("\nPredictions:")
    for name, prob in results:
        bar = "█" * int(prob * 30)
        print(f"  {name:20s} {prob*100:5.1f}% {bar}")

    print(f"\nInference time: {elapsed:.2f}ms")


def benchmark_clip(model_name: str, model_dir: str):
    """Benchmark CLIP across all backends."""
    model_path = Path(model_dir)

    if not (model_path / "vision_model.onnx").exists():
        export_clip_onnx(model_name, model_dir)

    print(f"\n{'='*70}")
    print(f"CLIP Benchmark: {model_name}")
    print(f"{'='*70}\n")

    # Prepare inputs
    dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    dummy_ids = np.random.randint(0, 49408, (1, 77)).astype(np.int64)
    dummy_mask = np.ones((1, 77), dtype=np.int64)

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

    print("Running benchmarks (Vision + Text encoders)...")
    print("-" * 70)

    for backend, device in test_configs:
        if backend not in pi.list_backends():
            continue

        try:
            backend_obj = pi.get_backend(backend)
            if not backend_obj.supports_device(device):
                continue

            vision_model = pi.load(
                str(model_path / "vision_model.onnx"),
                backend=backend,
                device=device,
            )
            text_model = pi.load(
                str(model_path / "text_model.onnx"),
                backend=backend,
                device=device,
            )

            # Warmup
            for _ in range(5):
                _ = vision_model(dummy_image)
                _ = text_model(dummy_ids, dummy_mask)

            # Benchmark vision
            vision_times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = vision_model(dummy_image)
                vision_times.append((time.perf_counter() - start) * 1000)

            # Benchmark text
            text_times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = text_model(dummy_ids, dummy_mask)
                text_times.append((time.perf_counter() - start) * 1000)

            vision_ms = np.mean(vision_times)
            text_ms = np.mean(text_times)
            total_ms = vision_ms + text_ms

            results.append({
                "backend": vision_model.backend_name,
                "vision_ms": vision_ms,
                "text_ms": text_ms,
                "total_ms": total_ms,
            })

            print(f"  {vision_model.backend_name:<25} Vision: {vision_ms:>6.2f}ms  "
                  f"Text: {text_ms:>6.2f}ms  Total: {total_ms:>6.2f}ms")

        except Exception as e:
            print(f"  {backend}/{device}: Error - {e}")

    print("-" * 70)

    if results:
        results.sort(key=lambda x: x["total_ms"])

        print(f"\n{'='*70}")
        print("RESULTS (sorted by total time)")
        print(f"{'='*70}")
        print(f"{'Backend':<25} {'Vision':>10} {'Text':>10} {'Total':>10} {'Speedup':>10}")
        print("-" * 70)

        baseline = results[-1]["total_ms"]
        for r in results:
            speedup = baseline / r["total_ms"]
            print(f"{r['backend']:<25} {r['vision_ms']:>8.2f}ms {r['text_ms']:>8.2f}ms "
                  f"{r['total_ms']:>8.2f}ms {speedup:>9.1f}x")

        print("-" * 70)
        print(f"\nFastest: {results[0]['backend']} ({results[0]['total_ms']:.2f}ms)")


def main():
    parser = argparse.ArgumentParser(
        description="CLIP Image-Text Embeddings with PolyInfer"
    )
    parser.add_argument(
        "--model",
        default="clip-vit-base-patch32",
        choices=list(CLIP_MODELS.keys()),
        help="CLIP model variant",
    )
    parser.add_argument(
        "--model-dir",
        default="./models/clip-vit-base-patch32-onnx",
        help="Directory for ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text description for similarity",
    )
    parser.add_argument(
        "--classify",
        type=str,
        help="Comma-separated class names for zero-shot classification",
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
    if args.model_dir == "./models/clip-vit-base-patch32-onnx":
        args.model_dir = f"./models/{args.model}-onnx"

    print("=" * 70)
    print(f"CLIP Embeddings - PolyInfer ({args.model})")
    print("=" * 70)

    # Export only
    if args.export:
        export_clip_onnx(args.model, args.model_dir)
        return

    # Benchmark mode
    if args.benchmark:
        benchmark_clip(args.model, args.model_dir)
        return

    # Similarity mode
    if args.image and args.text:
        run_similarity(args)
        return

    # Classification mode
    if args.image and args.classify:
        run_classification(args)
        return

    # Default: show usage
    print("\nUsage examples:")
    print("  Similarity:     python clip_embeddings.py --image photo.jpg --text \"a cat\"")
    print("  Classification: python clip_embeddings.py --image photo.jpg --classify \"cat,dog,bird\"")
    print("  Benchmark:      python clip_embeddings.py --benchmark")
    print("  Export:         python clip_embeddings.py --export")


if __name__ == "__main__":
    main()
