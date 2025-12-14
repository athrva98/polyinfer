#!/usr/bin/env python3
"""Stable Diffusion VAE with PolyInfer.

This example demonstrates:
1. Loading Stable Diffusion's VAE (Variational Autoencoder)
2. Encoding images to latent space
3. Decoding latents back to images
4. Benchmarking VAE performance across backends

The VAE is the component that converts between pixel space and latent space.
It's used in all Stable Diffusion pipelines and is often the bottleneck for
image generation speed.

Models:
- stabilityai/sd-vae-ft-mse: Fine-tuned VAE with MSE loss (better reconstruction)
- stabilityai/sd-vae-ft-ema: Fine-tuned VAE with EMA (smoother outputs)
- runwayml/stable-diffusion-v1-5: Original SD 1.5 VAE
- stabilityai/stable-diffusion-xl-base-1.0: SDXL VAE (larger, higher quality)

Backend Limitations:
- IREE (all devices): Not supported. VAE uses GroupNorm and attention layers that
  IREE's ONNX importer cannot compile. Use ONNX Runtime or OpenVINO instead.
- Recommended: ONNX Runtime (CPU/CUDA/TensorRT), OpenVINO (CPU/Intel-GPU)

Requirements:
    pip install polyinfer[cpu]  # or [nvidia], [intel]
    pip install diffusers transformers pillow

Usage:
    python stable_diffusion_vae.py --encode image.png
    python stable_diffusion_vae.py --decode latents.npy --output decoded.png
    python stable_diffusion_vae.py --roundtrip image.png
    python stable_diffusion_vae.py --benchmark
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


# VAE model configurations
VAE_MODELS = {
    "sd-vae-ft-mse": "stabilityai/sd-vae-ft-mse",
    "sd-vae-ft-ema": "stabilityai/sd-vae-ft-ema",
    "sd-1.5": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}

# Latent scaling factor (SD uses this for numerical stability)
LATENT_SCALE = 0.18215
SDXL_LATENT_SCALE = 0.13025


def load_image(image_path: str, size: int = 512) -> np.ndarray:
    """Load and preprocess image for VAE.

    Args:
        image_path: Path to image file
        size: Target size (must be divisible by 8)

    Returns:
        Preprocessed image tensor (1, 3, H, W) in range [-1, 1]
    """
    try:
        from PIL import Image
    except ImportError:
        print("Please install pillow: pip install pillow")
        raise

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize to target size (center crop to maintain aspect ratio)
    w, h = image.size

    # Resize so smallest side is size
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to square
    w, h = image.size
    left = (w - size) // 2
    top = (h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    # Convert to numpy and normalize to [-1, 1]
    img_array = np.array(image).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0

    # HWC -> CHW -> NCHW
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]

    return img_array.astype(np.float32)


def save_image(tensor: np.ndarray, output_path: str):
    """Save tensor as image.

    Args:
        tensor: Image tensor (1, 3, H, W) or (3, H, W) in range [-1, 1]
        output_path: Output file path
    """
    try:
        from PIL import Image
    except ImportError:
        print("Please install pillow: pip install pillow")
        raise

    # Handle batch dimension
    if tensor.ndim == 4:
        tensor = tensor[0]

    # CHW -> HWC
    img_array = tensor.transpose(1, 2, 0)

    # Denormalize from [-1, 1] to [0, 255]
    img_array = ((img_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    # Save
    Image.fromarray(img_array).save(output_path)
    print(f"Saved: {output_path}")


def export_vae_onnx(model_name: str, output_dir: str) -> tuple[Path, Path]:
    """Export VAE encoder and decoder to ONNX.

    Args:
        model_name: Model name from VAE_MODELS
        output_dir: Directory to save ONNX files

    Returns:
        Paths to (encoder.onnx, decoder.onnx)
    """
    output_path = Path(output_dir)
    encoder_path = output_path / "vae_encoder.onnx"
    decoder_path = output_path / "vae_decoder.onnx"

    if encoder_path.exists() and decoder_path.exists():
        print(f"Model already exists: {output_path}")
        return encoder_path, decoder_path

    print(f"Exporting {model_name} VAE to ONNX...")
    print("This may take a few minutes...")

    try:
        import torch
        from diffusers import AutoencoderKL
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install diffusers torch")
        raise

    hf_model = VAE_MODELS.get(model_name, model_name)

    # Load VAE
    print(f"Loading VAE from {hf_model}...")
    if "vae" in hf_model:
        vae = AutoencoderKL.from_pretrained(hf_model)
    else:
        vae = AutoencoderKL.from_pretrained(hf_model, subfolder="vae")
    vae.eval()

    output_path.mkdir(parents=True, exist_ok=True)

    # Determine image size
    is_sdxl = "xl" in model_name.lower()
    img_size = 1024 if is_sdxl else 512
    latent_size = img_size // 8

    # Export encoder
    print("Exporting encoder...")

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, x):
            return self.vae.encode(x).latent_dist.sample()

    encoder = EncoderWrapper(vae)
    encoder.eval()

    dummy_image = torch.randn(1, 3, img_size, img_size)
    # Use legacy exporter (dynamo=False) - VAE's sample() has random ops
    torch.onnx.export(
        encoder,
        dummy_image,
        str(encoder_path),
        input_names=["sample"],
        output_names=["latent"],
        dynamic_axes={
            "sample": {0: "batch", 2: "height", 3: "width"},
            "latent": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Export decoder
    print("Exporting decoder...")

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, z):
            return self.vae.decode(z).sample

    decoder = DecoderWrapper(vae)
    decoder.eval()

    dummy_latent = torch.randn(1, 4, latent_size, latent_size)
    # Use legacy exporter (dynamo=False) for consistency
    torch.onnx.export(
        decoder,
        dummy_latent,
        str(decoder_path),
        input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 2: "height", 3: "width"},
            "sample": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )

    print(f"Model exported to: {output_path}")
    return encoder_path, decoder_path


class VAEInference:
    """VAE inference wrapper using PolyInfer."""

    def __init__(
        self,
        model_dir: str,
        backend: str = "onnxruntime",
        device: str = "cpu",
        is_sdxl: bool = False,
    ):
        """Initialize VAE inference.

        Args:
            model_dir: Directory containing ONNX models
            backend: Backend to use
            device: Device to use
            is_sdxl: Whether this is SDXL VAE
        """
        model_path = Path(model_dir)
        encoder_path = model_path / "vae_encoder.onnx"
        decoder_path = model_path / "vae_decoder.onnx"

        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(f"VAE models not found in {model_dir}")

        print(f"Loading VAE with {backend}/{device}...")
        self.encoder = pi.load(str(encoder_path), backend=backend, device=device)
        self.decoder = pi.load(str(decoder_path), backend=backend, device=device)
        self.backend_name = self.encoder.backend_name

        self.scale_factor = SDXL_LATENT_SCALE if is_sdxl else LATENT_SCALE

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode image to latent space.

        Args:
            image: Image tensor (1, 3, H, W) in range [-1, 1]

        Returns:
            Latent tensor (1, 4, H/8, W/8)
        """
        latent = self.encoder(image)
        return latent * self.scale_factor

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent to image.

        Args:
            latent: Latent tensor (1, 4, H/8, W/8)

        Returns:
            Image tensor (1, 3, H, W) in range [-1, 1]
        """
        latent = latent / self.scale_factor
        return self.decoder(latent)

    def roundtrip(self, image: np.ndarray) -> np.ndarray:
        """Encode then decode (useful for testing reconstruction quality).

        Args:
            image: Image tensor (1, 3, H, W)

        Returns:
            Reconstructed image tensor (1, 3, H, W)
        """
        latent = self.encode(image)
        return self.decode(latent)


def run_encode(args):
    """Encode image to latent space."""
    model_path = Path(args.model_dir)
    if not (model_path / "vae_encoder.onnx").exists():
        export_vae_onnx(args.model, args.model_dir)

    is_sdxl = "xl" in args.model.lower()
    vae = VAEInference(args.model_dir, args.backend, args.device, is_sdxl)
    print(f"Loaded: {vae.backend_name}")

    # Load and encode image
    size = 1024 if is_sdxl else 512
    image = load_image(args.encode, size)
    print(f"\nInput shape: {image.shape}")

    start = time.perf_counter()
    latent = vae.encode(image)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Latent shape: {latent.shape}")
    print(f"Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
    print(f"Encode time: {elapsed:.2f}ms")

    # Save latent
    if args.output:
        np.save(args.output, latent)
        print(f"Saved latent to: {args.output}")


def run_decode(args):
    """Decode latent to image."""
    model_path = Path(args.model_dir)
    if not (model_path / "vae_decoder.onnx").exists():
        export_vae_onnx(args.model, args.model_dir)

    is_sdxl = "xl" in args.model.lower()
    vae = VAEInference(args.model_dir, args.backend, args.device, is_sdxl)
    print(f"Loaded: {vae.backend_name}")

    # Load latent
    latent = np.load(args.decode)
    print(f"\nLatent shape: {latent.shape}")

    start = time.perf_counter()
    image = vae.decode(latent)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Output shape: {image.shape}")
    print(f"Decode time: {elapsed:.2f}ms")

    # Save image
    output_path = args.output or "./models/outputs/decoded.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output_path)


def run_roundtrip(args):
    """Encode then decode image (test reconstruction)."""
    model_path = Path(args.model_dir)
    if not (model_path / "vae_encoder.onnx").exists():
        export_vae_onnx(args.model, args.model_dir)

    is_sdxl = "xl" in args.model.lower()
    vae = VAEInference(args.model_dir, args.backend, args.device, is_sdxl)
    print(f"Loaded: {vae.backend_name}")

    # Load image
    size = 1024 if is_sdxl else 512
    image = load_image(args.roundtrip, size)
    print(f"\nInput shape: {image.shape}")

    # Roundtrip
    start = time.perf_counter()
    latent = vae.encode(image)
    reconstructed = vae.decode(latent)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Total time: {elapsed:.2f}ms")

    # Compute reconstruction error
    mse = np.mean((image - reconstructed) ** 2)
    psnr = 10 * np.log10(4.0 / mse)  # max value is 2 (range [-1, 1])
    print(f"Reconstruction PSNR: {psnr:.2f} dB")

    # Save comparison
    output_path = args.output or "./models/outputs/roundtrip.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Create side-by-side comparison
    original = ((image[0].transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    recon = ((reconstructed[0].transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    try:
        from PIL import Image
        comparison = Image.new("RGB", (size * 2, size))
        comparison.paste(Image.fromarray(original), (0, 0))
        comparison.paste(Image.fromarray(recon), (size, 0))
        comparison.save(output_path)
        print(f"Saved comparison (left=original, right=reconstructed): {output_path}")
    except ImportError:
        save_image(reconstructed, output_path)


def benchmark_vae(model_name: str, model_dir: str):
    """Benchmark VAE across all backends."""
    model_path = Path(model_dir)

    if not (model_path / "vae_encoder.onnx").exists():
        export_vae_onnx(model_name, model_dir)

    is_sdxl = "xl" in model_name.lower()
    img_size = 1024 if is_sdxl else 512
    latent_size = img_size // 8

    print(f"\n{'='*70}")
    print(f"VAE Benchmark: {model_name} ({img_size}x{img_size})")
    print(f"{'='*70}\n")

    # Prepare inputs
    dummy_image = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    dummy_latent = np.random.randn(1, 4, latent_size, latent_size).astype(np.float32)

    print(f"Image shape: {dummy_image.shape}")
    print(f"Latent shape: {dummy_latent.shape}")
    print(f"Available backends: {pi.list_backends()}")
    print()

    # Test configurations
    # NOTE: IREE excluded - VAE's GroupNorm/attention layers fail ONNX import
    test_configs = [
        # CPU backends
        ("onnxruntime", "cpu"),
        ("openvino", "cpu"),
        # ("iree", "cpu"),  # Not supported - GroupNorm/attention compilation fails
        # NVIDIA GPU
        ("onnxruntime", "cuda"),
        ("onnxruntime", "tensorrt"),
        # ("iree", "cuda"),  # Not supported
        # ("iree", "vulkan"),  # Not supported
        # Intel GPU
        ("openvino", "intel-gpu"),
        # AMD/Intel Windows GPU
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
                str(model_path / "vae_encoder.onnx"),
                backend=backend,
                device=device,
            )
            decoder = pi.load(
                str(model_path / "vae_decoder.onnx"),
                backend=backend,
                device=device,
            )

            # Warmup
            for _ in range(3):
                _ = encoder(dummy_image)
                _ = decoder(dummy_latent)

            # Benchmark encoder
            encode_times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = encoder(dummy_image)
                encode_times.append((time.perf_counter() - start) * 1000)

            # Benchmark decoder
            decode_times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = decoder(dummy_latent)
                decode_times.append((time.perf_counter() - start) * 1000)

            encode_ms = np.mean(encode_times)
            decode_ms = np.mean(decode_times)

            results.append({
                "backend": encoder.backend_name,
                "encode_ms": encode_ms,
                "decode_ms": decode_ms,
            })

            print(f"  {encoder.backend_name:<25} Encode: {encode_ms:>7.2f}ms  "
                  f"Decode: {decode_ms:>7.2f}ms")

        except Exception as e:
            print(f"  {backend}/{device}: Error - {e}")

    print("-" * 70)

    if results:
        # Sort by decode time (most common operation in SD)
        results.sort(key=lambda x: x["decode_ms"])

        print(f"\n{'='*70}")
        print("RESULTS (sorted by decode time)")
        print(f"{'='*70}")
        print(f"{'Backend':<25} {'Encode':>10} {'Decode':>10} {'Speedup':>10}")
        print("-" * 70)

        baseline = results[-1]["decode_ms"]
        for r in results:
            speedup = baseline / r["decode_ms"]
            print(f"{r['backend']:<25} {r['encode_ms']:>8.2f}ms {r['decode_ms']:>8.2f}ms "
                  f"{speedup:>9.1f}x")

        print("-" * 70)
        print(f"\nFastest decode: {results[0]['backend']} ({results[0]['decode_ms']:.2f}ms)")


def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion VAE with PolyInfer"
    )
    parser.add_argument(
        "--model",
        default="sd-vae-ft-mse",
        choices=list(VAE_MODELS.keys()),
        help="VAE model variant",
    )
    parser.add_argument(
        "--model-dir",
        default="./models/sd-vae-ft-mse-onnx",
        help="Directory for ONNX model",
    )
    parser.add_argument(
        "--encode",
        type=str,
        help="Image to encode to latent space",
    )
    parser.add_argument(
        "--decode",
        type=str,
        help="Latent .npy file to decode to image",
    )
    parser.add_argument(
        "--roundtrip",
        type=str,
        help="Image for encode->decode roundtrip test",
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
    if args.model_dir == "./models/sd-vae-ft-mse-onnx":
        args.model_dir = f"./models/{args.model}-onnx"

    print("=" * 70)
    print(f"Stable Diffusion VAE - PolyInfer ({args.model})")
    print("=" * 70)

    # Export only
    if args.export:
        export_vae_onnx(args.model, args.model_dir)
        return

    # Benchmark mode
    if args.benchmark:
        benchmark_vae(args.model, args.model_dir)
        return

    # Encode mode
    if args.encode:
        run_encode(args)
        return

    # Decode mode
    if args.decode:
        run_decode(args)
        return

    # Roundtrip mode
    if args.roundtrip:
        run_roundtrip(args)
        return

    # Default: show usage
    print("\nUsage examples:")
    print("  Encode:    python stable_diffusion_vae.py --encode image.png --output latent.npy")
    print("  Decode:    python stable_diffusion_vae.py --decode latent.npy --output decoded.png")
    print("  Roundtrip: python stable_diffusion_vae.py --roundtrip image.png")
    print("  Benchmark: python stable_diffusion_vae.py --benchmark")
    print("  Export:    python stable_diffusion_vae.py --export")


if __name__ == "__main__":
    main()
