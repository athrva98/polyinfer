#!/usr/bin/env python3
"""Whisper Audio Transcription with PolyInfer.

This example demonstrates:
1. Loading OpenAI's Whisper model for speech-to-text
2. Transcribing audio files across different backends
3. Benchmarking transcription speed

Models:
- whisper-tiny: 39M params, fastest, lower accuracy
- whisper-base: 74M params, good balance
- whisper-small: 244M params, better accuracy
- whisper-medium: 769M params, high accuracy
- whisper-large-v3: 1.5B params, best accuracy

Requirements:
    pip install polyinfer[cpu]  # or [nvidia], [intel]
    pip install optimum[onnxruntime] librosa soundfile

Export model to ONNX (one-time):
    optimum-cli export onnx --model openai/whisper-tiny whisper-tiny-onnx/
    optimum-cli export onnx --model openai/whisper-base whisper-base-onnx/

Usage:
    python whisper_transcription.py --audio speech.wav
    python whisper_transcription.py --audio speech.mp3 --model whisper-base
    python whisper_transcription.py --benchmark
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


# Whisper model configurations
WHISPER_MODELS = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large-v3": "openai/whisper-large-v3",
}

# Sample rate for Whisper
SAMPLE_RATE = 16000
N_MELS = 80
CHUNK_LENGTH = 30  # seconds


def load_audio(audio_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load and preprocess audio file.

    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
        sr: Target sample rate

    Returns:
        Audio waveform as numpy array
    """
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        print("Please install librosa: pip install librosa")
        raise


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute log-mel spectrogram for Whisper.

    Args:
        audio: Audio waveform
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        Log-mel spectrogram (1, n_mels, time_frames)
    """
    try:
        import librosa
    except ImportError:
        print("Please install librosa: pip install librosa")
        raise

    # Pad or trim to 30 seconds
    target_length = CHUNK_LENGTH * sr
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=8000,
    )

    # Convert to log scale
    log_mel = np.log10(np.clip(mel, a_min=1e-10, a_max=None))

    # Normalize
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    # Add batch dimension
    return log_mel[np.newaxis, ...].astype(np.float32)


def export_whisper_onnx(model_name: str, output_dir: str) -> Path:
    """Export Whisper model to ONNX format.

    Args:
        model_name: Model name (e.g., "whisper-tiny")
        output_dir: Directory to save ONNX model

    Returns:
        Path to the encoder ONNX model
    """
    output_path = Path(output_dir)

    # Check if already exported
    encoder_path = output_path / "encoder_model.onnx"
    if encoder_path.exists():
        print(f"Model already exists: {output_path}")
        return encoder_path

    print(f"Exporting {model_name} to ONNX...")
    print("This may take a few minutes...")

    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor

        hf_model = WHISPER_MODELS.get(model_name, model_name)

        # Export model
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            hf_model,
            export=True,
        )
        model.save_pretrained(output_path)

        # Also save processor
        processor = WhisperProcessor.from_pretrained(hf_model)
        processor.save_pretrained(output_path)

        print(f"Model exported to: {output_path}")
        return encoder_path

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install optimum[onnxruntime] transformers")
        raise


def transcribe_with_transformers(
    audio_path: str,
    model_dir: str,
    backend: str = "onnxruntime",
    device: str = "cpu",
) -> tuple[str, float]:
    """Transcribe audio using transformers pipeline with PolyInfer backend.

    This uses the full Whisper pipeline including decoder for accurate results.

    Args:
        audio_path: Path to audio file
        model_dir: Directory containing ONNX model
        backend: Backend to use
        device: Device to use

    Returns:
        Tuple of (transcription, inference_time_ms)
    """
    try:
        from transformers import WhisperProcessor, pipeline
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    except ImportError:
        print("Please install: pip install optimum[onnxruntime] transformers")
        raise

    model_path = Path(model_dir)

    # Load model and processor
    print(f"Loading model from {model_path}...")

    # For ONNX Runtime, we use optimum's pipeline
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )

    # Transcribe
    start = time.perf_counter()
    result = pipe(audio_path, return_timestamps=True)
    elapsed = (time.perf_counter() - start) * 1000

    return result["text"], elapsed


def transcribe_encoder_only(
    audio_path: str,
    model_dir: str,
    backend: str = "onnxruntime",
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Run just the encoder for benchmarking purposes.

    This doesn't produce text output but benchmarks the encoder speed.

    Args:
        audio_path: Path to audio file
        model_dir: Directory containing ONNX model
        backend: Backend to use
        device: Device to use

    Returns:
        Tuple of (encoder_output, inference_time_ms)
    """
    model_path = Path(model_dir)
    encoder_path = model_path / "encoder_model.onnx"

    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    # Load audio and compute mel spectrogram
    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    mel = compute_mel_spectrogram(audio)

    print(f"Mel spectrogram shape: {mel.shape}")

    # Load encoder with PolyInfer
    print(f"Loading encoder with {backend}/{device}...")
    model = pi.load(str(encoder_path), backend=backend, device=device)
    print(f"Loaded: {model}")

    # Warmup
    for _ in range(3):
        _ = model(mel)

    # Inference
    start = time.perf_counter()
    output = model(mel)
    elapsed = (time.perf_counter() - start) * 1000

    return output, elapsed


def benchmark_whisper(model_name: str, model_dir: str, audio_path: str | None = None):
    """Benchmark Whisper encoder across all backends.

    Args:
        model_name: Model name
        model_dir: Directory containing ONNX model
        audio_path: Optional audio file (uses dummy input if not provided)
    """
    model_path = Path(model_dir)
    encoder_path = model_path / "encoder_model.onnx"

    if not encoder_path.exists():
        export_whisper_onnx(model_name, model_dir)

    print(f"\n{'='*70}")
    print(f"Whisper Encoder Benchmark: {model_name}")
    print(f"{'='*70}\n")

    # Prepare input
    if audio_path:
        audio = load_audio(audio_path)
        mel = compute_mel_spectrogram(audio)
    else:
        # Dummy mel spectrogram (30 seconds of audio)
        mel = np.random.randn(1, N_MELS, 3000).astype(np.float32)

    print(f"Input shape: {mel.shape}")
    print(f"Available backends: {pi.list_backends()}")
    print()

    # Test configurations
    test_configs = [
        ("onnxruntime", "cpu"),
        ("openvino", "cpu"),
        ("onnxruntime", "cuda"),
        ("onnxruntime", "tensorrt"),
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

            model = pi.load(str(encoder_path), backend=backend, device=device)

            # Warmup
            for _ in range(5):
                _ = model(mel)

            # Benchmark
            times = []
            for _ in range(20):
                start = time.perf_counter()
                _ = model(mel)
                times.append((time.perf_counter() - start) * 1000)

            mean_ms = np.mean(times)
            rtf = mean_ms / (CHUNK_LENGTH * 1000)  # Real-time factor

            results.append({
                "backend": model.backend_name,
                "mean_ms": mean_ms,
                "std_ms": np.std(times),
                "rtf": rtf,
            })

            print(f"  {model.backend_name:<30} {mean_ms:>8.2f}ms  (RTF: {rtf:.3f}x)")

        except Exception as e:
            print(f"  {backend}/{device}: Error - {e}")

    print("-" * 70)

    if results:
        results.sort(key=lambda x: x["mean_ms"])

        print(f"\n{'='*70}")
        print("RESULTS (sorted by speed)")
        print(f"{'='*70}")
        print(f"{'Backend':<30} {'Latency':>10} {'RTF':>10} {'Speedup':>10}")
        print("-" * 70)

        baseline = results[-1]["mean_ms"]
        for r in results:
            speedup = baseline / r["mean_ms"]
            print(f"{r['backend']:<30} {r['mean_ms']:>8.2f}ms {r['rtf']:>9.3f}x {speedup:>9.1f}x")

        print("-" * 70)
        print(f"\nFastest: {results[0]['backend']} ({results[0]['mean_ms']:.2f}ms)")
        print(f"RTF < 1.0 means faster than real-time")


def main():
    parser = argparse.ArgumentParser(
        description="Whisper Audio Transcription with PolyInfer"
    )
    parser.add_argument(
        "--model",
        default="whisper-tiny",
        choices=list(WHISPER_MODELS.keys()),
        help="Whisper model size",
    )
    parser.add_argument(
        "--model-dir",
        default="./models/whisper-tiny-onnx",
        help="Directory for ONNX model",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file to transcribe",
    )
    parser.add_argument(
        "--backend",
        default="onnxruntime",
        help="Backend: onnxruntime, openvino",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu, cuda, directml",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark encoder across all backends",
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Run encoder only (for benchmarking)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export model to ONNX and exit",
    )

    args = parser.parse_args()

    # Update model directory based on model name
    if args.model_dir == "./models/whisper-tiny-onnx":
        args.model_dir = f"./models/{args.model}-onnx"

    print("=" * 70)
    print(f"Whisper Transcription - PolyInfer ({args.model})")
    print("=" * 70)

    # Export only
    if args.export:
        export_whisper_onnx(args.model, args.model_dir)
        return

    # Benchmark mode
    if args.benchmark:
        # Ensure model is exported
        encoder_path = Path(args.model_dir) / "encoder_model.onnx"
        if not encoder_path.exists():
            export_whisper_onnx(args.model, args.model_dir)

        benchmark_whisper(args.model, args.model_dir, args.audio)
        return

    # Transcription mode
    if args.audio:
        # Ensure model is exported
        encoder_path = Path(args.model_dir) / "encoder_model.onnx"
        if not encoder_path.exists():
            export_whisper_onnx(args.model, args.model_dir)

        if args.encoder_only:
            # Encoder-only benchmark
            output, elapsed = transcribe_encoder_only(
                args.audio, args.model_dir, args.backend, args.device
            )
            print(f"\nEncoder output shape: {output.shape}")
            print(f"Inference time: {elapsed:.2f}ms")
        else:
            # Full transcription
            print(f"\nTranscribing: {args.audio}")
            text, elapsed = transcribe_with_transformers(
                args.audio, args.model_dir, args.backend, args.device
            )
            print(f"\nTranscription ({elapsed:.0f}ms):")
            print("-" * 40)
            print(text)
            print("-" * 40)
    else:
        print("\nUsage:")
        print("  Transcribe: python whisper_transcription.py --audio speech.wav")
        print("  Benchmark:  python whisper_transcription.py --benchmark")
        print("  Export:     python whisper_transcription.py --export")


if __name__ == "__main__":
    main()
