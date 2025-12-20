"""Benchmarking and comparison utilities for PolyInfer."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from polyinfer.discovery import get_backend, list_backends


def benchmark(
    model_path: str | Path,
    inputs: tuple[np.ndarray, ...] | np.ndarray,
    backend: str,
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 100,
    **kwargs,
) -> dict[str, Any]:
    """Benchmark a model on a specific backend.

    Args:
        model_path: Path to ONNX model
        inputs: Input tensor(s) for inference
        backend: Backend to use
        device: Target device
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        **kwargs: Backend-specific options

    Returns:
        Dictionary with benchmark results
    """
    if isinstance(inputs, np.ndarray):
        inputs = (inputs,)

    try:
        backend_instance = get_backend(backend)
        model = backend_instance.load(str(model_path), device=device, **kwargs)

        # Warmup
        for _ in range(warmup):
            model(*inputs)

        # Benchmark
        times_list: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(*inputs)
            elapsed = (time.perf_counter() - start) * 1000
            times_list.append(elapsed)

        times = np.array(times_list)
        return {
            "backend": backend,
            "device": device,
            "status": "success",
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "p90_ms": float(np.percentile(times, 90)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000 / np.mean(times)),
            "iterations": iterations,
        }
    except Exception as e:
        return {
            "backend": backend,
            "device": device,
            "status": "error",
            "error": str(e),
        }


def compare(
    model_path: str | Path,
    inputs: tuple[np.ndarray, ...] | np.ndarray | None = None,
    input_shape: tuple[int, ...] | None = None,
    backends: list[str] | None = None,
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 100,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Compare model performance across multiple backends.

    Args:
        model_path: Path to ONNX model
        inputs: Input tensor(s) for inference (auto-generated if None)
        input_shape: Shape to use for auto-generated inputs
        backends: List of backends to compare (None = all available)
        device: Target device
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        verbose: Print results as they're collected

    Returns:
        List of benchmark results for each backend

    Example:
        >>> results = pi.compare("yolov8n.onnx", input_shape=(1, 3, 640, 640))
        Backend Comparison for yolov8n.onnx
        ============================================================
        openvino-cpu          :  15.50 ms ( 64.5 FPS) <-- FASTEST
        onnxruntime-cpu       :  17.61 ms ( 56.8 FPS)
    """
    model_path = Path(model_path)

    # Generate inputs if not provided
    if inputs is None:
        if input_shape is None:
            # Try to get shape from model
            import onnx

            model = onnx.load(str(model_path))
            input_info = model.graph.input[0]
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)  # Replace dynamic dims with 1
            input_shape = tuple(shape)

        inputs = np.random.rand(*input_shape).astype(np.float32)

    if isinstance(inputs, np.ndarray):
        inputs = (inputs,)

    # Get backends to test
    if backends is None:
        backends = list_backends(available_only=True)

    # Filter to backends that support the device
    device_type = device.split(":")[0] if ":" in device else device
    valid_backends = []
    for name in backends:
        try:
            backend = get_backend(name)
            if backend.supports_device(device_type):
                valid_backends.append(name)
        except Exception:
            continue

    if verbose:
        print(f"\nBackend Comparison for {model_path.name}")
        print(f"Device: {device}")
        print("=" * 60)

    # Run benchmarks
    results = []
    for backend_name in valid_backends:
        if verbose:
            print(f"Testing {backend_name}...", end=" ", flush=True)

        result = benchmark(
            model_path,
            inputs,
            backend=backend_name,
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
        results.append(result)

        if verbose:
            if result["status"] == "success":
                print(f"{result['mean_ms']:.2f} ms ({result['fps']:.1f} FPS)")
            else:
                print(f"ERROR: {result['error']}")

    # Sort by speed
    successful = [r for r in results if r["status"] == "success"]
    successful.sort(key=lambda r: r["mean_ms"])

    if verbose and successful:
        print()
        print("Results (sorted by speed):")
        print("-" * 60)
        fastest = successful[0]["mean_ms"]
        for r in successful:
            marker = " <-- FASTEST" if r["mean_ms"] == fastest else ""
            slowdown = f" ({r['mean_ms'] / fastest:.2f}x)" if r["mean_ms"] != fastest else ""
            print(
                f"{r['backend']:25s}: {r['mean_ms']:6.2f} ms ({r['fps']:5.1f} FPS){slowdown}{marker}"
            )

    return results


def compare_all_devices(
    model_path: str | Path,
    input_shape: tuple[int, ...],
    warmup: int = 10,
    iterations: int = 100,
) -> dict[str, list[dict]]:
    """Compare model across all available backends and devices.

    Args:
        model_path: Path to ONNX model
        input_shape: Shape of input tensor
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        Dictionary mapping device to list of benchmark results
    """
    from polyinfer.discovery import list_devices

    results = {}
    inputs = np.random.rand(*input_shape).astype(np.float32)

    for device_info in list_devices():
        device = device_info.name
        print(f"\n{'=' * 60}")
        print(f"Device: {device}")
        print(f"{'=' * 60}")

        device_results = []
        for backend_name in device_info.backends:
            result = benchmark(
                model_path,
                inputs,
                backend=backend_name,
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
            device_results.append(result)

            if result["status"] == "success":
                print(f"  {backend_name}: {result['mean_ms']:.2f} ms ({result['fps']:.1f} FPS)")
            else:
                print(f"  {backend_name}: ERROR - {result['error']}")

        results[device] = device_results

    return results
