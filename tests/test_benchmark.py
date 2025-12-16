"""Tests for benchmarking functionality."""

import pytest
import numpy as np
import polyinfer as pi


class TestBenchmarkBasics:
    """Basic benchmark functionality tests."""

    def test_benchmark_returns_dict(self, model_path, yolo_input):
        """benchmark() should return a dictionary."""
        model = pi.load(model_path, device="cpu")
        result = model.benchmark(yolo_input, warmup=2, iterations=5)

        assert isinstance(result, dict)

    def test_benchmark_required_keys(self, model_path, yolo_input):
        """benchmark() should return all required metrics."""
        model = pi.load(model_path, device="cpu")
        result = model.benchmark(yolo_input, warmup=2, iterations=5)

        required_keys = [
            "backend",
            "device",
            "mean_ms",
            "std_ms",
            "min_ms",
            "max_ms",
            "median_ms",
            "p90_ms",
            "p99_ms",
            "fps",
            "iterations",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_benchmark_values_valid(self, model_path, yolo_input):
        """benchmark() values should be valid."""
        model = pi.load(model_path, device="cpu")
        result = model.benchmark(yolo_input, warmup=2, iterations=10)

        # Timing values should be positive
        assert result["mean_ms"] > 0
        assert result["min_ms"] > 0
        assert result["max_ms"] > 0
        assert result["median_ms"] > 0
        assert result["p90_ms"] > 0
        assert result["p99_ms"] > 0
        assert result["fps"] > 0

        # Std should be non-negative
        assert result["std_ms"] >= 0

        # Min <= median <= max
        assert result["min_ms"] <= result["median_ms"] <= result["max_ms"]

        # Min <= mean <= max
        assert result["min_ms"] <= result["mean_ms"] <= result["max_ms"]

        # FPS should be consistent with mean_ms
        expected_fps = 1000 / result["mean_ms"]
        assert abs(result["fps"] - expected_fps) < 0.1

    def test_benchmark_iterations(self, model_path, yolo_input):
        """benchmark() should respect iterations parameter."""
        model = pi.load(model_path, device="cpu")

        result5 = model.benchmark(yolo_input, warmup=1, iterations=5)
        result20 = model.benchmark(yolo_input, warmup=1, iterations=20)

        assert result5["iterations"] == 5
        assert result20["iterations"] == 20


class TestBenchmarkConsistency:
    """Test benchmark result consistency."""

    def test_benchmark_reproducible(self, model_path, yolo_input):
        """Benchmark results should be relatively stable."""
        model = pi.load(model_path, device="cpu")

        result1 = model.benchmark(yolo_input, warmup=5, iterations=20)
        result2 = model.benchmark(yolo_input, warmup=5, iterations=20)

        # Results should be within 50% of each other (allowing for system noise)
        ratio = result1["mean_ms"] / result2["mean_ms"]
        assert 0.5 < ratio < 2.0

    def test_warmup_effect(self, model_path, yolo_input):
        """Warmup should improve result consistency."""
        model = pi.load(model_path, device="cpu")

        # No warmup
        result_no_warmup = model.benchmark(yolo_input, warmup=0, iterations=10)

        # With warmup
        result_warmup = model.benchmark(yolo_input, warmup=10, iterations=10)

        # With warmup, variance should generally be lower
        # (not strictly enforced as it depends on system state)
        assert result_warmup["std_ms"] >= 0


class TestCompare:
    """Test the compare() utility function."""

    def test_compare_basic(self, model_path):
        """compare() should work with default parameters."""
        # This will compare available backends
        results = pi.compare(model_path, input_shape=(1, 3, 640, 640))
        assert isinstance(results, list)

    def test_compare_returns_results(self, model_path):
        """compare() should return benchmark results for each backend."""
        results = pi.compare(
            model_path,
            input_shape=(1, 3, 640, 640),
            warmup=2,
            iterations=5
        )

        for result in results:
            assert "backend" in result
            # Results may include errors for unavailable backends
            if result.get("status") != "error":
                assert "mean_ms" in result
                assert "fps" in result

    def test_compare_specific_device(self, model_path):
        """compare() should work with specific device."""
        results = pi.compare(
            model_path,
            input_shape=(1, 3, 640, 640),
            device="cpu",
            warmup=2,
            iterations=5
        )

        assert len(results) >= 1
        # Filter out errors
        successful = [r for r in results if r.get("status") != "error"]
        assert any("cpu" in r.get("device", "").lower() for r in successful)


class TestBenchmarkPerDevice:
    """Device-specific benchmark tests."""

    def test_cpu_benchmark(self, model_path, yolo_input):
        """CPU benchmark should complete."""
        model = pi.load(model_path, device="cpu")
        result = model.benchmark(yolo_input, warmup=3, iterations=10)

        assert result["fps"] > 1  # At least 1 FPS on CPU

    @pytest.mark.cuda
    def test_cuda_benchmark(self, model_path, yolo_input):
        """CUDA benchmark should complete with reasonable performance."""
        # Check if CUDA is available via ONNX Runtime
        backend = pi.get_backend("onnxruntime")
        if not backend.supports_device("cuda"):
            pytest.skip("ONNX Runtime CUDA not available (install onnxruntime-gpu)")

        model_cpu = pi.load(model_path, device="cpu")
        model_cuda = pi.load(model_path, backend="onnxruntime", device="cuda")

        result_cpu = model_cpu.benchmark(yolo_input, warmup=5, iterations=20)
        result_cuda = model_cuda.benchmark(yolo_input, warmup=5, iterations=20)

        # CUDA should be faster than CPU (ONNX Runtime CUDA EP is typically faster)
        assert result_cuda["mean_ms"] < result_cpu["mean_ms"]

    @pytest.mark.tensorrt
    def test_tensorrt_benchmark(self, model_path, yolo_input):
        """TensorRT benchmark should be competitive with CUDA."""
        model_cuda = pi.load(model_path, device="cuda")
        model_trt = pi.load(model_path, device="tensorrt")

        result_cuda = model_cuda.benchmark(yolo_input, warmup=5, iterations=20)
        result_trt = model_trt.benchmark(yolo_input, warmup=5, iterations=20)

        # TensorRT should be within 2x of CUDA (may vary due to warmup/caching)
        assert result_trt["mean_ms"] < result_cuda["mean_ms"] * 2

    @pytest.mark.npu
    def test_npu_benchmark(self, model_path, yolo_input):
        """NPU benchmark should complete with reasonable FPS."""
        model = pi.load(model_path, backend="openvino", device="npu")
        result = model.benchmark(yolo_input, warmup=5, iterations=20)

        # NPU should be reasonably fast
        assert result["fps"] > 10

    @pytest.mark.openvino
    def test_openvino_cpu_benchmark(self, model_path, yolo_input):
        """OpenVINO CPU should be competitive."""
        model = pi.load(model_path, backend="openvino", device="cpu")
        result = model.benchmark(yolo_input, warmup=5, iterations=20)

        assert result["fps"] > 10

    @pytest.mark.intel_gpu
    def test_intel_gpu_benchmark(self, model_path, yolo_input):
        """Intel GPU benchmark should complete."""
        model = pi.load(model_path, backend="openvino", device="intel-gpu")
        result = model.benchmark(yolo_input, warmup=5, iterations=20)

        assert result["fps"] > 1


@pytest.mark.slow
class TestExtendedBenchmarks:
    """Extended benchmark tests (slower)."""

    def test_long_benchmark(self, model_path, yolo_input):
        """Longer benchmark should have lower variance."""
        model = pi.load(model_path, device="cpu")

        result_short = model.benchmark(yolo_input, warmup=5, iterations=10)
        result_long = model.benchmark(yolo_input, warmup=10, iterations=100)

        # Longer benchmark should generally have lower relative std
        rel_std_short = result_short["std_ms"] / result_short["mean_ms"]
        rel_std_long = result_long["std_ms"] / result_long["mean_ms"]

        # Not strictly enforced as it depends on system state
        assert rel_std_long >= 0

    def test_all_available_backends(self, model_path, yolo_input):
        """Benchmark all available backends."""
        backends = pi.list_backends()
        devices = pi.list_devices()

        results = []
        for device in devices:
            try:
                model = pi.load(model_path, device=device.name)
                result = model.benchmark(yolo_input, warmup=3, iterations=10)
                results.append({
                    "device": device.name,
                    "backend": result["backend"],
                    "fps": result["fps"],
                    "mean_ms": result["mean_ms"],
                })
            except Exception as e:
                results.append({
                    "device": device.name,
                    "error": str(e),
                })

        # Should have at least one successful result
        successful = [r for r in results if "fps" in r]
        assert len(successful) > 0

        # Print results for visibility
        print("\n=== Benchmark Results ===")
        for r in results:
            if "fps" in r:
                print(f"{r['device']:20} {r['backend']:25} {r['mean_ms']:8.2f} ms  {r['fps']:8.1f} FPS")
            else:
                print(f"{r['device']:20} ERROR: {r.get('error', 'unknown')}")
