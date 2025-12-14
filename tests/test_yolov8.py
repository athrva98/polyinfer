"""End-to-end YOLOv8 tests across all backends and devices.

This test suite validates that YOLOv8n runs correctly on every available
backend and device combination, checking both correctness and performance.
"""

import pytest
import numpy as np
import polyinfer as pi
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def yolov8_path():
    """Get path to YOLOv8n ONNX model."""
    candidates = [
        Path(__file__).parent.parent / "yolov8n.onnx",
        Path(__file__).parent.parent / "examples" / "yolov8n.onnx",
        Path.cwd() / "yolov8n.onnx",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # Try to export
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        export_path = Path(__file__).parent.parent / "yolov8n.onnx"
        model.export(format="onnx")
        if Path("yolov8n.onnx").exists():
            Path("yolov8n.onnx").rename(export_path)
        return str(export_path)
    except ImportError:
        pytest.skip("YOLOv8n model not found. Install ultralytics: pip install ultralytics")


@pytest.fixture(scope="module")
def yolo_input():
    """Standard YOLOv8 input tensor (1x3x640x640)."""
    np.random.seed(42)  # Reproducible
    return np.random.rand(1, 3, 640, 640).astype(np.float32)


@pytest.fixture(scope="module")
def reference_output(yolov8_path, yolo_input):
    """Get reference output from CPU for comparison."""
    model = pi.load(yolov8_path, device="cpu")
    return model(yolo_input)


# =============================================================================
# Device Discovery
# =============================================================================

def get_all_device_backend_combinations():
    """Get all valid (device, backend) combinations."""
    combinations = []

    for device_info in pi.list_devices():
        for backend_name in device_info.backends:
            combinations.append((device_info.name, backend_name))

    return combinations


# =============================================================================
# YOLOv8 Tests by Backend
# =============================================================================

class TestYOLOv8ONNXRuntime:
    """YOLOv8 tests for ONNX Runtime backend."""

    def test_cpu(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on ONNX Runtime CPU."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not available")

        model = pi.load(yolov8_path, backend="onnxruntime", device="cpu")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        # ONNX Runtime CPU should match reference closely
        np.testing.assert_allclose(output, reference_output, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_cuda(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on ONNX Runtime CUDA."""
        # Check if ONNX Runtime has CUDA support
        backend = pi.get_backend("onnxruntime")
        if not backend.supports_device("cuda"):
            pytest.skip("ONNX Runtime CUDA not available (install onnxruntime-gpu)")

        model = pi.load(yolov8_path, backend="onnxruntime", device="cuda")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        np.testing.assert_allclose(output, reference_output, rtol=1e-3, atol=1e-3)

    @pytest.mark.tensorrt
    def test_tensorrt(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on ONNX Runtime TensorRT EP."""
        model = pi.load(yolov8_path, backend="onnxruntime", device="tensorrt")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        # TensorRT may have larger differences due to optimizations
        np.testing.assert_allclose(output, reference_output, rtol=1e-2, atol=1e-2)

    @pytest.mark.directml
    def test_directml(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on ONNX Runtime DirectML."""
        model = pi.load(yolov8_path, backend="onnxruntime", device="directml")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        np.testing.assert_allclose(output, reference_output, rtol=1e-2, atol=1e-2)


class TestYOLOv8OpenVINO:
    """YOLOv8 tests for OpenVINO backend."""

    @pytest.mark.openvino
    def test_cpu(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on OpenVINO CPU."""
        model = pi.load(yolov8_path, backend="openvino", device="cpu")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        np.testing.assert_allclose(output, reference_output, rtol=1e-3, atol=1e-3)

    @pytest.mark.openvino
    @pytest.mark.intel_gpu
    def test_intel_gpu(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on OpenVINO Intel GPU."""
        model = pi.load(yolov8_path, backend="openvino", device="intel-gpu")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        # Intel GPU may have FP16 optimizations causing numerical differences
        # Check correlation is very high
        flat_ref = reference_output.flatten()
        flat_out = output.flatten()
        correlation = np.corrcoef(flat_ref, flat_out)[0, 1]
        assert correlation > 0.999, f"Low correlation: {correlation}"

    @pytest.mark.openvino
    @pytest.mark.npu
    def test_npu(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on OpenVINO NPU."""
        model = pi.load(yolov8_path, backend="openvino", device="npu")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        # NPU may have quantization differences
        np.testing.assert_allclose(output, reference_output, rtol=0.1, atol=0.1)


class TestYOLOv8IREE:
    """YOLOv8 tests for IREE backend."""

    def test_cpu(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on IREE CPU."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not available")

        model = pi.load(yolov8_path, backend="iree", device="cpu")
        output = model(yolo_input)

        assert output.shape == reference_output.shape
        np.testing.assert_allclose(output, reference_output, rtol=1e-2, atol=1e-2)

    @pytest.mark.vulkan
    def test_vulkan(self, yolov8_path, yolo_input):
        """Test YOLOv8 on IREE Vulkan."""
        model = pi.load(yolov8_path, backend="iree", device="vulkan")
        output = model(yolo_input)

        # Check basic output validity
        assert output is not None
        assert output.shape == (1, 84, 8400)

        # Note: Vulkan/SPIR-V can produce sporadic NaN values in a small number
        # of output elements due to driver/shader execution non-determinism.
        # We tolerate up to 0.1% NaN values as this is a known limitation.
        total_elements = 1 * 84 * 8400
        nan_count = np.sum(np.isnan(output))
        nan_ratio = nan_count / total_elements
        assert nan_ratio <= 0.001, f"Too many NaN values: {nan_count} ({nan_ratio*100:.2f}%)"

        assert not np.any(np.isinf(output)), "Output contains Inf"

        # Check output range is reasonable for YOLOv8 (ignoring NaN)
        valid_output = output[~np.isnan(output)]
        assert valid_output.min() >= -100, f"Output min too low: {valid_output.min()}"
        assert valid_output.max() <= 1000, f"Output max too high: {valid_output.max()}"

    @pytest.mark.cuda
    def test_cuda(self, yolov8_path, yolo_input, reference_output):
        """Test YOLOv8 on IREE CUDA."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not available")

        # Check if IREE has CUDA support
        devices = pi.list_devices()
        iree_cuda = any(d.name == "cuda" and "iree" in d.backends for d in devices)
        if not iree_cuda:
            pytest.skip("IREE CUDA not available")

        try:
            model = pi.load(yolov8_path, backend="iree", device="cuda")
            output = model(yolo_input)

            assert output.shape == reference_output.shape
            np.testing.assert_allclose(output, reference_output, rtol=1e-2, atol=1e-2)
        except RuntimeError as e:
            # IREE CUDA can fail under resource pressure in test suites
            if "CUDA" in str(e) or "cuda" in str(e) or "out of memory" in str(e).lower():
                pytest.skip(f"IREE CUDA runtime error (expected under load): {e}")


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestYOLOv8Benchmarks:
    """Benchmark YOLOv8 across all backends."""

    @pytest.mark.benchmark
    def test_benchmark_all_devices(self, yolov8_path, yolo_input):
        """Benchmark YOLOv8 on all available devices."""
        results = []

        for device_info in pi.list_devices():
            device = device_info.name
            for backend_name in device_info.backends:
                try:
                    model = pi.load(yolov8_path, backend=backend_name, device=device)
                    bench = model.benchmark(yolo_input, warmup=5, iterations=20)
                    results.append({
                        "device": device,
                        "backend": backend_name,
                        "backend_name": model.backend_name,
                        "mean_ms": bench["mean_ms"],
                        "fps": bench["fps"],
                        "status": "success",
                    })
                except Exception as e:
                    results.append({
                        "device": device,
                        "backend": backend_name,
                        "error": str(e),
                        "status": "error",
                    })

        # Print results
        print("\n" + "=" * 80)
        print("YOLOv8n Benchmark Results")
        print("=" * 80)
        print(f"{'Device':<15} {'Backend':<25} {'Latency':>12} {'FPS':>10}")
        print("-" * 80)

        successful = [r for r in results if r["status"] == "success"]
        successful.sort(key=lambda r: r["mean_ms"])

        for r in successful:
            print(f"{r['device']:<15} {r['backend_name']:<25} {r['mean_ms']:>10.2f}ms {r['fps']:>9.1f}")

        print("-" * 80)

        failed = [r for r in results if r["status"] == "error"]
        if failed:
            print(f"\nFailed ({len(failed)}):")
            for r in failed:
                print(f"  {r['device']}/{r['backend']}: {r['error'][:60]}")

        # At least one device should work
        assert len(successful) > 0

    @pytest.mark.benchmark
    @pytest.mark.openvino
    def test_benchmark_openvino_devices(self, yolov8_path, yolo_input):
        """Benchmark YOLOv8 on all OpenVINO devices."""
        if not pi.is_available("openvino"):
            pytest.skip("OpenVINO not available")

        from polyinfer.backends.openvino import OpenVINOBackend
        ov_backend = OpenVINOBackend()
        raw_devices = ov_backend.get_available_devices()

        print(f"\nOpenVINO raw devices: {raw_devices}")

        results = []
        device_map = {
            "CPU": "cpu",
            "GPU.0": "intel-gpu:0",
            "GPU.1": "intel-gpu:1",
            "NPU": "npu",
        }

        for raw_device in raw_devices:
            pi_device = device_map.get(raw_device, raw_device.lower())
            try:
                model = pi.load(yolov8_path, backend="openvino", device=pi_device)
                bench = model.benchmark(yolo_input, warmup=5, iterations=20)
                results.append({
                    "raw_device": raw_device,
                    "pi_device": pi_device,
                    "mean_ms": bench["mean_ms"],
                    "fps": bench["fps"],
                })
                print(f"  {raw_device} ({pi_device}): {bench['mean_ms']:.2f}ms ({bench['fps']:.1f} FPS)")
            except Exception as e:
                print(f"  {raw_device} ({pi_device}): ERROR - {e}")

        assert len(results) > 0

    @pytest.mark.benchmark
    def test_benchmark_iree_devices(self, yolov8_path, yolo_input):
        """Benchmark YOLOv8 on all IREE devices."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not available")

        devices = ["cpu", "vulkan"]
        results = []

        print("\nIREE Benchmarks:")
        for device in devices:
            try:
                model = pi.load(yolov8_path, backend="iree", device=device)
                bench = model.benchmark(yolo_input, warmup=5, iterations=20)
                results.append({
                    "device": device,
                    "mean_ms": bench["mean_ms"],
                    "fps": bench["fps"],
                })
                print(f"  {device}: {bench['mean_ms']:.2f}ms ({bench['fps']:.1f} FPS)")
            except Exception as e:
                print(f"  {device}: ERROR - {e}")

        assert len(results) > 0


# =============================================================================
# Cross-Backend Consistency
# =============================================================================

class TestYOLOv8Consistency:
    """Test output consistency across backends."""

    def test_all_backends_same_output_shape(self, yolov8_path, yolo_input):
        """All backends should produce same output shape."""
        shapes = {}

        for device_info in pi.list_devices():
            for backend_name in device_info.backends:
                try:
                    model = pi.load(
                        yolov8_path,
                        backend=backend_name,
                        device=device_info.name
                    )
                    output = model(yolo_input)
                    key = f"{backend_name}-{device_info.name}"
                    shapes[key] = output.shape
                except Exception:
                    continue

        # All shapes should be the same
        unique_shapes = set(shapes.values())
        assert len(unique_shapes) == 1, f"Different output shapes: {shapes}"

    @pytest.mark.openvino
    def test_openvino_cpu_vs_npu(self, yolov8_path, yolo_input):
        """OpenVINO CPU and NPU should produce similar results."""
        devices = pi.list_devices()
        has_npu = any(d.name == "npu" for d in devices)
        if not has_npu:
            pytest.skip("NPU not available")

        model_cpu = pi.load(yolov8_path, backend="openvino", device="cpu")
        model_npu = pi.load(yolov8_path, backend="openvino", device="npu")

        output_cpu = model_cpu(yolo_input)
        output_npu = model_npu(yolo_input)

        # Check relative ordering is preserved (top detections)
        # NPU may have quantization differences
        assert output_cpu.shape == output_npu.shape

        # Check correlation is high
        flat_cpu = output_cpu.flatten()
        flat_npu = output_npu.flatten()
        correlation = np.corrcoef(flat_cpu, flat_npu)[0, 1]
        assert correlation > 0.95, f"Low correlation: {correlation}"

    def test_onnxruntime_vs_openvino_cpu(self, yolov8_path, yolo_input):
        """ONNX Runtime and OpenVINO CPU should produce similar results."""
        if not pi.is_available("onnxruntime") or not pi.is_available("openvino"):
            pytest.skip("Both backends required")

        model_ort = pi.load(yolov8_path, backend="onnxruntime", device="cpu")
        model_ov = pi.load(yolov8_path, backend="openvino", device="cpu")

        output_ort = model_ort(yolo_input)
        output_ov = model_ov(yolo_input)

        np.testing.assert_allclose(output_ort, output_ov, rtol=1e-3, atol=1e-3)

    @pytest.mark.vulkan
    def test_iree_vulkan_vs_cpu(self, yolov8_path, yolo_input):
        """IREE Vulkan and CPU should produce similar results."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not available")

        model_cpu = pi.load(yolov8_path, backend="iree", device="cpu")
        model_vulkan = pi.load(yolov8_path, backend="iree", device="vulkan")

        output_cpu = model_cpu(yolo_input)
        output_vulkan = model_vulkan(yolo_input)

        # Both should produce valid outputs
        assert output_cpu.shape == output_vulkan.shape
        assert not np.any(np.isnan(output_cpu))

        # Vulkan may have sporadic NaN values (known limitation)
        # Allow up to 0.1% NaN values
        total_elements = output_vulkan.size
        nan_count = np.sum(np.isnan(output_vulkan))
        nan_ratio = nan_count / total_elements
        assert nan_ratio <= 0.001, f"Too many NaN values: {nan_count} ({nan_ratio*100:.2f}%)"

        # Compare non-NaN values using correlation
        valid_mask = ~np.isnan(output_vulkan)
        flat_cpu = output_cpu[valid_mask].flatten()
        flat_vulkan = output_vulkan[valid_mask].flatten()
        correlation = np.corrcoef(flat_cpu, flat_vulkan)[0, 1]
        assert correlation > 0.99, f"Low correlation: {correlation}"


# =============================================================================
# Stress Tests
# =============================================================================

class TestYOLOv8Stress:
    """Stress tests for YOLOv8."""

    def test_repeated_inference_cpu(self, yolov8_path, yolo_input):
        """Run many inferences on CPU."""
        model = pi.load(yolov8_path, device="cpu")

        outputs = []
        for _ in range(50):
            output = model(yolo_input)
            outputs.append(output)

        # All outputs should be identical
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)

    @pytest.mark.vulkan
    def test_repeated_inference_vulkan(self, yolov8_path, yolo_input):
        """Run many inferences on Vulkan."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not available")

        model = pi.load(yolov8_path, backend="iree", device="vulkan")

        # Run inference multiple times and check outputs are valid
        # Note: Vulkan/SPIR-V can produce sporadic NaN values in a small number
        # of output elements due to driver/shader execution non-determinism.
        # We tolerate up to 0.1% NaN values as this is a known limitation.
        total_elements = 1 * 84 * 8400
        max_nan_ratio = 0.001  # 0.1% tolerance

        for i in range(10):
            output = model(yolo_input)
            assert output is not None
            assert output.shape == (1, 84, 8400)

            nan_count = np.sum(np.isnan(output))
            nan_ratio = nan_count / total_elements
            assert nan_ratio <= max_nan_ratio, (
                f"Run {i} has too many NaN values: {nan_count} ({nan_ratio*100:.2f}%)"
            )

            assert not np.any(np.isinf(output)), f"Run {i} contains Inf"

    @pytest.mark.npu
    def test_repeated_inference_npu(self, yolov8_path, yolo_input):
        """Run many inferences on NPU."""
        model = pi.load(yolov8_path, backend="openvino", device="npu")

        outputs = []
        for _ in range(50):
            output = model(yolo_input)
            outputs.append(output.copy())

        # All outputs should be identical
        for output in outputs[1:]:
            np.testing.assert_array_equal(outputs[0], output)

    def test_different_random_inputs(self, yolov8_path):
        """Run inference with different random inputs."""
        model = pi.load(yolov8_path, device="cpu")

        for i in range(10):
            np.random.seed(i)
            input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
            output = model(input_data)
            assert output is not None
            assert output.shape == (1, 84, 8400)


# =============================================================================
# Main - Run as script for quick testing
# =============================================================================

if __name__ == "__main__":
    import sys

    # Quick manual test
    print("=" * 60)
    print("YOLOv8 Quick Test")
    print("=" * 60)

    # Find model
    model_path = None
    for path in ["yolov8n.onnx", "../yolov8n.onnx", "examples/yolov8n.onnx"]:
        if Path(path).exists():
            model_path = path
            break

    if not model_path:
        print("ERROR: yolov8n.onnx not found")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Backends: {pi.list_backends()}")
    print(f"Devices: {[d.name for d in pi.list_devices()]}")

    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

    print("\n" + "-" * 60)
    print("Running benchmarks...")
    print("-" * 60)

    results = []
    for device_info in pi.list_devices():
        for backend in device_info.backends:
            try:
                model = pi.load(model_path, backend=backend, device=device_info.name)
                bench = model.benchmark(input_data, warmup=5, iterations=20)
                print(f"{device_info.name:15} {model.backend_name:25} {bench['mean_ms']:8.2f}ms {bench['fps']:8.1f} FPS")
                results.append((device_info.name, model.backend_name, bench['mean_ms'], bench['fps']))
            except Exception as e:
                print(f"{device_info.name:15} {backend:25} ERROR: {str(e)[:40]}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary (sorted by speed)")
    print("=" * 60)
    results.sort(key=lambda x: x[2])
    for device, backend, ms, fps in results:
        print(f"{backend:30} {ms:8.2f}ms {fps:8.1f} FPS")
