"""Test polyinfer with Intel devices (CPU, iGPU, NPU).

These tests require Intel hardware and OpenVINO to be properly configured.
They are marked with intel_gpu or npu markers and will be skipped in CI.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import polyinfer as pi

# Check if OpenVINO is available
try:
    from polyinfer.backends.openvino import OpenVINOBackend

    OPENVINO_AVAILABLE = True
    ov_backend = OpenVINOBackend()
    AVAILABLE_DEVICES = ov_backend.get_available_devices()
except ImportError:
    OPENVINO_AVAILABLE = False
    AVAILABLE_DEVICES = []


def _get_test_model():
    """Get a test model path, or None if not available."""
    for path in ["yolov8n.onnx", "examples/yolov8n.onnx", "../yolov8n.onnx", "tests/yolov8n.onnx"]:
        if os.path.exists(path):
            return path
    return None


# Skip all tests if no model available or OpenVINO not installed
pytestmark = [
    pytest.mark.skipif(not OPENVINO_AVAILABLE, reason="OpenVINO not available"),
]


@pytest.fixture
def test_model():
    """Fixture providing a test model path."""
    model_path = _get_test_model()
    if model_path is None:
        pytest.skip("No test model available (yolov8n.onnx)")
    return model_path


@pytest.fixture
def test_input():
    """Fixture providing test input data for YOLOv8n (1x3x640x640)."""
    return np.random.rand(1, 3, 640, 640).astype(np.float32)


class TestIntelCPU:
    """Tests for Intel CPU inference."""

    def test_cpu_inference(self, test_model, test_input):
        """Test inference on CPU."""
        model = pi.load(test_model, backend="openvino", device="cpu")
        assert model.backend_name == "openvino"
        output = model(test_input)
        assert output is not None

    def test_cpu_benchmark(self, test_model, test_input):
        """Test benchmarking on CPU."""
        model = pi.load(test_model, backend="openvino", device="cpu")
        bench = model.benchmark(test_input, warmup=2, iterations=5)
        assert "mean_ms" in bench
        assert "fps" in bench
        assert bench["mean_ms"] > 0


@pytest.mark.intel_gpu
class TestIntelGPU:
    """Tests for Intel iGPU inference."""

    @pytest.fixture(autouse=True)
    def check_gpu_available(self):
        """Skip if Intel GPU not available."""
        if "GPU" not in AVAILABLE_DEVICES and "GPU.0" not in AVAILABLE_DEVICES:
            pytest.skip("Intel GPU not available")

    def test_igpu_inference(self, test_model, test_input):
        """Test inference on Intel iGPU."""
        model = pi.load(test_model, backend="openvino", device="intel-gpu")
        assert model.backend_name == "openvino"
        output = model(test_input)
        assert output is not None

    def test_igpu_benchmark(self, test_model, test_input):
        """Test benchmarking on Intel iGPU."""
        model = pi.load(test_model, backend="openvino", device="intel-gpu")
        bench = model.benchmark(test_input, warmup=2, iterations=5)
        assert "mean_ms" in bench
        assert "fps" in bench
        assert bench["mean_ms"] > 0


@pytest.mark.npu
class TestIntelNPU:
    """Tests for Intel NPU (AI Boost) inference."""

    @pytest.fixture(autouse=True)
    def check_npu_available(self):
        """Skip if Intel NPU not available."""
        if "NPU" not in AVAILABLE_DEVICES:
            pytest.skip("Intel NPU not available")

    def test_npu_inference(self, test_model, test_input):
        """Test inference on Intel NPU."""
        model = pi.load(test_model, backend="openvino", device="npu")
        assert model.backend_name == "openvino"
        output = model(test_input)
        assert output is not None

    def test_npu_benchmark(self, test_model, test_input):
        """Test benchmarking on Intel NPU."""
        model = pi.load(test_model, backend="openvino", device="npu")
        bench = model.benchmark(test_input, warmup=2, iterations=5)
        assert "mean_ms" in bench
        assert "fps" in bench
        assert bench["mean_ms"] > 0


if __name__ == "__main__":
    # When run as a script, print device info
    print("=" * 60)
    print("PolyInfer: Intel Device Test")
    print("=" * 60)

    print("\nAvailable backends:", pi.list_backends())
    print("Available devices:", pi.list_devices())

    if OPENVINO_AVAILABLE:
        print("\nOpenVINO raw devices:", AVAILABLE_DEVICES)
    else:
        print("\nOpenVINO not available")

    model_path = _get_test_model()
    if model_path:
        print(f"\nTest model found: {model_path}")
    else:
        print("\nNo test model found. Please provide yolov8n.onnx")
