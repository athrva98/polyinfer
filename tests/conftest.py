"""Pytest configuration and shared fixtures for polyinfer tests."""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polyinfer as pi
from polyinfer.backends.registry import get_all_backends, get_backend


# =============================================================================
# Test Model Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def model_path():
    """Get path to test model (YOLOv8n)."""
    # Check common locations
    candidates = [
        Path(__file__).parent.parent / "yolov8n.onnx",
        Path(__file__).parent.parent / "examples" / "yolov8n.onnx",
        Path.cwd() / "yolov8n.onnx",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    # Try to download/export
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        export_path = Path(__file__).parent.parent / "yolov8n.onnx"
        model.export(format="onnx")
        # Move to expected location
        if Path("yolov8n.onnx").exists():
            Path("yolov8n.onnx").rename(export_path)
        return str(export_path)
    except ImportError:
        pytest.skip("No test model available. Install ultralytics: pip install ultralytics")


@pytest.fixture(scope="session")
def simple_model_path(tmp_path_factory):
    """Create a simple ONNX model for basic tests."""
    try:
        import onnx
        from onnx import helper, TensorProto

        # Create a simple model: Y = X + 1
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 224, 224])

        # Constant tensor of ones
        ones = helper.make_tensor(
            "ones",
            TensorProto.FLOAT,
            [1, 3, 224, 224],
            [1.0] * (1 * 3 * 224 * 224)
        )

        add_node = helper.make_node("Add", ["X", "ones"], ["Y"])

        graph = helper.make_graph(
            [add_node],
            "simple_add",
            [X],
            [Y],
            [ones]
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        # Save to temp file
        tmp_dir = tmp_path_factory.mktemp("models")
        model_path = tmp_dir / "simple_add.onnx"
        onnx.save(model, str(model_path))

        return str(model_path)
    except ImportError:
        pytest.skip("onnx package required for simple model creation")


# =============================================================================
# Input Data Fixtures
# =============================================================================

@pytest.fixture
def yolo_input():
    """Create input tensor for YOLOv8 (1x3x640x640)."""
    return np.random.rand(1, 3, 640, 640).astype(np.float32)


@pytest.fixture
def simple_input():
    """Create input tensor for simple model (1x3x224x224)."""
    return np.random.rand(1, 3, 224, 224).astype(np.float32)


@pytest.fixture
def batch_input():
    """Create batched input tensor (4x3x640x640)."""
    return np.random.rand(4, 3, 640, 640).astype(np.float32)


# =============================================================================
# Backend/Device Discovery Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def available_backends():
    """Get list of available backends."""
    return pi.list_backends()


@pytest.fixture(scope="session")
def available_devices():
    """Get list of available devices."""
    return pi.list_devices()


@pytest.fixture(scope="session")
def all_backends():
    """Get all registered backends (available or not)."""
    return get_all_backends()


# =============================================================================
# Device-specific Fixtures
# =============================================================================

@pytest.fixture
def has_cuda():
    """Check if CUDA is available."""
    devices = pi.list_devices()
    return any(d.name == "cuda" or d.name.startswith("cuda:") for d in devices)


@pytest.fixture
def has_tensorrt():
    """Check if TensorRT is available."""
    devices = pi.list_devices()
    return any(d.name == "tensorrt" for d in devices)


@pytest.fixture
def has_directml():
    """Check if DirectML is available."""
    devices = pi.list_devices()
    return any(d.name == "directml" for d in devices)


@pytest.fixture
def has_openvino():
    """Check if OpenVINO is available."""
    return "openvino" in pi.list_backends()


@pytest.fixture
def has_intel_gpu():
    """Check if Intel GPU is available."""
    devices = pi.list_devices()
    return any(d.name.startswith("intel-gpu") for d in devices)


@pytest.fixture
def has_npu():
    """Check if Intel NPU is available."""
    devices = pi.list_devices()
    return any(d.name == "npu" for d in devices)


@pytest.fixture
def has_vulkan():
    """Check if Vulkan is available."""
    devices = pi.list_devices()
    return any(d.name == "vulkan" for d in devices)


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "tensorrt: mark test as requiring TensorRT")
    config.addinivalue_line("markers", "directml: mark test as requiring DirectML")
    config.addinivalue_line("markers", "openvino: mark test as requiring OpenVINO")
    config.addinivalue_line("markers", "intel_gpu: mark test as requiring Intel GPU")
    config.addinivalue_line("markers", "npu: mark test as requiring Intel NPU")
    config.addinivalue_line("markers", "vulkan: mark test as requiring Vulkan")
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available hardware."""
    import polyinfer as pi

    devices = pi.list_devices()
    device_names = {d.name for d in devices}
    backends = set(pi.list_backends())

    # Check device availability
    has_cuda = any(d.startswith("cuda") for d in device_names)
    has_tensorrt = "tensorrt" in device_names
    has_directml = "directml" in device_names
    has_openvino = "openvino" in backends
    has_intel_gpu = any(d.startswith("intel-gpu") for d in device_names)
    has_npu = "npu" in device_names
    has_vulkan = "vulkan" in device_names

    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_tensorrt = pytest.mark.skip(reason="TensorRT not available")
    skip_directml = pytest.mark.skip(reason="DirectML not available")
    skip_openvino = pytest.mark.skip(reason="OpenVINO not available")
    skip_intel_gpu = pytest.mark.skip(reason="Intel GPU not available")
    skip_npu = pytest.mark.skip(reason="Intel NPU not available")
    skip_vulkan = pytest.mark.skip(reason="Vulkan not available")

    for item in items:
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)
        if "tensorrt" in item.keywords and not has_tensorrt:
            item.add_marker(skip_tensorrt)
        if "directml" in item.keywords and not has_directml:
            item.add_marker(skip_directml)
        if "openvino" in item.keywords and not has_openvino:
            item.add_marker(skip_openvino)
        if "intel_gpu" in item.keywords and not has_intel_gpu:
            item.add_marker(skip_intel_gpu)
        if "npu" in item.keywords and not has_npu:
            item.add_marker(skip_npu)
        if "vulkan" in item.keywords and not has_vulkan:
            item.add_marker(skip_vulkan)
