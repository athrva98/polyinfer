"""Tests for device-specific functionality."""

import numpy as np
import pytest

import polyinfer as pi


class TestCPUDevice:
    """Tests for CPU device across all backends."""

    def test_load_cpu_auto(self, model_path, yolo_input):
        """Load model on CPU with auto backend selection."""
        model = pi.load(model_path, device="cpu")
        assert model is not None
        assert "cpu" in model.device.lower()

        # Run inference
        output = model(yolo_input)
        assert output is not None
        assert isinstance(output, np.ndarray)

    def test_load_cpu_onnxruntime(self, model_path, yolo_input):
        """Load model on CPU with ONNX Runtime."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not available")

        model = pi.load(model_path, backend="onnxruntime", device="cpu")
        assert "onnxruntime" in model.backend_name

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.openvino
    def test_load_cpu_openvino(self, model_path, yolo_input):
        """Load model on CPU with OpenVINO."""
        model = pi.load(model_path, backend="openvino", device="cpu")
        assert "openvino" in model.backend_name

        output = model(yolo_input)
        assert output is not None


class TestCUDADevice:
    """Tests for NVIDIA CUDA device."""

    @pytest.mark.cuda
    def test_load_cuda_auto(self, model_path, yolo_input):
        """Load model on CUDA with auto backend selection."""
        model = pi.load(model_path, device="cuda")
        assert model is not None

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.cuda
    def test_load_cuda_explicit(self, model_path, yolo_input):
        """Load model on CUDA with explicit backend."""
        # Check if ONNX Runtime has CUDA support
        backend = pi.get_backend("onnxruntime")
        if not backend.supports_device("cuda"):
            pytest.skip("ONNX Runtime CUDA not available (install onnxruntime-gpu)")

        model = pi.load(model_path, backend="onnxruntime", device="cuda")
        assert "cuda" in model.backend_name.lower()

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.cuda
    def test_load_cuda_device_id(self, model_path, yolo_input):
        """Load model on specific CUDA device."""
        # Check if CUDA is available
        devices = pi.list_devices()
        has_cuda = any(d.name == "cuda" for d in devices)
        if not has_cuda:
            pytest.skip("CUDA not available")

        model = pi.load(model_path, device="cuda:0")
        output = model(yolo_input)
        assert output is not None

    @pytest.mark.cuda
    def test_device_aliases(self, model_path, yolo_input):
        """Test device aliases (gpu, nvidia)."""
        # Check if CUDA is available via ONNX Runtime (needed for gpu/nvidia aliases)
        backend = pi.get_backend("onnxruntime")
        if not backend.supports_device("cuda"):
            pytest.skip("ONNX Runtime CUDA not available (install onnxruntime-gpu)")

        # 'gpu' should map to cuda
        model = pi.load(model_path, device="gpu")
        assert model is not None

        # 'nvidia' should map to cuda
        model = pi.load(model_path, device="nvidia")
        assert model is not None


class TestTensorRTDevice:
    """Tests for TensorRT device."""

    @pytest.mark.tensorrt
    def test_load_tensorrt(self, model_path, yolo_input):
        """Load model with TensorRT."""
        model = pi.load(model_path, device="tensorrt")
        assert model is not None
        assert "tensorrt" in model.backend_name.lower()

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.tensorrt
    def test_tensorrt_backend_alias(self, model_path, yolo_input):
        """Test tensorrt backend alias."""
        model = pi.load(model_path, backend="tensorrt", device="cuda")
        assert "tensorrt" in model.backend_name.lower()

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.tensorrt
    def test_tensorrt_fp16(self, model_path, yolo_input):
        """Test TensorRT with FP16."""
        model = pi.load(model_path, device="tensorrt", fp16=True)
        output = model(yolo_input)
        assert output is not None


class TestDirectMLDevice:
    """Tests for DirectML device."""

    @pytest.mark.directml
    def test_load_directml(self, model_path, yolo_input):
        """Load model with DirectML."""
        model = pi.load(model_path, device="directml")
        assert model is not None
        assert "dml" in model.backend_name.lower() or "directml" in model.backend_name.lower()

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.directml
    def test_directml_device_id(self, model_path, yolo_input):
        """Load model on specific DirectML device."""
        model = pi.load(model_path, device="directml:0")
        output = model(yolo_input)
        assert output is not None


class TestIntelGPUDevice:
    """Tests for Intel integrated GPU."""

    @pytest.mark.intel_gpu
    def test_load_intel_gpu(self, model_path, yolo_input):
        """Load model on Intel GPU."""
        model = pi.load(model_path, backend="openvino", device="intel-gpu")
        assert model is not None
        assert "openvino" in model.backend_name.lower()

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.intel_gpu
    def test_load_intel_gpu_explicit_id(self, model_path, yolo_input):
        """Load model on specific Intel GPU."""
        model = pi.load(model_path, backend="openvino", device="intel-gpu:0")
        output = model(yolo_input)
        assert output is not None


class TestNPUDevice:
    """Tests for Intel NPU (AI Boost)."""

    @pytest.mark.npu
    def test_load_npu(self, model_path, yolo_input):
        """Load model on Intel NPU."""
        model = pi.load(model_path, backend="openvino", device="npu")
        assert model is not None
        assert "npu" in model.device.lower()

        output = model(yolo_input)
        assert output is not None

    @pytest.mark.npu
    def test_npu_performance(self, model_path, yolo_input):
        """NPU should provide reasonable performance."""
        model = pi.load(model_path, backend="openvino", device="npu")
        bench = model.benchmark(yolo_input, warmup=5, iterations=20)

        # NPU should be at least somewhat fast
        assert bench["fps"] > 10, f"NPU too slow: {bench['fps']} FPS"


class TestVulkanDevice:
    """Tests for Vulkan device (IREE)."""

    @pytest.mark.vulkan
    def test_load_vulkan(self, model_path, yolo_input):
        """Load model with Vulkan backend."""
        model = pi.load(model_path, device="vulkan")
        assert model is not None

        output = model(yolo_input)
        assert output is not None


class TestDeviceNormalization:
    """Test device string normalization."""

    def test_lowercase(self, model_path):
        """Device strings should be case-insensitive."""
        model1 = pi.load(model_path, device="CPU")
        model2 = pi.load(model_path, device="cpu")
        model3 = pi.load(model_path, device="Cpu")

        assert model1.backend_name == model2.backend_name == model3.backend_name

    def test_whitespace(self, model_path):
        """Device strings should be trimmed."""
        model1 = pi.load(model_path, device="cpu")
        model2 = pi.load(model_path, device=" cpu ")

        assert model1.backend_name == model2.backend_name

    @pytest.mark.cuda
    def test_gpu_alias(self, model_path):
        """'gpu' should map to 'cuda'."""
        model = pi.load(model_path, device="gpu")
        # Should use CUDA backend
        assert "cuda" in model.device.lower() or "cuda" in model.backend_name.lower()

    @pytest.mark.cuda
    def test_nvidia_alias(self, model_path):
        """'nvidia' should map to 'cuda'."""
        model = pi.load(model_path, device="nvidia")
        assert "cuda" in model.device.lower() or "cuda" in model.backend_name.lower()

    @pytest.mark.tensorrt
    def test_trt_alias(self, model_path):
        """'trt' should map to 'tensorrt'."""
        model = pi.load(model_path, device="trt")
        assert "tensorrt" in model.backend_name.lower()


class TestInvalidDevice:
    """Test error handling for invalid devices."""

    def test_invalid_device(self, model_path):
        """Invalid device should raise error."""
        with pytest.raises((ValueError, RuntimeError)):
            pi.load(model_path, device="nonexistent_device_xyz")

    def test_invalid_backend(self, model_path):
        """Invalid backend should raise error."""
        with pytest.raises((ValueError, KeyError)):
            pi.load(model_path, backend="nonexistent_backend_xyz", device="cpu")

    def test_backend_device_mismatch(self, model_path):
        """Mismatched backend/device should raise error."""
        # OpenVINO doesn't support CUDA
        if pi.is_available("openvino"):
            with pytest.raises(ValueError):
                pi.load(model_path, backend="openvino", device="cuda")
