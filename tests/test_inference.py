"""Tests for inference correctness and consistency."""

import pytest
import numpy as np
import polyinfer as pi


class TestBasicInference:
    """Basic inference functionality tests."""

    def test_single_input_single_output(self, model_path, yolo_input):
        """Test simple single input, single output inference."""
        model = pi.load(model_path, device="cpu")
        output = model(yolo_input)

        assert output is not None
        assert isinstance(output, np.ndarray)
        assert output.dtype == np.float32

    def test_output_shape(self, model_path, yolo_input):
        """Output shape should be consistent."""
        model = pi.load(model_path, device="cpu")
        output = model(yolo_input)

        # YOLOv8n output shape: (1, 84, 8400) for detection
        assert len(output.shape) == 3
        assert output.shape[0] == 1  # Batch size

    def test_deterministic_output(self, model_path, yolo_input):
        """Same input should produce same output."""
        model = pi.load(model_path, device="cpu")

        output1 = model(yolo_input)
        output2 = model(yolo_input)

        np.testing.assert_array_almost_equal(output1, output2, decimal=5)

    def test_different_inputs(self, model_path):
        """Different inputs should produce different outputs."""
        model = pi.load(model_path, device="cpu")

        input1 = np.zeros((1, 3, 640, 640), dtype=np.float32)
        input2 = np.ones((1, 3, 640, 640), dtype=np.float32)

        output1 = model(input1)
        output2 = model(input2)

        assert not np.allclose(output1, output2)


class TestNamedInputOutput:
    """Test named input/output interface."""

    def test_run_with_dict(self, model_path, yolo_input):
        """Test run() method with dictionary input."""
        model = pi.load(model_path, device="cpu")

        # Get input name
        input_name = model.input_names[0]
        inputs = {input_name: yolo_input}

        outputs = model.run(inputs)

        assert isinstance(outputs, dict)
        assert len(outputs) > 0
        for name, arr in outputs.items():
            assert isinstance(name, str)
            assert isinstance(arr, np.ndarray)

    def test_input_output_names(self, model_path):
        """Model should expose input/output names."""
        model = pi.load(model_path, device="cpu")

        assert len(model.input_names) > 0
        assert len(model.output_names) > 0
        assert all(isinstance(n, str) for n in model.input_names)
        assert all(isinstance(n, str) for n in model.output_names)

    def test_input_output_shapes(self, model_path):
        """Model should expose input/output shapes."""
        model = pi.load(model_path, device="cpu")

        assert len(model.input_shapes) > 0
        assert len(model.output_shapes) > 0


class TestCrossBackendConsistency:
    """Test that different backends produce consistent results."""

    def test_onnxruntime_vs_openvino_cpu(self, model_path, yolo_input):
        """ONNX Runtime and OpenVINO should produce similar results on CPU."""
        if not pi.is_available("onnxruntime") or not pi.is_available("openvino"):
            pytest.skip("Both ONNX Runtime and OpenVINO required")

        model_ort = pi.load(model_path, backend="onnxruntime", device="cpu")
        model_ov = pi.load(model_path, backend="openvino", device="cpu")

        output_ort = model_ort(yolo_input)
        output_ov = model_ov(yolo_input)

        # Allow for small numerical differences
        np.testing.assert_allclose(output_ort, output_ov, rtol=1e-3, atol=1e-3)

    @pytest.mark.cuda
    def test_cpu_vs_cuda(self, model_path, yolo_input):
        """CPU and CUDA should produce similar results."""
        # Check if CUDA is available via ONNX Runtime
        backend = pi.get_backend("onnxruntime")
        if not backend.supports_device("cuda"):
            pytest.skip("ONNX Runtime CUDA not available (install onnxruntime-gpu)")

        model_cpu = pi.load(model_path, backend="onnxruntime", device="cpu")
        model_cuda = pi.load(model_path, backend="onnxruntime", device="cuda")

        output_cpu = model_cpu(yolo_input)
        output_cuda = model_cuda(yolo_input)

        np.testing.assert_allclose(output_cpu, output_cuda, rtol=1e-3, atol=1e-3)

    @pytest.mark.tensorrt
    def test_cuda_vs_tensorrt(self, model_path, yolo_input):
        """CUDA and TensorRT should produce similar results."""
        model_cuda = pi.load(model_path, device="cuda")
        model_trt = pi.load(model_path, device="tensorrt")

        output_cuda = model_cuda(yolo_input)
        output_trt = model_trt(yolo_input)

        # TensorRT may have slightly different results due to optimizations
        np.testing.assert_allclose(output_cuda, output_trt, rtol=1e-2, atol=1e-2)

    @pytest.mark.npu
    def test_cpu_vs_npu(self, model_path, yolo_input):
        """CPU and NPU should produce similar results."""
        model_cpu = pi.load(model_path, backend="openvino", device="cpu")
        model_npu = pi.load(model_path, backend="openvino", device="npu")

        output_cpu = model_cpu(yolo_input)
        output_npu = model_npu(yolo_input)

        # NPU may have significant quantization differences
        # Just check shapes match and values are in reasonable range
        assert output_cpu.shape == output_npu.shape
        # Allow for larger differences due to NPU quantization
        np.testing.assert_allclose(output_cpu, output_npu, rtol=0.1, atol=0.1)


class TestInputValidation:
    """Test input validation and error handling."""

    def test_wrong_shape(self, model_path):
        """Wrong input shape should raise error."""
        model = pi.load(model_path, device="cpu")

        wrong_input = np.random.rand(1, 3, 320, 320).astype(np.float32)
        # Behavior depends on backend - some may resize, others error
        # Just ensure it doesn't crash silently
        try:
            output = model(wrong_input)
            # If it succeeds, output should still be valid
            assert output is not None
        except (ValueError, RuntimeError):
            pass  # Expected for strict backends

    def test_wrong_dtype(self, model_path):
        """Wrong dtype should be handled."""
        model = pi.load(model_path, device="cpu")

        # Int input instead of float
        int_input = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8)

        # Should either convert or raise clear error
        try:
            output = model(int_input)
            assert output is not None
        except (ValueError, TypeError, RuntimeError):
            pass  # Expected

    def test_non_contiguous(self, model_path):
        """Non-contiguous array should be handled."""
        model = pi.load(model_path, device="cpu")

        # Create non-contiguous array via transpose
        base = np.random.rand(1, 640, 640, 3).astype(np.float32)
        non_contiguous = np.transpose(base, (0, 3, 1, 2))  # NHWC -> NCHW

        # The transposed array may or may not be contiguous depending on numpy version
        # Make it explicitly non-contiguous
        non_contiguous = np.asarray(non_contiguous)

        # Should work (backend should handle by making contiguous copy)
        output = model(non_contiguous)
        assert output is not None


class TestModelProperties:
    """Test model property accessors."""

    def test_backend_name(self, model_path):
        """backend_name should return string."""
        model = pi.load(model_path, device="cpu")
        assert isinstance(model.backend_name, str)
        assert len(model.backend_name) > 0

    def test_device(self, model_path):
        """device should return string."""
        model = pi.load(model_path, device="cpu")
        assert isinstance(model.device, str)

    def test_repr(self, model_path):
        """Model should have useful repr."""
        model = pi.load(model_path, device="cpu")
        repr_str = repr(model)

        assert "Model" in repr_str
        assert "backend" in repr_str.lower() or model.backend_name in repr_str

    def test_model_path(self, model_path):
        """model_path should be accessible."""
        model = pi.load(model_path, device="cpu")
        assert model.model_path.exists()


class TestMultipleInferences:
    """Test multiple inference calls."""

    def test_repeated_inference(self, model_path, yolo_input):
        """Multiple inference calls should work."""
        model = pi.load(model_path, device="cpu")

        for _ in range(10):
            output = model(yolo_input)
            assert output is not None

    def test_varying_batch_size(self, model_path):
        """Different batch sizes should work (if model supports)."""
        model = pi.load(model_path, device="cpu")

        # Try batch size 1
        input1 = np.random.rand(1, 3, 640, 640).astype(np.float32)
        output1 = model(input1)
        assert output1.shape[0] == 1

        # Some models support dynamic batch, some don't
        try:
            input2 = np.random.rand(2, 3, 640, 640).astype(np.float32)
            output2 = model(input2)
            assert output2.shape[0] == 2
        except (ValueError, RuntimeError):
            pass  # Model has fixed batch size

    def test_concurrent_inference_same_model(self, model_path, yolo_input):
        """Model should handle rapid consecutive calls."""
        model = pi.load(model_path, device="cpu")

        outputs = []
        for _ in range(5):
            outputs.append(model(yolo_input))

        # All outputs should be identical
        for output in outputs[1:]:
            np.testing.assert_array_almost_equal(outputs[0], output, decimal=5)
