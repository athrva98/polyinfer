"""Tests for quantization functionality."""

import importlib.util

import numpy as np
import pytest

import polyinfer as pi

# Check if onnxruntime quantization is available
try:
    from onnxruntime.quantization import quantize_dynamic as _  # noqa: F401

    ONNXRUNTIME_QUANT_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_QUANT_AVAILABLE = False

# Check if any backend can load models
_HAS_ANY_BACKEND = len(pi.list_backends()) > 0


class TestQuantizationAPI:
    """Test quantization API availability and basic functionality."""

    def test_quantize_function_exists(self):
        """Test that quantize function is exported."""
        assert hasattr(pi, "quantize")
        assert hasattr(pi, "quantize_dynamic")
        assert hasattr(pi, "quantize_static")
        assert hasattr(pi, "convert_to_fp16")

    def test_quantization_types_exported(self):
        """Test that quantization types are exported."""
        assert hasattr(pi, "QuantizationResult")
        assert hasattr(pi, "QuantizationConfig")
        assert hasattr(pi, "QuantizationMethod")
        assert hasattr(pi, "QuantizationType")
        assert hasattr(pi, "CalibrationMethod")

    def test_quantization_method_values(self):
        """Test QuantizationMethod enum values."""
        assert pi.QuantizationMethod.DYNAMIC.value == "dynamic"
        assert pi.QuantizationMethod.STATIC.value == "static"

    def test_quantization_type_values(self):
        """Test QuantizationType enum values."""
        assert pi.QuantizationType.INT8.value == "int8"
        assert pi.QuantizationType.UINT8.value == "uint8"
        assert pi.QuantizationType.INT4.value == "int4"
        assert pi.QuantizationType.FP16.value == "fp16"
        assert pi.QuantizationType.BF16.value == "bf16"

    def test_calibration_method_values(self):
        """Test CalibrationMethod enum values."""
        assert pi.CalibrationMethod.MINMAX.value == "minmax"
        assert pi.CalibrationMethod.ENTROPY.value == "entropy"
        assert pi.CalibrationMethod.PERCENTILE.value == "percentile"


@pytest.mark.skipif(
    not ONNXRUNTIME_QUANT_AVAILABLE, reason="onnxruntime quantization not installed"
)
class TestDynamicQuantization:
    """Test dynamic quantization (no calibration needed)."""

    @pytest.fixture
    def simple_model(self, tmp_path):
        """Create a simple ONNX model for testing."""
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            pytest.skip("ONNX not installed")

        # Create simple model: output = input * 2
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
        const_tensor = helper.make_tensor("const", TensorProto.FLOAT, [1], [2.0])
        mul_node = helper.make_node("Mul", ["input", "const"], ["output"], name="mul")
        graph = helper.make_graph(
            [mul_node], "test_model", [input_tensor], [output_tensor], [const_tensor]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(model_path))
        return model_path

    def test_dynamic_quantization_int8(self, simple_model, tmp_path):
        """Test dynamic INT8 quantization."""
        output_path = tmp_path / "model_int8.onnx"

        result = pi.quantize_dynamic(
            simple_model,
            output_path,
            dtype="int8",
        )

        assert output_path.exists()
        assert result.backend == "onnxruntime"
        assert result.method == "dynamic"
        assert result.dtype == "int8"
        assert result.output_path == output_path

    def test_dynamic_quantization_uint8(self, simple_model, tmp_path):
        """Test dynamic UINT8 quantization."""
        output_path = tmp_path / "model_uint8.onnx"

        result = pi.quantize_dynamic(
            simple_model,
            output_path,
            dtype="uint8",
        )

        assert output_path.exists()
        assert result.dtype == "uint8"

    def test_quantize_function_dynamic(self, simple_model, tmp_path):
        """Test quantize() with method='dynamic'."""
        output_path = tmp_path / "model_quant.onnx"

        result = pi.quantize(
            simple_model,
            output_path,
            method="dynamic",
            dtype="int8",
        )

        assert output_path.exists()
        assert result.method == "dynamic"

    @pytest.mark.skipif(not _HAS_ANY_BACKEND, reason="No backends installed")
    def test_quantized_model_loads(self, simple_model, tmp_path):
        """Test that quantized model can be loaded and run."""
        output_path = tmp_path / "model_int8.onnx"

        pi.quantize_dynamic(simple_model, output_path, dtype="int8")

        # Load and run the quantized model
        model = pi.load(str(output_path), device="cpu")
        input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)
        output = model(input_data)

        assert output is not None
        assert output.shape == (1, 3, 32, 32)


@pytest.mark.skipif(
    not ONNXRUNTIME_QUANT_AVAILABLE, reason="onnxruntime quantization not installed"
)
class TestStaticQuantization:
    """Test static quantization (requires calibration)."""

    @pytest.fixture
    def simple_model(self, tmp_path):
        """Create a simple ONNX model for testing."""
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            pytest.skip("ONNX not installed")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
        const_tensor = helper.make_tensor("const", TensorProto.FLOAT, [1], [2.0])
        mul_node = helper.make_node("Mul", ["input", "const"], ["output"], name="mul")
        graph = helper.make_graph(
            [mul_node], "test_model", [input_tensor], [output_tensor], [const_tensor]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(model_path))
        return model_path

    @pytest.fixture
    def calibration_data(self):
        """Create calibration data."""
        return [np.random.rand(1, 3, 32, 32).astype(np.float32) for _ in range(10)]

    def test_static_quantization_requires_data(self, simple_model, tmp_path):
        """Test that static quantization fails without calibration data."""
        output_path = tmp_path / "model_int8.onnx"

        with pytest.raises(ValueError, match="calibration_data"):
            pi.quantize(
                simple_model,
                output_path,
                method="static",
            )

    def test_static_quantization_with_list(self, simple_model, tmp_path, calibration_data):
        """Test static quantization with list of arrays."""
        output_path = tmp_path / "model_int8.onnx"

        result = pi.quantize_static(
            simple_model,
            output_path,
            calibration_data=calibration_data,
            dtype="int8",
        )

        assert output_path.exists()
        assert result.method == "static"
        assert result.dtype == "int8"

    def test_static_quantization_with_dict_list(self, simple_model, tmp_path):
        """Test static quantization with list of dicts."""
        output_path = tmp_path / "model_int8.onnx"

        calibration_data = [
            {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)} for _ in range(10)
        ]

        pi.quantize_static(
            simple_model,
            output_path,
            calibration_data=calibration_data,
            dtype="int8",
        )

        assert output_path.exists()

    def test_static_quantization_per_channel(self, simple_model, tmp_path, calibration_data):
        """Test static quantization with per-channel option."""
        output_path = tmp_path / "model_int8.onnx"

        pi.quantize_static(
            simple_model,
            output_path,
            calibration_data=calibration_data,
            per_channel=True,
        )

        assert output_path.exists()

    def test_static_quantization_entropy_calibration(
        self, simple_model, tmp_path, calibration_data
    ):
        """Test static quantization with entropy calibration."""
        output_path = tmp_path / "model_int8.onnx"

        pi.quantize(
            simple_model,
            output_path,
            method="static",
            calibration_data=calibration_data,
            calibration_method="entropy",
        )

        assert output_path.exists()


class TestFP16Conversion:
    """Test FP16 conversion."""

    @pytest.fixture
    def simple_model(self, tmp_path):
        """Create a simple ONNX model for testing."""
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            pytest.skip("ONNX not installed")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
        const_tensor = helper.make_tensor("const", TensorProto.FLOAT, [1], [2.0])
        mul_node = helper.make_node("Mul", ["input", "const"], ["output"], name="mul")
        graph = helper.make_graph(
            [mul_node], "test_model", [input_tensor], [output_tensor], [const_tensor]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(model_path))
        return model_path

    def test_fp16_conversion(self, simple_model, tmp_path):
        """Test FP16 conversion."""
        if importlib.util.find_spec("onnxconverter_common") is None:
            pytest.skip("onnxconverter-common not installed")

        output_path = tmp_path / "model_fp16.onnx"

        result = pi.convert_to_fp16(simple_model, output_path)

        assert output_path.exists()
        assert result.dtype == "fp16"

    def test_fp16_via_quantize(self, simple_model, tmp_path):
        """Test FP16 conversion via quantize()."""
        if importlib.util.find_spec("onnxconverter_common") is None:
            pytest.skip("onnxconverter-common not installed")

        output_path = tmp_path / "model_fp16.onnx"

        result = pi.quantize(
            simple_model,
            output_path,
            dtype="fp16",
        )

        assert output_path.exists()
        assert result.dtype == "fp16"

    def test_fp16_model_runs(self, simple_model, tmp_path):
        """Test that FP16 model can be loaded and run."""
        if importlib.util.find_spec("onnxconverter_common") is None:
            pytest.skip("onnxconverter-common not installed")

        output_path = tmp_path / "model_fp16.onnx"
        pi.convert_to_fp16(simple_model, output_path)

        model = pi.load(str(output_path), device="cpu")
        input_data = np.random.rand(1, 3, 32, 32).astype(np.float32)
        output = model(input_data)

        assert output is not None


@pytest.mark.skipif(
    not ONNXRUNTIME_QUANT_AVAILABLE, reason="onnxruntime quantization not installed"
)
class TestQuantizationResult:
    """Test QuantizationResult dataclass."""

    @pytest.fixture
    def simple_model(self, tmp_path):
        """Create a simple ONNX model for testing."""
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            pytest.skip("ONNX not installed")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
        const_tensor = helper.make_tensor("const", TensorProto.FLOAT, [1], [2.0])
        mul_node = helper.make_node("Mul", ["input", "const"], ["output"], name="mul")
        graph = helper.make_graph(
            [mul_node], "test_model", [input_tensor], [output_tensor], [const_tensor]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(model_path))
        return model_path

    def test_result_has_all_fields(self, simple_model, tmp_path):
        """Test that QuantizationResult has all expected fields."""
        output_path = tmp_path / "model_int8.onnx"

        result = pi.quantize_dynamic(simple_model, output_path)

        assert hasattr(result, "input_path")
        assert hasattr(result, "output_path")
        assert hasattr(result, "backend")
        assert hasattr(result, "method")
        assert hasattr(result, "dtype")
        assert hasattr(result, "original_size_mb")
        assert hasattr(result, "quantized_size_mb")
        assert hasattr(result, "compression_ratio")

    def test_compression_ratio_positive(self, simple_model, tmp_path):
        """Test that compression ratio is positive."""
        output_path = tmp_path / "model_int8.onnx"

        result = pi.quantize_dynamic(simple_model, output_path)

        assert result.compression_ratio > 0
        assert result.original_size_mb > 0
        assert result.quantized_size_mb > 0


class TestOpenVINOQuantization:
    """Test OpenVINO NNCF quantization."""

    @pytest.fixture
    def simple_model(self, tmp_path):
        """Create a simple ONNX model for testing."""
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            pytest.skip("ONNX not installed")

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
        const_tensor = helper.make_tensor("const", TensorProto.FLOAT, [1], [2.0])
        mul_node = helper.make_node("Mul", ["input", "const"], ["output"], name="mul")
        graph = helper.make_graph(
            [mul_node], "test_model", [input_tensor], [output_tensor], [const_tensor]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        model_path = tmp_path / "test_model.onnx"
        onnx.save(model, str(model_path))
        return model_path

    @pytest.fixture
    def calibration_data(self):
        """Create calibration data."""
        return [np.random.rand(1, 3, 32, 32).astype(np.float32) for _ in range(10)]

    @pytest.mark.openvino
    def test_openvino_quantization(self, simple_model, tmp_path, calibration_data):
        """Test OpenVINO NNCF quantization."""
        if importlib.util.find_spec("nncf") is None or importlib.util.find_spec("openvino") is None:
            pytest.skip("OpenVINO/NNCF not installed")

        output_path = tmp_path / "model_int8.onnx"

        result = pi.quantize(
            simple_model,
            output_path,
            method="static",
            calibration_data=calibration_data,
            backend="openvino",
        )

        assert result.backend == "openvino"


class TestErrorHandling:
    """Test error handling in quantization."""

    def test_model_not_found(self, tmp_path):
        """Test error when model file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            pi.quantize(
                tmp_path / "nonexistent.onnx",
                tmp_path / "output.onnx",
            )

    def test_invalid_method(self, tmp_path):
        """Test error for invalid quantization method."""
        # Create a dummy file
        model_path = tmp_path / "dummy.onnx"
        model_path.write_bytes(b"dummy")

        with pytest.raises(ValueError):
            pi.quantize(
                model_path,
                tmp_path / "output.onnx",
                method="invalid_method",
            )

    def test_invalid_dtype(self, tmp_path):
        """Test error for invalid dtype."""
        model_path = tmp_path / "dummy.onnx"
        model_path.write_bytes(b"dummy")

        with pytest.raises(ValueError):
            pi.quantize(
                model_path,
                tmp_path / "output.onnx",
                dtype="invalid_dtype",
            )

    def test_invalid_backend(self, tmp_path):
        """Test error for invalid backend."""
        model_path = tmp_path / "dummy.onnx"
        model_path.write_bytes(b"dummy")

        with pytest.raises(ValueError, match="Unknown quantization backend"):
            pi.quantize(
                model_path,
                tmp_path / "output.onnx",
                backend="invalid_backend",
            )
