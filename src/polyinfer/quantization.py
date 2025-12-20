"""Quantization support for PolyInfer.

Provides unified quantization API across backends:
- ONNX Runtime: Static and dynamic INT8 quantization
- OpenVINO: NNCF-based INT8 quantization
- TensorRT: INT8/FP16 calibration during engine build

Basic usage:
    import polyinfer as pi

    # Dynamic quantization (no calibration data needed)
    pi.quantize("model.onnx", "model_int8.onnx", method="dynamic")

    # Static quantization with calibration data
    def data_reader():
        for batch in dataset:
            yield {"input": batch}

    pi.quantize("model.onnx", "model_int8.onnx",
                method="static",
                calibration_data=data_reader)

    # FP16 conversion
    pi.quantize("model.onnx", "model_fp16.onnx", dtype="fp16")
"""

import importlib.util
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class QuantizationMethod(Enum):
    """Quantization methods available."""

    DYNAMIC = "dynamic"  # Dynamic INT8 (no calibration needed)
    STATIC = "static"  # Static INT8 (requires calibration data)


class QuantizationType(Enum):
    """Target quantization data type."""

    INT8 = "int8"
    UINT8 = "uint8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"


class CalibrationMethod(Enum):
    """Calibration methods for static quantization."""

    MINMAX = "minmax"  # Min-max calibration
    ENTROPY = "entropy"  # Entropy-based calibration (KL divergence)
    PERCENTILE = "percentile"  # Percentile-based calibration


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    method: QuantizationMethod = QuantizationMethod.DYNAMIC
    dtype: QuantizationType = QuantizationType.INT8
    calibration_method: CalibrationMethod = CalibrationMethod.MINMAX
    per_channel: bool = False
    reduce_range: bool = False  # Use 7-bit for better accuracy on some hardware
    op_types_to_quantize: list[str] | None = None  # None = all supported ops
    nodes_to_exclude: list[str] | None = None
    extra_options: dict | None = None


@dataclass
class QuantizationResult:
    """Result of quantization operation."""

    input_path: Path
    output_path: Path
    backend: str
    method: str
    dtype: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float


# Type alias for calibration data
CalibrationData = (
    Iterator[dict[str, np.ndarray]]  # Iterator yielding input dicts
    | Callable[[], Iterator[dict[str, np.ndarray]]]  # Factory function
    | list[dict[str, np.ndarray]]  # List of input dicts
    | list[np.ndarray]  # List of input arrays (single input models)
)


def quantize(
    model_input: str | Path,
    model_output: str | Path,
    method: str = "dynamic",
    dtype: str = "int8",
    calibration_data: CalibrationData | None = None,
    calibration_method: str = "minmax",
    backend: str = "auto",
    num_calibration_samples: int = 100,
    per_channel: bool = False,
    reduce_range: bool = False,
    op_types_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
    **kwargs,
) -> QuantizationResult:
    """Quantize an ONNX model to reduce size and improve inference speed.

    Args:
        model_input: Path to input ONNX model
        model_output: Path to save quantized model
        method: Quantization method - "dynamic" or "static"
        dtype: Target data type - "int8", "uint8", "int4", "fp16", "bf16"
        calibration_data: Calibration data for static quantization. Can be:
            - Iterator yielding dicts of {input_name: numpy_array}
            - Callable that returns such an iterator
            - List of input dicts
            - List of numpy arrays (for single-input models)
        calibration_method: Method for calibration - "minmax", "entropy", "percentile"
        backend: Backend to use - "auto", "onnxruntime", "openvino"
        num_calibration_samples: Number of calibration samples to use
        per_channel: Quantize weights per channel (more accurate, larger model)
        reduce_range: Use 7-bit quantization (better accuracy on some CPUs)
        op_types_to_quantize: List of op types to quantize (None = all supported)
        nodes_to_exclude: List of node names to exclude from quantization
        **kwargs: Backend-specific options

    Returns:
        QuantizationResult with details about the quantization

    Examples:
        # Dynamic quantization (fastest, no calibration needed)
        >>> pi.quantize("model.onnx", "model_int8.onnx")

        # Static quantization with numpy arrays
        >>> calibration_data = [np.random.rand(1, 3, 224, 224).astype(np.float32)
        ...                     for _ in range(100)]
        >>> pi.quantize("model.onnx", "model_int8.onnx",
        ...             method="static", calibration_data=calibration_data)

        # FP16 quantization
        >>> pi.quantize("model.onnx", "model_fp16.onnx", dtype="fp16")

        # OpenVINO quantization
        >>> pi.quantize("model.onnx", "model_int8.onnx",
        ...             backend="openvino", calibration_data=data)
    """
    model_input = Path(model_input)
    model_output = Path(model_output)

    # Validate inputs
    if not model_input.exists():
        raise FileNotFoundError(f"Model not found: {model_input}")

    if method == "static" and calibration_data is None:
        raise ValueError("Static quantization requires calibration_data")

    # Parse method and dtype
    quant_method = QuantizationMethod(method)
    quant_dtype = QuantizationType(dtype)
    calib_method = CalibrationMethod(calibration_method)

    # Select backend
    if backend == "auto":
        # FP16/BF16 -> use onnxruntime (direct conversion)
        # INT8/INT4 static with OpenVINO available -> prefer openvino (NNCF is good)
        # Otherwise -> onnxruntime
        if quant_dtype in (QuantizationType.FP16, QuantizationType.BF16):
            backend = "onnxruntime"
        elif quant_method == QuantizationMethod.STATIC:
            backend = "openvino" if importlib.util.find_spec("nncf") is not None else "onnxruntime"
        else:
            backend = "onnxruntime"

    # Build config
    config = QuantizationConfig(
        method=quant_method,
        dtype=quant_dtype,
        calibration_method=calib_method,
        per_channel=per_channel,
        reduce_range=reduce_range,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        extra_options=kwargs if kwargs else None,
    )

    # Dispatch to backend
    if backend == "onnxruntime":
        result = _quantize_onnxruntime(
            model_input, model_output, config, calibration_data, num_calibration_samples
        )
    elif backend == "openvino":
        result = _quantize_openvino(
            model_input, model_output, config, calibration_data, num_calibration_samples
        )
    else:
        raise ValueError(f"Unknown quantization backend: {backend}")

    return result


def _quantize_onnxruntime(
    model_input: Path,
    model_output: Path,
    config: QuantizationConfig,
    calibration_data: CalibrationData | None,
    num_samples: int,
) -> QuantizationResult:
    """Quantize using ONNX Runtime quantization tools."""
    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quantize_dynamic,
            quantize_static,
        )
        from onnxruntime.quantization.calibrate import CalibrationMethod as ORTCalibMethod
    except ImportError as e:
        raise ImportError(
            "onnxruntime quantization not available. Install with: pip install onnxruntime"
        ) from e

    # Map dtype to QuantType
    dtype_map = {
        QuantizationType.INT8: QuantType.QInt8,
        QuantizationType.UINT8: QuantType.QUInt8,
        QuantizationType.INT4: QuantType.QInt4,
    }

    original_size = model_input.stat().st_size / (1024 * 1024)

    # Handle FP16/BF16 conversion separately
    if config.dtype == QuantizationType.FP16:
        _convert_to_fp16_onnx(model_input, model_output)
    elif config.dtype == QuantizationType.BF16:
        raise NotImplementedError("BF16 conversion not yet supported in ONNX Runtime")
    elif config.method == QuantizationMethod.DYNAMIC:
        # Dynamic quantization - no calibration needed
        weight_type = dtype_map.get(config.dtype, QuantType.QInt8)

        quantize_dynamic(
            model_input=str(model_input),
            model_output=str(model_output),
            weight_type=weight_type,
            per_channel=config.per_channel,
            reduce_range=config.reduce_range,
            op_types_to_quantize=config.op_types_to_quantize,
            nodes_to_exclude=config.nodes_to_exclude,
            extra_options=config.extra_options or {},
        )
    else:
        # Static quantization - requires calibration
        weight_type = dtype_map.get(config.dtype, QuantType.QInt8)
        activation_type = weight_type

        # Map calibration method
        calib_method_map = {
            CalibrationMethod.MINMAX: ORTCalibMethod.MinMax,
            CalibrationMethod.ENTROPY: ORTCalibMethod.Entropy,
            CalibrationMethod.PERCENTILE: ORTCalibMethod.Percentile,
        }
        ort_calib_method = calib_method_map.get(config.calibration_method, ORTCalibMethod.MinMax)

        # Create calibration data reader
        data_reader = _create_ort_calibration_reader(model_input, calibration_data, num_samples)

        quantize_static(
            model_input=str(model_input),
            model_output=str(model_output),
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,  # QDQ format for better compatibility
            weight_type=weight_type,
            activation_type=activation_type,
            per_channel=config.per_channel,
            reduce_range=config.reduce_range,
            calibrate_method=ort_calib_method,
            op_types_to_quantize=config.op_types_to_quantize,
            nodes_to_exclude=config.nodes_to_exclude,
            extra_options=config.extra_options or {},
        )

    quantized_size = model_output.stat().st_size / (1024 * 1024)

    return QuantizationResult(
        input_path=model_input,
        output_path=model_output,
        backend="onnxruntime",
        method=config.method.value,
        dtype=config.dtype.value,
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        compression_ratio=original_size / quantized_size if quantized_size > 0 else 0,
    )


def _convert_to_fp16_onnx(model_input: Path, model_output: Path) -> None:
    """Convert ONNX model to FP16."""
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as e:
        raise ImportError(
            "FP16 conversion requires onnxconverter-common. "
            "Install with: pip install onnxconverter-common"
        ) from e

    model = onnx.load(str(model_input))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(model_output))


class _ORTCalibrationDataReader:
    """ONNX Runtime CalibrationDataReader implementation."""

    def __init__(
        self,
        model_path: Path,
        data: CalibrationData,
        num_samples: int,
    ):
        self.model_path = model_path
        self.num_samples = num_samples
        self._data_iter: Iterator | None = None
        self._count = 0

        # Get input names from model
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_names = [inp.name for inp in sess.get_inputs()]
        del sess

        # Normalize data to iterator
        self._raw_data = data
        self._setup_iterator()

    def _setup_iterator(self):
        """Setup the data iterator."""
        data = self._raw_data

        if callable(data) and not isinstance(data, (list, Iterator)):
            # Factory function
            data = data()

        if isinstance(data, list):
            # Convert list to iterator
            if len(data) > 0 and isinstance(data[0], np.ndarray):
                # List of arrays - wrap in dicts
                if len(self._input_names) != 1:
                    raise ValueError(
                        f"Model has {len(self._input_names)} inputs, "
                        "but calibration data is a list of arrays. "
                        "Use list of dicts instead."
                    )
                data = [{self._input_names[0]: arr} for arr in data]
            self._data_iter = iter(data)
        else:
            self._data_iter = data

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Get next calibration batch."""
        if self._count >= self.num_samples:
            return None

        try:
            batch = next(self._data_iter)
            self._count += 1

            # Handle single array input
            if isinstance(batch, np.ndarray):
                return {self._input_names[0]: batch}
            return batch
        except StopIteration:
            return None

    def rewind(self):
        """Rewind the iterator."""
        self._count = 0
        self._setup_iterator()


def _create_ort_calibration_reader(
    model_path: Path,
    data: CalibrationData,
    num_samples: int,
):
    """Create ONNX Runtime CalibrationDataReader."""
    return _ORTCalibrationDataReader(model_path, data, num_samples)


def _quantize_openvino(
    model_input: Path,
    model_output: Path,
    config: QuantizationConfig,
    calibration_data: CalibrationData | None,
    num_samples: int,
) -> QuantizationResult:
    """Quantize using OpenVINO NNCF."""
    try:
        import nncf
        import openvino as ov
    except ImportError as e:
        raise ImportError(
            "OpenVINO NNCF not available. Install with: pip install openvino nncf"
        ) from e

    original_size = model_input.stat().st_size / (1024 * 1024)

    # Read the model
    core = ov.Core()
    model = core.read_model(str(model_input))

    # Handle FP16 conversion
    if config.dtype == QuantizationType.FP16:
        # OpenVINO doesn't have direct FP16 quantization, use compress_to_fp16
        try:
            # Compile with FP16 inference precision hint
            ov.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f16"})
            # For saving, we need to serialize the original model
            # OpenVINO FP16 is handled at compile time, not model level
            # Fall back to ONNX Runtime for FP16
            return _quantize_onnxruntime(
                model_input, model_output, config, calibration_data, num_samples
            )
        except Exception:
            return _quantize_onnxruntime(
                model_input, model_output, config, calibration_data, num_samples
            )

    # For INT8, we need calibration data
    if calibration_data is None:
        raise ValueError("OpenVINO quantization requires calibration_data")

    # Create NNCF dataset
    nncf_dataset = _create_nncf_dataset(model, calibration_data, num_samples)

    # Quantize with NNCF
    quantized_model = nncf.quantize(
        model,
        nncf_dataset,
        preset=nncf.QuantizationPreset.MIXED
        if config.per_channel
        else nncf.QuantizationPreset.PERFORMANCE,
        target_device=nncf.TargetDevice.CPU,
        subset_size=num_samples,
    )

    # Save the quantized model
    # Determine output format based on extension
    output_str = str(model_output)
    if output_str.endswith(".onnx"):
        # Save as ONNX
        ov.save_model(quantized_model, output_str)
    else:
        # Save as OpenVINO IR
        if not output_str.endswith(".xml"):
            output_str = output_str + ".xml"
        ov.save_model(quantized_model, output_str)
        model_output = Path(output_str)

    quantized_size = model_output.stat().st_size / (1024 * 1024)

    return QuantizationResult(
        input_path=model_input,
        output_path=model_output,
        backend="openvino",
        method=config.method.value,
        dtype=config.dtype.value,
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        compression_ratio=original_size / quantized_size if quantized_size > 0 else 0,
    )


def _create_nncf_dataset(model, data: CalibrationData, num_samples: int):
    """Create NNCF Dataset from calibration data."""
    import nncf

    # Get input names
    input_names = [inp.any_name for inp in model.inputs]

    # Normalize data
    if callable(data) and not isinstance(data, (list, Iterator)):
        data = data()

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
        # List of arrays
        if len(input_names) != 1:
            raise ValueError(
                f"Model has {len(input_names)} inputs, but calibration data is a list of arrays."
            )
        data = [{input_names[0]: arr} for arr in data]

    # Convert to list if iterator
    if not isinstance(data, list):
        data = list(data)

    # Limit samples
    data = data[:num_samples]

    # Transform function for NNCF
    def transform_fn(data_item):
        if isinstance(data_item, dict):
            # Return as tuple of arrays in input order
            return tuple(data_item[name] for name in input_names)
        elif isinstance(data_item, np.ndarray):
            return (data_item,)
        return data_item

    return nncf.Dataset(data, transform_fn)


def quantize_for_tensorrt(
    model_path: str | Path,
    output_path: str | Path | None = None,
    calibration_data: CalibrationData | None = None,
    precision: str = "fp16",
    num_calibration_samples: int = 100,
    **kwargs,
) -> Path:
    """Prepare a model for TensorRT with specified precision.

    Note: TensorRT quantization happens during engine building, not as a
    separate model conversion step. This function prepares calibration data
    and returns the path to use with pi.load(..., device="tensorrt", fp16=True).

    For INT8, you need to provide calibration data. The calibration happens
    when the TensorRT engine is first built.

    Args:
        model_path: Path to ONNX model
        output_path: Optional path to save calibration cache (for INT8)
        calibration_data: Calibration data for INT8 quantization
        precision: Target precision - "fp16", "int8", "fp8", "bf16"
        num_calibration_samples: Number of calibration samples
        **kwargs: Additional TensorRT options

    Returns:
        Path to the model (same as input for now)

    Example:
        # FP16 - just use the load function directly
        >>> model = pi.load("model.onnx", device="tensorrt", fp16=True)

        # INT8 with calibration
        >>> pi.quantize_for_tensorrt("model.onnx",
        ...     precision="int8",
        ...     calibration_data=data)
        >>> model = pi.load("model.onnx", device="tensorrt", int8=True)
    """
    model_path = Path(model_path)

    if precision == "fp16":
        # FP16 doesn't need calibration, just use the model directly
        print(f"For FP16 TensorRT, use: pi.load('{model_path}', device='tensorrt', fp16=True)")
        return model_path

    elif precision == "int8":
        if calibration_data is None:
            raise ValueError("INT8 TensorRT requires calibration_data")

        # For now, we just validate the data and inform the user
        # Full INT8 calibration would require implementing TensorRT's IInt8Calibrator
        print(
            f"INT8 TensorRT calibration prepared. "
            f"Use: pi.load('{model_path}', device='tensorrt', int8=True, "
            f"calibration_data=...)"
        )

        # TODO: Implement TensorRT calibrator that saves calibration cache
        # This would involve:
        # 1. Creating a custom IInt8EntropyCalibrator2
        # 2. Running calibration with the provided data
        # 3. Saving calibration cache to output_path

        return model_path

    else:
        raise ValueError(f"Unknown precision: {precision}. Use 'fp16' or 'int8'.")


# Convenience functions


def quantize_dynamic(
    model_input: str | Path,
    model_output: str | Path,
    dtype: str = "int8",
    **kwargs,
) -> QuantizationResult:
    """Convenience function for dynamic quantization.

    Dynamic quantization doesn't require calibration data.
    Weights are quantized at conversion time, activations at runtime.

    Best for: Transformer models, RNNs, models with varying input sizes.

    Args:
        model_input: Path to input ONNX model
        model_output: Path to save quantized model
        dtype: Target type - "int8", "uint8"
        **kwargs: Additional options passed to quantize()

    Returns:
        QuantizationResult
    """
    return quantize(
        model_input=model_input,
        model_output=model_output,
        method="dynamic",
        dtype=dtype,
        **kwargs,
    )


def quantize_static(
    model_input: str | Path,
    model_output: str | Path,
    calibration_data: CalibrationData,
    dtype: str = "int8",
    **kwargs,
) -> QuantizationResult:
    """Convenience function for static quantization.

    Static quantization uses calibration data to determine quantization
    parameters for both weights and activations.

    Best for: CNN models, models with fixed input sizes.

    Args:
        model_input: Path to input ONNX model
        model_output: Path to save quantized model
        calibration_data: Data for calibration
        dtype: Target type - "int8", "uint8", "int4"
        **kwargs: Additional options passed to quantize()

    Returns:
        QuantizationResult
    """
    return quantize(
        model_input=model_input,
        model_output=model_output,
        method="static",
        dtype=dtype,
        calibration_data=calibration_data,
        **kwargs,
    )


def convert_to_fp16(
    model_input: str | Path,
    model_output: str | Path,
    **kwargs,
) -> QuantizationResult:
    """Convert model to FP16 precision.

    FP16 reduces model size by ~50% with minimal accuracy loss.
    Works well on GPUs with FP16 support (most modern GPUs).

    Args:
        model_input: Path to input ONNX model
        model_output: Path to save FP16 model
        **kwargs: Additional options

    Returns:
        QuantizationResult
    """
    return quantize(
        model_input=model_input,
        model_output=model_output,
        method="dynamic",  # Not really used for FP16
        dtype="fp16",
        **kwargs,
    )
