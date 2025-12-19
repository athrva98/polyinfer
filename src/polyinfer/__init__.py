"""PolyInfer: Unified ML inference across multiple backends.

Supports:
- ONNX Runtime (CPU, CUDA, DirectML, TensorRT EP)
- OpenVINO (CPU, Intel GPU)
- TensorRT (native)
- IREE (CPU, Vulkan, CUDA)

Basic usage:
    import polyinfer as pi

    # Auto-select best backend
    model = pi.load("model.onnx", device="cuda")
    output = model(input_tensor)

    # Explicit backend
    model = pi.load("model.onnx", backend="openvino", device="cpu")

    # List available backends
    print(pi.list_backends())
    print(pi.list_devices())

    # Export MLIR for custom hardware
    mlir = pi.export_mlir("model.onnx", "model.mlir")
"""

__version__ = "0.1.0"

# Auto-setup NVIDIA libraries BEFORE importing anything else
# This ensures CUDA, cuDNN, TensorRT DLLs are findable
from polyinfer import nvidia_setup as _nvidia_setup
from polyinfer.nvidia_setup import fix_onnxruntime_conflict, get_nvidia_info, setup_tensorrt_paths

from polyinfer.model import load, Model
from polyinfer.discovery import (
    list_backends,
    list_devices,
    get_backend,
    is_available,
)
from polyinfer.config import InferenceConfig
from polyinfer.compare import compare, benchmark
from polyinfer.mlir import export_mlir, compile_mlir, MLIROutput
from polyinfer.quantization import (
    quantize,
    quantize_dynamic,
    quantize_static,
    convert_to_fp16,
    quantize_for_tensorrt,
    QuantizationResult,
    QuantizationConfig,
    QuantizationMethod,
    QuantizationType,
    CalibrationMethod,
)
from polyinfer._logging import (
    get_logger,
    set_log_level,
    get_log_level,
    get_log_level_name,
    enable_logging,
    disable_logging,
    configure_logging,
    LogContext,
)

__all__ = [
    # Core API
    "load",
    "Model",
    # Discovery
    "list_backends",
    "list_devices",
    "get_backend",
    "is_available",
    # Config
    "InferenceConfig",
    # Utilities
    "compare",
    "benchmark",
    # MLIR
    "export_mlir",
    "compile_mlir",
    "MLIROutput",
    # Quantization
    "quantize",
    "quantize_dynamic",
    "quantize_static",
    "convert_to_fp16",
    "quantize_for_tensorrt",
    "QuantizationResult",
    "QuantizationConfig",
    "QuantizationMethod",
    "QuantizationType",
    "CalibrationMethod",
    # Setup helpers
    "fix_onnxruntime_conflict",
    "get_nvidia_info",
    "setup_tensorrt_paths",
    # Logging
    "get_logger",
    "set_log_level",
    "get_log_level",
    "get_log_level_name",
    "enable_logging",
    "disable_logging",
    "configure_logging",
    "LogContext",
    # Version
    "__version__",
]
