"""Auto-load and register available backends."""

import sys
from polyinfer.backends.registry import register_backend


def _should_skip_native_tensorrt() -> bool:
    """Check if native TensorRT import should be skipped to avoid conflicts.

    The native TensorRT backend imports cuda.bindings/cuda.cudart which can
    load CUDA libraries that conflict with PyTorch's bundled libraries.
    This causes 'undefined symbol: ncclCommWindowRegister' errors.

    We skip native TensorRT if:
    1. PyTorch is already loaded (torch in sys.modules)
    2. We're on Linux (where the conflicts are most severe)

    Users who want native TensorRT can still use it by:
    - Importing polyinfer before torch
    - Using backend="tensorrt" explicitly after ensuring no conflicts
    - Using ONNX Runtime's TensorRT EP instead (recommended, no conflicts)
    """
    # Skip if PyTorch is already imported to avoid loading conflicting CUDA libs
    if "torch" in sys.modules:
        return True

    # On Linux, the cuda.bindings import can pollute the process with
    # incompatible CUDA libraries. Skip by default.
    if sys.platform.startswith("linux"):
        return True

    return False


def register_all():
    """Register all available backends.

    This function attempts to import and register each backend.
    Backends that fail to import (missing dependencies) are silently skipped.
    """
    # ONNX Runtime backend (Tier 1)
    try:
        from polyinfer.backends.onnxruntime import ONNXRuntimeBackend

        register_backend("onnxruntime", ONNXRuntimeBackend)
    except ImportError:
        pass

    # OpenVINO backend (Tier 1)
    try:
        from polyinfer.backends.openvino import OpenVINOBackend

        register_backend("openvino", OpenVINOBackend)
    except ImportError:
        pass

    # TensorRT backend (Tier 2)
    # Skip on Linux or if PyTorch is loaded to avoid CUDA library conflicts.
    # Users can still use ONNX Runtime's TensorRT EP (device="tensorrt").
    if not _should_skip_native_tensorrt():
        try:
            from polyinfer.backends.tensorrt import TensorRTBackend

            register_backend("tensorrt", TensorRTBackend)
        except ImportError:
            pass

    # IREE backend (Tier 2)
    try:
        from polyinfer.backends.iree import IREEBackend

        register_backend("iree", IREEBackend)
    except ImportError:
        pass
