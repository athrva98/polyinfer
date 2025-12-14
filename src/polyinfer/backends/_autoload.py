"""Auto-load and register available backends."""

from polyinfer.backends.registry import register_backend


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
