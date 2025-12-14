"""ONNX Runtime backend for PolyInfer.

Supports multiple execution providers:
- CPUExecutionProvider: Default CPU inference
- CUDAExecutionProvider: NVIDIA GPU via CUDA
- TensorrtExecutionProvider: NVIDIA GPU via TensorRT
- DmlExecutionProvider: DirectML for Windows (AMD, Intel, NVIDIA)
"""

from polyinfer.backends.onnxruntime.backend import ONNXRuntimeBackend, ONNXRuntimeModel

__all__ = ["ONNXRuntimeBackend", "ONNXRuntimeModel"]
