"""TensorRT backend for PolyInfer (Tier 2).

Native TensorRT backend for maximum performance on NVIDIA GPUs.
Requires:
- CUDA Toolkit
- TensorRT library
- tensorrt Python package
"""

from polyinfer.backends.tensorrt.backend import TensorRTBackend, TensorRTModel

__all__ = ["TensorRTBackend", "TensorRTModel"]
