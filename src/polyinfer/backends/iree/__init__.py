"""IREE backend for PolyInfer (Tier 2).

Google's IREE compiler and runtime supporting:
- CPU (llvm-cpu)
- Vulkan (vulkan-spirv)
- CUDA (cuda)

MLIR emission for custom hardware support:
- emit_mlir() to convert ONNX to IREE MLIR
- compile_mlir() to compile MLIR to VMFB
- load_vmfb() to load pre-compiled VMFB files
"""

from polyinfer.backends.iree.backend import (
    DEVICE_TO_DRIVER,
    DEVICE_TO_TARGET,
    IREEBackend,
    IREEModel,
    MLIROutput,
)

__all__ = [
    "IREEBackend",
    "IREEModel",
    "MLIROutput",
    "DEVICE_TO_TARGET",
    "DEVICE_TO_DRIVER",
]
