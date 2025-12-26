"""IREE backend for PolyInfer (Tier 2).

Google's IREE compiler and runtime supporting:
- CPU (llvm-cpu)
- Vulkan (vulkan-spirv) with GPU-specific targets (AMD, NVIDIA, Intel, ARM, Qualcomm)
- CUDA (cuda)

MLIR emission for custom hardware support:
- emit_mlir() to convert ONNX to IREE MLIR
- compile_mlir() to compile MLIR to VMFB
- load_vmfb() to load pre-compiled VMFB files

Vulkan GPU targets:
- list_vulkan_targets() to see all supported GPU presets
- Pass vulkan_target='rtx4090' (or 'rdna3', 'ampere', etc.) to load()

Reference: https://iree.dev/
"""

from polyinfer.backends.iree.backend import (
    DEVICE_TO_DRIVER,
    DEVICE_TO_TARGET,
    VULKAN_TARGETS,
    IREEBackend,
    IREECompilationError,
    IREECompileOptions,
    IREEModel,
    MLIROutput,
    VulkanGPUVendor,
    VulkanTarget,
)

__all__ = [
    # Backend and model
    "IREEBackend",
    "IREEModel",
    # MLIR output
    "MLIROutput",
    # Compilation options
    "IREECompileOptions",
    "IREECompilationError",
    # Vulkan targets
    "VulkanTarget",
    "VulkanGPUVendor",
    "VULKAN_TARGETS",
    # Device mappings
    "DEVICE_TO_TARGET",
    "DEVICE_TO_DRIVER",
]
