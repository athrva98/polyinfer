"""MLIR emission and compilation utilities.

This module provides high-level functions for working with MLIR,
enabling custom hardware support and kernel injection workflows.

Example:
    >>> import polyinfer as pi
    >>>
    >>> # Export ONNX model to MLIR
    >>> mlir = pi.export_mlir("model.onnx", "model.mlir")
    >>> print(mlir.path)
    model.mlir
    >>>
    >>> # Compile MLIR for a specific device
    >>> vmfb_path = pi.compile_mlir("model.mlir", device="vulkan")
    >>>
    >>> # Load and run. A compiled .vmfb must go through the IREE backend's
    >>> # load_vmfb(); pi.load() expects an ONNX model and would try to
    >>> # recompile it.
    >>> model = pi.get_backend("iree").load_vmfb(vmfb_path, device="vulkan")
    >>> output = model(input_data)
"""

from pathlib import Path
from typing import TYPE_CHECKING

from polyinfer.exceptions import BackendNotAvailableError

if TYPE_CHECKING:
    from polyinfer.backends.iree.backend import MLIROutput


def export_mlir(
    model_path: str | Path,
    output_path: str | Path | None = None,
    *,
    opset_version: int | None = None,
    load_content: bool = False,
) -> "MLIROutput":
    """Convert an ONNX model to IREE MLIR.

    This is the first step in the MLIR compilation pipeline, useful for:
    - Inspecting the intermediate representation
    - Custom kernel injection
    - MLIR pass analysis and transformation
    - Targeting custom hardware accelerators

    Args:
        model_path: Path to ONNX file
        output_path: Where to save the MLIR file. If None, saves alongside
                    the ONNX file with .mlir extension.
        opset_version: Upgrade the ONNX model to this opset before import.
                    Useful when IREE fails to legalize an older operator.
        load_content: If True, also load MLIR content into memory

    Returns:
        MLIROutput containing path and optionally content

    Raises:
        BackendNotAvailableError: If IREE is not available. Also a RuntimeError.
        CompilationError: If conversion fails. Also a RuntimeError.
        FileNotFoundError: If the model file doesn't exist

    Example:
        >>> import polyinfer as pi
        >>>
        >>> # Basic export
        >>> mlir = pi.export_mlir("model.onnx")
        >>> print(mlir.path)  # model.mlir
        >>>
        >>> # Export with custom path
        >>> mlir = pi.export_mlir("model.onnx", "output/model_v1.mlir")
        >>>
        >>> # Export and load content for inspection
        >>> mlir = pi.export_mlir("model.onnx", load_content=True)
        >>> print(mlir.content[:200])  # First 200 chars of MLIR
        >>>
        >>> # Save to another location
        >>> mlir.save("backup/model.mlir")
    """
    from polyinfer.backends.iree.backend import IREEBackend

    # Validate inputs before checking optional dependencies, so a mistyped
    # path reports the missing file rather than "IREE is not available".
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    backend = IREEBackend()

    if not backend.is_available():
        raise BackendNotAvailableError(
            f"IREE backend is not available: {backend.unavailable_reason}\n"
            "Install with: pip install iree-base-compiler iree-base-runtime"
        )

    return backend.emit_mlir(
        str(model_path),
        output_path,
        opset_version=opset_version,
        load_content=load_content,
    )


def compile_mlir(
    mlir_path: str | Path,
    device: str = "cpu",
    output_path: str | Path | None = None,
    *,
    opt_level: int = 2,
    **kwargs,
) -> Path:
    """Compile an MLIR file to executable VMFB.

    This is the second step in the MLIR compilation pipeline. It takes
    MLIR (possibly modified with custom kernels) and produces a VMFB
    binary that can be loaded for inference.

    Args:
        mlir_path: Path to MLIR file
        device: Target device (cpu, vulkan, cuda)
        output_path: Where to save the VMFB. If None, saves alongside
                    the MLIR file with .vmfb extension.
        opt_level: Optimization level (0-3, default: 2)
        **kwargs: Additional compilation options forwarded to the IREE
                    backend (see IREECompileOptions):
                    - vulkan_target: GPU target for device="vulkan", either a
                      preset name ('rtx4090', 'rdna3') or a raw IREE target
                    - data_tiling: Enable data tiling optimization
                    - extra_flags: Raw flags appended to iree-compile

    Returns:
        Path to compiled VMFB file

    Raises:
        BackendNotAvailableError: If IREE is not available. Also a RuntimeError.
        CompilationError: If compilation fails. Also a RuntimeError.
        FileNotFoundError: If the MLIR file doesn't exist

    Example:
        >>> import polyinfer as pi
        >>>
        >>> # Compile for CPU
        >>> vmfb = pi.compile_mlir("model.mlir", device="cpu")
        >>>
        >>> # Compile for Vulkan with high optimization
        >>> vmfb = pi.compile_mlir("model.mlir", device="vulkan", opt_level=3)
        >>>
        >>> # Compile for a specific GPU architecture
        >>> vmfb = pi.compile_mlir("model.mlir", device="vulkan",
        ...                        vulkan_target="rdna3", opt_level=3)
        >>>
        >>> # Load and run. Note: a .vmfb is loaded via the IREE backend's
        >>> # load_vmfb(), not pi.load(), which expects an ONNX model.
        >>> model = pi.get_backend("iree").load_vmfb(vmfb, device="vulkan")
        >>> output = model(input_data)
    """
    from polyinfer.backends.iree.backend import IREEBackend

    # Validate inputs before checking optional dependencies, so a mistyped
    # path reports the missing file rather than "IREE is not available".
    if not Path(mlir_path).exists():
        raise FileNotFoundError(f"MLIR file not found: {mlir_path}")

    backend = IREEBackend()

    if not backend.is_available():
        raise BackendNotAvailableError(
            f"IREE backend is not available: {backend.unavailable_reason}\n"
            "Install with: pip install iree-base-compiler iree-base-runtime"
        )

    return backend.compile_mlir(
        mlir_path,
        device=device,
        output_path=output_path,
        opt_level=opt_level,
        **kwargs,
    )


def __getattr__(name: str):
    """Lazy import for MLIROutput to avoid loading IREE at module import time."""
    if name == "MLIROutput":
        from polyinfer.backends.iree.backend import MLIROutput

        return MLIROutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["export_mlir", "compile_mlir", "MLIROutput"]
