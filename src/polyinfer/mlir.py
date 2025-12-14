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
    >>> # Load and run
    >>> model = pi.load(vmfb_path, device="vulkan")
    >>> output = model(input_data)
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polyinfer.backends.iree.backend import MLIROutput


def export_mlir(
    model_path: str | Path,
    output_path: str | Path | None = None,
    *,
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
        load_content: If True, also load MLIR content into memory

    Returns:
        MLIROutput containing path and optionally content

    Raises:
        RuntimeError: If IREE is not available or conversion fails
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

    backend = IREEBackend()

    if not backend.is_available():
        raise RuntimeError(
            "IREE backend is not available. Install with:\n"
            "  pip install iree-base-compiler iree-base-runtime"
        )

    return backend.emit_mlir(
        str(model_path),
        output_path,
        load_content=load_content,
    )


def compile_mlir(
    mlir_path: str | Path,
    device: str = "cpu",
    output_path: str | Path | None = None,
    *,
    opt_level: int = 2,
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

    Returns:
        Path to compiled VMFB file

    Raises:
        RuntimeError: If IREE is not available or compilation fails
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
        >>> # Load and run
        >>> model = pi.get_backend("iree").load_vmfb(vmfb, device="vulkan")
        >>> output = model(input_data)
    """
    from polyinfer.backends.iree.backend import IREEBackend

    backend = IREEBackend()

    if not backend.is_available():
        raise RuntimeError(
            "IREE backend is not available. Install with:\n"
            "  pip install iree-base-compiler iree-base-runtime"
        )

    return backend.compile_mlir(
        mlir_path,
        device=device,
        output_path=output_path,
        opt_level=opt_level,
    )


# Re-export MLIROutput for convenience
def __getattr__(name: str):
    if name == "MLIROutput":
        from polyinfer.backends.iree.backend import MLIROutput
        return MLIROutput
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For static type checkers
from polyinfer.backends.iree.backend import MLIROutput

__all__ = ["export_mlir", "compile_mlir", "MLIROutput"]
