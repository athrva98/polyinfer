"""IREE backend implementation."""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polyinfer._logging import get_logger
from polyinfer.backends.base import Backend, CompiledModel

_logger = get_logger("backends.iree")


@dataclass
class MLIROutput:
    """Container for emitted MLIR output.

    Attributes:
        path: Path to the saved MLIR file
        content: MLIR content as string (if loaded)
        source_model: Path to the source ONNX model
        dialect: The MLIR dialect used (e.g., 'iree')
    """

    path: Path
    content: str | None = None
    source_model: Path | None = None
    dialect: str = "iree"

    def __str__(self) -> str:
        if self.content:
            return self.content
        return self.path.read_text()

    def save(self, output_path: str | Path) -> Path:
        """Save MLIR to a new location.

        Args:
            output_path: Destination path for the MLIR file

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.content:
            output_path.write_text(self.content)
        else:
            shutil.copy(self.path, output_path)

        return output_path


# Check if IREE is available
try:
    import iree.runtime as iree_rt

    IREE_RUNTIME_AVAILABLE = True
    _logger.debug("IREE Runtime available")
except ImportError:
    IREE_RUNTIME_AVAILABLE = False
    iree_rt = None
    _logger.debug("IREE Runtime not installed")

try:
    import iree.compiler as iree_compiler

    IREE_COMPILER_AVAILABLE = True
    _logger.debug("IREE Compiler available")
except ImportError:
    IREE_COMPILER_AVAILABLE = False
    iree_compiler = None
    _logger.debug("IREE Compiler not installed")


def _find_iree_tool(tool_name: str) -> str | None:
    """Find IREE CLI tool, checking Python Scripts directory first."""
    # On Windows, pip installs scripts to Scripts/ in the Python prefix
    # On Linux/Mac, it's bin/
    if sys.platform == "win32":
        scripts_dir = Path(sys.prefix) / "Scripts"
        tool_path = scripts_dir / f"{tool_name}.exe"
    else:
        scripts_dir = Path(sys.prefix) / "bin"
        tool_path = scripts_dir / tool_name

    if tool_path.exists():
        return str(tool_path)

    # Try conda environment
    conda_prefix = Path(sys.prefix)
    if sys.platform == "win32":
        conda_tool = conda_prefix / "Scripts" / f"{tool_name}.exe"
    else:
        conda_tool = conda_prefix / "bin" / tool_name

    if conda_tool.exists():
        return str(conda_tool)

    # Fall back to PATH
    found = shutil.which(tool_name)
    if found:
        return found

    return None


def _get_iree_import_onnx() -> str | None:
    """Get path to iree-import-onnx tool."""
    return _find_iree_tool("iree-import-onnx")


def _get_iree_compile() -> str | None:
    """Get path to iree-compile tool."""
    return _find_iree_tool("iree-compile")


# Map device to IREE target
DEVICE_TO_TARGET = {
    "cpu": "llvm-cpu",
    "vulkan": "vulkan-spirv",
    "cuda": "cuda",
}

DEVICE_TO_DRIVER = {
    "cpu": "local-task",
    "vulkan": "vulkan",
    "cuda": "cuda",
}


class IREEModel(CompiledModel):
    """IREE compiled module wrapper."""

    # Common function names IREE generates from ONNX models
    FUNC_NAMES = ["main_graph", "main", "forward", "run", "inference"]

    def __init__(
        self,
        vmfb_path: Path,
        device_name: str,
        input_names: list[str],
        output_names: list[str],
    ):
        self._vmfb_path = vmfb_path
        self._device_name = device_name
        self._input_names = input_names
        self._output_names = output_names

        # Get the driver name for this device
        device_type = device_name.split(":")[0] if ":" in device_name else device_name
        driver = DEVICE_TO_DRIVER.get(device_type, "local-task")

        # Load the module using the simpler BoundModule API
        self._module = iree_rt.load_vm_flatbuffer_file(str(vmfb_path), driver=driver)

        # Find the main inference function
        self._func = None
        for func_name in self.FUNC_NAMES:
            try:
                self._func = self._module[func_name]
                self._func_name = func_name
                break
            except KeyError:
                continue

        if self._func is None:
            raise RuntimeError(
                f"Could not find inference function in IREE module. "
                f"Tried: {self.FUNC_NAMES}. Module: {self._module}"
            )

    @property
    def backend_name(self) -> str:
        return f"iree-{self._device_name}"

    @property
    def device(self) -> str:
        return self._device_name

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run inference."""
        # Ensure inputs are contiguous float32
        inputs = tuple(np.ascontiguousarray(inp, dtype=np.float32) for inp in inputs)

        # Run inference
        if self._func is None:
            raise RuntimeError("Model function not initialized")
        outputs = self._func(*inputs)

        # Convert outputs to numpy
        if isinstance(outputs, (list, tuple)):
            results = [np.asarray(o) for o in outputs]
            if len(results) == 1:
                return results[0]
            return tuple(results)
        return np.asarray(outputs)


class IREEBackend(Backend):
    """IREE backend supporting CPU, Vulkan, and CUDA."""

    @property
    def name(self) -> str:
        return "iree"

    @property
    def supported_devices(self) -> list[str]:
        if not IREE_RUNTIME_AVAILABLE:
            return []

        devices = ["cpu"]  # Always available

        # Check for Vulkan
        try:
            iree_rt.Config(driver_name="vulkan")
            devices.append("vulkan")
        except Exception:
            pass

        # Check for CUDA
        try:
            iree_rt.Config(driver_name="cuda")
            devices.append("cuda")
        except Exception:
            pass

        return devices

    @property
    def version(self) -> str:
        if IREE_RUNTIME_AVAILABLE:
            return getattr(iree_rt, "__version__", "unknown")
        return "not installed"

    @property
    def priority(self) -> int:
        # IREE is lower priority (experimental)
        return 40

    def is_available(self) -> bool:
        """Check if IREE is available (runtime + compiler tools)."""
        if not IREE_RUNTIME_AVAILABLE:
            return False

        # Need compiler tools or CLI tools as fallback
        return IREE_COMPILER_AVAILABLE or bool(_get_iree_import_onnx() and _get_iree_compile())

    def load(
        self,
        model_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> IREEModel:
        """Load an ONNX model with IREE.

        Args:
            model_path: Path to ONNX file
            device: Target device (cpu, vulkan, cuda)
            **kwargs: Additional options:
                - opt_level: Optimization level (0-3)
                - cache_dir: Directory for compiled artifacts
                - force_compile: Force recompilation
                - save_mlir: Save intermediate MLIR to cache_dir (default: False)
                - mlir_path: Custom path to save MLIR file

        Returns:
            Compiled IREE model
        """
        if not IREE_RUNTIME_AVAILABLE:
            _logger.error("IREE Runtime not installed")
            raise RuntimeError("iree-runtime not installed. Run: pip install iree-base-runtime")

        _logger.debug(f"Loading model: {model_path}")

        model_path_obj = Path(model_path)
        device_type = device.split(":")[0] if ":" in device else device

        # Determine paths
        target = DEVICE_TO_TARGET.get(device_type, "llvm-cpu")
        cache_dir = Path(kwargs.get("cache_dir", "."))
        vmfb_path = cache_dir / f"{model_path_obj.stem}_{target}.vmfb"

        _logger.debug(f"Target: {target}, cache path: {vmfb_path}")

        # Check for cached compilation
        if vmfb_path.exists() and not kwargs.get("force_compile", False):
            _logger.info(f"Loading cached VMFB: {vmfb_path}")
            return self._load_vmfb(vmfb_path, device)

        # Compile from ONNX
        _logger.info("Compiling ONNX to IREE VMFB...")
        if not IREE_COMPILER_AVAILABLE:
            # Try using CLI tools
            _logger.debug("Using CLI tools for compilation")
            vmfb_path = self._compile_with_cli(model_path_obj, target, vmfb_path, **kwargs)
        else:
            vmfb_path = self._compile_with_api(model_path_obj, target, vmfb_path, **kwargs)

        _logger.info(f"Compilation complete: {vmfb_path}")
        return self._load_vmfb(vmfb_path, device)

    def emit_mlir(
        self,
        model_path: str,
        output_path: str | Path | None = None,
        *,
        load_content: bool = False,
    ) -> MLIROutput:
        """Convert an ONNX model to IREE MLIR without compiling.

        This is useful for:
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

        Example:
            >>> backend = IREEBackend()
            >>> mlir = backend.emit_mlir("model.onnx", "model.mlir")
            >>> print(mlir.path)
            model.mlir
            >>> # Or load content for inspection
            >>> mlir = backend.emit_mlir("model.onnx", load_content=True)
            >>> print(mlir.content[:100])
            module @model {
              func.func @main_graph(...
        """
        model_path_obj = Path(model_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path_obj}")

        # Determine output path
        output_path_obj = (
            model_path_obj.with_suffix(".mlir") if output_path is None else Path(output_path)
        )

        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Find IREE import tool
        iree_import = _get_iree_import_onnx()
        if not iree_import:
            _logger.error("iree-import-onnx not found")
            raise RuntimeError(
                "iree-import-onnx not found. Install with: pip install iree-base-compiler\n"
                "Or ensure the tool is in your PATH."
            )

        # Convert ONNX to MLIR
        _logger.debug(f"Converting ONNX to MLIR: {model_path_obj} -> {output_path_obj}")
        try:
            subprocess.run(
                [iree_import, str(model_path_obj), "-o", str(output_path_obj)],
                check=True,
                capture_output=True,
                text=True,
            )
            _logger.debug("MLIR conversion successful")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            _logger.error(f"ONNX to MLIR conversion failed: {error_msg}")
            raise RuntimeError(f"ONNX to MLIR conversion failed: {error_msg}") from e

        # Load content if requested
        content = None
        if load_content:
            content = output_path_obj.read_text()

        return MLIROutput(
            path=output_path_obj,
            content=content,
            source_model=model_path_obj,
            dialect="iree",
        )

    def compile_mlir(
        self,
        mlir_path: str | Path,
        device: str = "cpu",
        output_path: str | Path | None = None,
        **kwargs,
    ) -> Path:
        """Compile an MLIR file to VMFB.

        This allows compiling custom or modified MLIR files.

        Args:
            mlir_path: Path to MLIR file
            device: Target device (cpu, vulkan, cuda)
            output_path: Where to save the VMFB. If None, saves alongside
                        the MLIR file with .vmfb extension.
            **kwargs: Additional options:
                - opt_level: Optimization level (0-3, default: 2)

        Returns:
            Path to compiled VMFB file

        Example:
            >>> backend = IREEBackend()
            >>> vmfb = backend.compile_mlir("model.mlir", device="vulkan")
            >>> model = backend.load_vmfb(vmfb, device="vulkan")
        """
        mlir_path = Path(mlir_path)

        if not mlir_path.exists():
            raise FileNotFoundError(f"MLIR file not found: {mlir_path}")

        device_type = device.split(":")[0] if ":" in device else device
        target = DEVICE_TO_TARGET.get(device_type, "llvm-cpu")

        # Determine output path
        if output_path is None:
            output_path = mlir_path.with_suffix(f".{target}.vmfb")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find IREE compile tool
        iree_compile = _get_iree_compile()
        if not iree_compile:
            raise RuntimeError(
                "iree-compile not found. Install with: pip install iree-base-compiler\n"
                "Or ensure the tool is in your PATH."
            )

        # Build compilation command
        opt_level = kwargs.get("opt_level", 2)
        cmd = [
            iree_compile,
            str(mlir_path),
            f"--iree-hal-target-backends={target}",
            f"--iree-opt-level=O{opt_level}",
            "-o",
            str(output_path),
        ]

        # Add target-specific flags
        if target == "llvm-cpu":
            cmd.append("--iree-llvmcpu-target-cpu=host")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"MLIR compilation failed: {error_msg}") from e

        return output_path

    def load_vmfb(
        self,
        vmfb_path: str | Path,
        device: str = "cpu",
    ) -> IREEModel:
        """Load a pre-compiled VMFB file.

        Args:
            vmfb_path: Path to VMFB file
            device: Target device (must match compilation target)

        Returns:
            Loaded IREE model ready for inference
        """
        if not IREE_RUNTIME_AVAILABLE:
            raise RuntimeError("iree-runtime not installed. Run: pip install iree-base-runtime")

        return self._load_vmfb(Path(vmfb_path), device)

    def _compile_with_cli(
        self,
        onnx_path: Path,
        target: str,
        vmfb_path: Path,
        **kwargs,
    ) -> Path:
        """Compile using IREE CLI tools."""
        # Find IREE tools
        iree_import = _get_iree_import_onnx()
        iree_compile = _get_iree_compile()

        if not iree_import:
            raise RuntimeError(
                "iree-import-onnx not found. Install with: pip install iree-base-compiler\n"
                "Or ensure the tool is in your PATH."
            )

        if not iree_compile:
            raise RuntimeError(
                "iree-compile not found. Install with: pip install iree-base-compiler\n"
                "Or ensure the tool is in your PATH."
            )

        # First convert ONNX to MLIR
        with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as mlir_file:
            mlir_path = Path(mlir_file.name)

        try:
            # ONNX -> MLIR
            subprocess.run(
                [iree_import, str(onnx_path), "-o", str(mlir_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            # MLIR -> VMFB
            opt_level = kwargs.get("opt_level", 2)
            cmd = [
                iree_compile,
                str(mlir_path),
                f"--iree-hal-target-backends={target}",
                f"--iree-opt-level=O{opt_level}",
                "-o",
                str(vmfb_path),
            ]

            # Add target-specific flags
            if target == "llvm-cpu":
                cmd.append("--iree-llvmcpu-target-cpu=host")

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return vmfb_path

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"IREE compilation failed: {error_msg}") from e

        finally:
            if mlir_path.exists():
                mlir_path.unlink()

    def _compile_with_api(
        self,
        onnx_path: Path,
        target: str,
        vmfb_path: Path,
        **kwargs,
    ) -> Path:
        """Compile using IREE Python API."""
        # This requires iree-compiler with ONNX import support
        # For now, fall back to CLI
        return self._compile_with_cli(onnx_path, target, vmfb_path, **kwargs)

    def _load_vmfb(self, vmfb_path: Path, device: str) -> IREEModel:
        """Load a compiled VMFB file."""
        # TODO: Extract actual input/output names from the module
        input_names = ["input"]
        output_names = ["output"]

        return IREEModel(
            vmfb_path=vmfb_path,
            device_name=device,
            input_names=input_names,
            output_names=output_names,
        )
