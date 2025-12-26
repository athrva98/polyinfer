"""IREE backend implementation with comprehensive Vulkan support."""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from polyinfer._logging import get_logger
from polyinfer.backends.base import Backend, CompiledModel

_logger = get_logger("backends.iree")


# =============================================================================
# Vulkan Target Configuration
# =============================================================================


class VulkanGPUVendor(str, Enum):
    """GPU vendor identifiers."""

    AMD = "amd"
    NVIDIA = "nvidia"
    INTEL = "intel"
    ARM = "arm"
    QUALCOMM = "qualcomm"
    UNKNOWN = "unknown"


@dataclass
class VulkanTarget:
    """Vulkan GPU target specification for IREE compilation.

    IREE supports multiple ways to specify Vulkan targets:
    - Architecture codes: rdna3, ampere, valhall4, adreno
    - LLVM targets: gfx1100, sm_86
    - Product names: rx7900xtx, rtx4090

    Reference: https://iree.dev/guides/deployment-configurations/gpu-vulkan/

    Attributes:
        target: The target specification string
        vendor: GPU vendor
        description: Human-readable description
    """

    target: str
    vendor: VulkanGPUVendor = VulkanGPUVendor.UNKNOWN
    description: str = ""

    def __str__(self) -> str:
        return self.target


# Pre-defined Vulkan targets for common GPUs
VULKAN_TARGETS = {
    # AMD RDNA3 (RX 7000 series)
    "rx7900xtx": VulkanTarget("rdna3", VulkanGPUVendor.AMD, "AMD RX 7900 XTX"),
    "rx7900xt": VulkanTarget("rdna3", VulkanGPUVendor.AMD, "AMD RX 7900 XT"),
    "rx7800xt": VulkanTarget("rdna3", VulkanGPUVendor.AMD, "AMD RX 7800 XT"),
    "rx7700xt": VulkanTarget("rdna3", VulkanGPUVendor.AMD, "AMD RX 7700 XT"),
    "rx7600": VulkanTarget("rdna3", VulkanGPUVendor.AMD, "AMD RX 7600"),
    "rdna3": VulkanTarget("rdna3", VulkanGPUVendor.AMD, "AMD RDNA3 architecture"),
    "gfx1100": VulkanTarget("gfx1100", VulkanGPUVendor.AMD, "AMD GFX1100 (RDNA3)"),
    "gfx1101": VulkanTarget("gfx1101", VulkanGPUVendor.AMD, "AMD GFX1101 (RDNA3)"),
    "gfx1102": VulkanTarget("gfx1102", VulkanGPUVendor.AMD, "AMD GFX1102 (RDNA3)"),
    # AMD RDNA2 (RX 6000 series)
    "rx6900xt": VulkanTarget("rdna2", VulkanGPUVendor.AMD, "AMD RX 6900 XT"),
    "rx6800xt": VulkanTarget("rdna2", VulkanGPUVendor.AMD, "AMD RX 6800 XT"),
    "rx6700xt": VulkanTarget("rdna2", VulkanGPUVendor.AMD, "AMD RX 6700 XT"),
    "rdna2": VulkanTarget("rdna2", VulkanGPUVendor.AMD, "AMD RDNA2 architecture"),
    "gfx1030": VulkanTarget("gfx1030", VulkanGPUVendor.AMD, "AMD GFX1030 (RDNA2)"),
    # NVIDIA Ada Lovelace (RTX 40 series)
    "rtx4090": VulkanTarget("sm_89", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 4090"),
    "rtx4080": VulkanTarget("sm_89", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 4080"),
    "rtx4070ti": VulkanTarget("sm_89", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 4070 Ti"),
    "rtx4070": VulkanTarget("sm_89", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 4070"),
    "rtx4060": VulkanTarget("sm_89", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 4060"),
    "ada": VulkanTarget("ada", VulkanGPUVendor.NVIDIA, "NVIDIA Ada Lovelace architecture"),
    "sm_89": VulkanTarget("sm_89", VulkanGPUVendor.NVIDIA, "NVIDIA SM 8.9 (Ada)"),
    # NVIDIA Ampere (RTX 30 series)
    "rtx3090": VulkanTarget("sm_86", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 3090"),
    "rtx3080ti": VulkanTarget("sm_86", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 3080 Ti"),
    "rtx3080": VulkanTarget("sm_86", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 3080"),
    "rtx3070": VulkanTarget("sm_86", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 3070"),
    "rtx3060": VulkanTarget("sm_86", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 3060"),
    "a100": VulkanTarget("sm_80", VulkanGPUVendor.NVIDIA, "NVIDIA A100"),
    "ampere": VulkanTarget("ampere", VulkanGPUVendor.NVIDIA, "NVIDIA Ampere architecture"),
    "sm_86": VulkanTarget("sm_86", VulkanGPUVendor.NVIDIA, "NVIDIA SM 8.6 (Ampere)"),
    "sm_80": VulkanTarget("sm_80", VulkanGPUVendor.NVIDIA, "NVIDIA SM 8.0 (Ampere)"),
    # NVIDIA Turing (RTX 20 series)
    "rtx2080ti": VulkanTarget("sm_75", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 2080 Ti"),
    "rtx2080": VulkanTarget("sm_75", VulkanGPUVendor.NVIDIA, "NVIDIA RTX 2080"),
    "turing": VulkanTarget("turing", VulkanGPUVendor.NVIDIA, "NVIDIA Turing architecture"),
    "sm_75": VulkanTarget("sm_75", VulkanGPUVendor.NVIDIA, "NVIDIA SM 7.5 (Turing)"),
    # Intel Arc
    "arc_a770": VulkanTarget("arc", VulkanGPUVendor.INTEL, "Intel Arc A770"),
    "arc_a750": VulkanTarget("arc", VulkanGPUVendor.INTEL, "Intel Arc A750"),
    "arc": VulkanTarget("arc", VulkanGPUVendor.INTEL, "Intel Arc architecture"),
    # ARM Mali
    "mali_g715": VulkanTarget("valhall4", VulkanGPUVendor.ARM, "ARM Mali G715"),
    "mali_g710": VulkanTarget("valhall4", VulkanGPUVendor.ARM, "ARM Mali G710"),
    "valhall4": VulkanTarget("valhall4", VulkanGPUVendor.ARM, "ARM Valhall Gen4"),
    "valhall": VulkanTarget("valhall", VulkanGPUVendor.ARM, "ARM Valhall"),
    # Qualcomm Adreno
    "adreno": VulkanTarget("adreno", VulkanGPUVendor.QUALCOMM, "Qualcomm Adreno"),
}


@dataclass
class IREECompileOptions:
    """Comprehensive IREE compilation options.

    Reference: https://iree.dev/reference/optimization-options/

    Attributes:
        opt_level: Optimization level (0-3). Higher = more optimization, longer compile.
        data_tiling: Enable data tiling optimization for better cache utilization.
        vulkan_target: Specific Vulkan GPU target (e.g., 'rdna3', 'ampere', 'sm_86').
        opset_version: ONNX opset version to upgrade model to before import.
        strip_debug: Strip debug info from compiled module (smaller size).
        enable_asserts: Include runtime assertions (useful for debugging).
        const_eval: Enable constant evaluation optimization.
        loop_unrolling: Enable loop unrolling.
        vectorize: Enable vectorization.
        extra_flags: Additional raw flags to pass to iree-compile.
    """

    opt_level: int = 2
    data_tiling: bool = True
    vulkan_target: str | None = None
    opset_version: int | None = None
    strip_debug: bool = False
    enable_asserts: bool = False
    const_eval: bool = True
    loop_unrolling: bool = True
    vectorize: bool = True
    extra_flags: list[str] = field(default_factory=list)

    def to_compile_flags(self, target: str) -> list[str]:
        """Convert options to iree-compile command line flags."""
        flags = [
            f"--iree-hal-target-backends={target}",
            f"--iree-opt-level=O{self.opt_level}",
        ]

        if self.data_tiling:
            flags.append("--iree-opt-data-tiling")

        if self.strip_debug:
            flags.append("--iree-llvmcpu-strip-executable-contents=true")

        if self.const_eval and self.opt_level >= 2:
            flags.append("--iree-opt-const-eval=true")

        # Target-specific flags
        if target == "llvm-cpu":
            flags.append("--iree-llvmcpu-target-cpu=host")
        elif target == "vulkan-spirv" and self.vulkan_target:
            # Resolve target if it's a known GPU name
            resolved = VULKAN_TARGETS.get(self.vulkan_target.lower())
            if resolved:
                flags.append(f"--iree-vulkan-target={resolved.target}")
            else:
                flags.append(f"--iree-vulkan-target={self.vulkan_target}")

        flags.extend(self.extra_flags)
        return flags

    def to_import_flags(self) -> list[str]:
        """Convert options to iree-import-onnx command line flags."""
        flags = []
        if self.opset_version:
            flags.append(f"--opset-version={self.opset_version}")
        return flags


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
        """Save MLIR to a new location."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.content:
            output_path.write_text(self.content)
        else:
            shutil.copy(self.path, output_path)

        return output_path


# =============================================================================
# IREE Availability Checks
# =============================================================================

# Check if IREE is available
try:
    import iree.runtime as iree_rt

    IREE_RUNTIME_AVAILABLE = True
    _logger.debug("IREE Runtime available")
except ImportError:
    IREE_RUNTIME_AVAILABLE = False
    iree_rt = None  # type: ignore[assignment]
    _logger.debug("IREE Runtime not installed")

try:
    import iree.compiler as iree_compiler

    IREE_COMPILER_AVAILABLE = True
    _logger.debug("IREE Compiler available")
except ImportError:
    IREE_COMPILER_AVAILABLE = False
    iree_compiler = None  # type: ignore[assignment]
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


def _get_iree_run_module() -> str | None:
    """Get path to iree-run-module tool."""
    return _find_iree_tool("iree-run-module")


# =============================================================================
# Device/Target Mappings
# =============================================================================

# Map device to IREE target backend
DEVICE_TO_TARGET = {
    "cpu": "llvm-cpu",
    "vulkan": "vulkan-spirv",
    "cuda": "cuda",
}

# Map device to IREE runtime driver
DEVICE_TO_DRIVER = {
    "cpu": "local-task",
    "vulkan": "vulkan",
    "cuda": "cuda",
}


# =============================================================================
# Error Handling
# =============================================================================


class IREECompilationError(RuntimeError):
    """IREE compilation failed with actionable error message."""

    def __init__(
        self,
        message: str,
        stderr: str = "",
        stage: str = "compilation",
        suggestions: list[str] | None = None,
    ):
        self.stderr = stderr
        self.stage = stage
        self.suggestions = suggestions or []

        # Build helpful error message
        full_msg = f"IREE {stage} failed: {message}"
        if stderr:
            full_msg += f"\n\nError output:\n{stderr[:2000]}"
        if self.suggestions:
            full_msg += "\n\nSuggestions:\n" + "\n".join(f"  - {s}" for s in self.suggestions)

        super().__init__(full_msg)


def _parse_compilation_error(stderr: str, stage: str = "compilation") -> IREECompilationError:
    """Parse IREE error output and provide actionable suggestions."""
    suggestions = []

    # Common error patterns and suggestions
    if "failed to legalize operation" in stderr:
        if "torch.operator" in stderr or "torch.aten" in stderr:
            suggestions.append("This ONNX operator is not yet supported by IREE")
            suggestions.append("Try upgrading opset version: opset_version=17")
            suggestions.append(
                "Check IREE ONNX Op Support: https://github.com/iree-org/iree/issues"
            )
        else:
            suggestions.append("An MLIR operation could not be lowered to the target")
            suggestions.append("This may be a bug in IREE - consider filing an issue")

    if "vulkan" in stderr.lower() and "driver" in stderr.lower():
        suggestions.append("Ensure Vulkan drivers are installed and up to date")
        suggestions.append("Try running: vulkaninfo to verify Vulkan support")
        suggestions.append("On Windows, update GPU drivers from manufacturer website")

    if "out of memory" in stderr.lower() or "OOM" in stderr:
        suggestions.append("Model too large for GPU memory")
        suggestions.append("Try reducing batch size or model size")
        suggestions.append("Consider using CPU target instead")

    if "opset" in stderr.lower():
        suggestions.append("ONNX opset version may be incompatible")
        suggestions.append("Try: opset_version=17 when loading")
        suggestions.append("Or upgrade your ONNX file using onnx.version_converter")

    if not suggestions:
        suggestions.append("Check IREE documentation: https://iree.dev/")
        suggestions.append("Try with --enable_asserts=True for more debug info")
        suggestions.append("Consider filing an issue: https://github.com/iree-org/iree/issues")

    return IREECompilationError(
        message="Compilation failed",
        stderr=stderr,
        stage=stage,
        suggestions=suggestions,
    )


# =============================================================================
# IREE Model Wrapper
# =============================================================================


class IREEModel(CompiledModel):
    """IREE compiled module wrapper."""

    # Common function names IREE generates from ONNX models
    FUNC_NAMES = ["main_graph", "main", "forward", "run", "inference", "predict"]

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
        try:
            self._module = iree_rt.load_vm_flatbuffer_file(str(vmfb_path), driver=driver)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load VMFB file '{vmfb_path}' with driver '{driver}': {e}\n"
                f"Ensure the VMFB was compiled for the correct target."
            ) from e

        # Find the main inference function
        self._func = None
        self._func_name = ""
        for func_name in self.FUNC_NAMES:
            try:
                self._func = self._module[func_name]
                self._func_name = func_name
                _logger.debug(f"Found inference function: {func_name}")
                break
            except KeyError:
                continue

        if self._func is None:
            # List available functions for debugging
            available = list(self._module.keys()) if hasattr(self._module, "keys") else []
            raise RuntimeError(
                f"Could not find inference function in IREE module.\n"
                f"Tried: {self.FUNC_NAMES}\n"
                f"Available functions: {available}"
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

        try:
            outputs = self._func(*inputs)
        except Exception as e:
            raise RuntimeError(
                f"IREE inference failed: {e}\n"
                f"Input shapes: {[inp.shape for inp in inputs]}\n"
                f"Function: {self._func_name}"
            ) from e

        # Convert outputs to numpy
        if isinstance(outputs, (list, tuple)):
            results = [np.asarray(o) for o in outputs]
            if len(results) == 1:
                return results[0]
            return tuple(results)
        return np.asarray(outputs)


# =============================================================================
# IREE Backend
# =============================================================================


class IREEBackend(Backend):
    """IREE backend supporting CPU, Vulkan, and CUDA.

    IREE (Intermediate Representation Execution Environment) is an MLIR-based
    end-to-end compiler and runtime for ML models.

    Key features:
    - Cross-platform: Linux, Windows, macOS, Android
    - Cross-vendor GPU support via Vulkan: AMD, NVIDIA, Intel, ARM, Qualcomm
    - Ahead-of-time compilation to VMFB (Virtual Machine FlatBuffer)
    - SPIR-V code generation for Vulkan targets

    Reference: https://iree.dev/
    """

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

    def list_vulkan_targets(self) -> dict[str, VulkanTarget]:
        """Get all available Vulkan target presets.

        Returns:
            Dictionary mapping target name to VulkanTarget configuration.

        Example:
            >>> backend = IREEBackend()
            >>> targets = backend.list_vulkan_targets()
            >>> print(targets['rtx4090'])
            VulkanTarget(target='sm_89', vendor='nvidia', ...)
        """
        return VULKAN_TARGETS.copy()

    def detect_vulkan_devices(self) -> list[dict[str, Any]]:
        """Detect available Vulkan devices using iree-run-module.

        Returns:
            List of device info dictionaries with name, vendor, etc.

        Note:
            Requires iree-run-module to be installed.
        """
        iree_run = _get_iree_run_module()
        if not iree_run:
            _logger.warning("iree-run-module not found, cannot detect Vulkan devices")
            return []

        try:
            result = subprocess.run(
                [iree_run, "--dump_devices"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse output (format varies by IREE version)
            devices = []
            for line in result.stdout.split("\n"):
                if "vulkan" in line.lower():
                    devices.append({"name": line.strip(), "driver": "vulkan"})
            return devices
        except Exception as e:
            _logger.warning(f"Failed to detect Vulkan devices: {e}")
            return []

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
                - opt_level: Optimization level 0-3 (default: 2)
                - vulkan_target: GPU target (e.g., 'rdna3', 'ampere', 'rtx4090')
                - opset_version: Upgrade ONNX to this opset before import
                - data_tiling: Enable data tiling optimization (default: True)
                - cache_dir: Directory for compiled artifacts
                - force_compile: Force recompilation even if cached
                - save_mlir: Save intermediate MLIR file (default: False)

        Returns:
            Compiled IREE model ready for inference

        Example:
            >>> model = backend.load("yolov8n.onnx", device="vulkan",
            ...                       vulkan_target="rtx4090", opt_level=3)
        """
        if not IREE_RUNTIME_AVAILABLE:
            raise RuntimeError(
                "IREE Runtime not installed.\nInstall with: pip install iree-base-runtime"
            )

        _logger.debug(f"Loading model: {model_path}")

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path_obj}")

        device_type = device.split(":")[0] if ":" in device else device

        # Build compile options
        options = IREECompileOptions(
            opt_level=kwargs.get("opt_level", 2),
            data_tiling=kwargs.get("data_tiling", True),
            vulkan_target=kwargs.get("vulkan_target"),
            opset_version=kwargs.get("opset_version"),
            extra_flags=kwargs.get("extra_flags", []),
        )

        # Determine paths
        target = DEVICE_TO_TARGET.get(device_type, "llvm-cpu")
        cache_dir = Path(kwargs.get("cache_dir", model_path_obj.parent))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Include vulkan target in cache filename if specified
        target_suffix = f"_{options.vulkan_target}" if options.vulkan_target else ""
        vmfb_path = cache_dir / f"{model_path_obj.stem}_{target}{target_suffix}.vmfb"

        _logger.debug(f"Target: {target}, cache path: {vmfb_path}")

        # Check for cached compilation
        if vmfb_path.exists() and not kwargs.get("force_compile", False):
            _logger.info(f"Loading cached VMFB: {vmfb_path}")
            return self._load_vmfb(vmfb_path, device)

        # Compile from ONNX
        _logger.info(f"Compiling ONNX to IREE VMFB (target={target})...")
        vmfb_path = self._compile_onnx(
            model_path_obj,
            target,
            vmfb_path,
            options,
            save_mlir=kwargs.get("save_mlir", False),
        )

        _logger.info(f"Compilation complete: {vmfb_path}")
        return self._load_vmfb(vmfb_path, device)

    def emit_mlir(
        self,
        model_path: str,
        output_path: str | Path | None = None,
        *,
        opset_version: int | None = None,
        load_content: bool = False,
    ) -> MLIROutput:
        """Convert an ONNX model to IREE MLIR without compiling.

        This is useful for:
        - Inspecting the intermediate representation
        - Custom kernel injection
        - MLIR pass analysis and transformation
        - Debugging compilation issues

        Args:
            model_path: Path to ONNX file
            output_path: Where to save MLIR. If None, uses .mlir extension.
            opset_version: Upgrade ONNX to this opset before import
            load_content: If True, also load MLIR content into memory

        Returns:
            MLIROutput containing path and optionally content
        """
        model_path_obj = Path(model_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path_obj}")

        output_path_obj = (
            model_path_obj.with_suffix(".mlir") if output_path is None else Path(output_path)
        )
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        iree_import = _get_iree_import_onnx()
        if not iree_import:
            raise RuntimeError(
                "iree-import-onnx not found.\nInstall with: pip install iree-base-compiler[onnx]"
            )

        # Build command
        cmd = [iree_import, str(model_path_obj), "-o", str(output_path_obj)]
        if opset_version:
            cmd.append(f"--opset-version={opset_version}")

        _logger.debug(f"Converting ONNX to MLIR: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise _parse_compilation_error(e.stderr or str(e), stage="ONNX import") from e

        content = output_path_obj.read_text() if load_content else None

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
            output_path: Where to save VMFB. If None, uses .vmfb extension.
            **kwargs: Compilation options (see IREECompileOptions)

        Returns:
            Path to compiled VMFB file
        """
        mlir_path = Path(mlir_path)

        if not mlir_path.exists():
            raise FileNotFoundError(f"MLIR file not found: {mlir_path}")

        device_type = device.split(":")[0] if ":" in device else device
        target = DEVICE_TO_TARGET.get(device_type, "llvm-cpu")

        if output_path is None:
            output_path = mlir_path.with_suffix(f".{target}.vmfb")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build options
        options = IREECompileOptions(
            opt_level=kwargs.get("opt_level", 2),
            data_tiling=kwargs.get("data_tiling", True),
            vulkan_target=kwargs.get("vulkan_target"),
            extra_flags=kwargs.get("extra_flags", []),
        )

        iree_compile = _get_iree_compile()
        if not iree_compile:
            raise RuntimeError(
                "iree-compile not found.\nInstall with: pip install iree-base-compiler"
            )

        # Build command
        cmd = [iree_compile, str(mlir_path)]
        cmd.extend(options.to_compile_flags(target))
        cmd.extend(["-o", str(output_path)])

        _logger.debug(f"Compiling MLIR: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise _parse_compilation_error(e.stderr or str(e), stage="MLIR compilation") from e

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
            raise RuntimeError(
                "IREE Runtime not installed.\nInstall with: pip install iree-base-runtime"
            )

        return self._load_vmfb(Path(vmfb_path), device)

    def _compile_onnx(
        self,
        onnx_path: Path,
        target: str,
        vmfb_path: Path,
        options: IREECompileOptions,
        save_mlir: bool = False,
    ) -> Path:
        """Compile ONNX to VMFB via MLIR."""
        iree_import = _get_iree_import_onnx()
        iree_compile = _get_iree_compile()

        if not iree_import:
            raise RuntimeError(
                "iree-import-onnx not found.\nInstall with: pip install iree-base-compiler[onnx]"
            )

        if not iree_compile:
            raise RuntimeError(
                "iree-compile not found.\nInstall with: pip install iree-base-compiler"
            )

        # Determine MLIR path
        if save_mlir:
            mlir_path = vmfb_path.with_suffix(".mlir")
            delete_mlir = False
        else:
            with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as mlir_file:
                mlir_path = Path(mlir_file.name)
            delete_mlir = True

        try:
            # Step 1: ONNX -> MLIR
            _logger.debug("Step 1/2: Converting ONNX to MLIR...")
            import_cmd = [iree_import, str(onnx_path), "-o", str(mlir_path)]
            import_cmd.extend(options.to_import_flags())

            _logger.debug(f"Import command: {' '.join(import_cmd)}")

            try:
                result = subprocess.run(
                    import_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if result.stderr:
                    _logger.debug(f"Import warnings: {result.stderr}")
            except subprocess.CalledProcessError as e:
                raise _parse_compilation_error(e.stderr or str(e), stage="ONNX import") from e

            # Step 2: MLIR -> VMFB
            _logger.debug("Step 2/2: Compiling MLIR to VMFB...")
            compile_cmd = [iree_compile, str(mlir_path)]
            compile_cmd.extend(options.to_compile_flags(target))
            compile_cmd.extend(["-o", str(vmfb_path)])

            _logger.debug(f"Compile command: {' '.join(compile_cmd)}")

            try:
                result = subprocess.run(
                    compile_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if result.stderr:
                    _logger.debug(f"Compile warnings: {result.stderr}")
            except subprocess.CalledProcessError as e:
                raise _parse_compilation_error(e.stderr or str(e), stage="MLIR compilation") from e

            return vmfb_path

        finally:
            if delete_mlir and mlir_path.exists():
                mlir_path.unlink()

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
