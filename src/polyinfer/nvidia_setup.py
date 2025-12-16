"""Automatic NVIDIA library setup for PolyInfer.

This module automatically finds and loads NVIDIA libraries (CUDA, cuDNN, TensorRT)
installed via pip packages, eliminating the need for manual PATH configuration.

The setup happens automatically when polyinfer is imported.
"""

import os
import sys
import warnings
from pathlib import Path


def _get_site_packages() -> Path:
    """Get the site-packages directory."""
    for path in sys.path:
        if "site-packages" in path:
            return Path(path)
    return Path(sys.prefix) / "Lib" / "site-packages"


def _find_nvidia_dll_dirs() -> list[Path]:
    """Find all directories containing NVIDIA DLLs."""
    site_packages = _get_site_packages()
    dll_dirs = []

    # Known NVIDIA package locations
    nvidia_packages = [
        "nvidia/cublas/bin",
        "nvidia/cuda_runtime/bin",
        "nvidia/cudnn/bin",
        "nvidia/cufft/bin",
        "nvidia/curand/bin",
        "nvidia/cusolver/bin",
        "nvidia/cusparse/bin",
        "nvidia/nccl/bin",
        "nvidia/nvjitlink/bin",
        "nvidia/nvrtc/bin",
        "tensorrt_libs",
        "tensorrt_bindings",
    ]

    for pkg in nvidia_packages:
        pkg_path = site_packages / pkg
        if pkg_path.exists():
            dll_dirs.append(pkg_path)

    # Also search for any nvidia subdirectory with DLLs
    nvidia_root = site_packages / "nvidia"
    if nvidia_root.exists():
        for subdir in nvidia_root.rglob("bin"):
            if subdir.is_dir() and subdir not in dll_dirs:
                # Check if it contains DLLs
                if any(subdir.glob("*.dll")):
                    dll_dirs.append(subdir)

    # TensorRT root
    tensorrt_root = site_packages / "tensorrt_libs"
    if tensorrt_root.exists() and tensorrt_root not in dll_dirs:
        dll_dirs.append(tensorrt_root)

    return dll_dirs


def _setup_dll_directories():
    """Add NVIDIA DLL directories to the DLL search path (Windows only)."""
    if sys.platform != "win32":
        return

    dll_dirs = _find_nvidia_dll_dirs()

    for dll_dir in dll_dirs:
        try:
            os.add_dll_directory(str(dll_dir))
        except (OSError, AttributeError):
            # os.add_dll_directory may not exist on older Python
            pass

    # Also add to PATH for subprocess calls
    if dll_dirs:
        path_additions = os.pathsep.join(str(d) for d in dll_dirs)
        current_path = os.environ.get("PATH", "")
        if path_additions not in current_path:
            os.environ["PATH"] = path_additions + os.pathsep + current_path


def _find_nvidia_lib_dirs() -> list[Path]:
    """Find all directories containing NVIDIA .so libraries."""
    site_packages = _get_site_packages()
    lib_dirs = []

    nvidia_root = site_packages / "nvidia"
    if nvidia_root.exists():
        for subdir in nvidia_root.rglob("lib"):
            if subdir.is_dir() and any(subdir.glob("*.so*")):
                lib_dirs.append(subdir)

    tensorrt_root = site_packages / "tensorrt_libs"
    if tensorrt_root.exists():
        lib_dirs.append(tensorrt_root)

    return lib_dirs


def _setup_ld_library_path():
    """Add NVIDIA library directories to LD_LIBRARY_PATH (Linux only).

    Note: We intentionally do NOT preload libraries with ctypes on Linux.
    Preloading can cause conflicts with PyTorch's CUDA libraries, especially
    when torch is pre-installed (e.g., on Google Colab). PyTorch and ONNX Runtime
    can find their own libraries without our help.
    """
    if sys.platform == "win32":
        return

    lib_dirs = _find_nvidia_lib_dirs()

    if not lib_dirs:
        return

    # Set LD_LIBRARY_PATH for any subprocesses (e.g., iree-compile)
    # This is safe and doesn't affect the current process's library loading
    lib_additions = os.pathsep.join(str(d) for d in lib_dirs)
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_additions not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_additions + os.pathsep + current_ld_path

    # NOTE: We do NOT preload libraries with ctypes anymore.
    # This was causing conflicts with PyTorch's bundled CUDA/NCCL libraries,
    # resulting in "undefined symbol: ncclCommWindowRegister" errors on Colab.


def setup_nvidia_libraries():
    """Setup NVIDIA libraries for use with PolyInfer.

    This function:
    1. Finds NVIDIA packages installed via pip (nvidia-cudnn-cu12, tensorrt-cu12-libs, etc.)
    2. Adds their DLL/library directories to the search path
    3. Makes CUDA, cuDNN, and TensorRT available to ONNX Runtime and other backends

    Called automatically when polyinfer is imported.
    """
    _setup_dll_directories()
    _setup_ld_library_path()


def _check_onnxruntime_conflicts():
    """Check for conflicting ONNX Runtime installations and warn/fix.

    On Windows, onnxruntime-gpu, onnxruntime-directml, and onnxruntime
    cannot coexist properly. This function detects conflicts and provides
    guidance or automatic fixes.
    """
    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata

    # Check which onnxruntime variants are installed
    installed = []
    for pkg in ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"]:
        try:
            metadata.version(pkg)
            installed.append(pkg)
        except metadata.PackageNotFoundError:
            pass

    # If multiple variants installed, check which one is actually active
    if len(installed) > 1:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()

            has_cuda = "CUDAExecutionProvider" in providers
            has_dml = "DmlExecutionProvider" in providers

            # Detect the conflict scenario
            if "onnxruntime-gpu" in installed and "onnxruntime-directml" in installed:
                if has_dml and not has_cuda:
                    # DirectML overwrote CUDA - this is the common problem
                    warnings.warn(
                        "\n\n"
                        "⚠️  ONNX Runtime Conflict Detected!\n"
                        "   Both 'onnxruntime-gpu' and 'onnxruntime-directml' are installed,\n"
                        "   but only DirectML is active. CUDA support is disabled.\n\n"
                        "   To fix, run:\n"
                        "     pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y\n"
                        "     pip install onnxruntime-gpu    # For CUDA\n"
                        "   OR:\n"
                        "     pip install onnxruntime-directml  # For DirectML\n\n"
                        "   You can only have ONE onnxruntime variant installed at a time.\n",
                        UserWarning,
                        stacklevel=3,
                    )
                elif has_cuda and not has_dml:
                    # CUDA overwrote DirectML - less common but possible
                    warnings.warn(
                        "\n\n"
                        "⚠️  ONNX Runtime Conflict Detected!\n"
                        "   Both 'onnxruntime-gpu' and 'onnxruntime-directml' are installed,\n"
                        "   but only CUDA is active. DirectML support is disabled.\n\n"
                        "   To fix, uninstall conflicting packages:\n"
                        "     pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y\n"
                        "     pip install onnxruntime-gpu\n",
                        UserWarning,
                        stacklevel=3,
                    )
        except ImportError:
            pass  # onnxruntime not importable, skip check


def get_nvidia_info() -> dict:
    """Get information about installed NVIDIA libraries.

    Returns:
        Dictionary with information about found NVIDIA packages and libraries.
    """
    site_packages = _get_site_packages()
    info = {
        "site_packages": str(site_packages),
        "library_directories": [],
        "libraries": {},
    }

    if sys.platform == "win32":
        lib_dirs = _find_nvidia_dll_dirs()
    else:
        lib_dirs = _find_nvidia_lib_dirs()

    info["library_directories"] = [str(d) for d in lib_dirs]

    # Find specific libraries
    if sys.platform == "win32":
        library_patterns = {
            "cublas": "cublas64_*.dll",
            "cudnn": "cudnn64_*.dll",
            "nvinfer": "nvinfer_*.dll",
            "cuda_runtime": "cudart64_*.dll",
        }
    else:
        library_patterns = {
            "cublas": "libcublas.so*",
            "cudnn": "libcudnn.so*",
            "nvinfer": "libnvinfer.so*",
            "cuda_runtime": "libcudart.so*",
        }

    for lib_name, pattern in library_patterns.items():
        for lib_dir in lib_dirs:
            matches = list(lib_dir.glob(pattern))
            if matches:
                info["libraries"][lib_name] = str(matches[0])
                break

    return info


def fix_onnxruntime_conflict(prefer: str = "cuda") -> bool:
    """Fix ONNX Runtime package conflicts by uninstalling conflicting packages.

    Args:
        prefer: Which variant to keep - "cuda" for onnxruntime-gpu,
                "directml" for onnxruntime-directml

    Returns:
        True if fix was applied, False if no fix needed

    Example:
        >>> import polyinfer as pi
        >>> pi.fix_onnxruntime_conflict(prefer="cuda")
        True
        >>> # Now restart Python and re-import
    """
    import subprocess

    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata

    # Check which variants are installed
    installed = []
    for pkg in ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"]:
        try:
            metadata.version(pkg)
            installed.append(pkg)
        except metadata.PackageNotFoundError:
            pass

    if len(installed) <= 1:
        print("No conflict detected - only one onnxruntime variant installed.")
        return False

    print(f"Found conflicting packages: {installed}")
    print(f"Preference: {prefer}")

    # Uninstall all variants
    print("\nUninstalling all onnxruntime variants...")
    for pkg in ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"]:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
            capture_output=True,
        )

    # Install the preferred variant
    if prefer == "cuda":
        pkg_to_install = "onnxruntime-gpu"
    elif prefer == "directml":
        pkg_to_install = "onnxruntime-directml"
    else:
        pkg_to_install = "onnxruntime"

    print(f"Installing {pkg_to_install}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg_to_install],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"\n✓ Successfully installed {pkg_to_install}")
        print("  Please restart Python to use the new package.")
        return True
    else:
        print(f"\n✗ Failed to install {pkg_to_install}")
        print(f"  Error: {result.stderr}")
        return False


# Auto-setup on import
setup_nvidia_libraries()

# Check for ONNX Runtime conflicts
_check_onnxruntime_conflicts()
