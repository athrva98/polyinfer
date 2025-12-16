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
    # Try to find an actual existing site-packages directory
    for path in sys.path:
        if "site-packages" in path:
            p = Path(path)
            if p.exists():
                return p

    # Fallback: try common locations
    candidates = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.prefix) / "Lib" / "site-packages",  # Windows
        Path("/usr/local/lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Last resort
    return Path(sys.prefix) / "lib" / "site-packages"


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
    """Setup for Linux - currently disabled to avoid PyTorch conflicts.

    On Linux, PyTorch and ONNX Runtime bundle their own CUDA libraries and
    handle library loading themselves. Modifying LD_LIBRARY_PATH can cause
    conflicts with PyTorch's bundled NCCL, resulting in errors like:
    "undefined symbol: ncclCommWindowRegister"

    This function is intentionally a no-op on Linux.
    """
    # Completely disabled on Linux to avoid PyTorch conflicts
    # PyTorch and ONNX Runtime can find their own libraries
    pass


# Track if TensorRT paths have been set up
_tensorrt_paths_configured = False
_preloaded_tensorrt_libs: list[str] = []


def _find_tensorrt_lib_dirs() -> list[Path]:
    """Find directories containing TensorRT libraries.

    Returns:
        List of paths that contain libnvinfer.so (Linux) or nvinfer*.dll (Windows)
    """
    tensorrt_dirs = []
    site_packages = _get_site_packages()

    # 1. Check pip-installed TensorRT packages
    tensorrt_libs = site_packages / "tensorrt_libs"
    if tensorrt_libs.exists():
        tensorrt_dirs.append(tensorrt_libs)

    # TensorRT bindings
    tensorrt_bindings = site_packages / "tensorrt_bindings"
    if tensorrt_bindings.exists():
        tensorrt_dirs.append(tensorrt_bindings)

    # Also check for tensorrt_cu12_libs (older package naming)
    for pattern in ["tensorrt*libs*", "tensorrt*"]:
        for path in site_packages.glob(pattern):
            if path.is_dir() and path not in tensorrt_dirs:
                if sys.platform == "win32":
                    if any(path.glob("nvinfer*.dll")):
                        tensorrt_dirs.append(path)
                else:
                    if any(path.glob("*.so*")):
                        tensorrt_dirs.append(path)

    # 2. Check system-level TensorRT installations (Linux only)
    if sys.platform != "win32":
        system_tensorrt_paths = [
            "/usr/lib/x86_64-linux-gnu",  # Debian/Ubuntu system libs
            "/usr/local/lib",
            "/usr/lib",
            "/opt/tensorrt/lib",  # Common TensorRT install location
            "/usr/lib64-nvidia",  # Colab uses this
        ]

        for sys_path in system_tensorrt_paths:
            p = Path(sys_path)
            if p.exists() and p not in tensorrt_dirs:
                if any(p.glob("libnvinfer.so*")):
                    tensorrt_dirs.append(p)

    return tensorrt_dirs


def setup_tensorrt_paths() -> bool:
    """Setup TensorRT libraries for ONNX Runtime TensorRT EP.

    This function should be called BEFORE using the TensorRT execution provider.
    On Linux, it preloads TensorRT libraries using ctypes.CDLL with RTLD_GLOBAL,
    which makes the symbols available to ONNX Runtime when it loads the TensorRT EP.

    On Windows, it adds directories to the DLL search path and PATH.

    This is safe to call after torch because:
    1. We only load TensorRT-specific libraries, not CUDA/NCCL
    2. PyTorch has already loaded its CUDA libraries
    3. TensorRT doesn't conflict with PyTorch's NCCL

    Returns:
        True if libraries were loaded/configured, False if already done or not needed
    """
    global _tensorrt_paths_configured

    if _tensorrt_paths_configured:
        return False

    tensorrt_dirs = _find_tensorrt_lib_dirs()

    if not tensorrt_dirs:
        _tensorrt_paths_configured = True
        return False

    if sys.platform == "win32":
        # Windows: Add to DLL search path
        for tensorrt_dir in tensorrt_dirs:
            try:
                os.add_dll_directory(str(tensorrt_dir))
            except (OSError, AttributeError):
                pass
            # Also add to PATH
            current_path = os.environ.get("PATH", "")
            tensorrt_path = str(tensorrt_dir)
            if tensorrt_path not in current_path:
                os.environ["PATH"] = tensorrt_path + os.pathsep + current_path
    else:
        # Linux: Preload TensorRT libraries using ctypes
        # Setting LD_LIBRARY_PATH at runtime doesn't work because the dynamic
        # linker only reads it at process startup. Instead, we use ctypes.CDLL
        # with RTLD_GLOBAL to load the libraries and make their symbols available.
        import ctypes

        # Libraries to preload in dependency order
        # ONNX Runtime TensorRT EP requires: libnvinfer, libnvonnxparser, libnvinfer_plugin
        # We try multiple version suffixes (.so.10, .so.8, .so) for compatibility
        #
        # Order matters! Dependencies must be loaded before dependents:
        # 1. libnvinfer (core) - depends on CUDA libs (already loaded by PyTorch/onnxruntime)
        # 2. libnvinfer_plugin - depends on libnvinfer
        # 3. libnvonnxparser - depends on libnvinfer
        # 4. libnvinfer_builder_resource - TensorRT builder resources (optional but helps)
        tensorrt_libs_to_load = [
            # Core TensorRT inference library (must be first)
            "libnvinfer.so.10",
            "libnvinfer.so.8",
            "libnvinfer.so",
            # TensorRT plugins - depends on libnvinfer, needed for custom ops
            "libnvinfer_plugin.so.10",
            "libnvinfer_plugin.so.8",
            "libnvinfer_plugin.so",
            # ONNX parser - required for ONNX Runtime to parse models
            "libnvonnxparser.so.10",
            "libnvonnxparser.so.8",
            "libnvonnxparser.so",
            # TensorRT builder resource library (helps with engine building)
            "libnvinfer_builder_resource.so.10",
            "libnvinfer_builder_resource.so.8",
            "libnvinfer_builder_resource.so.10.7.0",
            "libnvinfer_builder_resource.so",
            # TensorRT lean runtime (optional, for some configurations)
            "libnvinfer_lean.so.10",
            "libnvinfer_lean.so.8",
            "libnvinfer_lean.so",
            # TensorRT dispatch (optional, for multi-GPU)
            "libnvinfer_dispatch.so.10",
            "libnvinfer_dispatch.so.8",
            "libnvinfer_dispatch.so",
        ]

        loaded_libs = []
        for tensorrt_dir in tensorrt_dirs:
            for lib_name in tensorrt_libs_to_load:
                lib_path = tensorrt_dir / lib_name
                if lib_path.exists():
                    try:
                        # RTLD_GLOBAL makes symbols available to subsequently loaded libraries
                        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                        loaded_libs.append(str(lib_path))
                    except OSError as e:
                        # Library might have unmet dependencies, continue
                        # Only warn if it's a core library that failed
                        if "libnvinfer.so" in lib_name or "libnvinfer_plugin.so" in lib_name:
                            warnings.warn(
                                f"Failed to preload TensorRT library {lib_path}: {e}",
                                UserWarning,
                                stacklevel=3,
                            )

        # Store loaded libs for debugging
        global _preloaded_tensorrt_libs
        _preloaded_tensorrt_libs = loaded_libs

        # Also update LD_LIBRARY_PATH for any subprocess calls
        if tensorrt_dirs:
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            new_paths = [str(p) for p in tensorrt_dirs if str(p) not in current_ld_path]
            if new_paths:
                if current_ld_path:
                    os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + ":" + current_ld_path
                else:
                    os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths)

    _tensorrt_paths_configured = True
    return True


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

    IMPORTANT: This function avoids importing onnxruntime on Linux when torch
    is already loaded, as importing onnxruntime-gpu can load CUDA libraries
    that conflict with PyTorch's bundled NCCL.
    """
    # On Linux, skip this check entirely if PyTorch is already loaded.
    # Importing onnxruntime-gpu can load CUDA libraries that conflict with
    # PyTorch's bundled NCCL, causing "undefined symbol: ncclCommWindowRegister"
    if sys.platform.startswith("linux") and "torch" in sys.modules:
        return

    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata

    # Check which onnxruntime variants are installed (metadata only, no import)
    installed = []
    for pkg in ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"]:
        try:
            metadata.version(pkg)
            installed.append(pkg)
        except metadata.PackageNotFoundError:
            pass

    # Only check provider availability on Windows where the conflict is less severe
    # On Linux with onnxruntime-gpu, importing it can pollute CUDA environment
    if len(installed) > 1 and sys.platform == "win32":
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
    elif len(installed) > 1:
        # On Linux, just warn based on metadata without importing
        if "onnxruntime-gpu" in installed and "onnxruntime-directml" in installed:
            warnings.warn(
                "\n\n"
                "⚠️  Multiple ONNX Runtime variants detected!\n"
                "   Found: " + ", ".join(installed) + "\n"
                "   This may cause conflicts. Consider keeping only one:\n"
                "     pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y\n"
                "     pip install onnxruntime-gpu    # For CUDA\n",
                UserWarning,
                stacklevel=3,
            )


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
        "tensorrt_setup": {
            "configured": _tensorrt_paths_configured,
            "preloaded_libs": _preloaded_tensorrt_libs,
            "tensorrt_dirs": [str(d) for d in _find_tensorrt_lib_dirs()],
        },
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

# Preload TensorRT libraries BEFORE onnxruntime is imported
# ONNX Runtime checks for TensorRT EP at import time, so libs must be loaded first
setup_tensorrt_paths()

# Check for ONNX Runtime conflicts
_check_onnxruntime_conflicts()
