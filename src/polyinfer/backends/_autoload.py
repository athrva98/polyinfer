"""Auto-load and register available backends."""

import sys
from polyinfer.backends.registry import register_backend


def _verify_tensorrt_libs_available() -> bool:
    """Check if TensorRT libraries are actually available.

    This is a lightweight check that verifies libnvinfer can be loaded,
    without importing onnxruntime. Used by the lazy backend to avoid
    advertising tensorrt device when it won't actually work.

    Returns:
        True if TensorRT libraries are loadable, False otherwise.
    """
    import ctypes

    if sys.platform == "win32":
        for lib_name in ["nvinfer_10.dll", "nvinfer.dll"]:
            try:
                ctypes.CDLL(lib_name)
                return True
            except OSError:
                pass
        return False
    else:
        # On Linux, try to load libnvinfer
        for lib_name in ["libnvinfer.so.10", "libnvinfer.so.8", "libnvinfer.so"]:
            try:
                ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                pass
        return False


def _should_use_lazy_onnxruntime() -> bool:
    """Check if ONNX Runtime should use lazy loading to avoid CUDA conflicts.

    On Linux, importing onnxruntime-gpu loads CUDA libraries that conflict
    with PyTorch's bundled libraries (NCCL). This causes errors like:
    "undefined symbol: ncclCommWindowRegister"

    The conflict happens in BOTH directions:
    1. Import torch first, then onnxruntime-gpu -> torch breaks
    2. Import onnxruntime-gpu first, then torch -> torch breaks

    Therefore, on Linux we ALWAYS use lazy loading for ONNX Runtime to defer
    the actual import until the user explicitly loads a model. This allows:
    - Users to import polyinfer and torch in any order
    - ONNX Runtime import only happens when actually needed
    - If user only uses CPU, no CUDA conflict occurs

    We use lazy loading if:
    1. We're on Linux (where the CUDA library conflicts occur)
    2. onnxruntime-gpu is installed (not plain onnxruntime)

    On Windows, the DLL loading mechanism is different and doesn't cause
    these conflicts, so we can import eagerly.
    """
    if not sys.platform.startswith("linux"):
        return False

    # Check if onnxruntime-gpu is installed (vs plain onnxruntime)
    try:
        import importlib.metadata as metadata
        metadata.version("onnxruntime-gpu")
        return True  # onnxruntime-gpu installed, use lazy loading
    except Exception:
        pass

    return False  # Plain onnxruntime or not installed, safe to import eagerly


def _should_skip_native_tensorrt() -> bool:
    """Check if native TensorRT import should be skipped to avoid conflicts.

    The native TensorRT backend imports cuda.bindings/cuda.cudart which can
    load CUDA libraries that conflict with PyTorch's bundled libraries.
    This causes 'undefined symbol: ncclCommWindowRegister' errors.

    We skip native TensorRT if:
    1. PyTorch is already loaded (torch in sys.modules)
    2. We're on Linux (where the conflicts are most severe)

    Users who want native TensorRT can still use it by:
    - Importing polyinfer before torch
    - Using backend="tensorrt" explicitly after ensuring no conflicts
    - Using ONNX Runtime's TensorRT EP instead (recommended, no conflicts)
    """
    # Skip if PyTorch is already imported to avoid loading conflicting CUDA libs
    if "torch" in sys.modules:
        return True

    # On Linux, the cuda.bindings import can pollute the process with
    # incompatible CUDA libraries. Skip by default.
    if sys.platform.startswith("linux"):
        return True

    return False


def register_all():
    """Register all available backends.

    This function attempts to import and register each backend.
    Backends that fail to import (missing dependencies) are silently skipped.

    IMPORTANT: On Linux with onnxruntime-gpu, we use lazy loading to avoid
    importing onnxruntime at module load time. This prevents CUDA library
    conflicts with PyTorch's bundled NCCL, regardless of import order.
    """
    use_lazy_ort = _should_use_lazy_onnxruntime()

    # ONNX Runtime backend (Tier 1)
    # On Linux with onnxruntime-gpu, use lazy loading to avoid CUDA conflicts
    # with PyTorch. The actual import happens when a model is loaded.
    if not use_lazy_ort:
        try:
            from polyinfer.backends.onnxruntime import ONNXRuntimeBackend

            register_backend("onnxruntime", ONNXRuntimeBackend)
        except ImportError:
            pass
    else:
        # Register a lazy-loading placeholder that defers the actual import
        # This allows users to import polyinfer and torch in any order
        _register_lazy_onnxruntime()

    # OpenVINO backend (Tier 1) - CPU-focused, no CUDA conflict
    try:
        from polyinfer.backends.openvino import OpenVINOBackend

        register_backend("openvino", OpenVINOBackend)
    except ImportError:
        pass

    # TensorRT backend (Tier 2)
    # Skip on Linux or if PyTorch is loaded to avoid CUDA library conflicts.
    # Users can still use ONNX Runtime's TensorRT EP (device="tensorrt").
    if not _should_skip_native_tensorrt():
        try:
            from polyinfer.backends.tensorrt import TensorRTBackend

            register_backend("tensorrt", TensorRTBackend)
        except ImportError:
            pass

    # IREE backend (Tier 2)
    # On Linux, IREE runtime might load CUDA libraries when imported,
    # which can conflict with PyTorch. Use lazy loading on Linux.
    if not sys.platform.startswith("linux"):
        try:
            from polyinfer.backends.iree import IREEBackend

            register_backend("iree", IREEBackend)
        except ImportError:
            pass
    else:
        _register_lazy_iree()


def _register_lazy_iree():
    """Register a lazy-loading IREE backend wrapper.

    This is used on Linux to avoid importing iree.runtime at registration time,
    as it might load CUDA libraries that conflict with PyTorch.
    """
    from polyinfer.backends.base import Backend

    class LazyIREEBackend(Backend):
        """Lazy-loading wrapper for IREE backend."""

        _real_backend = None
        _import_attempted = False
        _import_error = None

        @classmethod
        def _ensure_loaded(cls):
            """Load the real backend on first use."""
            if cls._import_attempted:
                if cls._import_error:
                    raise cls._import_error
                return

            cls._import_attempted = True
            try:
                from polyinfer.backends.iree.backend import IREEBackend
                cls._real_backend = IREEBackend()
            except ImportError as e:
                cls._import_error = RuntimeError(
                    f"IREE not available: {e}. "
                    "Install with: pip install iree-base-runtime iree-base-compiler"
                )
                raise cls._import_error

        @property
        def name(self) -> str:
            return "iree"

        @property
        def supported_devices(self) -> list[str]:
            # Return expected devices without importing iree
            return ["cpu", "vulkan", "cuda"]

        @property
        def version(self) -> str:
            try:
                self._ensure_loaded()
                return self._real_backend.version
            except Exception:
                return "not loaded"

        @property
        def priority(self) -> int:
            return 40

        def is_available(self) -> bool:
            try:
                import importlib.metadata as metadata
                metadata.version("iree-base-runtime")
                return True
            except Exception:
                pass
            return False

        def load(self, model_path: str, device: str = "cpu", **kwargs):
            """Load model - this triggers the actual iree import."""
            self._ensure_loaded()
            return self._real_backend.load(model_path, device, **kwargs)

    try:
        register_backend("iree", LazyIREEBackend)
    except Exception:
        pass


def _register_lazy_onnxruntime():
    """Register a lazy-loading ONNX Runtime backend wrapper.

    This is used when we can't safely import onnxruntime at registration time
    (e.g., on Linux when PyTorch is already loaded). The actual import happens
    when the backend is first used.
    """
    from polyinfer.backends.base import Backend

    class LazyONNXRuntimeBackend(Backend):
        """Lazy-loading wrapper for ONNX Runtime backend.

        Defers the actual onnxruntime import until the backend is used,
        allowing users to control when CUDA libraries are loaded.
        """

        _real_backend = None
        _import_attempted = False
        _import_error = None

        @classmethod
        def _ensure_loaded(cls):
            """Load the real backend on first use."""
            if cls._import_attempted:
                if cls._import_error:
                    raise cls._import_error
                return

            cls._import_attempted = True
            try:
                from polyinfer.backends.onnxruntime.backend import ONNXRuntimeBackend
                cls._real_backend = ONNXRuntimeBackend()
            except ImportError as e:
                cls._import_error = RuntimeError(
                    f"ONNX Runtime not available: {e}. "
                    "Install with: pip install onnxruntime or onnxruntime-gpu"
                )
                raise cls._import_error

        @property
        def name(self) -> str:
            return "onnxruntime"

        @property
        def supported_devices(self) -> list[str]:
            # Return expected devices based on installed packages
            # without importing onnxruntime (which would load CUDA libs)
            devices = ["cpu"]
            try:
                import importlib.metadata as metadata
                # If onnxruntime-gpu is installed, CUDA devices are likely available
                metadata.version("onnxruntime-gpu")
                devices.append("cuda")
                # Only advertise tensorrt if libraries are actually available
                # TensorRT EP often fails even when ORT reports it as available
                if _verify_tensorrt_libs_available():
                    devices.append("tensorrt")
            except Exception:
                pass
            return devices

        @property
        def version(self) -> str:
            try:
                self._ensure_loaded()
                return self._real_backend.version
            except Exception:
                return "not loaded"

        @property
        def priority(self) -> int:
            return 60

        def is_available(self) -> bool:
            # Check if onnxruntime package exists without importing it
            try:
                import importlib.metadata as metadata
                metadata.version("onnxruntime")
                return True
            except Exception:
                pass
            try:
                import importlib.metadata as metadata
                metadata.version("onnxruntime-gpu")
                return True
            except Exception:
                pass
            return False

        def load(self, model_path: str, device: str = "cpu", **kwargs):
            """Load model - this triggers the actual onnxruntime import."""
            self._ensure_loaded()
            return self._real_backend.load(model_path, device, **kwargs)

    try:
        register_backend("onnxruntime", LazyONNXRuntimeBackend)
    except Exception:
        pass  # Registration failed, skip silently
