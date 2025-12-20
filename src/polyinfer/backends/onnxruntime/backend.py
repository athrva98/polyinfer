"""ONNX Runtime backend implementation."""

import numpy as np

from polyinfer._logging import get_logger
from polyinfer.backends.base import Backend, CompiledModel

_logger = get_logger("backends.onnxruntime")

# Check if onnxruntime is available
try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
    _logger.debug(f"ONNX Runtime {ort.__version__} available")
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ort = None
    _logger.debug("ONNX Runtime not installed")


# Map device types to execution providers
DEVICE_TO_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "rocm": ["ROCMExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


class ONNXRuntimeModel(CompiledModel):
    """ONNX Runtime inference session wrapper."""

    def __init__(
        self,
        session: "ort.InferenceSession",
        device: str,
        provider: str,
    ):
        self._session = session
        self._device = device
        self._provider = provider

        # Cache input/output metadata
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._output_names = [out.name for out in session.get_outputs()]
        self._input_shapes = [inp.shape for inp in session.get_inputs()]
        self._output_shapes = [out.shape for out in session.get_outputs()]

    @property
    def backend_name(self) -> str:
        return f"onnxruntime-{self._provider.lower().replace('executionprovider', '')}"

    @property
    def device(self) -> str:
        return self._device

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    @property
    def input_shapes(self) -> list[tuple]:
        return self._input_shapes

    @property
    def output_shapes(self) -> list[tuple]:
        return self._output_shapes

    @property
    def provider(self) -> str:
        """Return the active execution provider."""
        return self._provider

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run inference."""
        # Build input dict
        input_dict = {name: arr for name, arr in zip(self._input_names, inputs, strict=False)}

        # Run inference
        outputs = self._session.run(None, input_dict)

        if len(outputs) == 1:
            result: np.ndarray = outputs[0]
            return result
        return tuple(outputs)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with named inputs/outputs."""
        outputs = self._session.run(None, inputs)
        return dict(zip(self._output_names, outputs, strict=False))


def _verify_tensorrt_ep_works() -> bool:
    """Verify TensorRT EP actually works by checking library availability.

    ONNX Runtime may report TensorrtExecutionProvider as available even when
    the TensorRT libraries aren't properly installed or accessible. This causes
    session creation to fail with RegisterTensorRTPluginsAsCustomOps errors.

    Returns:
        True if TensorRT EP is likely to work, False otherwise.
    """
    import sys

    if sys.platform == "win32":
        # On Windows, check if nvinfer DLLs are findable
        import ctypes

        try:
            ctypes.CDLL("nvinfer_10.dll")
            return True
        except OSError:
            pass
        try:
            ctypes.CDLL("nvinfer.dll")
            return True
        except OSError:
            pass
        return False
    else:
        # On Linux, check if libnvinfer is loaded or loadable
        import ctypes

        # First check if already loaded (from our preload)
        try:
            # Try to find the symbol in already-loaded libraries
            _ = ctypes.CDLL(None).nvinfer_version
            return True
        except (OSError, AttributeError):
            pass

        # Try to load it
        for lib_name in ["libnvinfer.so.10", "libnvinfer.so.8", "libnvinfer.so"]:
            try:
                ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                pass

        return False


# Cache the TensorRT EP verification result
_tensorrt_ep_verified: bool | None = None


class ONNXRuntimeBackend(Backend):
    """ONNX Runtime backend supporting multiple execution providers."""

    @property
    def name(self) -> str:
        return "onnxruntime"

    @property
    def supported_devices(self) -> list[str]:
        """Return devices supported by available providers."""
        global _tensorrt_ep_verified

        if not ONNXRUNTIME_AVAILABLE:
            return []

        devices = ["cpu"]  # Always available
        providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in providers:
            devices.append("cuda")
        if "TensorrtExecutionProvider" in providers:
            # Verify TensorRT EP actually works before advertising it
            if _tensorrt_ep_verified is None:
                _tensorrt_ep_verified = _verify_tensorrt_ep_works()
            if _tensorrt_ep_verified:
                devices.append("tensorrt")
        if "DmlExecutionProvider" in providers:
            devices.append("directml")
        if "ROCMExecutionProvider" in providers:
            devices.append("rocm")
        if "CoreMLExecutionProvider" in providers:
            devices.append("coreml")

        return devices

    @property
    def version(self) -> str:
        if ONNXRUNTIME_AVAILABLE:
            return str(ort.__version__)
        return "not installed"

    @property
    def priority(self) -> int:
        # ONNX Runtime is a solid default
        return 60

    def is_available(self) -> bool:
        return ONNXRUNTIME_AVAILABLE

    def get_available_providers(self) -> list[str]:
        """Get list of available execution providers."""
        if not ONNXRUNTIME_AVAILABLE:
            return []
        return list(ort.get_available_providers())

    def load(
        self,
        model_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> ONNXRuntimeModel:
        """Load an ONNX model.

        Args:
            model_path: Path to ONNX file
            device: Target device (cpu, cuda, tensorrt, directml)
            **kwargs: Additional options:

                Session options:
                    providers (list): Explicit list of providers to try
                    provider_options (list[dict]): Provider-specific options (parallel to providers)
                    graph_optimization_level (int): 0=disable, 1=basic, 2=extended, 3=all. Default: 3
                    intra_op_num_threads (int): Threads for intra-op parallelism
                    inter_op_num_threads (int): Threads for inter-op parallelism
                    enable_mem_pattern (bool): Enable memory pattern optimization. Default: True
                    enable_cpu_mem_arena (bool): Enable CPU memory arena. Default: True

                CUDA EP options (device='cuda'):
                    device_id (int): GPU device ID. Default: 0
                    cuda_mem_limit (int): Max GPU memory in bytes
                    arena_extend_strategy (str): 'kNextPowerOfTwo' or 'kSameAsRequested'
                    cudnn_conv_algo_search (str): 'EXHAUSTIVE', 'HEURISTIC', 'DEFAULT'
                    do_copy_in_default_stream (bool): Use default CUDA stream. Default: True

                TensorRT EP options (device='tensorrt'):
                    fp16 (bool): Enable FP16 precision. Default: False
                    int8 (bool): Enable INT8 precision. Default: False
                    cache_dir (str): TensorRT engine cache directory
                    builder_optimization_level (int): 0-5. Default: 3
                    timing_cache_path (str): Path to timing cache
                    max_workspace_size (int): Max workspace in bytes. Default: 1GB
                    min_subgraph_size (int): Min nodes for TRT subgraph. Default: 5
                    max_partition_iterations (int): Max partitioning iterations. Default: 1000
                    dla_enable (bool): Enable DLA. Default: False
                    dla_core (int): DLA core to use. Default: 0
                    force_sequential_engine_build (bool): Build engines sequentially. Default: False

                DirectML EP options (device='directml'):
                    device_id (int): GPU device ID. Default: 0

        Returns:
            Loaded model ready for inference

        Example:
            >>> # TensorRT via ONNX Runtime with max optimization
            >>> model = pi.load("model.onnx", device="tensorrt",
            ...     fp16=True,
            ...     builder_optimization_level=5,
            ...     cache_dir="./trt_cache"
            ... )
        """
        if not ONNXRUNTIME_AVAILABLE:
            _logger.error("ONNX Runtime not installed")
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")

        _logger.debug(f"Loading model: {model_path}")

        # Normalize device
        device_type = device.split(":")[0] if ":" in device else device
        device_id = int(device.split(":")[1]) if ":" in device else 0

        # Setup TensorRT library paths if TensorRT is requested
        # This must happen BEFORE creating the session
        if device_type == "tensorrt":
            _logger.debug("Setting up TensorRT paths for TensorRT EP")
            from polyinfer.nvidia_setup import setup_tensorrt_paths

            setup_tensorrt_paths()

        # Get providers for device
        providers = kwargs.pop("providers", None)
        if providers is None:
            providers = DEVICE_TO_PROVIDERS.get(device_type, ["CPUExecutionProvider"])

        # Filter to available providers
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available]
        _logger.debug(f"Available providers: {list(available)}")
        _logger.debug(f"Selected providers: {providers}")

        if not providers:
            _logger.error(f"No execution provider available for device '{device}'")
            raise RuntimeError(
                f"No execution provider available for device '{device}'. "
                f"Available: {list(available)}"
            )

        # Build provider options
        provider_options = kwargs.pop("provider_options", None)
        if provider_options is None:
            provider_options = []
            for provider in providers:
                opts = {}

                if provider == "CUDAExecutionProvider":
                    opts["device_id"] = str(device_id)
                    if "cuda_mem_limit" in kwargs:
                        opts["gpu_mem_limit"] = str(kwargs["cuda_mem_limit"])
                    if "arena_extend_strategy" in kwargs:
                        opts["arena_extend_strategy"] = kwargs["arena_extend_strategy"]
                    if "cudnn_conv_algo_search" in kwargs:
                        opts["cudnn_conv_algo_search"] = kwargs["cudnn_conv_algo_search"]
                    if "do_copy_in_default_stream" in kwargs:
                        opts["do_copy_in_default_stream"] = str(
                            int(kwargs["do_copy_in_default_stream"])
                        )

                elif provider == "TensorrtExecutionProvider":
                    opts["device_id"] = str(device_id)
                    # Precision (newer ORT versions expect "True"/"False" not "1"/"0")
                    opts["trt_fp16_enable"] = "True" if kwargs.get("fp16", False) else "False"
                    opts["trt_int8_enable"] = "True" if kwargs.get("int8", False) else "False"
                    # Caching
                    opts["trt_engine_cache_enable"] = "True"
                    opts["trt_engine_cache_path"] = kwargs.get("cache_dir", "./trt_cache")
                    # Optimization
                    if "builder_optimization_level" in kwargs:
                        opts["trt_builder_optimization_level"] = str(
                            kwargs["builder_optimization_level"]
                        )
                    if "timing_cache_path" in kwargs:
                        opts["trt_timing_cache_path"] = kwargs["timing_cache_path"]
                        opts["trt_timing_cache_enable"] = "True"
                    if "max_workspace_size" in kwargs:
                        opts["trt_max_workspace_size"] = str(kwargs["max_workspace_size"])
                    else:
                        opts["trt_max_workspace_size"] = str(1 << 30)  # 1GB default
                    # Subgraph control
                    if "min_subgraph_size" in kwargs:
                        opts["trt_min_subgraph_size"] = str(kwargs["min_subgraph_size"])
                    if "max_partition_iterations" in kwargs:
                        opts["trt_max_partition_iterations"] = str(
                            kwargs["max_partition_iterations"]
                        )
                    # DLA
                    if kwargs.get("dla_enable", False):
                        opts["trt_dla_enable"] = "True"
                        opts["trt_dla_core"] = str(kwargs.get("dla_core", 0))
                    # Build options
                    if kwargs.get("force_sequential_engine_build", False):
                        opts["trt_force_sequential_engine_build"] = "True"

                elif provider == "DmlExecutionProvider" or provider == "ROCMExecutionProvider":
                    opts["device_id"] = str(device_id)

                provider_options.append(opts)

        # Session options
        sess_options = ort.SessionOptions()

        # Graph optimization
        opt_level = kwargs.get("graph_optimization_level", 99)
        if opt_level == 0:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        elif opt_level == 1:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif opt_level == 2:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Threading
        if "intra_op_num_threads" in kwargs:
            sess_options.intra_op_num_threads = kwargs["intra_op_num_threads"]
        if "inter_op_num_threads" in kwargs:
            sess_options.inter_op_num_threads = kwargs["inter_op_num_threads"]

        # Memory options
        if "enable_mem_pattern" in kwargs:
            sess_options.enable_mem_pattern = kwargs["enable_mem_pattern"]
        if "enable_cpu_mem_arena" in kwargs:
            sess_options.enable_cpu_mem_arena = kwargs["enable_cpu_mem_arena"]

        # Create session with fallback handling for TensorRT EP issues
        # TensorRT EP can fail during session creation even if it shows as available
        # (e.g., RegisterTensorRTPluginsAsCustomOps error). In this case, fall back
        # to CUDA EP if available.
        _logger.debug("Creating inference session...")
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options if provider_options else None,
            )
        except RuntimeError as e:
            error_msg = str(e)
            # Check if this is a TensorRT-specific error and we can fall back
            if "TensorRT" in error_msg or "RegisterTensorRTPluginsAsCustomOps" in error_msg:
                if "TensorrtExecutionProvider" in providers:
                    # Try falling back to CUDA EP
                    fallback_providers = [p for p in providers if p != "TensorrtExecutionProvider"]
                    fallback_options = (
                        [
                            opt
                            for i, opt in enumerate(provider_options)
                            if providers[i] != "TensorrtExecutionProvider"
                        ]
                        if provider_options
                        else None
                    )

                    if fallback_providers:
                        _logger.warning(
                            f"TensorRT EP failed, falling back to {fallback_providers[0]}"
                        )
                        import warnings

                        warnings.warn(
                            f"TensorRT EP failed ({error_msg[:100]}...), "
                            f"falling back to {fallback_providers[0]}",
                            UserWarning,
                            stacklevel=2,
                        )
                        session = ort.InferenceSession(
                            model_path,
                            sess_options=sess_options,
                            providers=fallback_providers,
                            provider_options=fallback_options,
                        )
                    else:
                        _logger.error(f"TensorRT EP failed with no fallback: {error_msg}")
                        raise
                else:
                    _logger.error(f"Session creation failed: {error_msg}")
                    raise
            else:
                _logger.error(f"Session creation failed: {error_msg}")
                raise

        # Get the actual provider being used
        active_provider = session.get_providers()[0]
        _logger.info(f"Model loaded with {active_provider}")

        return ONNXRuntimeModel(
            session=session,
            device=device,
            provider=active_provider,
        )
