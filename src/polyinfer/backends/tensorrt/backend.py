"""Native TensorRT backend implementation."""

from pathlib import Path

import numpy as np

from polyinfer._logging import get_logger
from polyinfer.backends.base import Backend, CompiledModel

_logger = get_logger("backends.tensorrt")

# Check if TensorRT is available
try:
    import tensorrt as trt

    # cuda-python 12.x uses cuda.cudart, 13.x+ uses cuda.bindings.runtime
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        import cuda.cudart as cudart

    TENSORRT_AVAILABLE = True
    _logger.debug(f"TensorRT {trt.__version__} available")
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cudart = None
    _logger.debug("TensorRT not installed")


class TensorRTModel(CompiledModel):
    """TensorRT engine wrapper."""

    def __init__(
        self,
        engine: "trt.ICudaEngine",
        context: "trt.IExecutionContext",
        device_id: int = 0,
    ):
        self._engine = engine
        self._context = context
        self._device_id = device_id

        # Get input/output info
        self._input_names = []
        self._output_names = []
        self._input_shapes = []  # May contain -1 for dynamic dims
        self._output_shapes = []  # May contain -1 for dynamic dims
        self._bindings = {}
        self._has_dynamic_shapes = False

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            mode = engine.get_tensor_mode(name)

            # Check for dynamic dimensions
            if -1 in shape or any(d < 0 for d in shape):
                self._has_dynamic_shapes = True

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                self._input_shapes.append(tuple(shape))
            else:
                self._output_names.append(name)
                self._output_shapes.append(tuple(shape))

            self._bindings[name] = {
                "shape": tuple(shape),
                "dtype": trt.nptype(engine.get_tensor_dtype(name)),
                "is_input": mode == trt.TensorIOMode.INPUT,
            }

        # Create CUDA stream
        err, self._stream = cudart.cudaStreamCreate()

        # For static shapes, pre-allocate GPU buffers
        # For dynamic shapes, allocate lazily on first inference
        self._d_inputs: dict[str, int] = {}
        self._d_outputs: dict[str, int] = {}
        self._h_outputs: dict[str, np.ndarray] = {}
        self._allocated_shapes: dict[str, tuple[int, ...]] = {}  # Track allocated buffer shapes

        if not self._has_dynamic_shapes:
            self._allocate_buffers()

    @property
    def backend_name(self) -> str:
        return f"tensorrt-cuda:{self._device_id}"

    @property
    def device(self) -> str:
        return f"cuda:{self._device_id}"

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

    def _allocate_buffers(self, input_shapes: dict[str, tuple] | None = None):
        """Allocate GPU buffers for inputs and outputs.

        For dynamic shapes, input_shapes must be provided to determine output shapes.
        """
        # If we have dynamic shapes, set input shapes on context first
        if input_shapes:
            for name, shape in input_shapes.items():
                self._context.set_input_shape(name, shape)

        # Allocate input buffers
        for name in self._input_names:
            shape = input_shapes[name] if input_shapes else self._bindings[name]["shape"]

            dtype = self._bindings[name]["dtype"]
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            # Check if we need to reallocate
            if name in self._allocated_shapes and self._allocated_shapes[name] == shape:
                continue  # Already allocated with correct shape

            # Free old buffer if exists
            if name in self._d_inputs:
                cudart.cudaFree(self._d_inputs[name])

            err, ptr = cudart.cudaMalloc(size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to allocate CUDA memory for {name}: {err}")
            self._d_inputs[name] = ptr
            self._allocated_shapes[name] = shape

        # Allocate output buffers based on context's computed shapes
        for name in self._output_names:
            if self._has_dynamic_shapes and input_shapes:
                # Get actual output shape from context after setting input shapes
                shape = tuple(self._context.get_tensor_shape(name))
            else:
                shape = self._bindings[name]["shape"]

            dtype = self._bindings[name]["dtype"]
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            # Check if we need to reallocate
            if name in self._allocated_shapes and self._allocated_shapes[name] == shape:
                continue

            # Free old buffer if exists
            if name in self._d_outputs:
                cudart.cudaFree(self._d_outputs[name])

            err, ptr = cudart.cudaMalloc(size)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to allocate CUDA memory for {name}: {err}")
            self._d_outputs[name] = ptr
            self._h_outputs[name] = np.empty(shape, dtype=dtype)
            self._allocated_shapes[name] = shape

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run inference."""
        # For dynamic shapes, ensure buffers are allocated for current input shapes
        if self._has_dynamic_shapes:
            input_shapes = {
                name: tuple(data.shape)
                for name, data in zip(self._input_names, inputs, strict=False)
            }
            self._allocate_buffers(input_shapes)

        # Copy inputs to GPU
        for name, data in zip(self._input_names, inputs, strict=False):
            data = np.ascontiguousarray(data)
            cudart.cudaMemcpyAsync(
                self._d_inputs[name],
                data.ctypes.data,
                data.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )

        # Set tensor addresses
        for name, ptr in self._d_inputs.items():
            self._context.set_tensor_address(name, ptr)
        for name, ptr in self._d_outputs.items():
            self._context.set_tensor_address(name, ptr)

        # Execute
        self._context.execute_async_v3(self._stream)

        # Copy outputs to host
        outputs = []
        for name in self._output_names:
            cudart.cudaMemcpyAsync(
                self._h_outputs[name].ctypes.data,
                self._d_outputs[name],
                self._h_outputs[name].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )
            outputs.append(self._h_outputs[name].copy())

        # Synchronize
        cudart.cudaStreamSynchronize(self._stream)

        if len(outputs) == 1:
            result: np.ndarray = outputs[0]
            return result
        return tuple(outputs)

    def __del__(self):
        """Clean up CUDA resources."""
        if hasattr(self, "_stream"):
            cudart.cudaStreamDestroy(self._stream)
        for ptr in getattr(self, "_d_inputs", {}).values():
            cudart.cudaFree(ptr)
        for ptr in getattr(self, "_d_outputs", {}).values():
            cudart.cudaFree(ptr)


class TensorRTBackend(Backend):
    """Native TensorRT backend for NVIDIA GPUs."""

    def __init__(self):
        self._logger = None

    @property
    def logger(self):
        """Lazy-initialize TensorRT logger."""
        if self._logger is None and TENSORRT_AVAILABLE:
            self._logger = trt.Logger(trt.Logger.WARNING)
        return self._logger

    @property
    def name(self) -> str:
        return "tensorrt"

    @property
    def supported_devices(self) -> list[str]:
        if not TENSORRT_AVAILABLE:
            return []
        return ["cuda", "tensorrt"]

    @property
    def version(self) -> str:
        if TENSORRT_AVAILABLE:
            return str(trt.__version__)
        return "not installed"

    @property
    def priority(self) -> int:
        # Native TensorRT has lower priority than ONNX Runtime's TensorRT EP
        # because tensorrt-cu12-libs causes CUDA conflicts with PyTorch.
        # Users who want native TensorRT can specify backend="tensorrt" explicitly.
        return 50

    def is_available(self) -> bool:
        return TENSORRT_AVAILABLE

    def load(
        self,
        model_path: str,
        device: str = "cuda:0",
        **kwargs,
    ) -> TensorRTModel:
        """Load an ONNX model and build TensorRT engine.

        Args:
            model_path: Path to ONNX file
            device: Target device (cuda:N)
            **kwargs: Additional options:

                Precision:
                    fp16 (bool): Enable FP16 precision. Default: False
                    int8 (bool): Enable INT8 precision. Default: False
                    tf32 (bool): Enable TF32 precision (Ampere+). Default: True
                    bf16 (bool): Enable BF16 precision (Ada+). Default: False
                    fp8 (bool): Enable FP8 precision (Hopper+). Default: False
                    strict_types (bool): Force layers to use specified precision. Default: False

                Optimization:
                    builder_optimization_level (int): 0-5, higher = more optimization time,
                        better runtime. Default: 3
                        - 0: Fastest build, no optimization
                        - 3: Default balance
                        - 5: Maximum optimization (longest build time)
                    workspace_size (int): Max workspace memory in bytes. Default: 1GB
                    avg_timing_iterations (int): Timing iterations for kernel selection. Default: 1
                    sparsity (bool): Enable structured sparsity (Ampere+). Default: False

                Caching:
                    cache_path (str): Path to save/load engine cache
                    timing_cache_path (str): Path to timing cache for faster rebuilds
                    force_rebuild (bool): Ignore cached engine, rebuild. Default: False

                Hardware:
                    dla_core (int): DLA core to use (-1 = GPU only). Default: -1
                    gpu_fallback (bool): Allow GPU fallback for unsupported DLA layers. Default: True

                Profiling:
                    profiling_verbosity (str): 'none', 'layer_names_only', 'detailed'. Default: 'none'
                    engine_capability (str): 'default', 'safe', 'dla_standalone'. Default: 'default'

                Dynamic shapes:
                    min_shapes (dict): Min shapes for dynamic inputs. {name: (N,C,H,W)}
                    opt_shapes (dict): Optimal shapes for dynamic inputs.
                    max_shapes (dict): Max shapes for dynamic inputs.

        Returns:
            TensorRT model ready for inference

        Example:
            >>> model = pi.load("model.onnx", backend="tensorrt", device="cuda",
            ...     fp16=True,
            ...     builder_optimization_level=5,
            ...     workspace_size=4 << 30,  # 4GB
            ...     timing_cache_path="./timing.cache"
            ... )
        """
        if not TENSORRT_AVAILABLE:
            _logger.error("TensorRT not installed")
            raise RuntimeError(
                "TensorRT not installed. Install CUDA, TensorRT, and run: pip install tensorrt"
            )

        _logger.debug(f"Loading model: {model_path}")

        # Parse device
        device_id = int(device.split(":")[1]) if ":" in device else 0
        cudart.cudaSetDevice(device_id)
        _logger.debug(f"Using CUDA device: {device_id}")

        # Check for cached engine
        model_path_obj = Path(model_path)
        cache_path = kwargs.get("cache_path")
        cache_path = (
            model_path_obj.with_suffix(".engine") if cache_path is None else Path(cache_path)
        )

        # Try to load cached engine (unless force_rebuild)
        if cache_path.exists() and not kwargs.get("force_rebuild", False):
            _logger.info(f"Loading cached engine: {cache_path}")
            return self._load_engine(cache_path, device_id)

        # Build engine from ONNX with full options
        _logger.info("Building TensorRT engine from ONNX (this may take a while)...")
        engine = self._build_engine(model_path_obj, **kwargs)

        # Cache the engine
        _logger.debug(f"Saving engine to: {cache_path}")
        self._save_engine(engine, cache_path)

        # Create execution context
        context = engine.create_execution_context()
        _logger.info("TensorRT engine built and ready")

        return TensorRTModel(engine, context, device_id)

    def _build_engine(
        self,
        onnx_path: Path,
        **kwargs,
    ) -> "trt.ICudaEngine":
        """Build TensorRT engine from ONNX with full options support.

        All options are passed via kwargs from load().
        """
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX - use parse_from_file to handle external data files (.onnx.data)
        _logger.debug(f"Parsing ONNX file: {onnx_path}")
        onnx_path_str = str(onnx_path.resolve())
        if not parser.parse_from_file(onnx_path_str):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            _logger.error(f"ONNX parse failed: {errors}")
            raise RuntimeError(f"ONNX parse failed: {errors}")

        # Build config
        config = builder.create_builder_config()

        # === Memory ===
        workspace_size = kwargs.get("workspace_size", 1 << 30)  # 1GB default
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # === Precision flags ===
        if kwargs.get("fp16", False):
            config.set_flag(trt.BuilderFlag.FP16)
            _logger.debug("FP16 precision enabled")
        if kwargs.get("int8", False):
            config.set_flag(trt.BuilderFlag.INT8)
            _logger.debug("INT8 precision enabled")
        if kwargs.get("bf16", False) and hasattr(trt.BuilderFlag, "BF16"):
            config.set_flag(trt.BuilderFlag.BF16)
        if kwargs.get("fp8", False) and hasattr(trt.BuilderFlag, "FP8"):
            config.set_flag(trt.BuilderFlag.FP8)
        if not kwargs.get("tf32", True) and hasattr(trt.BuilderFlag, "TF32"):
            # TF32 enabled by default on Ampere+
            config.clear_flag(trt.BuilderFlag.TF32)
        if kwargs.get("strict_types", False):
            if hasattr(trt.BuilderFlag, "STRICT_TYPES"):
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            elif hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        # === Optimization level (0-5) ===
        opt_level = kwargs.get("builder_optimization_level", 3)
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = opt_level

        # === Timing iterations ===
        avg_timing = kwargs.get("avg_timing_iterations", 1)
        if hasattr(config, "avg_timing_iterations"):
            config.avg_timing_iterations = avg_timing

        # === Sparsity (Ampere+) ===
        if kwargs.get("sparsity", False) and hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # === Timing cache ===
        timing_cache_path = kwargs.get("timing_cache_path")
        timing_cache = None
        if timing_cache_path:
            timing_cache_path = Path(timing_cache_path)
            if timing_cache_path.exists():
                with open(timing_cache_path, "rb") as f:
                    timing_cache = config.create_timing_cache(f.read())
            else:
                timing_cache = config.create_timing_cache(b"")
            config.set_timing_cache(timing_cache, ignore_mismatch=False)

        # === DLA (Deep Learning Accelerator) ===
        dla_core = kwargs.get("dla_core", -1)
        if dla_core >= 0:
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = dla_core
            if kwargs.get("gpu_fallback", True):
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # === Profiling verbosity ===
        verbosity = kwargs.get("profiling_verbosity", "none")
        if hasattr(config, "profiling_verbosity"):
            verbosity_map = {
                "none": trt.ProfilingVerbosity.NONE,
                "layer_names_only": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                "detailed": trt.ProfilingVerbosity.DETAILED,
            }
            if verbosity in verbosity_map:
                config.profiling_verbosity = verbosity_map[verbosity]

        # === Engine capability ===
        capability = kwargs.get("engine_capability", "default")
        if hasattr(config, "engine_capability"):
            cap_map = {}
            # TensorRT 10+ uses STANDARD, older versions use DEFAULT
            if hasattr(trt.EngineCapability, "STANDARD"):
                cap_map["default"] = trt.EngineCapability.STANDARD
                cap_map["standard"] = trt.EngineCapability.STANDARD
            elif hasattr(trt.EngineCapability, "DEFAULT"):
                cap_map["default"] = trt.EngineCapability.DEFAULT
            if hasattr(trt.EngineCapability, "SAFETY"):
                cap_map["safe"] = trt.EngineCapability.SAFETY
            if hasattr(trt.EngineCapability, "DLA_STANDALONE"):
                cap_map["dla_standalone"] = trt.EngineCapability.DLA_STANDALONE
            if capability in cap_map:
                config.engine_capability = cap_map[capability]

        # === Dynamic shapes ===
        min_shapes = kwargs.get("min_shapes", {})
        opt_shapes = kwargs.get("opt_shapes", {})
        max_shapes = kwargs.get("max_shapes", {})

        # Check if any input has dynamic dimensions
        has_dynamic = False
        for i in range(network.num_inputs):
            shape = network.get_input(i).shape
            if -1 in shape or any(d < 0 for d in shape):
                has_dynamic = True
                break

        # Always create optimization profile if network has dynamic shapes
        if has_dynamic or min_shapes or opt_shapes or max_shapes:
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                name = input_tensor.name
                shape = list(input_tensor.shape)

                # Check if this input has dynamic dimensions
                if -1 in shape or any(d < 0 for d in shape):
                    # Replace dynamic dims with defaults (1 for batch, keep others)
                    # Use batch=1 for all profiles by default - this is safest for models
                    # with internal reshape ops that may have fixed batch dimensions
                    default_shape = tuple(1 if d < 0 else d for d in shape)
                    min_shape = min_shapes.get(name, default_shape)
                    opt_shape = opt_shapes.get(name, default_shape)
                    # Default max to same as opt (batch=1) to avoid shape conflicts
                    # Users can override with max_shapes for true dynamic batching
                    max_shape = max_shapes.get(name, default_shape)
                    profile.set_shape(name, min_shape, opt_shape, max_shape)

            config.add_optimization_profile(profile)

        # Build engine
        _logger.debug("Building serialized network...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            _logger.error("Failed to build TensorRT engine")
            raise RuntimeError("Failed to build TensorRT engine")

        # Save timing cache if used
        if timing_cache is not None and timing_cache_path:
            serialized_cache = config.get_timing_cache().serialize()
            with open(timing_cache_path, "wb") as f:
                f.write(serialized_cache)

        # Deserialize
        runtime = trt.Runtime(self.logger)
        return runtime.deserialize_cuda_engine(engine_bytes)

    def _save_engine(self, engine: "trt.ICudaEngine", path: Path) -> None:
        """Save engine to file."""
        serialized = engine.serialize()
        with open(path, "wb") as f:
            f.write(serialized)

    def _load_engine(self, path: Path, device_id: int) -> TensorRTModel:
        """Load engine from cache."""
        with open(path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()
            return TensorRTModel(engine, context, device_id)
