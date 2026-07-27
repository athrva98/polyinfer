"""OpenVINO backend implementation."""

import numpy as np

from polyinfer._logging import get_logger
from polyinfer.backends.base import (
    Backend,
    CompiledModel,
    describe_import_error,
    translate_errors,
)
from polyinfer.exceptions import (
    BackendNotAvailableError,
    InvalidInputError,
    InvalidOptionError,
    ModelLoadError,
)

_logger = get_logger("backends.openvino")

# Check if OpenVINO is available
IMPORT_ERROR: str | None = None
try:
    import openvino as ov
    from openvino import CompiledModel as OVCompiledModel
    from openvino import Core
    from openvino import Tensor as OVTensor

    OPENVINO_AVAILABLE = True
    _logger.debug(f"OpenVINO {ov.__version__} available")
except ImportError as e:
    OPENVINO_AVAILABLE = False
    ov = None
    Core = None
    IMPORT_ERROR = describe_import_error(
        e,
        packages=("openvino",),
        install_hint="pip install openvino",
    )
    _logger.debug(f"OpenVINO unavailable: {IMPORT_ERROR}")


# Valid OpenVINO PERFORMANCE_HINT values, for the explicit `performance_hint`
# option. Prefer this over the coarse numeric `optimization_level` scale.
PERFORMANCE_HINTS = ("LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT")

# Mapping from the generic `optimization_level` scale to an OpenVINO
# performance hint: lower favours throughput, higher favours latency.
#
# NOTE: this table used to read {0: LATENCY, 1: THROUGHPUT, 2: LATENCY}, which
# was the inverse of the documented "0=throughput ... 2=latency" contract at
# levels 0 and 1 - asking for throughput got you a latency-tuned model.
# OpenVINO offers no distinct "balanced" hint, so level 1 maps to THROUGHPUT;
# use `performance_hint` when you need exact control.
PERF_HINTS = {
    0: "THROUGHPUT",  # Maximize throughput
    1: "THROUGHPUT",  # "Balanced" - no distinct OpenVINO hint exists
    2: "LATENCY",  # Default: minimize single-inference latency
    3: "LATENCY",  # Latency-focused
}


class OpenVINOModel(CompiledModel):
    """OpenVINO compiled model wrapper."""

    def __init__(
        self,
        compiled_model: "OVCompiledModel",
        device: str,
    ):
        self._model = compiled_model
        self._device = device
        self._infer_request = compiled_model.create_infer_request()

        # Cache input/output metadata
        self._input_names = [inp.any_name for inp in compiled_model.inputs]
        self._output_names = [out.any_name for out in compiled_model.outputs]
        self._input_shapes = [self._get_shape(inp) for inp in compiled_model.inputs]
        self._output_shapes = [self._get_shape(out) for out in compiled_model.outputs]

    @staticmethod
    def _get_shape(port) -> list:
        """Extract shape from port, handling dynamic dimensions."""
        partial_shape = port.partial_shape
        shape = []
        for dim in partial_shape:
            if dim.is_static:
                shape.append(dim.get_length())
            else:
                shape.append(-1)  # Dynamic dimension
        return shape

    @property
    def backend_name(self) -> str:
        return f"openvino-{self._device.lower()}"

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
        return [tuple(s) for s in self._input_shapes]

    @property
    def output_shapes(self) -> list[tuple]:
        return [tuple(s) for s in self._output_shapes]

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run inference."""
        self._check_input_count(inputs)

        with translate_errors(self.backend_name):
            # Set inputs (must wrap in OVTensor)
            for i, data in enumerate(inputs):
                tensor = OVTensor(np.ascontiguousarray(data))
                self._infer_request.set_input_tensor(i, tensor)

            # Run inference
            self._infer_request.infer()

            # Get outputs
            outputs = []
            for i in range(len(self._output_names)):
                output_tensor = self._infer_request.get_output_tensor(i)
                outputs.append(output_tensor.data.copy())

        if len(outputs) == 1:
            result: np.ndarray = outputs[0]
            return result
        return tuple(outputs)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with named inputs/outputs."""
        missing = [name for name in self._input_names if name not in inputs]
        if missing:
            raise InvalidInputError(
                f"Missing required input(s) for {self.backend_name}: {missing}\n"
                f"Expected: {self._input_names}\n"
                f"Got: {sorted(inputs)}"
            )

        with translate_errors(self.backend_name):
            # Set inputs by name
            for name, data in inputs.items():
                tensor = OVTensor(np.ascontiguousarray(data))
                self._infer_request.set_tensor(name, tensor)

            # Run inference
            self._infer_request.infer()

            # Get outputs by name
            results = {}
            for name in self._output_names:
                output_tensor = self._infer_request.get_tensor(name)
                results[name] = output_tensor.data.copy()

        return results


class OpenVINOBackend(Backend):
    """OpenVINO backend for Intel-optimized inference."""

    def __init__(self):
        self._core: Core | None = None

    @property
    def core(self) -> "Core":
        """Lazy-initialize OpenVINO Core."""
        if self._core is None:
            self._core = Core()
        return self._core

    @property
    def name(self) -> str:
        return "openvino"

    @property
    def supported_devices(self) -> list[str]:
        """Return devices supported by OpenVINO."""
        if not OPENVINO_AVAILABLE:
            return []

        devices = []
        available = self.core.available_devices

        # Map OpenVINO device names to our standard names
        for dev in available:
            if dev == "CPU":
                devices.append("cpu")
            elif dev.startswith("GPU"):
                devices.append(
                    f"intel-gpu:{dev.replace('GPU.', '')}" if "." in dev else "intel-gpu"
                )
            elif dev == "NPU":
                devices.append("npu")

        return devices if devices else ["cpu"]

    @property
    def version(self) -> str:
        if OPENVINO_AVAILABLE:
            return str(ov.__version__)
        return "not installed"

    @property
    def priority(self) -> int:
        # OpenVINO is great for CPU
        return 70

    def is_available(self) -> bool:
        return OPENVINO_AVAILABLE

    @property
    def unavailable_reason(self) -> str | None:
        if OPENVINO_AVAILABLE:
            return None
        return IMPORT_ERROR or "not installed (pip install openvino)"

    def get_available_devices(self) -> list[str]:
        """Get raw OpenVINO device names."""
        if not OPENVINO_AVAILABLE:
            return []
        return list(self.core.available_devices)

    def load(
        self,
        model_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> OpenVINOModel:
        """Load an ONNX model with OpenVINO.

        Args:
            model_path: Path to ONNX file
            device: Target device (cpu, intel-gpu, npu)
            **kwargs: Additional options:
                - performance_hint: Explicit OpenVINO hint - "LATENCY",
                  "THROUGHPUT", or "CUMULATIVE_THROUGHPUT". Takes precedence
                  over optimization_level. Preferred for exact control.
                - optimization_level: Coarse scale, 0-3. 0 and 1 map to
                  THROUGHPUT, 2 (default) and 3 map to LATENCY. Raises
                  ValueError if out of range.
                - num_threads: Number of inference threads (CPU only)
                - enable_caching: Enable model caching
                - cache_dir: Directory for cached models

        Returns:
            Compiled model ready for inference

        Raises:
            InvalidOptionError: If performance_hint or optimization_level is
                invalid. Also a ValueError.
        """
        if not OPENVINO_AVAILABLE:
            _logger.error("OpenVINO not installed")
            raise BackendNotAvailableError(f"openvino is not available: {self.unavailable_reason}")

        _logger.debug(f"Loading model: {model_path}")

        # Map our device names to OpenVINO device names
        device_map = {
            "cpu": "CPU",
            "intel-gpu": "GPU",
            "gpu": "GPU",
            "npu": "NPU",
        }

        # Handle device:id format
        device_type = device.split(":")[0] if ":" in device else device
        device_id = device.split(":")[1] if ":" in device else None

        ov_device = device_map.get(device_type, device_type.upper())
        if device_id:
            ov_device = f"{ov_device}.{device_id}"

        _logger.debug(f"Target OpenVINO device: {ov_device}")

        # Read the model
        _logger.debug("Reading model...")
        try:
            model = self.core.read_model(model_path)
        except Exception as e:
            raise ModelLoadError(f"OpenVINO could not read {model_path}: {e}") from e

        # Configure properties
        config = {}

        # Performance hint. An explicit `performance_hint` wins over the
        # coarse numeric `optimization_level` scale.
        perf_hint = kwargs.get("performance_hint")
        if perf_hint is not None:
            perf_hint = str(perf_hint).upper()
            if perf_hint not in PERFORMANCE_HINTS:
                raise InvalidOptionError(
                    f"Invalid performance_hint {perf_hint!r}. "
                    f"Expected one of {list(PERFORMANCE_HINTS)}."
                )
        else:
            opt_level = kwargs.get("optimization_level", 2)
            if opt_level not in PERF_HINTS:
                raise InvalidOptionError(
                    f"Invalid optimization_level {opt_level!r} for the openvino backend. "
                    f"Expected one of {sorted(PERF_HINTS)} "
                    "(lower favours throughput, higher favours latency), "
                    "or pass performance_hint= for explicit control."
                )
            perf_hint = PERF_HINTS[opt_level]

        config["PERFORMANCE_HINT"] = perf_hint

        # Threading (CPU only)
        if device_type == "cpu":
            num_threads = kwargs.get("num_threads", 0)
            if num_threads > 0:
                config["INFERENCE_NUM_THREADS"] = num_threads

        # Model caching
        if kwargs.get("enable_caching", False):
            cache_dir = kwargs.get("cache_dir", "./ov_cache")
            config["CACHE_DIR"] = cache_dir

        # Compile the model
        _logger.debug(f"Compiling model with config: {config}")
        try:
            compiled = self.core.compile_model(model, ov_device, config)
        except Exception as e:
            raise ModelLoadError(
                f"OpenVINO failed to compile {model_path} for device '{ov_device}': {e}"
            ) from e

        _logger.info(f"Model compiled on {ov_device}")

        return OpenVINOModel(
            compiled_model=compiled,
            device=device,
        )
