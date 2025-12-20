"""OpenVINO backend implementation."""

import numpy as np

from polyinfer._logging import get_logger
from polyinfer.backends.base import Backend, CompiledModel

_logger = get_logger("backends.openvino")

# Check if OpenVINO is available
try:
    import openvino as ov
    from openvino import CompiledModel as OVCompiledModel
    from openvino import Core
    from openvino import Tensor as OVTensor

    OPENVINO_AVAILABLE = True
    _logger.debug(f"OpenVINO {ov.__version__} available")
except ImportError:
    OPENVINO_AVAILABLE = False
    ov = None
    Core = None
    _logger.debug("OpenVINO not installed")


# Performance hint mapping
PERF_HINTS = {
    0: "LATENCY",  # Optimize for low latency
    1: "THROUGHPUT",  # Optimize for throughput
    2: "LATENCY",  # Default to latency
    3: "LATENCY",  # Max optimization = latency focused
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
        return self._input_shapes

    @property
    def output_shapes(self) -> list[tuple]:
        return self._output_shapes

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run inference."""
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
            return outputs[0]
        return tuple(outputs)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with named inputs/outputs."""
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
            return ov.__version__
        return "not installed"

    @property
    def priority(self) -> int:
        # OpenVINO is great for CPU
        return 70

    def is_available(self) -> bool:
        return OPENVINO_AVAILABLE

    def get_available_devices(self) -> list[str]:
        """Get raw OpenVINO device names."""
        if not OPENVINO_AVAILABLE:
            return []
        return self.core.available_devices

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
                - optimization_level: 0=throughput, 1=balanced, 2=latency (default)
                - num_threads: Number of inference threads
                - enable_caching: Enable model caching
                - cache_dir: Directory for cached models

        Returns:
            Compiled model ready for inference
        """
        if not OPENVINO_AVAILABLE:
            _logger.error("OpenVINO not installed")
            raise RuntimeError("openvino not installed. Run: pip install openvino")

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
        model = self.core.read_model(model_path)

        # Configure properties
        config = {}

        # Performance hint
        opt_level = kwargs.get("optimization_level", 2)
        perf_hint = PERF_HINTS.get(opt_level, "LATENCY")
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
        compiled = self.core.compile_model(model, ov_device, config)

        _logger.info(f"Model compiled on {ov_device}")

        return OpenVINOModel(
            compiled_model=compiled,
            device=device,
        )
