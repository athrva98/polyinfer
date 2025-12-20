"""Unified model loading and inference for PolyInfer."""

from pathlib import Path

import numpy as np

from polyinfer._logging import get_logger
from polyinfer.backends.base import CompiledModel
from polyinfer.config import InferenceConfig
from polyinfer.discovery import get_backend, select_backend

_logger = get_logger("model")


class Model:
    """Unified model wrapper that works with any backend.

    This is the main interface for loading and running inference.
    It provides a consistent API regardless of which backend is used.

    Example:
        >>> model = Model("yolov8n.onnx", device="cuda")
        >>> output = model(input_tensor)
        >>> print(f"Using: {model.backend_name}")
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        backend: str | None = None,
        config: InferenceConfig | None = None,
        **kwargs,
    ):
        """Load a model.

        Args:
            model_path: Path to ONNX model file
            device: Target device (cpu, cuda, cuda:0, directml, etc.)
            backend: Specific backend to use (None for auto-select)
            config: Inference configuration
            **kwargs: Backend-specific options
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            _logger.error(f"Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        _logger.debug(f"Loading model: {model_path}")

        # Merge config with kwargs
        if config:
            device = config.device
            backend = config.backend or backend
            kwargs.update(config.extra_options)

        # Normalize device and backend
        device = self._normalize_device(device)
        backend, device = self._normalize_backend(backend, device)
        self._device = device

        # Select backend
        if backend:
            _logger.debug(f"Using specified backend: {backend}")
            self._backend = get_backend(backend)
            if not self._backend.supports_device(device):
                _logger.error(f"Backend '{backend}' does not support device '{device}'")
                raise ValueError(
                    f"Backend '{backend}' does not support device '{device}'. "
                    f"Supported: {self._backend.supported_devices}"
                )
        else:
            _logger.debug(f"Auto-selecting backend for device: {device}")
            self._backend = select_backend(device)

        _logger.debug(
            f"Selected backend: {self._backend.name} (priority: {self._backend.priority})"
        )

        # Load the model
        _logger.debug(f"Loading with device: {device}")
        self._model: CompiledModel = self._backend.load(
            str(self.model_path),
            device=device,
            **kwargs,
        )

        _logger.info(f"Model loaded: {self.model_path.name} on {self._model.backend_name}")

    @staticmethod
    def _normalize_device(device: str) -> str:
        """Normalize device string."""
        device = device.lower().strip()
        # Aliases
        aliases = {
            "gpu": "cuda",
            "nvidia": "cuda",
            "trt": "tensorrt",
        }
        # Handle base device without index
        base = device.split(":")[0]
        if base in aliases:
            if ":" in device:
                return f"{aliases[base]}:{device.split(':')[1]}"
            return aliases[base]
        return device

    @staticmethod
    def _normalize_backend(backend: str | None, device: str) -> tuple[str | None, str]:
        """Normalize backend, handling special cases like tensorrt.

        Returns:
            Tuple of (backend, device) - both may be modified
        """
        if backend is None:
            return None, device

        backend = backend.lower().strip()

        # "tensorrt" backend should use onnxruntime with TensorRT EP
        # The native tensorrt backend requires separate TensorRT SDK installation
        # which is harder - so we route to onnxruntime by default
        if backend == "tensorrt":
            # Check if native tensorrt backend is available
            try:
                from polyinfer.backends.registry import _backends

                if "tensorrt" in _backends and _backends["tensorrt"].is_available():
                    return "tensorrt", device  # Use native
            except Exception:
                pass
            # Fall back to onnxruntime with TensorRT EP
            # Also set device to "tensorrt" so ONNX Runtime uses TensorRT EP
            return "onnxruntime", "tensorrt"

        return backend, device

    @property
    def backend_name(self) -> str:
        """Return the backend name being used."""
        return self._model.backend_name

    @property
    def device(self) -> str:
        """Return the device being used."""
        return self._model.device

    @property
    def input_names(self) -> list[str]:
        """Return input tensor names."""
        return self._model.input_names

    @property
    def output_names(self) -> list[str]:
        """Return output tensor names."""
        return self._model.output_names

    @property
    def input_shapes(self) -> list[tuple]:
        """Return input tensor shapes."""
        return self._model.input_shapes

    @property
    def output_shapes(self) -> list[tuple]:
        """Return output tensor shapes."""
        return self._model.output_shapes

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Run inference.

        Args:
            *inputs: Input numpy arrays in order

        Returns:
            Output array(s)
        """
        return self._model(*inputs)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with named inputs/outputs.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        return self._model.run(inputs)

    def benchmark(
        self,
        *inputs: np.ndarray,
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict:
        """Benchmark inference performance.

        Args:
            *inputs: Input numpy arrays
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with timing statistics
        """
        return self._model.benchmark(*inputs, warmup=warmup, iterations=iterations)

    def __repr__(self) -> str:
        return (
            f"Model(path={self.model_path.name!r}, "
            f"backend={self.backend_name!r}, "
            f"device={self.device!r})"
        )


def load(
    model_path: str | Path,
    device: str = "cpu",
    backend: str | None = None,
    **kwargs,
) -> Model:
    """Load a model for inference.

    This is the main entry point for loading models. It automatically
    selects the best available backend for the given device.

    Args:
        model_path: Path to ONNX model file
        device: Target device:
            - "cpu": CPU inference (default)
            - "cuda" or "cuda:0": NVIDIA GPU
            - "directml": DirectML (Windows, any GPU)
            - "tensorrt": TensorRT (NVIDIA, if available)
            - "vulkan": Vulkan (via IREE)
        backend: Specific backend to use (None for auto-select):
            - "onnxruntime": ONNX Runtime
            - "openvino": Intel OpenVINO
            - "tensorrt": NVIDIA TensorRT
            - "iree": Google IREE
        **kwargs: Backend-specific options

    Returns:
        Model instance ready for inference

    Example:
        >>> import polyinfer as pi

        # Auto-select best backend
        >>> model = pi.load("yolov8n.onnx", device="cpu")
        >>> print(model.backend_name)
        'openvino-cpu'

        # Explicit backend
        >>> model = pi.load("yolov8n.onnx", backend="onnxruntime", device="cuda")
        >>> output = model(input_tensor)
    """
    return Model(model_path, device=device, backend=backend, **kwargs)
