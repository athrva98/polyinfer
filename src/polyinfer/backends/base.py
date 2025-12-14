"""Base classes for all backends."""

from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import time


class CompiledModel(ABC):
    """Abstract base class for compiled/loaded models.

    All backend-specific model classes must inherit from this.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name (e.g., 'onnxruntime-cpu')."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this model runs on."""
        ...

    @property
    def input_names(self) -> list[str]:
        """Return input tensor names."""
        return []

    @property
    def output_names(self) -> list[str]:
        """Return output tensor names."""
        return []

    @property
    def input_shapes(self) -> list[tuple]:
        """Return input tensor shapes."""
        return []

    @property
    def output_shapes(self) -> list[tuple]:
        """Return output tensor shapes."""
        return []

    @abstractmethod
    def __call__(self, *inputs: np.ndarray) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
        """Run inference on input tensors.

        Args:
            *inputs: Input numpy arrays in order

        Returns:
            Single output array or tuple of output arrays
        """
        ...

    def run(
        self, inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Run inference with named inputs/outputs.

        Args:
            inputs: Dictionary mapping input names to numpy arrays

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Default implementation using positional call
        input_arrays = [inputs[name] for name in self.input_names]
        outputs = self(*input_arrays)

        if isinstance(outputs, np.ndarray):
            outputs = (outputs,)

        return dict(zip(self.output_names, outputs))

    def benchmark(
        self,
        *inputs: np.ndarray,
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict[str, Any]:
        """Benchmark inference performance.

        Args:
            *inputs: Input numpy arrays
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations

        Returns:
            Dictionary with timing statistics
        """
        # Warmup
        for _ in range(warmup):
            self(*inputs)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self(*inputs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        times = np.array(times)
        return {
            "backend": self.backend_name,
            "device": self.device,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "p90_ms": float(np.percentile(times, 90)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000 / np.mean(times)),
            "iterations": iterations,
        }


class Backend(ABC):
    """Abstract base class for inference backends.

    Each backend (ONNX Runtime, OpenVINO, TensorRT, etc.) must implement this.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'onnxruntime', 'openvino')."""
        ...

    @property
    @abstractmethod
    def supported_devices(self) -> list[str]:
        """Return list of supported device types (e.g., ['cpu', 'cuda'])."""
        ...

    @property
    def version(self) -> str:
        """Return the backend version."""
        return "unknown"

    @property
    def priority(self) -> int:
        """Return priority for auto-selection (higher = preferred).

        Default priorities:
        - TensorRT: 100 (fastest for NVIDIA)
        - ONNX Runtime CUDA: 80
        - OpenVINO: 70 (great for CPU)
        - ONNX Runtime CPU: 50
        - IREE: 40
        """
        return 50

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed)."""
        ...

    @abstractmethod
    def load(
        self,
        model_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> CompiledModel:
        """Load a model for inference.

        Args:
            model_path: Path to ONNX model
            device: Target device
            **kwargs: Backend-specific options

        Returns:
            Compiled model ready for inference
        """
        ...

    def supports_device(self, device: str) -> bool:
        """Check if this backend supports the given device."""
        # Exact match
        if device in self.supported_devices:
            return True

        # Check base type (e.g., "intel-gpu" matches "intel-gpu:0")
        device_type = device.split(":")[0] if ":" in device else device

        for supported in self.supported_devices:
            supported_type = supported.split(":")[0] if ":" in supported else supported
            if device_type == supported_type:
                return True

        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, devices={self.supported_devices})"
