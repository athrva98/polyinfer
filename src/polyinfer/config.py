"""Configuration classes for PolyInfer."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class InferenceConfig:
    """Configuration for model inference.

    Attributes:
        device: Target device ("cpu", "cuda", "cuda:0", "directml", "vulkan")
        backend: Specific backend to use (None for auto-select)
        precision: Model precision ("fp32", "fp16", "int8")
        optimization_level: Backend-specific optimization (0-3)
        num_threads: Number of CPU threads (0 = auto)
        enable_profiling: Enable performance profiling
        cache_dir: Directory for cached compiled models
        extra_options: Backend-specific options
    """

    device: str = "cpu"
    backend: str | None = None
    precision: Literal["fp32", "fp16", "int8"] = "fp32"
    optimization_level: int = 2
    num_threads: int = 0  # 0 = auto
    enable_profiling: bool = False
    cache_dir: str | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize device names
        self.device = self._normalize_device(self.device)

    def _normalize_device(self, device: str) -> str:
        """Normalize device string to standard format."""
        device = device.lower().strip()

        # Handle aliases
        aliases = {
            "gpu": "cuda:0",
            "cuda": "cuda:0",
            "dml": "directml",
            "directx": "directml",
        }

        return aliases.get(device, device)

    @property
    def device_type(self) -> str:
        """Get the device type (cpu, cuda, directml, vulkan)."""
        if ":" in self.device:
            return self.device.split(":")[0]
        return self.device

    @property
    def device_id(self) -> int:
        """Get the device ID (0 for CPU, N for cuda:N)."""
        if ":" in self.device:
            return int(self.device.split(":")[1])
        return 0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking.

    Attributes:
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of benchmark runs
        include_data_transfer: Include CPU<->GPU transfer time
        percentiles: Percentiles to compute (e.g., [50, 90, 99])
    """

    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    include_data_transfer: bool = True
    percentiles: list[int] = field(default_factory=lambda: [50, 90, 99])
