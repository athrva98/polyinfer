"""Configuration classes for PolyInfer."""

from dataclasses import dataclass, field
from typing import Any, Literal

from polyinfer._devices import device_index, device_type, normalize_device


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
        # Normalize device names using the canonical alias table.
        self.device = normalize_device(self.device)

    @property
    def device_type(self) -> str:
        """Get the device type (cpu, cuda, directml, vulkan)."""
        return device_type(self.device)

    @property
    def device_id(self) -> int:
        """Get the device ID (0 for CPU, N for cuda:N)."""
        return device_index(self.device)


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
