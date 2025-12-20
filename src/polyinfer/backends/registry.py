"""Backend registry for managing available inference backends."""

from dataclasses import dataclass

from polyinfer._logging import get_logger
from polyinfer.backends.base import Backend

_logger = get_logger("backends.registry")


@dataclass
class BackendInfo:
    """Information about a registered backend."""

    name: str
    backend_class: type[Backend]
    instance: Backend | None = None
    available: bool | None = None  # Lazily computed

    def get_instance(self) -> Backend:
        """Get or create backend instance."""
        if self.instance is None:
            self.instance = self.backend_class()
        return self.instance

    def is_available(self) -> bool:
        """Check if backend is available (cached)."""
        if self.available is None:
            self.available = self.get_instance().is_available()
        return self.available


# Global registry
_backends: dict[str, BackendInfo] = {}


def register_backend(name: str, backend_class: type[Backend]) -> None:
    """Register a backend class.

    Args:
        name: Backend name (e.g., 'onnxruntime', 'openvino')
        backend_class: Backend class to register
    """
    _backends[name] = BackendInfo(name=name, backend_class=backend_class)
    _logger.debug(f"Registered backend: {name}")


def get_backend(name: str) -> Backend:
    """Get a backend instance by name.

    Args:
        name: Backend name

    Returns:
        Backend instance

    Raises:
        KeyError: If backend not found
        RuntimeError: If backend not available
    """
    if name not in _backends:
        available = list(_backends.keys())
        _logger.error(f"Backend '{name}' not found. Available: {available}")
        raise KeyError(f"Backend '{name}' not found. Available: {available}")

    info = _backends[name]
    if not info.is_available():
        _logger.error(f"Backend '{name}' is not available")
        raise RuntimeError(
            f"Backend '{name}' is not available. Install it with: pip install polyinfer[{name}]"
        )

    _logger.debug(f"Retrieved backend: {name}")
    return info.get_instance()


def list_backends(available_only: bool = True) -> list[str]:
    """List registered backends.

    Args:
        available_only: If True, only return available backends

    Returns:
        List of backend names
    """
    if available_only:
        return [name for name, info in _backends.items() if info.is_available()]
    return list(_backends.keys())


def get_backends_for_device(device: str) -> list[Backend]:
    """Get all backends that support a given device.

    Args:
        device: Device string (e.g., 'cpu', 'cuda:0')

    Returns:
        List of backend instances, sorted by priority (highest first)
    """
    device_type = device.split(":")[0] if ":" in device else device

    matching = []
    for info in _backends.values():
        if not info.is_available():
            continue
        backend = info.get_instance()
        if backend.supports_device(device_type):
            matching.append(backend)

    # Sort by priority (higher = preferred)
    matching.sort(key=lambda b: b.priority, reverse=True)
    return matching


def get_best_backend(device: str) -> Backend:
    """Get the best available backend for a device.

    Args:
        device: Target device

    Returns:
        Best available backend

    Raises:
        RuntimeError: If no backend supports the device
    """
    backends = get_backends_for_device(device)
    if not backends:
        available = list_backends()
        raise RuntimeError(
            f"No backend available for device '{device}'. Available backends: {available}"
        )
    return backends[0]


def get_all_backends() -> dict[str, Backend]:
    """Get all registered backends (available or not).

    Returns:
        Dictionary mapping backend names to backend instances
    """
    return {name: info.get_instance() for name, info in _backends.items()}
