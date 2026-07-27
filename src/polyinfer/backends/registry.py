"""Backend registry for managing available inference backends."""

from dataclasses import dataclass

from polyinfer._logging import get_logger
from polyinfer.backends.base import Backend
from polyinfer.exceptions import BackendNotAvailableError, BackendNotFoundError

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

    def unavailable_reason(self) -> str | None:
        """Why this backend is unavailable, or None if it is available."""
        try:
            return self.get_instance().unavailable_reason
        except Exception as e:  # pragma: no cover - defensive
            return f"could not determine reason: {e}"


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
        BackendNotFoundError: If no backend is registered under this name.
            Also a KeyError.
        BackendNotAvailableError: If the backend is registered but its
            dependencies are missing or broken. Also a RuntimeError.
    """
    if name not in _backends:
        available = list(_backends.keys())
        _logger.error(f"Backend '{name}' not found. Available: {available}")
        raise BackendNotFoundError(f"Backend '{name}' not found. Available: {available}")

    info = _backends[name]
    if not info.is_available():
        reason = info.unavailable_reason()
        _logger.error(f"Backend '{name}' is not available: {reason}")
        raise BackendNotAvailableError(
            f"Backend '{name}' is not available: {reason}\n"
            f"Install it with: pip install polyinfer[{name}]"
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
        raise BackendNotAvailableError(_no_backend_message(device))
    return backends[0]


def get_unavailable_backends() -> dict[str, str]:
    """Map each unavailable backend name to the reason it is unavailable.

    Distinguishes "not installed" from "installed but failed to import",
    which is the difference between a setup step and a broken environment.
    """
    reasons = {}
    for name, info in _backends.items():
        if not info.is_available():
            reasons[name] = info.unavailable_reason() or "unknown"
    return reasons


def _no_backend_message(device: str) -> str:
    """Build an actionable error for when no backend supports a device."""
    available = list_backends()
    lines = [f"No backend available for device '{device}'."]

    if available:
        lines.append(f"Available backends: {available}")
        lines.append(
            "These are installed but none of them supports this device. "
            "Check `polyinfer info` for the supported device list."
        )
    else:
        lines.append("No backends are available.")

    unavailable = get_unavailable_backends()
    if unavailable:
        lines.append("")
        lines.append("Unavailable backends:")
        lines.extend(f"  - {name}: {reason}" for name, reason in unavailable.items())

    if not available:
        lines.append("")
        lines.append("Install one with, for example: pip install polyinfer[cpu]")

    return "\n".join(lines)


def get_all_backends() -> dict[str, Backend]:
    """Get all registered backends (available or not).

    Returns:
        Dictionary mapping backend names to backend instances
    """
    return {name: info.get_instance() for name, info in _backends.items()}
