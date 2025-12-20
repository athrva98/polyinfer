"""Device and backend discovery for PolyInfer."""

from dataclasses import dataclass

from polyinfer.backends.base import Backend
from polyinfer.backends.registry import (
    get_backend as _get_backend,
)
from polyinfer.backends.registry import (
    get_backends_for_device,
    get_best_backend,
)
from polyinfer.backends.registry import (
    list_backends as _list_backends,
)


@dataclass
class DeviceInfo:
    """Information about an available device."""

    name: str
    device_type: str
    backends: list[str]
    description: str = ""

    def __str__(self) -> str:
        backends_str = ", ".join(self.backends)
        return f"{self.name} ({self.device_type}) - backends: [{backends_str}]"


def list_backends(available_only: bool = True) -> list[str]:
    """List available backends.

    Args:
        available_only: If True, only return backends with installed dependencies

    Returns:
        List of backend names

    Example:
        >>> import polyinfer as pi
        >>> pi.list_backends()
        ['onnxruntime', 'openvino']
    """
    return _list_backends(available_only=available_only)


def get_backend(name: str) -> Backend:
    """Get a backend by name.

    Args:
        name: Backend name (e.g., 'onnxruntime', 'openvino')

    Returns:
        Backend instance

    Example:
        >>> backend = pi.get_backend('openvino')
        >>> model = backend.load('model.onnx', device='cpu')
    """
    return _get_backend(name)


def is_available(backend_name: str) -> bool:
    """Check if a backend is available.

    Args:
        backend_name: Name of the backend

    Returns:
        True if backend is installed and available
    """
    try:
        backend = _get_backend(backend_name)
        return backend.is_available()
    except (KeyError, RuntimeError):
        return False


def list_devices() -> list[DeviceInfo]:
    """List all available devices and their supported backends.

    Returns:
        List of DeviceInfo objects

    Example:
        >>> for device in pi.list_devices():
        ...     print(device)
        cpu (cpu) - backends: [onnxruntime, openvino]
        cuda:0 (cuda) - backends: [onnxruntime]
    """
    devices: dict[str, dict[str, list[str] | str]] = {}

    # Collect devices from all backends
    for backend_name in _list_backends(available_only=True):
        try:
            backend = _get_backend(backend_name)
            for device in backend.supported_devices:
                if device not in devices:
                    devices[device] = {
                        "backends": [],
                        "type": device.split(":")[0] if ":" in device else device,
                    }
                backends_list = devices[device]["backends"]
                if isinstance(backends_list, list):
                    backends_list.append(backend_name)
        except Exception:
            continue

    # Build DeviceInfo list
    result: list[DeviceInfo] = []
    for name, info in sorted(devices.items()):
        device_type = info["type"]
        backends = info["backends"]
        result.append(
            DeviceInfo(
                name=name,
                device_type=str(device_type),
                backends=list(backends) if isinstance(backends, list) else [backends],
            )
        )

    return result


def get_device_backends(device: str) -> list[str]:
    """Get backends that support a specific device.

    Args:
        device: Device name (e.g., 'cpu', 'cuda:0')

    Returns:
        List of backend names, sorted by priority
    """
    backends = get_backends_for_device(device)
    return [b.name for b in backends]


def select_backend(device: str, prefer: str | None = None) -> Backend:
    """Select the best backend for a device.

    Args:
        device: Target device
        prefer: Preferred backend (used if available)

    Returns:
        Selected backend instance

    Example:
        >>> backend = pi.select_backend('cuda', prefer='tensorrt')
    """
    if prefer:
        try:
            backend = _get_backend(prefer)
            if backend.supports_device(device):
                return backend
        except (KeyError, RuntimeError):
            pass  # Fall through to auto-selection

    return get_best_backend(device)


def system_info() -> dict:
    """Get detailed system and backend information.

    Returns:
        Dictionary with system info, available backends, and devices
    """
    import platform
    import sys

    info = {
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": sys.version,
            "architecture": platform.machine(),
        },
        "backends": {},
        "devices": [],
    }

    # Backend info
    backends_dict = info["backends"]
    for name in _list_backends(available_only=False):
        try:
            backend = _get_backend(name)
            if isinstance(backends_dict, dict):
                backends_dict[name] = {
                    "available": backend.is_available(),
                    "version": backend.version,
                    "devices": backend.supported_devices,
                    "priority": backend.priority,
                }
        except Exception as e:
            if isinstance(backends_dict, dict):
                backends_dict[name] = {
                    "available": False,
                    "error": str(e),
                }

    # Device info
    devices_list = info["devices"]
    for device in list_devices():
        if isinstance(devices_list, list):
            devices_list.append(
                {
                    "name": device.name,
                    "type": device.device_type,
                    "backends": device.backends,
                }
            )

    return info
