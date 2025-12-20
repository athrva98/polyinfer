"""Backend registry and management for PolyInfer."""

from polyinfer.backends.base import Backend, CompiledModel
from polyinfer.backends.registry import (
    BackendInfo,
    get_backend,
    get_backends_for_device,
    list_backends,
    register_backend,
)

__all__ = [
    "Backend",
    "CompiledModel",
    "register_backend",
    "get_backend",
    "list_backends",
    "get_backends_for_device",
    "BackendInfo",
]

# Auto-register available backends on import
from polyinfer.backends import _autoload

_autoload.register_all()
