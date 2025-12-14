"""Backend registry and management for PolyInfer."""

from polyinfer.backends.base import Backend, CompiledModel
from polyinfer.backends.registry import (
    register_backend,
    get_backend,
    list_backends,
    get_backends_for_device,
    BackendInfo,
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
