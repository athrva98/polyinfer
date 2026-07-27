"""Tests for the lazy backend wrappers used on Linux.

On Linux, _autoload registers proxy classes instead of the real backends, to
defer importing onnxruntime / iree.runtime until a model is actually loaded
(avoiding CUDA library conflicts with PyTorch).

Those proxies implemented only the abstract Backend surface plus load(), so
every other public method raised AttributeError. Since this path is
Linux-only and the backend job was continue-on-error, the gap went unnoticed:
the entire IREE MLIR API - emit_mlir, compile_mlir, load_vmfb - was
unreachable on Linux via pi.get_backend("iree"), including the load_vmfb()
call the README documents.

These construct the wrapper classes directly so they run on any platform.
"""

import pytest

from polyinfer.backends._autoload import (
    _make_lazy_iree_backend,
    _make_lazy_onnxruntime_backend,
)


def _public_api(cls) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


class TestLazyIREEBackend:
    @pytest.fixture
    def lazy(self):
        return _make_lazy_iree_backend()()

    def test_reports_identity_without_importing_iree(self, lazy):
        assert lazy.name == "iree"
        assert lazy.priority == 40
        assert "cpu" in lazy.supported_devices

    def test_proxies_entire_real_backend_surface(self, lazy):
        """Every public attribute of IREEBackend must be reachable."""
        from polyinfer.backends.iree.backend import IREEBackend

        missing = sorted(name for name in _public_api(IREEBackend) if not hasattr(lazy, name))
        assert not missing, f"lazy wrapper does not proxy: {missing}"

    @pytest.mark.parametrize(
        "method",
        ["emit_mlir", "compile_mlir", "load_vmfb", "list_vulkan_targets"],
    )
    def test_mlir_api_is_reachable(self, lazy, method):
        """Regression: these raised AttributeError on Linux."""
        assert callable(getattr(lazy, method))

    def test_emit_mlir_validates_path_rather_than_raising_attributeerror(self, lazy, tmp_path):
        """The exact CI failure: AttributeError instead of a real error."""
        try:
            lazy.emit_mlir("nonexistent_model.onnx", tmp_path / "out.mlir")
        except AttributeError as e:
            pytest.fail(f"lazy wrapper still missing emit_mlir: {e}")
        except Exception:
            pass  # FileNotFoundError or a backend-unavailable error is fine.

    def test_private_attributes_do_not_trigger_loading(self, lazy):
        """__getattr__ must not recurse via underscore lookups."""
        with pytest.raises(AttributeError):
            _ = lazy._not_a_real_private_attribute

    def test_unknown_public_attribute_still_raises(self, lazy):
        with pytest.raises(AttributeError):
            _ = lazy.definitely_not_a_method_on_the_backend


class TestLazyONNXRuntimeBackend:
    @pytest.fixture
    def lazy(self):
        return _make_lazy_onnxruntime_backend()()

    def test_reports_identity_without_importing_onnxruntime(self, lazy):
        assert lazy.name == "onnxruntime"
        assert lazy.priority == 60
        assert "cpu" in lazy.supported_devices

    def test_proxies_entire_real_backend_surface(self, lazy):
        from polyinfer.backends.onnxruntime.backend import ONNXRuntimeBackend

        missing = sorted(
            name for name in _public_api(ONNXRuntimeBackend) if not hasattr(lazy, name)
        )
        assert not missing, f"lazy wrapper does not proxy: {missing}"

    def test_get_available_providers_is_reachable(self, lazy):
        """Not part of the abstract surface, so it was missing entirely."""
        assert callable(lazy.get_available_providers)

    def test_private_attributes_do_not_trigger_loading(self, lazy):
        with pytest.raises(AttributeError):
            _ = lazy._not_a_real_private_attribute


class TestRegistrationIsSideEffectFree:
    """Building a wrapper class must not mutate the global registry."""

    def test_make_does_not_register(self):
        from polyinfer.backends.registry import _backends

        before = dict(_backends)
        _make_lazy_iree_backend()
        _make_lazy_onnxruntime_backend()
        assert dict(_backends) == before
