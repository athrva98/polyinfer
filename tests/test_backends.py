"""Tests for backend discovery and availability."""

import pytest

import polyinfer as pi
from polyinfer.backends.registry import get_all_backends, get_backend

# Check if any backend is available
_BACKENDS = pi.list_backends()
_HAS_ANY_BACKEND = len(_BACKENDS) > 0


class TestBackendDiscovery:
    """Test backend discovery functionality."""

    def test_list_backends_returns_list(self):
        """list_backends() should return a list."""
        backends = pi.list_backends()
        assert isinstance(backends, list)

    @pytest.mark.skipif(not _HAS_ANY_BACKEND, reason="No backends installed")
    def test_list_backends_not_empty(self):
        """At least one backend should be available."""
        backends = pi.list_backends()
        assert len(backends) > 0, "No backends available"

    def test_list_devices_returns_list(self):
        """list_devices() should return a list."""
        devices = pi.list_devices()
        assert isinstance(devices, list)

    @pytest.mark.skipif(not _HAS_ANY_BACKEND, reason="No backends installed")
    def test_cpu_always_available(self):
        """CPU device should always be available when backends are installed."""
        devices = pi.list_devices()
        device_names = [d.name for d in devices]
        assert "cpu" in device_names, "CPU device not found"

    def test_get_backend_valid(self):
        """get_backend should return backend for valid name."""
        backends = pi.list_backends()
        if backends:
            backend = get_backend(backends[0])
            assert backend is not None
            assert backend.name == backends[0]

    def test_get_backend_invalid(self):
        """get_backend should raise for invalid backend."""
        with pytest.raises((ValueError, KeyError)):
            get_backend("nonexistent_backend_xyz")

    def test_is_available(self):
        """is_available should return bool for any backend name."""
        assert isinstance(pi.is_available("onnxruntime"), bool)
        assert isinstance(pi.is_available("openvino"), bool)
        assert isinstance(pi.is_available("fake_backend"), bool)
        assert pi.is_available("fake_backend") is False


class TestImportErrorClassification:
    """A missing package and a broken install must not look the same.

    Both raise ImportError, but "pip install X" only fixes one of them.
    """

    def test_missing_package_reported_as_not_installed(self):
        from polyinfer.backends.base import describe_import_error

        reason = describe_import_error(
            ModuleNotFoundError("No module named 'openvino'", name="openvino"),
            packages=("openvino",),
            install_hint="pip install openvino",
        )
        assert "not installed" in reason
        assert "pip install openvino" in reason
        assert "failed to import" not in reason

    def test_broken_install_reported_as_import_failure(self):
        """A native-library failure must not be reported as 'not installed'."""
        from polyinfer.backends.base import describe_import_error

        reason = describe_import_error(
            ImportError("DLL load failed while importing onnxruntime_pybind11_state"),
            packages=("onnxruntime",),
            install_hint="pip install onnxruntime",
        )
        assert "failed to import" in reason
        assert "DLL load failed" in reason
        assert "not installed" not in reason

    def test_missing_transitive_dependency_is_distinguished(self):
        """A missing dep of the backend isn't the backend being absent."""
        from polyinfer.backends.base import describe_import_error

        reason = describe_import_error(
            ModuleNotFoundError("No module named 'numpy'", name="numpy"),
            packages=("onnxruntime",),
            install_hint="pip install onnxruntime",
        )
        assert "dependency is missing" in reason


class TestBackendErrorReporting:
    """polyinfer must explain why a backend is unavailable."""

    def test_backend_errors_is_exported(self):
        assert hasattr(pi, "backend_errors")
        assert isinstance(pi.backend_errors(), dict)

    def test_every_unavailable_backend_has_a_reason(self):
        errors = pi.backend_errors()
        available = set(pi.list_backends())

        for name, reason in errors.items():
            assert name not in available, f"{name} is available but reported as failing"
            assert isinstance(reason, str) and reason.strip(), f"{name} has an empty reason"

    def test_available_backends_are_absent_from_errors(self):
        assert set(pi.list_backends()) & set(pi.backend_errors()) == set()

    def test_no_backend_error_message_lists_reasons(self):
        """The 'no backend available' error must name the failing backends.

        It previously read "Available backends: []" with no indication of
        whether anything was installed or why it failed.
        """
        from polyinfer.backends.registry import _no_backend_message

        message = _no_backend_message("cpu")
        assert "cpu" in message

        for name in pi.backend_errors():
            assert name in message, f"{name} missing from diagnostic message"

    def test_get_backend_error_includes_reason(self):
        """get_backend() on an unavailable backend must say why."""
        errors = pi.backend_errors()
        if not errors:
            pytest.skip("all registered backends are available")

        name, reason = next(iter(errors.items()))
        with pytest.raises(RuntimeError) as exc:
            get_backend(name)

        # The first line of the reason should appear in the raised error.
        assert reason.split("\n")[0][:40] in str(exc.value)


class TestONNXRuntimeBackend:
    """Tests specific to ONNX Runtime backend."""

    def test_onnxruntime_available(self):
        """ONNX Runtime should be available."""
        # This may fail if not installed, which is fine
        backends = pi.list_backends()
        if "onnxruntime" not in backends:
            pytest.skip("ONNX Runtime not installed")

        backend = get_backend("onnxruntime")
        assert backend.is_available()

    def test_onnxruntime_supports_cpu(self):
        """ONNX Runtime should support CPU."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        backend = get_backend("onnxruntime")
        assert backend.supports_device("cpu")

    def test_onnxruntime_version(self):
        """ONNX Runtime should report version."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        backend = get_backend("onnxruntime")
        assert backend.version != "not installed"
        assert backend.version != "unknown"


class TestOpenVINOBackend:
    """Tests specific to OpenVINO backend."""

    @pytest.mark.openvino
    def test_openvino_available(self):
        """OpenVINO should be available when marked."""
        backend = get_backend("openvino")
        assert backend.is_available()

    @pytest.mark.openvino
    def test_openvino_supports_cpu(self):
        """OpenVINO should support CPU."""
        backend = get_backend("openvino")
        assert backend.supports_device("cpu")

    @pytest.mark.openvino
    def test_openvino_raw_devices(self):
        """OpenVINO should report raw device names."""
        backend = get_backend("openvino")
        raw_devices = backend.get_available_devices()
        assert "CPU" in raw_devices


class TestIREEBackend:
    """Tests specific to IREE backend."""

    def test_iree_registration(self):
        """IREE should be registered if available."""
        all_backends = get_all_backends()
        # IREE may or may not be available
        if "iree" in all_backends:
            backend = all_backends["iree"]
            # Just check it doesn't crash
            _ = backend.is_available()


class TestBackendPriority:
    """Test backend priority and auto-selection."""

    def test_backends_have_priority(self):
        """All backends should have a priority value."""
        all_backends = get_all_backends()
        for _name, backend in all_backends.items():
            assert isinstance(backend.priority, int)
            assert backend.priority >= 0

    @pytest.mark.skipif(not _HAS_ANY_BACKEND, reason="No backends installed")
    def test_select_backend_for_cpu(self):
        """Auto-selection should work for CPU."""
        from polyinfer.discovery import select_backend

        backend = select_backend("cpu")
        assert backend is not None
        assert backend.supports_device("cpu")

    @pytest.mark.cuda
    def test_select_backend_for_cuda(self):
        """Auto-selection should work for CUDA."""
        from polyinfer.discovery import select_backend

        backend = select_backend("cuda")
        assert backend is not None
        assert backend.supports_device("cuda")
