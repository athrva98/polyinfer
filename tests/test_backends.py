"""Tests for backend discovery and availability."""

import pytest

import polyinfer as pi
from polyinfer.backends.registry import get_all_backends, get_backend


class TestBackendDiscovery:
    """Test backend discovery functionality."""

    def test_list_backends_returns_list(self):
        """list_backends() should return a list."""
        backends = pi.list_backends()
        assert isinstance(backends, list)

    def test_list_backends_not_empty(self):
        """At least one backend should be available."""
        backends = pi.list_backends()
        assert len(backends) > 0, "No backends available"

    def test_list_devices_returns_list(self):
        """list_devices() should return a list."""
        devices = pi.list_devices()
        assert isinstance(devices, list)

    def test_cpu_always_available(self):
        """CPU device should always be available."""
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
