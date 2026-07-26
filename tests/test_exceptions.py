"""Tests for the unified exception hierarchy.

PolyInfer presents one API over four runtimes but used to leak each
runtime's own exception types. ONNX Runtime's InvalidArgument derives
directly from Exception - not RuntimeError or ValueError - so this, the
pattern the test suite itself used, silently caught nothing:

    try:
        model(bad_input)
    except (ValueError, RuntimeError):
        ...

Every PolyInfer error now derives from PolyInferError while keeping its
former builtin base, so old handlers still work.
"""

import numpy as np
import pytest

import polyinfer as pi
from polyinfer.exceptions import (
    BackendNotAvailableError,
    BackendNotFoundError,
    CompilationError,
    DeviceNotSupportedError,
    InferenceError,
    InvalidInputError,
    ModelLoadError,
    PolyInferError,
    QuantizationError,
)

ALL_ERRORS = [
    BackendNotFoundError,
    BackendNotAvailableError,
    DeviceNotSupportedError,
    ModelLoadError,
    CompilationError,
    InferenceError,
    InvalidInputError,
    QuantizationError,
]


class TestHierarchy:
    @pytest.mark.parametrize("exc", ALL_ERRORS)
    def test_everything_derives_from_polyinfer_error(self, exc):
        """One `except PolyInferError` must catch every PolyInfer failure."""
        assert issubclass(exc, PolyInferError)

    @pytest.mark.parametrize("exc", ALL_ERRORS)
    def test_all_exported_from_package_root(self, exc):
        assert getattr(pi, exc.__name__, None) is exc

    def test_compilation_error_is_a_load_error(self):
        """Compilation is part of loading, so it should be catchable as one."""
        assert issubclass(CompilationError, ModelLoadError)

    def test_invalid_input_is_an_inference_error(self):
        assert issubclass(InvalidInputError, InferenceError)

    def test_iree_compilation_error_joins_the_hierarchy(self):
        from polyinfer.backends.iree.backend import IREECompilationError

        assert issubclass(IREECompilationError, CompilationError)
        assert issubclass(IREECompilationError, PolyInferError)
        assert issubclass(IREECompilationError, RuntimeError)


class TestBackwardsCompatibility:
    """Pre-existing handlers must keep working."""

    @pytest.mark.parametrize(
        "exc,builtin",
        [
            (BackendNotFoundError, KeyError),
            (BackendNotAvailableError, RuntimeError),
            (DeviceNotSupportedError, ValueError),
            (ModelLoadError, RuntimeError),
            (CompilationError, RuntimeError),
            (InferenceError, RuntimeError),
            (InvalidInputError, RuntimeError),
            (InvalidInputError, ValueError),
            (QuantizationError, RuntimeError),
        ],
    )
    def test_retains_former_builtin_base(self, exc, builtin):
        assert issubclass(exc, builtin)

    def test_backend_not_found_message_is_not_repr_quoted(self):
        """KeyError.__str__ wraps messages in repr(); that is suppressed."""
        e = BackendNotFoundError("Backend 'x' not found. Available: []")
        assert str(e).startswith("Backend 'x' not found")
        assert not str(e).startswith('"')


class TestRegistryErrors:
    def test_unknown_backend_raises_typed_error(self):
        with pytest.raises(BackendNotFoundError):
            pi.get_backend("nonexistent_backend_xyz")

    def test_unknown_backend_still_catchable_as_keyerror(self):
        with pytest.raises(KeyError):
            pi.get_backend("nonexistent_backend_xyz")

    def test_unavailable_backend_raises_typed_error(self):
        errors = pi.backend_errors()
        if not errors:
            pytest.skip("all registered backends are available")
        name = next(iter(errors))
        with pytest.raises(BackendNotAvailableError):
            pi.get_backend(name)

    def test_no_backend_for_device_is_typed(self, model_path):
        with pytest.raises(PolyInferError):
            pi.load(model_path, device="nonexistent_device_xyz")


class TestTranslateErrors:
    """The backend error-translation helper."""

    def test_wraps_foreign_exception(self):
        from polyinfer.backends.base import translate_errors

        class ForeignError(Exception):
            """Mimics ONNX Runtime: derives straight from Exception."""

        with pytest.raises(InferenceError) as exc, translate_errors("fake-backend"):
            raise ForeignError("something went wrong")

        assert "fake-backend" in str(exc.value)
        assert "ForeignError" in str(exc.value)

    def test_preserves_original_as_cause(self):
        from polyinfer.backends.base import translate_errors

        original = ValueError("root cause")
        with pytest.raises(InferenceError) as exc, translate_errors("fake-backend"):
            raise original

        assert exc.value.__cause__ is original

    def test_passes_polyinfer_errors_through_unwrapped(self):
        """Already-typed errors must not be double-wrapped."""
        from polyinfer.backends.base import translate_errors

        original = InvalidInputError("bad input")
        with pytest.raises(InvalidInputError) as exc, translate_errors("fake-backend"):
            raise original

        assert exc.value is original

    @pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
    def test_never_swallows_control_flow_exceptions(self, exc_type):
        from polyinfer.backends.base import translate_errors

        with pytest.raises(exc_type), translate_errors("fake-backend"):
            raise exc_type()


class TestRealBackendErrors:
    """End-to-end: a genuine backend failure must be catchable portably."""

    def test_shape_mismatch_is_catchable_as_polyinfer_error(self, model_path):
        """The regression this hierarchy exists for.

        ONNX Runtime raises InvalidArgument, which is neither a RuntimeError
        nor a ValueError, so it escaped every handler in the codebase.
        """
        model = pi.load(model_path, device="cpu")
        bad = np.random.rand(1, 3, 99, 99).astype(np.float32)

        with pytest.raises(PolyInferError):
            model(bad)

    def test_shape_mismatch_preserves_backend_exception(self, model_path):
        model = pi.load(model_path, device="cpu")
        bad = np.random.rand(1, 3, 99, 99).astype(np.float32)

        with pytest.raises(InferenceError) as exc:
            model(bad)

        # The backend's own exception remains reachable for diagnosis.
        assert exc.value.__cause__ is not None

    def test_wrong_input_count_is_invalid_input_error(self, model_path):
        model = pi.load(model_path, device="cpu")
        arr = np.random.rand(1, 3, 640, 640).astype(np.float32)

        with pytest.raises(InvalidInputError):
            model(arr, arr)

    def test_missing_named_input_is_invalid_input_error(self, model_path):
        model = pi.load(model_path, device="cpu")

        with pytest.raises(InvalidInputError):
            model.run({"definitely_not_an_input": np.zeros((1,), dtype=np.float32)})

    def test_backend_device_mismatch_is_typed(self, model_path):
        if not pi.is_available("openvino"):
            pytest.skip("OpenVINO not installed")

        with pytest.raises(DeviceNotSupportedError):
            pi.load(model_path, backend="openvino", device="cuda")

    def test_missing_model_still_raises_filenotfound(self, tmp_path):
        """Standard errors that already fit stay unchanged."""
        with pytest.raises(FileNotFoundError):
            pi.load(tmp_path / "nope.onnx", device="cpu")
