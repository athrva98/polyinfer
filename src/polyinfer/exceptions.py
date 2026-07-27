"""Exception hierarchy for PolyInfer.

PolyInfer presents one API over four inference runtimes, but errors used to
leak each runtime's own exception types straight through. ONNX Runtime
raises ``onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument``,
which derives directly from ``Exception`` - not from ``RuntimeError`` or
``ValueError`` - while the TensorRT and OpenVINO paths raised
``RuntimeError``. Portable error handling was therefore impossible:

    try:
        output = model(data)
    except (ValueError, RuntimeError):   # never catches ONNX Runtime
        ...

Every error PolyInfer raises now derives from :class:`PolyInferError`, so a
single ``except PolyInferError`` works across backends.

Backwards compatibility
-----------------------
Each class also inherits from the builtin it used to be, so existing code
keeps working:

    ``InferenceError``            is a ``RuntimeError``
    ``InvalidInputError``         is a ``RuntimeError`` and a ``ValueError``
    ``ModelLoadError``            is a ``RuntimeError``
    ``BackendNotAvailableError``  is a ``RuntimeError``
    ``DeviceNotSupportedError``   is a ``ValueError``
    ``InvalidOptionError``        is a ``ValueError``
    ``BackendNotFoundError``      is a ``KeyError``

The originating backend exception is always attached as ``__cause__``, so
backend-specific details remain available:

    try:
        model(data)
    except pi.InferenceError as e:
        print(e)            # unified message
        print(e.__cause__)  # the backend's own exception
"""


class PolyInferError(Exception):
    """Base class for every error raised by PolyInfer.

    Catch this to handle any PolyInfer failure regardless of backend.
    """


class BackendError(PolyInferError):
    """Base for problems with a backend itself, rather than with a model."""


class BackendNotFoundError(BackendError, KeyError):
    """No backend is registered under the requested name.

    Also a KeyError, which is what the registry raised previously.
    """

    def __str__(self) -> str:
        # KeyError.__str__ wraps the message in repr(), turning a sentence
        # into "'a sentence'". Bypass that.
        return self.args[0] if self.args else ""


class BackendNotAvailableError(BackendError, RuntimeError):
    """A backend is registered but unusable.

    Either its package is not installed or it failed to import. See
    :func:`polyinfer.backend_errors` for the specific reason.
    """


class DeviceNotSupportedError(PolyInferError, ValueError):
    """The requested device is not supported by the selected backend."""


class InvalidOptionError(PolyInferError, ValueError):
    """A backend option was given an invalid value.

    This is a caller mistake - a bad `optimization_level`, an unrecognized
    `performance_hint` - not a failure to load the model, so it is a
    ValueError and is deliberately *not* wrapped in ModelLoadError when
    raised from a backend's load().
    """


class ModelLoadError(PolyInferError, RuntimeError):
    """A model could not be loaded, compiled, or prepared for inference."""


class CompilationError(ModelLoadError):
    """Ahead-of-time compilation failed (IREE, TensorRT engine build)."""


class InferenceError(PolyInferError, RuntimeError):
    """Inference failed."""


class InvalidInputError(InferenceError, ValueError):
    """Inputs were rejected: wrong count, names, shapes, or dtypes.

    Both an InferenceError and a ValueError, so it is caught by handlers
    written against either.
    """


class QuantizationError(PolyInferError, RuntimeError):
    """Quantization failed or was requested in an unsupported configuration."""


__all__ = [
    "PolyInferError",
    "BackendError",
    "BackendNotFoundError",
    "BackendNotAvailableError",
    "DeviceNotSupportedError",
    "InvalidOptionError",
    "ModelLoadError",
    "CompilationError",
    "InferenceError",
    "InvalidInputError",
    "QuantizationError",
]
