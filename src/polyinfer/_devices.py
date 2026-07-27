"""Canonical device-string normalization.

There is exactly one alias table for the whole library. Previously
``Model._normalize_device`` and ``InferenceConfig._normalize_device``
each had their own, and they disagreed with each other and with the
documented alias list in the README.

Canonical device names:
    cpu, cuda, tensorrt, directml, intel-gpu, npu, vulkan, rocm, coreml

Any of these may carry an index suffix (``cuda:1``, ``intel-gpu:0``).
"""

# Alias -> canonical base device name.
#
# Keep this in sync with the "Device Normalization" table in README.md;
# tests/test_devices.py::TestDeviceNormalization asserts they match.
DEVICE_ALIASES = {
    # NVIDIA
    "gpu": "cuda",
    "nvidia": "cuda",
    "trt": "tensorrt",
    # DirectML (Windows, any vendor)
    "dml": "directml",
    "directx": "directml",
    # Intel integrated GPU
    "igpu": "intel-gpu",
    "intel-igpu": "intel-gpu",
    "intel_igpu": "intel-gpu",
    "intel_gpu": "intel-gpu",
    # AMD
    "amd": "rocm",
    # Apple
    "metal": "coreml",
}


def normalize_device(device: str) -> str:
    """Normalize a device string to its canonical form.

    Lowercases, strips surrounding whitespace, and resolves aliases while
    preserving any device index suffix.

    Args:
        device: A device string, e.g. "GPU", " cuda:1 ", "igpu:0".

    Returns:
        The canonical device string.

    Example:
        >>> normalize_device("GPU")
        'cuda'
        >>> normalize_device("gpu:1")
        'cuda:1'
        >>> normalize_device("intel-igpu:0")
        'intel-gpu:0'
        >>> normalize_device("cpu")
        'cpu'
    """
    device = device.lower().strip()

    base, sep, index = device.partition(":")
    base = DEVICE_ALIASES.get(base, base)

    return f"{base}{sep}{index}" if sep else base


def device_type(device: str) -> str:
    """Return the base device type, dropping any index suffix.

    Example:
        >>> device_type("cuda:1")
        'cuda'
        >>> device_type("cpu")
        'cpu'
    """
    return device.partition(":")[0]


def device_index(device: str) -> int:
    """Return the device index, defaulting to 0 when unspecified.

    Example:
        >>> device_index("cuda:1")
        1
        >>> device_index("cuda")
        0
    """
    _, sep, index = device.partition(":")
    if not sep or not index:
        return 0
    try:
        return int(index)
    except ValueError as e:
        raise ValueError(f"Invalid device index in {device!r}: {index!r}") from e
