"""OpenVINO backend for PolyInfer.

Supports:
- CPU inference (highly optimized for Intel, works on AMD/ARM too)
- Intel integrated/discrete GPU
- Intel NPU (Neural Processing Unit)
"""

from polyinfer.backends.openvino.backend import OpenVINOBackend, OpenVINOModel

__all__ = ["OpenVINOBackend", "OpenVINOModel"]
