"""Test polyinfer with Intel devices (CPU, iGPU, NPU)."""

import os
import sys

sys.path.insert(0, "src")

import numpy as np

import polyinfer as pi
from polyinfer.backends.openvino import OpenVINOBackend

# Check what's available
print("=" * 60)
print("PolyInfer: Intel Device Test")
print("=" * 60)

print("\nAvailable backends:", pi.list_backends())
print("Available devices:", pi.list_devices())

# Get OpenVINO backend directly to see raw device names
ov_backend = OpenVINOBackend()
print("\nOpenVINO raw devices:", ov_backend.get_available_devices())

# Test model path, use YOLOv8n if available
model_path = None
for path in ["yolov8n.onnx", "examples/yolov8n.onnx", "../yolov8n.onnx"]:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    print("\nNo test model found. Downloading yolov8n.onnx...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.export(format="onnx")
        model_path = "yolov8n.onnx"
    except ImportError:
        print(
            "Please provide a model: pip install ultralytics && yolo export model=yolov8n.pt format=onnx"
        )
        sys.exit(1)

print(f"\nUsing model: {model_path}")

# Create test input (YOLOv8n expects 1x3x640x640)
input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Test each device
devices_to_test = [
    ("cpu", "CPU (Intel Core Ultra 9)"),
    ("intel-gpu", "Intel iGPU"),
    ("intel-gpu:0", "Intel iGPU (explicit)"),
    ("npu", "Intel NPU (AI Boost)"),
]

print("\n" + "=" * 60)
print("Running benchmarks...")
print("=" * 60)

results = []
for device, description in devices_to_test:
    try:
        print(f"\n[{device}] {description}")
        model = pi.load(model_path, backend="openvino", device=device)
        print(f"  Backend: {model.backend_name}")

        # Benchmark
        bench = model.benchmark(input_data, warmup=5, iterations=20)
        print(f"  Latency: {bench['mean_ms']:.2f} ms ({bench['fps']:.1f} FPS)")
        results.append((device, description, bench["mean_ms"], bench["fps"]))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append((device, description, None, None))

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"{'Device':<20} {'Description':<30} {'Latency':>10} {'FPS':>10}")
print("-" * 70)
for device, desc, latency, fps in results:
    if latency:
        print(f"{device:<20} {desc:<30} {latency:>8.2f}ms {fps:>9.1f}")
    else:
        print(f"{device:<20} {desc:<30} {'FAILED':>10} {'-':>10}")
