```
██████╗  ██████╗ ██╗  ██╗   ██╗██╗███╗   ██╗███████╗███████╗██████╗
██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝██║████╗  ██║██╔════╝██╔════╝██╔══██╗
██████╔╝██║   ██║██║   ╚████╔╝ ██║██╔██╗ ██║█████╗  █████╗  ██████╔╝
██╔═══╝ ██║   ██║██║    ╚██╔╝  ██║██║╚██╗██║██╔══╝  ██╔══╝  ██╔══██╗
██║     ╚██████╔╝███████╗██║   ██║██║ ╚████║██║     ███████╗██║  ██║
╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝  ╚═══╝╚═╝     ╚══════╝╚═╝  ╚═╝
```

# PolyInfer

Unified ML inference across multiple backends.

### Installation

**From PyPI** (coming soon):
```bash
pip install polyinfer[nvidia]   # NVIDIA GPU (CUDA + cuDNN via onnxruntime-gpu)
pip install polyinfer[intel]    # Intel CPU/GPU/NPU
pip install polyinfer[amd]      # AMD GPU (Windows DirectML)
pip install polyinfer[cpu]      # CPU only
pip install polyinfer[all]      # Everything
pip install polyinfer[examples] # Dependencies for running examples (torch, PIL, etc.)
```

**Native TensorRT** (optional, for maximum performance):
```bash
# Install AFTER polyinfer[nvidia], then reinstall torch
pip install tensorrt-cu12 cuda-python
pip install torch torchvision --force-reinstall
```

**From source** (current):
```bash
git clone https://github.com/athrva98/polyinfer.git
cd polyinfer
pip install -e ".[nvidia]"      # Or any of the extras above
```

**No manual CUDA/cuDNN installation required.** Dependencies are automatically downloaded and configured. Works on Windows, Linux, WSL2, and Google Colab.

## Quick Start

```python
import polyinfer as pi

# List available backends and devices
print(pi.list_backends())  # ['onnxruntime', 'openvino']
print(pi.list_devices())   # [cpu, cuda, tensorrt, ...]

# Load model - auto-selects fastest backend
model = pi.load("model.onnx", device="cpu")        # Uses OpenVINO (fastest for CPU)
model = pi.load("model.onnx", device="cuda")       # Uses CUDA
model = pi.load("model.onnx", device="tensorrt")   # Uses TensorRT (450+ FPS on YoloV8n RTX5060!)

# Run inference
import numpy as np
output = model(np.random.rand(1, 3, 224, 224).astype(np.float32))

# Benchmark
results = model.benchmark(input_data, warmup=10, iterations=100)
print(f"{results['mean_ms']:.2f} ms ({results['fps']:.1f} FPS)")
```

## Device Options

```python
# CPU
model = pi.load("model.onnx", device="cpu")

# NVIDIA GPU
model = pi.load("model.onnx", device="cuda")       # CUDA
model = pi.load("model.onnx", device="cuda:0")     # Specific GPU
model = pi.load("model.onnx", device="tensorrt")   # TensorRT (generally the fastest for Nvidia)

# AMD/Intel/Any GPU on Windows
model = pi.load("model.onnx", device="directml")

# Vulkan (cross-platform GPU)
model = pi.load("model.onnx", device="vulkan")
```

## Backend Selection

```python
# Auto-select (recommended)
model = pi.load("model.onnx", device="cuda")

# Explicit backend
model = pi.load("model.onnx", backend="onnxruntime", device="cuda")
model = pi.load("model.onnx", backend="openvino", device="cpu")

# TensorRT options:
model = pi.load("model.onnx", device="tensorrt")              # ONNX Runtime TensorRT EP (recommended)
model = pi.load("model.onnx", backend="tensorrt", device="cuda")  # Native TensorRT (requires separate install)
```

## Compare Backends

```python
# Compare all available backends
pi.compare("model.onnx", input_shape=(1, 3, 640, 640))

# For YOLO V8n
# Output:
# Backend Comparison for model.onnx
# ============================================================
# onnxruntime-tensorrt    :   2.22 ms (450.0 FPS) <-- FASTEST
# onnxruntime-cuda        :   6.64 ms (150.7 FPS)
# openvino-cpu            :  16.19 ms ( 61.8 FPS)
# onnxruntime-cpu         :  22.56 ms ( 44.3 FPS)
```

## CLI

```bash
# Show system info and available backends
polyinfer info

# Benchmark a model
polyinfer benchmark model.onnx --device tensorrt

# Run inference
polyinfer run model.onnx --device cuda
```

## Quantization

Reduce model size and improve inference speed with INT8/FP16 quantization:

```python
import polyinfer as pi

# Dynamic quantization (no calibration data needed)
pi.quantize("model.onnx", "model_int8.onnx", method="dynamic")

# Static quantization with calibration data
calibration_data = [np.random.rand(1, 3, 224, 224).astype(np.float32) for _ in range(100)]
pi.quantize("model.onnx", "model_int8.onnx",
            method="static",
            calibration_data=calibration_data)

# FP16 conversion
pi.convert_to_fp16("model.onnx", "model_fp16.onnx")

# Load and run quantized model
model = pi.load("model_int8.onnx", device="cpu")
output = model(input_data)
```

**Supported quantization:**
- **ONNX Runtime**: Dynamic/Static INT8, UINT8, INT4, FP16
- **OpenVINO (NNCF)**: Static INT8 with calibration
- **TensorRT**: FP16/INT8 (via `pi.load(..., fp16=True, int8=True)`)

## Performance

### YOLOv8n @ 640x640 (RTX 5060)

| Backend | Latency | FPS | Speedup |
|---------|---------|-----|---------|
| TensorRT | 2.2 ms | **450** | 10x |
| CUDA | 6.6 ms | 151 | 3.4x |
| OpenVINO (CPU) | 16.2 ms | 62 | 1.4x |
| ONNX Runtime (CPU) | 22.6 ms | 44 | 1.0x |

### ResNet18 @ 224x224 (Google Colab T4)

| Backend | Latency | FPS | Speedup |
|---------|---------|-----|---------|
| TensorRT | 1.6 ms | **639** | 2.6x |
| CUDA | 4.1 ms | 245 | 1.0x |
| ONNX Runtime (CPU) | 43.7 ms | 23 | 0.09x |

## Supported Backends

| Backend | Devices | Install |
|---------|---------|---------|
| **ONNX Runtime** | CPU, CUDA, TensorRT, DirectML | `[cpu]`, `[nvidia]`, `[amd]` |
| **OpenVINO** | CPU, Intel GPU, NPU | `[cpu]`, `[intel]` |
| **IREE** | CPU, Vulkan, CUDA | `[all]` |

## MLIR Export (Custom Hardware)

Export models to MLIR for custom hardware targets, kernel injection, or advanced optimizations:

```python
import polyinfer as pi

# Export ONNX to MLIR
mlir = pi.export_mlir("model.onnx", "model.mlir")

# Compile for specific target
vmfb = pi.compile_mlir("model.mlir", device="vulkan")

# Load and run
backend = pi.get_backend("iree")
model = backend.load_vmfb(vmfb, device="vulkan")
```

## Why PolyInfer?

- **Zero configuration**: `pip install polyinfer[nvidia]` - CUDA, cuDNN, TensorRT all auto-installed
- **Auto backend selection**: Picks the fastest backend for your hardware
- **Unified API**: Same code works across all backends
- **Real performance**: 450 FPS with TensorRT, no manual optimization needed
- **MLIR support**: Export to MLIR for custom hardware and kernel development

---

# Development

## Installation Options

| Extra | What's Included | Use Case |
|-------|-----------------|----------|
| `[nvidia]` | ONNX Runtime GPU, IREE, torch | NVIDIA GPUs |
| `[intel]` | OpenVINO, IREE, torch | Intel CPU, iGPU, NPU |
| `[amd]` | ONNX Runtime DirectML, IREE, torch | AMD GPUs on Windows |
| `[cpu]` | ONNX Runtime, OpenVINO, IREE, torch | CPU-only systems |
| `[vulkan]` | IREE, torch | Cross-platform GPU via Vulkan |
| `[all]` | Everything above | Maximum compatibility |
| `[tensorrt]` | tensorrt-cu12, cuda-python | Native TensorRT (install separately) |
| `[examples]` | PIL, opencv, transformers, diffusers, segment-anything | Running example scripts |

**Note:** Native TensorRT is provided as a separate `[tensorrt]` extra because `tensorrt-cu12-libs` depends on `cuda-toolkit` which overwrites CUDA libraries and breaks PyTorch. Install it separately after `[nvidia]`, then reinstall torch:
```bash
pip install polyinfer[nvidia]
pip install tensorrt-cu12 cuda-python  # Or: pip install polyinfer[tensorrt]
pip install torch torchvision --force-reinstall  # Fix torch after TensorRT install
```

### Development Install

```bash
# Clone the repository
git clone https://github.com/athrva98/polyinfer.git
cd polyinfer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e ".[nvidia,dev]"

# Run tests
pytest tests/
```

### Platform-Specific Setup

#### Windows

```powershell
# Create conda environment (recommended)
conda create -n polyinfer python=3.11
conda activate polyinfer

# Clone and install with NVIDIA support
git clone https://github.com/athrva98/polyinfer.git
cd polyinfer
pip install -e ".[nvidia]"

# Or for AMD GPU
pip install -e ".[amd]"

# Verify installation
python -c "import polyinfer as pi; print(pi.list_devices())"
```

#### Linux / WSL2

```bash
# Create virtual environment
python3 -m venv ~/polyinfer_venv
source ~/polyinfer_venv/bin/activate

# Clone and install with NVIDIA support
git clone https://github.com/athrva98/polyinfer.git
cd polyinfer
pip install -e ".[nvidia]"

# Verify CUDA works
python -c "import polyinfer as pi; print(pi.list_devices())"
```

**WSL2 GPU Passthrough Requirements:**
- Windows 11 or Windows 10 21H2+
- WSL2 with Ubuntu (or other distro)
- NVIDIA GPU driver installed on Windows (not in WSL)
- No need to install CUDA in WSL, polyinfer handles it automatically

#### Google Colab

```python
# Install polyinfer with NVIDIA support
!pip install -q "polyinfer[nvidia] @ git+https://github.com/athrva98/polyinfer.git"

# Verify installation
import polyinfer as pi
print(pi.list_devices())
# Output: [cpu, cuda, tensorrt, vulkan]

# TensorRT works out of the box on Colab!
model = pi.load("model.onnx", device="tensorrt")  # 638 FPS on ResNet18!
```

**Note:** TensorRT EP is automatically configured on Colab. The `tensorrt-cu12-libs` package provides TensorRT libraries, and PolyInfer automatically preloads them via ctypes before ONNX Runtime is imported.

#### macOS

```bash
# Clone repository
git clone https://github.com/athrva98/polyinfer.git
cd polyinfer

# Install CPU-only (no GPU acceleration on macOS yet)
pip install -e ".[cpu]"

# IREE provides some Metal support (experimental)
pip install -e ".[vulkan]"
```

---

## Architecture

### Backend Hierarchy

```
polyinfer/
├── __init__.py          # Public API: load, list_backends, export_mlir, etc.
├── model.py             # Model class with unified inference interface
├── discovery.py         # Backend/device discovery
├── nvidia_setup.py      # Auto-configures NVIDIA libraries (CUDA, cuDNN, TensorRT)
├── mlir.py              # MLIR export/compile functions
├── compare.py           # Cross-backend comparison utilities
├── config.py            # Configuration classes
└── backends/
    ├── base.py          # Abstract Backend and CompiledModel classes
    ├── registry.py      # Backend registration system
    ├── _autoload.py     # Auto-discovers and registers backends
    ├── onnxruntime/     # ONNX Runtime backend (CPU, CUDA, TensorRT, DirectML)
    ├── openvino/        # OpenVINO backend (Intel CPU, GPU, NPU)
    ├── tensorrt/        # Native TensorRT backend
    └── iree/            # IREE backend (CPU, Vulkan, CUDA) + MLIR emission
```

### Backend Priority System

When you call `pi.load("model.onnx", device="cuda")`, polyinfer selects the best backend:

| Backend | Priority | Devices |
|---------|----------|---------|
| OpenVINO | 70 | cpu, intel-gpu, npu |
| ONNX Runtime | 60 | cpu, cuda, tensorrt, directml, rocm, coreml |
| TensorRT (native) | 50 | cuda, tensorrt |
| IREE | 40 | cpu, vulkan, cuda |

**Note:** ONNX Runtime's TensorRT Execution Provider is preferred over native TensorRT because it works out-of-the-box without dependency conflicts. For `device="tensorrt"`, ONNX Runtime's TensorRT EP is used by default. To use native TensorRT, specify `backend="tensorrt"` explicitly.

### Device Normalization

These aliases are automatically normalized:

| Input | Normalized To |
|-------|---------------|
| `gpu`, `nvidia` | `cuda` |
| `trt` | `tensorrt` |
| `dml` | `directml` |
| `igpu`, `intel-igpu` | `intel-gpu` |

---

## Backend Options Reference

All backends support passing options through `pi.load()`. Options are passed as keyword arguments.

### Native TensorRT Backend

For maximum NVIDIA performance. Supports full TensorRT configuration.

**Requires separate installation:**
```bash
pip install tensorrt-cu12 cuda-python
pip install torch torchvision --force-reinstall  # Fix torch after TensorRT install
```

```python
model = pi.load("model.onnx", backend="tensorrt", device="cuda",
    # Precision
    fp16=True,                      # FP16 (half precision)
    int8=False,                     # INT8 quantization
    tf32=True,                      # TF32 on Ampere+ (default)
    bf16=False,                     # BF16 on Ada+
    fp8=False,                      # FP8 on Hopper+
    strict_types=False,             # Force specified precision

    # Optimization
    builder_optimization_level=5,   # 0-5, higher = better perf, slower build
    workspace_size=4 << 30,         # 4GB workspace
    avg_timing_iterations=4,        # More iterations = better kernel selection
    sparsity=False,                 # Structured sparsity (Ampere+)

    # Caching
    cache_path="./model.engine",    # Engine cache path
    timing_cache_path="./timing.cache",  # Timing cache for faster rebuilds
    force_rebuild=False,            # Ignore cache, rebuild engine

    # Hardware
    dla_core=-1,                    # DLA core (-1 = GPU only)
    gpu_fallback=True,              # GPU fallback for unsupported DLA ops

    # Profiling
    profiling_verbosity="detailed", # 'none', 'layer_names_only', 'detailed'
    engine_capability="default",    # 'default', 'safe', 'dla_standalone'

    # Dynamic shapes (for models with dynamic batch/resolution)
    min_shapes={"input": (1, 3, 224, 224)},
    opt_shapes={"input": (4, 3, 640, 640)},
    max_shapes={"input": (16, 3, 1024, 1024)},
)
```

### ONNX Runtime Backend

Versatile backend with multiple execution providers.

#### CUDA Execution Provider

```python
model = pi.load("model.onnx", device="cuda",
    # Session options
    graph_optimization_level=3,     # 0=off, 1=basic, 2=extended, 3=all
    intra_op_num_threads=4,         # Threads for parallelism
    inter_op_num_threads=2,
    enable_mem_pattern=True,
    enable_cpu_mem_arena=True,

    # CUDA-specific
    cuda_mem_limit=4 << 30,         # 4GB GPU memory limit
    arena_extend_strategy="kNextPowerOfTwo",
    cudnn_conv_algo_search="EXHAUSTIVE",  # or 'HEURISTIC', 'DEFAULT'
    do_copy_in_default_stream=True,
)
```

#### TensorRT Execution Provider (via ONNX Runtime)

```python
model = pi.load("model.onnx", device="tensorrt",
    # Precision
    fp16=True,
    int8=False,

    # Optimization
    builder_optimization_level=5,   # 0-5
    max_workspace_size=4 << 30,     # 4GB
    timing_cache_path="./timing.cache",

    # Caching
    cache_dir="./trt_cache",        # Engine cache directory

    # Subgraph control
    min_subgraph_size=5,            # Min nodes for TRT subgraph
    max_partition_iterations=1000,

    # DLA (Jetson)
    dla_enable=False,
    dla_core=0,

    # Build options
    force_sequential_engine_build=False,
)
```

#### DirectML Execution Provider (Windows AMD/Intel GPU)

```python
model = pi.load("model.onnx", device="directml",
    device_id=0,                    # GPU index
)
```

### OpenVINO Backend

Optimized for Intel hardware.

```python
model = pi.load("model.onnx", backend="openvino", device="cpu",
    optimization_level=2,           # 0=throughput, 1=balanced, 2=latency
    num_threads=8,                  # CPU threads
    enable_caching=True,
    cache_dir="./ov_cache",
)
```

### IREE Backend

Cross-platform with MLIR export capability.

```python
model = pi.load("model.onnx", backend="iree", device="vulkan",
    opt_level=3,                    # 0-3
    cache_dir="./iree_cache",
    force_compile=False,
    save_mlir=True,                 # Save intermediate MLIR
    mlir_path="./model.mlir",
)

# MLIR export for custom hardware
mlir = pi.export_mlir("model.onnx", "model.mlir", load_content=True)
vmfb = pi.compile_mlir("model.mlir", device="vulkan", opt_level=3)
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_yolov8.py

# Run specific test class
pytest tests/test_yolov8.py::TestYOLOv8IREE

# Run tests for a specific device
pytest tests/ -m cuda
pytest tests/ -m vulkan
pytest tests/ -m npu

# Run benchmarks
pytest tests/test_benchmark.py -v
```

### Test Markers

Tests are tagged with pytest markers for selective execution:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.cuda` | Requires CUDA GPU |
| `@pytest.mark.tensorrt` | Requires TensorRT |
| `@pytest.mark.vulkan` | Requires Vulkan GPU |
| `@pytest.mark.directml` | Requires DirectML (Windows) |
| `@pytest.mark.openvino` | Requires OpenVINO |
| `@pytest.mark.iree` | Requires IREE |
| `@pytest.mark.npu` | Requires Intel NPU |
| `@pytest.mark.intel_gpu` | Requires Intel integrated GPU |
| `@pytest.mark.benchmark` | Performance benchmark tests |

### Test Files

| File | Purpose |
|------|---------|
| `test_backends.py` | Backend discovery and registration |
| `test_backend_options.py` | Backend options passthrough |
| `test_devices.py` | Device-specific loading and inference |
| `test_inference.py` | Cross-backend consistency |
| `test_benchmark.py` | Performance benchmarks |
| `test_yolov8.py` | End-to-end YOLOv8 tests on all devices |
| `test_mlir.py` | MLIR export and compilation |
| `test_intel_devices.py` | Intel GPU and NPU device tests |

---

## MLIR & Custom Hardware

PolyInfer can emit MLIR for custom hardware targets, kernel injection, and advanced optimization workflows.

### Basic MLIR Workflow

```python
import polyinfer as pi

# 1. Export ONNX to MLIR
mlir = pi.export_mlir("model.onnx", "model.mlir", load_content=True)
print(mlir.content[:500])  # Inspect MLIR

# 2. (Optional) Modify MLIR for custom kernels
# ... your custom transformations ...

# 3. Compile MLIR to executable
vmfb = pi.compile_mlir("model.mlir", device="vulkan")

# 4. Load and run
backend = pi.get_backend("iree")
model = backend.load_vmfb(vmfb, device="vulkan")
output = model(input_data)
```

### MLIROutput Class

```python
@dataclass
class MLIROutput:
    path: Path              # Path to saved MLIR file
    content: str | None     # MLIR content (if load_content=True)
    source_model: Path      # Original ONNX model path
    dialect: str            # MLIR dialect (e.g., "iree")

    def save(self, output_path) -> Path:
        """Save MLIR to a new location."""

    def __str__(self) -> str:
        """Returns MLIR content."""
```

### Compilation Targets

| Device | IREE Target | Use Case |
|--------|-------------|----------|
| `cpu` | `llvm-cpu` | CPU with LLVM optimizations |
| `vulkan` | `vulkan-spirv` | Cross-platform GPU |
| `cuda` | `cuda` | NVIDIA GPU via CUDA |

---

## Troubleshooting

### Common Issues

#### "Backend 'onnxruntime' does not support device 'cuda'"

**Cause:** `onnxruntime-gpu` is not installed, or conflicting ONNX Runtime packages.

PolyInfer will automatically detect this conflict and show a warning on import:
```
⚠️  ONNX Runtime Conflict Detected!
   Both 'onnxruntime-gpu' and 'onnxruntime-directml' are installed,
   but only DirectML is active. CUDA support is disabled.
```

**Solution 1: Use the built-in fix helper**
```python
import polyinfer as pi
pi.fix_onnxruntime_conflict(prefer="cuda")  # or prefer="directml"
# Restart Python after running this
```

**Solution 2: Manual fix**
```bash
# Uninstall all onnxruntime variants
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml -y

# Install the one you need
pip install onnxruntime-gpu  # For CUDA
# or
pip install onnxruntime-directml  # For DirectML (AMD on Windows)
```

**Important:** On Windows, you can only have ONE onnxruntime variant installed at a time.
The packages `onnxruntime`, `onnxruntime-gpu`, and `onnxruntime-directml` share the same
module namespace and will overwrite each other.

#### "libcudnn.so.9: cannot open shared object file"

**Cause:** cuDNN libraries not found by ONNX Runtime.

**Solution:** This should be automatic with `polyinfer[nvidia]`. If not:
```python
# Check if libraries are detected
from polyinfer.nvidia_setup import get_nvidia_info
print(get_nvidia_info())
```

If libraries are found but still failing, the ctypes preload may not work for your setup. Set `LD_LIBRARY_PATH` manually:
```bash
export LD_LIBRARY_PATH=$(python -c "from polyinfer.nvidia_setup import get_nvidia_info; print(':'.join(get_nvidia_info()['library_directories']))")
python your_script.py
```

#### PyTorch breaks after installing TensorRT ("undefined symbol: ncclCommWindowRegister")

**Cause:** `tensorrt-cu12-libs` depends on `cuda-toolkit`, which overwrites CUDA libraries (nvidia-cuda-runtime, nvidia-nccl, etc.) with versions incompatible with PyTorch.

**Solution:** Reinstall PyTorch after installing TensorRT:
```bash
pip install torch torchvision --force-reinstall
```

**Prevention:** Use ONNX Runtime's TensorRT Execution Provider instead (works with `device="tensorrt"` by default). It provides similar performance without dependency conflicts. Only install native TensorRT if you need advanced TensorRT features.

#### TensorRT EP: "RegisterTensorRTPluginsAsCustomOps" error

**Cause:** ONNX Runtime can't find TensorRT libraries, even though `TensorrtExecutionProvider` shows as available.

**Solution 1: Check if TensorRT libraries are installed**
```bash
pip install tensorrt-cu12-libs  # Lightweight TensorRT libs (no cuda-python conflict)
```

**Solution 2: Check library detection**
```python
import polyinfer as pi
info = pi.get_nvidia_info()
print("TensorRT dirs:", info['tensorrt_setup']['tensorrt_dirs'])
print("Preloaded libs:", info['tensorrt_setup']['preloaded_libs'])
```

If `tensorrt_dirs` is empty, PolyInfer couldn't find the TensorRT libraries. This usually means:
- `tensorrt-cu12-libs` is not installed
- The libraries are in an unexpected location

**Solution 3: Manual preload (advanced)**
```python
import ctypes
from pathlib import Path
import sys

# Find tensorrt_libs directory
site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "dist-packages"
tensorrt_libs = site_packages / "tensorrt_libs"

# Preload before importing onnxruntime
for lib in ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"]:
    lib_path = tensorrt_libs / lib
    if lib_path.exists():
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

# Now import polyinfer
import polyinfer as pi
```

#### "iree-import-onnx not found"

**Cause:** IREE compiler tools not installed.

**Solution:**
```bash
pip install iree-base-compiler iree-base-runtime
```

#### Vulkan tests produce NaN values

**Known Issue:** IREE's Vulkan backend can produce sporadic NaN values (~0.01% of outputs) on some drivers.

**Workaround:** Tests are configured to tolerate up to 0.1% NaN values. For production, use CPU or CUDA backends for deterministic results.

#### WSL2: "CUDA not available" but nvidia-smi works

**Cause:** Python can't find CUDA libraries.

**Solution:** Ensure you installed with `[nvidia]`:
```bash
pip install polyinfer[nvidia]
```

The `nvidia_setup.py` module automatically configures library paths.

### Debugging

#### Check Available Backends and Devices

```python
import polyinfer as pi

print("Backends:", pi.list_backends())
print("Devices:")
for d in pi.list_devices():
    print(f"  {d.name}: {d.backends}")
```

#### Check NVIDIA Library Detection

```python
from polyinfer.nvidia_setup import get_nvidia_info
import json
print(json.dumps(get_nvidia_info(), indent=2))
```

#### Check TensorRT Setup

```python
import polyinfer as pi
info = pi.get_nvidia_info()

print("TensorRT Setup:")
print(f"  Configured: {info['tensorrt_setup']['configured']}")
print(f"  TensorRT dirs: {info['tensorrt_setup']['tensorrt_dirs']}")
print(f"  Preloaded libs: {info['tensorrt_setup']['preloaded_libs']}")

# Check if TensorRT EP is available
import onnxruntime as ort
providers = ort.get_available_providers()
print(f"\nONNX Runtime providers: {providers}")
print(f"TensorRT EP available: {'TensorrtExecutionProvider' in providers}")
```

#### Verbose ONNX Runtime Logging

```python
import onnxruntime as ort
ort.set_default_logger_severity(0)  # 0=Verbose, 1=Info, 2=Warning, 3=Error
```

#### Check IREE Tools

```bash
# Check if IREE tools are available
which iree-import-onnx
which iree-compile

# Or in Python
from polyinfer.backends.iree.backend import _get_iree_import_onnx, _get_iree_compile
print("iree-import-onnx:", _get_iree_import_onnx())
print("iree-compile:", _get_iree_compile())
```

---

## Contributing

### Code Style

We use `ruff` for linting and formatting:

```bash
# Check code
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Type Checking

```bash
mypy src/polyinfer/
```

### Adding a New Backend

1. Create a new directory under `src/polyinfer/backends/`:
   ```
   backends/
   └── mybackend/
       ├── __init__.py
       └── backend.py
   ```

2. Implement the `Backend` and `CompiledModel` interfaces:
   ```python
   from polyinfer.backends.base import Backend, CompiledModel

   class MyModel(CompiledModel):
       @property
       def backend_name(self) -> str:
           return "mybackend"

       @property
       def device(self) -> str:
           return self._device

       def __call__(self, *inputs):
           # Run inference
           pass

   class MyBackend(Backend):
       @property
       def name(self) -> str:
           return "mybackend"

       @property
       def supported_devices(self) -> list[str]:
           return ["cpu", "custom-device"]

       def is_available(self) -> bool:
           # Check if backend can be used
           pass

       def load(self, model_path, device, **kwargs) -> MyModel:
           # Load and compile model
           pass
   ```

3. Register the backend in `_autoload.py`:
   ```python
   try:
       from polyinfer.backends.mybackend import MyBackend
       register_backend("mybackend", MyBackend)
   except ImportError:
       pass
   ```

4. Add tests in `tests/test_mybackend.py`

5. Update `pyproject.toml` with optional dependencies

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code is formatted (`ruff format`)
- [ ] No linting errors (`ruff check`)
- [ ] Type hints added for public APIs
- [ ] Documentation updated if needed
- [ ] CHANGELOG updated (if applicable)

---

## Performance Tips

### Backend Selection

| Use Case | Recommended Backend |
|----------|---------------------|
| Maximum NVIDIA performance | TensorRT |
| Good NVIDIA performance + compatibility | ONNX Runtime CUDA |
| Intel CPU optimization | OpenVINO |
| Intel GPU/NPU | OpenVINO |
| Cross-platform GPU | IREE Vulkan |
| AMD GPU (Windows) | ONNX Runtime DirectML |

### Benchmarking

```python
import polyinfer as pi
import numpy as np

model = pi.load("model.onnx", device="cuda")
input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Warm up is important for GPU
bench = model.benchmark(input_data, warmup=50, iterations=200)

print(f"Mean: {bench['mean_ms']:.2f}ms")
print(f"Std:  {bench['std_ms']:.2f}ms")
print(f"Min:  {bench['min_ms']:.2f}ms")
print(f"Max:  {bench['max_ms']:.2f}ms")
print(f"FPS:  {bench['fps']:.1f}")
```

### Memory Optimization

```python
# Use specific device index to control GPU memory
model = pi.load("model.onnx", device="cuda:0")

# For ONNX Runtime, you can pass session options
model = pi.load("model.onnx", device="cuda",
                cuda_mem_limit=2 * 1024 * 1024 * 1024)  # 2GB
```

---

## Author

Athrva Pandhare

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
