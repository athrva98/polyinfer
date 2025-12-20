"""Tests for MLIR emission and compilation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import polyinfer as pi

# Check if IREE is available
IREE_AVAILABLE = pi.is_available("iree")

# Skip all tests in this module if IREE is not available
pytestmark = pytest.mark.skipif(not IREE_AVAILABLE, reason="IREE backend not available")

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def model_path():
    """Get path to test ONNX model."""
    candidates = [
        Path(__file__).parent.parent / "yolov8n.onnx",
        Path(__file__).parent.parent / "examples" / "yolov8n.onnx",
        Path.cwd() / "yolov8n.onnx",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    pytest.skip("No test model found. Download yolov8n.onnx first.")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    # Use ignore_cleanup_errors=True on Windows because IREE may hold
    # file handles to VMFB files even after the model is out of scope
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# MLIR Export Tests
# =============================================================================


class TestMLIRExport:
    """Tests for export_mlir functionality."""

    def test_export_mlir_basic(self, model_path, temp_dir):
        """Test basic MLIR export."""
        output_path = temp_dir / "model.mlir"
        mlir = pi.export_mlir(model_path, output_path)

        assert mlir.path == output_path
        assert mlir.path.exists()
        assert mlir.source_model == Path(model_path)
        assert mlir.dialect == "iree"

    def test_export_mlir_default_path(self, model_path, temp_dir):
        """Test MLIR export with default output path."""
        # Copy model to temp dir to test default path behavior
        import shutil

        temp_model = temp_dir / "test_model.onnx"
        shutil.copy(model_path, temp_model)

        mlir = pi.export_mlir(temp_model)

        expected_path = temp_dir / "test_model.mlir"
        assert mlir.path == expected_path
        assert mlir.path.exists()

    def test_export_mlir_load_content(self, model_path, temp_dir):
        """Test MLIR export with content loading."""
        output_path = temp_dir / "model.mlir"
        mlir = pi.export_mlir(model_path, output_path, load_content=True)

        assert mlir.content is not None
        assert len(mlir.content) > 0
        assert "module" in mlir.content
        assert "func.func" in mlir.content

    def test_export_mlir_content_matches_file(self, model_path, temp_dir):
        """Test that loaded content matches file content."""
        output_path = temp_dir / "model.mlir"
        mlir = pi.export_mlir(model_path, output_path, load_content=True)

        file_content = output_path.read_text()
        assert mlir.content == file_content

    def test_export_mlir_str_conversion(self, model_path, temp_dir):
        """Test MLIROutput string conversion."""
        output_path = temp_dir / "model.mlir"
        mlir = pi.export_mlir(model_path, output_path)

        # str() should return file content
        content = str(mlir)
        assert "module" in content
        assert "func.func" in content

    def test_export_mlir_save_method(self, model_path, temp_dir):
        """Test MLIROutput.save() method."""
        output_path = temp_dir / "model.mlir"
        mlir = pi.export_mlir(model_path, output_path, load_content=True)

        # Save to new location
        new_path = temp_dir / "backup" / "model_backup.mlir"
        saved_path = mlir.save(new_path)

        assert saved_path == new_path
        assert new_path.exists()
        assert new_path.read_text() == mlir.content

    def test_export_mlir_creates_parent_dirs(self, model_path, temp_dir):
        """Test that export creates parent directories."""
        output_path = temp_dir / "nested" / "dir" / "model.mlir"
        mlir = pi.export_mlir(model_path, output_path)

        assert mlir.path.exists()
        assert mlir.path.parent.exists()

    def test_export_mlir_file_not_found(self, temp_dir):
        """Test export with non-existent model."""
        with pytest.raises(FileNotFoundError):
            pi.export_mlir("nonexistent_model.onnx", temp_dir / "out.mlir")


# =============================================================================
# MLIR Compilation Tests
# =============================================================================


class TestMLIRCompilation:
    """Tests for compile_mlir functionality."""

    @pytest.fixture
    def mlir_file(self, model_path, temp_dir):
        """Create MLIR file for compilation tests."""
        output_path = temp_dir / "model.mlir"
        mlir = pi.export_mlir(model_path, output_path)
        return mlir.path

    def test_compile_mlir_cpu(self, mlir_file, temp_dir):
        """Test MLIR compilation for CPU."""
        vmfb_path = pi.compile_mlir(
            mlir_file, device="cpu", output_path=temp_dir / "model_cpu.vmfb"
        )

        assert vmfb_path.exists()
        assert vmfb_path.suffix == ".vmfb"

    @pytest.mark.vulkan
    def test_compile_mlir_vulkan(self, mlir_file, temp_dir):
        """Test MLIR compilation for Vulkan."""
        vmfb_path = pi.compile_mlir(
            mlir_file, device="vulkan", output_path=temp_dir / "model_vulkan.vmfb"
        )

        assert vmfb_path.exists()

    def test_compile_mlir_default_path(self, mlir_file):
        """Test MLIR compilation with default output path."""
        vmfb_path = pi.compile_mlir(mlir_file, device="cpu")

        # Should be next to MLIR file
        assert vmfb_path.exists()
        assert vmfb_path.parent == mlir_file.parent

        # Cleanup
        vmfb_path.unlink()

    def test_compile_mlir_opt_levels(self, mlir_file, temp_dir):
        """Test different optimization levels."""
        for opt_level in [0, 1, 2, 3]:
            vmfb_path = pi.compile_mlir(
                mlir_file,
                device="cpu",
                output_path=temp_dir / f"model_O{opt_level}.vmfb",
                opt_level=opt_level,
            )
            assert vmfb_path.exists()

    def test_compile_mlir_file_not_found(self, temp_dir):
        """Test compilation with non-existent MLIR file."""
        with pytest.raises(FileNotFoundError):
            pi.compile_mlir("nonexistent.mlir", device="cpu")


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestMLIRWorkflow:
    """End-to-end tests for MLIR workflow."""

    def test_export_compile_load_cpu(self, model_path, temp_dir):
        """Test full workflow: export -> compile -> load -> inference (CPU)."""
        # 1. Export MLIR
        mlir = pi.export_mlir(model_path, temp_dir / "model.mlir")

        # 2. Compile to VMFB
        vmfb_path = pi.compile_mlir(mlir.path, device="cpu", output_path=temp_dir / "model.vmfb")

        # 3. Load model
        backend = pi.get_backend("iree")
        model = backend.load_vmfb(vmfb_path, device="cpu")

        # 4. Run inference
        input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
        output = model(input_data)

        assert output is not None
        assert output.shape == (1, 84, 8400)

    @pytest.mark.vulkan
    def test_export_compile_load_vulkan(self, model_path, temp_dir):
        """Test full workflow: export -> compile -> load -> inference (Vulkan)."""
        # 1. Export MLIR
        mlir = pi.export_mlir(model_path, temp_dir / "model.mlir")

        # 2. Compile to VMFB
        vmfb_path = pi.compile_mlir(mlir.path, device="vulkan", output_path=temp_dir / "model.vmfb")

        # 3. Load model
        backend = pi.get_backend("iree")
        model = backend.load_vmfb(vmfb_path, device="vulkan")

        # 4. Run inference
        input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)
        output = model(input_data)

        assert output is not None
        assert output.shape == (1, 84, 8400)

    def test_mlir_output_consistency(self, model_path, temp_dir):
        """Test that MLIR workflow produces same results as direct load."""
        input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

        # Direct load
        model_direct = pi.load(model_path, backend="iree", device="cpu")
        output_direct = model_direct(input_data)

        # MLIR workflow
        mlir = pi.export_mlir(model_path, temp_dir / "model.mlir")
        vmfb_path = pi.compile_mlir(mlir.path, device="cpu", output_path=temp_dir / "model.vmfb")
        backend = pi.get_backend("iree")
        model_mlir = backend.load_vmfb(vmfb_path, device="cpu")
        output_mlir = model_mlir(input_data)

        # Should produce identical results
        np.testing.assert_array_equal(output_direct, output_mlir)


# =============================================================================
# Backend Method Tests
# =============================================================================


class TestIREEBackendMethods:
    """Tests for IREEBackend emit_mlir and compile_mlir methods."""

    def test_backend_emit_mlir(self, model_path, temp_dir):
        """Test IREEBackend.emit_mlir directly."""
        backend = pi.get_backend("iree")
        mlir = backend.emit_mlir(model_path, temp_dir / "model.mlir")

        assert mlir.path.exists()
        assert mlir.dialect == "iree"

    def test_backend_compile_mlir(self, model_path, temp_dir):
        """Test IREEBackend.compile_mlir directly."""
        backend = pi.get_backend("iree")

        # First export
        mlir = backend.emit_mlir(model_path, temp_dir / "model.mlir")

        # Then compile
        vmfb_path = backend.compile_mlir(
            mlir.path, device="cpu", output_path=temp_dir / "model.vmfb"
        )

        assert vmfb_path.exists()

    def test_backend_load_vmfb(self, model_path, temp_dir):
        """Test IREEBackend.load_vmfb directly."""
        backend = pi.get_backend("iree")

        # Export and compile
        mlir = backend.emit_mlir(model_path, temp_dir / "model.mlir")
        vmfb_path = backend.compile_mlir(mlir.path, device="cpu")

        # Load
        model = backend.load_vmfb(vmfb_path, device="cpu")

        assert model is not None
        assert "iree" in model.backend_name


# =============================================================================
# MLIR Content Analysis Tests
# =============================================================================


class TestMLIRContent:
    """Tests for MLIR content analysis."""

    def test_mlir_contains_expected_ops(self, model_path, temp_dir):
        """Test that MLIR contains expected operations for YOLOv8."""
        mlir = pi.export_mlir(model_path, temp_dir / "model.mlir", load_content=True)

        # YOLOv8 should have these operations
        content = mlir.content
        assert "func.func @main_graph" in content
        # Should have convolution operations
        assert "onnx.Conv" in content or "torch.operator" in content

    def test_mlir_input_output_shapes(self, model_path, temp_dir):
        """Test that MLIR shows expected input/output shapes."""
        mlir = pi.export_mlir(model_path, temp_dir / "model.mlir", load_content=True)

        content = mlir.content
        # YOLOv8n input shape
        assert "1,3,640,640" in content or "[1, 3, 640, 640]" in content
        # YOLOv8n output shape
        assert "1,84,8400" in content or "[1, 84, 8400]" in content
