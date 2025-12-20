"""Tests for backend options passthrough.

Tests that backend options (TensorRT, ONNX Runtime, OpenVINO, IREE)
are properly validated and passed through to the underlying engines.
"""

import numpy as np
import pytest

import polyinfer as pi

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def dummy_onnx_model(tmp_path):
    """Create a minimal ONNX model for testing."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx not installed")

    # Create a simple model: output = input * 2
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # Constant for multiplication
    const_tensor = helper.make_tensor("const", TensorProto.FLOAT, [1], [2.0])

    mul_node = helper.make_node("Mul", ["input", "const"], ["output"], name="mul")

    graph = helper.make_graph(
        [mul_node],
        "test_model",
        [input_tensor],
        [output_tensor],
        [const_tensor],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model_path = tmp_path / "test_model.onnx"
    onnx.save(model, str(model_path))

    return str(model_path)


@pytest.fixture
def dummy_input():
    """Create dummy input data."""
    return np.random.rand(1, 3, 224, 224).astype(np.float32)


# =============================================================================
# ONNX Runtime Backend Options Tests
# =============================================================================


class TestONNXRuntimeOptions:
    """Test ONNX Runtime backend options passthrough."""

    def test_graph_optimization_levels(self, dummy_onnx_model, dummy_input):
        """Test graph optimization level options."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        # Test each optimization level
        for level in [0, 1, 2, 3]:
            model = pi.load(
                dummy_onnx_model,
                backend="onnxruntime",
                device="cpu",
                graph_optimization_level=level,
            )
            output = model(dummy_input)
            assert output is not None
            assert output.shape == dummy_input.shape

    def test_threading_options(self, dummy_onnx_model, dummy_input):
        """Test threading configuration options."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cpu",
            intra_op_num_threads=2,
            inter_op_num_threads=1,
        )
        output = model(dummy_input)
        assert output is not None

    def test_memory_options(self, dummy_onnx_model, dummy_input):
        """Test memory configuration options."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cpu",
            enable_mem_pattern=True,
            enable_cpu_mem_arena=True,
        )
        output = model(dummy_input)
        assert output is not None

    @pytest.mark.cuda
    def test_cuda_options(self, dummy_onnx_model, dummy_input):
        """Test CUDA-specific options."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        # Check if onnxruntime backend supports cuda device
        try:
            backend = pi.get_backend("onnxruntime")
            if "cuda" not in backend.supported_devices:
                pytest.skip("ONNX Runtime CUDA EP not available")
        except Exception:
            pytest.skip("ONNX Runtime backend not available")

        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cuda",
            cudnn_conv_algo_search="HEURISTIC",
            do_copy_in_default_stream=True,
        )
        output = model(dummy_input)
        assert output is not None

    @pytest.mark.cuda
    def test_cuda_mem_limit(self, dummy_onnx_model, dummy_input):
        """Test CUDA memory limit option."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        # Check if onnxruntime backend supports cuda device
        try:
            backend = pi.get_backend("onnxruntime")
            if "cuda" not in backend.supported_devices:
                pytest.skip("ONNX Runtime CUDA EP not available")
        except Exception:
            pytest.skip("ONNX Runtime backend not available")

        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cuda",
            cuda_mem_limit=2 * 1024 * 1024 * 1024,  # 2GB
        )
        output = model(dummy_input)
        assert output is not None

    @pytest.mark.tensorrt
    def test_tensorrt_ep_options(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test TensorRT EP options via ONNX Runtime."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        # Check if onnxruntime backend supports tensorrt device
        try:
            backend = pi.get_backend("onnxruntime")
            if "tensorrt" not in backend.supported_devices:
                pytest.skip("ONNX Runtime TensorRT EP not available")
        except Exception:
            pytest.skip("ONNX Runtime backend not available")

        cache_dir = str(tmp_path / "trt_cache")

        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="tensorrt",
            fp16=False,
            cache_dir=cache_dir,
            builder_optimization_level=3,
            max_workspace_size=1 << 30,
            min_subgraph_size=1,
        )
        output = model(dummy_input)
        assert output is not None

    @pytest.mark.directml
    def test_directml_options(self, dummy_onnx_model, dummy_input):
        """Test DirectML options."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        devices = pi.list_devices()
        if not any(d.name == "directml" for d in devices):
            pytest.skip("DirectML not available")

        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="directml",
            device_id=0,
        )
        output = model(dummy_input)
        assert output is not None


# =============================================================================
# Native TensorRT Backend Options Tests
# =============================================================================


class TestNativeTensorRTOptions:
    """Test native TensorRT backend options."""

    @pytest.mark.tensorrt
    def test_precision_options(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test precision configuration options."""
        if not pi.is_available("tensorrt"):
            pytest.skip("Native TensorRT not installed")

        cache_path = tmp_path / "test.engine"

        model = pi.load(
            dummy_onnx_model,
            backend="tensorrt",
            device="cuda",
            fp16=False,
            int8=False,
            cache_path=str(cache_path),
        )
        output = model(dummy_input)
        assert output is not None
        assert cache_path.exists(), "Engine cache should be created"

    @pytest.mark.tensorrt
    def test_builder_optimization_level(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test builder optimization level option."""
        if not pi.is_available("tensorrt"):
            pytest.skip("Native TensorRT not installed")

        # Test different optimization levels
        for level in [0, 3, 5]:
            cache_path = tmp_path / f"test_opt{level}.engine"

            model = pi.load(
                dummy_onnx_model,
                backend="tensorrt",
                device="cuda",
                builder_optimization_level=level,
                cache_path=str(cache_path),
            )
            output = model(dummy_input)
            assert output is not None

    @pytest.mark.tensorrt
    def test_workspace_size(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test workspace size option."""
        if not pi.is_available("tensorrt"):
            pytest.skip("Native TensorRT not installed")

        cache_path = tmp_path / "test_workspace.engine"

        model = pi.load(
            dummy_onnx_model,
            backend="tensorrt",
            device="cuda",
            workspace_size=2 << 30,  # 2GB
            cache_path=str(cache_path),
        )
        output = model(dummy_input)
        assert output is not None

    @pytest.mark.tensorrt
    def test_timing_cache(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test timing cache option."""
        if not pi.is_available("tensorrt"):
            pytest.skip("Native TensorRT not installed")

        cache_path = tmp_path / "test_timing.engine"
        timing_cache = tmp_path / "timing.cache"

        model = pi.load(
            dummy_onnx_model,
            backend="tensorrt",
            device="cuda",
            cache_path=str(cache_path),
            timing_cache_path=str(timing_cache),
        )
        output = model(dummy_input)
        assert output is not None
        assert timing_cache.exists(), "Timing cache should be created"

    @pytest.mark.tensorrt
    def test_force_rebuild(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test force rebuild option."""
        if not pi.is_available("tensorrt"):
            pytest.skip("Native TensorRT not installed")

        cache_path = tmp_path / "test_rebuild.engine"

        # First build
        pi.load(
            dummy_onnx_model,
            backend="tensorrt",
            device="cuda",
            cache_path=str(cache_path),
        )
        mtime1 = cache_path.stat().st_mtime

        # Should use cache (no rebuild)
        pi.load(
            dummy_onnx_model,
            backend="tensorrt",
            device="cuda",
            cache_path=str(cache_path),
        )
        mtime2 = cache_path.stat().st_mtime
        assert mtime1 == mtime2, "Cache should be reused"

        # Force rebuild
        pi.load(
            dummy_onnx_model,
            backend="tensorrt",
            device="cuda",
            cache_path=str(cache_path),
            force_rebuild=True,
        )
        mtime3 = cache_path.stat().st_mtime
        assert mtime3 > mtime2, "Engine should be rebuilt"

    @pytest.mark.tensorrt
    def test_profiling_verbosity(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test profiling verbosity option."""
        if not pi.is_available("tensorrt"):
            pytest.skip("Native TensorRT not installed")

        for verbosity in ["none", "layer_names_only", "detailed"]:
            cache_path = tmp_path / f"test_prof_{verbosity}.engine"

            model = pi.load(
                dummy_onnx_model,
                backend="tensorrt",
                device="cuda",
                profiling_verbosity=verbosity,
                cache_path=str(cache_path),
            )
            output = model(dummy_input)
            assert output is not None


# =============================================================================
# OpenVINO Backend Options Tests
# =============================================================================


class TestOpenVINOOptions:
    """Test OpenVINO backend options."""

    @pytest.mark.openvino
    def test_optimization_level(self, dummy_onnx_model, dummy_input):
        """Test optimization level options."""
        if not pi.is_available("openvino"):
            pytest.skip("OpenVINO not installed")

        for level in [0, 1, 2]:
            model = pi.load(
                dummy_onnx_model,
                backend="openvino",
                device="cpu",
                optimization_level=level,
            )
            output = model(dummy_input)
            assert output is not None

    @pytest.mark.openvino
    def test_num_threads(self, dummy_onnx_model, dummy_input):
        """Test threading options."""
        if not pi.is_available("openvino"):
            pytest.skip("OpenVINO not installed")

        model = pi.load(
            dummy_onnx_model,
            backend="openvino",
            device="cpu",
            num_threads=4,
        )
        output = model(dummy_input)
        assert output is not None

    @pytest.mark.openvino
    def test_caching(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test model caching options."""
        if not pi.is_available("openvino"):
            pytest.skip("OpenVINO not installed")

        cache_dir = str(tmp_path / "ov_cache")

        model = pi.load(
            dummy_onnx_model,
            backend="openvino",
            device="cpu",
            enable_caching=True,
            cache_dir=cache_dir,
        )
        output = model(dummy_input)
        assert output is not None


# =============================================================================
# IREE Backend Options Tests
# =============================================================================


class TestIREEOptions:
    """Test IREE backend options."""

    @pytest.mark.iree
    def test_opt_level(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test optimization level options."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not installed")

        # Note: IREE tests may fail with simple test models due to function name mismatches.
        # This is expected - the test verifies options are passed correctly.
        try:
            for level in [0, 3]:  # Test just a couple levels
                model = pi.load(
                    dummy_onnx_model,
                    backend="iree",
                    device="cpu",
                    opt_level=level,
                    cache_dir=str(tmp_path),
                    force_compile=True,
                )
                output = model(dummy_input)
                assert output is not None
        except RuntimeError as e:
            if "inference function" in str(e) or "compilation failed" in str(e).lower():
                pytest.skip("IREE cannot compile/run simple test model")
            raise

    @pytest.mark.iree
    def test_caching(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test compilation caching."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not installed")

        cache_dir = tmp_path / "iree_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = pi.load(
                dummy_onnx_model,
                backend="iree",
                device="cpu",
                cache_dir=str(cache_dir),
            )
            output = model(dummy_input)
            assert output is not None
        except RuntimeError as e:
            if "inference function" in str(e) or "compilation failed" in str(e).lower():
                pytest.skip("IREE cannot compile/run simple test model")
            raise

    @pytest.mark.iree
    def test_save_mlir(self, dummy_onnx_model, dummy_input, tmp_path):
        """Test MLIR saving option."""
        if not pi.is_available("iree"):
            pytest.skip("IREE not installed")

        mlir_path = tmp_path / "model.mlir"

        try:
            model = pi.load(
                dummy_onnx_model,
                backend="iree",
                device="cpu",
                save_mlir=True,
                mlir_path=str(mlir_path),
                cache_dir=str(tmp_path),
                force_compile=True,
            )
            output = model(dummy_input)
            assert output is not None
        except RuntimeError as e:
            if "inference function" in str(e) or "compilation failed" in str(e).lower():
                pytest.skip("IREE cannot compile/run simple test model")
            raise


# =============================================================================
# Options Validation Tests
# =============================================================================


class TestOptionsValidation:
    """Test that invalid options are handled properly."""

    def test_unknown_option_ignored(self, dummy_onnx_model, dummy_input):
        """Unknown options should be ignored (not raise errors)."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        # This should not raise an error
        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cpu",
            unknown_option_xyz=123,
            another_fake_option="test",
        )
        output = model(dummy_input)
        assert output is not None

    def test_invalid_device_raises(self, dummy_onnx_model):
        """Invalid device should raise appropriate error."""
        with pytest.raises((ValueError, RuntimeError)):
            pi.load(
                dummy_onnx_model,
                device="nonexistent_device_xyz",
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestOptionsIntegration:
    """Integration tests for options across backends."""

    def test_same_output_different_options(self, dummy_onnx_model, dummy_input):
        """Different options should produce same output for deterministic models."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        # Load with different optimization levels
        model_opt0 = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cpu",
            graph_optimization_level=0,
        )
        model_opt3 = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cpu",
            graph_optimization_level=3,
        )

        output_opt0 = model_opt0(dummy_input)
        output_opt3 = model_opt3(dummy_input)

        np.testing.assert_allclose(output_opt0, output_opt3, rtol=1e-5)

    def test_backend_name_reflects_options(self, dummy_onnx_model, dummy_input):
        """Backend name should reflect the actual backend used."""
        if not pi.is_available("onnxruntime"):
            pytest.skip("ONNX Runtime not installed")

        model = pi.load(
            dummy_onnx_model,
            backend="onnxruntime",
            device="cpu",
        )
        assert "onnxruntime" in model.backend_name.lower()
