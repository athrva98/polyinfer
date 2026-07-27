"""Tests for IREE compiled-artifact metadata and cache keying.

These cover three defects that did not require IREE to be installed to
demonstrate:

1. _load_vmfb() hardcoded input_names=["input"], output_names=["output"],
   so Model.run() raised KeyError on every real IREE model and multi-output
   models lost outputs.
2. IREEModel.__call__ coerced every input to float32, silently corrupting
   integer tensors such as token IDs and attention masks.
3. The VMFB cache filename ignored opt_level, data_tiling, opset_version,
   extra_flags, and the source model's contents, so changing any of them
   reused a stale artifact.
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="onnx is a core dependency")

from polyinfer.backends.iree.backend import (  # noqa: E402
    IREECompileOptions,
    compilation_cache_key,
    read_io_metadata,
    read_onnx_io_metadata,
    write_io_metadata,
)


@pytest.fixture
def mixed_dtype_model(tmp_path):
    """ONNX model with float32 and int64 inputs and two outputs.

    Mirrors a transformer's signature, where the float32 coercion bug did
    real damage.
    """
    from onnx import TensorProto, helper

    embeddings = helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, [1, 4, 8])
    token_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 4])

    scaled = helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [1, 4, 8])
    doubled_ids = helper.make_tensor_value_info("doubled_ids", TensorProto.INT64, [1, 4])

    two_f = helper.make_tensor("two_f", TensorProto.FLOAT, [1], [2.0])
    two_i = helper.make_tensor("two_i", TensorProto.INT64, [1], [2])

    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["embeddings", "two_f"], ["scaled"]),
            helper.make_node("Mul", ["input_ids", "two_i"], ["doubled_ids"]),
        ],
        "mixed",
        [embeddings, token_ids],
        [scaled, doubled_ids],
        [two_f, two_i],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    path = tmp_path / "mixed.onnx"
    onnx.save(model, str(path))
    return path


class TestONNXIOMetadata:
    def test_extracts_real_input_names(self, mixed_dtype_model):
        """Regression: names were hardcoded to ["input"]."""
        meta = read_onnx_io_metadata(mixed_dtype_model)
        assert [spec["name"] for spec in meta["inputs"]] == ["embeddings", "input_ids"]

    def test_extracts_all_output_names(self, mixed_dtype_model):
        """Regression: outputs were hardcoded to a single ["output"]."""
        meta = read_onnx_io_metadata(mixed_dtype_model)
        assert [spec["name"] for spec in meta["outputs"]] == ["scaled", "doubled_ids"]

    def test_preserves_integer_dtypes(self, mixed_dtype_model):
        """Integer inputs must be recorded as integers, not floats."""
        meta = read_onnx_io_metadata(mixed_dtype_model)
        dtypes = {spec["name"]: spec["dtype"] for spec in meta["inputs"]}
        assert dtypes["embeddings"] == "float32"
        assert dtypes["input_ids"] == "int64"

    def test_records_shapes(self, mixed_dtype_model):
        meta = read_onnx_io_metadata(mixed_dtype_model)
        shapes = {spec["name"]: spec["shape"] for spec in meta["inputs"]}
        assert shapes["embeddings"] == [1, 4, 8]
        assert shapes["input_ids"] == [1, 4]

    def test_initializers_excluded_from_inputs(self, mixed_dtype_model):
        """Constants are not runtime inputs."""
        names = [spec["name"] for spec in read_onnx_io_metadata(mixed_dtype_model)["inputs"]]
        assert "two_f" not in names
        assert "two_i" not in names


class TestMetadataSidecar:
    def test_roundtrip(self, tmp_path, mixed_dtype_model):
        vmfb = tmp_path / "model.vmfb"
        vmfb.write_bytes(b"fake")

        original = read_onnx_io_metadata(mixed_dtype_model)
        write_io_metadata(vmfb, original)

        assert read_io_metadata(vmfb) == original

    def test_missing_sidecar_returns_none(self, tmp_path):
        vmfb = tmp_path / "nometa.vmfb"
        vmfb.write_bytes(b"fake")
        assert read_io_metadata(vmfb) is None

    def test_corrupt_sidecar_returns_none(self, tmp_path):
        """A damaged sidecar must degrade gracefully, not crash the load."""
        vmfb = tmp_path / "model.vmfb"
        vmfb.write_bytes(b"fake")
        (tmp_path / "model.vmfb.io.json").write_text("{not json", encoding="utf-8")

        assert read_io_metadata(vmfb) is None

    def test_sidecar_sits_beside_the_artifact(self, tmp_path):
        vmfb = tmp_path / "model.vmfb"
        vmfb.write_bytes(b"fake")
        write_io_metadata(vmfb, {"inputs": [], "outputs": []})

        assert (tmp_path / "model.vmfb.io.json").exists()


class TestCompilationCacheKey:
    """Every input that changes the artifact must change the key."""

    def test_key_is_deterministic(self):
        assert compilation_cache_key(None, "llvm-cpu", IREECompileOptions()) == (
            compilation_cache_key(None, "llvm-cpu", IREECompileOptions())
        )

    @pytest.mark.parametrize(
        "options",
        [
            IREECompileOptions(opt_level=3),
            IREECompileOptions(data_tiling=False),
            IREECompileOptions(opset_version=17),
            IREECompileOptions(extra_flags=["--custom"]),
            IREECompileOptions(strip_debug=True),
            IREECompileOptions(const_eval=False),
            IREECompileOptions(vulkan_target="rdna3"),
        ],
    )
    def test_option_change_changes_key(self, options):
        """Regression: only vulkan_target affected the cache filename."""
        baseline = compilation_cache_key(None, "llvm-cpu", IREECompileOptions())
        assert compilation_cache_key(None, "llvm-cpu", options) != baseline

    def test_target_change_changes_key(self):
        opts = IREECompileOptions()
        assert compilation_cache_key(None, "llvm-cpu", opts) != compilation_cache_key(
            None, "vulkan-spirv", opts
        )

    def test_source_model_edit_changes_key(self, tmp_path):
        """Editing the ONNX must invalidate the cached VMFB."""
        model = tmp_path / "m.onnx"
        model.write_bytes(b"a" * 100)
        before = compilation_cache_key(model, "llvm-cpu", IREECompileOptions())

        model.write_bytes(b"b" * 200)  # different size
        after = compilation_cache_key(model, "llvm-cpu", IREECompileOptions())

        assert before != after

    def test_key_is_filename_safe(self):
        key = compilation_cache_key(None, "vulkan-spirv", IREECompileOptions(extra_flags=["--x=y"]))
        assert key.isalnum()
        assert len(key) == 12


class TestInputDtypeHandling:
    """IREEModel must stop coercing everything to float32."""

    @staticmethod
    def _prepare(array, dtype):
        from polyinfer.backends.iree.backend import IREEModel

        # _prepare_input does not touch instance state, so call it unbound
        # to avoid constructing a model (which needs the IREE runtime).
        return IREEModel._prepare_input(None, array, dtype)

    def test_integer_input_preserved(self):
        """Regression: token IDs were reinterpreted as float32."""
        token_ids = np.array([[101, 2054, 2003, 102]], dtype=np.int64)
        assert self._prepare(token_ids, "int64").dtype == np.int64

    def test_float_input_preserved(self):
        data = np.zeros((1, 3), dtype=np.float32)
        assert self._prepare(data, "float32").dtype == np.float32

    def test_casts_to_expected_dtype(self):
        """float64 input for a float32 model should be narrowed."""
        data = np.zeros((1, 3), dtype=np.float64)
        assert self._prepare(data, "float32").dtype == np.float32

    def test_unknown_dtype_leaves_array_untouched(self):
        """Without metadata we must not guess - especially not float32."""
        token_ids = np.array([[1, 2, 3]], dtype=np.int64)
        assert self._prepare(token_ids, None).dtype == np.int64

    def test_result_is_contiguous(self):
        base = np.zeros((4, 8), dtype=np.float32)
        assert self._prepare(base.T, "float32").flags["C_CONTIGUOUS"]

    def test_integer_values_are_not_corrupted(self):
        token_ids = np.array([[50256, 1, 2]], dtype=np.int64)
        assert np.array_equal(self._prepare(token_ids, "int64"), token_ids)
