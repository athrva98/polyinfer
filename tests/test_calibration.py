"""Tests for calibration data handling and TensorRT quantization guards.

ONNX Runtime calls CalibrationDataReader.rewind() between calibration
passes, and the entropy and percentile methods need more than one pass.
_setup_iterator() previously re-assigned the *same* exhausted generator, so
every pass after the first saw nothing and calibration silently proceeded on
first-pass statistics alone.

These exercise _ORTCalibrationDataReader directly, which needs only
onnxruntime - not onnxruntime.quantization.
"""

import numpy as np
import pytest

pytest.importorskip("onnx", reason="onnx is a core dependency")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime required")

from polyinfer.quantization import (  # noqa: E402
    _ORTCalibrationDataReader,
    quantize_for_tensorrt,
)


@pytest.fixture
def single_input_model(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    const = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "two"], ["output"])],
        "m",
        [inp],
        [out],
        [const],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    return path


def _samples(n=5):
    return [np.full((1, 4), i, dtype=np.float32) for i in range(n)]


def _drain(reader):
    """Consume a reader fully, returning every batch it yielded."""
    batches = []
    while (batch := reader.get_next()) is not None:
        batches.append(batch)
    return batches


class TestRewindWithList:
    def test_list_replays_after_rewind(self, single_input_model):
        reader = _ORTCalibrationDataReader(single_input_model, _samples(5), num_samples=100)

        first = _drain(reader)
        reader.rewind()
        second = _drain(reader)

        assert len(first) == 5
        assert len(second) == 5, "list-backed reader failed to replay"


class TestRewindWithGenerator:
    """The case that silently broke multi-pass calibration."""

    def test_generator_replays_after_rewind(self, single_input_model):
        def gen():
            yield from _samples(5)

        reader = _ORTCalibrationDataReader(single_input_model, gen(), num_samples=100)

        first = _drain(reader)
        reader.rewind()
        second = _drain(reader)

        assert len(first) == 5
        assert len(second) == 5, (
            "generator-backed reader yielded nothing on the second pass; "
            "entropy/percentile calibration would silently see no data"
        )

    def test_generator_replay_is_identical(self, single_input_model):
        def gen():
            yield from _samples(3)

        reader = _ORTCalibrationDataReader(single_input_model, gen(), num_samples=100)
        first = _drain(reader)
        reader.rewind()
        second = _drain(reader)

        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a["input"], b["input"])

    def test_multiple_rewinds(self, single_input_model):
        def gen():
            yield from _samples(4)

        reader = _ORTCalibrationDataReader(single_input_model, gen(), num_samples=100)
        for _ in range(3):
            assert len(_drain(reader)) == 4
            reader.rewind()


class TestRewindWithFactory:
    def test_factory_is_reinvoked(self, single_input_model):
        calls = []

        def factory():
            calls.append(1)
            return iter(_samples(3))

        reader = _ORTCalibrationDataReader(single_input_model, factory, num_samples=100)
        assert len(_drain(reader)) == 3
        reader.rewind()
        assert len(_drain(reader)) == 3
        assert len(calls) >= 2, "factory was not re-invoked on rewind"


class TestSampleLimit:
    def test_num_samples_caps_a_list(self, single_input_model):
        reader = _ORTCalibrationDataReader(single_input_model, _samples(50), num_samples=10)
        assert len(_drain(reader)) == 10

    def test_num_samples_caps_a_generator(self, single_input_model):
        def gen():
            yield from _samples(50)

        reader = _ORTCalibrationDataReader(single_input_model, gen(), num_samples=10)
        assert len(_drain(reader)) == 10

    def test_infinite_generator_is_bounded(self, single_input_model):
        """Materialization must not hang on an unbounded source."""

        def endless():
            while True:
                yield np.zeros((1, 4), dtype=np.float32)

        reader = _ORTCalibrationDataReader(single_input_model, endless(), num_samples=7)
        assert len(_drain(reader)) == 7


class TestArrayWrapping:
    def test_bare_arrays_are_wrapped_with_input_name(self, single_input_model):
        reader = _ORTCalibrationDataReader(single_input_model, _samples(2), num_samples=10)
        batch = reader.get_next()
        assert set(batch) == {"input"}

    def test_dicts_pass_through(self, single_input_model):
        data = [{"input": np.zeros((1, 4), dtype=np.float32)} for _ in range(2)]
        reader = _ORTCalibrationDataReader(single_input_model, data, num_samples=10)
        assert set(reader.get_next()) == {"input"}


class TestTensorRTQuantizationGuards:
    def test_int8_refuses_instead_of_pretending(self, single_input_model):
        """Regression: this printed a success message and did nothing.

        Combined with pi.load(..., int8=True), which sets BuilderFlag.INT8
        with no calibrator, the result was a silently miscalibrated engine.
        """
        with pytest.raises(NotImplementedError, match="INT8"):
            quantize_for_tensorrt(
                single_input_model,
                precision="int8",
                calibration_data=_samples(3),
            )

    def test_int8_without_data_still_raises(self, single_input_model):
        with pytest.raises((NotImplementedError, ValueError)):
            quantize_for_tensorrt(single_input_model, precision="int8")

    def test_fp16_returns_model_path(self, single_input_model):
        assert quantize_for_tensorrt(single_input_model, precision="fp16") == single_input_model

    def test_unknown_precision_raises(self, single_input_model):
        with pytest.raises(ValueError, match="Unknown precision"):
            quantize_for_tensorrt(single_input_model, precision="fp4")

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            quantize_for_tensorrt(tmp_path / "nope.onnx", precision="fp16")
