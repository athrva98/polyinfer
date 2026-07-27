"""Tests for positional and named input binding.

Backends bound positional inputs with ``zip(names, inputs, strict=False)``,
which silently dropped extra inputs and left missing ones unbound. An arity
mistake surfaced as a confusing backend error - or, when a caller passed
inputs in the wrong order, as a silently wrong result.
"""

import numpy as np
import pytest

from polyinfer.backends.base import CompiledModel


class FakeModel(CompiledModel):
    """Minimal CompiledModel exercising the shared binding logic."""

    def __init__(self, input_names, output_names):
        self._input_names = list(input_names)
        self._output_names = list(output_names)
        self.received = None

    @property
    def backend_name(self) -> str:
        return "fake-backend"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def __call__(self, *inputs):
        self._check_input_count(inputs)
        self.received = inputs
        outputs = tuple(np.zeros((1,), dtype=np.float32) for _ in self._output_names)
        return outputs[0] if len(outputs) == 1 else outputs


def _arr():
    return np.zeros((1, 3), dtype=np.float32)


class TestPositionalInputCount:
    def test_correct_count_accepted(self):
        model = FakeModel(["a", "b"], ["out"])
        model(_arr(), _arr())
        assert len(model.received) == 2

    def test_too_few_inputs_raises(self):
        """Regression: a missing input was silently left unbound."""
        model = FakeModel(["a", "b", "c"], ["out"])
        with pytest.raises(ValueError, match="expects 3 input"):
            model(_arr(), _arr())

    def test_too_many_inputs_raises(self):
        """Regression: extra inputs were silently discarded."""
        model = FakeModel(["a"], ["out"])
        with pytest.raises(ValueError, match="expects 1 input"):
            model(_arr(), _arr())

    def test_error_lists_expected_names_in_order(self):
        model = FakeModel(["input_ids", "attention_mask"], ["out"])
        with pytest.raises(ValueError) as exc:
            model(_arr())

        message = str(exc.value)
        assert "input_ids" in message
        assert "attention_mask" in message
        # Must point the user at the name-based alternative.
        assert "run(" in message

    def test_no_validation_when_names_unknown(self):
        """Backends that cannot report input names must not be blocked."""
        model = FakeModel([], [])
        model(_arr(), _arr(), _arr())  # must not raise


class TestNamedInputBinding:
    def test_run_binds_by_name(self):
        model = FakeModel(["a", "b"], ["out"])
        model.run({"a": _arr(), "b": _arr()})
        assert len(model.received) == 2

    def test_run_missing_input_raises_clear_error(self):
        """Regression: this used to raise a bare KeyError."""
        model = FakeModel(["a", "b"], ["out"])
        with pytest.raises(ValueError) as exc:
            model.run({"a": _arr()})

        message = str(exc.value)
        assert "Missing required input" in message
        assert "'b'" in message or "b" in message

    def test_run_ignores_extra_named_inputs(self):
        """Extra keys are harmless as long as every required name is present."""
        model = FakeModel(["a"], ["out"])
        model.run({"a": _arr(), "unused": _arr()})
        assert len(model.received) == 1

    def test_run_is_order_independent(self):
        """Binding by name must not depend on dict insertion order.

        This is the failure that positional binding cannot catch: a dict
        built in a different order than the model declares its inputs.
        """
        model = FakeModel(["a", "b"], ["out"])
        first = _arr() + 1
        second = _arr() + 2

        model.run({"b": second, "a": first})
        assert np.array_equal(model.received[0], first)
        assert np.array_equal(model.received[1], second)

    def test_run_maps_outputs_to_names(self):
        model = FakeModel(["a"], ["logits", "probs"])
        outputs = model.run({"a": _arr()})
        assert set(outputs) == {"logits", "probs"}
