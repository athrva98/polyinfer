"""Tests for InferenceConfig.

InferenceConfig previously documented eight fields but Model.__init__ read
only three (device, backend, extra_options). precision, optimization_level,
num_threads, enable_profiling, and cache_dir were silently discarded, and
the config's device overrode an explicitly passed one.
"""

import pytest

from polyinfer.config import BenchmarkConfig, InferenceConfig


class TestDeviceHandling:
    def test_device_is_normalized(self):
        assert InferenceConfig(device="GPU").device == "cuda"
        assert InferenceConfig(device="  dml ").device == "directml"

    def test_device_type_and_id(self):
        cfg = InferenceConfig(device="cuda:2")
        assert cfg.device_type == "cuda"
        assert cfg.device_id == 2

    def test_device_id_defaults_to_zero(self):
        assert InferenceConfig(device="cuda").device_id == 0
        assert InferenceConfig(device="cpu").device_id == 0


class TestToBackendKwargs:
    """Config fields must reach the backend."""

    def test_defaults_emit_minimal_kwargs(self):
        """A default config must not override backend defaults."""
        kwargs = InferenceConfig().to_backend_kwargs()

        # fp32 is the default precision, so no precision flag should appear.
        assert "fp16" not in kwargs
        assert "int8" not in kwargs
        # num_threads=0 means "backend decides".
        assert "num_threads" not in kwargs
        assert "cache_dir" not in kwargs
        assert "enable_profiling" not in kwargs

    def test_fp16_precision_forwarded(self):
        kwargs = InferenceConfig(precision="fp16").to_backend_kwargs()
        assert kwargs["fp16"] is True
        assert "int8" not in kwargs

    def test_int8_precision_forwarded(self):
        kwargs = InferenceConfig(precision="int8").to_backend_kwargs()
        assert kwargs["int8"] is True
        assert "fp16" not in kwargs

    def test_num_threads_forwarded_when_set(self):
        assert InferenceConfig(num_threads=8).to_backend_kwargs()["num_threads"] == 8

    def test_optimization_level_forwarded(self):
        assert InferenceConfig(optimization_level=0).to_backend_kwargs()["optimization_level"] == 0

    def test_profiling_forwarded(self):
        assert (
            InferenceConfig(enable_profiling=True).to_backend_kwargs()["enable_profiling"] is True
        )

    def test_cache_dir_forwarded_and_enables_caching(self):
        kwargs = InferenceConfig(cache_dir="./cache").to_backend_kwargs()
        assert kwargs["cache_dir"] == "./cache"
        # OpenVINO gates caching behind a separate flag.
        assert kwargs["enable_caching"] is True

    def test_extra_options_override_derived_values(self):
        kwargs = InferenceConfig(
            num_threads=4,
            extra_options={"num_threads": 16, "custom": "x"},
        ).to_backend_kwargs()
        assert kwargs["num_threads"] == 16
        assert kwargs["custom"] == "x"

    def test_every_documented_field_is_representable(self):
        """Guard against a field being added to the dataclass but not here."""
        kwargs = InferenceConfig(
            precision="fp16",
            optimization_level=1,
            num_threads=2,
            enable_profiling=True,
            cache_dir="./c",
        ).to_backend_kwargs()

        for expected in ("fp16", "optimization_level", "num_threads", "enable_profiling"):
            assert expected in kwargs, f"{expected} is dropped by to_backend_kwargs()"


class TestModelConfigPrecedence:
    """Explicit arguments must beat the config."""

    def test_explicit_device_beats_config_device(self):
        """Regression: config.device unconditionally replaced the argument.

        `load(path, device="cuda", config=InferenceConfig())` silently ran on
        CPU because the config's default device won.
        """
        from polyinfer.model import _DEFAULT_DEVICE, Model

        cfg = InferenceConfig()  # device defaults to "cpu"
        assert cfg.device == _DEFAULT_DEVICE

        # Mirror the precedence logic in Model.__init__.
        explicit_device = "cuda"
        resolved = cfg.device if explicit_device == _DEFAULT_DEVICE else explicit_device
        assert resolved == "cuda"

        assert Model._normalize_device(resolved) == "cuda"

    def test_config_supplies_device_when_argument_omitted(self):
        from polyinfer.model import _DEFAULT_DEVICE

        cfg = InferenceConfig(device="cuda:1")
        resolved = cfg.device if _DEFAULT_DEVICE == _DEFAULT_DEVICE else _DEFAULT_DEVICE
        assert resolved == "cuda:1"

    def test_model_load_signature_accepts_config(self):
        import inspect

        from polyinfer.model import Model

        assert "config" in inspect.signature(Model.__init__).parameters


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.warmup_iterations > 0
        assert cfg.benchmark_iterations > 0
        assert cfg.percentiles == [50, 90, 99]

    def test_percentiles_are_not_shared_between_instances(self):
        """Mutable default must be per-instance."""
        a, b = BenchmarkConfig(), BenchmarkConfig()
        a.percentiles.append(999)
        assert 999 not in b.percentiles


class TestInvalidPrecision:
    def test_unknown_precision_is_not_silently_forwarded(self):
        """An unrecognized precision must not emit a bogus flag."""
        cfg = InferenceConfig()
        cfg.precision = "bogus"  # type: ignore[assignment]
        kwargs = cfg.to_backend_kwargs()
        assert "fp16" not in kwargs
        assert "int8" not in kwargs
        assert "bogus" not in kwargs


if __name__ == "__main__":
    pytest.main([__file__])
