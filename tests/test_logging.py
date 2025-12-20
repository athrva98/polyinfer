"""Tests for polyinfer logging functionality."""

import logging
import sys
from io import StringIO

import pytest


class TestLoggingImports:
    """Test that logging functions are properly exported."""

    def test_logging_exports_available(self):
        """Test all logging functions are exported from polyinfer."""
        import polyinfer as pi

        # Core logging functions
        assert hasattr(pi, "set_log_level")
        assert hasattr(pi, "get_log_level")
        assert hasattr(pi, "get_log_level_name")
        assert hasattr(pi, "enable_logging")
        assert hasattr(pi, "disable_logging")
        assert hasattr(pi, "get_logger")
        assert hasattr(pi, "configure_logging")
        assert hasattr(pi, "LogContext")

    def test_logging_module_import(self):
        """Test direct import from logging module."""
        from polyinfer._logging import (
            configure_logging,
            disable_logging,
            enable_logging,
            get_log_level,
            get_log_level_name,
            get_logger,
            set_log_level,
        )

        # All should be callable
        assert callable(get_logger)
        assert callable(set_log_level)
        assert callable(get_log_level)
        assert callable(get_log_level_name)
        assert callable(enable_logging)
        assert callable(disable_logging)
        assert callable(configure_logging)


class TestLogLevels:
    """Test log level configuration."""

    def test_set_log_level_string(self):
        """Test setting log level with string values."""
        import polyinfer as pi

        # Test various string levels
        pi.set_log_level("DEBUG")
        assert pi.get_log_level() == logging.DEBUG

        pi.set_log_level("INFO")
        assert pi.get_log_level() == logging.INFO

        pi.set_log_level("WARNING")
        assert pi.get_log_level() == logging.WARNING

        pi.set_log_level("ERROR")
        assert pi.get_log_level() == logging.ERROR

        pi.set_log_level("CRITICAL")
        assert pi.get_log_level() == logging.CRITICAL

        # Reset to default
        pi.set_log_level("WARNING")

    def test_set_log_level_case_insensitive(self):
        """Test that log level strings are case-insensitive."""
        import polyinfer as pi

        pi.set_log_level("debug")
        assert pi.get_log_level() == logging.DEBUG

        pi.set_log_level("Debug")
        assert pi.get_log_level() == logging.DEBUG

        pi.set_log_level("DEBUG")
        assert pi.get_log_level() == logging.DEBUG

        # Reset
        pi.set_log_level("WARNING")

    def test_set_log_level_integer(self):
        """Test setting log level with integer values."""
        import polyinfer as pi

        pi.set_log_level(10)  # DEBUG
        assert pi.get_log_level() == 10

        pi.set_log_level(20)  # INFO
        assert pi.get_log_level() == 20

        pi.set_log_level(30)  # WARNING
        assert pi.get_log_level() == 30

        # Reset
        pi.set_log_level("WARNING")

    def test_set_log_level_invalid_raises(self):
        """Test that invalid log level strings raise ValueError."""
        import polyinfer as pi

        with pytest.raises(ValueError):
            pi.set_log_level("INVALID")

        with pytest.raises(ValueError):
            pi.set_log_level("foo")

    def test_get_log_level_name(self):
        """Test getting log level as string name."""
        import polyinfer as pi

        pi.set_log_level("DEBUG")
        assert pi.get_log_level_name() == "DEBUG"

        pi.set_log_level("INFO")
        assert pi.get_log_level_name() == "INFO"

        pi.set_log_level("WARNING")
        assert pi.get_log_level_name() == "WARNING"

    def test_silent_level(self):
        """Test SILENT log level disables all output."""
        import polyinfer as pi

        pi.set_log_level("SILENT")
        assert pi.get_log_level() > logging.CRITICAL
        assert pi.get_log_level_name() == "SILENT"

        # Reset
        pi.set_log_level("WARNING")


class TestEnableDisable:
    """Test enable/disable logging convenience functions."""

    def test_enable_logging_default(self):
        """Test enable_logging sets INFO level by default."""
        import polyinfer as pi

        pi.set_log_level("WARNING")  # Start at warning
        pi.enable_logging()
        assert pi.get_log_level() == logging.INFO

        # Reset
        pi.set_log_level("WARNING")

    def test_enable_logging_custom_level(self):
        """Test enable_logging with custom level."""
        import polyinfer as pi

        pi.enable_logging("DEBUG")
        assert pi.get_log_level() == logging.DEBUG

        pi.enable_logging("ERROR")
        assert pi.get_log_level() == logging.ERROR

        # Reset
        pi.set_log_level("WARNING")

    def test_disable_logging(self):
        """Test disable_logging silences all output."""
        import polyinfer as pi

        pi.enable_logging("DEBUG")
        pi.disable_logging()
        assert pi.get_log_level() > logging.CRITICAL

        # Reset
        pi.set_log_level("WARNING")


class TestLogContext:
    """Test LogContext context manager."""

    def test_log_context_temporary_level(self):
        """Test LogContext temporarily changes log level."""
        import polyinfer as pi

        pi.set_log_level("WARNING")
        original_level = pi.get_log_level()

        with pi.LogContext("DEBUG"):
            assert pi.get_log_level() == logging.DEBUG

        # Level should be restored
        assert pi.get_log_level() == original_level

    def test_log_context_restores_on_exception(self):
        """Test LogContext restores level even on exception."""
        import polyinfer as pi

        pi.set_log_level("WARNING")
        original_level = pi.get_log_level()

        try:
            with pi.LogContext("DEBUG"):
                assert pi.get_log_level() == logging.DEBUG
                raise ValueError("test error")
        except ValueError:
            pass

        # Level should still be restored
        assert pi.get_log_level() == original_level


class TestGetLogger:
    """Test get_logger function."""

    def test_get_root_logger(self):
        """Test getting the root polyinfer logger."""
        import polyinfer as pi

        logger = pi.get_logger()
        assert logger.name == "polyinfer"

    def test_get_named_logger(self):
        """Test getting a named sub-logger."""
        import polyinfer as pi

        logger = pi.get_logger("model")
        assert logger.name == "polyinfer.model"

        logger = pi.get_logger("backends.onnxruntime")
        assert logger.name == "polyinfer.backends.onnxruntime"

    def test_logger_hierarchy(self):
        """Test that child loggers inherit from parent."""
        import polyinfer as pi

        pi.get_logger()
        child = pi.get_logger("model")

        # Child's effective level should match parent
        pi.set_log_level("ERROR")
        assert child.getEffectiveLevel() == logging.ERROR

        # Reset
        pi.set_log_level("WARNING")


class TestLoggingOutput:
    """Test actual logging output."""

    def test_logging_output_captured(self):
        """Test that log messages are properly output."""
        import polyinfer as pi

        # Create a string buffer to capture output
        stream = StringIO()
        pi.configure_logging(level="INFO", stream=stream)

        # Log a message
        logger = pi.get_logger("test")
        logger.info("Test message")

        output = stream.getvalue()
        assert "Test message" in output
        assert "polyinfer.test" in output

        # Reset to default
        pi.configure_logging(level="WARNING", stream=sys.stderr)

    def test_log_level_filtering(self):
        """Test that log levels are properly filtered."""
        import polyinfer as pi

        stream = StringIO()
        pi.configure_logging(level="WARNING", stream=stream)

        logger = pi.get_logger("test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        output = stream.getvalue()
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output

        # Reset
        pi.configure_logging(level="WARNING", stream=sys.stderr)


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_with_custom_format(self):
        """Test configuring with custom format."""
        import polyinfer as pi

        stream = StringIO()
        pi.configure_logging(
            level="INFO",
            format="%(levelname)s - %(message)s",
            stream=stream,
        )

        logger = pi.get_logger("test")
        logger.info("Custom format test")

        output = stream.getvalue()
        assert "INFO - Custom format test" in output

        # Reset
        pi.configure_logging(level="WARNING", stream=sys.stderr)


class TestModuleLoggers:
    """Test that module-level loggers exist and work."""

    def test_model_logger_exists(self):
        """Test that model module has a logger."""
        from polyinfer._logging import get_logger

        logger = get_logger("model")
        assert logger is not None
        assert logger.name == "polyinfer.model"

    def test_backends_logger_exists(self):
        """Test that backends modules have loggers."""
        from polyinfer._logging import get_logger

        # Check backend loggers
        loggers = [
            get_logger("backends.registry"),
            get_logger("backends.autoload"),
            get_logger("backends.onnxruntime"),
            get_logger("backends.openvino"),
            get_logger("backends.tensorrt"),
            get_logger("backends.iree"),
        ]

        for logger in loggers:
            assert logger is not None
            assert logger.name.startswith("polyinfer.backends")

    def test_nvidia_setup_logger_exists(self):
        """Test that nvidia_setup module has a logger."""
        from polyinfer._logging import get_logger

        logger = get_logger("nvidia_setup")
        assert logger is not None
        assert logger.name == "polyinfer.nvidia_setup"
