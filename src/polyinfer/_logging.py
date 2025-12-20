"""Structured logging with configurable verbosity for PolyInfer.

This module provides a unified logging interface for the entire library.
Users can configure verbosity levels to control debug output.

Usage:
    import polyinfer as pi

    # Set global log level
    pi.set_log_level("DEBUG")  # Most verbose
    pi.set_log_level("INFO")   # Normal operation info
    pi.set_log_level("WARNING")  # Warnings only (default)
    pi.set_log_level("ERROR")  # Errors only
    pi.set_log_level("CRITICAL")  # Critical errors only

    # Or use numeric levels
    pi.set_log_level(10)  # DEBUG

    # Enable/disable logging quickly
    pi.enable_logging()   # Set to INFO
    pi.disable_logging()  # Set to CRITICAL (effectively silent)

    # Get current logger for advanced usage
    logger = pi.get_logger()
"""

import logging
import sys

# Create the polyinfer logger hierarchy
_logger = logging.getLogger("polyinfer")

# Default: WARNING level (minimal output)
_logger.setLevel(logging.WARNING)

# Create console handler with formatting
_handler: logging.Handler = logging.StreamHandler(sys.stderr)
_handler.setLevel(logging.DEBUG)  # Handler passes everything, logger filters

# Format: [LEVEL] polyinfer.module: message
_formatter = logging.Formatter(
    fmt="[%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_handler.setFormatter(_formatter)

# Add handler (only once)
if not _logger.handlers:
    _logger.addHandler(_handler)

# Prevent propagation to root logger
_logger.propagate = False


# Log level mapping
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "silent": logging.CRITICAL + 10,  # Above critical = silent
    "off": logging.CRITICAL + 10,
}


def get_logger(name: str = "") -> logging.Logger:
    """Get a polyinfer logger.

    Args:
        name: Logger name suffix. Empty string returns root polyinfer logger.
              "model" returns "polyinfer.model" logger.

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger("backends.onnxruntime")
        >>> logger.info("Loading model...")
    """
    if name:
        return logging.getLogger(f"polyinfer.{name}")
    return _logger


def set_log_level(level: str | int) -> None:
    """Set the global polyinfer log level.

    Args:
        level: Log level - can be string ("DEBUG", "INFO", "WARNING", "ERROR",
               "CRITICAL", "SILENT") or integer (10, 20, 30, 40, 50)

    Example:
        >>> pi.set_log_level("DEBUG")  # Most verbose
        >>> pi.set_log_level("WARNING")  # Default, warnings only
        >>> pi.set_log_level("SILENT")  # No output
    """
    if isinstance(level, str):
        level_lower = level.lower()
        if level_lower not in _LEVEL_MAP:
            valid = list(_LEVEL_MAP.keys())
            raise ValueError(f"Invalid log level: {level}. Valid: {valid}")
        level = _LEVEL_MAP[level_lower]

    _logger.setLevel(level)


def get_log_level() -> int:
    """Get the current global log level.

    Returns:
        Current log level as integer
    """
    return _logger.level


def get_log_level_name() -> str:
    """Get the current log level name.

    Returns:
        Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = _logger.level
    if level >= logging.CRITICAL + 10:
        return "SILENT"
    return logging.getLevelName(level)


def enable_logging(level: str | int = "INFO") -> None:
    """Enable logging with specified level.

    Convenience function to quickly enable verbose logging.

    Args:
        level: Log level (default: INFO)

    Example:
        >>> pi.enable_logging()  # Enable INFO level
        >>> pi.enable_logging("DEBUG")  # Enable DEBUG level
    """
    set_log_level(level)


def disable_logging() -> None:
    """Disable all logging output.

    Sets log level to SILENT (above CRITICAL).
    """
    set_log_level("silent")


def configure_logging(
    level: str | int = "WARNING",
    format: str = "[%(levelname)s] %(name)s: %(message)s",
    stream=None,
    filename: str | None = None,
) -> None:
    """Configure polyinfer logging with custom settings.

    Args:
        level: Log level
        format: Log message format string
        stream: Output stream (default: sys.stderr)
        filename: If provided, log to file instead of stream

    Example:
        >>> pi.configure_logging(
        ...     level="DEBUG",
        ...     format="%(asctime)s %(levelname)s %(message)s",
        ...     filename="polyinfer.log"
        ... )
    """
    global _handler, _formatter

    # Remove existing handlers
    for handler in _logger.handlers[:]:
        _logger.removeHandler(handler)
        handler.close()

    # Create new formatter
    _formatter = logging.Formatter(fmt=format)

    # Create new handler
    if filename:
        _handler = logging.FileHandler(filename)
    else:
        _handler = logging.StreamHandler(stream or sys.stderr)

    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)

    # Set level
    set_log_level(level)


class LogContext:
    """Context manager for temporarily changing log level.

    Example:
        >>> with pi.LogContext("DEBUG"):
        ...     model = pi.load("model.onnx")  # Verbose output
        >>> # Back to previous level
    """

    def __init__(self, level: str | int):
        self.new_level = level
        self.old_level = None

    def __enter__(self):
        self.old_level = get_log_level()
        set_log_level(self.new_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_log_level(self.old_level)
        return False


# Convenience loggers for common operations
def _log_model_load(backend: str, device: str, model_path: str, **kwargs):
    """Log model loading operation."""
    logger = get_logger("model")
    logger.info(f"Loading model: {model_path}")
    logger.debug(f"  Backend: {backend}")
    logger.debug(f"  Device: {device}")
    if kwargs:
        for key, value in kwargs.items():
            logger.debug(f"  {key}: {value}")


def _log_backend_init(name: str, version: str, devices: list):
    """Log backend initialization."""
    logger = get_logger("backends")
    logger.debug(f"Initialized backend: {name} v{version}")
    logger.debug(f"  Supported devices: {devices}")


def _log_inference(backend: str, input_shapes: list, output_shapes: list, time_ms: float | None = None):
    """Log inference operation."""
    logger = get_logger("inference")
    if time_ms is not None:
        logger.debug(f"Inference [{backend}]: {time_ms:.2f}ms")
    else:
        logger.debug(f"Inference [{backend}]: inputs={input_shapes}, outputs={output_shapes}")
