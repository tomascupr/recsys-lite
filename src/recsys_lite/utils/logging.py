"""Centralized logging configuration for RecSys-Lite."""

import logging
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# Define standard log levels with descriptive names
class LogLevel(str, Enum):
    """Log level enum for RecSys-Lite."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Mapping from LogLevel enum to logging module levels
LOG_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


def configure_logging(
    level: Union[LogLevel, str] = LogLevel.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """Configure logging for RecSys-Lite.

    Args:
        level: Log level (default: INFO)
        log_file: Optional file path to write logs to
        log_format: Optional custom log format
    """
    # Convert string to enum if needed
    if isinstance(level, str) and not isinstance(level, LogLevel):
        try:
            level = LogLevel(level.upper())
        except ValueError:
            level = LogLevel.INFO

    # Get the numeric log level
    numeric_level = LOG_LEVEL_MAP.get(level, logging.INFO)

    # Define default log format if not provided
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    handlers: List[logging.Handler] = []

    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # Add file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance
    """
    # Ensure all loggers are prefixed with recsys-lite
    if not name.startswith("recsys-lite"):
        if name != "__main__":
            name = f"recsys-lite.{name}"
        else:
            name = "recsys-lite"

    return logging.getLogger(name)


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Optional[Exception] = None,
    level: LogLevel = LogLevel.ERROR,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an exception with consistent formatting.

    Args:
        logger: Logger instance
        message: Error message
        exc: Exception object (if available)
        level: Log level (default: ERROR)
        extra: Additional context to include in the log
    """
    log_method = getattr(logger, level.lower())

    if exc is not None:
        if level == LogLevel.ERROR or level == LogLevel.CRITICAL:
            logger.exception(f"{message}: {exc}", extra=extra)
        else:
            log_method(f"{message}: {exc}", extra=extra)
    else:
        log_method(message, extra=extra)


# Initialize package-level logger
logger = get_logger(__name__)
