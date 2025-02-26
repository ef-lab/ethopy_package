"""Central logging configuration for the ethopy package.

This module provides a consistent logging setup across all ethopy modules,
supporting both file and console output with appropriate formatting.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class LoggingManager:
    """Manages centralized logging configuration for ethopy package.

    This class provides a unified logging configuration that can be used
    across all modules of the ethopy package. It supports both file and
    console logging with different formats and log levels.

    Features:
    - Rotating file handler with size limits
    - Optional colored console output
    - Different formats for different log levels
    - Single configuration point for all modules
    """

    DEFAULT_LOG_DIR = "logs"
    DEFAULT_LOG_FILE = "ethopy.log"
    MAX_LOG_SIZE = 30 * 1024 * 1024  # 30 MB
    LOG_BACKUP_COUNT = 5

    def __init__(self, name):
        self._configured = False
        # Keep track of root logger
        self._root_logger = logging.getLogger()
        # Keep track of ethopy logger
        self._logger = logging.getLogger(name)

    def configure(
        self,
        log_dir: Optional[str] = None,
        console: bool = True,
        log_level: str = "INFO",
        log_file: str = None,
    ) -> None:
        """Configure logging for the entire application."""
        if self._configured:
            return

        # First, remove ALL existing handlers from root logger
        self._root_logger.handlers.clear()

        # Remove any existing handlers from our logger
        self._logger.handlers.clear()

        # Convert log level string to logging constant
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Setup log directory
        log_dir = log_dir or self.DEFAULT_LOG_DIR
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Setup log file
        log_file = log_file or self.DEFAULT_LOG_FILE
        log_file_path = log_path / log_file

        # Configure root logger
        self._root_logger.setLevel(numeric_level)

        # Add file handler to root logger
        file_handler = self._create_file_handler(log_file_path, numeric_level)
        self._root_logger.addHandler(file_handler)

        # Add console handler if enabled to root logger
        if console:
            console_handler = self._create_console_handler(numeric_level)
            self._root_logger.addHandler(console_handler)

        self._configured = True

        # Log initial message
        logging.info(
            "Logging system initialized: level=%s, file=%s", log_level, log_file_path
        )

    def _create_file_handler(
        self, log_file: Path, log_level: int
    ) -> RotatingFileHandler:
        """Create and configure the file handler.

        Args:
            log_file: Path to the log file
            log_level: Logging level for the handler

        Returns:
            Configured RotatingFileHandler instance
        """
        handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=self.MAX_LOG_SIZE,
            backupCount=self.LOG_BACKUP_COUNT,
        )
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        return handler

    def _create_console_handler(self, log_level: int) -> logging.StreamHandler:
        """Create and configure the console handler.

        Args:
            log_level: Logging level for the handler

        Returns:
            Configured StreamHandler instance with color formatting
        """
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(CustomFormatter())
        return handler


class CustomFormatter(logging.Formatter):
    """Custom formatter providing colored console output."""

    COLORS = {
        logging.DEBUG: "\x1b[38;21m",  # grey
        logging.INFO: "\x1b[38;21m",  # grey
        logging.WARNING: "\x1b[33;21m",  # yellow
        logging.ERROR: "\x1b[31;21m",  # red
        logging.CRITICAL: "\x1b[31;1m",  # bold red
    }
    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__()
        self.detailed_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        self.simple_fmt = "%(asctime)s - %(levelname)s - %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with appropriate color and format."""
        # Use detailed format for warnings and above
        if record.levelno >= logging.WARNING:
            fmt = self.detailed_fmt
        else:
            fmt = self.simple_fmt

        # Add color based on level
        color = self.COLORS.get(record.levelno, self.COLORS[logging.INFO])
        formatter = logging.Formatter(
            f"{color}{fmt}{self.RESET}", datefmt="%Y-%m-%d %H:%M:%S"
        )
        return formatter.format(record)


# Global manager instance
_manager = LoggingManager("ethopy")


def setup_logging(
    log_dir: Optional[str] = None,
    console: bool = True,
    log_level: str = "INFO",
    log_file: str = None,
) -> None:
    """
    Initialize logging for the ethopy package.
    Uses the LoggingManager with custom formatting for both file and console output.

    Args:
        log_dir: Directory for log files. Created if doesn't exist.
                Defaults to 'logs' in current directory.
        console: Whether to enable console output. Defaults to True.
        log_level: Minimum log level. Defaults to "INFO".
        log_file: Name of the log file. Defaults to "ethopy.log".
    """
    global _manager
    _manager.configure(
        log_dir=log_dir, console=console, log_level=log_level, log_file=log_file
    )
