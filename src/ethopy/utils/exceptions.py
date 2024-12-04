import logging
from typing import Optional


class LoggingError(Exception):
    """
    Base exception class for ethopy that includes logging functionality.
    All custom exceptions in ethopy should inherit from this class.

    This provides consistent error handling and logging across the package.

    Attributes:
        message -- explanation of the error
        details -- optional additional error details
        logger_name -- name of the logger to use (defaults to 'ethopy')
    """

    def __init__(
        self, message: str, details: Optional[str] = None, logger_name: str = "ethopy"
    ):
        """
        Initialize the exception with message, optional details, and logging.

        Args:
            message: Main error message explaining what went wrong
            details: Optional additional details about the error
            logger_name: Name of the logger to use
        """
        self.message = message
        self.details = details
        self.logging = logging.getLogger(logger_name)

        # Log the error
        self._log_error()

        # Call parent class constructor with formatted message
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the complete error message including details if available."""
        if self.details:
            return f"{self.__class__.__name__}: {self.message}\nDetails: {self.details}"
        return f"{self.__class__.__name__}: {self.message}"

    def _log_error(self) -> None:
        """Log the error message and details."""
        self.logging.error(self.format_message())


class ConfigurationError(LoggingError):
    """
    Exception raised for errors in the configuration system.

    This exception is used when:
    - Required configuration values are missing
    - Configuration files cannot be loaded or parsed
    - Invalid configuration values are provided
    - Configuration paths are invalid or inaccessible
    """

    pass

