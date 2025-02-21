# Logging in EthoPy

EthoPy provides a comprehensive logging system that handles both file and console output with configurable formats and levels. The logging system is centrally managed and provides consistent logging across all modules of the package.

## Features

- Rotating file logs with size limits
- Colored console output
- Different formats for different log levels
- Centralized configuration
- Automatic log directory creation
- Multiple backup log files

## Configuration

### Default Settings

```python
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "ethopy.log"
MAX_LOG_SIZE = 30 * 1024 * 1024  # 30 MB
LOG_BACKUP_COUNT = 5
```

### Local conf setting based on the local_conf.json
Logging is set up based on the parameters defined in the local_conf.json
```json
    "logging": {
        "level": "INFO",
        "directory": "~/.ethopy/",
        "filename": "ethopy.log"
    }
```

### Command Line Options

When running EthoPy from the command line, you can configure logging using these options:

```bash
ethopy [OPTIONS]
Options:
  --log-console        Enable console logging
  --log-level TEXT     Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
```


## Log Formats

### File Logs
All file logs use the detailed format:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)
```

Example:
```
2024-01-20 10:15:30 - ethopy - INFO - Experiment started (experiment.py:145)
```

### Console Logs

Console logs use two different formats based on the log level:

1. **Simple Format** (for INFO and DEBUG):
```
%(asctime)s - %(levelname)s - %(message)s
```

2. **Detailed Format** (for WARNING and above):
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)
```

### Color Coding

Console output is color-coded by log level:

- DEBUG: Grey
- INFO: Grey
- WARNING: Yellow
- ERROR: Red
- CRITICAL: Bold Red

## Log File Management

### Rotation

Log files are automatically rotated when they reach the maximum size:

- Maximum file size: 30 MB
- Number of backup files: 5
- Naming convention: ethopy.log, ethopy.log.1, ethopy.log.2, etc.

### Directory Structure

```
logs/
├── ethopy.log          # Current log file
├── ethopy.log.1        # Most recent backup
├── ethopy.log.2        # Second most recent backup
└── ...
```

## Usage Examples

### Basic Logging

```python
import logging

# Log messages at different levels
logging.debug("Detailed debug information")
logging.info("General information")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical error")
```

### Custom Logger

```python
import logging

# Create a logger for your module
logger = logging.getLogger(__name__)

# Use the logger
logger.info("Module specific information")
logger.error("Module specific error")
```

## Best Practices

1. **Log Level Selection**
            - Use DEBUG for detailed debugging information
            - Use INFO for general operational messages
            - Use WARNING for unexpected but handled situations
            - Use ERROR for errors that affect functionality
            - Use CRITICAL for errors that require immediate attention

2. **Message Content**
            - Include relevant context in log messages
            - Be specific about what happened
            - Include important variable values
            - Avoid logging sensitive information

3. **Performance Considerations**
            - Avoid logging in tight loops
            - Use appropriate log levels to control output volume
            - Consider log rotation settings for long-running applications

## Implementation Details

### LoggingManager Class

The `LoggingManager` class handles all logging configuration:

```python
from ethopy.utils.ethopy_logging import LoggingManager

# Create a manager instance
manager = LoggingManager("your_module_name")

# Configure logging
manager.configure(
    log_dir="logs",
    console=True,
    log_level="INFO",
    log_file="app.log"
)
```

### Custom Formatter

The logging system includes a custom formatter that provides:

- Color-coded output for different log levels
- Dynamic format selection based on log level
- Timestamp formatting
- File and line number information for warnings and errors

## Troubleshooting

1. **Missing Logs**
            - Check write permissions for log directory
            - Verify log level configuration
            - Ensure log directory exists

2. **Console Output Issues**
            - Verify console logging is enabled
            - Check terminal color support
            - Confirm log level settings

3. **Performance Issues**
            - Review log rotation settings
            - Check logging frequency
            - Consider adjusting log levels