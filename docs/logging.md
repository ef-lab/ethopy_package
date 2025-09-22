# Logging in EthoPy

EthoPy uses Python's standard logging system with centralized configuration. All modules automatically get properly formatted logging with the option for both file and console output.

## How to Use Logging

### Command Line Options

When running EthoPy from the command line, you can configure logging using these options:

```bash
ethopy [OPTIONS]
Options:
  --log-console        Enable console logging
  --log-level TEXT     Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
```

### In Your Module

```python
import logging

# Create a logger for your module
log = logging.getLogger(__name__)

# Use the logger
log.info("Module operation completed")
log.warning("Potential issue detected")
log.error("Operation failed")
log.debug("Detailed debugging info")
```

### How `logging.getLogger(__name__)` Works

Each module gets its own named logger based on the module path:

```python
# In src/ethopy/core/experiment.py
log = logging.getLogger(__name__)
# Creates logger: "ethopy.core.experiment"

# In src/ethopy/interfaces/Arduino.py
log = logging.getLogger(__name__)
# Creates logger: "ethopy.interfaces.Arduino"
```

**Benefits:**

- **Module identification**: Log messages show which module they came from
- **Automatic configuration**: All loggers inherit the same formatting and handlers
- **Hierarchical structure**: Organized by module structure

## Log Output Formats

### Console Output
- **INFO/DEBUG**: `2024-01-20 10:15:30 - INFO - Session started`
- **WARNING+**: `2024-01-20 10:15:30 - ethopy.core.experiment - WARNING - Task not found (experiment.py:145)`
- **Colors**: Grey (INFO/DEBUG), Yellow (WARNING), Red (ERROR), Bold Red (CRITICAL)

### File Output
All levels use detailed format:
```
2024-01-20 10:15:30 - ethopy.interfaces.Arduino - INFO - Connection established (Arduino.py:67)
```

## Configuration

### Automatic Setup
Logging is automatically configured when EthoPy starts. No manual setup needed in your modules.

### Log Files

- **Location**: `logs/ethopy.log`
- **Rotation**: 30MB max size, 5 backup files
- **Files**: `ethopy.log`, `ethopy.log.1`, `ethopy.log.2`, etc.

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General operational messages
- **WARNING**: Unexpected but handled situations
- **ERROR**: Errors that affect functionality
- **CRITICAL**: Errors requiring immediate attention

### Configure settings with the local_conf.json
Logging is set up based on the parameters defined in the local_conf.json
```json
    "logging": {
        "level": "INFO",
        "directory": "~/.ethopy/",
        "filename": "ethopy.log",
        "max_size": 31457280,
        "backup_count": 5
    }
```

## Best Practices

1. **Always use** `log = logging.getLogger(__name__)` in each module
2. **Include context** in log messages (variable values, state info)
3. **Use appropriate levels** - INFO for normal operations, WARNING for issues, ERROR for failures
4. **Don't log sensitive information** (passwords, keys, personal data)

## Technical Details

The logging system uses:

- **LoggingManager** class in `src/ethopy/utils/ethopy_logging.py`
- **Root logger configuration** that all module loggers inherit from
- **RotatingFileHandler** for automatic log file management
- **CustomFormatter** for dynamic console formatting and colors

All configuration is handled automatically - just use `logging.getLogger(__name__)` in your modules.
