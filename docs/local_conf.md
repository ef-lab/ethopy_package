# EthoPy Configuration Guide

## Quick Start

```python
from ethopy.core.config import ConfigurationManager

# Initialize with default configuration
config = ConfigurationManager()

# Get a configuration value
db_host = config.get('database.host')

# Set a configuration value
config.set('logging.level', 'DEBUG')

# Save changes
config.save()
```

## Configuration File Location

The configuration file is stored in:
- Linux/macOS: `~/.ethopy/local_conf.json`
- Windows: `%USERPROFILE%\.ethopy\local_conf.json`

## Basic Configuration Structure

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "your_password",
        "database.port": 3306
    },
    "source_path": "/path/to/data",
    "target_path": "/path/to/backup",
    "logging": {
        "level": "INFO",
        "directory": "~/.ethopy/",
        "filename": "ethopy.log"
    }
}
```

## Using the Configuration Manager

### Basic Operations

```python
# Initialize
config = ConfigurationManager()

# Get values (with optional default)
db_host = config.get('database.host', 'localhost')
log_level = config.get('logging.level', 'INFO')

# Set values
config.set('database.password', 'new_password')

# Save changes
config.save()

# Get complete DataJoint config
dj_config = config.get_datajoint_config()
```

### Working with Paths

```python
# Get standard paths
source = config.get('source_path')
target = config.get('target_path')

# Create directories automatically
from pathlib import Path
Path(source).mkdir(parents=True, exist_ok=True)
```

## Advanced Usage

### Environment Variables Override

Set environment variables to override configuration:

```bash
export ETHOPY_DB_PASSWORD="secret"
export ETHOPY_SOURCE_PATH="/custom/path"
```

### Custom Configuration File

```python
config = ConfigurationManager(config_file="custom_config.json")
```

### Configuration Validation

The ConfigurationManager automatically validates and sets defaults for:
- Database connection settings
- Schema names
- Logging configuration
- Required paths

## Configuration Sections

### Database Settings (`dj_local_conf`)

Essential settings for DataJoint database connection:

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "password",
        "database.port": 3306,
        "database.reconnect": true,
        "database.use_tls": false,
        "datajoint.loglevel": "WARNING"
    }
}
```

### Schema Mapping (`SCHEMATA`)

Maps internal schema names to database schemas:

```json
{
    "SCHEMATA": {
        "experiment": "lab_experiments",
        "stimulus": "lab_stimuli",
        "behavior": "lab_behavior",
        "recording": "lab_recordings",
        "mice": "lab_mice"
    }
}
```

### Logging Configuration

```json
{
    "logging": {
        "level": "INFO",
        "directory": "~/.ethopy/",
        "filename": "ethopy.log",
        "max_size": 31457280,
        "backup_count": 5
    }
}
```

### Hardware Channel Configuration (`channels`)

The `channels` configuration defines GPIO pin mappings for various hardware components. This is crucial for experiments that involve physical hardware interaction, particularly on Raspberry Pi systems.

```json
{
    "channels": {
        "Odor": {"1": 24, "2": 25},     // Odor delivery valves
        "Liquid": {"1": 22, "2": 23},    // Liquid reward valves
        "Lick": {"1": 17, "2": 27},      // Lick detection sensors
        "Proximity": {"3": 9, "1": 5, "2": 6},  // Proximity sensors
        "Sound": {"1": 13},              // Sound output
        "Sync": {                        // Synchronization pins
            "in": 21, 
            "rec": 26, 
            "out": 16
        },
        "Opto": 19,                      // Optogenetics control
        "Status": 20                     // Status LED
    }
}
```

#### Channel Types and Their Uses

1. **Odor Channels**
    - Purpose: Control odor delivery valves
    - Format: `{"port_number": gpio_pin}`
    - Example: `"1": 24` maps odor port 1 to GPIO pin 24

2. **Liquid Channels**
    - Purpose: Control water/liquid reward delivery
    - Format: `{"port_number": gpio_pin}`
    - Example: `"1": 22` maps liquid port 1 to GPIO pin 22

3. **Lick Channels**
    - Purpose: Detect animal licking behavior
    - Format: `{"port_number": gpio_pin}`
    - Example: `"1": 17` maps lick detector 1 to GPIO pin 17

4. **Proximity Channels**
    - Purpose: Detect animal presence/position
    - Format: `{"port_number": gpio_pin}`
    - Example: `"3": 9` maps proximity sensor 3 to GPIO pin 9

5. **Sound Channel**
    - Purpose: Control audio output
    - Format: `{"port_number": gpio_pin}`
    - Example: `"1": 13` maps sound output to GPIO pin 13

6. **Sync Channels**
    - Purpose: Synchronization with external devices
    - Components:
        - `"in"`: Input synchronization signal
        - `"rec"`: Recording trigger
        - `"out"`: Output synchronization signal

7. **Special Channels**
    - `"Opto"`: Single pin for optogenetics control
    - `"Status"`: LED indicator for system status

#### Hardware Setup Considerations

1. **Pin Conflicts**
    - Ensure no GPIO pin is assigned to multiple channels
    - Check pin capabilities (input/output/PWM support)
    - Avoid system-reserved pins

2. **Safety Checks**
   ```python
   # Example validation
   pins_used = set()
   for device, pins in config.get('channels').items():
       if isinstance(pins, dict):
           for pin in pins.values():
               if pin in pins_used:
                   raise ValueError(f"Pin {pin} used multiple times")
               pins_used.add(pin)
   ```

3. **Power Requirements**
    - Consider current limitations of GPIO pins
    - Use appropriate hardware drivers for high-power devices
    - Include safety resistors where needed

#### Common Hardware Configurations

1. **Basic Setup (2 ports)**
   ```json
   {
       "Liquid": {"1": 22, "2": 23},
       "Lick": {"1": 17, "2": 27}
   }
   ```

2. **Advanced Setup (with all features)**
   ```json
   {
       "Liquid": {"1": 22, "2": 23},
       "Lick": {"1": 17, "2": 27},
       "Proximity": {"1": 5, "2": 6},
       "Odor": {"1": 24, "2": 25},
       "Sound": {"1": 13},
       "Sync": {"in": 21, "rec": 26, "out": 16},
       "Opto": 19,
       "Status": 20
   }
   ```

### Paths Configuration

```json
{
    "source_path": "/path/to/data/source",
    "target_path": "/path/to/data/target",
    "plugin_path": "/path/to/plugins"
}
```

## Best Practices

1. **Security**
    - Never commit configuration files with sensitive data
    - Use environment variables for passwords
    - Keep backups of your configuration

2. **Path Management**
    - Use absolute paths when possible
    - Ensure write permissions for logs/data
    - Regularly check available disk space

3. **Error Handling**
    - Always check if paths exist before operations
    - Handle missing configuration values gracefully
    - Log configuration changes

## Common Issues and Solutions

### Database Connection Issues

Problem: Cannot connect to database
```python
# Check connection settings
config = ConfigurationManager()
print(config.get_datajoint_config())

# Verify database is reachable
import socket
s = socket.socket()
try:
    s.connect((config.get('database.host'), config.get('database.port')))
    print("Database reachable")
except Exception as e:
    print(f"Cannot reach database: {e}")
```

### Path Permission Issues

Problem: Cannot write to paths
```python
# Check path permissions
from pathlib import Path

path = Path(config.get('source_path'))
if not path.exists():
    print(f"Path does not exist: {path}")
elif not os.access(path, os.W_OK):
    print(f"Cannot write to path: {path}")
```

## Future Improvements

The configuration system is continually being improved. Planned features include:

1. Schema validation using Pydantic
2. Configuration migration tools
3. GUI configuration editor

## Contributing

To contribute to the configuration system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Testing Your Configuration

```python
from ethopy.core.config import ConfigurationManager

def test_config():
    config = ConfigurationManager()
    
    # Test database connection
    dj_config = config.get_datajoint_config()
    assert all(k in dj_config for k in ['database.host', 'database.user'])
    
    # Test paths
    assert Path(config.get('source_path')).exists()
    
    # Test logging
    assert config.get('logging.level') in ['DEBUG', 'INFO', 'WARNING', 'ERROR']

if __name__ == '__main__':
    test_config()
```