# Configuration Manager Documentation

The ConfigurationManager provides a unified way to manage local configurations in your ethopy project across all operating systems. All configurations are stored in a consistent location within the user's home directory.

## Configuration Location

All configuration files are stored in a `.ethopy` directory in your home folder:

- Windows: `C:\Users\<username>\.ethopy\local_conf.json`
- Linux/macOS: `/home/<username>/.ethopy/local_conf.json`

## Basic Usage

### Initializing the Configuration Manager

```python
from ethopy.core.config import ConfigurationManager

# Basic initialization
config = ConfigurationManager()

# Initialize with existing configuration
config = ConfigurationManager(import_config="path/to/your/local_conf.json")
```

### Configuration Structure

The configuration file uses JSON format with this structure:

```json
{
    "SCHEMATA": {
        "experiment": "lab_experiments",
        "stimulus": "lab_stimuli",
        "behavior": "lab_behavior",
        "recording": "lab_recordings",
        "mice": "lab_mice"
    },
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "your_password",
        "database.port": 3306,
        "database.reconnect": true,
        "database.use_tls": false,
        "datajoint.loglevel": "WARNING"
    },
    "source_path": "~/ethopy_data",
    "target_path": "/path to transfer data after completion",
}
```

## Working with Configurations

### Database Settings

```python
# Access database settings
db_host = config.db.host
db_user = config.db.user
db_port = config.db.port

# Get DataJoint-compatible configuration
dj_config = config.db.to_dict
```

### Schema Settings

```python
# Access schema names
experiment_schema = config.schema.experiment
behavior_schema = config.schema.behavior
```

### Managing Paths

```python
# Access basic paths
source_path = config.paths.source_path
target_path = config.paths.target_path

# Add custom paths (automatically creates directories)
config.paths.add_path('video_path', '~/ethopy_data/videos')

# Get custom paths
video_path = config.paths.get_path('video')
```

### Custom Parameters

```python
# Add custom parameters
config.add_custom_param('camera_id', 'CAM01')
config.add_custom_param('frame_rate', 30)
config.add_custom_param('settings', {
    'exposure': 100,
    'gain': 1.5
})

# Access custom parameters
camera_id = config.get_custom_param('camera_id')
frame_rate = config.get_custom_param('frame_rate', default=25)
```

## Configuration Management

### Importing Configuration

You can import an existing configuration file:

```python
# Method 1: During initialization
config = ConfigurationManager(import_config="path/to/local_conf.json")

# Method 2: After initialization
config = ConfigurationManager()
config.import_configuration("path/to/local_conf.json")
```

### Saving Configuration

Save your current configuration:

```python
config.save_configuration()
```

## Environment Variables

Override default settings using environment variables:

Windows (PowerShell):
```powershell
$env:ETHOPY_CONFIG = "C:\custom\path\local_conf.json"
$env:ETHOPY_DB_PASSWORD = "your_password"
```

Linux/macOS:
```bash
export ETHOPY_CONFIG="/custom/path/local_conf.json"
export ETHOPY_DB_PASSWORD="your_password"
```

## Error Handling

The ConfigurationManager uses custom exceptions for error handling:

```python
from ethopy.exceptions import ConfigurationError

try:
    config = ConfigurationManager()
    config.import_configuration("non_existent.json")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Complete Setup Example

Here's a complete example of setting up a configuration:

```python
from pathlib import Path
from ethopy.core.config import ConfigurationManager

def setup_experiment_config():
    # Initialize configuration manager
    config = ConfigurationManager()
    
    # Set up experiment paths
    base_path = Path.home() / "ethopy_data"
    config.paths.add_path('video', base_path / "videos")
    config.paths.add_path('interface', base_path / "interfaces")
    config.paths.add_path('calibration', base_path / "calibration")
    
    # Save configuration
    config.save_configuration()
    
    return config

if __name__ == "__main__":
    config = setup_experiment_config()
    print(f"Configuration saved to: {config.CONFIG_PATH}")
```

## Best Practices

1. **Path Management**
   - Use relative paths when possible
   - Use Path objects for path manipulation
   - Use the `~` shortcut for home directory paths

2. **Configuration Files**
   - Keep a template configuration in version control
   - Exclude actual configuration files with sensitive data
   - Use environment variables for sensitive information

3. **Cross-Platform Compatibility**
   - Always use Path objects for paths
   - Use os.path.join() or Path.joinpath() for path construction
   - Avoid hardcoded path separators

## Troubleshooting

Common issues and solutions:

1. **Configuration Not Found**
   - Check if `.ethopy` directory exists in your home folder
   - Verify file permissions
   - Check if the path in ETHOPY_CONFIG environment variable is correct

2. **Permission Issues**
   - Ensure you have write permission to your home directory
   - Check file permissions on the configuration file

3. **Invalid Configuration**
   - Verify JSON syntax
   - Check for missing required fields
   - Ensure all paths use correct separators for your OS

4. **Path Issues**
   - Use forward slashes (/) in paths, even on Windows
   - Use absolute paths when having issues with relative paths
   - Verify all referenced directories exist

## Getting Help

If you encounter issues:

1. Check the logs (ConfigurationManager uses Python's logging system)
2. Verify your configuration file structure
3. Ensure all paths exist and are accessible
4. Check file permissions
5. Verify environment variables are set correctly