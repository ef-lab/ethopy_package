# Plugin System

The Ethopy plugin system provides a flexible way to extend the functionality by adding custom modules, behaviors, experiments, interfaces, and stimuli. The system supports both core modules and user plugins with intelligent conflict resolution.

## Plugin Categories

Ethopy supports two types of plugins:

1. **Standalone Modules**: Individual Python files in the plugin directory
2. **Categorized Plugins**: Modules organized in specific categories:
        - `behaviors`: Custom behavior implementations
        - `experiments`: Experiment definitions
        - `interfaces`: Hardware interface modules
        - `stimuli`: Stimulus control modules

## Plugin Locations

Plugins can be placed in the following locations (in order of precedence):

1. Default locations:
    - `~/.ethopy/ethopy_plugins/` (User's home directory)

2. Custom locations specified by the `ETHOPY_PLUGIN_PATH` environment variable:
   ```bash
   export ETHOPY_PLUGIN_PATH=/path/to/plugins,/another/plugin/path
   ```

The plugin directory structure should follow this pattern:

```
ethopy_plugins/
├── mymodule.py                    # Standalone module
├── another_module.py              # Another standalone module
├── behaviors/                     # Behavior plugins
│   └── custom_behavior.py
├── experiments/                   # Experiment plugins
│   └── custom_experiment.py
├── interfaces/                    # Interface plugins
│   └── custom_interface.py
└── stimuli/                      # Stimulus plugins
    └── custom_stimulus.py
```

## Creating Plugins

### Plugin Naming

Plugins are imported using the `ethopy` namespace. For example:
    - Standalone module: `ethopy.mymodule`
    - Categorized plugin: `ethopy.behaviors.custom_behavior`

Make sure to avoid naming conflicts with core Ethopy modules, as core modules take precedence over plugins.

### Standalone Modules

Create a Python file in the root of your plugin directory:

```python
# ~/.ethopy/ethopy_plugins/mymodule.py
class MyModule:
    def __init__(self):
        self.name = "My Custom Module"
    
    def do_something(self):
        return "Hello from MyModule!"
```

### Behavior Plugins

Create a Python file in the `behaviors` directory:

```python
# ~/.ethopy/ethopy_plugins/behaviors/custom_behavior.py
from ethopy.core.behavior import Behavior

class CustomBehavior(Behavior):
    def __init__(self):
        super().__init__()
        # Your initialization code
    
    def run(self):
        # Your behavior implementation
        pass
```

### Experiment Plugins

Create a Python file in the `experiments` directory:

```python
# ~/.ethopy/ethopy_plugins/experiments/custom_experiment.py
from ethopy.core.experiment import ExperimentClass, State

class CustomExperiment(State, ExperimentClass):
    def __init__(self):
        super().__init__()
        # Your initialization code
    
    def run(self):
        # Your experiment implementation
        pass
```

### Plugin Registration

Plugins are automatically discovered and registered when:
1. They are placed in a recognized plugin directory
2. The file name doesn't start with an underscore
3. The file has a `.py` extension

## Using Plugins

### Importing Plugins

Import and use plugins just like regular Ethopy modules:

```python
# Import standalone module
from ethopy.mymodule import MyModule

# Import behavior plugin
from ethopy.behaviors.custom_behavior import CustomBehavior

# Import experiment plugin
from ethopy.experiments.custom_experiment import CustomExperiment

# Use plugins
my_module = MyModule()
behavior = CustomBehavior()
experiment = CustomExperiment()
```

### Plugin Management

The plugin system is managed by the `PluginManager` class, which handles:
    - Plugin discovery and registration
    - Import path management
    - Conflict resolution
    - Plugin information tracking

```python
from ethopy.plugin_manager import PluginManager

# Create plugin manager instance
plugin_manager = PluginManager()

# Add custom plugin path
plugin_manager.add_plugin_path('/path/to/plugins')

# List available plugins
plugins = plugin_manager.list_plugins(
    show_duplicates=True,  # Show duplicate plugin information
    include_core=True      # Include core Ethopy modules
)

# Print plugin information
for category, items in plugins.items():
    print(f"\n{category} plugins:")
    for plugin in items:
        print(f"  - {plugin['name']} ({plugin['path']})")
        if 'duplicates' in plugin:
            print("    Duplicate versions found in:")
            for dup in plugin['duplicates']:
                print(f"      - {dup}")

# Get information about a specific plugin
info = plugin_manager.get_plugin_info('ethopy.mymodule')
if info:
    print(f"Plugin: {info.name}")
    print(f"Path: {info.path}")
    print(f"Type: {info.type}")
    print(f"Is Core: {info.is_core}")
```

## Plugin Resolution

### Load Order

Plugins are loaded in the following order:
1. Core Ethopy modules (from main package)
2. Default plugin directories
3. Custom plugin paths from environment variable

### Conflict Resolution

The plugin system uses the following precedence rules:

1. **Core vs Plugin Conflicts**:
        - Core Ethopy modules always take precedence over plugins
        - Warning is issued when a plugin conflicts with a core module

2. **Plugin vs Plugin Conflicts**:
        - Later added paths take precedence over earlier ones
        - Warning is displayed showing which version is used/ignored

Example conflict warning:
```
WARNING: Plugin 'ethopy.mymodule' from /path/to/plugin conflicts with core ethopy module. Core module will be used.

WARNING: Duplicate plugin found for 'ethopy.behaviors.custom':
  Using:     /home/user/.ethopy/ethopy_plugins/behaviors/custom.py
  Ignoring:  /another/path/behaviors/custom.py
```

## Best Practices

### Plugin Development

1. **Namespace Awareness**:
        - Avoid using names that conflict with core Ethopy modules
        - Use descriptive, unique names for your plugins
        - Follow Python naming conventions

2. **Structure and Organization**:
        - Place plugins in the correct category directory
        - Use appropriate base classes for each plugin type
        - Keep plugin files focused and single-purpose

3. **Documentation**:
        - Add docstrings to your plugin classes and methods
        - Include usage examples in the documentation
        - Document any special requirements or dependencies

4. **Error Handling**:
        - Implement proper error handling in your plugins
        - Provide meaningful error messages
        - Handle resource cleanup properly

### Plugin Distribution

1. **Dependencies**:
        - Clearly specify any additional dependencies
        - Use standard Python package management
        - Test with different Python versions

2. **Version Control**:
        - Use version control for your plugins
        - Tag releases with version numbers
        - Maintain a changelog

3. **Testing**:
        - Write tests for your plugins
        - Test integration with Ethopy
        - Verify behavior with different configurations

## Troubleshooting

### Common Issues

1. **Plugin Not Found**:
        - Verify plugin directory location
        - Check file permissions
        - Ensure correct Python path
        - Validate plugin file naming

2. **Import Errors**:
        - Check for missing dependencies
        - Verify Python version compatibility
        - Look for syntax errors in plugin code
        - Check for circular imports

3. **Plugin Conflicts**:
        - Review plugin naming for conflicts
        - Check plugin path order
        - Examine duplicate warnings
        - Verify core module conflicts

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.getLogger('ethopy').setLevel(logging.DEBUG)
   ```

2. **Check Plugin Registration**:
   ```python
   # List all registered plugins
   plugins = plugin_manager.list_plugins(show_duplicates=True)
   
   # Check specific plugin
   info = plugin_manager.get_plugin_info('ethopy.mymodule')
   if info:
       print(f"Plugin registered at: {info.path}")
       print(f"Plugin type: {info.type}")
   else:
       print("Plugin not registered")
   ```

3. **Verify Plugin Paths**:
   ```python
   # Print current plugin search paths
   print("Plugin paths:")
   for path in plugin_manager._plugin_paths:
       print(f"- {path}")
   ```

## Additional Resources

1. **Documentation**:
        - [Ethopy Core Documentation](https://ef-lab.github.io/ethopy_package/)
        - [DataJoint Documentation](https://docs.datajoint.org/)
        - [Python Packaging Guide](https://packaging.python.org/guides/distributing-packages-using-setuptools/)

2. **Community**:
        - [GitHub Issues](https://github.com/ef-lab/ethopy_package/issues)
        - [Contributing Guidelines](https://github.com/ef-lab/ethopy_package/blob/main/CONTRIBUTING.md)
