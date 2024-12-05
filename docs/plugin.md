# Plugin System

The ethopy plugin system allows users to extend the functionality of ethopy by adding custom modules, behaviors, experiments, interfaces, and stimuli. This guide explains how to create and use plugins.

## Directory Structure

Plugins can be placed in any of these locations:

1. User's home directory: `~/ethopy_plugins/`
2. Current working directory: `./ethopy_plugins/`
3. Custom locations specified by the `ETHOPY_PLUGIN_PATH` environment variable

The plugin directory can contain both standalone modules and categorized plugins:

```
ethopy_plugins/
├── mymodule.py                    # Standalone module
├── another_module.py              # Another standalone module
├── Behaviors/                     # Behavior plugins
│   └── custom_behavior.py
├── Experiments/                   # Experiment plugins
│   └── custom_experiment.py
├── Interfaces/                    # Interface plugins
│   └── custom_interface.py
└── Stimuli/                      # Stimulus plugins
    └── custom_stimulus.py
```

## Creating Plugins

### Standalone Modules

Create a Python file in the root of your plugin directory:

```python
# ~/ethopy_plugins/mymodule.py
class MyModule:
    def __init__(self):
        self.name = "My Custom Module"
    
    def do_something(self):
        return "Hello from MyModule!"
```

### Behavior Plugins

Create a Python file in the `Behaviors` directory:

```python
# ~/ethopy_plugins/Behaviors/custom_behavior.py
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

Create a Python file in the `Experiments` directory:

```python
# ~/ethopy_plugins/Experiments/custom_experiment.py
from ethopy.core.experiment import ExperimentClass, State

class Experiment(State, ExperimentClass):
    def __init__(self):
        super().__init__()
        # Your initialization code
    
    def run(self):
        # Your experiment implementation
        pass
```

## Using Plugins

Import and use plugins just like regular ethopy modules:

```python
# Import standalone module
from ethopy.mymodule import MyModule

# Import behavior plugin
from ethopy.Behaviors.custom_behavior import CustomBehavior

# Use standalone module
my_module = MyModule()
print(my_module.do_something())

# Use behavior plugin
behavior = CustomBehavior()
```

## Plugin Discovery

### Setting Plugin Paths

You can add plugin directories in several ways:

1. Environment variable:
```bash
export ETHOPY_PLUGIN_PATH=/path/to/plugins,/another/plugin/path
```

2. Programmatically:
```python
from ethopy.core.plugin_manager import plugin_manager
plugin_manager.add_plugin_path('/path/to/plugins')
```

### Listing Available Plugins

```python
from ethopy.core.plugin_manager import plugin_manager

# List all plugins with duplicate information
plugins = plugin_manager.list_plugins(show_duplicates=True)

# Print plugin information
for category, items in plugins.items():
    print(f"\n{category} plugins:")
    for plugin in items:
        print(f"  - {plugin['name']} ({plugin['path']})")
        if 'duplicates' in plugin:
            print("    Duplicate versions found in:")
            for dup in plugin['duplicates']:
                print(f"      - {dup}")
```

## Duplicate Handling

The plugin system handles duplicates with the following precedence rules:

1. Core ethopy modules take precedence over plugins with the same name
2. For conflicts between plugins:
   - Later added paths take precedence over earlier ones
   - Environment variable paths override default paths

When duplicates are found, warnings are displayed:

```
WARNING: Duplicate plugin found for 'ethopy.mymodule':
  Using:     /home/user/ethopy_plugins/mymodule.py
  Ignoring:  /current/dir/ethopy_plugins/mymodule.py
```

## Best Practices

1. **Avoid Used Names**: Don't create plugins with the same names as ethopy modules
2. **Use Consistent Structure**: Follow the directory structure for different plugin types
3. **Clear Naming**: Use descriptive names for your plugins to avoid conflicts
4. **Inheritance**: Extend appropriate base classes for behaviors, experiments, etc.
5. **Documentation**: Document your plugins with docstrings and comments

## Common Issues

1. **Plugin Not Found**: Check if the plugin directory is in the correct location
2. **Import Errors**: Ensure all dependencies are installed
3. **Duplicate Warnings**: Review plugin paths for unintended duplicates
4. **Core Conflicts**: Avoid using names that match core ethopy modules

## Plugin Development Tips

1. Use the plugin manager to check for existing plugins:
```python
plugins = plugin_manager.list_plugins()
print("Existing plugins:", plugins)
```

2. Get information about specific plugins:
```python
info = plugin_manager.get_plugin_info('ethopy.mymodule')
if info:
    print(f"Plugin: {info.name}")
    print(f"Path: {info.path}")
    print(f"Type: {info.type}")
```

3. During development, you can reload plugins:
```python
plugin_manager.reload_plugins()
```

## Examples
