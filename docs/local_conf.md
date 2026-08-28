# EthoPy Configuration Guide

## What is local_conf.json?

The `local_conf.json` file stores **device-specific settings** that are unique to each computer or experimental setup. These are settings that:

- Are tied to your specific hardware (file paths, GPIO pins, database credentials)
- Need to be configured once per machine, not per experiment

**Key distinction**: Experimental parameters and data go in the database. Machine-specific settings go in `local_conf.json`.

## How EthoPy is Configured

EthoPy reads its settings from **one JSON configuration file**, plus a small number of overrides that apply on top of it. There are four mechanisms in total:

| # | Mechanism | What it controls | Scope |
|---|-----------|------------------|-------|
| 1 | `~/.ethopy/local_conf.json` | Everything: database, paths, logging, schemata, plugins, hardware | Default for every run on this machine |
| 2 | `ethopy -c /path/to/conf.json` | Same as above, from a different file | A single run |
| 3 | Environment variable `ETHOPY_PLUGIN_PATH` | Extra plugin directories only | The shell session |

Command-line flags (`--log-level`, `--log-console`) override individual logging settings for one run, but they are not a general configuration mechanism.

### 1. The default configuration file

With no other instruction, EthoPy loads:

- **Mac/Linux**: `~/.ethopy/local_conf.json`
- **Windows**: `%USERPROFILE%\.ethopy\local_conf.json`

The file is read once, at `import ethopy`. If it does not exist, EthoPy runs entirely on its [built-in defaults](#built-in-defaults).

### 2. A different config file on the command line

Every EthoPy run can point at another file with `-c` / `--config`:

```bash
ethopy -p my_task.py -c /path/to/rig2_conf.json
```

This is the supported way to keep several configurations on one machine, one per rig, or one per database. The file must exist; EthoPy exits with an error if it does not.

**Important**: the custom file *replaces* the default one, it is not merged with `~/.ethopy/local_conf.json`. Any key you leave out falls back to the built-in default, not to the value in your home-directory file. Custom config files should therefore be self-contained.

### 3. From Python

The configuration object is available as `ethopy.local_conf` and can be read anywhere in your own code, including plugins:

```python
from ethopy import local_conf

source = local_conf.get("source_path")              # top-level key
host = local_conf.get("database.host")              # key inside dj_local_conf
level = local_conf.get("logging.level", "INFO")     # nested key, with a fallback
```

`get()` resolves a key in this order, returning the first hit:

1. A top-level key of the JSON file (`"source_path"`, `"SCHEMATA"`, `"Channels"`, ...)
2. A key inside `dj_local_conf` (so `"database.host"` works without a prefix)
3. Dot notation walked through nested objects (`"logging.level"`)
4. The `default` argument you passed, or `None`

To load a different file explicitly — for example in a notebook or an export script:

```python
from ethopy.config import ConfigurationManager

conf = ConfigurationManager(config_file="/path/to/rig2_conf.json")
print(conf)                       # prints every resolved setting and the file it came from
```

By itself this only affects the `conf` object. To make the rest of EthoPy (DataJoint connection, schema names, plugin manager) use it as well, apply it globally:

```python
conf.update_global_config()
```

`update_global_config()` re-points `ethopy.local_conf`, pushes `dj_local_conf` into DataJoint, replaces `ethopy.SCHEMATA`, and rebuilds the plugin manager from the new `plugin_path`. This is exactly what `ethopy -c` does internally.

The NWB exporter takes its own `config_path` argument for the same purpose — see [Export to NWB](nwb_docs.md).

### 4. Environment variables

EthoPy reads exactly one environment variable of its own:

- **`ETHOPY_PLUGIN_PATH`** — a comma-separated list of extra plugin directories.

```bash
export ETHOPY_PLUGIN_PATH=/path/to/plugins,/another/plugin/path
```

There is **no environment variable for the config file location** — use `ethopy -c` instead.

Note that DataJoint's own environment variables (`DJ_HOST`, `DJ_USER`, `DJ_PASS`, ...) have no effect on EthoPy. EthoPy pushes its `dj_local_conf` block into `dj.config` at import, after DataJoint has read the environment, so the JSON file — or its default — wins in every case. Set database credentials in `dj_local_conf`, not in the environment.

## Built-in Defaults

Any key absent from your configuration file is filled in with a default, so a minimal file is enough to get started. The defaults are:

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "",
        "database.port": 3306,
        "database.reconnect": true,
        "database.use_tls": false,
        "datajoint.loglevel": "WARNING"
    },
    "SCHEMATA": {
        "experiment": "lab_experiments",
        "stimulus": "lab_stimuli",
        "behavior": "lab_behavior",
        "interface": "lab_interface",
        "recording": "lab_recordings"
    },
    "logging": {
        "level": "INFO",
        "directory": "~/.ethopy",
        "filename": "ethopy.log",
        "max_size": 31457280,
        "backup_count": 5
    },
    "source_path": "~/EthoPy_Files",
    "target_path": "/",
    "plugin_path": "~/.ethopy/ethopy_plugins"
}
```

Merging happens **one level deep**. If your file defines `"logging": {"level": "DEBUG"}`, you keep the default `directory`, `filename`, `max_size` and `backup_count` and only override `level`. The same applies to `dj_local_conf` and `SCHEMATA`.

Keys with no default — `Channels`, `server.*`, `video_source_path`, `video_target_path` — are simply absent unless you define them, and code that needs them supplies its own fallback.

## Precedence Order

For a normal `ethopy` run, a setting is resolved like this — first match wins:

1. **Command-line flag**, where one exists for that setting (`--log-level`, `--log-console`)
2. **The active configuration file** — the `-c/--config` file if given, otherwise
   `~/.ethopy/local_conf.json`
3. **Built-in default** for that key

Only one configuration file is ever active; files are never merged with each other.

**Plugin directories are the exception** — they accumulate rather than override. All three sources are scanned, in this order:

1. `plugin_path` from the active configuration file
2. `~/.ethopy/ethopy_plugins` (the default directory)
3. Each path in `ETHOPY_PLUGIN_PATH`, left to right

When two directories contain a module with the same import name, **the one scanned later wins**, so `ETHOPY_PLUGIN_PATH` overrides the default directory, which overrides `plugin_path`. Core EthoPy modules always win over plugins, regardless of source. Each conflict is logged as a warning naming both files. See the [Plugin System](plugin.md) guide for details.

## Essential Configuration

### Minimal Setup

Here's the minimum configuration needed to get EthoPy running:

```json
{
    "dj_local_conf": {
        "database.host": "your_database_address",
        "database.user": "your_username",
        "database.password": "your_password_here",
        "database.port": 3306
    },
    "source_path": "/path/to/your/data",
    "target_path": "/path/to/your/backup"
}
```

**What each part does:**

- **`dj_local_conf`**: Database connection settings (required)
- **`source_path`**: Where experimental data files are saved on this machine
- **`target_path`**: Where backup copies should be saved

## Configuration Sections

### 1. Database Settings (Required)

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "your_mysql_password",
        "database.port": 3306,
        "database.reconnect": true,
        "database.use_tls": false,
        "database.enable_python_native_blobs": true,
        "datajoint.loglevel": "WARNING"
    }
}
```

**Settings explanation:**

- **`database.host`**: Database server IP address or hostname
    - **Default**: `"127.0.0.1"` (localhost - your computer)
    - **Examples**: `"192.168.1.100"`, `"lab-database.university.edu"`

- **`database.user`**: MySQL username for database connection
    - **Default**: `"root"`

- **`database.password`**: MySQL password for the specified user
    - **Default**: `""` (empty string)

- **`database.port`**: MySQL server port number
    - **Default**: `3306` (standard MySQL port)
    - **Note**: Only change if your MySQL uses a different port

- **`database.reconnect`**: Automatically reconnect if connection is lost
    - **Default**: `true`

- **`database.use_tls`**: Use encrypted TLS connection to database, more details [here](https://docs.datajoint.com/core/datajoint-python/latest/client/settings/#tls-configuration)
    - **Default**: `false`

- **`datajoint.loglevel`**: DataJoint library logging verbosity
    - **Default**: `"WARNING"`
    - **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`

### 2. File Paths (Required)

```json
{
    "source_path": "/Users/yourname/experiment_data",
    "target_path": "/Users/yourname/experiment_backup"
}
```

**Settings explanation:**

- **`source_path`**: Local directory where experimental data files are saved
    - **Default**: `"~/EthoPy_Files"` (EthoPy_Files folder in your home directory)
    - **Purpose**: All recorded data (videos, sensor data, etc.) is stored here during experiments
    - **Examples**: `"/Users/yourname/experiment_data"`, `"/home/pi/data"`

- **`target_path`**: Directory where backup copies of data should be moved after experiments
    - **Default**: `"/"` (root directory - usually needs to be changed)
    - **Purpose**: Automatic backup/archival location for completed experiments
    - **Examples**: `"/mnt/lab_storage"`

**Important**: Always use full paths starting from your drive root.

### 3. Logging Settings (Optional)

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

**Settings explanation:**

- **`level`**: Minimum log level to record
    - **Default**: `"INFO"`
    - **Options**: `"DEBUG"` (most verbose), `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"` (least verbose)
    - **Purpose**: Controls how much detail is logged

- **`directory`**: Directory where log files are stored
    - **Default**: `"~/.ethopy/"` (hidden .ethopy folder in your home directory)
    - **Examples**: `"/var/log/ethopy"`, `"/Users/yourname/logs"`

- **`filename`**: Name of the main log file
    - **Default**: `"ethopy.log"`
    - **Note**: Backup files will be named `ethopy.log.1`, `ethopy.log.2`, etc.

- **`max_size`**: Maximum size of log file before rotation (in bytes)
    - **Default**: `31457280` (30 MB)
    - **Purpose**: Prevents log files from growing too large

- **`backup_count`**: Number of old log files to keep
    - **Default**: `5`
    - **Purpose**: Maintains history while preventing unlimited disk usage

### 4. Hardware Setup (Optional - Raspberry Pi only)

If you're using physical hardware (valves, sensors, LEDs), specify GPIO pin connections:

```json
{
    "Channels": {
        "Liquid": {"1": 22, "2": 23},
        "Lick": {"1": 17, "2": 27},
        "Odor": {"1": 24, "2": 25}
    }
}
```

**Settings explanation:**

- **`Channels`**: Maps hardware types to GPIO pin assignments
    - **Default**: Not set (no hardware channels configured)
    - **Purpose**: Tells EthoPy which GPIO pins control which hardware devices

**Hardware types:**

- **`Liquid`**: Water delivery pumps/valves for reward delivery
    - **Format**: `{"port_number": gpio_pin_number}`
    - **Example**: `{"1": 22, "2": 23}` means port 1 uses GPIO pin 22, port 2 uses GPIO pin 23

- **`Lick`**: Lick detection sensors for behavioral monitoring
    - **Format**: `{"sensor_number": gpio_pin_number}`
    - **Example**: `{"1": 17, "2": 27}` means lick sensor 1 on GPIO pin 17, sensor 2 on GPIO pin 27

- **`Odor`**: Odor delivery valves for olfactory experiments
    - **Format**: `{"valve_number": gpio_pin_number}`
    - **Example**: `{"1": 24, "2": 25}` means valve 1 on GPIO pin 24, valve 2 on GPIO pin 25

**Important notes:**

- Each GPIO pin number can only be used once across all hardware types

### 5. Custom Schema Names (Optional)

If your database uses custom schema names:

```json
{
    "SCHEMATA": {
        "experiment": "my_experiments",
        "behavior": "my_behavior_data",
        "stimulus": "my_stimuli",
        "interface": "my_interface",
        "recording": "my_recordings"
    }
}
```

**Settings explanation:**

- **`SCHEMATA`**: Maps EthoPy data types to your custom database schema names
    - **Purpose**: Allows EthoPy to work with existing databases that use different naming conventions

**Schema types and defaults:**

- **`experiment`**: Main experimental session data
    - **Default**: `"lab_experiments"`
    - **Contains**: Session info, trial data, animal information

- **`behavior`**: Behavioral measurement data
    - **Default**: `"lab_behavior"`
    - **Contains**: Lick detection, movement tracking, response data

- **`stimulus`**: Stimulus presentation information
    - **Default**: `"lab_stimuli"`
    - **Contains**: Visual/auditory stimuli parameters, timing

- **`interface`**: Hardware interface configurations
    - **Default**: `"lab_interface"`
    - **Contains**: Hardware setup parameters, calibration data

- **`recording`**: Data recording metadata
    - **Default**: `"lab_recordings"`
    - **Contains**: File paths, recording parameters, data format info

**Note**: Most users can skip this section - EthoPy will use the default schema names.

### 6. Plugin Path (Optional)

```json
{
    "plugin_path": "/Users/yourname/.ethopy/ethopy_plugins"
}
```

**Settings explanation:**

- **`plugin_path`**: Directory where EthoPy plugins are stored
    - **Default**: `"~/.ethopy/ethopy_plugins"` (plugins folder in your .ethopy directory)
    - **Purpose**: Location for custom EthoPy extensions and plugins
    - **Examples**: `"/Users/yourname/my_plugins"`, `"/opt/ethopy_plugins"`

**Note**: Only needed if you're using custom plugins or want to store them in a different
location. `plugin_path` is one of three plugin sources — the default
`~/.ethopy/ethopy_plugins` directory and `ETHOPY_PLUGIN_PATH` are also scanned. All three
are used together; see [Precedence Order](#precedence-order).

## Common Setup Scenarios

### Local Database Setup (Most Common)

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "your_mysql_password",
        "database.port": 3306
    },
    "source_path": "/Users/yourname/experiment_data",
    "target_path": "/Users/yourname/experiment_backup"
}
```

### Remote Database Setup

```json
{
    "dj_local_conf": {
        "database.host": "192.168.1.100",
        "database.user": "lab_user",
        "database.password": "lab_password",
        "database.port": 3306
    },
    "source_path": "/Users/yourname/experiment_data",
    "target_path": "/Users/yourname/experiment_backup"
}
```

### Hardware Experiment Setup

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "your_password",
        "database.port": 3306
    },
    "source_path": "/home/pi/experiment_data",
    "target_path": "/home/pi/experiment_backup",
    "Channels": {
        "Liquid": {"1": 22, "2": 23},
        "Lick": {"1": 17, "2": 27}
    }
}
```

## Troubleshooting

### Problem: "Cannot connect to database"

**Solutions:**

1. **Check your password** - Verify the password matches your MySQL password

2. **Advanced troubleshooting**: For remote databases or lab setups, contact your system administrator or IT support to verify database server status, network connectivity, and firewall settings. **Check if MySQL is running**
    - If `database.host` is `127.0.0.1` or `localhost`: Run `mysql -u root -p` on the same machine as EthoPy
    - If `database.host` is a remote IP (like `192.168.1.100`): Run the command on that remote database server
    - The command should ask for your password and connect successfully
    - If you get "command not found", MySQL client is not installed
    - If you get "connection refused", MySQL server is not running

3. **Check the database address** - For `127.0.0.1`, MySQL must be on your computer

4. **Check the port number** - MySQL usually uses 3306

### Problem: "Cannot find data path"

**Solutions:**

1. **Check the folder exists** - Verify the folder exists in your file system

2. **Use full paths** - Use `/Users/yourname/data/` not `data/`

3. **Check permissions** - Ensure you can read and write to the folder

4. **Create the folder** - Create the folder if it doesn't exist

### Problem: "Hardware not responding"

**Solutions:**

1. **Check physical connections** - Verify all wires are properly connected

2. **Check pin numbers** - Ensure pin numbers match your hardware setup

3. **Check for conflicts** - Make sure no pin number is used twice

4. **Test with simple LED** - Verify basic GPIO functionality

### Problem: "Configuration file not found"

**Solutions:**

1. **Check file location** - Ensure `local_conf.json` is in `~/.ethopy/`, or pass your file
   explicitly with `ethopy -c /path/to/conf.json`

2. **Check JSON format** - Verify proper JSON syntax (no missing commas/brackets). A file
   that fails to parse is logged as an error and EthoPy falls back to the built-in
   defaults, which usually shows up as a failed connection to `127.0.0.1`

3. **Confirm what was loaded** - Run
   `python -c "import ethopy; print(ethopy.local_conf)"` to print the active file path and
   every resolved setting

4. **Start simple** - Copy one of the examples from this guide

## Security Best Practices

- **Never share your config file** - It contains database passwords
- **Use strong passwords** - Protect your database access
- **Keep backups** - Save a copy of your working configuration
- **Use full paths** - Avoid relative paths that might break

## Local vs Database Settings

**Store in local_conf.json:**

  - Database connection details
  - File paths specific to this machine
  - Hardware GPIO pin assignments
  - Logging preferences
  - Machine-specific settings

**Store in database:**

  - Experimental parameters
  - Trial configurations
  - Animal information
  - Session data
  - Results and measurements

This separation keeps your experiments portable while maintaining machine-specific configurations.