# EthoPy Local Configuration Guide

## What is local_conf.json?

The `local_conf.json` file stores **device-specific settings** that are unique to each computer or experimental setup. These are settings that:

- Are tied to your specific hardware (file paths, GPIO pins, database credentials)
- Need to be configured once per machine, not per experiment

**Key distinction**: Experimental parameters and data go in the database. Machine-specific settings go in `local_conf.json`.

## File Location

EthoPy automatically looks for your configuration file here:

- **Mac/Linux**: `~/.ethopy/local_conf.json`
- **Windows**: `%USERPROFILE%\.ethopy\local_conf.json`

You can also specify a custom location using the environment variable:

- **Environment variable**: Set `ETHOPY_CONFIG_PATH` to point to your custom config file location

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

**Note**: Only needed if you're using custom plugins or want to store them in a different location.

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

1. **Check file location** - Ensure `local_conf.json` is in `~/.ethopy/`

2. **Check JSON format** - Verify proper JSON syntax (no missing commas/brackets)

3. **Start simple** - Copy one of the examples from this guide

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