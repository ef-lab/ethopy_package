# EthoPy Configuration Guide

## What is Configuration?

Configuration tells EthoPy how to connect to your database, where to find your data, and how to configure your hardware. Think of it as a settings file that contains all the important information EthoPy needs to run your experiments.

## Where is the Configuration File?

EthoPy automatically looks for a file called `local_conf.json` in a special folder:
- **Mac/Linux**: In your home folder under `.ethopy/local_conf.json`
- **Windows**: In your user folder under `.ethopy\local_conf.json`

## Quick Start Guide

### Step 1: Basic Setup
When you first start EthoPy, you'll need to create a configuration file. Here's a simple example to get you started:

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "root",
        "database.password": "your_password_here",
        "database.port": 3306
    },
    "source_path": "/path/to/your/data",
    "target_path": "/path/to/your/backup"
}
```

### Step 2: What Each Part Means
- **database settings**: How to connect to your MySQL database
- **source_path**: Where your experimental data is stored
- **target_path**: Where backup copies should be saved

## Complete Configuration Structure

Here's what a full configuration file looks like with all the optional settings:

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
    },
    "channels": {
        "Liquid": {"1": 22, "2": 23},
        "Lick": {"1": 17, "2": 27}
    }
}
```

## Understanding Each Section

### 1. Database Settings (`dj_local_conf`)

This section tells EthoPy how to connect to your MySQL database:

```json
{
    "dj_local_conf": {
        "database.host": "127.0.0.1",     // Database server address (127.0.0.1 means your computer)
        "database.user": "root",          // Your database username
        "database.password": "password",  // Your database password
        "database.port": 3306            // Database port number (3306 is standard for MySQL)
    }
}
```

**What you need to change:**
- Replace `"password"` with your actual MySQL password
- If your database is on another computer, change `"127.0.0.1"` to that computer's address

### 2. File Paths

These tell EthoPy where to find and save your data:

```json
{
    "source_path": "/Users/yourname/experiment_data",  // Where your data files are stored
    "target_path": "/Users/yourname/backup_data"       // Where to save backup copies
}
```

**Important:** Use full paths (starting from the root of your drive) to avoid confusion.

### 3. Logging Settings (Optional)

This controls how EthoPy saves information about what it's doing:

```json
{
    "logging": {
        "level": "INFO",              // How much detail to log (DEBUG, INFO, WARNING, ERROR)
        "directory": "~/.ethopy/",    // Where to save log files
        "filename": "ethopy.log"      // Name of the log file
    }
}
```

### 4. Schema Names

If your database uses custom names for different parts of your experiment data:

```json
{
    "SCHEMATA": {
        "experiment": "my_experiments",   // Custom name for experiment schema
        "behavior": "my_behavior_data",   // Custom name for behavior schema
        "recording": "my_recordings"      // Custom name for recording schema
    }
}
```

**Most users can skip this section** - EthoPy will use standard names.

### 5. Hardware Setup (`channels`) - For Raspberry Pi Users

**Skip this section if you're not using physical hardware like valves, sensors, or LEDs.**

If you're running experiments with physical hardware (like water valevs, lick detectors, or LEDs), you need to tell EthoPy which pins on your Raspberry Pi connect to which devices.

```json
{
    "channels": {
        "Liquid": {"1": 22, "2": 23},    // Water valve connections
        "Lick": {"1": 17, "2": 27},      // Lick sensor connections
    }
}
```

#### Common Hardware Types

**Liquid Delivery (Water Pumps)**
- Controls water rewards for your animals
- Example: `"Liquid": {"1": 22, "2": 23}` means pump #1 is connected to pin 22, pump #2 to pin 23

**Lick Detection (Sensors)**
- Detects when animals lick at reward ports
- Example: `"Lick": {"1": 17, "2": 27}` means sensor #1 is on pin 17, sensor #2 on pin 27

**Odor Delivery (Valves)**
- Controls scent delivery for olfactory experiments
- Example: `"Odor": {"1": 24, "2": 25}` means valve #1 is on pin 24, valve #2 on pin 25


#### Simple Hardware Configurations

**Basic Setup (just water and lick detection):**
```json
{
    "channels": {
        "Liquid": {"1": 22, "2": 23},
        "Lick": {"1": 17, "2": 27}
    }
}
```

**Full Setup (all hardware types):**
```json
{
    "channels": {
        "Liquid": {"1": 22, "2": 23},
        "Lick": {"1": 17, "2": 27},
        "Odor": {"1": 24, "2": 25},
    }
}
```

**Important Notes:**
- Each pin number can only be used once
- Make sure your hardware is properly connected before running experiments
- If you're not sure about pin numbers, check your circuit diagram or ask your hardware setup person

## Common Configuration Scenarios

### Scenario 1: Basic Local Setup (Most Users)
You have MySQL running on your computer and want to store data locally:

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

### Scenario 2: Remote Database Setup
Your database is on a different computer in the lab:

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

### Scenario 3: Hardware Experiments with Raspberry Pi
You're running behavioral experiments with physical hardware:

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
    "channels": {
        "Liquid": {"1": 22, "2": 23},
        "Lick": {"1": 17, "2": 27},
    }
}
```

## Security and Best Practices

### Keep Your Password Safe
- Never share your configuration file with others
- Use a strong password for your database
- Consider using environment variables for sensitive information


### File Organization
- Use full paths (like `/Users/yourname/data`) instead of relative paths (like `../data`)
- Make sure the folders you specify actually exist
- Keep backups of your configuration file

## Troubleshooting

### Problem: "Cannot connect to database"

**Symptoms:** EthoPy says it can't connect to your database

**Solutions to try:**
1. **Check your password** - Make sure the password in your config file matches your MySQL password
2. **Check if MySQL is running** - Open a terminal and try: `mysql -u root -p`
3. **Check the database address** - If using `127.0.0.1`, make sure MySQL is running on your computer
4. **Check the port number** - MySQL usually uses port 3306, but yours might be different

### Problem: "Cannot find data path"

**Symptoms:** EthoPy says it can't find your data folder

**Solutions to try:**
1. **Check the folder exists** - Go to your file browser and make sure the folder actually exists
2. **Use full paths** - Instead of `data/`, use `/Users/yourname/data/`
3. **Check permissions** - Make sure you can read and write to the folder
4. **Create the folder** - If it doesn't exist, create it first

### Problem: "Hardware not responding"

**Symptoms:** Your pumps, sensors, or LEDs aren't working

**Solutions to try:**
1. **Check physical connections** - Make sure all wires are properly connected
2. **Check pin numbers** - Verify the pin numbers in your config match your hardware setup
3. **Check for conflicts** - Make sure no pin number is used twice
4. **Test with a simple LED** - Connect a simple LED to verify basic functionality

### Problem: "Configuration file not found"

**Symptoms:** EthoPy says it can't find your configuration

**Solutions to try:**
1. **Check the file location** - Make sure `local_conf.json` is in the right folder (`.ethopy` in your home directory)
2. **Check file format** - Make sure your JSON file is properly formatted (no missing commas or brackets)
3. **Start with a simple config** - Copy one of the examples from this guide

### Getting Help

If you're still having trouble:
1. Check the EthoPy log file (usually in `~/.ethopy/ethopy.log`)
2. Ask your lab's technical support person
3. Make sure you're using the latest version of EthoPy

## Advanced Topics

### Using Environment Variables
If you want to keep your database password separate from your config file, you can use environment variables:

```bash
# In your terminal before running EthoPy:
export ETHOPY_DB_PASSWORD="your_secret_password"
```

Then in your config file, you can leave the password field empty - EthoPy will use the environment variable instead.

### Custom Configuration Locations
By default, EthoPy looks for configuration in `~/.ethopy/local_conf.json`. If you need to use a different location, you can specify it when starting EthoPy (ask your technical support person for help with this).