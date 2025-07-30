<!-- ToDo -->

# Raspberry Pi Setup Guide

This guide provides essential commands for setting up Ethopy on a Raspberry Pi (RP) device.

## Prerequisites

- Raspberry Pi image
  Follow the instructions in the [Raspberry Pi documentation](https://www.raspberrypi.com/documentation/) to install the image and set up your Raspberry Pi.

- SSH Setup
  Enable the SSH service to allow remote access to your Raspberry Pi via the terminal:

  ```bash
  sudo systemctl enable ssh  # Enables SSH to start automatically at boot
  sudo systemctl start ssh   # Starts SSH service immediately
  ```

## EthoPy setup

### Step 1: Ethopy Installation

Once your Raspberry Pi is set up, you can connect to it from your computer's terminal based on [Raspberry Pi documentation](https://www.raspberrypi.com/documentation/).

1. Verify python version:
   ```bash
   python --version
   ```

EthoPy requires Python >=3.8, < 3.12

2. Setting Up a Virtual Environment

=== "Conda"
    ```sh
    conda create --name myenv python=3.x
    conda activate myenv
    ```
    [More details](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

=== "venv (pip)"
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
    [More details](https://docs.python.org/3/library/venv.html)

=== "uv"
    ```sh
    uv venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```
    [More details](https://github.com/astral-sh/uv)

Once activated, proceed with installation.

3. Install Ethopy:

    ```bash
    pip install ethopy
    ```

4. Create configuration file at `~/.ethopy/local_conf.json`:
   ```json
   {
       "dj_local_conf": {
           "database.host": "YOUR DATABASE",
           "database.user": "USERNAME",
           "database.password": "PASSWORD",
           "database.port": 3306,
           "database.reconnect": true,
           "database.enable_python_native_blobs": true
       },
       "source_path": "LOCAL_RECORDINGS_DIRECTORY",
       "target_path": "TARGET_RECORDINGS_DIRECTORY"
   }
   ```

For detailed desciption of configuration files, see [Local configuration](local_conf.md).

### Step 2: Database connection

```bash
ethopy-db-connection     # Tests database connection to verify setup
```

### Step 3: GPIO Hardware Support

Enable pigpio daemon for GPIO control:

```bash
sudo systemctl enable pigpiod.service  # Enables pigpio daemon to start at boot
sudo systemctl start pigpiod.service   # Starts pigpio daemon for immediate GPIO access
```

Install GPIO libraries:

```bash
pip install pigpio              # Python library for pigpio daemon communication
sudo apt-get install python3-rpi.gpio  # Alternative GPIO library for Raspberry Pi
```

### Step 4: Display Configuration

Configure display settings for GUI applications via SSH:

```bash
export DISPLAY=:0                           # Sets display to primary screen
sed -i -e '$aexport DISPLAY=:0' ~/.profile  # Persists DISPLAY setting in profile
sed -i -e '$axhost +  > /dev/null' ~/.profile  # Allows X11 forwarding access
```

### Step 5: Screen Blanking Disable

To prevent screen from turning off, run raspi-config:

```bash
sudo raspi-config
```

Navigate to "_Display Options_" → "_Screen Blanking_" → Set to "No"

### Step 6: Run your first experiment

```
ethopy --task-path grating_test.py --log-console
```

## Troubleshooting

### Common Issues

1. **Display Issues**

   - Ensure DISPLAY is set correctly in ~/.profile
   - Check X server is running
   - Verify permissions with `xhost +`

1. **GPIO Access**

   - Verify pigpiod service is running: `systemctl status pigpiod`
   - Check user permissions for GPIO access

1. **Database Connection**

   - Test connection: `ethopy-db-connection`
   - Check network connectivity to database server
   - Verify credentials in local_conf.json

## Where to Go Next

Now that you have a basic understanding of EthoPy:

1. [Creating Custom Components](creating_custom_components.md)
1. How to [create a Task ](https://ef-lab.github.io/ethopy_package/task_setup/)
1. Explore the [Plugin System](plugin.md) to extend functionality
1. Dive deeper into [Local Configuration](local_conf.md) for advanced settings
1. Understand [setup configuration index](setup_configuration_idx.md)
1. Learn more about [Database Setup](database_setup.md)
1. Study the [API Reference](API/logger.md) for detailed documentation
1. Check [Contributing](contributing.md) if you want to help improve EthoPy
