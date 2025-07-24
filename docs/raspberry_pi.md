<!-- ToDo -->
# Raspberry Pi Setup Guide

This guide provides essential commands for setting up Ethopy on a Raspberry Pi device.

## SSH Setup
Enable SSH service for remote access:
```bash
sudo systemctl enable ssh  # Enables SSH to start automatically at boot
sudo systemctl start ssh   # Starts SSH service immediately
```

## Ethopy Installation

1. Install Ethopy:
   ```bash
   pip install ethopy
   ```

2. Create configuration file at `~/.ethopy/local_conf.json`:
   ```json
   {
       "dj_local_conf": {
           "database.host": "YOUR DATABASE",
           "database.user": "USERNAME",
           "database.password": "PASSWORD",
           "database.port": "PORT",
           "database.reconnect": true,
           "database.enable_python_native_blobs": true
       },
       "source_path": "LOCAL_RECORDINGS_DIRECTORY",
       "target_path": "TARGET_RECORDINGS_DIRECTORY"
   }
   ```
For detailed desciption of configuration files, see [Local configuration](local_conf.md). 


## Database connection:
```bash
ethopy-db-connection     # Tests database connection to verify setup
```

## GPIO Hardware Support
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

## Display Configuration
Configure display settings for GUI applications via SSH:
```bash
export DISPLAY=:0                           # Sets display to primary screen
sed -i -e '$aexport DISPLAY=:0' ~/.profile  # Persists DISPLAY setting in profile
sed -i -e '$axhost +  > /dev/null' ~/.profile  # Allows X11 forwarding access
```

## Screen Blanking Disable
To prevent screen from turning off, run raspi-config:
```bash
sudo raspi-config
```
Navigate to "Display Options" → "Screen Blanking" → Set to "No"

## Troubleshooting

### Common Issues

1. **Display Issues**
   - Ensure DISPLAY is set correctly in ~/.profile
   - Check X server is running
   - Verify permissions with `xhost +`

2. **GPIO Access**
   - Verify pigpiod service is running: `systemctl status pigpiod`
   - Check user permissions for GPIO access

3. **Database Connection**
   - Test connection: `ethopy-db-connection`
   - Check network connectivity to database server
   - Verify credentials in local_conf.json