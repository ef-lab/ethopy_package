# Troubleshooting Guide

This guide addresses common issues you might encounter when using EthoPy and provides solutions to help you resolve them quickly.

## Installation Issues

### Package Installation Failures

**Problem**: Installation fails with dependency conflicts.

**Solution**: 
1. Try creating a fresh virtual environment:
   ```bash
   python -m venv ethopy_env
   source ethopy_env/bin/activate  # On Windows: ethopy_env\Scripts\activate
   pip install ethopy
   ```

2. If specific dependencies are failing, try installing them manually first:
   ```bash
   pip install panda3d numpy pygame
   pip install ethopy
   ```

### ImportError After Installation

**Problem**: You get `ImportError: No module named 'ethopy'` after installation.

**Solution**:
1. Verify the package is installed:
   ```bash
   pip list | grep ethopy
   ```

2. Make sure you're using the same Python environment where you installed the package.
3. If installed in development mode, check that your working directory is properly set.

## Database Connection Issues

### Cannot Connect to Database

**Problem**: Error when connecting to MySQL database.

**Solution**:
1. Verify your connection information in `local_conf.json`:
   ```json
   {
    "dj_local_conf": {
        "database.host": "127.0.0.1",
        "database.user": "your_username",
        "database.password": "your_password",
        "database.port": 3306,
        "database.reconnect": true,
        "database.use_tls": false,
        "datajoint.loglevel": "WARNING"
    },
   }
   ```

2. Make sure MySQL is running:
   ```bash
   # For Linux/Mac
   sudo service mysql status
   # or
   sudo systemctl status mysql
   
   # For Windows (check in services)
   ```

3. Test connection with MySQL client:
   ```bash
   mysql -u your_username -p
   ```

### Schema Creation Errors

**Problem**: Errors when creating the database schema.

**Solution**:
1. Make sure the user has privileges to create databases and tables:
   ```sql
   GRANT ALL PRIVILEGES ON *.* TO 'your_username'@'localhost';
   FLUSH PRIVILEGES;
   ```

2. Try manually running the schema setup:
   ```bash
      python3 -c 'from ethopy.core.experiment import *'
      python3 -c 'from ethopy.core.stimulus import *'
      python3 -c 'from ethopy.core.sehavior import *'
      python3 -c 'from ethopy.stimuli import *'
      python3 -c 'from ethopy.behaviors import *'
      python3 -c 'from ethopy.experiments import *'
   ```

3. Check for database encoding issues. EthoPy requires UTF-8 encoding.

## Hardware Interface Issues

### Port Communication Problems

**Problem**: Cannot communicate with hardware ports.

Solutions:
ToDo

<!-- **Solution**:
1. Check permissions for USB devices (Linux/Mac):
   ```bash
   sudo chmod a+rw /dev/ttyUSB0  # Replace with your port
   ```

2. Verify port configuration in your setup:
   ```bash
   ethopy --list-ports
   ```

3. For Arduino interfaces, make sure the correct firmware is uploaded. -->

### Raspberry Pi Specific Issues

**Problem**: Issues when running on Raspberry Pi.

<!-- **Solution**:
1. Make sure you have the latest Raspberry Pi OS.
2. Enable I2C and SPI interfaces:
   ```bash
   sudo raspi-config
   # Navigate to Interface Options > I2C/SPI and enable
   ```
3. Check GPIO permissions:
   ```bash
   sudo usermod -a -G gpio your_username
   ``` -->

## Experiment Execution Issues

### Experiment Fails to Start

**Problem**: `ethopy -p your_task.py` fails to start the experiment.

**Solution**:
1. Check for syntax errors in your task file.
2. Verify that experiment, behavior, and stimulus classes are correctly imported and assigned.
3. Run with debug logging:
   ```bash
   ethopy -p your_task.py --log-console --log-level DEBUG
   ```

### Reward Delivery Failures

**Problem**: Water rewards are not being delivered correctly.

**Solution**:
1. Run a calibration task to test the ports:
   ```bash
   ethopy -p calibrate_ports.py
   ```
2. Check solenoid connections and power supply.
3. Verify port configuration in your setup.

## Data Logging Issues

### Missing Trial Data

**Problem**: Some trial data is not being logged to the database.

**Solution**:
1. Check database connection during experiment execution.
2. Verify that your experiment states are correctly called.

## Common Error Messages

### "No task found with idx X"

**Problem**: `ethopy --task-idx X` returns "No task found with idx X".

**TODO: Solution**:
1. Verify the task exists in the database:
   ```bash
   ethopy --list-tasks
   ```
2. If missing, add your task to the database:
   ```bash
   ethopy --add-task your_task.py
   ```

### "Multiple instances detected"

**Problem**: Attempt to run EthoPy when an instance is already running.

**Solution**:
1. Find and close the existing EthoPy process:
   ```bash
   # Linux/Mac
   ps aux | grep ethopy
   kill <pid>
   
   # Windows
   tasklist | findstr ethopy
   taskkill /F /PID <pid>
   ```

## Getting Help

If you're still experiencing issues:

1. **Check the documentation**: Review related sections in the documentation for guidance.

2. **Search GitHub Issues**: Check if your issue has been reported and addressed:
   [EthoPy GitHub Issues](https://github.com/ef-lab/ethopy_package/issues)

3. **Submit an Issue**: If your problem persists, submit a detailed issue on GitHub with:
   - EthoPy version (`ethopy --version`)
   - Python version (`python --version`)
   - Operating system details
   - Complete error message and traceback
   - Steps to reproduce the problem

4. **Contact Maintainers**: For urgent issues, contact the package maintainers directly.

## Appendix: Log Files

EthoPy creates log files that can be valuable for troubleshooting.

Reviewing these logs can provide insights into issues not apparent from console output.