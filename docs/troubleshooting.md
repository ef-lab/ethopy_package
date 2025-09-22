# Troubleshooting Guide

This guide helps you solve common problems when using EthoPy. Start with the problem that matches what you're seeing.

## Installation Problems

### "pip install ethopy" doesn't work

**What you see**: Error messages during installation

**Try these steps**:

1. **Update pip first**:
   ```bash
   pip install --upgrade pip
   ```

2. **Try installing in a clean environment**:
   ```bash
   python -m venv ethopy_env
   source ethopy_env/bin/activate  # On Windows: ethopy_env\Scripts\activate
   pip install ethopy
   ```

3. **If you still get errors**, ask your lab's technical support for help.

### "No module named 'ethopy'" after installation

**What you see**: `ImportError: No module named 'ethopy'` when trying to use EthoPy

**Try these steps**:

1. **Check if EthoPy is installed**:
   ```bash
   pip list | grep ethopy
   ```
   If you don't see ethopy in the list, reinstall it.

2. **Make sure you're in the right environment** where you installed EthoPy.

## Database Connection Problems

### "Cannot connect to database"

**What you see**: Error messages about database connection failing

**Try these steps**:

1. **Check your password** - Make sure the password in your `local_conf.json` file is correct

2. **Check your configuration file** (see [Local Configuration Guide](local_conf.md)):
   ```json
   {
       "dj_local_conf": {
           "database.host": "127.0.0.1",
           "database.user": "root",
           "database.password": "your_actual_password",
           "database.port": 3306
       }
   }
   ```

3. **For local database**: Try running `mysql -u root -p` in your terminal
   - If this doesn't work, MySQL isn't running on your computer
   - Ask your technical support to help start MySQL

4. **For remote database**: Contact your lab's database administrator

## Experiment Won't Start

### "Experiment fails to start"

**What you see**: Errors when running `ethopy --task-path your_task.py`

**Try these steps**:

1. **Test with a simple example first**:
   ```bash
   ethopy --task-path grating_test.py --log-console
   ```

2. **If the example works but your task doesn't**, check your task file for errors

3. **Get more information** by adding debug logging:
   ```bash
   ethopy --task-path your_task.py --log-console --log-level DEBUG
   ```

4. **Look at the error messages** - they usually tell you what's wrong

### "Setup configuration not found"

**What you see**: Error about missing setup_conf_idx

**Try these steps**:

1. **Use simulation mode** (setup_conf_idx = 0) for testing:
   ```python
   # In your task file
   setup_conf_idx = 0  # Use simulation mode
   ```

2. **Create your hardware configuration** - see [Setup Configuration Guide](setup_configuration_idx.md)

## Hardware Not Working

### "Hardware not responding" (Raspberry Pi/Arduino)

**What you see**: Sensors or valves not working

**Try these steps**:

1. **Start with simulation mode** (setup_conf_idx = 0) to test your experiment logic

2. **Check hardware connections** - make sure all cables are properly connected

3. **Verify GPIO pin configuration** in your `local_conf.json` file

4. **Contact your hardware setup person** - they know your specific hardware best

### "Reward delivery not working"

**What you see**: No water/reward coming out during experiments

**Try these steps**:

1. **Check if water reservoir is full**

2. **Test ports manually** if you have calibration software

3. **Verify port configuration** matches your hardware setup

4. **Contact your lab technician** for hardware issues

## File and Data Problems

### "Cannot find data path"

**What you see**: Errors about missing data directories

**Try these steps**:

1. **Check if the folders exist** - go to your file manager and look for the paths in your config

2. **Create missing folders**:
   ```bash
   mkdir -p /path/to/your/data
   mkdir -p /path/to/your/backup
   ```

3. **Use full paths** in your `local_conf.json` (like `/Users/yourname/data` not just `data`)

### "Permission denied" errors

**What you see**: Can't write to files or folders

**Try these steps**:

1. **Choose a folder in your home directory** for data storage

2. **On Mac/Linux**, make sure you can write to the folder:
   ```bash
   ls -la /path/to/your/folder
   ```

3. **Ask your system administrator** if you can't access the folders you need

## Common Error Messages

### "Task not found"

**What you see**: Can't find your task file

**Solution**: Use the full path to your task file:
```bash
ethopy --task-path /full/path/to/your_task.py --log-console
```

### "Already running"

**What you see**: EthoPy says it's already running

**Solution**:
1. **Close any other EthoPy windows** that might be open
2. **Restart your computer** if you're not sure
3. **Wait a minute** and try again

## Getting Help

### When to ask for help

Ask your lab's technical support when you see:
- Database connection errors (after checking your password)
- Hardware not responding
- Permission/access errors
- Installation problems that persist

### How to ask for help effectively

When asking for help, include:

1. **What you were trying to do**: "I was trying to run my task file..."
2. **What happened**: "I got this error message: [copy the exact error]"
3. **What you already tried**: "I checked my password and restarted EthoPy"

### Information that helps

- **EthoPy version**: Run `python -c "import ethopy; print(ethopy.__version__)"`
- **Your operating system**: Windows, Mac, or Linux
- **The exact error message**: Copy and paste the whole error

### Where to get help

1. **Your lab's technical support** - they know your specific setup
2. **EthoPy documentation** - check related guides for your issue
3. **GitHub Issues**: [Report bugs here](https://github.com/ef-lab/ethopy_package/issues)

## Quick Fixes

### Before asking for help, try these:

1. **Restart EthoPy** - close it completely and start again
2. **Check your internet connection** - needed for some database connections
3. **Try simulation mode** - use `setup_conf_idx = 0` to test without hardware
4. **Use example tasks** - make sure EthoPy works with provided examples first
5. **Check log files** - look in `~/.ethopy/ethopy.log` for error details

### Most problems are caused by:

- **Wrong passwords** in configuration files
- **Missing folders** for data storage
- **Hardware not properly connected**
- **Typos in file paths**
- **Using the wrong setup_conf_idx** for your hardware

Remember: It's better to ask for help early than to spend hours stuck on a problem!