# Setup Configuration in EthoPy

## What is setup_conf_idx?

The `setup_conf_idx` (Setup Configuration Index) is a **unique number** that defines a complete hardware setup for your experiments. Think of it as a "recipe" that tells EthoPy:

- What type of hardware interface to use (simulation, Raspberry Pi, PC ports, etc.)
- How many ports you have and what they do (lick detection, reward delivery, etc.)
- Screen settings for visual stimuli
- Camera settings for recording
- Audio settings for sound stimuli

**Key concept**: Each `setup_conf_idx` represents one complete hardware configuration. You can have multiple configurations (idx 0, 1, 2, etc.) for different experimental setups.

## Why Use Different Setup Configurations?

Different experiments may need different hardware:

- **setup_conf_idx = 0**: Simulation mode (no real hardware, keyboard input)
- **setup_conf_idx = 1**: Basic Raspberry Pi setup with 2 lick ports and 1 reward valve
- **setup_conf_idx = 2**: Advanced setup with multiple cameras, speakers, and VR ball
- **setup_conf_idx = 3**: Simple PC setup for visual-only experiments

## What Does setup_conf_idx Define?

### 1. Main Configuration

```sql
# SetupConfiguration
setup_conf_idx      : tinyint      # unique ID number
---
interface           : enum('DummyPorts','RPPorts', 'PCPorts', 'RPVR')
description         : varchar(256)
```

**Interface types:**

- **`DummyPorts`**: Simulation mode (keyboard input, no real hardware)
- **`RPPorts`**: Raspberry Pi with GPIO pins
- **`PCPorts`**: PC with serial/USB connections
- **`RPVR`**: Raspberry Pi with virtual reality ball

### 2. Port Configuration

Defines your behavioral ports (lick detectors, reward valves, etc.):

```sql
port                   : tinyint                  # port number (1, 2, 3...)
type="Lick"            : enum('Lick','Proximity') # what the port does
-> SetupConfiguration
---
ready=0                : tinyint       # can detect "ready" state
response=0             : tinyint       # can detect responses
reward=0               : tinyint       # can deliver rewards
invert=0               : tinyint       # invert signal (0=normal, 1=inverted)
description            : varchar(256)
```

**Port types:**

- **`Lick`**: Detects when animal licks (infrared sensor, capacitive sensor, etc.)
- **`Proximity`**: Detects when animal is in position (motion sensor, etc.)

**Port flags (what each port can do):**

- **`ready=1`**: Port can detect when animal is in "ready" position for a trial
- **`response=1`**: Port can register behavioral responses (licks, touches, etc.)
- **`reward=1`**: Port can deliver rewards (water, food, etc.)
- **`invert=1`**: Flip the signal (useful if your sensor gives opposite readings)

### 3. Screen Configuration

```sql
screen_idx             : tinyint      # screen number
-> SetupConfiguration
---
intensity             : tinyint UNSIGNED    # brightness (0-255)
distance              : float               # distance from animal (cm)
center_x              : float               # screen center X position
center_y              : float               # screen center Y position
resolution_x          : smallint            # screen width (pixels)
resolution_y          : smallint            # screen height (pixels)
fps                   : tinyint UNSIGNED    # frame rate
fullscreen            : tinyint             # 0=windowed, 1=fullscreen
description           : varchar(256)
```

### 4. Camera Configuration

```sql
camera_idx            : tinyint      # camera number
-> SetupConfiguration
---
fps                   : tinyint UNSIGNED    # recording frame rate
resolution_x          : smallint            # video width
resolution_y          : smallint            # video height
video_aim             : enum('eye','body','openfield')  # what to record
description           : varchar(256)
```

### 5. Other Components
- **Ball**: For virtual reality experiments
- **Speaker**: For audio stimuli

## How to Create a New Setup Configuration

**Alternative Method**: Instead of using Python code, you can use [DBBeaver](https://dbeaver.io/) (a free database management tool) to create and modify setup configurations through a user-friendly graphical interface. This can be easier for users who prefer visual database editing over writing code.

---

### Step 1: Choose Your setup_conf_idx Number

Pick an unused number (check existing configurations first):
```python
# Check what setup_conf_idx numbers are already used
from ethopy import interface
existing = interface.SetupConfiguration().fetch('setup_conf_idx')
print("Existing configurations:", existing)

# Pick next available number
new_idx = max(existing) + 1  # e.g., if you have 0,1,2 then use 3
```

### Step 2: Add Main Configuration

```python
# Add your main setup configuration
interface.SetupConfiguration.insert1({
    'setup_conf_idx': 3,
    'interface': 'RPPorts',  # or 'DummyPorts', 'PCPorts', 'RPVR'
    'description': 'My new Raspberry Pi setup'
})
```

### Step 3: Add Port Configuration

**Important**: You MUST add at least one port configuration.

```python
# Example: 2 lick ports + 1 proximity sensor
interface.SetupConfiguration.Port.insert([
    # Port 1: Left lick port with reward
    {
        'setup_conf_idx': 3,
        'port': 1,
        'type': 'Lick',
        'ready': 0,
        'response': 1,    # Can detect licks
        'reward': 1,      # Can give rewards
        'invert': 0,
        'description': 'Left reward port'
    },
    # Port 2: Right lick port with reward
    {
        'setup_conf_idx': 3,
        'port': 2,
        'type': 'Lick',
        'ready': 0,
        'response': 1,    # Can detect licks
        'reward': 1,      # Can give rewards
        'invert': 0,
        'description': 'Right reward port'
    },
    # Port 3: Proximity sensor
    {
        'setup_conf_idx': 3,
        'port': 3,
        'type': 'Proximity',
        'ready': 1,       # Indicates animal is ready for trial
        'response': 0,
        'reward': 0,
        'invert': 0,
        'description': 'Animal proximity sensor'
    }
])
```

### Step 4: Add Screen Configuration (If Needed)

```python
# Only needed if your experiment shows visual stimuli
interface.SetupConfiguration.Screen.insert1({
    'setup_conf_idx': 3,
    'screen_idx': 1,
    'intensity': 100,        # Screen brightness
    'distance': 15.0,        # 15 cm from animal
    'center_x': 0,           # Centered horizontally
    'center_y': 0,           # Centered vertically
    'resolution_x': 1920,    # HD resolution
    'resolution_y': 1080,
    'fps': 60,               # 60 FPS
    'fullscreen': 1,         # Fullscreen mode
    'description': 'Main stimulus screen'
})
```

### Step 5: Add Camera Configuration (If Needed)

```python
# Only needed if you want to record behavior
interface.SetupConfiguration.Camera.insert1({
    'setup_conf_idx': 3,
    'camera_idx': 1,
    'fps': 30,               # 30 FPS recording
    'resolution_x': 640,
    'resolution_y': 480,
    'video_aim': 'body',     # Record body movements
    'description': 'Behavior recording camera'
})
```

## Using Your New Configuration

Once created, use your new setup in tasks:

```python
# In your task file
setup_conf_idx = 3  # Use your new configuration

# Now your task will use the hardware defined in setup_conf_idx 3
```

## Pre-built Configuration

### Default Simulation (setup_conf_idx = 0)
Perfect for testing tasks without hardware:
- **Interface**: DummyPorts (keyboard simulation)
- **Ports**: 2 lick ports + 1 proximity (controlled by arrow keys and spacebar)
- **Screen**: Basic simulation screen

## Troubleshooting

### Common Issues

1. **"Configuration not found" error**
   - Check if your `setup_conf_idx` exists in the database
   - Verify you inserted the main configuration correctly

2. **Hardware not responding**
   - Verify your `interface` type matches your actual hardware
   - For RPPorts: Check GPIO pin connections in `local_conf.json`
   - For simulation: Use DummyPorts interface

### Getting Help

- **Check existing configurations**: Look at working setups (especially setup_conf_idx = 0)
- **Start simple**: Begin with simulation mode, then add real hardware
- **Ask for help**: Contact your lab's technical support for hardware-specific issues

## Summary

The `setup_conf_idx` system lets you:

1. **Define different hardware setups** for different experiments
2. **Switch between configurations** easily (simulation vs real hardware)
3. **Share configurations** across different experimental computers
4. **Maintain consistency** in hardware setup across experiments

Each setup_conf_idx is completely independent - changing one doesn't affect others. This makes it safe to experiment with new configurations while keeping working setups intact.

## Important Note: Configuration History and Data Integrity

**Configuration parameters are automatically saved for each experimental session.** When you start a session, EthoPy copies all configuration parameters from the setup tables into the `lab_interface` schema tables connected to that specific session. This means:

- **Data integrity**: Even if you modify a setup configuration later, your historical data remains linked to the exact parameters that were used during each session
- **Reproducibility**: You can always see exactly what hardware settings were used for any past experiment
- **Safe updates**: You can safely update setup configurations without affecting the analysis of previous experiments
- **Audit trail**: All configuration changes are preserved, preventing data drift issues

This automatic parameter logging ensures that your experimental data always maintains a complete record of the hardware setup used, making your research more reproducible and reliable.

