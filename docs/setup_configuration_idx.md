# Setup Configuration in EthoPy

The setup configuration system in EthoPy is managed through the `setup_conf_idx` (Setup Configuration Index) and associated tables. This system allows users to define and manage different hardware configurations for experimental setups.

## Setup Configuration Index (`setup_conf_idx`)

The `setup_conf_idx` is a unique identifier that links together all components of a particular setup configuration. It's defined in the main `SetupConfiguration` table and referenced by all related configuration tables.

### Main Configuration Table

```sql
# SetupConfiguration
setup_conf_idx      : tinyint      # configuration version
---
interface           : enum('DummyPorts','RPPorts', 'PCPorts', 'RPVR')
discription         : varchar(256)
```

## Component Configuration Tables

### 1. Port Configuration (`SetupConfiguration.Port`)

Defines the configuration for input/output ports.

```sql
port                   : tinyint                  # port id
type="Lick"            : enum('Lick','Proximity') # port type
-> SetupConfiguration
---
ready=0                : tinyint       # ready flag
response=0             : tinyint       # response flag
reward=0               : tinyint       # reward flag
invert=0               : tinyint       # invert flag
discription            : varchar(256)
```

#### Port Types and Flags
- **Types**: 
    - `Lick`: For lick detection
    - `Proximity`: For proximity detection
- **Flags**:
    - `ready`: Port can be used for is_ready function which indicate that the port is being activate for specific duration
    - `response`: Port can register responses
    - `reward`: Port can deliver rewards
    - `invert`: Invert port signal

### 2. Screen Configuration (`SetupConfiguration.Screen`)

Defines display settings for visual stimuli.

```sql
screen_idx             : tinyint
-> SetupConfiguration
---
intensity             : tinyint UNSIGNED
distance              : float
center_x              : float
center_y              : float
aspect                : float
size                  : float
fps                   : tinyint UNSIGNED
resolution_x          : smallint
resolution_y          : smallint
description           : varchar(256)
fullscreen            : tinyint
```

### 3. Ball Configuration (`SetupConfiguration.Ball`)

Defines settings for ball-based interfaces (e.g., virtual reality).

```sql
-> SetupConfiguration
---
ball_radius=0.125     : float                   # in meters
material="styrofoam"  : varchar(64)             # ball material
coupling="bearings"   : enum('bearings','air')  # mechanical coupling
discription           : varchar(256)
```

### 4. Speaker Configuration (`SetupConfiguration.Speaker`)

Defines audio output settings.

```sql
speaker_idx           : tinyint
-> SetupConfiguration
---
sound_freq=10000     : int           # in Hz
duration=500         : int           # in ms
volume=50            : tinyint       # 0-100 percentage
discription         : varchar(256)
```

### 5. Camera Configuration (`SetupConfiguration.Camera`)

Defines camera settings for behavioral recording.

```sql
camera_idx            : tinyint
-> SetupConfiguration
---
fps                   : tinyint UNSIGNED
resolution_x          : smallint
resolution_y          : smallint
shutter_speed         : smallint
iso                   : smallint
file_format           : varchar(256)
video_aim             : enum('eye','body','openfield')
discription           : varchar(256)
```

## Creating a New Setup Configuration

To create a new setup configuration:

1. **Add Main Configuration Entry**
```python
# Add to SetupConfiguration.contents
[
    setup_conf_idx,    # Unique identifier
    interface_type,    # e.g., "RPPorts"
    description        # Setup description
]
```

2. **Add Component Configurations**
```python
# Example: Adding port configuration
SetupConfiguration.Port.insert1({
    'setup_conf_idx': your_idx,
    'port': port_number,
    'type': 'Lick',
    'ready': 0,
    'response': 1,
    'reward': 1,
    'invert': 0,
    'discription': 'Reward port'
})
```

3. **Add Required Components**
   - Add entries to relevant component tables (Screen, Camera, etc.)
   - Each component must reference the same `setup_conf_idx`

## Usage Example

```python
# Example of a complete setup configuration
setup_config = {
    'setup_conf_idx': 1,
    'interface': 'RPPorts',
    'discription': 'Raspberry Pi Setup'
}

port_config = {
    'setup_conf_idx': 1,
    'port': 1,
    'type': 'Lick',
    'ready': 1,
    'response': 1,
    'reward': 1,
    'discription': 'Main reward port'
}

screen_config = {
    'setup_conf_idx': 1,
    'screen_idx': 1,
    'intensity': 100,
    'distance': 10.0,
    'resolution_x': 1920,
    'resolution_y': 1080,
    'fullscreen': 1
}
```

## Troubleshooting

- **Component Initialization Failures**
    - Check `setup_conf_idx` references
    - Verify hardware connections

## Default Configuration

EthoPy includes a default simulation configuration:
```python
# Default setup (setup_conf_idx = 0)
contents = [
    [0, "DummyPorts", "Simulation"]
]

# Default ports
Port.contents = [
    [1, "Lick", 0, 0, 1, 1, 0, "probe"],
    [2, "Lick", 0, 0, 1, 1, 0, "probe"],
    [3, "Proximity", 0, 1, 0, 0, 0, "probe"]
]

# Default screen
Screen.contents = [
    [1, 0, 64, 5.0, 0, -0.1, 1.66, 7.0, 30, 800, 480, "Simulation", 0]
]
```