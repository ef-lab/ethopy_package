# Getting Started with EthoPy

This guide will walk you through the process of setting up and running your first experiment with EthoPy. After completing this guide, you'll have a solid understanding of how to configure and run basic behavioral experiments.

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher (but less than 3.12) installed
- Basic understanding of Python programming
- Docker (for database setup)
- For hardware experiments, appropriate hardware setup (Raspberry Pi, ports, etc.)

## Installation

1. **Install EthoPy package:**

   For the basic installation with core functionality:

   ```bash
   pip install ethopy
   ```

   From source (latest development version):

   ```bash
   pip install git+https://github.com/ef-lab/ethopy_package
   ```

   For development installation:

   ```bash
   git clone https://github.com/ef-lab/ethopy_package.git
   cd ethopy_package
   pip install -e ".[dev,docs]"
   ```

2. **Verify installation:**

   ```bash
   ethopy --version
   ```

## Database Setup

EthoPy relies on a database for experiment configuration and data logging. If there is not a databse availabe, here is a quick setup of setting mysql databse with docker:

1. **Start the database container:**

   ```bash
   ethopy-setup-djdocker
   ```

> **Note:** By default, Docker requires sudo because the Docker daemon runs as root.
This command adds your user to the docker group, so you can run Docker commands without sudo.

```bash
sudo usermod -aG docker $USER
```

restart your session (log out and back in) or run:
```bash
newgrp docker
```

2. **Configure the database connection:**

   Create a configuration file at:
   - Linux/macOS: `~/.ethopy/local_conf.json`
   - Windows: `%USERPROFILE%\.ethopy\local_conf.json`

   ```json
   {
       "dj_local_conf": {
           "database.host": "127.0.0.1",
           "database.user": "root",
           "database.password": "your_password",
           "database.port": 3306
       },
     "SCHEMATA": {
       "experiment": "lab_experiments",
       "stimulus": "lab_stimuli",
       "behavior": "lab_behavior",
       "recording": "lab_recordings",
       "mice": "lab_mice"
     }
   }
   ```

3. **Verify database connection:**

   ```bash
   ethopy-db-connection
   ```

4. **Create required schemas:**

   ```bash
   ethopy-setup-schema
   ```

> **Note:** For testing without a database connection, see the simplified configuration in the next section.

## Setting Up Your First Experiment

### 1. Local Configuration

Create a local configuration file to specify your database connection and hardware setup.

1. **Create a configuration file** at:
   - Linux/macOS: `~/.ethopy/local_conf.json`
   - Windows: `%USERPROFILE%\.ethopy\local_conf.json`

   Basic configuration structure:

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
     "source_path": "/path/to/data",
     "target_path": "/path/to/backup",
     "logging": {
       "level": "INFO",
       "directory": "~/.ethopy/",
       "filename": "ethopy.log"
     },
     "SCHEMATA": {
       "experiment": "lab_experiments",
       "stimulus": "lab_stimuli",
       "behavior": "lab_behavior",
       "recording": "lab_recordings",
       "mice": "lab_mice"
     }
   }
   ```

2. **For hardware experiments**, configure GPIO pin mappings (Raspberry Pi):

   ```json
   {
     "channels": {
       "Liquid": {"1": 22, "2": 23},      // Liquid reward valves
       "Lick": {"1": 17, "2": 27},        // Lick detection sensors
       "Proximity": {"1": 5, "2": 6},     // Proximity sensors
       "Odor": {"1": 24, "2": 25},        // Odor delivery valves
       "Sound": {"1": 13},                // Sound output
       "Sync": {"in": 21, "rec": 26, "out": 16},  // Synchronization pins
       "Status": 20                       // Status LED
     }
   }
   ```

3. **Programmatically accessing configuration** in your scripts:

   ```python
   from ethopy.core.config import ConfigurationManager
   
   # Initialize with default configuration
   config = ConfigurationManager()
   
   # Get a configuration value
   db_host = config.get('database.host')
   log_level = config.get('logging.level', 'INFO')  # With default value
   
   # Set a configuration value
   config.set('logging.level', 'DEBUG')
   
   # Save changes
   config.save()
   ```

### 2. Create a Simple Task

Let's create a simple task that displays a stimulus and waits for a response.

1. Create a file named `simple_task.py`:

   ```python
   # Orientation discrimination experiment
   from ethopy.behaviors.multi_port import MultiPort
   from ethopy.experiments.match_port import Experiment
   from ethopy.stimuli.grating import Grating

   # define session parameters
   session_params = {
      "trial_selection": "staircase",
      "max_reward": 3000,
      "min_reward": 30,
      "setup_conf_idx": 0,
   }

   exp = Experiment()
   exp.setup(logger, MultiPort, session_params)

   # define stimulus conditions
   key = {
      "contrast": 100,
      "spatial_freq": 0.05,  # cycles/deg
      "square": 0,  # squarewave or Guassian
      "temporal_freq": 0,  # cycles/sec
      "flatness_correction": 1,  # adjustment of spatiotemporal frequencies based on animal distance
      "duration": 5000,
      "difficulty": 1,
      "trial_duration": 5000,
      "intertrial_duration": 0,
      "reward_amount": 8,
   }

   conditions = []

   ports = {1: 0,
            2: 90}

   block = exp.Block(difficulty=1, next_up=1, next_down=1, trial_selection='staircase', metric='dprime', stair_up=1, stair_down=0.5)

   # For port 1 and theta 0
   conditions += exp.make_conditions(stim_class=Grating(),
                                     conditions={
                                       **block.dict(),
                                       **key,
                                       'theta'        : 0,
                                       'reward_port'  : 1,
                                       'response_port': 1
                                       }
                                    )

   # For port 2 and theta 90
   conditions += exp.make_conditions(stim_class=Grating(),
                                     conditions={
                                       **block.dict(),
                                       **key,
                                       'theta'        : 90,
                                       'reward_port'  : 2,
                                       'response_port': 2
                                       }
                                    )

   # run experiments
   exp.push_conditions(conditions)
   exp.start()
   ```

### 3. Run Your Experiment

Now that you have your task defined, you can run it:

```bash
ethopy -p simple_task.py --log-console
```

This will:
1. Set up your experiment with the specified configuration
2. Initialize the behavior and stimulus classes
3. Begin the experiment state machine
4. Log all events and data to the database

## Common Experiment Types

EthoPy comes with several built-in experiment types that you can use or customize:

1. **FreeWater**: Simple reward delivery when an animal licks a port
   ```python
   from ethopy.experiments.freewater import Experiment
   ```

2. **MatchPort**: Associate specific stimuli with reward ports
   ```python
   from ethopy.experiments.match_port import Experiment
   ```

3. **Passive**: Present stimuli without requiring responses
   ```python
   from ethopy.experiments.passive import Experiment
   ```

4. **Calibrate**: Calibrate water delivery amounts
   ```python
   from ethopy.experiments.calibrate import Experiment
   ```

Each experiment type has a predefined state machine with states like Entry, Trial, Response, Reward, and Exit. These states control the flow of the experiment.

## Behavior and Stimulus Classes

EthoPy's modular design allows you to combine different experiment types with behavior and stimulus classes:

### Behavior Types
- **MultiPort**: Standard setup with lick, liquid delivery, and proximity ports
  ```python
  from ethopy.behaviors,multi_port import MultiPort
  ```

- **MultiPort**: Standard setup with lick, liquid delivery, and proximity ports
  ```python
  from ethopy.behaviors.head_fixed import HeadFixed
  ```

### Stimulus Types
- **Grating**: Orientation gratings
  ```python
  from ethopy.stimuli.grating import Grating
  ```
- **Bar**: Moving bar for retinotopic mapping
  ```python
  from ethopy.stimuli.bar import Bar
  ```
- **Dot**: Moving dot patterns
  ```python
  from ethopy.stimuli.dot import Dot
  ```

## Next Steps

After successfully running your first experiment, you can:

1. **Customize your task** by modifying parameters
2. **Create a new experiment type** by subclassing from the base classes
3. **Add hardware interfaces** for real behavioral experiments
4. **Configure database logging** for data analysis

## State Machine Architecture

Understanding the state machine is crucial for working with EthoPy:

1. Each experiment has a **StateMachine** dictionary mapping state names to methods
2. The current state is tracked in the **State** variable
3. Each state function is called repeatedly until it changes the State
4. Every state function can have four special methods:
   - **entry_state**: Called once when entering the state
   - **run_state**: Called repeatedly while in the state
   - **exit_state**: Called once when exiting the state
   - **state**: Combined method that manages all the above

## Sample Projects and Templates

Explore these sample projects in the `ethopy/task/` directory:

1. **calibrate_ports.py** - Calibrate water delivery ports
2. **free_water.py** - Reward delivery without complex stimuli
3. **no_stimulus.py** - Run experiment without visual stimuli
4. **grating_test.py** - Grating stimulus presentation
5. **bar_test.py** - Moving bar stimulus
6. **dot_test.py** - Moving dot patterns

## Troubleshooting and Help

If you encounter issues, refer to the [Troubleshooting Guide](troubleshooting.md).

For specific questions, check the:
- [API Reference](logger.md) for detailed module documentation
- [GitHub Issues](https://github.com/ef-lab/ethopy_package/issues) for known problems

---

## Where to Go Next

Now that you have a basic understanding of EthoPy:

1. Dive deeper into [Local Configuration](local_conf.md) for advanced settings
2. Learn about [Database Setup](database.md) for data storage
3. Explore the [Plugin System](plugin.md) to extend functionality
4. Study the [API Reference](logger.md) for detailed documentation
5. Check [Contributing](contributing.md) if you want to help improve EthoPy