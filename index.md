# Ethopy

[![PyPI Version](https://img.shields.io/pypi/v/ethopy.svg)](https://pypi.python.org/pypi/ethopy)
[![Python Versions](https://img.shields.io/pypi/pyversions/ethopy.svg)](https://pypi.org/project/ethopy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ethopy is a state control system for automated, high-throughput behavioral training based on Python. It provides a flexible framework for designing and running behavioral experiments with:

- Tight integration with database storage & control using [Datajoint]
- Cross-platform support (Linux, macOS, Windows)
- Optimized for Raspberry Pi boards
- Modular architecture with overridable components
- Built-in support for various experiment types, stimuli, and behavioral interfaces

## Features

- **Modular Design**: Comprised of several overridable modules that define the structure of experiments, stimuli, and behavioral control
- **Database Integration**: Automatic storage and management of experimental data using Datajoint
- **Multiple Experiment Types**: Support for various experiment paradigms (MatchToSample, Navigation, Passive Viewing, etc.)
- **Hardware Integration**: Interfaces with multiple hardware setups (MultiPort, VRBall, Touchscreen)
- **Stimulus Control**: Various stimulus types supported (Gratings, Movies, Olfactory, 3D Objects)
- **Real-time Control**: State-based experiment control with precise timing
- **Extensible**: Easy to add new experiment types, stimuli, or behavioral interfaces

## System Architecture

The following diagram illustrates the relationship between the core modules:

<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/modules.iuml">

[Datajoint]: https://github.com/datajoint/datajoint-python

--- 

## Core modules:

### Experiment
Main state experiment Empty class that is overriden by other classes depending on the type of experiment.

This class can have various State classes. An Entry and Exit State are necessary, all the rest can be customized.
 
A typical experiment state diagram:

<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/states.iuml">

Each of the states is discribed by 4 overridable funcions:

<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/state_functions.iuml">

Tables that are needed for the experiment that discribe the setup:

> SetupConfiguration  
> SetupConfiguration.Port  
> SetupConfiguration.Screen

The experiment parameters are specified in *.py script configuration files that are entered in the Task table within the lab_experriment schema.
 Some examples are in the conf folder but any folder that is accessible to the system can be used. Each protocol has a unique task_idx identifier that is used uppon running. 

Implemented experiment types:  
* MatchToSample: Experiment with Cue/Delay/Response periods 
* MatchPort: Stimulus matched to ports
* Navigate: Navigation experiment
* Passive: Passive stimulus presentation experiment
* FreeWater: Free water delivery experiment
* Calibrate: Port Calibration of water amount
* PortTest: Testing port for water delivery

### Behavior
Empty class that handles the animal behavior in the experiment.  

IMPORTANT: Liquid calibration protocol needs to be run frequently for accurate liquid delivery

Implemented Behavior types:
* MultiPort:  Default RP setup with lick, liquid delivery and proximity port
* VRBall (beta): Ball for 2D environments with single lick/liquid delivery port
* Touch (beta): Touchscreen interface

### Stimulus
Empty class that handles the stimuli used in the experiment.

Implemented stimulus types:
* Grating: Orientation gratings
* Bar: Moving bar for retinotopic mapping
* Movies: Movie presentation
* Olfactory: Odor persentation
* Panda: Object presentation
* VROdors: Virtual environment with odors
* SmellyObjects: Odor-Visual objects


Non-overridable classes:
### Logger (non-overridable)
Handles all database interactions and it is shared across Experiment/Behavior/Stimulus classes
non-overridable

Data are storred in tables within 3 different schemata that are automatically created:

lab_experiments:  
<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/experiments.iuml">
  

lab_behavior:  
<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/behavior.iuml">
  
lab_stimuli:  
<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/stimuli.iuml">

### Interface (non-overridable)
Handles all communication with hardware

---

## How to run

You can run experiments in two modes:

1. **Service Mode**: Controlled by the Control table in the database
2. **Direct Mode**: Run a specific task directly

Example of running a task:
```bash
# Run a grating test experiment
ethopy -p grating_test.py

# Run a specific task by ID
ethopy --task-idx 1
```

This process can be automated by either a bash script that runs on startup or through control from a salt server. 
