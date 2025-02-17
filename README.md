# Ethopy
State control system for automated, high-throughput behavioral training based on Python. 
It is tightly integrated with Database storage & control using the [Datajoint] framework. 
It can run on Linux, MacOS, Windows and it is optimized for use with Raspberry pi boards. 

It is comprised of several overridable modules that define the structure of experiment, stimulus and behavioral control.

[![image](https://img.shields.io/pypi/v/ethopy.svg)](https://pypi.python.org/pypi/ethopy)

A diagram that illustrates the relationship between the core modules:

<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/modules.iuml">

[Datajoint]: https://github.com/datajoint/datajoint-python

--- 

## Intallation: how to use EthoPy
Can be run either as a service that is controled by the Control table
```bash
pip install ethopy
```

### setup database in docker
```bash
ethopy-setup-djdocker
```
### Define local configuration:
The configuration file by default is stored in:
- Linux/macOS: `~/.ethopy/local_conf.json`
- Windows: `%USERPROFILE%\.ethopy\local_conf.json`

#### Basic Configuration Structure
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
        "filename": "ethopy.log"
    }
}
```

### Check the connection with the database:
```
ethopy-db-connection
```

### create schema
```bash
ethopy-setup-schema
```

### Run a task on local machine for test
or can specify a task_idx to run directly. After it completes, the process ends.
```bash
ethopy -p grating_test.py
```

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

