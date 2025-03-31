# Ethopy

[![PyPI Version](https://img.shields.io/pypi/v/ethopy.svg)](https://pypi.python.org/pypi/ethopy)
[![Python Versions](https://img.shields.io/pypi/pyversions/ethopy.svg)](https://pypi.org/project/ethopy/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://ef-lab.github.io/ethopy_package/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ethopy is a state control system for automated, high-throughput behavioral training based on Python. It provides a flexible framework for designing and running behavioral experiments with:

- Tight integration with database storage & control using [Datajoint](https://docs.datajoint.org/python/)
- Cross-platform support (Linux, macOS, Windows)
- Optimized for Raspberry Pi boards
- Modular architecture with overridable components
- Built-in support for various experiment types, stimuli, and behavioral interfaces

The full documentation is available at:

👉 [Documentation](https://ef-lab.github.io/ethopy_package/)

## Features

- **Modular Design**: Comprised of several overridable modules that define the structure of experiments, stimuli, and behavioral control
- **Database Integration**: Automatic storage and management of experimental data using Datajoint
- **Multiple Experiment Types**: Support for various experiment paradigms (match to sample, 2AFC, open field, etc.)
- **Hardware Integration**: Interfaces with multiple hardware setups (raspberry, arduino, desktop computer, screen, camera etc.)
- **Stimulus Control**: Various stimulus types supported (Gratings, Movies, Olfactory, 3D Objects)
- **Real-time Control**: State-based experiment control with precise timing
- **Extensible**: Easy to add new experiment types, stimuli, or behavioral interfaces

## System Architecture

The following diagram illustrates the relationship between the core modules:

<img src="docs/plantuml/modules_uml.png">

[Datajoint]: https://github.com/datajoint/datajoint-python

--- 

## Installation & Setup

### Requirements

- Python 3.8 or higher
- Maria DB Database (instructions for [database setup](database.md))


### Basic Installation

```bash
pip install ethopy
```

### Running Experiments

Example of running a task:
```bash
# Run a grating test experiment
ethopy -p grating_test.py

# Run a specific task by ID
ethopy --task-idx 1
```

---

## Core Architecture

Understanding Ethopy's core architecture is essential for both using the system effectively and extending it for your needs. Ethopy is built around four core modules that work together to provide a flexible and extensible experimental framework. Each module handles a specific aspect of the experiment, from controlling the overall flow to managing stimuli and recording behavior.

### 1. Experiment Module

The base experiment module defines the state control system. Each experiment is composed of multiple states, with Entry and Exit states being mandatory.

#### Example of a State Machine Diagram

This diagram illustrates a simple state machine, a computational model that transitions between discrete states in response to inputs or events. State machines are used to model systems with a finite number of possible states and well-defined transitions between them.
- Entry: The initial state, marking the start of the process.
- PreTrial: A preparatory state before the actual trial begins.
- Trial: The main state where the trial takes place. It can loop back to itself, representing iterative actions within the trial.
- Abort, Reward, Punish: Possible outcomes of the trial, leading to different branches.
- InterTrial: A state between trials, possibly for processing results or preparing for the next trial.
- Exit: The final state, indicating the end of the process.

<div style="text-align: left;">
  <p style="font-weight: bold; margin-bottom: 10px;">State Machine Diagram</p>
  <img src="docs/plantuml/states_uml.png" alt="Image Description" style="max-width: 500px;">
</div>

Each state has four overridable functions that control its behavior:
<div style="text-align: left;">
  <p style="font-weight: bold; margin-bottom: 10px;">State Functions</p>
  <img src="docs/plantuml/state_functions_uml.png" alt="Image Description" style="max-width: 500px;">
</div>


#### Available Experiment Types

- **MatchPort**: Stimulus-port matching experiments
- **Passive**: Passive stimulus presentation
- **FreeWater**: Water delivery experiments
- **Calibrate**: Port calibration for water delivery

### 2. Behavior Module

Handles animal behavior tracking and response processing.

#### Available Behavior Types

- **MultiPort**: Standard setup with lick detection, liquid delivery, and proximity sensing
- **HeadFixed**: Passive head fixed setup
> **Important**: Regular liquid calibration is essential for accurate reward delivery. We recommend calibrating at least once per week to ensure consistent reward volumes and reliable experimental results.

### 3. Stimulus Module

Controls stimulus presentation and management.

#### Available Stimulus Types

- **Visual**
  - Grating: Orientation gratings
  - Bar: Moving bars for retinotopic mapping
  - Dot: Moving dots


### Logger Module (Non-overridable)
Manages all database interactions across modules. Data is stored in three schemas:

**lab_experiments**:  
<img src="docs/plantuml/experiment_uml.png">

**lab_behavior**:  
<img src="docs/plantuml/behavior_uml.png">

**lab_stimuli**:  
<img src="docs/plantuml/stimuli_uml.png">

**interface**:  
<img src="docs/plantuml/interface_uml.png">

### Interface Module (Non-overridable)
Manages hardware interactions and the configuration of hardware based on setup index.

#### Available Interfaces
- **Arduino**: Interfaces with Arduino microcontrollers for reward delivery, lick detection, and proximity sensing
- **Ball**: Tracks animal movement on a spherical treadmill using two mice for position, orientation, and speed
- **Camera**: Manages camera recordings with timestamping and HTTP streaming capabilities
- **dlc**: DeepLabCut integration for real-time animal pose tracking and arena detection
- **DummyPorts**: Simulates animal interaction ports for testing and development without physical hardware
- **PCPorts**: Controls PC hardware interfaces via serial connections for synchronization and stimulation
- **RPPorts**: Interfaces with Raspberry Pi GPIO pins to control lick ports, valves, sensors, and stimulation
- **RPVR**: Extends RPPorts for virtual reality experiments with additional odor control capabilities


## Development & Contributing

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ef-lab/ethopy_package/  # Main repository
cd ethopy
```

2. Install development dependencies:
```bash
pip install -e ".[dev,docs]"
```

### Code Quality

The project uses several tools to maintain code quality:

- **ruff**: Code formatting and linting
- **isort**: Import sorting
- **mypy**: Static type checking
- **pytest**: Testing and test coverage

Run tests:
```bash
pytest tests/
```

### Documentation

Documentation is built using MkDocs. Install documentation dependencies and serve locally:

```bash
pip install ".[docs]"
mkdocs serve
```

### License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ef-lab/ethopy_package/blob/master/LICENSE) file for details.

### Support

For questions and support:
- Open an issue on [GitHub](https://github.com/ef-lab/ethopy_package/issues)
- Check the [full documentation](https://ef-lab.github.io/ethopy_package/)
