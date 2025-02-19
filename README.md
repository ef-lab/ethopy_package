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
- **Multiple Experiment Types**: Support for various experiment paradigms (MatchToSample, Navigation, Passive Viewing, etc.)
- **Hardware Integration**: Interfaces with multiple hardware setups
- **Stimulus Control**: Various stimulus types supported (Gratings, Movies, Olfactory, 3D Objects)
- **Real-time Control**: State-based experiment control with precise timing
- **Extensible**: Easy to add new experiment types, stimuli, or behavioral interfaces

## System Architecture

The following diagram illustrates the relationship between the core modules:

<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/modules.iuml">

[Datajoint]: https://github.com/datajoint/datajoint-python

--- 

## Installation & Setup

### Requirements

- Python 3.8 or higher
- Docker (for database setup)
- Dependencies: numpy, pandas, datajoint, pygame, pillow, and more (automatically installed)

### Basic Installation

```bash
pip install ethopy
```

For optional features:
```bash
# For 3D object support
pip install "ethopy[obj]"

# For development
pip install "ethopy[dev]"

# For documentation
pip install "ethopy[docs]"
```

### Database Setup

1. Start the database container:
```bash
ethopy-setup-djdocker  # This will start a MySQL container for data storage
```

2. Configure the database connection (this tells Ethopy how to connect to the database):

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
    "source_path": "/path/to/data",
    "target_path": "/path/to/backup",
    "logging": {
        "level": "INFO",
        "filename": "ethopy.log"
    }
}
```

3. Verify database connection:
```bash
ethopy-db-connection  # Ensures Ethopy can connect to the database
```

4. Create required schemas:
```bash
ethopy-setup-schema  # Sets up all necessary database tables for experiments
```

After completing these steps, your database will be ready to store experiment data, configurations, and results.

### Running Experiments

1. **Service Mode**: Controlled by the Control table in the database
2. **Direct Mode**: Run a specific task directly

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

<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/states.iuml">

Each state has four overridable functions that control its behavior:

<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/state_functions.iuml">

#### Available Experiment Types

- **MatchPort**: Stimulus-port matching experiments
- **Passive**: Passive stimulus presentation
- **FreeWater**: Water delivery experiments
- **Calibrate**: Port calibration for water delivery

#### Configuration

Experiments require setup configuration through:
- `SetupConfiguration`
- `SetupConfiguration.Port`
- `SetupConfiguration.Screen`

Experiment parameters are defined in Python configuration files and stored in the `Task` table within the `lab_experiment` schema.

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

### 4. Core System Modules

#### Logger Module (Non-overridable)
Manages all database interactions across modules. Data is stored in three schemas:

**lab_experiments**:  
<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/experiments.iuml">

**lab_behavior**:  
<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/behavior.iuml">

**lab_stimuli**:  
<img src="http://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/ef-lab/EthoPy/master/utils/plantuml/stimuli.iuml">

#### Interface Module (Non-overridable)
Manages hardware communication and control.

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
