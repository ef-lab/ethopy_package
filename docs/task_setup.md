# Tasks

Tasks in Ethopy define the experimental protocol by combining experiments, behaviors, and stimuli with specific parameters. They serve as configuration files that specify how an experiment should run.

## Task Structure

A typical task file consists of three main parts:

1. **Session Parameters**: Global settings for the experiment
2. **Stimulus/Behavior/Experiment Conditions**: Parameters
3. **Experiment Configuration**: Setup and execution of the experiment

### Basic Structure
```python
# Import required components
from ethopy.behaviors import SomeBehavior
from ethopy.experiments import SomeExperiment
from ethopy.stimuli import SomeStimulus

# 1. Session Parameters
session_params = {
    'setup_conf_idx': 0,
    # ... other session parameters
}

# 2. Initialize Experiment
exp = SomeExperiment()
exp.setup(logger, SomeBehavior, session_params)

# 3. Define Experiment/Stimulus/Behavior Conditions
conditions = []
# ... condition setup

# 4. Run Experiment
exp.push_conditions(conditions)
exp.start()
```

## Creating Tasks

### 1. Session Parameters

Session parameters control the overall experiment behavior:

```python
session_params = {
    # Required Parameters
    'setup_conf_idx': 0,  # Setup configuration index
    
    # Optional Parameters
    'max_reward': 3000,    # Maximum reward amount
    'min_reward': 30,      # Minimum reward amount
}
```

### 2. Stimulus Conditions

Define the parameters for your stimuli:

```python
# Example from grating_test.py
key = {
    'contrast': 100,
    'spatial_freq': 0.05,        # cycles/deg
    'temporal_freq': 0,          # cycles/sec
    'duration': 5000,            # ms
    'trial_duration': 5000,      # ms
    'intertrial_duration': 0,    # ms
    'reward_amount': 8,
    # ... other stimulus parameters
}
```

### 3. Creating Conditions

Use the experiment's Block class and make_conditions method:

```python
# Create a block with specific parameters
block = exp.Block(
    difficulty=1,
    next_up=1,
    next_down=1,
    trial_selection='staircase',
    metric='dprime',
    stair_up=1,
    stair_down=0.5
)

# Create conditions
conditions = exp.make_conditions(
    stim_class=SomeStimulus(),
    conditions={**block.dict(), **key, 'other_param': value}
)
```

## Helper Functions

Ethopy provides helper functions for task creation:

### Get Parameters
```python
from ethopy.utils.task_helper_funcs import get_parameters

# Get required and default parameters for a class
parameters = get_parameters(SomeClass())
```

### Format Parameters
```python
from ethopy.utils.task_helper_funcs import format_params_print

# Pretty print parameters including numpy arrays
formatted_params = format_params_print(parameters)
```

## Example Tasks

### 1. Grating Test
Visual orientation discrimination experiment:

```python
from ethopy.behaviors.multi_port import MultiPort
from ethopy.experiments.match_port import Experiment
from ethopy.stimuli.grating import Grating

# Session setup
session_params = {
    'max_reward': 3000,
    'setup_conf_idx': 0,
}

exp = Experiment()
exp.setup(logger, MultiPort, session_params)

# Stimulus conditions
key = {
    'contrast': 100,
    'spatial_freq': 0.05,
    'duration': 5000,
}

# Port mapping
ports = {1: 0, 2: 90}  # Port number: orientation

# Create conditions
block = exp.Block(difficulty=1, trial_selection='staircase')
conditions = []
for port in ports:
    conditions += exp.make_conditions(
        stim_class=Grating(),
        conditions={
            **block.dict(),
            **key,
            'theta': ports[port],
            'reward_port': port,
            'response_port': port
        }
    )

# Run
exp.push_conditions(conditions)
exp.start()
```

## Best Practices

1. **Parameter Organization**:
        - Group related parameters together
        - Use descriptive variable names
        - Document units in comments

2. **Error Handling**:
        - Validate parameters before running
        - Use helper functions to get required parameters
        - Check for missing or invalid values

3. **Documentation**:
        - Comment complex parameter combinations
        - Document dependencies
        - Include example usage

4. **Testing**:
        - Test with different parameter combinations
        - Verify stimulus timing
        - Check reward delivery

## Common Issues

1. **Parameter Errors**:
        - Missing required parameters
        - Incorrect parameter types
        - Invalid parameter combinations

2. **Timing Issues**:
        - Incorrect duration values
        - Mismatched trial/stimulus timing
        - Intertrial interval problems

3. **Hardware Configuration**:
        - Wrong setup_conf_idx
        - Uncalibrated rewad ports
        - Missing hardware components

## Additional Resources

- [Example Tasks](https://github.com/ef-lab/ethopy_package/tree/main/src/ethopy/task)
