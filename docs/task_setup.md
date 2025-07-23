# Tasks

Tasks in Ethopy define the experimental protocol by combining experiments, behaviors, and stimuli and specifying their parameters. They serve as configuration files of the experiments.

## Task Structure

A typical task file consists of three main parts:

1. **Session Parameters**: Global settings of the experiment
2. **Stimulus/Behavior/Experiment Conditions**: Parameters of the respective condition tables
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

## Using of Task Templates in Ethopy

### Overview
The `ethopy-create-task` command generates a Python template file for an Ethopy experiment. This template includes default parameters and placeholders that you need to customize for your specific experiment.

### Generating a Template
To create a task template, run the following command in your terminal:

```bash
ethopy-create-task
```

You will be prompted to enter the module paths and class names for the experiment, behavior, and stimuli components. The generated file will include all required parameters with placeholders (`...`) that need to be filled.

#### Template Generation Process
The script follows these steps:

1. **Prompt for Module Paths and Class Names**
   - Enter the paths relative to `ethopy` for:
     - Experiment module (e.g., `experiments.match_port`)
     - Behavior module (e.g., `behaviors.multi_port`)
     - Stimulus module (e.g., `stimuli.grating`)
   - Enter corresponding class names for each module.

2. **Validate Imports**
   - The script attempts to import the specified modules and classes.
   - If an import fails, an error message is displayed.

3. **Extract Default Parameters**
   - The script retrieves the parameters from the experiment, behavior, and stimulus classes.

4. **Generate a Template File**
   - A Python file is created with structured sections:
     - **Session Parameters**: General experiment settings
     - **Experiment Setup**: Instantiating the experiment
     - **Trial Conditions**: Configuration for experiments, behaviors, and stimuli
     - **Condition Merging**: Combining all conditions for trial generation
     - **Execution**: Running the experiment

5. **Save the File**
   - The template is saved with a default filename (`task_<stimulus>_<date>.py`) or a user-specified name.

## Next Steps
After generating the template:

1. **Open the generated file** in a text editor.
2. **Fill in missing parameters** where indicated by `...`
3. **Customize trial conditions** to match your experiment's requirements.
4. **Run the script** to execute the experiment.

By following these steps, you can quickly set up an Ethopy experiment with minimal manual configuration.


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
