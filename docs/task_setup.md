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


## Task Identification and Database Integration

### The `task_idx` System

Ethopy uses a `task_idx` (task index) system to uniquely identify and manage experiment configurations. This index serves as the primary key linking tasks across the Control and Task database tables.

#### How `task_idx` Works

1. **Task Table Storage**:

    - Each experimental configuration is stored in the `Task` table with a unique `task_idx`
    - The `Task` table contains:
        - `task_idx` (primary key): Unique identifier for the task
        - `path`: The actual task file
        - `description`: Human-readable description of the task
        - `timestamp`: The timestamp of the task creation

2. **Control Table Usage**:

    - The `Control` table uses `task_idx` to specify which experiment configuration to run
    - When you set `task_idx` in the Control table, the system loads the corresponding task configuration file


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

### 4. How Conditions Are Made

A **condition** is one fully specified trial: a single dictionary that holds one value for every stimulus, behavior, and experiment parameter. You rarely write conditions out one by one. Instead you pass `make_conditions` a compact dictionary that describes *ranges* of values, and Ethopy expands (factorizes) it into the full set of individual conditions.

#### What `make_conditions` does with your dictionary

When you call `make_conditions`, the single dictionary you pass is split by parameter ownership and factorized in three independent groups:

1. **Stimulus** parameters (the fields of your stim_class, e.g. contrast, spatial_freq).
2. **Behavior** parameters (the fields of the behavior class, e.g. reward and port settings).
3. **Experiment** parameters (the Block fields and other experiment-level settings).

Each group is factorized on its own, and then the three groups are combined by a final **cartesian product**. So the total number of conditions is:

```
n_stimulus  ×  n_behavior  ×  n_experiment
```

This split is why the same flat dictionary can carry stimulus, behavior, and block parameters at once, each parameter is routed to the class that owns it, and any key that no class claims is still attached to every condition.

#### The factorization rule

Within each group, expansion is done by the `factorize` helper, which generates the **cartesian product** of the parameters so you can define many conditions with a single call.

A single rule governs the expansion:

> A `list` value is treated as a set of alternatives to expand over. Any other type  (`tuple`, `int`, `float`, `str`) is treated as a single fixed value.

| You write | Interpreted as | Result |
|-----------|----------------|--------|
| `'x': [1, 2, 3]` | three alternatives | three conditions (`x=1`, `x=2`, `x=3`) |
| `'x': 5` | one fixed value | one condition |
| `'x': (1, 2)` | one fixed value (a tuple) | one condition, `x` stays `(1, 2)` |

For example:

```python
factorize({'freq': [1, 2], 'contrast': [10, 20], 'duration': 500})
# 2 x 2 x 1 = 4 conditions:
#   {'freq': 1, 'contrast': 10, 'duration': 500}
#   {'freq': 1, 'contrast': 20, 'duration': 500}
#   {'freq': 2, 'contrast': 10, 'duration': 500}
#   {'freq': 2, 'contrast': 20, 'duration': 500}
```

#### Fields that must stay together (lists vs. tuples)

Some condition tables have fields that represent **one value made of several aligned elements** rather than a set of alternatives. A common case is a color field such as `bg_level`, an `[R, G, B]` triple whose three elements must stay together as a single value.

For these fields use a **tuple**, so `factorize` keeps them intact instead of expanding them:

```python
'bg_level': (1, 1, 1),   # ONE color (white)
'bg_level': [1, 1, 1],   # THREE separate conditions: 1, 1, 1
```

To vary over several such arrays in one call, wrap the tuples (or inner lists) in an outer list. The outer list is the set of alternatives, each inner array becomes one condition's aligned value (inner lists are converted to tuples automatically):

```python
'bg_level': [[1, 1, 1]]            # one condition,  bg_level = (1, 1, 1)
'bg_level': [[1, 1, 1], [0, 0, 0]] # two conditions: (1, 1, 1) and (0, 0, 0)
```

You can see this pattern in the shipped `dot_test.py` example task, where each stimulus keeps an aligned RGB triple as a single value while other fields expand:

```python
# src/ethopy/task/dot_test.py
key = {
    'bg_level' : [[1, 1, 1]],
    'dot_level': [[0, 0, 0]],
    'dot_x'    : list(np.linspace(-.45, .45, 10)),  # 10 alternatives
    'dot_y'    : list(np.linspace(-.27, .27, 6)),   # 6 alternatives
    # ...
}
# 10 x 6 = 60 conditions; bg_level / dot_level stay intact as single RGB triples
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
---

**Note**: In your tasks, the `setup_conf_idx` parameter defines which hardware configuration your experiment will use. Learn more about configuring different hardware setups in the [Setup Configuration Index](setup_configuration_idx.md) guide.

## Additional Resources

- [Example Tasks](https://github.com/ef-lab/ethopy_package/tree/main/src/ethopy/task)
- [Plugins](https://github.com/ef-lab/ethopy_plugins/)

