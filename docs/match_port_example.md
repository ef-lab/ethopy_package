# Creating a Custom Experiment: Match Port Example

This guide explains how to create a custom experiment in Ethopy by examining the `match_port.py` example, which implements a 2-Alternative Forced Choice (2AFC) task.

## Understanding States and Experiments in Ethopy

### What is a State?

A State in Ethopy represents a distinct phase of an experiment with specific behaviors and transitions. States are fundamental building blocks of the experimental flow and follow these principles:

- **Encapsulated Logic**: Each state handles a specific part of the experiment (e.g., waiting for a response, delivering a reward)
- **Well-defined Transitions**: States define clear rules for when to transition to other states
- **Shared Context**: All states share experiment variables through a shared state dictionary
- **Lifecycle Methods**: States implement standard methods (`entry()`, `run()`, `next()`, `exit()`) for predictable behavior

The State pattern allows complex behavioral experiments to be broken down into manageable, reusable components that can be combined to create sophisticated experimental flows.

### Why Use States?

States provide several advantages in experimental design:

1. **Modularity**: Each state handles one specific aspect of the experiment
2. **Reusability**: States can be reused across different experiments
3. **Maintainability**: Easier to debug and modify individual states without affecting others
4. **Readability**: Experimental flow becomes clearer when broken into distinct states
5. **Robustness**: State transitions are explicitly defined, reducing the chance of unexpected behavior

### Required Components for Experiments

Every Ethopy experiment requires:

1. **Condition Tables**: DataJoint tables that define the experimental parameters
2. **Entry and Exit States**: Special states required by the StateMachine
3. **Base Experiment Class**: Inherits from both `State` and `ExperimentClass`
4. **Custom States**: Implementation of experiment-specific states

The StateMachine in Ethopy automatically finds all state classes that inherit from the base experiment class and handles the flow between them based on their `next()` method returns.

## Overview of the Match Port Experiment

The Match Port experiment challenges animals to correctly choose between ports based on stimuli. 

It implements:
- Adaptive difficulty using staircase methods
- Reward and punishment handling
- Sleep/wake cycle management
- State machine architecture for task flow

## Defining Database Tables with DataJoint

Each experiment requires defining conditions tables that store parameters. Here's how the Match Port experiment defines its table:

```python
@experiment.schema
class Condition(dj.Manual):
    class MatchPort(dj.Part):
        definition = """
        # 2AFC experiment conditions
        -> Condition
        ---
        max_reward=3000             : smallint
        min_reward=500              : smallint
        hydrate_delay=0             : int # delay hydration in minutes

        trial_selection='staircase' : enum('fixed','block','random','staircase', 'biased')
        difficulty                  : int
        bias_window=5               : smallint
        staircase_window=20         : smallint
        stair_up=0.7                : float
        stair_down=0.55             : float
        noresponse_intertrial=1     : tinyint(1)
        incremental_punishment=1    : tinyint(1)
        next_up=0                   : tinyint
        next_down=0                 : tinyint
        metric='accuracy'           : enum('accuracy','dprime')
        antibias=1                  : tinyint(1)

        init_ready                  : int
        trial_ready                 : int
        intertrial_duration         : int
        trial_duration              : int
        reward_duration             : int
        punish_duration             : int
        abort_duration              : int
        """
```

This table definition:
- Uses the `@experiment.schema` decorator to associate with the experiment database
- Creates a part table `MatchPort` under the parent `Condition` table (all parameters of the experiements are part table of the `Condition` table)
- Defines fields with default values and data types
- Documents the purpose with a comment at the top

for more details check [datajoint documentation](https://datajoint.com/docs/core/datajoint-python/latest/)

## Creating the Base Experiment Class

The main experiment class sets up the foundation for all states and defines required parameters:

```python
class Experiment(State, ExperimentClass):
    cond_tables = ["MatchPort"]
    required_fields = ["difficulty"]
    default_key = {
        "max_reward": 3000,
        "min_reward": 500,
        "hydrate_delay": 0,
        "trial_selection": "staircase",
        "init_ready": 0,
        "trial_ready": 0,
        "intertrial_duration": 1000,
        "trial_duration": 1000,
        "reward_duration": 2000,
        "punish_duration": 1000,
        "abort_duration": 0,
        "noresponse_intertrial": True,
        "incremental_punishment": 0,
        **ExperimentClass.Block().dict(),
    }

    def entry(self):
        self.logger.curr_state = self.name()
        self.start_time = self.logger.log("Trial.StateOnset", {"state": self.name()})
        self.resp_ready = False
        self.state_timer.start()
```

Key components:
- `cond_tables`: Lists the condition tables this experiment uses and define/store the parameters of the experiment.
- `required_fields`: Specifies which fields must be provided in the task file
- `default_key`: Sets default values for parameters
required_fields + default_key must have all the parameters from the Condition.MatchPort table 
- `entry()`: Common initialization for all states. The entry function initializes the state by logging the state transition in the Trial.StateOnset in the experiment schema, resetting the response readiness flag, and starting a timer to track the duration of the state.
 
## Creating Individual State Classes

Each state is implemented as a class inheriting from the base `Experiment` class. Let's analyze in detail the `Trial` state, which manages stimulus presentation and response detection:

```python
class Trial(Experiment):
    def entry(self):
        self.resp_ready = False
        super().entry()
        self.stim.start()

    def run(self):
        self.stim.present()  # Start Stimulus
        self.has_responded = self.beh.get_response(self.start_time)
        if (
            self.beh.is_ready(self.stim.curr_cond["trial_ready"], self.start_time)
            and not self.resp_ready
        ):
            self.resp_ready = True
            self.stim.ready_stim()

    def next(self):
        if not self.resp_ready and self.beh.is_off_proximity():  # did not wait
            return "Abort"
        elif (
            self.has_responded and not self.beh.is_correct()
        ):  # response to incorrect probe
            return "Punish"
        elif self.has_responded and self.beh.is_correct():  # response to correct probe
            return "Reward"
        elif self.state_timer.elapsed_time() > self.stim.curr_cond["trial_duration"]:
            return "Abort"
        elif self.is_stopped():
            return "Exit"
        else:
            return "Trial"

    def exit(self):
        self.stim.stop()  # stop stimulus when timeout
```

Let's break down the Trial state's functionality in detail:

### 1. `entry()` method

```python
def entry(self):
    self.resp_ready = False
    super().entry()
    self.stim.start()
```

- `self.resp_ready = False`: Resets the response readiness flag at the beginning of each trial
- `super().entry()`: Calls the parent class's entry method which:
  - Records the current state in the logger
  - Logs a state transition event with timestamp
  - Starts the state timer
- `self.stim.start()`: Initializes the stimulus for the current trial, activating any hardware or software components needed

### 2. `run()` method

```python
def run(self):
    self.stim.present()  # Start Stimulus
    self.has_responded = self.beh.get_response(self.start_time)
    if (
        self.beh.is_ready(self.stim.curr_cond["trial_ready"], self.start_time)
        and not self.resp_ready
    ):
        self.resp_ready = True
        self.stim.ready_stim()
```

This method is called repeatedly while in the Trial state:

- `self.stim.present()`: Updates the stimulus presentation on each iteration, which might involve:
  - Moving visual elements
  - Updating sound playback
  - Refreshing displays
  
- `self.beh.get_response(self.start_time)`: Checks if the animal has made a response since the trial started
  - Returns a boolean value indicating response status
  - The `start_time` parameter allows measuring response time relative to trial start
  
- Response readiness check:
  - `self.beh.is_ready(...)`: Determines if the animal is in position and ready for the stimulus
  - Uses `trial_ready` parameter from current condition to check timing requirements
  - If the animal is ready AND we haven't yet marked the trial as ready:
    - Sets `self.resp_ready = True` to indicate the animal is in position
    - Calls `self.stim.ready_stim()` to potentially modify the stimulus (e.g., changing color, activating a cue)

### 3. `next()` method

```python
def next(self):
    if not self.resp_ready and self.beh.is_off_proximity():  # did not wait
        return "Abort"
    elif (
        self.has_responded and not self.beh.is_correct()
    ):  # response to incorrect probe
        return "Punish"
    elif self.has_responded and self.beh.is_correct():  # response to correct probe
        return "Reward"
    elif self.state_timer.elapsed_time() > self.stim.curr_cond["trial_duration"]:
        return "Abort"
    elif self.is_stopped():
        return "Exit"
    else:
        return "Trial"
```

This method determines the next state based on the animal's behavior and timing:

- **Early Withdrawal Check**: 
  - `if not self.resp_ready and self.beh.is_off_proximity()`: Checks if the animal left before being ready
  - If true → transition to "Abort" state
  
- **Incorrect Response Check**: 
  - `elif self.has_responded and not self.beh.is_correct()`: Animal responded but to the wrong port
  - If true → transition to "Punish" state
  
- **Correct Response Check**:
  - `elif self.has_responded and self.beh.is_correct()`: Animal responded correctly
  - If true → transition to "Reward" state
  
- **Timeout Check**:
  - `elif self.state_timer.elapsed_time() > self.stim.curr_cond["trial_duration"]`: Trial ran too long
  - If true → transition to "Abort" state
  
- **Experiment Stop Check**:
  - `elif self.is_stopped()`: Checks if experiment has been externally requested to stop
  - If true → transition to "Exit" state
  
- **Default Case**:
  - `else: return "Trial"`: If none of the above conditions are met, stay in the Trial state

### 4. `exit()` method

```python
def exit(self):
    self.stim.stop()  # stop stimulus when timeout
```

- `self.stim.stop()`: Stops the stimulus presentation
  - Cleans up resources
  - Resets hardware components if necessary
  - Prepares for the next state

## State Machine Flow

The experiment implements a state machine that flows through these main states:

1. `Entry` → Initial setup
2. `PreTrial` → Prepare stimulus and wait for animal readiness
3. `Trial` → Present stimulus and detect response
4. `Reward` or `Punish` → Handle correct or incorrect responses
5. `InterTrial` → Pause between trials
6. `Hydrate` → Ensure animal drink the minimum reward amount
7. `Offtime` → Handle sleep periods
8. `Exit` → Clean up and end experiment

## How to Create Your Own Experiment

To create your own experiment:

1. Define your condition table with DataJoint
2. Create a base experiment class that inherits from `State` and `ExperimentClass`
3. Specify `cond_tables`, `required_fields`, and `default_key`
4. Implement the `entry()` method for common state initialization
5. Create individual state classes for each stage of your experiment
6. Implement the state machine flow through the `next()` methods

By following this pattern, you can create complex behavioral experiments that handle stimulus presentation, animal responses, and data logging in a structured manner.