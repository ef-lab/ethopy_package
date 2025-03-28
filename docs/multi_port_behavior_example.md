# Creating a Custom Behavior: MultiPort Example

This guide explains how to create a custom behavior handler in Ethopy by examining the `multi_port.py` example, which implements a behavior system for experiments with multiple response and reward ports.

## Overview of the MultiPort Behavior

The MultiPort behavior manages interactions with multiple response ports in an experimental setup. It's designed for:

- Tracking which ports an animal interacts with
- Validating responses based on experiment conditions
- Managing reward delivery at specific ports
- Logging behavioral data throughout the experiment

This behavior is particularly useful for:
- Two-alternative forced choice (2AFC) tasks
- Multiple-choice experiments
- Complex reward contingency studies
- Tasks requiring specific response sequences

## Defining the Behavior Database Tables

The MultiPort behavior uses multiple tables to define its parameters:

```python
@behavior.schema
class MultiPort(Behavior, dj.Manual):
    definition = """
    # This class handles the behavior variables for RP
    ->behavior.BehCondition
    """

    class Response(dj.Part):
        definition = """
        # Lick response condition
        -> MultiPort
        response_port              : tinyint          # response port id
        """

    class Reward(dj.Part):
        definition = """
        # reward port conditions
        -> MultiPort
        ---
        reward_port               : tinyint          # reward port id
        reward_amount=0           : float            # reward amount
        reward_type               : varchar(16)      # reward type
        """
```

This database definition consists of:

1. **Main MultiPort table**:
   - Inherits from both the `Behavior` base class and `dj.Manual` (DataJoint)
   - Links to the parent `behavior.BehCondition` table with a foreign key

2. **Response part table**: 
   - Defines which ports can be used for responses
   - Contains the `response_port` field that identifies valid response ports

3. **Reward part table**:
   - Specifies which ports can deliver rewards
   - Defines the reward amount and type for each port
   - Links back to the parent MultiPort table

## Implementing the Behavior Class

The MultiPort class implements several key methods that manage the animal's interaction with the experimental apparatus:

```python
def __init__(self):
    super().__init__()
    self.cond_tables = ["MultiPort", "MultiPort.Response", "MultiPort.Reward"]
    self.required_fields = ["response_port", "reward_port", "reward_amount"]
    self.default_key = {"reward_type": "water"}
```

### 1. `__init__()` method

This initializes the behavior handler and specifies:

- `self.cond_tables`: List of tables used for conditions (main table and part tables)
- `self.required_fields`: Parameters that must be provided for this behavior
- `self.default_key`: Default values for optional parameters (like reward type)

### 2. `is_ready()` method

```python
def is_ready(self, duration, since=False):
    position, ready_time, tmst = self.interface.in_position()
    if duration == 0:
        return True
    elif position == 0 or position.ready == 0:
        return False
    elif not since:
        return ready_time > duration  # in position for specified duration
    elif tmst >= since:
        # has been in position for specified duration since timepoint
        return ready_time > duration
    else:
        # has been in position for specified duration since timepoint
        return (ready_time + tmst - since) > duration
```

This method determines if the animal is properly positioned and ready for the experiment:

1. Gets position information from the interface
2. Handles different timing scenarios:
   - If duration is 0, animal is always considered ready
   - If no position is detected or the position is not ready, returns False
   - Checks if the animal has been in position for the required duration
   - If a 'since' timestamp is provided, calculates readiness relative to that time

### 3. `is_correct()` method

```python
def is_correct(self):
    """Check if the response port is correct.

    if current response port is -1, then any response port is correct
    otherwise if the response port is equal to the current response port/ports,
    then it is correct

    Returns:
        bool: True if correct, False otherwise

    """
    return self.curr_cond['response_port'] == -1 or \
        np.any(np.equal(self.response.port, self.curr_cond['response_port']))
```

This method validates if the animal's response was correct:
- If the condition's response_port is -1, any response is considered correct
- Otherwise, checks if the animal's response port matches the expected port in the current condition
- Uses numpy's array comparison to support multiple correct ports

### 4. `reward()` method

```python
def reward(self, tmst=0):
    """Give reward at latest licked port.

    After the animal has made a correct response, give the reward at the
    first port that animal has licked and is definded as reward.

    Args:
        tmst (int, optional): Time in milliseconds. Defaults to 0.

    Returns:
        bool: True if rewarded, False otherwise

    """
    # check that the last licked port is also a reward port
    licked_port = self.is_licking(since=tmst, reward=True)
    if licked_port == self.curr_cond["reward_port"]:
        self.interface.give_liquid(licked_port)
        self.log_reward(self.reward_amount[self.licked_port])
        self.update_history(self.response.port, self.reward_amount[self.licked_port])
        return True
    return False
```

This method handles reward delivery:

1. Gets the port the animal is licking (since a specified time)
2. If the licked port matches the reward port in the current condition:
   - Triggers the hardware to deliver liquid reward through the interface
   - Logs the reward amount
   - Updates the history with the response port and reward amount
   - Returns True (reward was given)
3. Returns False if no reward was given

### 5. `punish()` method

```python
def punish(self):
    port = self.response.port if self.response.port > 0 else np.nan
    self.update_history(port, punish=True)
```

This method handles punishment for incorrect responses:
- Gets the response port (or NaN if no response)
- Updates the history to record the punishment event

### 6. `exit()` method

```python
def exit(self):
    super().exit()
    self.interface.cleanup()
```

This method cleans up when the behavior handler is no longer needed:
- Calls the parent class's exit method
- Tells the interface to perform cleanup operations (e.g., closing ports, stopping pumps)

## Behavior Handler Lifecycle

The MultiPort behavior handler follows this lifecycle:

1. **Initialization**: Set up parameters, define required fields, set defaults
2. **Setup**: Connect to experiment, logger, and interface components
3. **Operation**: During the experiment, it:
   - Checks if the animal is ready
   - Validates responses
   - Delivers rewards for correct responses
   - Records punishments for incorrect responses
4. **Cleanup**: Release resources when the experiment ends

## How to Create Your Own Behavior Handler

To create your own behavior handler:

1. **Define database tables** appropriate for your experimental paradigm
   - Consider what behavioral parameters need to be configured
   - Create part tables for related groups of parameters

2. **Create a behavior class** that inherits from the base `Behavior` class
   - Specify required fields and defaults in `__init__`
   - Implement core behavior methods based on your experiment's needs

3. **Implement these essential methods**:
   - `is_ready()`: Determine if the animal is positioned and ready
   - `is_correct()`: Validate if the animal's response was correct
   - `reward()`: Handle reward delivery for correct responses
   - `punish()`: Handle consequences for incorrect responses
   - `exit()`: Clean up resources when done

By following this pattern, you can create behavior handlers for diverse experimental paradigms, from simple single-port setups to complex multi-stage behavioral tasks.