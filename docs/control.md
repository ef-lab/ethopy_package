# Using of the Control Table

The Control table is a critical component in EthoPy that manages experiment execution and setup status. It's part of the `lab_experiments` schema and is used primarily when running EthoPy in Service Mode.

## Control Table Fields

The Control table contains the following important fields:

1. `setup` (primary key)
      - The hostname of the machine running the experiment
      - Used to identify different experimental setups

2. `status`
      - Current status of the setup
      - Possible values:
         - "ready" - Setup is in Welcome gui and ready for a new experiment
         - "running" - Experiment is currently running
         - "stop" - Request to stop the current experiment
         - "exit" - An error has occured and it is in exit

3. `last_ping`
      - Timestamp of the last status update
      - Format: "YYYY-MM-DD HH:MM:SS"
      - Updated every 5 seconds by default

4. `queue_size`
      - Number of pending operations in the queue
      - Indicates the backlog of data waiting to be written to the database

5. `trials`
      - Current trial index in the session
      - Tracks progress through the experiment

6. `total_liquid`
      - Total amount of reward delivered in the session
      - Used for tracking reward delivery

7. `state`
      - Current state of the experiment
      - Reflects which part of the experiment is currently executing (check experiment states)

8. `task_idx`
      - Index of the task to be executed
      - Used to determine which experiment configuration to load

## How to Use the Control Table

### 1. Service Mode Operation

The Control table is automatically updated by the Logger class. You don't need to modify it directly in most cases.

### 2. Monitoring Experiment Status

```python
# Example of checking setup status
control_entry = (experiment.Control & {'setup': setup_name}).fetch1()
current_status = control_entry['status']
current_state = control_entry['state']
```

### 3. Controlling Experiments
The user only change the status of the experiment from running to stop and from ready to running. Also can change the animal_id and the task_id.

```python
# To start an experiment on a setup
experiment.Control.update1({
    'setup': setup_name,
    'status': 'running',
    'task_idx': your_task_id
})

# To stop an experiment
experiment.Control.update1({
    'setup': setup_name,
    'status': 'stop'
})
```

## Important Notes

1. **Automatic Updates**: The Control table is automatically updated by the Logger class every 5 seconds (default update_period = 5000ms)

2. **Status Flow**:
      - Normal flow: ready -> running
      - Stop flow: running -> stop -> ready
      - Exit flow: any_status (raised error) -> exit

3. **Error Handling**:
      - If an error occurs during experiment execution, the state field will show "ERROR!"
      - Additional error details will be stored in the notes field

4. **Monitoring**:
      - The `last_ping` field can be used to monitor if a setup is active
      - If a setup hasn't updated its status for a long time, it might indicate issues

5. **Thread Safety**:
      - All Control table operations are thread-safe
      - Updates are protected by a thread lock to prevent race conditions

## Example Usage in Service Mode

```python
from ethopy.core.logger import Logger

# Initialize logger
logger = Logger()

# Logger will automatically:
# 1. Create an entry in Control table for this setup
# 2. Monitor Control table for status changes
# 3. Update setup status every 5 seconds
# 4. Execute tasks based on task_idx when status is 'running'
```

## Implementation Details

The Control table is managed primarily by the Logger class (`ethopy.core.logger.Logger`). Key implementation details include:

1. **Status Synchronization**:
      - The `_sync_control_table` method runs in a separate thread
      - Updates occur every 5 seconds by default
      - Uses thread locks to ensure thread-safe operations

2. **Setup Information Updates**:
   ```python
   # Example of information updated in each cycle
   info = {
       'last_ping': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
       'queue_size': self.queue.qsize(),
       'trials': self.trial_key['trial_idx'],
       'total_liquid': self.total_reward,
       'state': self.curr_state,
   }
   ```

3. **Error Recovery**:
      - The system includes automatic error recovery mechanisms
      - Failed database operations are retried with increased priority
      - Persistent failures trigger system shutdown with error logging

## Best Practices

1. **Status Monitoring**:
      - Regularly check `last_ping` to ensure setups are active
      - Monitor `queue_size` to detect potential bottlenecks
      - Use `state` field to track experiment progress

2. **Error Handling**:
      - Implement monitoring for "ERROR!" states
      - Check notes field for detailed error information
      - check ethopy.log to track the issue

3. **Resource Management**:
      - Monitor `total_liquid` to ensure proper reward delivery
      - Track `trials` to ensure experiment progress
      - Use `task_idx` to verify correct experiment execution