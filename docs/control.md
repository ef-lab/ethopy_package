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
      - References a specific task configuration path stored in the Task table at lab_experiment
      - The system automatically loads the corresponding task configuration when the session starts

9. `start_time`
      - Scheduled start time for the experiment session
      - Format: "HH:MM:SS" (e.g., "09:00:00")
      - Default: "00:00:00" (midnight)
      - Used to control when experiments can begin running

10. `stop_time`
      - Scheduled stop time for the experiment session
      - Format: "HH:MM:SS" (e.g., "17:00:00") 
      - Default: "23:59:00" (11:59 PM)
      - Used to control when experiments should go to stop/sleep
      - Must be defined if `start_time` is specified

## How to Use the Control Table
The Control table is automatically updated by the Logger class. You don't need to modify it directly in most cases.

The user only change the status of the experiment from running to stop and from ready to running. The user can aslo change the animal_id, the task_id, start_time and stop_time.

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