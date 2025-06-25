# NWB Export Documentation

## Overview

The NWB export module provides functionality to export experimental data from Ethopy DataJoint tables to NWB (Neurodata Without Borders) format files. This documentation covers the main export functions and their usage.

## Main Functions

### `export_to_nwb()`

The primary function for exporting a single experimental session to NWB format.

#### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `animal_id` | `int` | Unique identifier for the animal |
| `session_id` | `int` | Session identifier |

#### Optional Parameters

##### NWB File Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experimenter` | `str` | `"Unknown"` | Name of the experimenter |
| `lab` | `str` | `"Your Lab Name"` | Laboratory name |
| `institution` | `str` | `"Your Institution"` | Institution name |
| `session_description` | `str` | Auto-generated | Description of the experimental session |

##### Subject Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `age` | `str` | `"Unknown"` | Age in ISO 8601 duration format |
| `subject_description` | `str` | `"laboratory mouse"` | Description of the subject |
| `species` | `str` | `Unknown"` | Species of the subject |
| `sex` | `str` | `"U"` | Sex: "M", "F", "U" (unknown), or "O" (other) |

##### Additional Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_filename` | `str` | Auto-generated | Output filename for the NWB file |
| `overwrite` | `bool` | `False` | Whether to overwrite existing files |
| `return_nwb_object` | `bool` | `False` | Return both filename and NWB object |
| `config_path` | `str` | `~/.ethopy/local_conf.json` | Path to the local configuration file for DataJoint and schema setup. If not provided, the default Ethopy config is used. This parameter is passed to the DataJoint connection setup and allows exporting from different databases or configurations. |

#### Returns

- `str`: Path to the saved NWB file (default)
- `Tuple[str, NWBFile]`: Path and NWB object if `return_nwb_object=True`

#### Raises

- `ValueError`: If no session is found for the provided animal_id and session_id
- `FileExistsError`: If output file exists and `overwrite=False`

### `batch_export_to_nwb()`

Export multiple sessions to NWB format in batch.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_list` | `List[Tuple[int, int]]` | List of (animal_id, session_id) tuples |
| `output_directory` | `str` | Directory to save NWB files (default: "nwb_exports") |
| `**kwargs` | | Additional parameters passed to `export_to_nwb()` |

#### Returns

- `List[str]`: List of successfully exported filenames

## Usage Examples

### Basic Usage

Export with minimal parameters using all defaults:

```python
filename = export_to_nwb(animal_id=123, session_id=1)
```

This will create a file named `nwb_animal_123_session_1.nwb` with default metadata.

### Custom Parameters

Export with custom metadata and subject information:

```python
filename = export_to_nwb(
    animal_id=123,
    session_id=1,
    experimenter="Alex Smith",
    lab="Systems Neuroscience Lab",
    institution="FORTH IMBB",
    session_description="2AFC task",
    age="P120D",  # 120 days old
    subject_description="Wild-type C57BL/6J mouse, head-fixed",
    sex="F",
    overwrite=True
)
```

### Custom Filename

Specify a custom output filename:

```python
filename = export_to_nwb(
    animal_id=123,
    session_id=1,
    output_filename="TwoAFCexperiment_session_1.nwb",
    experimenter="Alex"
)
```

### Clarify the configuration path
```python
filename = export_to_nwb(
    animal_id=123,
    session_id=1,
    config_path="/path/to/custom_config.json"
)
```
Example of configuration file

The configuration file is a JSON file that specifies how to connect to your DataJoint database and which schema names to use for your experiment. You can provide a custom configuration file using the `config_path` parameter in `export_to_nwb`. This allows you to export data from different databases or with different schema setups, without changing your code.

**Parameter explanations:**

- **dj_local_conf**:  
  Contains the database connection settings for DataJoint.
  - `database.host`: The hostname or IP address of your MySQL server.
  - `database.user`: The username for connecting to the database.
  - `database.password`: The password for the database user.
  - `database.port`: The port number for the database (default is usually 3306).
  - `database.reconnect`: Whether to automatically reconnect if the connection drops.
  - `database.use_tls`: Whether to use TLS/SSL for the connection.
  - `datajoint.loglevel`: The logging level for DataJoint messages (e.g., "WARNING", "INFO").

- **SCHEMATA**:  
  Maps logical names to the actual schema names in your database. This tells Ethopy which schemas to use for each data type.
  - `experiment`: The schema name for experiment/session tables.
  - `stimulus`: The schema name for stimulus tables.
  - `behavior`: The schema name for behavior tables.

**Example:**
```json
{
    "dj_local_conf": {
        "database.host": "localhost",
        "database.user": "myuser",
        "database.password": "mypassword",
        "database.port": 3306,
        "database.reconnect": true,
        "database.use_tls": false,
        "datajoint.loglevel": "WARNING"
    },
    "SCHEMATA": {
        "experiment": "lab_experiments",
        "stimulus": "lab_stimuli",
        "behavior": "lab_behavior",
    }
}
```

### Return NWB Object for Further Processing

Get both the filename and NWB object for additional processing:

```python
filename, nwb_obj = export_to_nwb(
    animal_id=123,
    session_id=1,
    return_nwb_object=True
)

# Now you can access the NWB object directly
print(f"NWB file contains {len(nwb_obj.trials)} trials")
```
### Overwriting
The code raises FileExistsError if the file exists and overwrite=False.

```python
filename, nwb_obj = export_to_nwb(
    animal_id=123,
    session_id=1,
    overwrite=True
)
```

### Batch Export

Export multiple sessions at once:

```python
# Define sessions to export
animal_session_list = [
                        (123, 1),
                        (123, 2),
                        (124, 1),
                        (124, 2)
                    ]

# Batch export with common parameters
exported_files = batch_export_to_nwb(
    sessions,
    experimenter="Dr. Smith",
    lab="Vision Lab",
    institution="FORTH IMBB",
    output_directory="my_nwb_exports"
)

print(f"Successfully exported {len(exported_files)} files")
```

## Parameter Guidelines

### Age Format (ISO 8601 Duration)

The age parameter should follow ISO 8601 duration format:

- `P90D` = 90 days
- `P3M` = 3 months
- `P1Y6M` = 1 year and 6 months
- `P2Y` = 2 years
- `P1Y2M15D` = 1 year, 2 months, and 15 days

### Sex Values

| Value | Description |
|-------|-------------|
| `"M"` | Male |
| `"F"` | Female |
| `"U"` | Unknown |
| `"O"` | Other |

## Data Included in NWB Files

The export function includes the following data types:

### Core Data
- **Session metadata**: Timestamps, experimenter, lab, institution
- **Subject information**: Species, age, sex, description
- **Trials**: Trial timing, conditions, and metadata

### Experimental Data
- **Conditions**: Experiment, stimulus, and behavior condition parameters
- **Activity data**: Behavioral activity measurements
- **Reward data**: Reward delivery timestamps and amounts
- **Stimulus data**: Stimulus presentation timing and parameters
- **States data**: State transition timestamps for each trial

### Data Organization
- **Processing modules**: Data organized into logical modules (Conditions, Activity, Reward, States)
- **Dynamic tables**: Flexible storage for experimental parameters
- **Time series**: Timestamped behavioral events
- **Behavioral events**: Reward delivery and other behavioral markers

## Error Handling

The export functions provide comprehensive error handling:

### Common Errors

1. **Session not found**: If the specified animal_id and session_id combination doesn't exist
2. **File exists**: If the output file already exists and `overwrite=False`
3. **No valid trials**: If no trials with both PreTrial and InterTrial states are found
4. **Database connection**: If DataJoint connection fails

**Missing parameters**
If certain data (such as specific trial fields, behavioral events, or condition tables) are missing or incomplete for a session, the export function will skip those data elements and continue the export process. 
A warning message will be logged to inform you about any missing or incomplete data, so you can review and address these issues if needed. This ensures that a valid NWB file is still created even when some parts of the dataset are unavailable.

### Error Recovery

- The function will attempt to create the NWB file even if some data is missing
- Warning messages are displayed for missing or incomplete data
- Batch export continues processing remaining sessions if individual exports fail

## Best Practices

### File Naming
- Use descriptive filenames that include animal ID, session ID, and experiment type
- Use consistent naming conventions across your lab

### Metadata Completeness
- Always provide experimenter name and institution
- Include detailed session descriptions
- Specify accurate subject information (age, sex, strain)

### Batch Processing
- Use batch export for multiple sessions with similar parameters
- Create separate output directories for different experiments
- Monitor the console output for failed exports

### File Management
- Regularly backup NWB files
- Use version control for analysis scripts
- Document any post-processing steps applied to NWB files

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all required packages are installed (pynwb, datajoint, numpy, pandas)
2. **Database connection**: Verify DataJoint configuration and database access
3. **Memory issues**: For large sessions, consider processing in smaller batches
4. **File permissions**: Ensure write permissions for the output directory

## Advanced Usage

### Custom Processing

If you need to modify the NWB file after creation:

```python
# Get the NWB object
filename, nwb_obj = export_to_nwb(
    animal_id=123,
    session_id=1,
    return_nwb_object=True
)

# Add custom data
from pynwb import TimeSeries
custom_data = TimeSeries(
    name="custom_signal",
    data=[1, 2, 3, 4, 5],
    timestamps=[0.1, 0.2, 0.3, 0.4, 0.5],
    unit="arbitrary_units"
)
nwb_obj.add_acquisition(custom_data)

# Save the modified file
from pynwb import NWBHDF5IO
with NWBHDF5IO("modified_" + filename, "w") as io:
    io.write(nwb_obj)
```

### Integration with Analysis Pipelines

The exported NWB files can be easily integrated with analysis pipelines:

```python
from pynwb import NWBHDF5IO

# Read the NWB file
with NWBHDF5IO("nwb_animal_123_session_1.nwb", "r") as io:
    nwb_file = io.read()
    
    # Access trial data
    trials_df = nwb_file.trials.to_dataframe()
    
    # Access behavioral events
    rewards = nwb_file.processing['Reward']['BehavioralEvents']['response_reward']
    
    # Access conditions
    conditions = nwb_file.processing['Conditions']

    # get Experiment conditions for each trial (similar for Stimulus/stim_hash, Behavior/beh_hash)
    exp_conds = nwb_file.processing['Conditions'].get('Experiment').to_dataframe()
    trials = nwb_file.trials.to_dataframe()
    merged_df = trials.merge(exp_conds, on="cond_hash", how="inner")
```