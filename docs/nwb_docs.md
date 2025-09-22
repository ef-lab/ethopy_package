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
| `species` | `str` | `"Unknown"` | Species of the subject |
| `sex` | `str` | `"U"` | Sex: "M", "F", "U" (unknown), or "O" (other) |

##### Additional Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_filename` | `str` | Auto-generated | Output filename for the NWB file |
| `overwrite` | `bool` | `False` | Whether to overwrite existing files |
| `return_nwb_object` | `bool` | `False` | Return both filename and NWB object |
| `config_path` | `str` | `None` | Path to configuration file (uses default if not specified) |

## Usage Examples

### Basic Export

Export a single session with default parameters:

```python
from ethopy.utils.export.nwb import export_to_nwb

# Basic export with minimal parameters
filename = export_to_nwb(animal_id=123, session_id=1)
print(f"NWB file saved as: {filename}")
```

### Export with Custom Metadata

Include detailed metadata about the experiment:

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

### Custom Configuration File

Use a custom configuration file for database connection:

```python
filename = export_to_nwb(
    animal_id=123,
    session_id=1,
    config_path="/path/to/custom_config.json"
)
```

The configuration file is a JSON file that specifies how to connect to your DataJoint database and which schema names to use for your experiment. You can provide a custom configuration file using the `config_path` parameter in `export_to_nwb`. If no `config_path` is specified, EthoPy will automatically search for the configuration file in the default location (`~/.ethopy/local_conf.json`).

For detailed information about configuration options and file structure, see the [Local Configuration Guide](local_conf.md).

### Return NWB Object for Further Processing

Get both the filename and NWB object for additional processing:

```python
filename, nwb_obj = export_to_nwb(
    animal_id=123,
    session_id=1,
    return_nwb_object=True
)

# Use the NWB object for additional processing
print(f"NWB file contains {len(nwb_obj.trials)} trials")
```

### Batch Export

Export multiple sessions at once:

```python
from ethopy.utils.export.nwb import batch_export_to_nwb

# Define sessions to export
animal_session_list = [
                        (123, 1),
                        (123, 2),
                        (124, 1),
                        (124, 2)
                    ]

exported_files = batch_export_to_nwb(
    animal_session_list,
    experimenter="Dr. Smith",
    lab="Vision Lab",
    institution="FORTH IMBB",
    output_directory="my_nwb_exports"
)

print(f"Successfully exported {len(exported_files)} sessions")
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
- **Trials**: Trial timing, conditions, and metadata
- **Subject Information**: Animal details (age, species, sex, etc.)
- **Session Metadata**: Experimenter, lab, institution, timestamps

### Experimental Data
- **Conditions**: Experiment, stimulus, and behavior conditions
- **Activity Data**: Behavioral measurements and responses
- **Reward Data**: Reward delivery timestamps and amounts
- **Stimulus Data**: Stimulus parameters and timing
- **States Data**: Trial state transitions and timing

### Special Features
- **Compound Stimuli**: Automatically handles multi-component stimuli (e.g., "Tones_Grating")
- **Fallback Timing**: Robust trial timing extraction with multiple strategies
- **Data Validation**: Comprehensive error checking and data integrity validation

## Advanced Features

### Compound Stimulus Support

The export function automatically detects and handles compound stimuli (stimuli with multiple components separated by underscores):

```python
# If your session uses compound stimuli like "Tones_Grating"
# The export will automatically:
# 1. Detect the compound stimulus
# 2. Split it into components ("Tones" and "Grating")
# 3. Validate all components exist in the database
# 4. Export each component separately
filename = export_to_nwb(animal_id=123, session_id=1)
```

### Robust Trial Timing

The system uses multiple strategies to extract trial timing:

1. **Primary Strategy**: Uses PreTrial and InterTrial states when available
2. **Fallback Strategy**: Uses first and last available states for trials missing standard timing
3. **Comprehensive Logging**: Detailed information about timing extraction for each trial

### Error Handling

The export function provides detailed error messages and logging:

```python
try:
    filename = export_to_nwb(animal_id=123, session_id=1)
except ValueError as e:
    print(f"Invalid session: {e}")
except FileExistsError as e:
    print(f"File already exists: {e}")
except Exception as e:
    print(f"Export failed: {e}")
```

## Output File Structure

The generated NWB file contains:

### Processing Modules
- **Conditions**: Experiment, behavior, and stimulus condition parameters
- **Activity data**: Custom behavioral metadata tables
- **Reward**: Reward delivery time series data
- **States**: Trial state transition timestamps

### Core NWB Components
- **Trials table**: Trial-by-trial data with timing and conditions
- **Subject**: Animal information and metadata
- **Stimulus tables**: Stimulus parameters and presentation timing

## Troubleshooting

### Common Issues

1. **"No session found" error**
   - Verify animal_id and session_id exist in your database
   - Check database connection in your configuration file

2. **"File already exists" error**
   - Use `overwrite=True` parameter or choose a different filename

3. **Missing data warnings**
   - Some warnings are normal (e.g., missing reward data for non-reward sessions)
   - Check logs for specific missing components

4. **Configuration errors**
   - Verify your `local_conf.json` file has correct database credentials
   - Use `config_path` parameter to specify custom configuration

### Data Validation

The export function performs extensive validation:
- Checks for session existence
- Validates stimulus components for compound stimuli
- Verifies trial timing data consistency
- Reports missing or incomplete data with detailed logging

### Performance Notes

- Large sessions may take several minutes to export
- The function uses efficient DataJoint queries to minimize memory usage
- Progress is logged throughout the export process

## Related Functions

- `batch_export_to_nwb()`: Export multiple sessions efficiently
- `setup_datajoint_connection()`: Set up database connections with custom configurations
- `create_nwb_file()`: Create base NWB file structure (internal function)

For more details about the underlying data structures and database schema, see the main EthoPy documentation.