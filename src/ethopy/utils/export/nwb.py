"""
NWB File Export Module for Ethopy Data

This module provides functionality to export experimental data from DataJoint tables
to NWB (Neurodata Without Borders) format files.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Optional, NamedTuple
from uuid import uuid4
from pathlib import Path
from contextlib import contextmanager
from functools import reduce

import datajoint as dj
import numpy as np
from dateutil import tz
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents
from pynwb.core import DynamicTable
from pynwb.file import Subject
import pandas as pd

from ethopy.config import ConfigurationManager
from ethopy.utils.helper_functions import create_virtual_modules

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NWBExportError(Exception):
    """Custom exception for NWB export errors."""

    pass


class SessionClasses(NamedTuple):
    """Container for session class information."""

    stimulus: np.ndarray
    behavior: np.ndarray
    experiment: np.ndarray


class TrialData(NamedTuple):
    """Container for processed trial data."""

    pretrial_times: List[float]
    intertrial_times: List[float]
    valid_indices: List[int]


def inhomogeneous_columns(nwbfile: NWBFile) -> List[str]:
    """
    Find columns with inhomogeneous array data that will fail HDF5 write.

    HDF5 format requires all arrays in a column to have the same shape. This function
    validates the NWB file by checking trials, processing modules, and stimulus tables
    for columns that contain arrays with inconsistent dimensions.

    Args:
        nwbfile: The NWB file object to validate

    Returns:
        List of problem descriptions in the format "table.column: error_message"
        Empty list if no problems are found

    Example:
        >>> problems = inhomogeneous_columns(nwbfile)
        >>> if problems:
        ...     print(f"Found {len(problems)} inhomogeneous columns")
    """
    problems = []

    # Check trials table
    if nwbfile.trials is not None:
        for col_name in nwbfile.trials.colnames:
            col_data = nwbfile.trials[col_name].data
            try:
                np.array(col_data)
            except ValueError as e:
                problems.append(f"trials.{col_name}: {e}")

    # Check processing modules
    for module_name, module in nwbfile.processing.items():
        for container_name, container in module.data_interfaces.items():
            if isinstance(container, DynamicTable):
                for col_name in container.colnames:
                    col_data = container[col_name].data
                    try:
                        np.array(col_data)
                    except ValueError as e:
                        problems.append(
                            f"processing.{module_name}.{container_name}.{col_name}: {e}"
                        )

    # Check stimulus tables
    for stim_name, stim in nwbfile.stimulus.items():
        if isinstance(stim, DynamicTable):
            for col_name in stim.colnames:
                col_data = stim[col_name].data
                try:
                    np.array(col_data)
                except ValueError as e:
                    problems.append(f"stimulus.{stim_name}.{col_name}: {e}")

    return problems


def _get_session_timestamp(experiment: Any, session_key: Dict[str, Any]) -> datetime:
    """
    Fetch the session timestamp for a given session key.

    Retrieves the session start timestamp from the DataJoint database and ensures
    it has timezone information. If the timestamp is timezone-naive, it assumes
    the local timezone.

    Args:
        experiment: DataJoint experiment module containing Session table
        session_key: Primary key dict identifying the session (e.g., {'animal_id': 1, 'session': 1})

    Returns:
        datetime: Session timestamp with timezone information (timezone-aware)

    Raises:
        NWBExportError: If no session is found for the provided key
    """
    session_tmst = (experiment.Session & session_key).fetch("session_tmst")
    if len(session_tmst) == 0:
        raise NWBExportError(
            f"No session found for the provided session key: {session_key}"
        )

    tmst = session_tmst[0]
    if tmst.tzinfo is None:
        tmst = tmst.replace(tzinfo=tz.tzlocal())
    return tmst


def milliseconds_to_seconds(
    milliseconds: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert milliseconds to seconds.

    Args:
        milliseconds: Time value(s) in milliseconds. Can be a single float,
                     numpy array, or pandas Series

    Returns:
        Time value(s) in seconds, maintaining the same type as input
    """
    return milliseconds / 1000.0


def get_non_empty_children(
    table: dj.Table,
    session_hash: Optional[dj.Table] = None,
    session_key: Optional[Dict[str, Any]] = None,
) -> List[dj.Table]:
    """
    Return all child tables of a parent table that are non-empty after restriction.

    This function retrieves all child tables from a DataJoint parent table and filters
    them based on either a session_key dictionary or a session_hash table restriction.
    Only children that contain data after the restriction are returned.

    Args:
        table: Parent DataJoint table to get children from
        session_hash: Optional DataJoint table restriction to apply to children
        session_key: Optional dict with primary keys to restrict children
                    (e.g., {'animal_id': 1, 'session': 1})

    Returns:
        List of restricted child tables that contain data

    Raises:
        ValueError: If both session_key and session_hash are provided
        ValueError: If neither session_key nor session_hash is provided

    Example:
        >>> children = get_non_empty_children(stimulus.StimCondition,
        ...                                   session_key={'animal_id': 1, 'session': 1})
    """
    if session_key is not None and session_hash is not None:
        raise ValueError("Provide either session_key or session_hash, not both.")

    if session_key is None and session_hash is None:
        raise ValueError("Either session_key or session_hash must be provided.")

    restricted_children = []

    for child in table.children(as_objects=True):
        restriction = (
            child & session_key if session_key is not None else child & session_hash
        )

        if len(restriction) > 0:
            restricted_children.append(restriction)

    return restricted_children


def parse_compound_stimulus(stimulus_class: str) -> List[str]:
    """
    Parse compound stimulus names separated by underscores.

    Compound stimuli are represented by multiple stimulus types separated by underscores.
    This function splits them into individual components.

    Args:
        stimulus_class: Stimulus class name that may contain multiple components
                       (e.g., 'Tones_Grating', 'Visual_Auditory_Tactile')

    Returns:
        List of individual stimulus component names
        (e.g., ['Tones', 'Grating'], ['Visual', 'Auditory', 'Tactile'])

    Example:
        >>> parse_compound_stimulus('Tones_Grating')
        ['Tones', 'Grating']
        >>> parse_compound_stimulus('SimpleStimulus')
        ['SimpleStimulus']
    """
    return stimulus_class.split("_")


def combine_children_tables(children: List[dj.Table]) -> dj.Table:
    """
    Combine all child tables using the DataJoint join operator.

    Uses functional reduction to join multiple DataJoint tables into a single table.
    The join operator (*) in DataJoint performs a natural join based on common attributes.

    Args:
        children: List of DataJoint table objects to combine

    Returns:
        Single DataJoint table containing the joined result of all children

    Raises:
        TypeError: If children list is empty (reduce on empty sequence)

    Example:
        >>> child1 = experiment.Trial & session_key
        >>> child2 = experiment.Condition & session_key
        >>> combined = combine_children_tables([child1, child2])
    """
    return reduce(lambda x, y: x * y, children)


def get_stimulus_conditions(
    stimulus_module: Any, session_key: Dict[str, Any], class_name: str
) -> dj.Table:
    """
    Fetch stimulus conditions for a given stimulus class.

    Retrieves stimulus condition data from the database by joining the trial-level
    stimulus conditions with the specific stimulus class parameters. Handles both
    simple and complex stimulus hierarchies with multiple child tables.

    Args:
        stimulus_module: DataJoint stimulus module containing StimCondition and stimulus classes
        session_key: Primary key dict identifying the session
                    (e.g., {'animal_id': 1, 'session': 1})
        class_name: Name of the stimulus class to retrieve (e.g., 'Grating', 'Tones')

    Returns:
        DataJoint table joining StimCondition.Trial with the stimulus class parameters
        for the specified session

    Note:
        If multiple child tables exist for the stimulus class, they are automatically
        combined using natural joins.
    """
    stim_class = getattr(stimulus_module, class_name)
    session_hash = stimulus_module.StimCondition.Trial & session_key
    children = get_non_empty_children(stim_class, session_hash=session_hash)

    if len(children) > 1:
        comb_tables = combine_children_tables(children)
    elif len(children) == 1:
        comb_tables = children[0]
    else:
        comb_tables = stim_class
        logger.warning(f"No children found for stimulus class {class_name}")

    return (stimulus_module.StimCondition.Trial & session_key) * comb_tables


def validate_stimulus_components(
    stimulus_module: Any, class_name: str
) -> Tuple[bool, List[str]]:
    """
    Validate that all components of a compound stimulus exist in the stimulus module.

    Args:
        stimulus_module: DataJoint stimulus module
        class_name: Name of the compound stimulus class (e.g., 'Tones_Grating')

    Returns:
        Tuple of (all_exist: bool, missing_components: List[str])
    """
    stimulus_components = parse_compound_stimulus(class_name)
    missing_components = []

    for component in stimulus_components:
        try:
            getattr(stimulus_module, component)
            logger.debug(f"Stimulus component '{component}' found in module")
        except AttributeError:
            missing_components.append(component)
            logger.error(
                f"Stimulus component '{component}' not found in stimulus module"
            )

    all_exist = len(missing_components) == 0
    return all_exist, missing_components


def get_multiple_stimulus_conditions(
    stimulus_module: Any, session_key: Dict[str, Any], class_name: str
) -> Dict[str, dj.Table]:
    """
    Fetch stimulus conditions for compound stimulus classes (e.g., 'Tones_Grating').

    Args:
        stimulus_module: DataJoint stimulus module
        session_key: Primary key identifying the session
        class_name: Name of the compound stimulus class (e.g., 'Tones_Grating')

    Returns:
        Dictionary mapping component names to their condition tables

    Raises:
        NWBExportError: If any stimulus components are missing from the database
    """
    # First validate that all components exist
    all_exist, missing_components = validate_stimulus_components(
        stimulus_module, class_name
    )

    if not all_exist:
        raise NWBExportError(
            f"Missing stimulus components in database: {missing_components}. "
            f"Cannot export compound stimulus '{class_name}'"
        )

    stimulus_components = parse_compound_stimulus(class_name)
    conditions_dict = {}

    for component in stimulus_components:
        component_conditions = get_stimulus_conditions(
            stimulus_module, session_key, component
        )
        if len(component_conditions) > 0:
            conditions_dict[component] = component_conditions
            logger.info(
                f"Found {len(component_conditions)} conditions for stimulus component '{component}'"
            )
        else:
            logger.warning(
                f"No conditions found for stimulus component '{component}' in session"
            )

    if not conditions_dict:
        logger.error(
            f"No stimulus conditions found for any component of '{class_name}' in session"
        )

    return conditions_dict


def get_experiment_conditions(
    experiment_module: Any, session_key: Dict[str, Any], class_name: str
) -> dj.Table:
    """
    Fetch experiment conditions for a given experiment class.

    Joins trial data with condition parameters for a specific experiment class type.
    This retrieves experimental parameters (e.g., task difficulty, reward schedule)
    that varied across trials.

    Args:
        experiment_module: DataJoint experiment module containing Trial and Condition tables
        session_key: Primary key dict identifying the session
                    (e.g., {'animal_id': 1, 'session': 1})
        class_name: Name of the experiment condition class
                   (e.g., 'OperantConditioning', 'DetectionTask')

    Returns:
        DataJoint table joining Trial, Condition, and the specific experiment class
        for the specified session
    """
    exp_class = getattr(experiment_module.Condition, class_name)
    return (
        (experiment_module.Trial() & session_key) * experiment_module.Condition
    ) * exp_class


def get_behavior_conditions(
    behavior_module: Any, session_key: Dict[str, Any], class_name: str
) -> dj.Table:
    """
    Fetch behavior conditions for a given behavior class.

    Retrieves behavioral condition data by joining trial-level behavior conditions
    with specific behavior class parameters. Handles behavior hierarchies with
    multiple child tables by combining them.

    Args:
        behavior_module: DataJoint behavior module containing BehCondition and behavior classes
        session_key: Primary key dict identifying the session
                    (e.g., {'animal_id': 1, 'session': 1})
        class_name: Name of the behavior class to retrieve
                   (e.g., 'Licking', 'Running', 'Choice')

    Returns:
        DataJoint table joining BehCondition.Trial with the behavior class parameters
        for the specified session

    Note:
        If multiple child tables exist for the behavior class, they are automatically
        combined using natural joins. Logs a warning if no children are found.
    """
    beh_class = getattr(behavior_module, class_name)
    children = beh_class.children(as_objects=True)

    if len(children) > 1:
        comb_tables = combine_children_tables(children)
    elif len(children) == 1:
        comb_tables = children[0]
    else:
        comb_tables = beh_class
        logger.warning(f"No children found for behavior class {class_name}")

    return (behavior_module.BehCondition.Trial() & session_key) * comb_tables


def get_table_columns(table: dj.Table) -> List[str]:
    """
    Fetch all column names from a DataJoint table.

    Args:
        table: DataJoint table object

    Returns:
        List of column names from the table heading

    Example:
        >>> columns = get_table_columns(experiment.Trial)
        >>> print(columns)
        ['animal_id', 'session', 'trial_idx', 'trial_start_time']
    """
    return table.heading.names


def remove_columns(table: dj.Table, columns_to_remove: List[str]) -> List[str]:
    """
    Get table column names excluding specified columns.

    Args:
        table: DataJoint table object
        columns_to_remove: List of column names to exclude

    Returns:
        List of column names that are not in columns_to_remove

    Example:
        >>> remaining = remove_columns(trial_table, ['animal_id', 'session'])
    """
    return [col for col in table.heading.names if col not in columns_to_remove]


def get_children_tables(table: dj.Table) -> List[dj.Table]:
    """
    Get all non-empty child tables of a DataJoint table.

    Retrieves child tables that contain at least one row of data.

    Args:
        table: Parent DataJoint table to get children from

    Returns:
        List of non-empty child table objects

    Note:
        This function does NOT apply any session restrictions. Use get_non_empty_children()
        if you need to filter by session.
    """
    children = table.children(as_objects=True)
    return [child for child in children if len(child) > 0]


def create_nwb_file(
    experiment: Any,
    session_key: Dict[str, Any],
    session_description: str,
    experimenter: str,
    lab: str = "Your Lab Name",
    institution: str = "Your Institution",
) -> NWBFile:
    """
    Create the base NWB file with metadata.

    Args:
        experiment: DataJoint experiment module
        session_key: Primary key identifying the session
        session_description: Description of the experimental session
        experimenter: Name of the experimenter
        lab: Laboratory name
        institution: Institution name

    Returns:
        The created NWBFile object
    """
    session_start_time = _get_session_timestamp(experiment, session_key)

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=session_start_time,
        experimenter=[experimenter],
        lab=lab,
        institution=institution,
        file_create_date=datetime.now(tz=tz.UTC),
        timestamps_reference_time=session_start_time,
        was_generated_by=["Ethopy"],
    )

    return nwbfile


def create_subject(
    animal_id: int,
    age: str = "Unknown",
    description: str = "laboratory mouse",
    species: str = "Mus musculus",
    sex: str = "U",
) -> Subject:
    """
    Create a Subject object for the NWB file.

    Args:
        animal_id: Unique identifier for the animal
        age: Age of the subject
        description: Description of the subject
        species: Species of the subject
        sex: Sex of the subject

    Returns:
        Subject object
    """
    return Subject(
        subject_id=str(animal_id),
        age=age,
        description=description,
        species=species,
        sex=sex,
    )


def process_trial_states(experiment: Any, session_key: Dict[str, Any]) -> TrialData:
    """
    Process trial states to extract timing information.

    Args:
        experiment: DataJoint experiment module
        session_key: Primary key identifying the session

    Returns:
        TrialData containing pretrial times, intertrial times, and valid trial indices
    """
    states_df = (
        (experiment.Trial.StateOnset & session_key).fetch(format="frame").reset_index()
    )
    if len(states_df) < 3:
        raise Exception(f"Not enough Trials {states_df}")
    if states_df.empty:
        logger.warning("No trial states found for session")
        return TrialData([], [], [])

    # Filter for relevant states
    relevant_states = states_df[states_df["state"].isin(["PreTrial", "InterTrial"])]

    if relevant_states.empty:
        logger.warning("No PreTrial or InterTrial states found")
        return TrialData([], [], [])

    # Pivot to get both states per trial
    trial_states_pivot = relevant_states.pivot_table(
        index="trial_idx", columns="state", values="time", aggfunc="first"
    )

    # Filter trials that have both required states
    complete_trials = trial_states_pivot.dropna(subset=["PreTrial", "InterTrial"])

    # Log problematic trials (those missing timing data) and attempt fallback recovery
    incomplete_trials = trial_states_pivot[trial_states_pivot.isnull().any(axis=1)]

    # Initialize collections for combined data
    all_pretrial_times = []
    all_intertrial_times = []
    all_valid_indices = []
    fallback_count = 0

    # Process trials with both PreTrial and InterTrial states (primary strategy)
    if not complete_trials.empty:
        primary_pretrial_times = milliseconds_to_seconds(
            complete_trials["PreTrial"]
        ).tolist()
        primary_intertrial_times = milliseconds_to_seconds(
            complete_trials["InterTrial"]
        ).tolist()
        primary_valid_indices = complete_trials.index.tolist()

        all_pretrial_times.extend(primary_pretrial_times)
        all_intertrial_times.extend(primary_intertrial_times)
        all_valid_indices.extend(primary_valid_indices)

    # Process incomplete trials using fallback strategy
    if not incomplete_trials.empty:
        logger.warning(
            f"Found {len(incomplete_trials)} trials with incomplete PreTrial/InterTrial timing data."
        )
        logger.info("Attempting fallback recovery using first/last states...")

        for trial_idx, row in incomplete_trials.iterrows():
            missing_states = []
            if pd.isna(row.get("PreTrial")):
                missing_states.append("PreTrial")
            if pd.isna(row.get("InterTrial")):
                missing_states.append("InterTrial")

            # Get all states for this trial for fallback
            trial_states = states_df[states_df["trial_idx"] == trial_idx].copy()
            trial_states = trial_states.sort_values("time")

            if len(trial_states) < 2:
                logger.warning(
                    f"  Trial {trial_idx}: Only {len(trial_states)} state(s) found, cannot determine trial duration."
                    f"  Skipping."
                )
                continue

            # Use fallback timing: first and last available states
            start_time = trial_states["time"].iloc[0]
            end_time = trial_states["time"].iloc[-1]
            available_states = trial_states["state"].tolist()

            # Log detailed fallback information
            logger.info(
                f"  Trial {trial_idx}: Using fallback timing (missing {', '.join(missing_states)})"
            )
            logger.info(f"    Available states: {available_states}")
            logger.info(
                f"    Using: {available_states[0]} (start: {start_time}ms) → "
                f"{available_states[-1]} (end: {end_time}ms)"
            )

            # Add to combined dataset
            all_pretrial_times.append(milliseconds_to_seconds(start_time))
            all_intertrial_times.append(milliseconds_to_seconds(end_time))
            all_valid_indices.append(trial_idx)
            fallback_count += 1

    # Check if we have any valid trials at all
    if not all_valid_indices:
        logger.warning(
            "No trials found with sufficient timing data (neither PreTrial/InterTrial nor fallback states)"
        )
        return TrialData([], [], [])

    # Log processing results with breakdown
    total_trials = len(states_df["trial_idx"].unique())
    primary_count = len(complete_trials) if not complete_trials.empty else 0

    logger.info(
        f"Trial timing summary: {primary_count} trials used PreTrial/InterTrial, "
        f"{fallback_count} trials used fallback (first/last state)"
    )
    logger.info(f"Total processed: {len(all_valid_indices)}/{total_trials} trials")

    if fallback_count > 0:
        logger.info(
            "Fallback trials may have different timing characteristics. Review data quality as needed."
        )

    logger.info(f"Valid trial indices: {len(all_valid_indices)}")

    return TrialData(all_pretrial_times, all_intertrial_times, all_valid_indices)


def add_trials_to_nwb(
    nwbfile: NWBFile,
    trial_hash: dj.Table,
    trial_data: TrialData,
    keep_columns: List[str],
) -> None:
    """
    Add trial information to the NWB file.

    Args:
        nwbfile: NWB file object
        trial_hash: DataJoint table with trial information
        trial_data: Processed trial timing data
        keep_columns: List of columns to keep in the trial table
    """
    if len(trial_hash) == 0:
        logger.warning(
            "Trial hashes are mising check experiment,condition and stimulus hashes "
        )
        return

    if not trial_data.valid_indices:
        logger.warning("No valid trial indices provided")
        return
    logger.info(f"trial_data {len(trial_data)}, {len(trial_data.pretrial_times)}")

    # Add trial columns
    trial_columns = {
        tag: {"name": tag, "description": trial_hash.heading.attributes[tag].comment}
        for tag in trial_hash.heading.names
        if tag in keep_columns
    }

    for column_info in trial_columns.values():
        nwbfile.add_trial_column(**column_info)

    # Add trial data
    all_columns = set(trial_hash.heading.names)
    columns_to_remove = all_columns - set(keep_columns)
    logger.info(f"all_columns {all_columns}")

    # Create a mapping from trial_idx to timing data indices for efficient lookup
    valid_indices_to_timing_idx = {
        trial_idx: i for i, trial_idx in enumerate(trial_data.valid_indices)
    }

    # Get all trial_idx values from trial_hash for debugging
    trial_hash_indices = [t.get("trial_idx") for t in trial_hash.fetch(as_dict=True)]
    logger.info(f"Trial hash contains {len(trial_hash_indices)}")
    logger.info(f"Timing data available for {len(trial_data.valid_indices)})")

    # Find mismatches between trial_hash and timing data
    missing_timing = set(trial_hash_indices) - set(trial_data.valid_indices)
    missing_trials = set(trial_data.valid_indices) - set(trial_hash_indices)

    if missing_timing:
        logger.warning(
            f"Trials in trial_hash without timing data: "
            f" {sorted(missing_timing)[:10]}{'...' if len(missing_timing) > 10 else ''} (total: {len(missing_timing)})"
        )
    if missing_trials:
        logger.info(
            f"Trials with timing data not in trial_hash: "
            f"{sorted(missing_trials)[:10]}{'...' if len(missing_trials) > 10 else ''} (total: {len(missing_trials)})"
        )

    trials_added = 0
    trials_skipped = 0
    for trial in trial_hash.fetch(as_dict=True):
        trial_idx = trial.get("trial_idx")

        # Find corresponding timing data by trial_idx
        timing_idx = valid_indices_to_timing_idx.get(trial_idx)

        if timing_idx is None:
            logger.warning(f"No timing data found for trial_idx {trial_idx}, skipping")
            trials_skipped += 1
            continue

        trial.update(
            {
                "start_time": float(trial_data.pretrial_times[timing_idx]),
                "stop_time": float(trial_data.intertrial_times[timing_idx]),
                "id": trial_idx,
            }
        )

        # Remove unwanted columns
        for col in columns_to_remove:
            trial.pop(col, None)

        nwbfile.add_trial(**trial)
        trials_added += 1

    logger.info(
        f"Added {trials_added} trials to NWB file, skipped {trials_skipped} trials due to missing timing data"
    )


def get_array_shape(val) -> tuple:
    """Get shape of array-like value."""
    if isinstance(val, np.ndarray):
        return val.shape
    elif isinstance(val, (list, tuple)):
        try:
            return np.array(val).shape
        except ValueError:
            return (len(val), "varied")
    return ("scalar",)  # Return identifiable shape for scalars


def is_array_like(val) -> bool:
    """Check if value is array-like."""
    return isinstance(val, (np.ndarray, list, tuple))


def analyze_array_column(series: pd.Series, col_name: str) -> dict:
    """
    Analyze a column for homogeneity (works with mixed scalar/array columns).

    Returns:
        dict with keys: is_homogeneous, needs_conversion, shapes, shape_counts, message
    """
    # Check if ANY value in the column is array-like
    has_arrays = series.apply(is_array_like).any()

    if not has_arrays:
        # All scalars, no conversion needed
        return {
            "is_homogeneous": True,
            "needs_conversion": False,
            "message": f"Column '{col_name}': All scalar values",
        }

    # Get shapes for all values
    shapes = series.apply(get_array_shape)
    shape_counts = shapes.value_counts()

    is_homogeneous = len(shape_counts) == 1 and ("scalar",) not in shape_counts.index

    # Need conversion if: inhomogeneous OR mixed scalars and arrays
    has_mixed_types = ("scalar",) in shape_counts.index and len(shape_counts) > 1
    needs_conversion = not is_homogeneous or has_mixed_types

    result = {
        "is_homogeneous": is_homogeneous,
        "needs_conversion": needs_conversion,
        "shape_counts": shape_counts.to_dict(),
        "total_rows": len(series),
    }

    if not needs_conversion:
        result["message"] = (
            f"Column '{col_name}': All {len(series)} rows have shape {shape_counts.index[0]}"
        )
    else:
        shape_details = ", ".join(
            f"{shape}: {count} rows" for shape, count in shape_counts.items()
        )
        result["message"] = (
            f"Column '{col_name}': INHOMOGENEOUS - {len(shape_counts)} different shapes found:\n"
            f"    {shape_details}"
        )

    return result


def add_conditions_module(
    nwbfile: NWBFile,
    exp_conditions: dj.Table,
    stim_conditions: dj.Table,
    beh_conditions: dj.Table,
    class_names: SessionClasses,
) -> None:
    """
    Create and add conditions metadata module to NWB file.
    """
    logger.info("Add condition parameters for experiment, behavior and stimuli")
    meta_data = nwbfile.create_processing_module(
        name="Conditions",
        description="Conditions parameters for experiment, behavior and stimuli",
    )

    def add_condition_table(
        conditions: dj.Table, name: str, description: str, columns_to_remove: List[str]
    ):
        df = conditions.fetch(format="frame").reset_index()
        columns_of_interest = [
            col for col in conditions.heading.names if col not in columns_to_remove
        ]

        # Find the hash column for deduplication
        hash_cols = [col for col in columns_of_interest if "_hash" in col]

        unique_combinations = (
            df[columns_of_interest].drop_duplicates(subset=hash_cols).copy()
        )

        if unique_combinations.empty:
            logger.warning(f"No data found for {name} conditions")
            return

        # Analyze ALL columns for array content (not just first value)
        for col in columns_of_interest:
            analysis = analyze_array_column(unique_combinations[col], col)

            if analysis["needs_conversion"]:
                logger.warning(f"[{name}] {analysis['message']}")
                logger.info(
                    f"[{name}] Converting '{col}' to string for HDF5 compatibility"
                )
                unique_combinations[col] = unique_combinations[col].apply(str)
            elif "shape_counts" in analysis:
                logger.debug(analysis["message"])

        table = DynamicTable(name=name, description=description, id=[])

        trial_columns = {
            tag: {
                "name": tag,
                "description": conditions.heading.attributes[tag].comment,
            }
            for tag in conditions.heading.names
            if tag in columns_of_interest
        }
        for column_info in trial_columns.values():
            table.add_column(
                name=column_info["name"], description=column_info["description"]
            )

        for trial in unique_combinations.to_dict(orient="records"):
            table.add_row(**trial)

        meta_data.add(table)

    skip_cols = ["animal_id", "session", "trial_idx", "time"]
    add_condition_table(beh_conditions, "Behavior", class_names.behavior[0], skip_cols)
    add_condition_table(
        exp_conditions, "Experiment", class_names.experiment[0], skip_cols
    )
    add_condition_table(
        stim_conditions,
        "Stimulus",
        class_names.stimulus[0],
        skip_cols + ["start_time", "end_time"],
    )


def create_dynamic_table_from_dj_table(
    table: dj.Table,
    table_name: str,
    description: str,
    skip_columns: Optional[List[str]] = None,
    id_column: str = "trial_idx",
) -> DynamicTable:
    """
    Create a PyNWB DynamicTable from a DataJoint table.

    Args:
        table: DataJoint table
        table_name: Name for the dynamic table
        description: Description of the table
        skip_columns: Columns to skip
        id_column: Column to use as ID

    Returns:
        DynamicTable object
    """
    logger.debug(f"Creating dynamic table: {table_name, table}")
    skip_columns = skip_columns or []

    # Create dynamic table
    dynamic_table = DynamicTable(
        name=table_name,
        description=description,
        id=[],
    )

    # Add columns
    trial_columns = {
        tag: {"name": tag, "description": table.heading.attributes[tag].comment}
        for tag in table.heading.names
        if tag not in skip_columns
    }

    for column_info in trial_columns.values():
        logger.debug(
            f"Adding column: {column_info['name'], column_info['description']}"
        )
        dynamic_table.add_column(
            name=column_info["name"], description=column_info["description"]
        )

    # Add rows
    _table = table.fetch(as_dict=True)
    for trial in _table:
        trial["id"] = trial.get(id_column, len(dynamic_table))
        for key in skip_columns:
            trial.pop(key, None)
        dynamic_table.add_row(**trial)

    return dynamic_table


def add_activity_data(
    nwbfile: NWBFile, behavior_module: Any, session_key: Dict[str, Any]
) -> None:
    """
    Add activity data to NWB file.

    Args:
        nwbfile: NWB file object
        behavior_module: DataJoint behavior module
        session_key: Primary key identifying the session
    """
    logger.info("Add Activity data")
    activity_module = nwbfile.create_processing_module(
        name="Activity data", description="Custom behavioral metadata"
    )

    activity_tables = get_non_empty_children(
        behavior_module.Activity, session_key=session_key
    )

    for table in activity_tables:
        dynamic_table = create_dynamic_table_from_dj_table(
            table,
            table._table_name,
            str(table._heading),
            skip_columns=["animal_id", "session", "start_time", "stop_time"],
        )
        activity_module.add(dynamic_table)


def add_reward_data(
    nwbfile: NWBFile, behavior_module: Any, session_key: Dict[str, Any]
) -> None:
    """
    Add reward delivery data to NWB file.

    Args:
        nwbfile: NWB file object
        behavior_module: DataJoint behavior module
        session_key: Primary key identifying the session
    """
    logger.info("Add reward data")
    reward_data = (behavior_module.Rewards & session_key).fetch(
        "time", "reward_type", "reward_amount"
    )

    if not reward_data[0].size:  # Check if any data exists
        logger.warning("No reward data found for session")
        return

    time, reward_type, reward_amount = reward_data

    behavior_module_nwb = nwbfile.create_processing_module(
        name="Reward", description="Reward delivery data"
    )

    time_series = TimeSeries(
        name="response_reward",
        data=reward_amount.tolist(),
        timestamps=milliseconds_to_seconds(time).tolist(),
        description="The water amount the subject received as a reward.",
        unit="ml",
    )

    behavioral_events = BehavioralEvents(
        time_series=time_series, name="BehavioralEvents"
    )

    behavior_module_nwb.add(behavioral_events)


def add_stimulus_data(
    nwbfile: NWBFile, stim_conditions: dj.Table, stimulus_class: str
) -> None:
    """
    Add stimulus data to NWB file - one row per stimulus presentation.
    """
    logger.info("Add Stimulus per trial")
    df = stim_conditions.fetch(format="frame").reset_index()

    columns_of_interest = ["trial_idx", "start_time", "end_time", "stim_hash"]

    available_cols = [col for col in columns_of_interest if col in df.columns]
    subset_df = df[available_cols].copy()

    # Explode array columns to get one row per presentation
    for col in ["start_time", "end_time"]:
        if col in subset_df.columns:
            first_val = subset_df[col].iloc[0]
            if isinstance(first_val, (np.ndarray, list)):
                subset_df = subset_df.explode(col)

    subset_df = subset_df.reset_index(drop=True)

    if subset_df.empty:
        logger.warning("No stimulus data found")
        return

    table = DynamicTable(
        name=stimulus_class,
        description=f"Stimulus presentation data for {stimulus_class}",
    )

    for col in available_cols:
        if col != "trial_idx":
            description = stim_conditions.heading.attributes.get(col, {})
            table.add_column(name=col, description=getattr(description, "comment", col))

    for idx, row in subset_df.iterrows():
        row_data = {col: row[col] for col in available_cols if col != "trial_idx"}
        row_data["id"] = idx  # Use row index as id since trial_idx may repeat
        table.add_row(**row_data)

    nwbfile.add_stimulus(table)


def add_multiple_stimulus_data(
    nwbfile: NWBFile, stimulus_conditions_dict: Dict[str, dj.Table]
) -> None:
    """
    Add multiple stimulus components to NWB file for compound stimuli.

    Args:
        nwbfile: NWB file object
        stimulus_conditions_dict: Dictionary mapping component names to their condition tables
    """
    if not stimulus_conditions_dict:
        logger.warning("No stimulus components found")
        return

    for component_name, stim_conditions in stimulus_conditions_dict.items():
        logger.info(f"Adding stimulus data for component: {component_name}")
        add_stimulus_data(nwbfile, stim_conditions, component_name)


def add_states_data(
    nwbfile: NWBFile,
    experiment_module: Any,
    session_key: Dict[str, Any],
    valid_trial_indices: List[int],
) -> None:
    """
    Add states data to NWB file.

    Args:
        nwbfile: NWB file object
        experiment_module: DataJoint experiment module
        session_key: Primary key identifying the session
        valid_trial_indices: List of valid trial indices
    """
    logger.info("Add states per trial")
    if not valid_trial_indices:
        logger.warning("No valid trial indices for states data")
        return

    states_module = nwbfile.create_processing_module(
        name="States", description="States timestamps for each trial"
    )

    states_table = (
        experiment_module.Trial.StateOnset
        & session_key
        & f"trial_idx<={max(valid_trial_indices)}"
    )

    dynamic_table = create_dynamic_table_from_dj_table(
        states_table,
        "States Onset",
        str(states_table._heading),
        skip_columns=["animal_id", "session"],
    )

    states_module.add(dynamic_table)


def save_nwb_file(nwbfile: NWBFile, filename: str) -> None:
    """
    Save NWB file to disk.

    Args:
        nwbfile: NWB file object
        filename: Output filename
    """
    if os.path.exists(filename):
        logger.warning(f"Warning: File '{filename}' already exists and cannot be overwritten.")
        return
    with NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)


def setup_datajoint_connection(
    config_path: Optional[str] = None,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Set up DataJoint connection and create virtual modules.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of virtual modules (experiment, stimulus, behavior, recording, interface)
    """
    config = ConfigurationManager(config_path)
    dj_conf = config.get_datajoint_config()
    logger.info(f"Connecting to database: {dj_conf['database.host']}")
    dj.config.update(dj_conf)

    schemata = config.get("SCHEMATA")
    virtual_modules, _ = create_virtual_modules(schemata)

    return (
        virtual_modules["experiment"],
        virtual_modules["stimulus"],
        virtual_modules["behavior"],
    )


def get_session_classes(experiment: Any, session_key: Dict[str, Any]) -> SessionClasses:
    """
    Get the classes for the session_key.

    Args:
        experiment_module: DataJoint experiment module
        session_key: Primary key identifying the session

    Returns:
        SessionClasses containing stimulus, behavior, and experiment class arrays

    Raises:
        NWBExportError: If no classes found for session
    """
    session_classes = (experiment.Condition * experiment.Trial) & session_key

    if not session_classes:
        raise NWBExportError(f"No classes found for session {session_key}")

    return SessionClasses(
        stimulus=np.unique(session_classes.fetch("stimulus_class")),
        behavior=np.unique(session_classes.fetch("behavior_class")),
        experiment=np.unique(session_classes.fetch("experiment_class")),
    )


@contextmanager
def nwb_file_writer(filename: Union[str, Path], overwrite: bool = False):
    """
    Context manager for writing NWB files.

    Args:
        filename: Output filename
        overwrite: Whether to overwrite existing files

    Yields:
        NWBHDF5IO writer object

    Raises:
        FileExistsError: If file exists and overwrite=False
    """
    filename = Path(filename)

    if filename.exists() and not overwrite:
        raise FileExistsError(
            f"File '{filename}' already exists. Use overwrite=True to replace it."
        )

    if overwrite and filename.exists():
        filename.unlink()

    try:
        with NWBHDF5IO(str(filename), "w") as io:
            yield io
    except Exception as e:
        logger.error(f"Failed to save NWB file {filename}: {e}")
        raise


def export_to_nwb(
    animal_id: int,
    session_id: int,
    output_filename: Optional[str] = None,
    # NWB file parameters
    experimenter: str = "Unknown",
    lab: str = "Your Lab Name",
    institution: str = "Your Institution",
    session_description: Optional[str] = None,
    # Subject parameters
    age: str = "Unknown",
    subject_description: str = "laboratory mouse",
    species: str = "Unknown",
    sex: str = "U",
    # Additional options
    overwrite: bool = False,
    return_nwb_object: bool = False,
    config_path: Optional[str] = None,
) -> Union[str, Tuple[str, NWBFile]]:
    """
    Export experimental data from DataJoint tables to NWB format.

    This function creates an NWB file containing all experimental data for a specific
    animal and session, including trials, conditions, activity, rewards, stimuli, and states.

    Args:
        animal_id: Unique identifier for the animal
        session_id: Session identifier
        output_filename: Output filename. If None, auto-generates based on animal_id and session_id

        # NWB File Parameters:
        experimenter: Name of the experimenter (default: "Unknown")
        lab: Laboratory name (default: "Your Lab Name")
        institution: Institution name (default: "Your Institution")
        session_description: Description of the session. If None, auto-generates

        # Subject Parameters:
        age: Age of the subject in ISO 8601 format (default: "Unknown")
        subject_description: Description of the subject (default: "laboratory mouse")
        species: Species of the subject (default: "Unknown")
        sex: Sex of the subject - "M", "F", "U" for unknown, or "O" for other (default: "U")

        # Additional Options:
        overwrite: Whether to overwrite existing files (default: False)
        return_nwb_object: Whether to return the NWBFile object along with filename (default: False)

    Returns:
        str: Path to the saved NWB file
        or
        Tuple[str, NWBFile]: Path and NWBFile object if return_nwb_object=True

    Raises:
        ValueError: If no session is found for the provided animal_id and session_id
        FileExistsError: If output file exists and overwrite=False

    """
    # Create session key
    session_key = {"animal_id": animal_id, "session": session_id}

    # Generate default filename if not provided
    if output_filename is None:
        output_filename = f"nwb_animal_{animal_id}_session_{session_id}.nwb"

    # Generate default session description if not provided
    if session_description is None:
        session_description = f"Ethopy experimental session - Animal ID: {animal_id}, Session: {session_id}"

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting data for Animal {animal_id}, Session {session_id}")
    logger.info(f"Output file: {output_path}")

    try:
        # Set up DataJoint connection
        # ToDo: add parameters from recording and interface schemas
        experiment, stimulus, behavior = setup_datajoint_connection(config_path)

        # Create NWB file
        nwbfile = create_nwb_file(
            experiment, session_key, session_description, experimenter, lab, institution
        )

        # Add subject information
        nwbfile.subject = create_subject(
            animal_id, age, subject_description, species, sex
        )

        # Get class information
        class_names = get_session_classes(experiment, session_key)
        logger.info(
            f"Session classes - Stimulus: {class_names.stimulus}, "
            f"Behavior: {class_names.behavior}, Experiment: {class_names.experiment}"
        )

        trial_data = process_trial_states(experiment, session_key)

        if trial_data.valid_indices:
            logger.info(f"Processing {len(trial_data.valid_indices)} valid trials")

            # Get conditions
            exp_conditions = get_experiment_conditions(
                experiment, session_key, class_names.experiment[0]
            )

            # Check if stimulus is compound (contains underscore)
            stimulus_class = class_names.stimulus[0]
            is_compound_stimulus = "_" in stimulus_class

            if is_compound_stimulus:
                logger.info(f"Detected compound stimulus: {stimulus_class}")
                stimulus_conditions_dict = get_multiple_stimulus_conditions(
                    stimulus, session_key, stimulus_class
                )
                # For trial hash, use the first available stimulus conditions
                if stimulus_conditions_dict:
                    stim_conditions = next(iter(stimulus_conditions_dict.values()))
                else:
                    logger.error(
                        f"No stimulus conditions found for any component of {stimulus_class}"
                    )
                    stim_conditions = None
            else:
                stim_conditions = get_stimulus_conditions(
                    stimulus, session_key, stimulus_class
                )
                stimulus_conditions_dict = {stimulus_class: stim_conditions}

            beh_conditions = get_behavior_conditions(
                behavior, session_key, class_names.behavior[0]
            )

            # Create trial hash and add trials
            trial_hash = (
                exp_conditions
                * stim_conditions.proj(time_stim="time")
                * beh_conditions.proj(time_beh="time")
            )
            trial_hash = trial_hash & f"trial_idx<={max(trial_data.valid_indices)}"

            keep_columns = [
                "trial_idx",
                "stimulus_class",
                "behavior_class",
                "experiment_class",
                "cond_hash",
                "stim_hash",
                "beh_hash",
            ]

            # Deduplicate using dj.U() - get unique combinations of keep_columns
            trial_hash_unique = dj.U(*keep_columns) & trial_hash

            add_trials_to_nwb(nwbfile, trial_hash_unique, trial_data, keep_columns)
            # add_trials_to_nwb(nwbfile, trial_hash, trial_data, keep_columns)

            # Add conditions metadata
            add_conditions_module(
                nwbfile, exp_conditions, stim_conditions, beh_conditions, class_names
            )

            # Add stimulus data
            if is_compound_stimulus and stimulus_conditions_dict:
                logger.info(
                    f"Adding multiple stimulus components: {list(stimulus_conditions_dict.keys())}"
                )
                add_multiple_stimulus_data(nwbfile, stimulus_conditions_dict)
            elif stim_conditions is not None:
                add_stimulus_data(nwbfile, stim_conditions, stimulus_class)
            else:
                logger.warning("No stimulus data to add")

            # Add states data
            add_states_data(nwbfile, experiment, session_key, trial_data.valid_indices)
        else:
            logger.warning(
                "No valid trials found with both PreTrial and InterTrial states"
            )

        # Add activity and reward data
        add_activity_data(nwbfile, behavior, session_key)
        add_reward_data(nwbfile, behavior, session_key)

        # In export_to_nwb, before the write:
        inhomogeneous_clmns = inhomogeneous_columns(nwbfile)
        if inhomogeneous_clmns:
            for c in inhomogeneous_clmns:
                logger.error(f"Inhomogeneous column: {c}")
            raise NWBExportError(
                f"Found {len(inhomogeneous_clmns)} columns with inhomogeneous data"
            )

        # Save the file
        with nwb_file_writer(output_path, overwrite) as io:
            io.write(nwbfile)

        logger.info(f"Successfully exported NWB file: {output_path}")

        return (str(output_path), nwbfile) if return_nwb_object else str(output_path)

    except Exception as e:
        logger.error(f"Error during NWB export: {e}")
        raise NWBExportError(f"Export failed: {e}") from e


def batch_export_to_nwb(
    animal_session_list: List[Tuple[int, int]],
    output_directory: str = "nwb_exports",
    **kwargs,
) -> List[str]:
    """
    Export multiple sessions to NWB format in batch.

    Args:
        animal_session_list: List of (animal_id, session_id) tuples
        output_directory: Directory to save NWB files
        **kwargs: Additional parameters passed to export_to_nwb()

    Returns:
        List of successfully exported filenames
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    exported_files = []
    failed_exports = []

    for animal_id, session_id in animal_session_list:
        try:
            filename = output_dir / f"nwb_animal_{animal_id}_session_{session_id}.nwb"

            result = export_to_nwb(
                animal_id=animal_id,
                session_id=session_id,
                output_filename=str(filename),
                **kwargs,
            )

            exported_files.append(result)
            logger.info(f"Exported: Animal {animal_id}, Session {session_id}")

        except Exception as e:
            failed_exports.append((animal_id, session_id, str(e)))
            logger.error(f"Failed: Animal {animal_id}, Session {session_id} - {e}")

    # Summary
    logger.info(
        f"Batch Export Summary: {len(exported_files)} succeeded, {len(failed_exports)} failed"
    )

    if failed_exports:
        logger.error("Failed sessions:")
        for animal_id, session_id, error in failed_exports:
            logger.error(f"  Animal {animal_id}, Session {session_id}: {error}")

    return exported_files


if __name__ == "__main__":
    try:
        animal_id = int(input("Enter animal_id: "))
        session_id = int(input("Enter session_id: "))
        export_to_nwb(animal_id=animal_id, session_id=session_id, overwrite=True)
    except KeyboardInterrupt:
        logger.info("Export cancelled by user")
    except Exception as e:
        logger.error(f"Export failed: {e}")
