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


def _get_session_timestamp(experiment: Any, session_key: Dict[str, Any]) -> datetime:
    """
    Fetch the session timestamp for a given session key.

    Args:
        experiment: DataJoint experiment module
        session_key: Primary key identifying the session

    Returns:
        The session timestamp with timezone information

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


def milliseconds_to_seconds(milliseconds: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert milliseconds to seconds."""
    return milliseconds / 1000.0


def get_non_empty_children(table: dj.Table) -> List[dj.Table]:
    """Get all non-empty children of a DataJoint table."""
    children = table.children(as_objects=True)
    return [child for child in children if len(child) > 0]


def combine_children_tables(children: List[dj.Table]) -> dj.Table:
    """Combine all children tables using the join operator."""
    return reduce(lambda x, y: x * y, children)


def get_stimulus_conditions(stimulus_module: Any, session_key: Dict[str, Any], class_name: str) -> dj.Table:
    """
    Fetch stimulus conditions for a given class.

    Args:
        stimulus_module: DataJoint stimulus module
        session_key: Primary key identifying the session
        class_name: Name of the stimulus class

    Returns:
        DataJoint table with stimulus conditions
    """
    stim_class = getattr(stimulus_module, class_name)
    return (stimulus_module.StimCondition.Trial & session_key) * stim_class


def get_experiment_conditions(experiment_module: Any, session_key: Dict[str, Any], class_name: str) -> dj.Table:
    """
    Fetch experiment conditions for a given class.

    Args:
        experiment_module: DataJoint experiment module
        session_key: Primary key identifying the session
        class_name: Name of the experiment class

    Returns:
        DataJoint table with experiment conditions
    """
    exp_class = getattr(experiment_module.Condition, class_name)
    return ((experiment_module.Trial() & session_key) * experiment_module.Condition) * exp_class


def get_behavior_conditions(behavior_module: Any, session_key: Dict[str, Any], class_name: str) -> dj.Table:
    """
    Fetch behavior conditions for a given class.

    Args:
        behavior_module: DataJoint behavior module
        session_key: Primary key identifying the session
        class_name: Name of the behavior class

    Returns:
        DataJoint table with behavior conditions

    Raises:
        NWBExportError: If no children found for behavior class
    """
    beh_class = getattr(behavior_module, class_name)
    children = beh_class.children(as_objects=True)

    if len(children) > 1:
        comb_tables = combine_children_tables(children)
    elif len(children) == 1:
        comb_tables = children[0]
    else:
        raise NWBExportError(f"No children found for behavior class {class_name}")

    return (behavior_module.BehCondition.Trial() & session_key) * comb_tables


def get_table_columns(table: dj.Table) -> List[str]:
    """
    Fetch columns from a DataJoint table.

    Args:
        table: DataJoint table

    Returns:
        List of column names
    """
    return table.heading.names


def remove_columns(table: dj.Table, columns_to_remove: List[str]) -> List[str]:
    """
    Remove specified columns from a DataJoint table.

    Args:
        table: DataJoint table
        columns_to_remove: List of column names to remove

    Returns:
        List of remaining column names
    """
    return [col for col in table.heading.names if col not in columns_to_remove]


def convert_milliseconds_to_seconds(
    milliseconds: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert milliseconds to seconds.

    Args:
        milliseconds: Time in milliseconds

    Returns:
        Time in seconds
    """
    return milliseconds / 1000.0


def get_children_tables(table: dj.Table) -> List[dj.Table]:
    """
    Get all non-empty children of a DataJoint table.

    Args:
        table: The table to get children from

    Returns:
        List of child tables as DataJoint objects
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
        (experiment.Trial.StateOnset & session_key)
        .fetch(format="frame")
        .reset_index()
    )

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

    if complete_trials.empty:
        logger.warning("No trials found with both PreTrial and InterTrial states")
        return TrialData([], [], [])

    # Convert to seconds and extract times
    pretrial_times = milliseconds_to_seconds(complete_trials["PreTrial"]).tolist()
    intertrial_times = milliseconds_to_seconds(complete_trials["InterTrial"]).tolist()
    valid_trial_indices = complete_trials.index.tolist()

    # Log processing results
    total_trials = len(states_df["trial_idx"].unique())
    logger.info(f"Processed {len(complete_trials)}/{total_trials} trials")

    return TrialData(pretrial_times, intertrial_times, valid_trial_indices)


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
        logger.warning("Trial hashes are mising check experiment,condition and stimulus hashes ")
        return

    if not trial_data.valid_indices:
        logger.warning("No valid trial indices provided")
        return
    logger.info(f'trial_data {len(trial_data)}, {len(trial_data.pretrial_times)}')

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
    logging.info(f"all_columns {all_columns}")

    for i, trial in enumerate(trial_hash.fetch(as_dict=True)):
        # if i >= len(trial_data.pretrial_times) or i >= len(trial_data.intertrial_times):
        #     logger.warning(f"Timing data missing for trial {i}")
        #     continue

        trial.update({
            "start_time": float(trial_data.pretrial_times[i]),
            "stop_time": float(trial_data.intertrial_times[i]),
            "id": trial_data.valid_indices[i]
        })

        # Remove unwanted columns
        for col in columns_to_remove:
            trial.pop(col, None)

        nwbfile.add_trial(**trial)


def add_conditions_module(
    nwbfile: NWBFile,
    exp_conditions: dj.Table,
    stim_conditions: dj.Table,
    beh_conditions: dj.Table,
    class_names: SessionClasses,
) -> None:
    """
    Create and add conditions metadata module to NWB file.

    Args:
        nwbfile: NWB file object
        exp_conditions: Experiment conditions table
        stim_conditions: Stimulus conditions table
        beh_conditions: Behavior conditions table
        class_names: Session class names
    """
    logger.info("Add condition parameters for experiment, behavior and stimuli")
    meta_data = nwbfile.create_processing_module(
        name="Conditions",
        description="Conditions parameters for experiment, behavior and stimuli",
    )

    # Helper function to add condition table
    def add_condition_table(conditions: dj.Table, name: str, description: str, columns_to_remove: List[str]):
        df = conditions.fetch(format="frame").reset_index()
        columns_of_interest = [col for col in conditions.heading.names if col not in columns_to_remove]
        unique_combinations = df[columns_of_interest].drop_duplicates()

        if unique_combinations.empty:
            logger.warning(f"No data found for {name} conditions")
            return

        table = DynamicTable(name=name, description=description, id=[])

        # Add columns
        trial_columns = {
            tag: {"name": tag, "description": conditions.heading.attributes[tag].comment}
            for tag in conditions.heading.names
            if tag in columns_of_interest
        }

        for column_info in trial_columns.values():
            table.add_column(
                name=column_info["name"], description=column_info["description"]
            )

        # Add rows
        for trial in unique_combinations.to_dict(orient="records"):
            table.add_row(**trial)

        meta_data.add(table)

    # Add condition tables
    skip_cols = ["animal_id", "session", "trial_idx", "time"]

    add_condition_table(
        beh_conditions, "Behavior", class_names.behavior[0], skip_cols
    )
    add_condition_table(
        exp_conditions, "Experiment", class_names.experiment[0], skip_cols
    )
    add_condition_table(
        stim_conditions, "Stimulus", class_names.stimulus[0],
        skip_cols + ["start_time", "end_time"]
    )


def create_dynamic_table_from_dj_table(
    table: dj.Table,
    table_name: str,
    description: str,
    skip_columns: Optional[List[str]] = None,
    id_column: str = "trial_idx"
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
        dynamic_table.add_column(
            name=column_info["name"],
            description=column_info["description"]
        )

    # Add rows
    for trial in table.fetch(as_dict=True):
        trial["id"] = trial.get(id_column, len(dynamic_table))
        for key in skip_columns:
            trial.pop(key, None)
        dynamic_table.add_row(**trial)

    return dynamic_table


def add_activity_data(nwbfile: NWBFile, behavior_module: Any, session_key: Dict[str, Any]) -> None:
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

    activity_tables = get_non_empty_children(behavior_module.Activity & session_key)

    for table in activity_tables:
        dynamic_table = create_dynamic_table_from_dj_table(
            table,
            table._table_name,
            str(table._heading),
            skip_columns=["animal_id", "session", "start_time", "stop_time"]
        )
        activity_module.add(dynamic_table)


def add_reward_data(nwbfile: NWBFile, behavior_module: Any, session_key: Dict[str, Any]) -> None:
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
    Add stimulus data to NWB file.

    Args:
        nwbfile: NWB file object
        stim_conditions: Stimulus conditions table
        stimulus_class: Name of stimulus class
    """
    logger.info("Add Stimulus per trial")
    df = stim_conditions.fetch(format="frame").reset_index()
    columns_of_interest = ["trial_idx", "start_time", "end_time", "stim_hash"]
    unique_combinations = df[columns_of_interest].drop_duplicates()

    if unique_combinations.empty:
        logger.warning("No stimulus data found")
        return

    table = create_dynamic_table_from_dj_table(
        stim_conditions, stimulus_class, str(stim_conditions._heading)
    )
    nwbfile.add_stimulus(table)


def add_states_data(
    nwbfile: NWBFile,
    experiment_module: Any,
    session_key: Dict[str, Any],
    valid_trial_indices: List[int]
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
        skip_columns=["animal_id", "session"]
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
        print(f"Warning: File '{filename}' already exists and cannot be overwritten.")
        return
    with NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)
    print(f"NWB file saved as: {filename}")


def setup_datajoint_connection(config_path: Optional[str] = None) -> Tuple[Any, Any, Any, Any, Any]:
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
        experiment=np.unique(session_classes.fetch("experiment_class"))
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
        logger.info(f"NWB file saved as: {filename}")
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
    config_path: Optional[str] = None
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
    print("session_key ", session_key)
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
            exp_conditions = get_experiment_conditions(experiment, session_key, class_names.experiment[0])
            stim_conditions = get_stimulus_conditions(stimulus, session_key, class_names.stimulus[0])
            beh_conditions = get_behavior_conditions(behavior, session_key, class_names.behavior[0])

            # Create trial hash and add trials
            trial_hash = (
                exp_conditions
                * stim_conditions.proj(time_stim="time")
                * beh_conditions.proj(time_beh="time")
            )
            trial_hash = trial_hash & f"trial_idx<={max(trial_data.valid_indices)}"

            keep_columns = [
                "trial_idx", "stimulus_class", "behavior_class",
                "experiment_class", "cond_hash", "stim_hash", "beh_hash"
            ]

            add_trials_to_nwb(nwbfile, trial_hash, trial_data, keep_columns)

            # Add conditions metadata
            add_conditions_module(nwbfile, exp_conditions, stim_conditions, beh_conditions, class_names)

            # Add stimulus data
            add_stimulus_data(nwbfile, stim_conditions, class_names.stimulus[0])

            # Add states data
            add_states_data(nwbfile, experiment, session_key, trial_data.valid_indices)
        else:
            logger.warning("No valid trials found with both PreTrial and InterTrial states")

        # Add activity and reward data
        add_activity_data(nwbfile, behavior, session_key)
        add_reward_data(nwbfile, behavior, session_key)

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
    **kwargs
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
    logger.info(f"Batch Export Summary: {len(exported_files)} succeeded, {len(failed_exports)} failed")

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
