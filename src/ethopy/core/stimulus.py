"""Module for handling stimulus presentation in behavioral experiments.

This module provides a base Stimulus class that handles the presentation of various
types of stimuli during behavioral experiments. It includes functionality for stimulus
preparation, presentation, logging, and cleanup.
"""

import os
from typing import Any, Dict, List, Optional

import datajoint as dj

from ethopy.core.logger import (  # pylint: disable=W0611, # noqa: F401
    experiment,
    stimulus,
)
from ethopy.utils.helper_functions import DictStruct, FillColors
from ethopy.utils.presenter import Presenter
from ethopy.utils.timer import Timer


class Stimulus:
    """Base class for handling stimulus presentation in behavioral experiments.

    This class provides the core functionality for managing stimuli in behavioral
    experiments, including initialization, presentation, logging, and cleanup. It can be
    subclassed to implement specific types of stimuli.

    Attributes:
        cond_tables (List[str]): List of condition table names.
        required_fields (List[str]): List of required fields for stimulus conditions.
        default_key (Dict[str, Any]): Default key-value pairs for stimulus conditions.
        curr_cond (Dict[str, Any]): Current stimulus condition parameters.
        conditions (List[Dict[str, Any]]): List of all stimulus conditions.
        timer (Timer): Timer object for tracking stimulus timing.
        period (str): Current experimental period ('Trial' by default).
        in_operation (bool): Flag indicating if stimulus is currently active.
        flip_count (int): Counter for screen flips.
        photodiode (bool): Flag for photodiode triggering.
        rec_fliptimes (bool): Flag for recording flip times.
        fill_colors (DictStruct): Structure containing color values for different states

    """

    def __init__(self) -> None:
        """Initialize stimulus attributes."""
        self.cond_tables: List[str] = []
        self.required_fields: List[str] = []
        self.default_key: Dict[str, Any] = {}
        self.curr_cond: Dict[str, Any] = {}
        self.timer: Timer = Timer()
        self.period: str = "Trial"
        self.in_operation: bool = False
        self.flip_count: int = 0
        self.photodiode: bool = False
        self.rec_fliptimes: bool = False
        self.fill_colors: FillColors = FillColors()
        self.logger = None
        self.exp = None
        self.monitor = None
        self.Presenter = None
        self.start_time: Optional[float] = None

    def init(self, exp) -> None:
        """Initialize stimulus with experiment object and setup screen properties.

        Args:
            exp (Experiment): Experiment object containing logger and interface
                components.

        """
        self.logger = exp.logger
        self.exp = exp
        screen_properties = self.logger.get(
            schema="interface",
            table="SetupConfiguration.Screen",
            key=self.exp.params,
            as_dict=True,
        )
        self.monitor = DictStruct(screen_properties[0])
        if self.logger.is_pi:
            cmd = (
                "echo %d > /sys/class/backlight/rpi_backlight/brightness"
                % self.monitor.intensity
            )
            os.system(cmd)
            exp.interface.setup_touch_exit()

    def setup(self) -> None:
        """Set up stimulus presentation environment.

        Initializes the Presenter object with monitor settings and background color.
        Should be called before starting the experiment.
        """
        self.Presenter = Presenter(
            self.logger,
            self.monitor,
            background_color=self.fill_colors.background,
            photodiode=self.photodiode,
            rec_fliptimes=self.rec_fliptimes,
        )

    def prepare(self, curr_cond=False, stim_period="") -> None:
        """Prepare stuff for presentation before trial starts."""
        self.curr_cond = curr_cond if stim_period == "" else curr_cond[stim_period]
        self.period = stim_period

    def start(self) -> None:
        """Start stimulus."""
        self.in_operation = True
        self.log_start()
        self.timer.start()

    def present(self) -> None:
        """Present stimulus.

        This is a placeholder method that should be overridden by subclasses
        to implement specific stimulus presentation logic.
        """

    def fill(self, color=False) -> None:
        """Stimulus hidding method."""
        if not color:
            color = self.fill_colors.background
        if self.fill_colors.background:
            self.Presenter.fill(color)

    def stop(self) -> None:
        """Stop stimulus."""
        self.fill()
        self.log_stop()
        self.in_operation = False

    def exit(self) -> None:
        """Exit stimulus stuff."""
        self.Presenter.quit()

    def ready_stim(self) -> None:
        """Stim Cue for ready."""
        if self.fill_colors.ready:
            self.fill(self.fill_colors.ready)

    def reward_stim(self) -> None:
        """Stim Cue for reward."""
        if self.fill_colors.reward:
            self.fill(self.fill_colors.reward)

    def punish_stim(self) -> None:
        """Stim Cue for punishment."""
        if self.fill_colors.punish:
            self.fill(self.fill_colors.punish)

    def start_stim(self) -> None:
        """Stim Cue for start."""
        if self.fill_colors.start:
            self.fill(self.fill_colors.start)

    def log_start(self) -> None:
        """Start timer for the log of stimlus condition."""
        self.start_time = self.logger.logger_timer.elapsed_time()
        self.exp.interface.sync_out(True)

    def log_stop(self) -> None:
        """Log stimulus condition start & stop time."""
        stop_time = self.logger.logger_timer.elapsed_time()
        self.exp.interface.sync_out(False)
        self.logger.log(
            "StimCondition.Trial",
            dict(
                period=self.period,
                stim_hash=self.curr_cond["stim_hash"],
                start_time=self.start_time,
                end_time=stop_time,
            ),
            schema="stimulus",
        )

    def make_conditions(
        self, conditions: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate and store stimulus conditions.

        Args:
            conditions: List of condition dictionaries to process.

        Returns:
            List of processed condition dictionaries with hashes.

        Raises:
            AssertionError: If required fields are missing from any condition.

        """
        # check for any missing field from the required fields and add the default keys
        for cond in conditions:
            missing_fields = [
                field for field in self.required_fields if field not in cond
            ]
            assert not missing_fields, (
                f"Missing Stimulus required fields: {missing_fields}"
            )
            cond.update({**self.default_key, **cond})
        # log stim conditions
        conditions = self.exp.log_conditions(
            conditions,
            schema="stimulus",
            hash_field="stim_hash",
            condition_tables=["StimCondition"] + self.cond_tables,
        )
        return conditions

    def name(self) -> str:
        """Get the name of the stimulus class.

        Returns:
            (str):Name of the current stimulus class.

        """
        return type(self).__name__


@stimulus.schema
class StimCondition(dj.Manual):
    """Datajoint table for the stimulus presentation hash."""

    definition = """
    # This class handles the stimulus presentation use function overrides for each
    # stimulus class
    stim_hash            : char(24)   # unique stimulus condition hash
    """

    class Trial(dj.Part):
        """Datajoint table for the Stimulus onset timestamps."""

        definition = """
        # Stimulus onset timestamps
        -> experiment.Trial
        period='Trial'       : varchar(16)
        ---
        -> StimCondition
        start_time           : int   # start time from session start (ms)
        end_time=NULL        : int   # end time from session start (ms)
        """
