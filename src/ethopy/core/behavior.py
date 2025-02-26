"""Core behavior module handles behavioral variables, responses, and reward management.

This module provides the core functionality for managing animal behavior in experimental
setups, including response tracking, reward delivery, and behavioral data logging. It
interfaces with hardware components and maintains experiment state.
"""

from dataclasses import dataclass, fields
from dataclasses import field as datafield
from datetime import datetime, timedelta
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import datajoint as dj
import numpy as np

from ethopy.core.experiment import ExperimentClass
from ethopy.core.logger import (  # pylint: disable=W0611, # noqa: F401
    behavior,
    experiment,
)


class Behavior:
    """Manages behavioral variables and interactions in experimental setups.

    This class handles all aspects of behavioral monitoring and control:
    - Response tracking and validation
    - Reward delivery and management
    - Behavioral data logging through logger module
    - Condition management and history tracking
    - Interface with hardware components through experiment module

    Attributes:
        interface (List[Any]): List of interface objects
        required_fields (List[str]): Required fields for condition validation
        curr_cond (List[Dict]): Current condition parameters
        response (List[Any]): Current response data
        licked_port (int): ID of the most recently licked port
        logging (bool): Whether logging is enabled
        reward_amount (Dict[int, float]): Reward amounts by port
        choice_history (List[float]): History of animal choices
        reward_history (List[float]): History of rewards given

    """

    def __init__(self) -> None:
        """Initialize the behavior class with default values.

        Attributes:
            cond_tables (List[str]): A list of condition table names.
            default_key (Dict[str, Any]): A dictionary for default key-value pairs.
            interface (Any): The interface object (initially None).
            required_fields (List[str]): A list of required field names.
            curr_cond (List[Dict[str, Any]]): A list of current trial conditions.
            response (List[Any]): A list for storing responses.
            licked_port (int): The port is that was licked.
            logging (bool): A flag indicating if logging is enabled.
            reward_amount (Dict[int, float]): A dictionary mapping port numbers to
                reward amounts.
            choice_history (List[float]): A list of choice history values.
            reward_history (List[float]): A list of reward history values.
            punish_history (List[float]): A list of punishment history values.
            choices (np.ndarray): An array of choices.
            response_queue (Queue): A queue for storing responses with a maximum size
                of 4.
            last_lick: The last lick event (initially None).
            params: Parameters for the experiment (initially None).
            exp: The experiment object (initially None).
            logger: The logger object (initially None).

        """
        self.cond_tables: List[str] = []
        self.default_key: Dict[str, Any] = {}
        self.interface = None
        self.required_fields: List[str] = []
        self.curr_cond: List[Dict[str, Any]] = []
        self.response: List[Any] = []
        self.licked_port: int = 0
        self.logging: bool = False
        self.reward_amount: Dict[int, float] = {}
        self.choice_history: List[float] = []
        self.reward_history: List[float] = []
        self.punish_history: List[float] = []
        self.choices = np.array([])
        self.response_queue: Queue = Queue(maxsize=4)
        self.last_lick = None

        self.params = None
        self.exp = None
        self.logger = None

    def setup(self, exp: ExperimentClass) -> None:
        """Set up behavior."""
        self.params = exp.params
        self.exp = exp
        self.logger = exp.logger
        self.interface = exp.interface

        self.choices = np.array(np.empty(0))
        self.choice_history = []  # History term for bias calculation
        self.reward_history = []  # History term for performance calculation
        self.punish_history = []
        self.reward_amount = dict()
        self.response, self.last_lick = BehActivity(), BehActivity()
        self.response_queue = Queue(maxsize=4)
        self.logging = True

    def is_ready(self, duration: int, since: int = 0) -> Tuple[bool, int]:
        """Check if has been in position for a duration."""
        return True, 0

    def get_response(self, since: int = 0, clear: bool = True) -> bool:
        """Check for valid behavioral responses since a given time point.

        Args:
            since: Time reference point in milliseconds
            clear: Whether to clear existing responses before checking

        Returns:
            Whether a valid response was detected

        """
        # set a flag to indicate whether there is a valid response since the given time
        _valid_response = False

        # clear existing response if clear is True
        if clear:
            self.response = BehActivity()
            self.licked_port = 0

        while not self.response_queue.empty():
            _response = self.response_queue.get()
            if not _valid_response and _response.time >= since and _response.port:
                self.response = _response
                _valid_response = True

        return _valid_response

    def is_licking(
        self, since: int = 0, reward: bool = False, clear: bool = True
    ) -> int:
        """Check for licking activity since a given time point.

        This method can be used in two ways:
        1. To detect any licking activity since the given time
        2. To check for rewarded licking (when reward=True) where only licks at reward
        ports count

        Args:
            since (int, optional): Time reference point in milliseconds. Defaults to 0.
            reward (bool, optional): Whether to only count licks at reward ports.
                Defaults to False.
            clear (bool, optional): Whether to reset last_lick after checking.
                Defaults to True.

        Returns:
            int: Port number of valid lick (0 if no valid lick detected)

        """
        # check if there is any licking since the given time
        if self.last_lick.time >= since and self.last_lick.port:
            # if reward == False return the licked port number
            # if reward == True check if the licked port is alse a reward port
            if not reward or (reward and self.last_lick.reward):
                self.licked_port = self.last_lick.port
            else:
                self.licked_port = 0
        else:
            self.licked_port = 0
        # by default if it licked since the last time this function was called
        if clear:
            self.last_lick = BehActivity()

        return self.licked_port

    def reward(self) -> None:
        """Reward action."""
        return True

    def punish(self) -> None:
        """Punish action."""

    def exit(self) -> None:
        """Clean up and exit the behavior module."""
        self.logging = False

    def log_activity(self, activity_key: dict) -> int:
        """Log behavioral activity to the database.

        Updates last_lick and licked_port variables, manages response queue,
        and logs the activity in the database.

        Args:
            activity_key (dict): Dictionary containing activity parameters

        Returns:
            (int):Timestamp of the logged activity in milliseconds

        """
        activity = BehActivity(**activity_key)
        # if activity.time is not set, set it to the current time
        if not activity.time:
            activity.time = self.logger.logger_timer.elapsed_time()
        key = {**self.logger.trial_key, **activity.__dict__}
        # log the activity in the database
        if self.exp.in_operation and self.logging:
            self.logger.log("Activity", key, schema="behavior", priority=10)
            self.logger.log("Activity." + activity.type, key, schema="behavior")
        # if activity.type == 'Response': append to the response queue
        if activity.response:
            if self.response_queue.full():
                self.response_queue.get()
            self.response_queue.put(activity)
        # get the last lick and licked port to use it in is_licking function
        if activity.type == "Lick":
            self.last_lick = activity
            self.licked_port = activity.port
        return key["time"]

    def log_reward(self, reward_amount: float) -> None:
        """Log delivered reward to the database.

        Args:
            reward_amount (float): Amount of reward delivered

        """
        if isinstance(self.curr_cond["reward_port"], list):
            self.curr_cond["reward_port"] = [self.licked_port]
            self.curr_cond["response_port"] = [self.licked_port]
        self.logger.log(
            "Rewards",
            {**self.curr_cond, "reward_amount": reward_amount},
            schema="behavior",
        )

    def make_conditions(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate, update with default_key and generate hash for stimulus conditions.

        Args:
            conditions: List of condition dictionaries

        Returns:
            Dictionary containing processed conditions and metadata

        """
        for cond in conditions:
            missing_fields = [
                field for field in self.required_fields if field not in cond
            ]
            assert not missing_fields, (
                f"Missing behavior required fields: {missing_fields}"
            )
            cond.update({**self.default_key, **cond})

        if self.cond_tables:
            return self.exp.log_conditions(
                conditions=conditions,
                condition_tables=["BehCondition"] + self.cond_tables,
                schema="behavior",
                hash_field="beh_hash",
            )

        return self.exp.log_conditions(
            conditions=conditions, condition_tables=[], schema="behavior"
        )

    def prepare(self, condition: Dict[str, Any]) -> None:
        """Prepare for a new trial with given conditions.

        Args:
            condition: Dictionary of trial conditions

        """
        self.curr_cond = condition
        self.reward_amount = self.interface.calc_pulse_dur(condition["reward_amount"])
        self.logger.log(
            "BehCondition.Trial",
            dict(beh_hash=self.curr_cond["beh_hash"]),
            schema="behavior",
        )

    def update_history(
        self, choice: float = np.nan, reward: float = np.nan, punish: float = np.nan
    ) -> None:
        """Update choice and reward history.

        Args:
            choice: Choice made (port number)
            reward: Reward amount
            punish: Punishment value

        """
        if (
            np.isnan(choice)
            and (~np.isnan(reward) or ~np.isnan(punish))
            and self.response.time > 0
        ):
            choice = self.response.port
        self.choice_history.append(choice)
        self.reward_history.append(reward)
        self.punish_history.append(punish)
        self.logger.total_reward = np.nansum(self.reward_history)

    def get_false_history(self, h: int = 10) -> float:
        """Get history of false responses.

        Args:
            h: Number of trials to look back

        Returns:
            Cumulative product of false responses

        """
        idx = np.nan_to_num(self.punish_history)
        return np.nansum(np.cumprod(np.flip(idx[-h:], axis=0)))

    def is_sleep_time(self) -> bool:
        """Check if current time is within sleep period.

        Returns:
            Whether current time is in sleep period

        """
        now = datetime.now()
        start_time = self.logger.setup_info["start_time"]
        if isinstance(start_time, str):
            dt = datetime.strptime(start_time, "%H:%M:%S")
            start_time = timedelta(seconds=dt.hour * 3600 + dt.minute * 60 + dt.second)
        stop_time = self.logger.setup_info["stop_time"]
        if isinstance(stop_time, str):
            dt = datetime.strptime(stop_time, "%H:%M:%S")
            stop_time = timedelta(seconds=dt.hour * 3600 + dt.minute * 60 + dt.second)

        start = now.replace(hour=0, minute=0, second=0) + start_time
        stop = now.replace(hour=0, minute=0, second=0) + stop_time
        if stop < start:
            stop = stop + timedelta(days=1)
        time_restriction = now < start or now > stop
        return time_restriction

    def is_hydrated(self, rew: Optional[float] = None) -> bool:
        """Check if animal has received enough reward.

        Args:
            rew: Optional override for maximum reward amount

        Returns:
            Whether maximum reward threshold has been reached

        """
        if rew:
            return self.logger.total_reward >= rew
        elif self.params["max_reward"]:
            return self.logger.total_reward >= self.params["max_reward"]
        else:
            return False


@dataclass
class BehActivity:
    """Dataclass for tracking behavioral activity.

    Attributes:
        port: Port number where activity occurred
        type: Type of activity (e.g., 'Lick', 'Touch')
        time: Timestamp of activity
        in_position: Position status
        loc_x: X coordinate of activity
        loc_y: Y coordinate of activity
        theta: Angular position
        ready: Ready status
        reward: Whether activity was rewarded
        response: Whether activity was a valid response

    """

    port: int = datafield(compare=True, default=0, hash=True)
    type: str = datafield(compare=True, default="", hash=True)
    time: int = datafield(compare=False, default=0)
    in_position: int = datafield(compare=False, default=0)
    loc_x: int = datafield(compare=False, default=0)
    loc_y: int = datafield(compare=False, default=0)
    theta: int = datafield(compare=False, default=0)
    ready: bool = datafield(compare=False, default=False)
    reward: bool = datafield(compare=False, default=False)
    response: bool = datafield(compare=False, default=False)

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """Initialize the behavior object with the provided keyword arguments."""
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@behavior.schema
class Rewards(dj.Manual):
    """DataJoint table for tracking reward trials."""

    definition = """
    # reward trials
    -> experiment.Trial
    time			        : int 	           # time from session start (ms)
    ---
    reward_type             : varchar(16)
    reward_amount           : float            # reward amount
    """


@behavior.schema
class Activity(dj.Manual):
    """DataJoint table for tracking behavioral responses."""

    definition = """
    # Mouse behavioral response
    -> experiment.Trial
    """

    class Proximity(dj.Part):
        """DataJoint table for tracking proximity port information."""

        definition = """
        # Center port information
        -> Activity
        port                 : tinyint          # port id
        time	     	  	 : int           	# time from session start (ms)
        ---
        in_position          : tinyint
        """

    class Lick(dj.Part):
        """DataJoint table for licking."""

        definition = """
        # Lick timestamps
        -> Activity
        port                 : tinyint          # port id
        time	     	  	 : int           	# time from session start (ms)
        """

    class Touch(dj.Part):
        """DataJoint table for touch timestamps."""

        definition = """
        # Touch timestamps
        -> Activity
        loc_x               : int               # x touch location
        loc_y               : int               # y touch location
        time	     	    : int           	# time from session start (ms)
        """

    class Position(dj.Part):
        """DataJoint table for 2D possition timestamps."""

        definition = """
        # 2D possition timestamps
        -> Activity
        loc_x               : float             # x 2d location
        loc_y               : float             # y 2d location
        theta               : float             # direction in space
        time	     	    : int           	# time from session start (ms)
        """


@behavior.schema
class BehCondition(dj.Manual):
    """Datajoint table with a hash defining all the conditions."""

    definition = """
    # reward probe conditions
    beh_hash               : char(24)                     # unique reward hash
    """

    class Trial(dj.Part):
        """Datajoint table for keeping the hash for each trial."""

        definition = """
        # movie clip conditions
        -> experiment.Trial
        -> BehCondition
        time			      : int 	                # time from session start (ms)
        """
