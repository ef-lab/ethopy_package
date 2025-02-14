"""Core experiment module for experiment control.

This module provides the base classes and functionality for running behavioral
experiments. It includes:
- State machine implementation for experiment flow control
- Condition management and randomization
- Trial preparation and execution
- Performance tracking and analysis
- Integration with stimulus and behavior modules

The module is built around two main classes:
- State: Base class for implementing experiment states
- ExperimentClass: Base class for experiment implementation
"""

import logging
import time
from dataclasses import dataclass, field
from importlib import import_module
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import datajoint as dj
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from ethopy.core.interface import Interface
from ethopy.core.logger import Logger, experiment, mice
from ethopy.core.stimulus import Stimulus
from ethopy.utils.helper_functions import factorize, make_hash
from ethopy.utils.task_helper_funcs import format_params_print, get_parameters
from ethopy.utils.timer import Timer

log = logging.getLogger(__name__)


class State:
    """Base class for implementing experiment states.

    This class provides the template for creating states in the experiment state
    machine. Each state should inherit from this class and implement the required
    methods.

    Attributes:
        state_timer: Timer instance shared across all states
        __shared_state: Dictionary containing shared state variables

    """

    state_timer: Timer = Timer()
    __shared_state: Dict[str, Any] = {}

    def __init__(self, parent: Optional["ExperimentClass"] = None) -> None:
        """Initialize state with optional parent experiment.

        Args:
            parent: Parent experiment instance this state belongs to

        """
        self.__dict__ = self.__shared_state
        if parent:
            self.__dict__.update(parent.__dict__)

    def entry(self) -> None:
        """Execute actions when entering this state."""
        pass

    def run(self) -> None:
        """Execute the main state logic."""
        pass

    def next(self) -> str:
        """Determine the next state to transition to.

        Returns:
            Name of the next state to transition to

        Raises:
            AssertionError: If next() is not implemented by child class

        """
        raise AssertionError("next not implemented")

    def exit(self) -> None:
        """Execute actions when exiting this state."""
        pass


class StateMachine:
    """State machine implementation for experiment control flow.

    Manages transitions between experiment states and ensures proper execution
    of state entry/exit hooks. The state machine runs until it reaches the exit
    state.

    Attributes:
        states (Dict[str, State]): Mapping of state names to state instances
        futureState (State): Next state to transition to
        currentState (State): Currently executing state
        exitState (State): Final state that ends the state machine

    """

    def __init__(self, states: Dict[str, State]) -> None:
        """Initialize the state machine.

        Args:
            states: Dictionary mapping state names to state instances

        Raises:
            ValueError: If required states (Entry, Exit) are missing

        """
        if "Entry" not in states or "Exit" not in states:
            raise ValueError("StateMachine requires Entry and Exit states")

        self.states = states
        self.futureState = states["Entry"]
        self.currentState = states["Entry"]
        self.exitState = states["Exit"]

    # # # # Main state loop # # # # #
    def run(self) -> None:
        """Execute the state machine until reaching exit state.

        The machine will:
        1. Check for state transition
        2. Execute exit hook of current state if transitioning
        3. Execute entry hook of new state if transitioning
        4. Execute the current state's main logic
        5. Determine next state

        Raises:
            KeyError: If a state requests transition to non-existent state
            RuntimeError: If a state's next() method raises an exception

        """
        try:
            while self.futureState != self.exitState:
                if self.currentState != self.futureState:
                    self.currentState.exit()
                    self.currentState = self.futureState
                    self.currentState.entry()

                self.currentState.run()

                next_state = self.currentState.next()
                if next_state not in self.states:
                    raise KeyError(f"Invalid state transition to: {next_state}")

                self.futureState = self.states[next_state]

            self.currentState.exit()
            self.exitState.run()

        except Exception as e:
            raise RuntimeError(
                f"""State machine error in state
                    {self.currentState.__class__.__name__}: {str(e)}"""
            ) from e


class ExperimentClass:
    """Parent Experiment."""

    curr_trial = 0  # the current trial number in the session
    cur_block = 0  # the current block number in the session
    states = {}  # dictionary wiht all states of the experiment
    stims = {}  # dictionary with all stimulus classes
    stim = False  # the current condition stimulus class
    sync = False  # a boolean to make synchronization available
    un_choices = []
    blocks = []
    iter = []
    curr_cond = {}
    block_h = []
    has_responded = False
    resp_ready = False
    required_fields = []
    default_key = {}
    conditions = []
    cond_tables = []
    quit = False
    in_operation = False
    cur_block_sz = 0
    params = None
    logger = None
    conditions = []
    iter = []
    setup_conf_idx = 0
    interface = None
    beh = None
    trial_start = 0  # time in ms of the trial start

    def setup(self, logger: Logger, behavior_class, session_params: Dict):
        self.in_operation = False
        self.conditions = []
        self.iter = []
        self.quit = False
        self.curr_cond = {}
        self.block_h = []
        self.stims = dict()
        self.curr_trial = 0
        self.cur_block_sz = 0
        self.setup_conf_idx = session_params.get("setup_conf_idx", 0)
        session_params["setup_conf_idx"] = self.setup_conf_idx

        self.params = {**self.default_key, "setup_conf_idx": self.setup_conf_idx}

        self.logger = logger
        self.beh = behavior_class()
        self.interface = self._interface_setup(
            self.beh, self.logger, self.setup_conf_idx
        )
        self.interface.load_calibration()
        self.beh.setup(self)

        self.logger.log_session(
            session_params, experiment_type=self.cond_tables[0], log_task=True
        )

        self.session_timer = Timer()

        np.random.seed(0)  # fix random seed, it can be overidden in the task file

    def _interface_setup(
        self, beh, logger: Logger, setup_conf_idx: int
    ) -> Interface:
        interface_module = logger.get(
            schema="interface",
            table="SetupConfiguration",
            fields=["interface"],
            key={"setup_conf_idx": setup_conf_idx},
        )[0]
        log.info(f"Interface: {interface_module}")
        interface = getattr(
            import_module(f"ethopy.interfaces.{interface_module}"), interface_module
        )

        return interface(exp=self, beh=beh)

    def start(self):
        states = dict()
        for state in self.__class__.__subclasses__():  # Initialize states
            states.update({state().__class__.__name__: state(self)})
        state_control = StateMachine(states)
        self.interface.set_operation_status(True)
        state_control.run()

    def stop(self):
        self.stim.exit()
        self.interface.release()
        self.beh.exit()
        if self.sync:
            while self.interface.is_recording():
                log.info("Waiting for recording to end...")
                time.sleep(1)
        self.logger.closeDatasets()
        self.in_operation = False

    def is_stopped(self):
        self.quit = self.quit or self.logger.setup_status in ["stop", "exit"]
        if self.quit and self.logger.setup_status not in ["stop", "exit"]:
            self.logger.update_setup_info({"status": "stop"})
        if self.quit:
            self.in_operation = False
        return self.quit

    def _stim_init(self, stim_class: Stimulus, stims: Dict) -> Dict:
        # get stimulus class name
        stim_name = stim_class.name()
        if stim_name not in stims:
            stim_class.init(self)
            stims[stim_name] = stim_class
        return stims

    def get_keys_from_dict(self, data: Dict, keys: List) -> Dict:
        """Efficiently extract specific keys from a dictionary.

        Args:
            data (dict): The input dictionary.
            keys (list): The list of keys to extract.

        Returns:
            dict: A new dictionary with only the specified keys if they exist.

        """
        keys_set = set(keys)  # Convert list to set for O(1) lookup
        return {key: data[key] for key in keys_set.intersection(data)}

    def _get_task_classes(self, stim_class: Stimulus) -> Dict:
        exp_name = {"experiment_class": self.cond_tables[0]}
        beh_name = {
            "behavior_class": self.beh.cond_tables[0] if self.cond_tables else None
        }
        stim_name = {"stimulus_class": stim_class.name()}
        return {**exp_name, **beh_name, **stim_name}

    def make_conditions(
        self,
        stim_class: Stimulus,
        conditions: Dict[str, Any],
        stim_periods: List[str] = None,
    ):
        """Create experiment conditions by combining stimulus, behavior, and experiment."""
        log.debug("-------------- Make conditions --------------")
        self.stims = self._stim_init(stim_class, self.stims)
        used_keys = set()  # all the keys used from dictionary conditions

        # Handle stimulus conditions
        stim_conditions, stim_keys = self._process_stim_conditions(
            stim_class, conditions, stim_periods
        )
        used_keys.update(stim_keys)

        # Process behavior conditions
        beh_conditions, beh_keys = self._process_behavior_conditions(conditions)
        used_keys.update(beh_keys)

        # Process experiment conditions
        exp_conditions, exp_keys = self._process_experiment_conditions(
            self._get_task_classes(stim_class), conditions
        )
        used_keys.update(exp_keys)

        # Combine results and handle unused parameters
        partial_results = [exp_conditions, beh_conditions, stim_conditions]
        unused_conditions = self._handle_unused_parameters(conditions, used_keys)
        if unused_conditions:
            partial_results.append(unused_conditions)
        log.debug("-----------------------------------------------")
        return [
            {k: v for d in comb for k, v in d.items()}
            for comb in product(*partial_results)
        ]

    def _process_stim_conditions(
        self, stim_class: Stimulus, conditions: Dict, stim_periods: List
    ) -> Tuple[List, List]:
        """Process stimulus-specific conditions."""
        if stim_periods:
            period_conditions = {}
            for period in stim_periods:
                stim_dict = self.get_keys_from_dict(
                    conditions[period], get_parameters(stim_class).keys()
                )
                log.debug(
                    f"Stimulus period: {period} use default conditions:"
                    f"\n{get_parameters(stim_class).keys() - stim_dict.keys()}"
                )
                period_conditions[period] = factorize(stim_dict)
                period_conditions[period] = self.stims[
                    stim_class.name()
                ].make_conditions(period_conditions[period])
                for i, stim_condition in enumerate(period_conditions[period]):
                    log.debug(
                        f"Stimulus condition {i}:\n {format_params_print(stim_condition)}"
                    )
            stim_conditions = factorize(period_conditions)
            return stim_conditions, stim_periods

        stim_dict = self.get_keys_from_dict(
            conditions, get_parameters(stim_class).keys()
        )
        log.debug(
            f"Stimulus use default conditions:\n"
            f"{get_parameters(stim_class).keys() - stim_dict.keys()}"
        )
        stim_conditions = factorize(stim_dict)
        stim_conditions = self.stims[stim_class.name()].make_conditions(stim_conditions)
        for i, stim_condition in enumerate(stim_conditions):
            log.debug(f"Stimulus condition {i}:\n{format_params_print(stim_condition)}")
        return stim_conditions, stim_dict.keys()

    def _process_behavior_conditions(self, conditions: Dict) -> Tuple[List, List]:
        """Process behavior-related conditions."""
        beh_dict = self.get_keys_from_dict(conditions, get_parameters(self.beh).keys())
        log.debug(
            f"Behavior use default conditions:\n{get_parameters(self.beh).keys() - beh_dict.keys()}"
        )
        beh_conditions = factorize(beh_dict)
        beh_conditions = self.beh.make_conditions(beh_conditions)
        for i, beh_condition in enumerate(beh_conditions):
            log.debug(f"Behavior condition {i}:\n{format_params_print(beh_condition)}")
        return beh_conditions, beh_dict.keys()

    def _process_experiment_conditions(
        self, task_classes: List, conditions: Dict
    ) -> Tuple[List, list]:
        """Process experiment-wide conditions."""
        exp_dict = self.get_keys_from_dict(conditions, get_parameters(self).keys())
        exp_dict.update(task_classes)
        log.debug(
            f"Experiment use default conditions:\n{get_parameters(self).keys() - exp_dict.keys()}"
        )
        exp_conditions = factorize(exp_dict)

        for cond in exp_conditions:
            self.validate_condition(cond)
            cond.update({**self.default_key, **self.params, **cond})
        cond_tables = ["Condition." + table for table in self.cond_tables]
        conditions_list = self.log_conditions(
            exp_conditions, condition_tables=["Condition"] + cond_tables
        )
        for i, exp_condition in enumerate(exp_conditions):
            log.debug(
                f"Experiment condition {i}:\n{format_params_print(exp_condition)}"
            )
        return conditions_list, exp_dict.keys()

    def _handle_unused_parameters(self, conditions, used_keys):
        """Process any unused parameters."""
        unused_keys = set(conditions.keys()) - used_keys
        if unused_keys:
            log.warning(
                f"Keys: {unused_keys} are in condition but are not used from Experiment, Behavior or Stimulus"
            )
            unused_dict = self.get_keys_from_dict(conditions, unused_keys)
            return factorize(unused_dict)
        return None

    def validate_condition(self, condition: Dict) -> None:
        """Validate a condition dictionary against the required fields.

        Args:
            condition (Dict): The condition dictionary to validate.

        Raises:
            ValueError: If required fields are missing from the condition.

        """
        missing_fields = set(self.required_fields) - set(condition)
        if missing_fields:
            raise ValueError(f"Missing experiment required fields: {missing_fields}")

    def push_conditions(self, conditions: List):
        log.info(f"Number of conditions: {len(conditions)}")
        self.conditions = conditions
        resp_cond = (
            self.params["resp_cond"] if "resp_cond" in self.params else "response_port"
        )
        self.blocks = np.array([cond["difficulty"] for cond in self.conditions])
        if np.all([resp_cond in cond for cond in conditions]):
            self.choices = np.array(
                [make_hash([d[resp_cond], d["difficulty"]]) for d in conditions]
            )
            self.un_choices, un_idx = np.unique(self.choices, axis=0, return_index=True)
            self.un_blocks = self.blocks[un_idx]
        #  select random condition for first trial initialization
        self.cur_block = min(self.blocks)
        self.curr_cond = np.random.choice(
            [i for (i, v) in zip(self.conditions, self.blocks == self.cur_block) if v]
        )

    def prepare_trial(self):
        old_cond = self.curr_cond
        self._get_new_cond()

        if not self.curr_cond or self.logger.thread_end.is_set():
            self.quit = True
            return
        if (
            "stimulus_class" not in old_cond
            or self.curr_trial == 0
            or old_cond["stimulus_class"] != self.curr_cond["stimulus_class"]
        ):
            if "stimulus_class" in old_cond and self.curr_trial != 0:
                self.stim.exit()
            self.stim = self.stims[self.curr_cond["stimulus_class"]]
            log.debug("setting up stimulus")
            self.stim.setup()
            log.debug("stimuli is done")
        self.curr_trial += 1
        self.logger.update_trial_idx(self.curr_trial)
        self.trial_start = self.logger.logger_timer.elapsed_time()
        self.logger.log(
            "Trial",
            dict(cond_hash=self.curr_cond["cond_hash"], time=self.trial_start),
            priority=3,
        )
        if not self.in_operation:
            self.in_operation = True

    def name(self):
        return type(self).__name__

    def make_cond_hash(
        self, conditions: List[Dict], hash_field: str, schema: dj.schema, condition_tables: List
    ) -> List[Dict]:
        # get all fields from condition tables except hash
        fields_key = {
            key
            for ctable in condition_tables
            for key in self.logger.get_table_keys(schema, ctable)
        }
        fields_key.discard(hash_field)
        for condition in conditions:
            # find all dependant fields and generate hash
            key = {k: condition[k] for k in fields_key if k in condition}
            condition.update({hash_field: make_hash(key)})
        return conditions

    def log_conditions(
        self,
        conditions,
        condition_tables=None,
        schema="experiment",
        hash_field="cond_hash",
        priority=2,
    ) -> List[Dict]:
        """Log experimental conditions to specified tables with hashes tracking.

        Args:
            conditions (List): List of condition dictionaries to log
            condition_tables (List): List of table names to log to
            schema (db.shcema): Database schema name
            hash_field (str): Name of the hash field
            priority (int): for the insertion order of the logger

        Returns:
            List of processed conditions with added hashes

        """
        if not conditions:
            return []

        if condition_tables is None:
            condition_tables = ["Condition"]

        conditions = self.make_cond_hash(
            conditions, hash_field, schema, condition_tables
        )

        processed_conditions = conditions.copy()
        for condition in processed_conditions:
            # insert conditions fields to the correspond table
            for ctable in condition_tables:
                # Get table metadata
                fields = set(self.logger.get_table_keys(schema, ctable))
                primary_keys = set(
                    self.logger.get_table_keys(schema, ctable, key_type="primary")
                )
                core = [key for key in primary_keys if key != hash_field]

                # Validate condition has all required fields
                missing_keys = set(fields) - set(condition.keys())
                if missing_keys:
                    log.warning(f"Skipping {ctable}, Missing keys:{missing_keys}")
                    continue

                # check if there is a primary key which is not hash and it is iterable
                if core and hasattr(condition[core[0]], "__iter__"):
                    # TODO make a function for this and clarify it
                    # If any of the primary keys is iterable all the rest should be.
                    # The first element of the iterable will be matched with the first
                    # element of the rest of the keys
                    for idx, _ in enumerate(condition[core[0]]):
                        cond_key = {}
                        for k in fields:
                            if isinstance(condition[k], (int, float, str)):
                                cond_key[k] = condition[k]
                            else:
                                cond_key[k] = condition[k][idx]

                        self.logger.put(
                            table=ctable,
                            tuple=cond_key,
                            schema=schema,
                            priority=priority,
                        )

                else:
                    self.logger.put(
                        table=ctable, tuple=condition, schema=schema, priority=priority
                    )

                # Increment the priority for each subsequent table
                # to ensure they are inserted in the correct order
                priority += 1

        return processed_conditions

    def _anti_bias(self, choice_h, un_choices):
        choice_h = np.array(
            [make_hash(c) for c in choice_h[-self.curr_cond["bias_window"] :]]
        )
        if len(choice_h) < self.curr_cond["bias_window"]:
            choice_h = self.choices
        fixed_p = 1 - np.array([np.mean(choice_h == un) for un in un_choices])
        if sum(fixed_p) == 0:
            fixed_p = np.ones(np.shape(fixed_p))
        return np.random.choice(un_choices, 1, p=fixed_p / sum(fixed_p))

    def _get_performance(self):
        idx = np.logical_or(
            ~np.isnan(self.beh.reward_history), ~np.isnan(self.beh.punish_history)
        )  # select valid
        rew_h = np.asarray(self.beh.reward_history)
        rew_h = rew_h[idx]
        choice_h = np.int64(np.asarray(self.beh.choice_history)[idx])
        perf = np.nan
        window = self.curr_cond["staircase_window"]
        if self.curr_cond["metric"] == "accuracy":
            perf = np.nanmean(np.greater(rew_h[-window:], 0))
        elif self.curr_cond["metric"] == "dprime":
            y_true = [
                c if r > 0 else c % 2 + 1
                for (c, r) in zip(choice_h[-window:], rew_h[-window:])
            ]
            if len(np.unique(y_true)) > 1:
                perf = np.sqrt(2) * stats.norm.ppf(
                    roc_auc_score(y_true, np.array(choice_h[-window:]))
                )
            if self.logger.manual_run:
                log.debug(
                    f"perf: {perf if not np.isnan(perf) else 0} "
                    f"accuracy: {np.nanmean(np.greater(rew_h[-window:], 0)) or 0}"
                )
        else:
            log.error("Performance method not implemented!")
            self.quit = True
        choice_h = [[c, d] for c, d in zip(choice_h, np.asarray(self.block_h)[idx])]
        return perf, choice_h

    def _get_new_cond(self):
        """Get curr condition & create random block of all conditions"""
        if self.curr_cond["trial_selection"] == "fixed":
            self.curr_cond = [] if len(self.conditions) == 0 else self.conditions.pop(0)
        elif self.curr_cond["trial_selection"] == "block":
            if np.size(self.iter) == 0:
                self.iter = np.random.permutation(np.size(self.conditions))
            cond = self.conditions[self.iter[0]]
            self.iter = self.iter[1:]
            self.curr_cond = cond
        elif self.curr_cond["trial_selection"] == "random":
            self.curr_cond = np.random.choice(self.conditions)
        elif self.curr_cond["trial_selection"] == "staircase":
            perf, choice_h = self._get_performance()
            if np.size(self.beh.choice_history) and self.beh.choice_history[-1:][0] > 0:
                self.cur_block_sz += 1  # current block trial counter
            if self.cur_block_sz >= self.curr_cond["staircase_window"]:
                if perf >= self.curr_cond["stair_up"]:
                    self.cur_block = self.curr_cond["next_up"]
                    self.cur_block_sz = 0
                    self.logger.update_setup_info({"difficulty": self.cur_block})
                elif perf < self.curr_cond["stair_down"]:
                    self.cur_block = self.curr_cond["next_down"]
                    self.cur_block_sz = 0
                    self.logger.update_setup_info({"difficulty": self.cur_block})
            if self.curr_cond["antibias"]:
                anti_bias = self._anti_bias(
                    choice_h, self.un_choices[self.un_blocks == self.cur_block]
                )
                condition_idx = np.logical_and(
                    self.choices == anti_bias, self.blocks == self.cur_block
                )
            else:
                condition_idx = self.blocks == self.cur_block
            self.curr_cond = np.random.choice(
                [i for (i, v) in zip(self.conditions, condition_idx) if v]
            )
            self.block_h.append(self.cur_block)
        elif self.curr_cond["trial_selection"] == "biased":
            perf, choice_h = self._get_performance()
            condition_idx = self.choices == self._anti_bias(choice_h, self.un_choices)
            self.curr_cond = np.random.choice(
                [i for (i, v) in zip(self.conditions, condition_idx) if v]
            )
        else:
            log.error("Selection method not implemented!")
            self.quit = True

    @dataclass
    class Block:
        difficulty: int = field(compare=True, default=0, hash=True)
        stair_up: float = field(compare=False, default=0.7)
        stair_down: float = field(compare=False, default=0.55)
        next_up: int = field(compare=False, default=0)
        next_down: int = field(compare=False, default=0)
        staircase_window: int = field(compare=False, default=20)
        bias_window: int = field(compare=False, default=5)
        trial_selection: str = field(compare=False, default="fixed")
        metric: str = field(compare=False, default="accuracy")
        antibias: bool = field(compare=False, default=True)
        noresponse_intertrial: bool = field(compare=False, default=True)
        incremental_punishment: bool = field(compare=False, default=False)

        def dict(self):
            return self.__dict__


@experiment.schema
class Session(dj.Manual):
    definition = """
    # Session info
    animal_id                        : smallint UNSIGNED            # animal id
    session                          : smallint UNSIGNED            # session number
    ---
    user_name                        : varchar(16)      # user performing the experiment
    setup=null                       : varchar(256)     # computer id
    experiment_type                  : varchar(128)
    session_tmst=CURRENT_TIMESTAMP   : timestamp        # session timestamp
    """

    class Task(dj.Part):
        definition = """
        # Task info
        -> Session
        ---
        task_name        : varchar(256)                 # task filename
        task_file        : blob                         # task text file
        git_hash             : varchar(32)              # github hash
        """

    class Notes(dj.Part):
        definition = """
        # File session info
        -> Session
        timestamp=CURRENT_TIMESTAMP : timestamp         # timestamp
        ---
        note=null                   : varchar(2048)     # session notes
        """

    class Excluded(dj.Part):
        definition = """
        # Excluded sessions
        -> Session
        ---
        reason=null                 : varchar(2048)      # notes for exclusion
        timestamp=CURRENT_TIMESTAMP : timestamp
        """


@experiment.schema
class Condition(dj.Manual):
    definition = """
    # unique stimulus conditions
    cond_hash             : char(24)                 # unique condition hash
    ---
    stimulus_class        : varchar(128)
    behavior_class        : varchar(128)
    experiment_class      : varchar(128)
    """


@experiment.schema
class Trial(dj.Manual):
    definition = """
    # Trial information
    -> Session
    trial_idx            : smallint UNSIGNED       # unique trial index
    ---
    -> Condition
    time                 : int                     # start time from session start (ms)
    """

    class Aborted(dj.Part):
        definition = """
        # Aborted Trials
        -> Trial
        """

    class StateOnset(dj.Part):
        definition = """
        # Trial state timestamps
        -> Trial
        time			    : int 	            # time from session start (ms)
        state               : varchar(64)
        """


@experiment.schema
class Control(dj.Lookup):
    definition = """
    # Control table
    setup                       : varchar(256)                 # Setup name
    ---
    status="exit"               : enum('ready',
                                        'running',
                                        'stop',
                                        'sleeping',
                                        'exit',
                                        'offtime',
                                        'wakeup')
    animal_id=0                 : int                       # animal id
    task_idx=0                  : int                       # task identification number
    session=0                   : int
    trials=0                    : int
    total_liquid=0              : float
    state='none'                : varchar(255)
    difficulty=0                : smallint
    start_time='00:00:00'       : time
    stop_time='23:59:00'        : time
    last_ping=CURRENT_TIMESTAMP : timestamp
    notes=''                    : varchar(256)
    queue_size=0                : int
    ip=null                     : varchar(16)                  # setup IP address
    """


@experiment.schema
class Task(dj.Lookup):
    definition = """
    # Experiment parameters
    task_idx                    : int           # task identification number
    ---
    task                        : varchar(4095) # presented stimuli(array of dicts)
    description=""              : varchar(2048) # task description
    timestamp=CURRENT_TIMESTAMP : timestamp
    """


@mice.schema
class MouseWeight(dj.Manual):
    definition = """
    animal_id                       : int unsigned                 # id number
    timestamp=CURRENT_TIMESTAMP     : timestamp                    # timestamp of weight
    ---
    weight                          : double(5,2)                  # weight in grams
    """
