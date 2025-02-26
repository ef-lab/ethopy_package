"""A comprehensive logging system.

Managing experimental data, database connections, and data flow control in an
experimental setup. It includes functionality for establishing database connections,
managing logging sessions, handling data insertion, and synchronizing setup status.

Classes:
    Logger: Manages logging and data handling in an experimental setup.
    PrioritizedItem: Represents an item with a priority for logging purposes.

Functions:
    _set_connection: Establishes connection to the database and initializes global
        variables for virtual modules.
"""

import importlib
import inspect
import logging
import os
import platform
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as datafield
from datetime import datetime
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Dict, List, Optional

import datajoint as dj
import numpy as np
from pyfiglet import figlet_format

from ethopy import SCHEMATA, __version__, local_conf, plugin_manager
from ethopy.utils.helper_functions import (
    create_virtual_modules,
    get_code_version_info,
    get_environment_info,
    rgetattr,
)
from ethopy.utils.task import Task, resolve_task
from ethopy.utils.timer import Timer
from ethopy.utils.writer import Writer

log = logging.getLogger(__name__)


def _set_connection() -> None:
    """Establish connection to database.

    Establishes connections to database, creates virtual modules based on the provided
    schemata and assigns them to global variables. It also initializes the `public_conn`
    global variable.

    Globals:
        experiment: The virtual module for experiment.
        stimulus: The virtual module for stimulus.
        behavior: The virtual module for behavior.
        interface: The virtual module for interface.
        recording: The virtual module for recording.
        mice: The virtual module for mice.
        public_conn: The connection object for public access.

    Returns:
        None

    """
    global experiment, stimulus, behavior, interface, recording, mice, public_conn
    virtual_modules, public_conn = create_virtual_modules(SCHEMATA)
    experiment = virtual_modules["experiment"]
    stimulus = virtual_modules["stimulus"]
    behavior = virtual_modules["behavior"]
    recording = virtual_modules["recording"]
    mice = virtual_modules["mice"]
    interface = virtual_modules["interface"]


_set_connection()


class Logger:
    """Logger class for managing logging and data handling in an experimental setup.

    This class is designed to handle the logging of experimental data, manage database
    connections, and control the flow of data from source to target locations. It
    supports both manual and automatic running modes of a session, integrates with a
    Python logging setup, and manages threads for data insertion and setup status
    updates.

    Attributes:
        setup (str): The hostname of the machine running the experiment.
        is_pi (bool): Flag indicating if the current machine is a Raspberry Pi.
        task_idx (int): Task index
        task_path (str): Path to the task file.
        manual_run (bool): Flag indicating if the experiment is run manually.
        setup_status (str): Current status of the setup (e.g. 'running', 'ready').
        private_conn (Connection): Connection for internal database communication.
        writer (Writer): Writer class instance for handling data writing.
        rec_fliptimes (bool): Flag indicating if flip times should be recorded.
        trial_key (dict): Dictionary containing identifiers for the current trial.
        setup_info (dict): Dictionary containing setup information.
        datasets (dict): Dictionary containing datasets.
        lock (bool): Lock flag for thread synchronization.
        queue (PriorityQueue): Queue for managing data insertion order.
        ping_timer (Timer): Timer for managing pings.
        logger_timer (Timer): Timer for managing logging intervals.
        total_reward (int): Total reward accumulated.
        curr_state (str): Current state of the logger.
        thread_exception (Exception): Exception caught in threads, if any.
        source_path (str): Path where data are saved.
        target_path (str): Path where data will be moved after the session ends.
        thread_end (Event): Event to signal thread termination.
        thread_lock (Lock): Lock for thread synchronization.
        inserter_thread (Thread): Thread for inserting data into the database.
        getter_thread (Thread): Thread for periodically updating setup status.

    Methods:
        __init__(task=False): Initializes the Logger instance.
        _check_if_raspberry_pi(): Checks if the current machine is a Raspberry Pi.
        _inserter(): Inserts data into the database.
        _log_setup_info(setup, status): Logs setup information.
        _get_setup_status(): Get setup status.

    """

    def __init__(self, task: bool = False) -> None:
        """Initialize the Logger."""
        self.setup = socket.gethostname()
        self.is_pi = self._check_if_raspberry_pi()

        self.task = task or Task(path=None, id=None)
        self.manual_run = bool(self.task.path or self.task.id)
        self.setup_status = "running" if self.manual_run else "ready"

        # separate connection for internal communication
        self._schemata, self.private_conn = create_virtual_modules(
            SCHEMATA, create_tables=False, create_schema=False
        )

        self.writer = Writer
        self.rec_fliptimes = True
        self.trial_key = {"animal_id": 0, "session": 1, "trial_idx": 0}
        self.setup_info = {}
        self.datasets = {}
        self.lock = False
        self.queue = PriorityQueue()
        self.ping_timer = Timer()
        self.logger_timer = Timer()
        self.total_reward = 0
        self.curr_state = ""
        self.thread_exception = None
        self.update_status = threading.Event()
        self.update_status.clear()

        # source path is the local path that data are saved
        self.source_path = local_conf.get("source_path")
        # target path is the path that data will be moved after the session ends
        self.target_path = local_conf.get("target_path")

        # inserter_thread read the queue and insert the data in the database
        self.thread_end, self.thread_lock = threading.Event(), threading.Lock()
        self.inserter_thread = threading.Thread(target=self._inserter)
        self.inserter_thread.start()

        # _log_setup_info needs to run after the inserter_thread is started
        self._log_setup_info(self.setup, self.setup_status)

        # before starting the getter thread we need to _log_setup_info
        self.update_thread = threading.Thread(target=self._sync_control_table)
        self.update_thread.start()
        self.logger_timer.start()

    @property
    def task_path(self) -> Optional[Path]:
        """Get the task path."""
        return self.task.path

    def get_task(self) -> bool:
        """Get the task configuration.

        Returns:
            bool: True if task is available and valid

        """
        if not self.manual_run:
            self.task = resolve_task(task_id=self.get_setup_info("task_idx"))

        return self.task_path is not None

    def _check_if_raspberry_pi(self) -> bool:
        system = platform.uname()
        return (
            system.machine.startswith("arm") or system.machine == "aarch64"
            if system.system == "Linux"
            else False
        )

    def setup_schema(self, extra_schema: Dict[str, Any]) -> None:
        """Set up additional schema.

        Args:
            extra_schema (Dict[str, Any]): The additional schema to set up.

        """
        for schema, value in extra_schema.items():
            globals()[schema] = dj.create_virtual_module(
                schema, value, create_tables=True, create_schema=True
            )
            self._schemata.update(
                {
                    schema: dj.create_virtual_module(
                        schema, value, connection=self.private_conn
                    )
                }
            )

    def put(self, **kwargs: Dict[str, Any]) -> None:
        """Put an item in the queue.

        This method creates a `PrioritizedItem` from the given keyword arguments and
        puts it into the queue. After putting an item in the queue, it checks the
        'block' attribute of the item. If 'block' is False, it marks the item as
        processed by calling `task_done()`. This is useful in scenarios where items are
        processed asynchronously, and the queue needs to be notified that a task is
        complete. If 'block' is True, it waits for all items in the queue to be
        processed by calling `join()`.

        Args:
            **kwargs (Any): The keyword arguments used to create a `PrioritizedItem` and
                put it in the queue.

        """
        item = PrioritizedItem(**kwargs)
        self.queue.put(item)
        if not item.block:
            self.queue.task_done()
        else:
            self.queue.join()

    def _insert_item(self, item, table) -> None:
        """Insert an item into the specified table.

        Args:
            item: The item to be inserted.
            table: The table to insert the item into.

        Returns:
            None

        """
        table.insert1(
            item.tuple,
            ignore_extra_fields=item.ignore_extra_fields,
            skip_duplicates=False if item.replace else True,
            replace=item.replace,
        )

    def _validate_item(self, item, table) -> None:
        """Validate an item against a table."""
        if item.validate:  # validate tuple exists in database
            key = {k: v for (k, v) in item.tuple.items() if k in table.primary_key}
            if "status" in item.tuple.keys():
                key["status"] = item.tuple["status"]
            while not len(table & key) > 0:
                time.sleep(0.5)

    def _handle_insert_error(self, item, table, exception, queue) -> None:
        """Handle databse insert errors.

        Handles an error by logging the error message, set the item.error=True, increase
        priority and add the item again in the queue for re-trying to insert later.

        Args:
            item : Description of parameter `item`.
            table : Description of parameter `table`.
            exception (Exception): The exception that was raised.
            thread_end : Description of parameter `thread_end`.
            queue : Description of parameter `queue`.

        """
        log.warning(
            "Failed to insert:\n%s in %s\n With error:%s\nWill retry later",
            item.tuple,
            table,
            exception,
            exc_info=True,
        )
        item.error = True
        item.priority = item.priority + 2
        queue.put(item)

    @contextmanager
    def acquire_lock(self, lock):  # noqa: ANN201
        """Acquire a lock, yield control, and release the lock.

        This context manager ensures that the given lock is acquired before
        entering the block of code and released after exiting the block, even
        if an exception is raised within the block.

        Args:
            lock (threading.Lock): The lock object to acquire and release.

        """
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _inserter(self) -> None:
        """Insert continuously items from the queue into their respective tables.

        It runs in a loop until the thread_end event is set. In each iteration, it
        checks if the queue is empty. If it is, it sleeps for 0.5 seconds and then
        continues to the next iteration.
        If the queue is not empty, it gets an item from the queue, acquires the thread
        lock, and tries to insert the item into it's table.
        If an error occurs during the insertion, it handles the error. After the
        insertion, it releases the thread lock. If the item was marked to block, it
        marks the task as done.

        Returns:
            None

        """
        while not self.thread_end.is_set():
            if self.queue.empty():
                time.sleep(0.5)
                continue
            item = self.queue.get()
            table = rgetattr(self._schemata[item.schema], item.table)
            with self.acquire_lock(self.thread_lock):
                try:
                    self._insert_item(item, table)
                    self._validate_item(item, table)
                except Exception as insert_error:
                    if item.error:
                        self.thread_end.set()
                        log.error(
                            "Second time failed to insert:\n %s in %s With error:\n %s",
                            item.tuple,
                            table,
                            insert_error,
                            exc_info=True,
                        )
                        self.thread_exception = insert_error
                        break
                    self._handle_insert_error(item, table, insert_error, self.queue)
            if item.block:
                self.queue.task_done()

    def _sync_control_table(self, update_period: float = 5000) -> None:
        """Synchronize the Control table.

        Synchronize the Control table by continuously fetching the setup status
        from the experiment schema and periodically updating the setup info.

        Runs in a loop until the thread_end event is set.

        Args:
            update_period (float): Time in milliseconds between Control table updates.

        """
        while not self.thread_end.is_set():
            with self.thread_lock:
                if self.update_status.is_set():
                    continue
                try:
                    self._fetch_setup_info()
                    self._update_setup_info(update_period)
                except Exception as error:
                    log.exception("Error during Control table sync: %s", error)
                    self.thread_exception = error

            time.sleep(1)  # Cycle once a second

    def _fetch_setup_info(self) -> None:
        self.setup_info = (
            self._schemata["experiment"].Control() & {"setup": self.setup}
        ).fetch1()
        self.setup_status = self.setup_info["status"]

    def _update_setup_info(self, update_period: float) -> None:
        """Update the setup information if the elapsed time exceeds the update period.

        This method checks if the elapsed time since the last ping exceeds the given
        update period. If it does, it resets the ping timer and updates the setup
        information with the current state, queue size, trial index, total liquid
        reward, and the current timestamp. The updated information is then stored
        in the "Control" table with a priority of 1.
        """
        if self.ping_timer.elapsed_time() >= update_period:
            self.ping_timer.start()
            info = {
                "last_ping": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "queue_size": self.queue.qsize(),
                "trials": self.trial_key["trial_idx"],
                "total_liquid": self.total_reward,
                "state": self.curr_state,
            }
            self.setup_info.update(info)
            self.put(table="Control", tuple=self.setup_info, replace=True, priority=1)

    def log(
        self,
        table: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ) -> float:
        """Log the given data into the specified table in the experiment database.

        It first gets the elapsed time from the logger timer and adds it to the data
        dictionary. It then puts the data into the specified table.

        Args:
            table (str): The name of the table in the experiment database.
            data (dict, optional): The data to be logged. Defaults to an empty
                dictionary.
            **kwargs: Additional keyword arguments to be passed to the put method.

        Returns:
            float: The elapsed time from the logger timer.

        """
        tmst = self.logger_timer.elapsed_time()
        data = data or {}  # if data is None or False use an empty dictionary
        self.put(table=table, tuple={**self.trial_key, "time": tmst, **data}, **kwargs)
        if table == "Trial.StateOnset":
            log.info("State: %s", data["state"])
        return tmst

    def _log_setup_info(self, setup: str, setup_status: str = "running") -> None:
        """Log setup information into the Control table in the experiment database.

        It first fetches the control information for the current setup. If no control
        information is found, it creates a new dictionary with the setup information.
        It then adds the IP and status information to the key.

        The method finally puts the key into the Control table, replacing any existing
        entry. Because it blocks the queue until the operation is complete it needs the
        inserter_thread to be running.

        Args:
            setup (str): The setup name.
            setup_status (str): The current status fo the setup. Defaults to running.

        Returns:
            None

        """
        rel = experiment.Control() & dict(setup=setup)
        key = rel.fetch1() if np.size(rel.fetch()) else dict(setup=setup)
        key = {**key, "ip": self.get_ip(), "status": setup_status}
        self.put(
            table="Control",
            tuple=key,
            replace=True,
            priority=1,
            block=True,
            validate=True,
        )

    def _get_last_session(self) -> int:
        """Fetch last session for a given animal_id from the experiment.

        It first fetches all sessions for the given animal_id. If no sessions are found,
        it returns 0.
        If sessions are found, it returns the maximum session number, which corresponds
        to the last session.

        Returns:
            int: The last session number or 0 if no sessions are found.

        """
        last_sessions = (
            experiment.Session() & dict(animal_id=self.get_setup_info("animal_id"))
        ).fetch("session")
        return 0 if np.size(last_sessions) == 0 else np.max(last_sessions)

    def log_session(
        self,
        session_params: Dict[str, Any],
        experiment_type: str,
        log_task: bool = False,
    ) -> None:
        """Log session with the given parameters and optionally log the task.

        Args:
            session_params (Dict[str, Any]): Parameters for the session.
            experiment_type (str): current experiment running in session.
            log_task (bool): Whether to log the task information.

        """
        # Initializes session parameters and logs the session start.
        self._init_session_params(
            session_params.get("user_name", "bot"), experiment_type
        )

        # Save the task file, name and the git_hash in the database.
        if log_task:
            self._log_task_details()

        # update the configuration tables
        self.log_session_configs(session_params["setup_conf_idx"])

        #  Init the informations(e.g. trial_id=0, session) in control table
        self._init_control_table(session_params)

        self.logger_timer.start()  # Start session time

    def _init_session_params(self, user_name: str, experiment_type: str) -> None:
        """Initialize session parameters and log the session start.

        This method initializes the session parameters by setting the total reward to
        zero and creating a trial key with the animal ID, trial index set to zero, and
        the session number incremented by one from the last session. It logs the trial
        key and creates a session key by merging the trial key with the provided session
        parameters, setup information, and a default or provided user name. The session
        key is then logged and stored in the database.

        Args:
            user_name (Dict[str, Any]): A string defininng user.
            experiment_type: str: name of the expertiment.

        """
        self.total_reward = 0
        self.trial_key = {
            "animal_id": self.get_setup_info("animal_id"),
            "trial_idx": 0,
            "session": self._get_last_session() + 1,
        }

        session_key = {
            "animal_id": self.get_setup_info("animal_id"),
            "session": self._get_last_session() + 1,
            "user_name": user_name,
            "setup": self.setup,
            "experiment_type": experiment_type,
        }

        # Convert np.int64 values to native Python int
        session_key_cleaned = {
            k: int(v) if isinstance(v, np.integer) else v
            for k, v in session_key.items()
        }

        log.info("\n%s", figlet_format("EthoPy"))
        log.info(
            "\n%s%s%s\n%s\n%s",
            "-" * 22,
            " Basic Session informations ",
            "-" * 22,
            "\n".join(f"{k}: {v}" for k, v in session_key_cleaned.items()),
            "-" * 72,
        )

        # Logs the new session id to the database
        self.put(
            table="Session", tuple=session_key, priority=1, validate=True, block=True
        )

    @staticmethod
    def get_inner_classes_list(outer_class: Any) -> List[str]:
        """Retrieve a list of names of all inner classes defined within an outer class.

        Args:
            outer_class: The class object of the outer class containing the inner
                classes.

        Returns:
            A list of strings, each representing the fully qualified name of an inner
                class defined within the outer class.

        """
        outer_class_dict_values = outer_class.__dict__.values()
        inner_classes = [
            value for value in outer_class_dict_values if isinstance(value, type)
        ]
        return [outer_class.__name__ + "." + cls.__name__ for cls in inner_classes]

    def log_session_configs(self, setup_conf_idx: int) -> None:
        """Log parameter of a session into the appropriate schema tables.

        This method performs several key operations to ensure that the configuration of
        a session, including behavior and stimulus settings, is accurately logged into
        the database. It involves the following steps:
        1. Identifies the relevant modules (e.g., ethopy.core.interface) that contain
        Configuration classes.
        2. Derives schema names from these modules, assuming the schema name matches the
        class name in lowercase.
        3. Logs the session and animal_id into the Configuration tables of the
        identified schemas.
        4. Creates a dictionary mapping each schema to its respective Configuration
        class's inner classes.
        5. Calls a helper method to log the configuration of sub-tables for each schema.
        """
        # modules that have a Configuration classes
        _modules = ["ethopy.core.interface"]
        # consider that the module have the same name as the schema but in lower case
        # (e.g for class Behaviour the schema is the behavior)
        _schemas = [_module.split(".")[2].lower() for _module in _modules]

        # Logs the session and animal_id in configuration tables of behavior/stimulus.
        for schema in _schemas:
            self.put(
                table="Configuration",
                tuple=self.trial_key,
                schema=schema,
                priority=2,
                validate=True,
                block=True,
            )

        # create a dict with the configuration as key and the subclasses as values
        conf_table_schema = {}
        for _schema, _module in zip(_schemas, _modules, strict=True):
            conf = importlib.import_module(_module).Configuration
            # Find the inner classes of the class Configuration
            conf_table_schema[_schema] = self.get_inner_classes_list(conf)

        # update the sub tables of Configuration table
        for schema, config_tables in conf_table_schema.items():
            self._log_sub_tables_config(setup_conf_idx, config_tables, schema)

    def _log_sub_tables_config(
        self, setup_conf_idx: int, config_tables: List[str], schema: str
    ) -> None:
        """Log conifguration data in the respective tables.

        This method iterates over a list of configuration tables, retrieves the
        configuration data for each table based on the provided parameters, and then
        logs this data into the respective table within the given schema.

        Args:
            setup_conf_idx (int): index indication the setup configuration
            config_tables (list): The part table to be recorded (e.g., Port, Screen).
            schema (str): The schema for the configuration.

        """
        for config_table in config_tables:
            configuration_data = (
                getattr(interface.SetupConfiguration, config_table.split(".")[1])
                & {"setup_conf_idx": setup_conf_idx}
            ).fetch(as_dict=True)
            # put the configuration data in the configuration table
            # it can be a list of configurations (e.g have two ports with different ids)
            for conf in configuration_data:
                self.put(
                    table=config_table,
                    tuple={**conf, **self.trial_key},
                    schema=schema,
                )

    def _init_control_table(self, params: Dict[str, Any]) -> None:
        """Set the control table informations for the setup.

        This method sets various parameters related to the session setup, including
        session ID, number of trials, total liquid, difficulty level, and state. It also
        optionally sets start and stop times if they are provided in the `params`
        argument.

        The start and stop times are expected to be in "%H:%M:%S" format. If they are
        provided, this method calculates the time delta from "00:00:00" for each and
        updates the setup information accordingly.

        Args:
            params (Dict[str, Any]): A dictionary containing parameters for the session
            setup. This may include 'start_time' and 'stop_time' among other setup
            parameters.

        """
        key = {
            "session": self.trial_key["session"],
            "trials": 0,
            "total_liquid": 0,
            "difficulty": 1,
            "state": "",
        }
        #  TODO if task is the path of the config there is no update in Control table
        if self.task.id and isinstance(self.task.id, int):
            key["task_idx"] = self.task.id

        # if in the start_time is defined in the configuration use this
        # otherwise use the Control table
        if "start_time" in params:

            def _tdelta(t: str) -> datetime:
                return datetime.strptime(t, "%H:%M:%S") - datetime.strptime(
                    "00:00:00", "%H:%M:%S"
                )

            key.update(
                {
                    "start_time": str(_tdelta(params["start_time"])),
                    "stop_time": str(_tdelta(params["stop_time"])),
                }
            )

        self.update_setup_info({**key, "status": self.setup_info["status"]})

    def update_setup_info(
        self, info: Dict[str, Any], key: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the setup information in Control table with the provided info and key.

        It first fetches the existing setup information from the experiment's Control
        table, then updates it with the provided info. If 'status' is in the provided
        info, it blocks and validates the update operation.

        Args:
            info (dict): The information to update the setup with.
            key (dict, optional): Additional keys to fetch the setup information with.
                Defaults to None.

        Side Effects:
            Updates the setup_info attribute with the new setup information.
            Updates the setup_status attribute with the new status.

        """
        if self.thread_exception:
            self.thread_exception = None
            raise Exception("Thread exception occurred: %s", self.thread_exception)
        if key is None:
            key = dict()

        if not public_conn.is_connected:
            _set_connection()

        block = True if "status" in info else False
        if block:
            self.update_status.set()
            caller = inspect.stack()[1]
            caller_info = (
                f"Function called by {caller.function} "
                f"in {caller.filename} at line {caller.lineno}"
            )
            log.debug("Update status is set %s\n%s", info["status"], caller_info)

        self.setup_info = {
            **(experiment.Control() & {"setup": self.setup, **key}).fetch1(),
            **info,
        }

        char_len = 255
        if "notes" in info and len(info["notes"]) > char_len:
            info["notes"] = info["notes"][:char_len]

        self.put(
            table="Control",
            tuple=self.setup_info,
            replace=True,
            priority=1,
            block=block,
            validate=block,
        )
        self.setup_status = self.setup_info["status"]
        self.update_status.clear()

    def _log_task_details(self) -> None:
        """Save the task file, name and git_hash in the database."""
        version_info = get_code_version_info()
        self.put(table="Session.Version", tuple={**self.trial_key, **version_info})
        log.debug(f"Code version: {version_info}")

        env_info = get_environment_info()
        self.put(table="Session.Enviroment", tuple={**self.trial_key, **env_info})
        log.debug(f"Enviroment info: {env_info}")

        for path in plugin_manager.plugin_paths:
            plugin_version_info = get_code_version_info(path)
            if plugin_version_info["source_type"] is None:
                log.warning(f"Plugin {path} is not a git repository")
            else:
                log.debug(f"Plugin code version: {plugin_version_info}")
            self.put(
                table="Session.Version", tuple={**self.trial_key, **plugin_version_info}
            )

        self.put(
            table="Session.Task",
            tuple={
                **self.trial_key,
                "task_name": self.task_path,
                "task_file": np.fromfile(self.task_path, dtype=np.int8),
                "git_hash": version_info["version"],
            },
        )

    def get_setup_info(self, field: str) -> np.int64:
        """Retrieve specific setup information from an experiment control table.

        Args:
            field (str): The name of the field to fetch from the experiment control
                setup.

        Returns:
            The value of the specified field from the experiment control setup.

        """
        return (experiment.Control() & dict(setup=self.setup)).fetch1(field)

    def get(
        self,
        schema: str = "experiment",
        table: str = "Control",
        fields: Optional[List] = None,
        key: Optional[Dict] = None,
        **kwargs: Dict[str, Any],
    ) -> np.ndarray:
        """Fetch data from a specified table in a schema.

        Args:
            schema (str): The schema to fetch data from. Defaults to "experiment".
            table (str): The table to fetch data from. Defaults to "Control".
            fields (dict): The fields to fetch. Defaults to "".
            key (dict): The key used to fetch data. Defaults to an empty dict.
            **kwargs: Additional keyword arguments.

        Returns:
            (numpy.ndarray): The fetched data.

        """
        if key is None:
            key = dict()
        if fields is None:
            fields = []
        table = rgetattr(eval(schema), table)  # noqa: S307
        return (table() & key).fetch(*fields, **kwargs)

    def get_table_keys(
        self,
        schema: str = "experiment",
        table: str = "Control",
        key: Optional[Dict] = None,
        key_type: Optional[str] = None,
    ) -> List[str]:
        """Retrieve the primary key of a specified table within a given schema.

        Args:
            schema (str): The schema name where the table is located. Default is
                'experiment'.
            table (str): The table name from which to retrieve the keys. Default is
                'Control'.
            key (dict): A dict with the key to filter the table. Default is an empty
                dictionary.
            key_type (str): type of keys to return from the table

        Returns:
            list: The primary key of the specified table.

        """
        if key is None:
            key = []
        table = rgetattr(globals()[schema], table)  # noqa: S307
        if key_type == "primary":
            return (table() & key).primary_key
        return (table() & key).heading.names

    def update_trial_idx(self, trial_idx: int) -> None:
        """Update trial index.

        Updates the trial index in the trial_key dictionary and check if there is any
        exception in the threads.

        Args:
            trial_idx (int): The new trial index to be updated.

        """
        self.trial_key["trial_idx"] = trial_idx
        log.info("\nTrial idx: %s", self.trial_key["trial_idx"])
        if self.thread_exception:
            self.thread_exception = None
            raise Exception("Thread exception occurred: %s", self.thread_exception)

    def cleanup(self) -> None:
        """Wait for the logging queue to be empty and signals the logging thread to end.

        This method checks if the logging queue is empty, and if not, it waits until it
        becomes empty. Once the queue is empty, it sets the thread_end event to signal
        the logging thread to terminate.
        """
        while not self.queue.empty() and not self.thread_end.is_set():
            log.info("Waiting for empty queue... qsize: %d", self.queue.qsize())
            time.sleep(1)
        self.thread_end.set()

        if not self.queue.empty():
            log.warning("Clean up finished but queue size is: %d", self.queue.qsize())

    def createDataset(
        self,
        dataset_name: str,
        dataset_type: type,
        filename: Optional[str] = None,
        db_log: Optional[bool] = True,
    ) -> Dict:
        """Create a dataset and return the dataset object.

        Args:
            dataset_name (str): The name of the dataset.
            dataset_type (type): The datatype of the dataset.
            filename (str, optional): The filename for the h5 file. If not provided,
                a default filename will be generated based on the dataset name,
                animal ID, session, and current timestamp.
            db_log (bool, optional): If True call the log_recording

        Returns:
            Tuple[str, Any]: A tuple containing the filename and the dataset object.

        """
        folder = (
            f"Recordings/{self.trial_key['animal_id']}_{self.trial_key['session']}/"
        )
        path = self.source_path + folder
        if not os.path.isdir(path):
            os.makedirs(path)  # create path if necessary

        if not os.path.isdir(self.target_path):
            log.info("No target directory set! Autocopying will not work.")
            target_path = False
        else:
            target_path = self.target_path + folder
            if not os.path.isdir(target_path):
                os.makedirs(target_path)

        # Generate filename if not provided
        if filename is None:
            filename = (
                f"{dataset_name}_{self.trial_key['animal_id']}_"
                f"{self.trial_key['session']}_"
                f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.h5"
            )
        if filename not in self.datasets:
            # create h5 file if not exists
            self.datasets[filename] = self.writer(path + filename, target_path)

        # create new dataset in the h5 files
        self.datasets[filename].createDataset(
            dataset_name, shape=(1,), dtype=dataset_type
        )

        if db_log:
            rec_key = dict(
                rec_aim=dataset_name,
                software="EthoPy",
                version=__version__,
                filename=filename,
                source_path=path,
                target_path=target_path,
            )
            self.log_recording(rec_key)

        return self.datasets[filename]

    def log_recording(self, rec_key: Dict) -> None:
        """Log a new recording entry with an incremented recording index.

        This method retrieves the current recordings associated with the trial,
        calculates the next recording index (rec_idx) by finding the maximum
        recording index and adding one, and logs the new recording entry with
        the provided recording key (rec_key) and the calculated recording index.

        Args:
            rec_key (dict): A dictionary containing the key information for the
                recording entry.

        The method assumes the existence of a `get` method to retrieve existing
        recordings and a `log` method to log the new recording entry.

        """
        recs = self.get(
            schema="recording",
            table="Recording",
            key=self.trial_key,
            fields=["rec_idx"],
        )
        rec_idx = 1 if not recs else max(recs) + 1
        self.log("Recording", data={**rec_key, "rec_idx": rec_idx}, schema="recording")

    def closeDatasets(self) -> None:
        """Close all datasets managed by this instance.

        Iterates through the datasets dictionary, calling the `exit` method on each
        dataset object to properly close them.
        """
        for _, dataset in self.datasets.items():
            dataset.exit()

    @staticmethod
    def get_ip() -> str:
        """Retrieve the local IP address of the machine.

        Attempts to establish a dummy connection to a public DNS server (8.8.8.8) to
        determine the local network IP address of the machine. If the connection fails,
        defaults to localhost (127.0.0.1).

        Returns:
            str: The local IP address.

        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip


@dataclass(order=True)
class PrioritizedItem:
    """A class used to represent an item with a priority for logging purposes."""

    table: str = datafield(compare=False)
    tuple: Any = datafield(compare=False)
    field: str = datafield(compare=False, default="")
    value: Any = datafield(compare=False, default="")
    schema: str = datafield(compare=False, default="experiment")
    replace: bool = datafield(compare=False, default=False)
    block: bool = datafield(compare=False, default=False)
    validate: bool = datafield(compare=False, default=False)
    priority: int = datafield(default=50)
    error: bool = datafield(compare=False, default=False)
    ignore_extra_fields: bool = datafield(compare=False, default=True)
