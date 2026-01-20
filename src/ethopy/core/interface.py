"""Core interface module for EthoPy.

This module provides the base interface for hardware interaction and configuration of
hardware based on setup index.
"""

import logging
from dataclasses import dataclass, fields
from dataclasses import field as datafield
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

import datajoint as dj
import numpy as np

from ethopy.core.logger import (  # pylint: disable=W0611, # noqa: F401
    experiment,
    interface,
)
from ethopy.utils.helper_functions import reverse_lookup
from ethopy.utils.timer import Timer

log = logging.getLogger()


class Interface:
    """Base interface class for hardware interaction in experimental setups.

    This class manages hardware interfaces including ports, cameras, and other
    peripherals.
    It provides methods for stimulus delivery, reward management, and hardware control.

    Attributes:
        port (int): Current active port
        resp_tmst (int): Response timestamp
        ready_dur (int): Ready duration
        activity_tmst (int): Activity timestamp
        ready_tmst (int): Ready state timestamp
        pulse_rew (Dict[int, Dict]): Reward pulse settings by port
        ports (List[Port]): List of available ports
        response (List): List of responses
        duration (Dict[int, float]): Duration settings by port
        ready (bool): Ready state flag
        timer_ready (Timer): Timer for ready state
        weight_per_pulse (Dict[int, float]): Calibration weights per pulse
        pulse_dur (Dict[int, float]): Pulse durations by port
        channels (Dict[str, Any]): Channel mappings
        position_dur (int): Position duration

    """

    def __init__(
        self,
        exp: Optional = None,
        beh: Optional = None,
        callbacks: bool = True,
    ) -> None:
        """Initialize the interface with experiment and behavior objects.

        Args:
            exp: Experiment object containing parameters and logger
            beh: Behavior object for tracking responses
            callbacks: Whether to enable callback functions

        """
        # Initialize basic attributes
        self.callbacks = callbacks
        self.beh = beh
        self.exp = exp
        self.logger = exp.logger if exp else None
        self.position = Port()
        self.position_tmst: int = 0
        self.camera = None
        self.ports: List[Port] = []
        self.pulse_rew: Dict[int, Dict] = {}
        self.duration: Dict[int, float] = {}
        self.weight_per_pulse: Dict[int, float] = {}
        self.pulse_dur: Dict[int, float] = {}

        # Initialize timing variables
        self.port: int = 0
        self.resp_tmst: int = 0
        self.ready_dur: int = 0
        self.activity_tmst: int = 0
        self.ready_tmst: int = 0
        self.position_dur: int = 0

        # Initialize state variables
        self.ready: bool = False
        self.timer_ready = Timer()
        self.response: List[Any] = []

        # Remote camera management
        self.remote_camera_configs: List[Dict] = []
        self.remote_cameras: List = []

        if exp and hasattr(exp, "session_params"):
            self._initialize_hardware()

    def _initialize_hardware(self) -> None:
        """Initialize hardware components based on setup configuration.

        This method sets up ports and camera if configured in the experiment parameters.
        """
        # Initialize ports
        port_configs = self.logger.get(
            schema="interface",
            table="SetupConfiguration.Port",
            key=f"setup_conf_idx={self.exp.setup_conf_idx}",
            as_dict=True,
        )

        for port_config in port_configs:
            self.ports.append(Port(**port_config))

        self.ports = np.array(self.ports)
        self.proximity_ports = np.array(
            [p.port for p in self.ports if p.type == "Proximity"]
        )
        self.rew_ports = np.array([p.port for p in self.ports if p.reward])

    def register_remote_camera(
        self,
        remote_host: str,
        setup_conf_idx: int,
        node_id: str = None,
        command_port: int = None,
        response_port: int = None,
        **kwargs
    ) -> "Interface":
        """Register a remote camera arguments.

        Args:
            remote_host: Hostname or IP of remote machine
            setup_conf_idx: Setup configuration index for remote camera
            node_id: Unique identifier for this camera node
            command_port: ZMQ port for commands (auto-assigned if None)
            response_port: ZMQ port for responses (auto-assigned if None)
            **kwargs: Additional camera configuration

        Returns:
            Self for method chaining
        """
        camera_config = {
            "remote_host": remote_host,
            "setup_conf_idx": setup_conf_idx,
            "node_id": node_id,
            "command_port": command_port,
            "response_port": response_port,
            **kwargs
        }
        self.remote_camera_configs.append(camera_config)
        log.info(f"Registered remote camera: {node_id or 'unnamed'} at {remote_host}")
        return self

    def setup_cameras(self) -> None:
        """Setup all cameras (both local from database and registered remote cameras).

        This method should be called after session metadata is available.
        It handles:
        - Local camera from database configuration
        - All registered remote cameras (InterfaceProxy instances)
        """
        # Setup local camera from database
        self._setup_local_camera()

        # Setup registered remote cameras
        self._setup_remote_cameras()

    def _setup_remote_cameras(self) -> None:
        """Setup all registered remote cameras with session metadata.

        Creates InterfaceProxy for each registered remote camera and initializes
        with correct session metadata. Called after session metadata is available.

        Can be called multiple times for different sessions - clears old cameras first.
        """
        if not self.remote_camera_configs:
            log.debug("No remote cameras registered")
            return

        # Clear old camera proxies from previous session (if any)
        # The InterfaceProxy objects stay in memory but are no longer referenced
        # Their NetworkServers remain alive for reuse
        if self.remote_cameras:
            log.info(f"Clearing {len(self.remote_cameras)} camera proxies from previous session")
            self.remote_cameras.clear()

        log.info(f"Setting up {len(self.remote_camera_configs)} remote cameras...")

        from ethopy.utils.interface_proxy import InterfaceProxy

        for i, cam_config in enumerate(self.remote_camera_configs):
            # IMPORTANT: Port allocation MUST be consistent across sessions!
            # Remote nodes store these ports and reconnect to them.
            # If we change ports between sessions, remote won't find us!
            #
            # Start default port at 5561 to avoid conflict with InterfaceProxy
            # defaults (5557/5558)
            command_port = cam_config["command_port"] or (5561 + i * 2)
            response_port = cam_config["response_port"] or (5562 + i * 2)
            node_id = cam_config["node_id"] or f"remote_camera_{i}"

            log.info(
                f"Creating camera proxy: {node_id} at {cam_config['remote_host']}:"
                f"{command_port}/{response_port}"
            )

            camera_proxy = InterfaceProxy(
                Interface,
                remote_host=cam_config["remote_host"],
                remote_setup_conf_idx=cam_config["setup_conf_idx"],
                node_id=node_id,
                command_port=command_port,
                response_port=response_port,
            )

            # Initialize with session metadata
            camera_proxy.init_local(self.exp, self.beh)
            self.remote_cameras.append(camera_proxy)

            log.info(f"Remote camera {node_id} initialized successfully")

    def _setup_local_camera(self) -> None:
        """Setup local camera if configured in setup."""
        setup_cameras = self.logger.get(
            schema="interface",
            table="SetupConfiguration.Camera",
            fields=["setup_conf_idx"],
        )

        if self.exp.setup_conf_idx in setup_cameras:
            camera_params = self.logger.get(
                schema="interface",
                table="SetupConfiguration.Camera",
                key=f"setup_conf_idx={self.exp.setup_conf_idx}",
                as_dict=True,
            )[0]

            camera_class = getattr(
                import_module("ethopy.interfaces.Camera"), camera_params["discription"]
            )

            self.camera = camera_class(
                filename=f"{self.logger.trial_key['animal_id']}"
                f"_{self.logger.trial_key['session']}",
                logger=self.logger,
                logger_timer=self.logger.logger_timer,
                video_aim=camera_params.pop("video_aim"),
                **camera_params,
            )

    def give_liquid(self, port: int, duration: Optional[float] = 0) -> None:
        """Deliver liquid reward through specified port.

        Args:
            port: Port number for delivery
            duration: Duration of delivery in milliseconds

        """

    def give_odor(self, odor_idx: int, duration: float) -> None:
        """Deliver odor stimulus.

        Args:
            odor_idx: Index of odor to deliver
            duration: Duration of delivery in milliseconds

        """

    def give_sound(self, sound_freq: float, duration: float, dutycycle: float) -> None:
        """Generate sound stimulus.

        Args:
            sound_freq: Frequency of sound in Hz
            duration: Duration of sound in milliseconds
            dutycycle: Duty cycle for sound generation (0-1)

        """

    def in_position(self) -> Tuple[bool, float]:
        """Check if subject is in correct position.

        Returns:
            Tuple of (position status, position time)

        """
        return True, 0

    def create_pulse(self, port: int, duration: float) -> None:
        """Create a pulse for stimulus delivery.

        Args:
            port: Port number for pulse
            duration: Duration of pulse in milliseconds

        """

    def sync_out(self, state: bool = False) -> None:
        """Send synchronization signal.

        Args:
            state: Synchronization state to set

        """

    def set_operation_status(self, operation_status: bool) -> None:
        """Set operation status of interface.

        Args:
            operation_status: Status to set

        """

    def cleanup(self) -> None:
        """Clean up interface resources."""

    def stop_cameras(self) -> None:
        """Stop recording on all cameras (local and remote).

        This should be called when the experiment stops.
        """
        log.debug("STOP_CAMERAS CALLED")

        # Stop local camera
        if self.camera:
            log.debug("Stopping LOCAL camera recording")
            if self.camera.recording.is_set():
                self.camera.stop_rec()
                log.debug("✅ Local camera stop_rec() complete")
            else:
                log.warning("Local camera was not recording")
        else:
            log.debug("No local camera configured")

        # Stop remote cameras
        if hasattr(self, "remote_cameras") and self.remote_cameras:
            log.debug(f"Stopping {len(self.remote_cameras)} REMOTE cameras")
            for i, camera_proxy in enumerate(self.remote_cameras):
                try:
                    log.debug(f"Remote camera {i+1}/{len(self.remote_cameras)}: "
                              f"{camera_proxy.node_id}")
                    log.debug(f"Type: {type(camera_proxy).__name__}")

                    # Release stops the camera recording
                    log.debug("Calling release() on InterfaceProxy...")
                    camera_proxy.release()
                    log.debug(f"Remote camera {camera_proxy.node_id} release()")

                    # Cleanup destroys the interface object, returns remote to idle
                    log.debug("Calling cleanup() on InterfaceProxy...")
                    camera_proxy.cleanup()
                    log.debug(f"Remote camera {camera_proxy.node_id} cleanup()")

                    # IMPORTANT: Shutdown the NetworkServer
                    # This releases ports 5561/5562 so they can be reused
                    # in the next session. Remote node will detect disconnection
                    # and automatically reconnect when we create a new server.
                    log.debug("Calling shutdown() on InterfaceProxy...")
                    camera_proxy.shutdown()
                    log.debug(f"Remote camera {camera_proxy.node_id} server shutdown")

                except Exception as e:
                    log.error(f"❌ Error stopping remote camera "
                              f"{camera_proxy.node_id}: {e}")
                    import traceback
                    log.error(traceback.format_exc())
        else:
            log.debug("No remote cameras configured")

        log.info("STOP_CAMERAS COMPLETE")

    def release(self) -> None:
        """Release hardware resources (local camera and remote cameras)."""
        log.debug("INTERFACE.RELEASE() CALLED")

        # Stop and release local camera
        self.stop_cameras()

        log.info("INTERFACE.RELEASE() COMPLETE")

    def load_calibration(self) -> None:
        """Load port calibration data from database.

        This method loads the most recent calibration data for each reward port,
        including pulse durations and weights.

        Raises:
            RuntimeError: If no calibration data is found

        """
        for port in list(set(self.rew_ports)):
            self.pulse_rew[port] = dict()
            key = dict(setup=self.logger.setup, port=port)
            dates = self.logger.get(
                schema="interface",
                table="PortCalibration.Liquid",
                key=key,
                fields=["date"],
                order_by="date",
            )
            if np.size(dates) < 1:
                log.error("No PortCalibration found!")
                self.exp.quit = True
                break

            key["date"] = dates[-1]  # use most recent calibration

            self.pulse_dur[port], pulse_num, weight = self.logger.get(
                schema="interface",
                table="PortCalibration.Liquid",
                key=key,
                fields=["pulse_dur", "pulse_num", "weight"],
            )
            self.weight_per_pulse[port] = np.divide(weight, pulse_num)

    def calc_pulse_dur(self, reward_amount: float) -> Dict[int, float]:
        """Calculate pulse duration for desired reward amount.

        Args:
            reward_amount: Desired reward amount in microliters

        Returns:
            Dictionary mapping ports to actual reward amounts

        """
        log.info(f"calc_pulse_dur called with reward_amount={reward_amount}, rew_ports={self.rew_ports}")
        actual_rew = {}
        for port in self.rew_ports:
            if reward_amount not in self.pulse_rew[port]:
                self.duration[port] = np.interp(
                    reward_amount / 1000,
                    self.weight_per_pulse[port],
                    self.pulse_dur[port],
                )
                self.pulse_rew[port][reward_amount] = (
                    np.max((np.min(self.weight_per_pulse[port]), reward_amount / 1000))
                    * 1000
                )  # in uL
            actual_rew[port] = self.pulse_rew[port][reward_amount]
        return actual_rew

    def _channel2port(self, channel: Optional[int], category: str = "Proximity"):
        """Convert channel number to port object.

        Args:
            channel: Channel number to convert
            category: Port category to match

        Returns:
            Corresponding port or None if not found

        """
        port = reverse_lookup(self.channels[category], channel) if channel else 0
        if port:
            port = self.ports[Port(type=category, port=port) == self.ports][0]
        return port


@dataclass
class Port:
    """Dataclass representing a hardware port configuration.

    Attributes:
        port (int): Port identifier
        type (str): Port type (e.g., 'Lick', 'Proximity')
        ready (bool): Whether port is in ready state
        reward (bool): Whether port can deliver rewards
        response (bool): Whether port accepts responses
        invert (bool): Whether to invert port signal
        state (bool): Current port state

    """

    port: int = datafield(compare=True, default=0, hash=True)
    type: str = datafield(compare=True, default="", hash=True)
    ready: bool = datafield(compare=False, default=False)
    reward: bool = datafield(compare=False, default=False)
    response: bool = datafield(compare=False, default=False)
    invert: bool = datafield(compare=False, default=False)
    state: bool = datafield(compare=False, default=False)

    def __init__(self, **kwargs):
        """Initialize the instance with the given keyword arguments.

        This constructor dynamically sets the attributes of the instance
        based on the provided keyword arguments. Only attributes that are
        defined as fields of the class will be set.
        """
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@interface.schema
class SetupConfiguration(dj.Lookup, dj.Manual):
    """DataJoint table for configuring the setup interfaces.

    The user can define all harware configuration by defining only the setup index.
    """

    definition = """
    # Setup configuration
    setup_conf_idx           : tinyint      # configuration version
    ---
    interface                : enum('DummyPorts','RPPorts', 'PCPorts', 'RPVR')
    discription              : varchar(256)
    """

    contents = [
        [0, "DummyPorts", "Simulation"],
    ]

    class Port(dj.Lookup, dj.Part):
        """Port configuration table."""

        definition = """
        # Probe identityrepeat_n = 1

        port                     : tinyint                      # port id
        type="Lick"              : enum('Lick','Proximity')     # port type
        -> SetupConfiguration
        ---
        ready=0                  : tinyint                      # ready flag
        response=0               : tinyint                      # response flag
        reward=0                 : tinyint                      # reward flag
        invert=0                 : tinyint                      # invert flag
        discription              : varchar(256)
        """

        contents = [
            [1, "Lick", 0, 0, 1, 1, 0, "probe"],
            [2, "Lick", 0, 0, 1, 1, 0, "probe"],
            [3, "Proximity", 0, 1, 0, 0, 0, "probe"],
        ]

    class Screen(dj.Lookup, dj.Part):
        """Screen configuration table."""

        definition = """
        # Screen information
        screen_idx               : tinyint
        -> SetupConfiguration
        ---
        intensity                : tinyint UNSIGNED
        distance                 : float
        center_x                 : float
        center_y                 : float
        aspect                   : float
        size                     : float
        fps                      : tinyint UNSIGNED
        resolution_x             : smallint
        resolution_y             : smallint
        description              : varchar(256)
        fullscreen               : tinyint
        """

        contents = [
            [1, 0, 64, 5.0, 0, -0.1, 1.66, 7.0, 30, 800, 480, "Simulation", 0],
        ]

    class Ball(dj.Lookup, dj.Part):
        """Ball configuration table."""

        definition = """
        # Ball information
        -> SetupConfiguration
        ---
        ball_radius=0.125        : float                   # in meters
        material="styrofoam"     : varchar(64)             # ball material
        coupling="bearings"      : enum('bearings','air')  # mechanical coupling
        discription              : varchar(256)
        """

    class Speaker(dj.Lookup, dj.Part):
        """Speaker configuration table."""

        definition = """
        # Speaker information
        speaker_idx             : tinyint
        -> SetupConfiguration
        ---
        sound_freq=10000        : int           # in Hz
        duration=500            : int           # in ms
        volume=50               : tinyint       # 0-100 percentage
        discription             : varchar(256)
        """

    class Camera(dj.Lookup, dj.Part):
        """Camera configuration table."""

        definition = """
        # Camera information
        camera_idx               : tinyint
        -> SetupConfiguration
        ---
        fps                      : tinyint UNSIGNED
        resolution_x             : smallint
        resolution_y             : smallint
        shutter_speed            : smallint
        iso                      : smallint
        file_format              : varchar(256)
        video_aim                : enum('eye','body','openfield')
        discription              : varchar(256)
        """


@interface.schema
class Configuration(dj.Manual):
    """DataJoint table for saving setup configurations for each session."""

    definition = """
    # Session behavior configuration info
    -> experiment.Session
    """

    class Port(dj.Part):
        """Port configuration table."""

        definition = """
        # Probe identity
        -> Configuration
        port                     : tinyint                      # port id
        type="Lick"              : varchar(24)                 # port type
        ---
        ready=0                  : tinyint                      # ready flag
        response=0               : tinyint                      # response flag
        reward=0                 : tinyint                      # reward flag
        discription              : varchar(256)
        """

    class Ball(dj.Part):
        """Ball configuration table."""

        definition = """
        # Ball information
        -> Configuration
        ---
        ball_radius=0.125        : float                   # in meters
        material="styrofoam"     : varchar(64)             # ball material
        coupling="bearings"      : enum('bearings','air')  # mechanical coupling
        discription              : varchar(256)
        """

    class Screen(dj.Part):
        """Screen configuration table."""

        definition = """
        # Screen information
        -> Configuration
        screen_idx               : tinyint
        ---
        intensity                : tinyint UNSIGNED
        distance         : float
        center_x         : float
        center_y         : float
        aspect           : float
        size             : float
        fps                      : tinyint UNSIGNED
        resolution_x             : smallint
        resolution_y             : smallint
        description              : varchar(256)
        """

    class Speaker(dj.Part):
        """Speaker configuration table."""

        definition = """
        # Speaker information
        speaker_idx             : tinyint
        -> Configuration
        ---
        sound_freq=10000        : int           # in Hz
        duration=500            : int           # in ms
        volume=50               : tinyint       # 0-100 percentage
        discription             : varchar(256)
        """


@interface.schema
class PortCalibration(dj.Manual):
    """Liquid delivery calibration sessions for each port with water availability."""

    definition = """
    # Liquid delivery calibration sessions for each port with water availability
    setup                        : varchar(256)  # Setup name
    port                         : tinyint       # port id
    date                         : date # session date (only one per day is allowed)
    """

    class Liquid(dj.Part):
        """Datajoint table for volume per pulse duty cycle estimation."""

        definition = """
        # Data for volume per pulse duty cycle estimation
        -> PortCalibration
        pulse_dur                    : int       # duration of pulse in ms
        ---
        pulse_num                    : int       # number of pulses
        weight                       : float     # weight of total liquid released in gr
        timestamp=CURRENT_TIMESTAMP  : timestamp # timestamp
        pressure=0                   : float     # air pressure (PSI)
        """

    class Test(dj.Part):
        """Datajoint table for Lick Test."""

        definition = """
        # Lick timestamps
        setup                        : varchar(256)                 # Setup name
        port                         : tinyint                      # port id
        timestamp=CURRENT_TIMESTAMP  : timestamp
        ___
        result=null                  : enum('Passed','Failed')
        pulses=null                  : int
        """
