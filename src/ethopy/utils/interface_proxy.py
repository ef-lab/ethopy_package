"""Network proxy for running interfaces on remote computers.

This module provides an InterfaceProxy class that wraps any Interface to run it
on a remote computer via network. Similar to ProcessProxy for stimuli, but for
distributed interfaces across networked computers.
"""
import logging
import time
from typing import Type

from ethopy.utils.network import NetworkClient, NetworkServer

log = logging.getLogger(__name__)


class BehaviorProxy:
    """Proxy for behavior object on remote node that forwards events to master.

    When the remote interface calls self.beh.log_activity(), this proxy
    intercepts the call and sends it to the master via the network.

    This proxy provides stub methods (is_licking, get_response) that can be
    wrapped by DummyPorts to inject event polling, but return dummy values
    since the real behavior logic runs on the master.

    Attributes:
        client: NetworkClient for sending events to master
        logger: Logger proxy with timer for calculating timestamps
    """

    def __init__(self, client: 'NetworkClient', logger=None):
        """Initialize behavior proxy.

        Args:
            client: NetworkClient connected to master
            logger: Logger proxy with timer (set later via set_logger)
        """
        self.client = client
        self.logger = logger

    def is_licking(self, since=0, reward=False, clear=True):
        """Stub method for compatibility with DummyPorts wrapping.

        In remote mode, this does nothing since the real behavior logic
        runs on the master. DummyPorts can wrap this method to inject
        _get_events() calls for event polling.

        Returns:
            0 (no lick detected - master handles actual lick detection)
        """
        return 0

    def get_response(self, *args, **kwargs):
        """Stub method for compatibility with DummyPorts wrapping.

        In remote mode, this does nothing since the real behavior logic
        runs on the master. DummyPorts can wrap this method to inject
        _get_events() calls for event polling.

        Returns:
            False (no response - master handles actual response detection)
        """
        return False

    def log_activity(self, activity_key: dict) -> int:
        """Log activity by sending event to master.

        This is called by the interface when events occur (button presses, etc.).
        It forwards the event to the master's real behavior object.

        Uses the logger timer (synchronized with master's start_time) to calculate
        correct relative timestamps.

        Args:
            activity_key: Event data dict (port, time, in_position, etc.)

        Returns:
            Timestamp of the event (milliseconds since session start)
        """
        # Calculate timestamp using logger timer (synchronized with master)
        if self.logger and hasattr(self.logger, 'logger_timer'):
            timestamp = self.logger.logger_timer.elapsed_time()
        else:
            log.warning("Logger not set on BehaviorProxy, using 0 timestamp")
            timestamp = 0

        # Set timestamp in activity_key
        if "time" not in activity_key or not activity_key.get("time"):
            activity_key["time"] = timestamp

        # Send event to master via response channel
        # This uses the existing REQ-REP socket to send the event
        try:
            log.info(f"📤 REMOTE -> MASTER: Sending event {activity_key} (timestamp={timestamp}ms)")
            self.client._send_response("log_event", activity_key, timeout=1.0)
        except Exception as e:
            log.error(f"❌ Failed to send event to master: {e}")

        return activity_key["time"]


class InterfaceProxy:
    """Proxy that runs an interface on a remote computer via network.

    This class provides transparent access to interface objects running on remote
    computers. It forwards method calls over the network and handles responses.

    Attributes:
        _interface_class: Original interface class to be proxied
        remote_host: IP address of remote computer
        remote_setup_conf_idx: Setup configuration index for remote
        server: NetworkServer for communication with remote

    Class Attributes:
        LOCAL_METHODS: Methods that run locally, not forwarded to remote
        LOCAL_ATTRIBUTES: Attributes accessed locally
    """

    # Methods that should run locally (master side), not forwarded to remote
    # Note: calc_pulse_dur should run on remote where calibration data is
    LOCAL_METHODS = {'load_calibration'}

    # Attributes that are local only
    LOCAL_ATTRIBUTES = {'logger', 'exp', 'beh'}

    def __init__(
        self,
        interface_class: Type,
        remote_host: str,
        remote_setup_conf_idx: int,
        node_id: str = None,
        command_port: int = 5557,
        response_port: int = 5558
    ):
        """Initialize interface proxy (deferred initialization).

        Creates network connection to remote but does NOT initialize the
        interface yet. Call init_local() with session metadata to complete
        initialization.

        Args:
            interface_class: Interface class to run remotely (e.g., RPPorts, Camera)
            remote_host: IP address of remote computer
            remote_setup_conf_idx: Setup configuration index on remote
            node_id: Unique identifier for remote node
            command_port: Port for sending commands
            response_port: Port for receiving responses
        """
        self._interface_class = interface_class
        self.remote_host = remote_host
        self.remote_setup_conf_idx = remote_setup_conf_idx
        self.node_id = node_id or f"remote_{interface_class.__name__}"
        self._remote_initialized = False

        # Create network server on master
        self.server = NetworkServer(command_port=command_port, response_port=response_port)

        log.info(f"InterfaceProxy created for {interface_class.__name__} on {remote_host}")

        # Local attributes for compatibility (will be set in init_local)
        self.logger = None
        self.exp = None
        self.beh = None
        self.camera = None
        self.ports = []
        self.rew_ports = []
        self.proximity_ports = []

        # Register event handler for log_event (events from remote)
        self.server.register_event_handler("log_event", self._handle_log_event)

    def _wait_for_remote_connection(self, timeout: float):
        """Wait for remote node to connect.

        The heartbeat monitor thread will receive the initial heartbeat from the
        remote node and add it to connected_nodes. We just poll that dict.

        Args:
            timeout: Seconds to wait for connection

        Raises:
            TimeoutError: If remote doesn't connect within timeout
        """
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if heartbeat monitor has registered the node
            if self.node_id in self.server.connected_nodes:
                log.info(f"Remote node '{self.node_id}' connected")
                return

            # Small sleep to avoid busy-waiting
            time.sleep(0.1)

        raise TimeoutError(f"Remote node '{self.node_id}' failed to connect within {timeout}s")

    def _initialize_remote_interface(self):
        """Send initialization command to remote node."""
        init_data = {
            "interface_class": f"{self._interface_class.__module__}.{self._interface_class.__name__}",
            "setup_conf_idx": self.remote_setup_conf_idx
        }

        self.server.send_command("init_interface", init_data, target_node=self.node_id)
        response = self.server.get_response(
            node_id=self.node_id,
            command_type="init_interface",
            timeout=10.0
        )

        if not response or response.get("result", {}).get("status") != "initialized":
            raise RuntimeError(f"Failed to initialize remote interface: {response}")

        log.info(f"Remote interface {self._interface_class.__name__} initialized")

    def _handle_log_event(self, event_data: dict):
        """Handle log_event from remote node.

        This is called by the NetworkServer when it receives a log_event message
        from the remote node. It forwards the event to the real behavior object.

        Args:
            event_data: Event data from remote (activity_key dict)
        """
        if self.beh:
            try:
                log.info(f"MASTER <- REMOTE: Received event {event_data}")
                self.beh.log_activity(event_data)
                log.info("Event logged to behavior database")
            except Exception as e:
                log.error(f"Error logging remote event: {e}")
        else:
            log.warning(f"Received event but beh not initialized: {event_data}")

    def init_local(self, exp, beh):
        """Initialize remote interface with session metadata.

        Sends init_interface command to remote WITH session metadata included.
        This ensures remote camera gets correct animal_id, session, start_time
        before recording starts.

        Can be called multiple times for different sessions - remote will cleanup
        and re-initialize with new metadata.

        Args:
            exp: Experiment object (must have logger with trial_key)
            beh: Behavior object

        Raises:
            RuntimeError: If remote initialization fails
        """
        self.exp = exp
        self.beh = beh
        self.logger = exp.logger

        # Prepare session metadata
        session_metadata = {
            "animal_id": int(self.logger.trial_key["animal_id"]),
            "session": int(self.logger.trial_key["session"]),
            "start_time": float(self.logger.logger_timer.start_time),
            "setup": str(self.logger.setup)
        }

        # Prepare init command WITH session metadata
        init_data = {
            "interface_class": f"{self._interface_class.__module__}.{self._interface_class.__name__}",
            "setup_conf_idx": self.remote_setup_conf_idx,
            "session_metadata": session_metadata  # Include metadata!
        }

        log.info(f"Initializing remote interface with session: "
                 f"animal_id={session_metadata['animal_id']}, "
                 f"session={session_metadata['session']}")

        # Wait for remote to be fully ready (heartbeat + SUB socket synced)
        # The is_node_ready() check ensures the remote has:
        # 1. Sent a heartbeat (connected_nodes)
        # 2. Completed SUB socket sync (ready_nodes)
        # No sleep workarounds needed - sync is explicit via sync_ready message
        log.info(f"Waiting for remote node {self.node_id} to be ready...")
        max_wait = 15.0
        wait_start = time.time()
        while time.time() - wait_start < max_wait:
            if self.server.is_node_ready(self.node_id):
                log.info(f"Remote node {self.node_id} is ready (SUB socket synced)")
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Remote node {self.node_id} did not connect within {max_wait}s")

        # Send init command to remote
        self.server.send_command("init_interface", init_data, target_node=self.node_id)
        response = self.server.get_response(
            node_id=self.node_id,
            command_type="init_interface",
            timeout=15.0
        )

        if not response or response.get("result", {}).get("status") != "initialized":
            raise RuntimeError(f"Failed to initialize remote interface: {response}")

        self._remote_initialized = True
        log.info(f"Remote interface {self._interface_class.__name__} initialized successfully")

    def _deserialize_result(self, result):
        """Deserialize result from JSON format.

        Converts dictionaries that look like Port objects back to Port instances.

        Args:
            result: Result to deserialize

        Returns:
            Deserialized result
        """
        from ethopy.core.interface import Port

        # Check if this is a Port dict
        if isinstance(result, dict) and 'port' in result and 'type' in result:
            return Port(**result)

        # Handle tuples (e.g., in_position returns (Port, int, int))
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            # Check if first element is a Port dict
            if isinstance(result[0], dict) and 'port' in result[0]:
                return tuple(self._deserialize_result(item) for item in result)
            return result

        # Already deserialized or primitive type
        else:
            return result

    def __getattr__(self, name):
        """Intercept method calls and forward to remote interface."""
        # Avoid infinite recursion
        if name in ("_interface_class", "remote_host", "remote_setup_conf_idx",
                    "node_id", "server", "logger", "exp", "beh"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Local methods run on master
        if name in self.LOCAL_METHODS:
            # Return dummy implementation
            def local_method(*args, **kwargs):
                log.debug(f"Local method {name} called (not forwarded to remote)")
                return None
            return local_method

        # Forward all other methods to remote
        def remote_method(*args, **kwargs):
            """Forward method call to remote interface."""
            log.debug(f"Forwarding {name}(*{args}, **{kwargs}) to remote")

            # Check if server is still running before sending command
            if not self.server.running:
                log.warning(f"Server already shut down, skipping remote call to {name}()")
                return None

            command_data = {
                "method": name,
                "args": args,
                "kwargs": kwargs
            }
            if name not in ['in_position', 'sync_out']:
                log.debug(f"MASTER -> REMOTE: Calling {name}({args}, {kwargs})")

            self.server.send_command("call_method", command_data, target_node=self.node_id)
            response = self.server.get_response(
                node_id=self.node_id,
                command_type="call_method",
                timeout=5.0
            )

            if not response:
                log.error(f"❌ Timeout waiting for {name}() response")
                raise TimeoutError(f"Remote did not respond to {name}() call")

            if name not in ['in_position', 'sync_out']:
                log.info(f"🔵 MASTER <- REMOTE: {name}() returned {response.get('result', {})}")

            result = response.get("result", {})
            if result.get("status") == "error":
                raise RuntimeError(f"Remote error in {name}(): {result.get('message')}")

            return_value = result.get("return_value")

            # Deserialize Port objects if needed
            return_value = self._deserialize_result(return_value)

            return return_value

        return remote_method

    def cleanup(self):
        """Clean up remote interface resources (does NOT shutdown server).

        Sends cleanup command to remote to destroy interface object.
        Server stays alive for reuse in next session.
        Call shutdown() when truly done to close network connections.
        """
        log.info("Cleaning up remote interface (server stays alive)")
        self.server.send_command("cleanup", {}, target_node=self.node_id)
        # Don't shutdown server - keep it alive for next session

    def shutdown(self):
        """Shutdown network server and close all connections.

        Call this when completely done with the remote interface,
        not between sessions. For between-session cleanup, use cleanup().
        """
        log.info("Shutting down InterfaceProxy network server")
        self.server.shutdown()


class RemoteInterfaceNode:
    """Remote node that runs the actual interface and responds to master commands.

    This class runs on the remote computer (e.g., Raspberry Pi) and creates
    the real interface object. It receives commands from InterfaceProxy and
    executes them on the interface.

    Attributes:
        master_host: IP address of master computer
        node_id: Unique identifier for this node
        client: NetworkClient for communication with master
        interface: Actual interface object (e.g., RPPorts instance)
        logger: Logger proxy for remote logging
        exp: Experiment proxy for remote
        beh: Behavior proxy for remote
    """

    def __init__(
        self,
        master_host,
        node_id: str,
        command_port: int = 5557,
        response_port: int = 5558
    ):
        """Initialize remote interface node.

        Args:
            master_host: IP address(es) of master computer(s). Can be:
                - Single IP: "192.168.1.10"
                - List of IPs: ["192.168.1.10", "192.168.1.20"]
                - IP range: "192.168.1.10-30" (auto-expands to IPs 10-30)
                Remote will connect to whichever master responds first.
            node_id: Unique identifier for this node.
            command_port: Port for receiving commands.
            response_port: Port for sending responses.
        """
        self.master_host = master_host
        self.node_id = node_id
        self.interface = None
        self.logger = None
        self.exp = None
        self.beh = None

        # Connect to master
        self.client = NetworkClient(
            master_host=master_host,
            command_port=command_port,
            response_port=response_port,
            node_id=node_id
        )

        # Register command handlers
        self.client.register_handler("init_interface", self._handle_init_interface)
        self.client.register_handler("set_session", self._handle_set_session)
        self.client.register_handler("call_method", self._handle_call_method)
        self.client.register_handler("cleanup", self._handle_cleanup)

        # Register disconnect callback to cleanup interface immediately on connection loss
        self.client.on_disconnect(self._on_disconnect)

        # Mark client as ready to process commands
        # This allows heartbeats to be sent, signaling to master that we're ready
        self.client.mark_ready()

        log.info(f"RemoteInterfaceNode '{node_id}' connected to {master_host}")

    def _on_disconnect(self) -> None:
        """Called when connection to master is lost.

        Immediately cleanup interface (stop camera, transfer files) rather than
        waiting for reconnection and new init_interface command.
        """
        log.info("Connection lost - cleaning up interface immediately")
        if self.interface:
            try:
                if hasattr(self.interface, 'cleanup'):
                    self.interface.cleanup()
                if hasattr(self.interface, 'release'):
                    self.interface.release()
                log.info("Interface cleanup complete")
            except Exception as e:
                log.error(f"Error during disconnect cleanup: {e}")
            finally:
                self.interface = None

    def _handle_init_interface(self, data: dict) -> dict:
        """Initialize interface WITH session metadata.

        Session metadata MUST be included in data. This ensures camera
        gets correct animal_id, session, and start_time before recording.

        Can be called multiple times for different sessions - will cleanup
        existing interface first.

        Args:
            data: Contains interface_class, setup_conf_idx, and session_metadata

        Returns:
            Status dict
        """
        try:
            # If interface already exists, cleanup first for re-initialization
            if self.interface:
                log.info("Cleaning up existing interface for re-initialization")
                if hasattr(self.interface, 'cleanup'):
                    self.interface.cleanup()
                if hasattr(self.interface, 'release'):
                    self.interface.release()
                self.interface = None

            from importlib import import_module

            # Import interface class
            class_path = data["interface_class"]
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            interface_class = getattr(module, class_name)

            # Create logger
            from ethopy.core.logger import Logger
            self.logger = Logger()

            # Session metadata MUST be provided
            session_meta = data.get("session_metadata")
            if not session_meta:
                raise ValueError("session_metadata is required in init_interface command")

            # Update logger with session metadata BEFORE creating interface
            self.logger.trial_key["animal_id"] = session_meta["animal_id"]
            self.logger.trial_key["session"] = session_meta["session"]
            self.logger.logger_timer.start_time = session_meta["start_time"]

            log.info(f"Session metadata: animal_id={session_meta['animal_id']}, "
                     f"session={session_meta['session']}, "
                     f"start_time={session_meta['start_time']}")

            # Create minimal exp proxy
            class ExpProxy:
                def __init__(self, logger, setup_conf_idx):
                    self.logger = logger
                    self.setup_conf_idx = setup_conf_idx
                    self.params = {}
                    self.sync = False
                    self.curr_trial = 0
                    self.curr_cond = {}
                    self.quit = False
                    self.in_operation = False
                    self.session_params = {"setup_conf_idx": setup_conf_idx}

            self.exp = ExpProxy(self.logger, data["setup_conf_idx"])

            # Create behavior proxy
            self.beh = BehaviorProxy(self.client, logger=self.logger)

            # Create interface (ports will be initialized)
            self.interface = interface_class(exp=self.exp, beh=self.beh)

            # Initialize camera NOW (after metadata is ready)
            if hasattr(self.interface, '_setup_local_camera'):
                self.interface._setup_local_camera()
                log.info("Camera initialized with correct session metadata")

                # Auto-start camera recording if camera exists
                if self.interface.camera:
                    log.info("Camera recording will start automatically via camera_process")
                    log.info(f"Video file: {self.interface.camera.filename}.mp4")
                    log.info(f"Source path: {self.interface.camera.source_path}")

            log.info(f"Interface {class_name} initialized on remote")

            return {"status": "initialized", "interface": class_name}

        except Exception as e:
            log.error(f"Failed to initialize interface: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_set_session(self, data: dict) -> dict:
        """Update session metadata from master.

        Args:
            data: Session metadata (animal_id, session, start_time, etc.)

        Returns:
            Status dict
        """
        try:
            # Update logger with session info
            self.logger.trial_key["animal_id"] = data["animal_id"]
            self.logger.trial_key["session"] = data["session"]
            self.logger.logger_timer.start_time = data["start_time"]

            log.info(f"Session metadata updated: {data}")

            return {"status": "updated"}

        except Exception as e:
            log.error(f"Failed to set session: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_call_method(self, data: dict) -> dict:
        """Execute method on interface.

        Args:
            data: Contains method name, args, kwargs

        Returns:
            Result dict with return value or error
        """
        try:
            method_name = data["method"]
            args = data.get("args", ())
            kwargs = data.get("kwargs", {})

            if not self.interface:
                return {"status": "error", "message": "Interface not initialized"}

            # Call method on interface
            method = getattr(self.interface, method_name)
            result = method(*args, **kwargs)

            log.debug(f"Executed {method_name}() -> {result}")

            return {"status": "success", "return_value": result}

        except Exception as e:
            log.error(f"Error calling {data.get('method')}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_cleanup(self, data: dict) -> dict:
        """Clean up interface resources and return to idle.

        Args:
            data: Optional cleanup parameters

        Returns:
            Status dict
        """
        try:
            if self.interface:
                if hasattr(self.interface, 'cleanup'):
                    self.interface.cleanup()
                if hasattr(self.interface, 'release'):
                    self.interface.release()
                self.interface = None

            log.info("Interface cleaned up, ready for next session")

            return {"status": "cleaned"}

        except Exception as e:
            log.error(f"Cleanup error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def run(self):
        """Main loop processing commands from master."""
        log.info("RemoteInterfaceNode running, waiting for commands...")

        try:
            while True:
                # Process commands (blocks with timeout)
                self.client.process_commands(timeout=0.1)

        except KeyboardInterrupt:
            log.info("RemoteInterfaceNode stopped by user")
        finally:
            self.client.shutdown()
