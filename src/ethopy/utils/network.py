"""ZMQ-based networking for coordinating distributed nodes.

One master sends commands to one or more remotes and collects their
responses. Remotes run continuously, staying in standby between sessions
and reconnecting whenever a master becomes available.

Communication uses two ZMQ patterns:
  - PUB/SUB for commands: master broadcasts to all remotes or a specific
    node. Commands are confirmed via get_response(), so lost commands
    surface as timeouts rather than silent failures.
  - REQ/REP for responses and heartbeats: guaranteed delivery (blocks
    until acknowledged). Used to return results and detect dead nodes.

This design suits: one coordinator sending tasks to N persistent workers,
where responses are always expected.
It does NOT suit:
  - Peer-to-peer communication: remotes cannot send commands to each
    other, only to the master.
  - Streaming or large data transfer: all messages are JSON-serialized,
    so binary blobs (e.g. video frames) should use a separate channel.
  - Message queuing when remotes are offline: commands sent while a
    remote is disconnected are lost; the master will receive a timeout.

Key classes:
    NetworkServer         -- runs on master, sends commands and receives responses
    NetworkClient         -- runs on remote, receives commands and sends responses
    NodeDisconnectedError -- raised when communicating with a timed-out node

For usage examples see docs/network_module_guide.md.

Requires pyzmq: pip install pyzmq
"""
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False
    logging.warning("ZMQ not available - network features disabled")

log = logging.getLogger(__name__)


class NodeDisconnectedError(Exception):
    """Raised when attempting to communicate with a disconnected node."""
    pass


# =============================================================================
# NetworkServer (Master side)
# =============================================================================


class NetworkServer:
    """Runs on the master computer to coordinate remote nodes.

    Broadcasts commands to connected remotes and collects their responses.
    A background thread monitors heartbeats to detect disconnected nodes.

    Typical usage::

        server = NetworkServer(command_port=5555, response_port=5556)
        server.add_node("worker", timeout=15.0)  # block until remote connects

        req_id = server.send_command("start", {"param": 1}, target_node="worker")
        response = server.get_response("worker", "start", req_id, timeout=5.0)
        server.shutdown()

    Attributes:
        nodes: Connected nodes mapped to last heartbeat time {node_id: timestamp}.
               A node appears here only after sending sync_ready.
        node_ips: IP address of each connected node {node_id: ip}.
        running: False after shutdown() is called.
    """

    def __init__(self, command_port: int = 5555, response_port: int = 5556):
        """Initialize and start the network server.

        Args:
            command_port: Port for broadcasting commands to remotes.
            response_port: Port for receiving responses and heartbeats from remotes.

        Raises:
            RuntimeError: If pyzmq is not installed.
        """
        if not HAVE_ZMQ:
            raise RuntimeError("ZMQ not installed - run: pip install pyzmq")

        self.context = zmq.Context()

        # PUB socket for broadcasting commands to all remotes
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{command_port}")

        # REP socket for receiving responses from remotes
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://*:{response_port}")

        self.running = True

        # Track server start time to filter stale heartbeats from previous sessions
        self.start_time = time.time()

        # Node tracking - only populated on sync_ready (not on heartbeat)
        self.nodes: Dict[str, float] = {}  # {node_id: last_heartbeat_time}
        self.node_ips: Dict[str, str] = {}  # {node_id: client_ip}

        # Response cache for async retrieval.
        self._request_counter = 0
        self._response_cache: Dict[tuple, dict] = {}  # {(node_id, command_type, request_id)}
        self._response_lock = threading.Lock()

        # Event handlers for async events from remotes (e.g., log_event)
        self._event_handlers: Dict[str, Callable] = {}

        # Disconnect handlers - called when a node times out
        self._disconnect_handlers: List[Callable[[str], None]] = []

        log.info(f"NetworkServer started on ports {command_port} (cmd), {response_port} (resp)")

        # Start response monitor thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register handler for async events from remotes.

        Args:
            event_type: Type of event (e.g., "log_event")
            handler: Function(event_data) to call when event received
        """
        self._event_handlers[event_type] = handler
        log.debug(f"Registered event handler: {event_type}")

    def on_node_disconnect(self, callback: Callable[[str], None]) -> None:
        """Register callback when any node disconnects.

        The callback receives the node_id of the disconnected node.
        Callbacks run in the monitor thread - keep them quick.

        Args:
            callback: Function(node_id: str) to call when node disconnects
        """
        self._disconnect_handlers.append(callback)

    def send_command(
        self,
        command_type: str,
        data: Optional[Dict] = None,
        target_node: str = "all",
    ) -> int:
        """Send command to remote node(s) via PUB socket.

        Args:
            command_type: Command identifier (e.g., "init_interface", "call_method")
            data: Command payload
            target_node: Node ID or "all" for broadcast

        Returns:
            request_id: Unique ID for this request, pass to get_response()

        # TODO: add fire_and_forget=False parameter — when True, skip get_response()
        # and immediately mark request as abandoned so its late response is discarded.
        """
        self._request_counter += 1
        request_id = self._request_counter
        message = {
            "target": target_node,
            "type": command_type,
            "request_id": request_id,
            "data": data or {},
            "timestamp": time.time(),
        }
        self.pub_socket.send_json(message)
        log.debug(f"Sent command '{command_type}' to {target_node} (req_id={request_id})")
        return request_id

    def get_response(
        self,
        node_id: str,
        command_type: str,
        request_id: int,
        timeout: float = 5.0,
        clear: bool = True
    ) -> Optional[Dict]:
        """Wait for and retrieve cached response from a node.

        The monitor thread continuously receives responses and caches them.
        This method polls the cache until the expected response arrives.

        Args:
            node_id: ID of node to get response from
            command_type: Type of command to get response for
            request_id: ID returned by send_command(), prevents stale response collisions
            timeout: Max seconds to wait
            clear: Remove response from cache after retrieval

        Returns:
            Response dict or None if timeout

        Raises:
            NodeDisconnectedError: If node is not currently connected
        """
        # Check if node is still connected
        if node_id not in self.nodes:
            raise NodeDisconnectedError(
                f"Node '{node_id}' is disconnected. Cannot get response for '{command_type}'."
            )

        cache_key = (node_id, command_type, request_id)
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._response_lock:
                if cache_key in self._response_cache:
                    response = self._response_cache[cache_key]
                    if clear:
                        del self._response_cache[cache_key]
                    return response
            time.sleep(0.05)

        log.warning(f"Response timeout: {node_id}/{command_type} (req_id={request_id})")
        return None

    def add_node(self, node_id: str, timeout: float = 30.0) -> None:
        """Wait for specific node to connect and be ready.

        Blocks until the node sends sync_ready (SUB socket established).
        Provides informative error if a different node connects instead.

        Args:
            node_id: ID of node to wait for
            timeout: Maximum seconds to wait

        Raises:
            TimeoutError: If node doesn't connect in time, with details about
                what nodes DID connect (helps debug node_id mismatches)
        """
        log.info(f"Waiting for node '{node_id}'...")
        start = time.time()
        while time.time() - start < timeout:
            if node_id in self.nodes:
                log.info(f"Node '{node_id}' is ready")
                return
            time.sleep(0.5)

        # Timeout - provide informative error about what actually connected
        connected = list(self.nodes.keys())
        if connected:
            # Show node_id -> IP mapping for debugging
            node_info = [f"'{nid}' (IP: {self.node_ips.get(nid, 'unknown')})"
                         for nid in connected]
            raise TimeoutError(
                f"Expected node '{node_id}' but connected node(s): {', '.join(node_info)}. "
                f"Hint: Remote client defaults to socket.gethostname() if node_id not set."
            )
        raise TimeoutError(f"No nodes connected within {timeout}s")

    def _monitor_loop(self) -> None:
        """Receive and process all messages from remotes.

        Runs in dedicated thread. Handles:
        - Heartbeats: updates timestamp for known nodes, ignores unknown
        - sync_ready: adds node to self.nodes
        - Responses: caches for retrieval via get_response()
        - Events: dispatches to registered handlers
        """
        while self.running:
            try:
                # Wait for message with timeout
                self.rep_socket.setsockopt(zmq.RCVTIMEO, 1000)
                message = self.rep_socket.recv_json()
                self.rep_socket.send_json({"status": "ack"})

                node_id = message.get("node_id")
                msg_type = message.get("command_type")
                timestamp = message.get("timestamp", 0)

                # Ignore stale messages from before server start
                if timestamp < self.start_time:
                    log.debug(f"Ignoring stale message from {node_id}")
                    continue

                # Handle heartbeats - only update known nodes
                if msg_type == "heartbeat":
                    if node_id not in self.nodes:
                        log.debug(f"Heartbeat from unknown node: {node_id}")
                    else:
                        self.nodes[node_id] = time.time()
                    continue

                # Handle sync_ready - add node to tracking
                if msg_type == "sync_ready":
                    client_ip = message.get("client_ip", "unknown")
                    self.nodes[node_id] = time.time()
                    self.node_ips[node_id] = client_ip
                    log.info(f"Node connected: {node_id} (IP: {client_ip})")
                    continue

                # Handle async events (log_event, etc.)
                if msg_type in self._event_handlers:
                    try:
                        self._event_handlers[msg_type](message.get("result", {}))
                    except Exception as e:
                        log.error(f"Event handler error ({msg_type}): {e}")
                    continue

                # Cache response for get_response() retrieval
                with self._response_lock:
                    request_id = message.get("request_id")
                    cache_key = (node_id, msg_type, request_id)
                    self._response_cache[cache_key] = message

            except zmq.Again:
                pass  # Timeout, check for dead nodes
            except Exception as e:
                if self.running:
                    log.debug(f"Monitor error: {e}")

            # Check for dead nodes (no heartbeat in 10s)
            self._check_dead_nodes()

    def _check_dead_nodes(self, timeout: float = 5.0) -> None:
        """Remove nodes that haven't sent heartbeat recently."""
        now = time.time()
        dead = [nid for nid, last in self.nodes.items() if now - last > timeout]
        for node_id in dead:
            log.warning(f"Node '{node_id}' disconnected (no heartbeat response)")
            del self.nodes[node_id]
            self.node_ips.pop(node_id, None)
            # Clean stale cached responses for this node
            with self._response_lock:
                stale_keys = [k for k in self._response_cache if k[0] == node_id]
                for key in stale_keys:
                    del self._response_cache[key]
            # Invoke disconnect callbacks
            for handler in self._disconnect_handlers:
                try:
                    handler(node_id)
                except Exception as e:
                    log.error(f"Disconnect handler error: {e}")

    def shutdown(self) -> None:
        """Shutdown server gracefully.

        Sends shutdown notification to remotes before closing sockets.
        """
        log.info("NetworkServer shutting down...")

        # Notify remotes of shutdown
        try:
            self.send_command("master_shutdown", {})
            time.sleep(0.3)
        except Exception as e:
            log.debug(f"Could not send shutdown notification: {e}")

        self.running = False
        self.pub_socket.close()
        self.rep_socket.close()
        self.context.term()
        log.info("NetworkServer shutdown complete")


# =============================================================================
# NetworkClient (Remote side)
# =============================================================================


class ConnectionState(Enum):
    """Connection states for NetworkClient."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()


@dataclass
class OutboxMessage:
    """Message to be sent via REQ socket.

    Attributes:
        msg_type: Message type identifier
        data: Message payload
        done_event: Set when message is sent (for sync sends)
        error: Set if send fails
    """
    msg_type: str
    data: dict
    done_event: Optional[threading.Event] = None
    error: Optional[Exception] = None
    request_id: Optional[int] = None


class NetworkClient:
    """Runs on a remote node to receive commands from the master.

    Connects to a NetworkServer, stays in standby between sessions, and
    reconnects automatically whenever the master becomes available.

    Typical usage::

        client = NetworkClient(master_host="xxx.xxx.x.xx", node_id="rpi_camera_1")
        client.register_handler("ping", lambda data: {"response": "pong"})
        client.connect()  # blocks until master acknowledges

        try:
            while True:
                client.process_commands(timeout=0.1)
        finally:
            client.shutdown()

    Attributes:
        node_id: Unique identifier for this node (defaults to hostname).
        state: Current connection state, one of ConnectionState.DISCONNECTED,
               CONNECTING, CONNECTED, or RECONNECTING.
        running: False after shutdown() is called.
        command_handlers: Handlers registered via register_handler() {command_type: callable}.
    """

    # Configuration
    HEARTBEAT_INTERVAL = 2.0  # Seconds between heartbeats
    DISCOVERY_TIMEOUT = 2.0   # Timeout for master probe
    SEND_TIMEOUT = 2.0        # Timeout for REQ-REP round-trip
    RECONNECT_DELAY = 5.0     # Delay between reconnection attempts

    def __init__(
        self,
        master_host: Union[str, List[str]],
        command_port: int = 5555,
        response_port: int = 5556,
        node_id: Optional[str] = None,
    ):
        """Initialize the network client.

        Args:
            master_host: IP address(es) of the master. Accepted formats:
                - Single IP:  "xxx.xxx.x.xx"
                - List:       ["xxx.xxx.x.xx", "xxx.xxx.x.yy"]
                - Range:      "xxx.xxx.x.10-30" (expands to .10 through .30)
                All candidates are probed in parallel; first to respond wins.
            command_port: Port for receiving commands (must match server's command_port).
            response_port: Port for sending responses (must match server's response_port).
            node_id: Unique identifier for this node. Defaults to hostname.
                     Must match the node_id used on the master side.

        Raises:
            RuntimeError: If pyzmq is not installed.
        """
        if not HAVE_ZMQ:
            raise RuntimeError("ZMQ not installed - run: pip install pyzmq")

        # ZMQ context - manages all sockets, one per process
        self.context = zmq.Context()

        # Connection parameters (stored for reconnection)
        self.node_id = node_id or socket.gethostname()
        self.client_ip = self._get_local_ip()
        self.command_port = command_port   # Master's PUB port (we SUB to it)
        self.response_port = response_port  # Master's REP port (we REQ to it)

        # List of master IPs to try (supports failover)
        self._master_candidates = self._parse_hosts(master_host)

        # Connection state machine (thread-safe via _state_lock)
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()

        # Command handlers: {"command_type": handler_func}
        # Called by process_commands() when master sends a command
        self.command_handlers: Dict[str, Callable] = {}

        # Disconnect callback: called once when entering RECONNECTING state
        # Use this to cleanup resources (e.g., stop camera) before reconnection
        self._on_disconnect_callback: Optional[Callable] = None
        self._disconnect_handled = False  # Ensures callback runs only once per disconnect

        # Flag for graceful shutdown - adds delay before reconnect to let master fully close
        self._graceful_shutdown = False

        # Outbox queue: main thread queues messages, outbox thread sends them
        # This avoids lock contention on the REQ socket
        self._outbox: queue.Queue[OutboxMessage] = queue.Queue()

        # Current master IP (set by outbox thread during reconnect, read by main thread)
        self._current_master: Optional[str] = None

        # ZMQ sockets with SPLIT OWNERSHIP to avoid cross-thread crashes:
        # - sub_socket: owned by MAIN thread (created/closed in process_commands)
        # - req_socket: owned by OUTBOX thread (created/closed in _reconnect)
        self.sub_socket: Optional[zmq.Socket] = None
        self.req_socket: Optional[zmq.Socket] = None

        # Control flags
        self.running = True   # Set to False in shutdown() to stop all threads

        # Handle graceful master shutdown (triggers reconnection)
        self.register_handler("master_shutdown", self._on_master_shutdown)

        # Thread created in connect()
        self._outbox_thread: Optional[threading.Thread] = None

        log.info(f"NetworkClient '{self.node_id}' initialized")

    @property
    def state(self) -> ConnectionState:
        """Current connection state (thread-safe)."""
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, value: ConnectionState) -> None:
        with self._state_lock:
            old = self._state
            self._state = value
            if old != value:
                log.debug(f"State: {old.name} -> {value.name}")

        # Call disconnect callback when entering RECONNECTING (outside lock)
        if value == ConnectionState.RECONNECTING and not self._disconnect_handled:
            self._disconnect_handled = True
            if self._on_disconnect_callback:
                try:
                    log.info("Calling disconnect callback...")
                    self._on_disconnect_callback()
                except Exception as e:
                    log.error(f"Disconnect callback error: {e}")

        # Reset flag when connected again
        if value == ConnectionState.CONNECTED:
            self._disconnect_handled = False

    def on_disconnect(self, callback: Callable) -> None:
        """Register callback to be called when connection is lost.

        The callback is called once when entering RECONNECTING state,
        before reconnection attempts begin. Use this to cleanup resources
        like stopping cameras or saving data.

        Args:
            callback: Function with no arguments to call on disconnect
        """
        self._on_disconnect_callback = callback

    def _get_local_ip(self) -> str:
        """Get local IP via system routing table (avoids DNS hostname issues)."""
        try:
            # Create UDP socket and "connect" to external address
            # This doesn't send anything, just determines route
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            # Fallback to hostname resolution
            return socket.gethostbyname(socket.gethostname())

    def _parse_hosts(self, master_host: Union[str, List[str]]) -> List[str]:
        """Parse master_host into list of IP addresses.

        Supports single IP, list of IPs, or IP range notation.

        Args:
            master_host: IP specification

        Returns:
            List of IP addresses to probe

        Raises:
            ValueError: If format is invalid
        """
        if isinstance(master_host, list):
            return master_host

        if isinstance(master_host, str):
            # Check for range notation: "192.168.1.10-20"
            if "-" in master_host and master_host.count(".") == 3:
                base, range_part = master_host.rsplit(".", 1)
                if "-" in range_part:
                    try:
                        start, end = range_part.split("-")
                        return [f"{base}.{i}" for i in range(int(start), int(end) + 1)]
                    except ValueError:
                        pass
            return [master_host]

        raise ValueError(f"Invalid master_host: {master_host}")

    def _probe_host(self, host: str) -> bool:
        """Check if master is responsive at given host.

        Creates temporary REQ socket, sends probe, waits for ACK.

        Args:
            host: IP address to probe

        Returns:
            True if master responded
        """
        sock = None
        try:
            sock = self.context.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{host}:{self.response_port}")

            sock.send_json({
                "node_id": self.node_id,
                "client_ip": self.client_ip,
                "command_type": "heartbeat",
                "result": {"status": "probe"},
                "timestamp": time.time()
            })

            sock.setsockopt(zmq.RCVTIMEO, int(self.DISCOVERY_TIMEOUT * 1000))
            sock.recv_json()
            return True

        except Exception:
            return False
        finally:
            if sock:
                sock.close()

    def _discover_master(self) -> str:
        """Find first responsive master from candidates.

        Probes all candidates in parallel, returns first responder.

        Returns:
            IP address of responsive master

        Raises:
            TimeoutError: If no master responds
        """
        result_queue: queue.Queue[str] = queue.Queue()
        found = threading.Event()

        def probe(host: str) -> None:
            if found.is_set():
                return
            if self._probe_host(host):
                if not found.is_set():
                    found.set()
                    result_queue.put(host)

        # Probe all candidates in parallel
        threads = [
            threading.Thread(target=probe, args=(h,), daemon=True)
            for h in self._master_candidates
        ]
        for t in threads:
            t.start()

        try:
            master = result_queue.get(timeout=self.DISCOVERY_TIMEOUT + 1)
            log.info(f"Discovered master: {master}")
            return master
        except queue.Empty:
            raise TimeoutError(f"No master found: {self._master_candidates}")

    # -------------------------------------------------------------------------
    # Socket management (split ownership: SUB=main thread, REQ=outbox thread)
    # -------------------------------------------------------------------------

    def _create_sub_socket(self, master: str) -> None:
        """Create SUB socket. Called by MAIN thread only."""
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{master}:{self.command_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def _create_req_socket(self, master: str) -> None:
        """Create REQ socket. Called by OUTBOX thread only."""
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{master}:{self.response_port}")

    def _close_sub_socket(self) -> None:
        """Close SUB socket. Called by MAIN thread only."""
        if self.sub_socket:
            try:
                self.sub_socket.setsockopt(zmq.LINGER, 0)
                self.sub_socket.close()
                self.sub_socket = None
            except Exception:
                pass

    def _close_req_socket(self) -> None:
        """Close REQ socket. Called by OUTBOX thread only."""
        if self.req_socket:
            try:
                self.req_socket.setsockopt(zmq.LINGER, 0)
                self.req_socket.close()
                self.req_socket = None
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------

    def _connect(self) -> None:
        """Initial connection to master (called from __init__, before threads start)."""
        self.state = ConnectionState.CONNECTING
        log.info("Connecting to master...")

        start = time.time()
        last_log = start

        while True:
            try:
                self._current_master = self._discover_master()
                # Safe to create both sockets here - no threads running yet
                self._create_sub_socket(self._current_master)
                self._create_req_socket(self._current_master)
                self.state = ConnectionState.CONNECTED
                return
            except TimeoutError:
                now = time.time()
                if now - last_log >= 30:
                    log.info(f"Still waiting for master... ({now - start:.0f}s)")
                    last_log = now
                time.sleep(self.RECONNECT_DELAY)

    def _handle_graceful_shutdown(self) -> None:
        """Wait for master to fully close before reconnecting.

        When master sends 'master_shutdown' command, _on_master_shutdown()
        sets _graceful_shutdown=True. This method adds a delay so the master
        has time to close its sockets before we attempt reconnection.

        See: _on_master_shutdown() where _graceful_shutdown is set to True.
        """
        if not self._graceful_shutdown:
            return

        log.info("Graceful shutdown - waiting for master to close...")
        time.sleep(0.5)
        self._graceful_shutdown = False

    def _log_reconnect_status(self, attempts: int, log_interval: int) -> None:
        """Log reconnection status without flooding logs.

        Logs immediately on first attempt, then once per hour to avoid
        filling logs during extended waits (e.g., master offline for days).

        Args:
            attempts: Number of reconnection attempts so far
            log_interval: Attempts between hourly log messages
        """
        if attempts == 1:
            log.info("No master available, waiting...")
        elif attempts % log_interval == 0:
            hours = (attempts * self.RECONNECT_DELAY) / 3600
            log.info(f"Still waiting for master ({hours:.1f}h, {attempts} attempts)")

    def _reconnect(self) -> None:
        """Reconnect after connection loss. Called by OUTBOX thread only.

        Only manages REQ socket. Main thread manages its own SUB socket
        when it detects RECONNECTING state in process_commands().
        """
        self.state = ConnectionState.RECONNECTING
        log.warning("Reconnecting to master...")

        self._handle_graceful_shutdown()
        self._close_req_socket()

        attempts = 0
        hourly_attempts = int(3600 / self.RECONNECT_DELAY)

        while self.running:
            try:
                self._current_master = self._discover_master()
                self._create_req_socket(self._current_master)
                # Don't send sync_ready here — SUB socket hasn't been created yet.
                # process_commands() sends sync_ready after creating the SUB socket.

                self.state = ConnectionState.CONNECTED
                log.info(f"Reconnected to {self._current_master}")
                return
            except TimeoutError:
                attempts += 1
                self._log_reconnect_status(attempts, hourly_attempts)
                time.sleep(self.RECONNECT_DELAY)
            except Exception as e:
                log.warning(f"Reconnect error: {e}")
                time.sleep(self.RECONNECT_DELAY)

    def _do_send(self, msg_type: str, data: Any, request_id: Optional[int] = None) -> None:
        """Send message via REQ socket (outbox thread only).

        Args:
            msg_type: Message type identifier
            data: Message payload
            request_id: Echo back the master's request_id so response can be matched

        Raises:
            Exception: If send or recv fails
        """
        message = {
            "node_id": self.node_id,
            "client_ip": self.client_ip,
            "command_type": msg_type,
            "result": self._serialize(data),
            "timestamp": time.time(),
            "request_id": request_id,
        }

        self.req_socket.send_json(message)
        self.req_socket.setsockopt(zmq.RCVTIMEO, int(self.SEND_TIMEOUT * 1000))
        self.req_socket.recv_json()  # Wait for ACK

    def _serialize(self, data: Any) -> Any:
        """Convert data to JSON-serializable format.

        Handles Port objects, numpy scalars, nested structures.
        """
        if hasattr(data, '__dict__') and hasattr(data, 'port'):
            return data.__dict__
        elif isinstance(data, tuple):
            return tuple(self._serialize(x) for x in data)
        elif isinstance(data, list):
            return [self._serialize(x) for x in data]
        elif isinstance(data, dict):
            return {
                (k.item() if hasattr(k, 'item') else k): self._serialize(v)
                for k, v in data.items()
            }
        elif hasattr(data, 'item'):
            return data.item()
        return data

    def _outbox_loop(self) -> None:
        """Process outbox queue and send heartbeats.

        This is the ONLY thread that uses the REQ socket.
        Runs continuously until shutdown.
        """
        while self.running:
            # Handle reconnection
            if self.state == ConnectionState.RECONNECTING:
                self._reconnect()
                continue

            # Check for broken connection
            if self.state != ConnectionState.CONNECTED:
                time.sleep(0.1)
                continue

            try:
                # Get queued message or timeout for heartbeat
                try:
                    msg = self._outbox.get(timeout=self.HEARTBEAT_INTERVAL)
                except queue.Empty:
                    # No queued messages - send heartbeat
                    self._do_send("heartbeat", {"status": "alive"})
                    continue

                # Send queued message
                try:
                    self._do_send(msg.msg_type, msg.data, msg.request_id)
                    if msg.done_event:
                        msg.done_event.set()
                except Exception as e:
                    msg.error = e
                    if msg.done_event:
                        msg.done_event.set()
                    raise

            except Exception as e:
                log.warning(f"Outbox error: {e}")
                self.state = ConnectionState.RECONNECTING

    def register_handler(self, command_type: str, handler: Callable) -> None:
        """Register handler for a command type.

        Args:
            command_type: Command identifier
            handler: Function(data) -> result to call
        """
        self.command_handlers[command_type] = handler
        log.debug(f"Registered handler: {command_type}")

    def connect(self) -> None:
        """Connect to master and block until acknowledged.

        Call this after registering all command handlers. This method:
        1. Discovers and connects to the master
        2. Starts the outbox thread for sending messages
        3. Blocks until master acknowledges the connection

        Raises:
            RuntimeError: If already connected
        """
        if self.state == ConnectionState.CONNECTED:
            raise RuntimeError("Already connected")

        # Discover master and create sockets
        self._connect()

        # Start outbox thread
        self._outbox_thread = threading.Thread(target=self._outbox_loop, daemon=True)
        self._outbox_thread.start()

        # Block until master ACKs (REQ/REP is synchronous)
        log.info("Syncing with master...")
        self.send("sync_ready", {"status": "initial"}, wait=True, timeout=20.0)
        log.info(f"Client '{self.node_id}' connected to {self._current_master}")

    def send(self, msg_type: str, data: dict, wait: bool = True, timeout: float = 5.0,
             request_id: Optional[int] = None) -> None:
        """Queue a message to be sent to master.

        Args:
            msg_type: Message type identifier
            data: Message payload
            wait: If True, block until sent
            timeout: Max seconds to wait (if wait=True)
            request_id: Echo master's request_id so response can be matched in cache

        Raises:
            RuntimeError: If not connected (call connect() first)
            TimeoutError: If wait=True and send times out
            Exception: If send fails
        """
        if self.state == ConnectionState.DISCONNECTED:
            raise RuntimeError("Not connected - call connect() first")

        if wait:
            event = threading.Event()
            msg = OutboxMessage(msg_type, data, event, request_id=request_id)
            self._outbox.put(msg)

            if not event.wait(timeout):
                raise TimeoutError(f"Send timeout: {msg_type}")
            if msg.error:
                raise msg.error
        else:
            self._outbox.put(OutboxMessage(msg_type, data, request_id=request_id))

    def process_commands(self, timeout: float = 1.0) -> bool:
        """Poll for and handle commands from master.

        This method owns the SUB socket lifecycle. When reconnection is needed,
        it closes its SUB socket, waits for outbox thread to reconnect,
        then creates a new SUB socket.

        Args:
            timeout: Seconds to wait for command

        Returns:
            True if command was processed
        """
        # Handle reconnection (we own SUB socket)
        if self.state == ConnectionState.RECONNECTING:
            self._close_sub_socket()
            # Wait for outbox thread to reconnect
            while self.state != ConnectionState.CONNECTED and self.running:
                time.sleep(0.1)
            if not self.running:
                return False
            # Create new SUB socket using master discovered by outbox thread
            self._create_sub_socket(self._current_master)
            # Wait for TCP connection to establish before announcing readiness.
            # ZMQ connect() is non-blocking; messages published before the handshake
            # completes are silently dropped (slow joiner problem).
            time.sleep(0.5)
            log.info("SUB socket reconnected")
            # Only NOW is the SUB socket ready — tell master we're ready to receive commands
            self.send("sync_ready", {"status": "reconnected"}, wait=False)

        # Skip if not connected
        if self.state != ConnectionState.CONNECTED:
            time.sleep(0.1)
            return False

        try:
            self.sub_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            message = self.sub_socket.recv_json()

            # Check if command is for us
            target = message.get("target", "all")
            if target not in ["all", self.node_id]:
                return False

            cmd_type = message.get("type")
            data = message.get("data", {})
            request_id = message.get("request_id")

            # Dispatch to handler
            if cmd_type in self.command_handlers:
                result = self.command_handlers[cmd_type](data)
                self.send(cmd_type, result, wait=False, request_id=request_id)
                return True

            log.warning(f"No handler for: {cmd_type}")
            return False

        except zmq.Again:
            return False  # Timeout (normal)
        except zmq.ZMQError as e:
            log.warning(f"SUB socket error: {e}")
            self.state = ConnectionState.RECONNECTING
            return False
        except Exception as e:
            log.error(f"Command processing error: {e}", exc_info=True)
            return False

    def _on_master_shutdown(self, data: dict) -> dict:
        """Handle graceful master shutdown notification."""
        log.info("Master shutdown received - will reconnect after delay")
        self._graceful_shutdown = True  # Signal reconnect to wait
        self.state = ConnectionState.RECONNECTING
        return {"status": "acknowledged"}

    def shutdown(self) -> None:
        """Shutdown client and release resources."""
        log.info("NetworkClient shutting down...")
        self.running = False
        # Close both sockets (safe during shutdown - no operations in progress)
        self._close_sub_socket()
        self._close_req_socket()
        self.context.term()
        log.info("NetworkClient shutdown complete")
