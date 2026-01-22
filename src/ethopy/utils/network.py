"""Network communication utilities for distributed nodes.

This module provides ZMQ-based networking capabilities for coordinating
distributed ethopy nodes.

Architecture
------------
Master (NetworkServer):
    - PUB socket: broadcasts commands to all remotes
    - REP socket: receives responses/heartbeats from remotes

Remote (NetworkClient):
    - SUB socket: receives commands from master
    - REQ socket: sends responses/heartbeats to master

Threading Model
---------------
NetworkClient uses a single "outbox thread" that owns the REQ socket:
    - Sends queued messages (responses, events)
    - Sends heartbeats when queue is empty
    - Handles reconnection (owns socket lifecycle)

Main thread:
    - Receives commands via SUB socket (process_commands)
    - Queues responses via send() method

This eliminates lock contention and prevents deadlocks during reconnection.

The module requires pyzmq to be installed:
    pip install pyzmq
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


# =============================================================================
# NetworkServer (Master side)
# =============================================================================


class NetworkServer:
    """ZMQ server for master node to coordinate remote nodes.

    The server broadcasts commands to remotes via PUB socket and receives
    responses/heartbeats via REP socket.

    Attributes:
        context: ZMQ context for socket management
        pub_socket: Publisher socket for broadcasting commands
        rep_socket: Reply socket for receiving responses
        running: Flag indicating server is running
        connected_nodes: Map of node_id to last heartbeat time
        ready_nodes: Set of nodes that completed SUB socket sync

    Example:
        server = NetworkServer(command_port=5555, response_port=5556)
        server.send_command("start", {"param": 1}, target_node="rpi_camera_1")
        response = server.get_response("rpi_camera_1", "start", timeout=5.0)
        server.shutdown()
    """

    def __init__(self, command_port: int = 5555, response_port: int = 5556):
        """Initialize network server.

        Args:
            command_port: Port for publishing commands to remotes
            response_port: Port for receiving responses from remotes

        Raises:
            RuntimeError: If ZMQ is not installed
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

        # Node tracking
        self.connected_nodes: Dict[str, float] = {}  # {node_id: last_heartbeat_time}
        self.node_ips: Dict[str, str] = {}  # {node_id: client_ip}
        self.ready_nodes: set = set()  # Nodes that completed SUB socket sync

        # Response cache for async retrieval
        self._response_cache: Dict[tuple, dict] = {}  # {(node_id, command_type): response}
        self._response_lock = threading.Lock()

        # Event handlers for async events from remotes (e.g., log_event)
        self._event_handlers: Dict[str, Callable] = {}

        log.info(f"NetworkServer started on ports {command_port} (cmd), {response_port} (resp)")

        # Give ZMQ time to bind sockets before accepting connections
        time.sleep(0.3)

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

    def send_command(
        self,
        command_type: str,
        data: Optional[Dict] = None,
        target_node: str = "all",
    ) -> None:
        """Send command to remote node(s) via PUB socket.

        Args:
            command_type: Command identifier (e.g., "init_interface", "call_method")
            data: Command payload
            target_node: Node ID or "all" for broadcast
        """
        message = {
            "target": target_node,
            "type": command_type,
            "data": data or {},
            "timestamp": time.time(),
        }
        self.pub_socket.send_json(message)
        log.debug(f"Sent command '{command_type}' to {target_node}")

    def get_response(
        self,
        node_id: str,
        command_type: str,
        timeout: float = 5.0,
        clear: bool = True
    ) -> Optional[Dict]:
        """Wait for and retrieve cached response from a node.

        The monitor thread continuously receives responses and caches them.
        This method polls the cache until the expected response arrives.

        Args:
            node_id: ID of node to get response from
            command_type: Type of command to get response for
            timeout: Max seconds to wait
            clear: Remove response from cache after retrieval

        Returns:
            Response dict or None if timeout
        """
        cache_key = (node_id, command_type)
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._response_lock:
                if cache_key in self._response_cache:
                    response = self._response_cache[cache_key]
                    if clear:
                        del self._response_cache[cache_key]
                    return response
            time.sleep(0.05)

        log.warning(f"Response timeout: {node_id}/{command_type}")
        return None

    def is_node_ready(self, node_id: str) -> bool:
        """Check if node is fully connected and ready for PUB commands.

        A node is ready when:
        1. It has sent a heartbeat (in connected_nodes)
        2. It has completed SUB socket sync (in ready_nodes)

        Args:
            node_id: ID of node to check

        Returns:
            True if node is ready to receive commands
        """
        return node_id in self.connected_nodes and node_id in self.ready_nodes

    def wait_for_node(self, node_id: str, timeout: float = 15.0) -> None:
        """Wait for specific node to connect and be ready.

        Blocks until the node is fully ready (heartbeat received + SUB socket synced).
        Provides informative error if a different node connects instead.

        Args:
            node_id: ID of node to wait for
            timeout: Maximum seconds to wait

        Raises:
            TimeoutError: If node doesn't connect in time, with details about
                what nodes DID connect (helps debug node_id mismatches)
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.is_node_ready(node_id):
                log.info(f"Node '{node_id}' is ready")
                return
            time.sleep(0.5)

        # Timeout - provide informative error about what actually connected
        connected = list(self.connected_nodes.keys())
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
        - Heartbeats: updates connected_nodes
        - sync_ready: marks node as ready for PUB commands
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

                # Update node tracking
                if node_id:
                    client_ip = message.get("client_ip", "unknown")
                    if node_id not in self.connected_nodes:
                        log.info(f"Node connected: {node_id} (IP: {client_ip})")
                    self.connected_nodes[node_id] = time.time()
                    self.node_ips[node_id] = client_ip

                # Handle sync_ready - node's SUB socket is established
                if msg_type == "sync_ready":
                    self.ready_nodes.add(node_id)
                    log.debug(f"Node ready: {node_id} (SUB socket synced)")
                    continue

                # Handle heartbeats (just tracking, already done above)
                if msg_type == "heartbeat":
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
                    self._response_cache[(node_id, msg_type)] = message

            except zmq.Again:
                pass  # Timeout, check for dead nodes
            except Exception as e:
                if self.running:
                    log.debug(f"Monitor error: {e}")

            # Check for dead nodes (no heartbeat in 10s)
            self._check_dead_nodes()

    def _check_dead_nodes(self) -> None:
        """Remove nodes that haven't sent heartbeat recently."""
        now = time.time()
        dead = [nid for nid, last in self.connected_nodes.items() if now - last > 10.0]
        for node_id in dead:
            log.warning(f"Node timeout: {node_id}")
            del self.connected_nodes[node_id]
            self.node_ips.pop(node_id, None)
            self.ready_nodes.discard(node_id)

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


class NetworkClient:
    """ZMQ client for remote nodes to communicate with master.

    Connects to a NetworkServer, receives commands via SUB socket,
    and sends responses/heartbeats via REQ socket. Handles automatic
    reconnection when master restarts.

    Attributes:
        node_id: Unique identifier for this node
        state: Current connection state (DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING)
        command_handlers: Map of command types to handler functions

    Example:
        client = NetworkClient(
            master_host="192.168.1.10",  # or ["192.168.1.10", "192.168.1.20"]
            node_id="rpi_camera_1"
        )
        client.register_handler("start", handle_start)
        client.sync_with_master()

        while True:
            client.process_commands(timeout=0.1)
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
        """Initialize network client.

        Args:
            master_host: IP address(es) of master node(s). Formats:
                - Single IP: "192.168.1.10"
                - List: ["192.168.1.10", "192.168.1.20"]
                - Range: "192.168.1.10-30" (expands to .10 through .30)
            command_port: Port for receiving commands (master's PUB)
            response_port: Port for sending responses (master's REP)
            node_id: Unique identifier (defaults to hostname)

        Raises:
            RuntimeError: If ZMQ is not installed
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
        log.info(f"NetworkClient '{self.node_id}' initialized")
        log.info(f"  Master candidates: {len(self._master_candidates)}")

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
        self._ready = False   # Set by sync_with_master() to enable outbox thread

        # Establish initial connection (blocks until master found)
        self._connect()

        # Handle graceful master shutdown (triggers reconnection)
        self.register_handler("master_shutdown", self._on_master_shutdown)

        # Outbox thread: sends queued messages and heartbeats
        self._outbox_thread = threading.Thread(target=self._outbox_loop, daemon=True)
        self._outbox_thread.start()

        log.info(f"NetworkClient '{self.node_id}' connected")

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

        log.info("Graceful shutdown - waiting 2s for master to close...")
        time.sleep(2.0)
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
                self._do_send("sync_ready", {"status": "reconnected"})

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

    def _do_send(self, msg_type: str, data: Any) -> None:
        """Send message via REQ socket (outbox thread only).

        Args:
            msg_type: Message type identifier
            data: Message payload

        Raises:
            Exception: If send or recv fails
        """
        message = {
            "node_id": self.node_id,
            "client_ip": self.client_ip,
            "command_type": msg_type,
            "result": self._serialize(data),
            "timestamp": time.time(),
        }

        self.req_socket.send_json(message)
        self.req_socket.setsockopt(zmq.RCVTIMEO, int(self.SEND_TIMEOUT * 1000))
        self.req_socket.recv_json()  # Wait for ACK

    def _serialize(self, data: Any) -> Any:
        """Convert data to JSON-serializable format.

        Handles Port objects, numpy types, nested structures.
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
            # Wait until ready
            if not self._ready:
                time.sleep(0.1)
                continue

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
                    self._do_send(msg.msg_type, msg.data)
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

    def sync_with_master(self) -> None:
        """Synchronize with master to signal readiness.

        Call this after registering all command handlers.
        Sends sync_ready message to master confirming SUB socket is established
        and client is ready to receive commands.
        """
        self._ready = True

        # Queue sync_ready message
        self.send("sync_ready", {"status": "initial"}, wait=False)

        log.info(f"Client '{self.node_id}' synced with master")

    def send(self, msg_type: str, data: dict, wait: bool = True, timeout: float = 5.0) -> None:
        """Queue a message to be sent to master.

        Args:
            msg_type: Message type identifier
            data: Message payload
            wait: If True, block until sent
            timeout: Max seconds to wait (if wait=True)

        Raises:
            TimeoutError: If wait=True and send times out
            Exception: If send fails
        """
        if wait:
            event = threading.Event()
            msg = OutboxMessage(msg_type, data, event)
            self._outbox.put(msg)

            if not event.wait(timeout):
                raise TimeoutError(f"Send timeout: {msg_type}")
            if msg.error:
                raise msg.error
        else:
            self._outbox.put(OutboxMessage(msg_type, data))

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
            log.info("SUB socket reconnected")

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

            # Dispatch to handler
            if cmd_type in self.command_handlers:
                result = self.command_handlers[cmd_type](data)
                self.send(cmd_type, result, wait=False)
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


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# Keep old method names working
def _send_response(self, command_type: str, result: Any, timeout: float = 2.0) -> None:
    """Backward compatible alias for send()."""
    self.send(command_type, result, wait=True, timeout=timeout)


NetworkClient._send_response = _send_response
