"""Network communication utilities for distributed nodes.

This module provides ZMQ-based networking capabilities for coordinating
distributed ethopy nodes. It implements a publish-subscribe pattern for
master-to-remote commands and request-reply for responses.

The module requires pyzmq to be installed:
    pip install pyzmq
"""
import logging
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False
    logging.warning("ZMQ not available - network features disabled")

log = logging.getLogger(__name__)


class NetworkServer:
    """ZMQ server for master node to coordinate remote nodes.

    Master creates one server that accepts connections from multiple remote nodes.
    Uses PUB-SUB pattern for commands and REQ-REP for responses.

    Attributes:
        context (zmq.Context): ZMQ context for socket management
        pub_socket (zmq.Socket): Publisher socket for broadcasting commands
        rep_socket (zmq.Socket): Reply socket for receiving responses
        running (bool): Flag indicating server is running
        connected_nodes (dict): Map of node_id to last heartbeat time
        heartbeat_thread (threading.Thread): Thread monitoring node heartbeats

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

        # Track server start time to identify stale heartbeats from previous sessions
        # Only heartbeats with timestamp >= start_time are considered valid
        self.start_time = time.time()

        self.connected_nodes = {}  # {node_id: last_heartbeat_time}

        # Response routing: store latest response per command type per node
        # Format: {(node_id, command_type): response_dict}
        self._response_cache = {}
        self._response_lock = threading.Lock()

        # Event handlers for asynchronous events from remote nodes
        # Format: {command_type: handler_function}
        self._event_handlers = {}

        log.info(f"Network server started on ports {command_port}, {response_port}")

        # Give ZMQ time to fully bind sockets
        time.sleep(0.5)

        # Start heartbeat monitor thread
        self.heartbeat_thread = threading.Thread(
            target=self._monitor_heartbeats, daemon=True
        )
        self.heartbeat_thread.start()

    def register_event_handler(self, command_type: str, handler: Callable) -> None:
        """Register handler for asynchronous events from remote nodes.

        Args:
            command_type: Type of event to handle (e.g., "log_event")
            handler: Function to call when event received
        """
        self._event_handlers[command_type] = handler
        log.debug(f"Registered event handler for '{command_type}'")

    def send_command(
        self,
        command_type: str,
        data: Optional[Dict] = None,
        target_node: str = "all",
    ) -> None:
        """Send command to remote node(s).

        Args:
            command_type: Type of command (start_session, stop_session, etc.)
            data: Optional data payload
            target_node: Node ID or "all" for broadcast

        """
        message = {
            "target": target_node,
            "type": command_type,
            "data": data or {},
            "timestamp": time.time(),
        }

        self.pub_socket.send_json(message)
        # log.info(f"Sent command '{command_type}' type {command_type}, data {data}  to {target_node}")

    def wait_for_response(
        self,
        timeout: float = 5.0,
        node_id: str = None,
        command_type: str = None
    ) -> Optional[Dict]:
        """Wait for response from remote node.

        This method should ONLY be called by the heartbeat monitor thread.
        Other code should use get_response() to retrieve cached responses.

        Args:
            timeout: Seconds to wait for response
            node_id: Expected node ID (for validation)
            command_type: Expected command type (for validation)

        Returns:
            Response dict or None if timeout

        """
        try:
            self.rep_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            message = self.rep_socket.recv_json()
            self.rep_socket.send_json({"status": "ack"})

            # Update node heartbeat
            msg_node_id = message.get("node_id")
            msg_command_type = message.get("command_type")

            if msg_node_id:
                self.connected_nodes[msg_node_id] = time.time()

                # Cache response for retrieval
                with self._response_lock:
                    cache_key = (msg_node_id, msg_command_type)
                    self._response_cache[cache_key] = message

            return message
        except zmq.Again:
            log.debug("Network response timeout")
            return None

    def get_response(
        self,
        node_id: str,
        command_type: str,
        timeout: float = 5.0,
        clear: bool = True
    ) -> Optional[Dict]:
        """Get cached response from a node for a specific command.

        This polls the response cache that's populated by the heartbeat monitor thread.

        Args:
            node_id: ID of node to get response from
            command_type: Type of command to get response for
            timeout: Seconds to wait for response to appear in cache
            clear: Whether to remove response from cache after retrieval

        Returns:
            Response dict or None if timeout

        """
        start_time = time.time()
        cache_key = (node_id, command_type)

        while time.time() - start_time < timeout:
            with self._response_lock:
                if cache_key in self._response_cache:
                    response = self._response_cache[cache_key]
                    if clear:
                        del self._response_cache[cache_key]
                    return response

            time.sleep(0.05)  # Poll every 50ms

        log.warning(f"Response timeout for {node_id}/{command_type}")
        return None

    def _monitor_heartbeats(self) -> None:
        """Monitor remote node heartbeats by actively receiving them.

        Only accepts heartbeats with timestamp >= server start time to avoid
        treating stale heartbeats from previous sessions as valid connections.
        """
        while self.running:
            try:
                # Actively wait for heartbeats/responses from nodes
                response = self.wait_for_response(timeout=1.0)

                if response:
                    node_id = response.get("node_id")
                    command_type = response.get("command_type")
                    timestamp = response.get("timestamp", 0)

                    # Ignore stale heartbeats from before server start
                    # This prevents race conditions where old heartbeats make us think
                    # a node is connected when it's still reconnecting
                    if timestamp < self.start_time:
                        log.debug(f"Ignoring stale heartbeat from {node_id} (timestamp: {timestamp:.2f} < start: {self.start_time:.2f})")
                        continue

                    # Register node on heartbeat or any response
                    if node_id:
                        if node_id not in self.connected_nodes:
                            log.info(f"New node connected: {node_id}")
                        self.connected_nodes[node_id] = time.time()

                    # Handle asynchronous events (like log_event)
                    if command_type in self._event_handlers:
                        try:
                            handler = self._event_handlers[command_type]
                            handler(response.get("result", {}))
                        except Exception as e:
                            log.error(f"Error in event handler for {command_type}: {e}")

            except Exception as e:
                log.debug(f"Heartbeat monitor error: {e}")

            # Check for node timeouts
            now = time.time()
            dead_nodes = [
                nid for nid, last_time in self.connected_nodes.items()
                if now - last_time > 10.0  # 10 second timeout
            ]

            for node_id in dead_nodes:
                log.warning(f"Node '{node_id}' heartbeat timeout")
                del self.connected_nodes[node_id]

    def shutdown(self) -> None:
        """Shutdown server.

        Sends graceful shutdown notification to all connected nodes
        before closing sockets. This allows nodes to prepare for
        reconnection rather than detecting it via heartbeat timeouts.
        """
        log.info("Shutting down network server...")

        # Send shutdown notification to all connected nodes
        # This provides immediate notification rather than waiting for heartbeat failures
        try:
            self.send_command("master_shutdown", {})
            time.sleep(0.5)  # Give time for message to be sent and processed
        except Exception as e:
            log.warning(f"Could not send shutdown notification: {e}")

        self.running = False
        self.pub_socket.close()
        self.rep_socket.close()
        self.context.term()
        log.info("Network server shutdown complete")


class NetworkClient:
    """ZMQ client for remote node to connect to master.

    Remote nodes subscribe to master's commands and send responses/heartbeats.

    Attributes:
        context (zmq.Context): ZMQ context for socket management
        master_host (str): IP address of master node
        node_id (str): Unique identifier for this node
        sub_socket (zmq.Socket): Subscriber socket for receiving commands
        req_socket (zmq.Socket): Request socket for sending responses
        running (bool): Flag indicating client is running
        command_handlers (dict): Map of command types to handler functions
        heartbeat_thread (threading.Thread): Thread sending periodic heartbeats

    """

    def __init__(
        self,
        master_host: str,
        command_port: int = 5555,
        response_port: int = 5556,
        node_id: str = None,
    ):
        """Initialize network client.

        Args:
            master_host: IP address of master node
            command_port: Port for receiving commands
            response_port: Port for sending responses
            node_id: Unique identifier for this node

        Raises:
            RuntimeError: If ZMQ is not installed

        """
        if not HAVE_ZMQ:
            raise RuntimeError("ZMQ not installed - run: pip install pyzmq")

        self.context = zmq.Context()
        self.master_host = master_host
        self.node_id = node_id or socket.gethostname()

        # IMPORTANT: Store ports for reconnection
        # When master shuts down between sessions and recreates server,
        # remote needs to reconnect to the SAME ports
        self.command_port = command_port
        self.response_port = response_port

        # SUB socket for receiving commands from master
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{master_host}:{command_port}")
        self.sub_socket.setsockopt_string(
            zmq.SUBSCRIBE, ""
        )  # Subscribe to all messages

        # REQ socket for sending responses to master
        self.req_socket = self.context.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{master_host}:{response_port}")

        self.running = True
        self.command_handlers = {}  # {command_type: handler_function}

        # Lock to protect REQ socket from concurrent access
        # REQ sockets must alternate send-recv strictly, so we need to serialize access
        self._req_lock = threading.Lock()

        # Simple flag to indicate connection is broken and needs reconnection
        # Only the heartbeat thread performs reconnection (single owner pattern)
        # Main thread just sets this flag when it detects errors
        self._connection_broken = False

        log.info(f"Network client '{self.node_id}' connected to {master_host}")

        # Try to send initial heartbeat (wait indefinitely for master)
        def _send_initial_heartbeat():
            """Send initial heartbeat and recreate socket on failure."""
            try:
                self._send_response("heartbeat", {"status": "alive"})
            except Exception:
                # Recreate socket and retry
                self.req_socket.close()
                self.req_socket = self.context.socket(zmq.REQ)
                self.req_socket.connect(f"tcp://{self.master_host}:{self.response_port}")
                raise  # Re-raise to trigger retry

        self._wait_for_connection(
            f"master at {self.master_host}",
            _send_initial_heartbeat,
            retry_delay=1.0,
        )

        # Register internal handler for graceful master shutdown
        self.register_handler("master_shutdown", self._handle_master_shutdown)

        # Start heartbeat thread for periodic updates
        self.heartbeat_thread = threading.Thread(
            target=self._send_heartbeats, daemon=True
        )
        self.heartbeat_thread.start()

    def _wait_for_connection(
        self,
        operation_name: str,
        retry_func: Callable,
        retry_delay: float = 1.0,
        log_interval: float = 30.0,
    ) -> None:
        """Wait for connection with time-based progress logging.

        Args:
            operation_name: Description of operation (e.g., "master at 192.168.1.100")
            retry_func: Function to retry; should raise exception on failure
            retry_delay: Seconds to wait between retry attempts
            log_interval: Seconds between progress log messages

        """
        start_time = time.time()
        last_log_time = start_time

        log.info(f"Waiting for {operation_name}...")

        while True:
            try:
                retry_func()
                elapsed = time.time() - start_time
                log.info(f"✓ Connected to {operation_name} after {elapsed:.1f}s")
                break
            except Exception as e:
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    elapsed = current_time - start_time
                    log.info(f"Still waiting... ({elapsed:.0f}s elapsed)")
                    last_log_time = current_time
                time.sleep(retry_delay)

    def register_handler(self, command_type: str, handler: Callable) -> None:
        """Register command handler.

        Args:
            command_type: Type of command to handle
            handler: Function to call when command received

        """
        self.command_handlers[command_type] = handler
        log.debug(f"Registered handler for '{command_type}'")

    def process_commands(self, timeout: float = 1.0) -> bool:
        """Poll for commands and process them.

        Args:
            timeout: Seconds to wait for command

        Returns:
            True if command processed, False if timeout

        """
        # Skip if connection is broken - let heartbeat thread handle reconnection
        if self._connection_broken:
            time.sleep(0.1)
            return False

        try:
            self.sub_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            message = self.sub_socket.recv_json()

            # Check if command is for us
            target = message.get("target", "all")
            if target not in ["all", self.node_id]:
                return False

            command_type = message.get("type")
            data = message.get("data", {})

            # log.info(f"Received command: {command_type}")

            # Call registered handler
            if command_type in self.command_handlers:
                result = self.command_handlers[command_type](data)

                # Send response
                self._send_response(command_type, result)
                return True
            log.warning(f"No handler for command type: {command_type}")
            return False

        except zmq.Again:
            # Timeout - no command received (this is NORMAL)
            return False
        except zmq.ZMQError as e:
            # Connection broken - master shutdown or network issue
            # Don't reconnect directly - just mark as broken and let heartbeat thread handle it
            log.warning(f"Network error: {e}")
            self._connection_broken = True
            return False
        except Exception as e:
            log.error(f"Error processing command: {e}", exc_info=True)
            return False

    def _serialize_result(self, result: Any) -> Any:
        """Serialize result for JSON transmission.

        Converts non-JSON-serializable objects (like Port, numpy types) to
        JSON-compatible formats.

        Args:
            result: Result to serialize

        Returns:
            JSON-serializable version of result
        """
        # Handle Port objects from ethopy.core.interface
        if hasattr(result, '__dict__') and hasattr(result, 'port'):
            return result.__dict__

        # Handle tuples (e.g., in_position returns (Port, int, int))
        elif isinstance(result, tuple):
            return tuple(self._serialize_result(item) for item in result)

        # Handle lists
        elif isinstance(result, list):
            return [self._serialize_result(item) for item in result]

        # Handle dicts (convert numpy keys to native Python types)
        elif isinstance(result, dict):
            serialized_dict = {}
            for k, v in result.items():
                # Convert numpy keys to native Python types
                if hasattr(k, 'item'):  # numpy scalar key
                    key = k.item()
                else:
                    key = k
                serialized_dict[key] = self._serialize_result(v)
            return serialized_dict

        # Handle numpy types
        elif hasattr(result, 'item'):  # numpy scalar
            return result.item()

        # Already JSON-serializable
        else:
            return result

    def _send_response(self, command_type: str, result: Any, timeout: float = 2.0) -> None:
        """Send response to master.

        Args:
            command_type: Type of command being responded to
            result: Result from command handler
            timeout: Seconds to wait for acknowledgment

        Raises:
            zmq.Again: If no acknowledgment received within timeout

        """
        # Serialize result to JSON-compatible format
        serialized_result = self._serialize_result(result)

        message = {
            "node_id": self.node_id,
            "command_type": command_type,
            "result": serialized_result,
            "timestamp": time.time(),
        }

        # Lock REQ socket to prevent concurrent send-recv cycles
        # REQ sockets must alternate send-recv, so this ensures atomicity
        with self._req_lock:
            # log.info(f"🔌 CLIENT SEND: {command_type} - {serialized_result}")
            self.req_socket.send_json(message)
            # Wait for ack with timeout
            self.req_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            ack = self.req_socket.recv_json()
            # log.info(f"🔌 CLIENT RECV ACK: {ack}")

    def _reconnect(self) -> None:
        """Reconnect to master after connection loss.

        This method is called when the master shuts down its server
        between sessions. The remote node will continuously retry
        connecting until the master comes back online.

        IMPORTANT:
        - This allows remote nodes to run for days, waiting for master to start/restart
        - Only called from heartbeat thread (single owner pattern)
        - No locks needed since only one thread calls this
        """
        log.warning(f"Lost connection to master at {self.master_host}")

        def _perform_reconnection():
            """Reconnect sockets and test connection with heartbeat."""
            # Close old broken sockets
            # IMPORTANT: Must close before creating new ones to avoid
            # "socket already in use" errors
            # Use _req_lock to prevent any pending heartbeat sends during close
            with self._req_lock:
                try:
                    self.sub_socket.close()
                    self.req_socket.close()
                except:
                    pass  # Ignore errors if already closed

                # Create fresh sockets
                self.sub_socket = self.context.socket(zmq.SUB)
                self.sub_socket.connect(f"tcp://{self.master_host}:{self.command_port}")
                self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

                self.req_socket = self.context.socket(zmq.REQ)
                self.req_socket.connect(f"tcp://{self.master_host}:{self.response_port}")

                # Test connection with heartbeat
                # IMPORTANT: This verifies master is actually listening,
                # not just that we connected (ZMQ connects even if no server)
                self.req_socket.send_json({
                    "node_id": self.node_id,
                    "command_type": "heartbeat",
                    "result": {"status": "reconnected"},
                    "timestamp": time.time()
                })
                self.req_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2s timeout
                ack = self.req_socket.recv_json()

        self._wait_for_connection(
            f"master at {self.master_host}",
            _perform_reconnection,
            retry_delay=2.0,
        )

        # Reset connection broken flag - reconnection complete
        self._connection_broken = False
        log.info("Reconnection complete")

        # Note: We don't send a post-reconnection heartbeat here to avoid race conditions
        # The regular heartbeat thread (running every 5s) will naturally send the next heartbeat
        # The master filters stale heartbeats using server.start_time, ensuring only fresh ones count

    def _handle_master_shutdown(self, data: Dict) -> Dict:
        """Handle graceful master shutdown notification.

        When master sends shutdown command before closing, this provides
        immediate notification rather than waiting for heartbeat failures.

        Args:
            data: Shutdown data (currently unused)

        Returns:
            Acknowledgment dict
        """
        log.info("Master shutting down gracefully - marking connection as broken")
        # Mark connection as broken - heartbeat thread will reconnect immediately
        self._connection_broken = True
        return {"status": "acknowledged"}

    def _send_heartbeats(self) -> None:
        """Send periodic heartbeats to master.

        Monitors connection health and triggers reconnection when needed.
        This is the ONLY place that calls _reconnect() (single owner pattern).

        Reconnection is triggered by:
        1. Heartbeat failure (network error)
        2. Connection broken flag set by main thread (command processing error)
        3. Graceful shutdown notification from master
        """
        while self.running:
            # Check if reconnection needed (set by main thread or previous heartbeat failure)
            if self._connection_broken:
                log.info("Connection broken detected, reconnecting...")
                self._reconnect()
                # Flag is reset inside _reconnect() after successful reconnection
                continue  # Skip this heartbeat iteration, will send next cycle

            # Send normal heartbeat
            try:
                self._send_response("heartbeat", {"status": "alive"})
            except Exception as e:
                log.warning(f"Heartbeat failed: {e}")
                # Mark connection as broken and reconnect on next iteration
                self._connection_broken = True
                # Don't sleep here - reconnect immediately on next iteration
                continue

            time.sleep(2)

    def shutdown(self) -> None:
        """Shutdown client."""
        self.running = False
        self.sub_socket.close()
        self.req_socket.close()
        self.context.term()
        log.info("Network client shutdown")
