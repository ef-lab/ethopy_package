# Distributed Networking Architecture

This document explains the distributed networking system in ethopy, which enables running interfaces (cameras, sensors, etc.) on remote computers while the main experiment runs on a master computer.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
  - [network.py - Network Communication](#networkpy---network-communication)
  - [interface_proxy.py - Remote Interface Control](#interface_proxypy---remote-interface-control)
  - [interface.py - Interface Integration](#interfacepy---interface-integration)
- [Communication Flow](#communication-flow)
- [Threading & Synchronization](#threading--synchronization)
- [Reconnection System](#reconnection-system)
- [Heartbeat Mechanism](#heartbeat-mechanism)
- [Session Lifecycle](#session-lifecycle)
- [Common Issues & Solutions](#common-issues--solutions)

---

## Overview

The distributed networking system allows you to:
- Run cameras/sensors on remote computers (e.g., Raspberry Pis)
- Coordinate multiple distributed nodes from a single master
- Automatically reconnect when sessions end and restart
- Synchronize session metadata across all nodes
- Forward hardware events back to master for logging

**Key Design Goals:**
1. **Persistence**: Remote nodes run continuously, waiting for master across multiple sessions
2. **Robustness**: Automatic reconnection when master restarts between sessions
3. **Thread Safety**: Multiple threads access sockets safely using locks
4. **Transparency**: Remote interfaces appear as local objects via proxy pattern

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Master Computer                          │
│                                                                 │
│  ┌──────────────┐         ┌──────────────────────────────────┐  │
│  │ Experiment   │────────▶│ Interface                        │  │
│  │ (main loop)  │         │                                  │  │
│  └──────────────┘         │  ┌──────────────┐                │  │
│                           │  │ NetworkServer│                │  │
│                           │  │  (pub/rep)   │                │  │
│                           │  └──────┬───────┘                │  │
│                           │         │                        │  │
│                           │  ┌──────▼───────────────────┐    │  │
│                           │  │ InterfaceProxy           │    │  │
│                           │  │ (appears like local obj) │    │  │
│                           │  └──────────────────────────┘    │  │
│                           └──────────────────────────────────┘  │
└───────────────────────────────────┬─────────────────────────────┘
                                    │ ZMQ over TCP/IP
                                    │ (commands + responses)
┌───────────────────────────────────▼──────────────────────────────┐
│                     Remote Computer (Raspberry Pi)               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ RemoteInterfaceNode                                        │  │
│  │                                                            │  │
│  │  ┌──────────────┐        ┌────────────────┐                │  │
│  │  │ NetworkClient│───────▶│ Real Interface │                │  │
│  │  │  (sub/req)   │        │  (Camera, etc) │                │  │
│  │  └──────────────┘        └────────────────┘                │  │
│  │                                                            │  │
│  │  Runs continuously, waiting for master commands...         │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Communication Patterns:**
- **PUB/SUB**: Master publishes commands → Remotes subscribe and receive
- **REQ/REP**: Remotes send responses/heartbeats → Master replies with ACK

---

## Core Modules

### network.py - Network Communication

Provides low-level ZMQ networking for master-remote coordination.

#### NetworkServer (Master Side)

**Purpose**: Coordinate multiple remote nodes from master computer

**Key Components:**

```python
class NetworkServer:
    def __init__(self, command_port=5555, response_port=5556):
        # PUB socket - broadcasts commands to all remotes
        self.pub_socket = zmq.PUB

        # REP socket - receives responses/heartbeats from remotes
        self.rep_socket = zmq.REP

        # Track when server started (for filtering stale heartbeats)
        self.start_time = time.time()

        # Track which nodes are connected
        self.connected_nodes = {}  # {node_id: last_heartbeat_time}

        # Cache responses for retrieval by command handlers
        self._response_cache = {}  # {(node_id, cmd_type): response}
```

**Key Methods:**

1. **send_command(command_type, data, target_node)**
   - Broadcasts command to specific node or all nodes
   - Uses PUB socket (fire-and-forget)
   - Commands: `init_interface`, `call_method`, `cleanup`, `master_shutdown`

2. **get_response(node_id, command_type, timeout)**
   - Polls response cache populated by heartbeat monitor thread
   - Blocks until response arrives or timeout
   - Returns response dict or None

3. **_monitor_heartbeats()** (runs in background thread)
   - Continuously receives heartbeats/responses on REP socket
   - Filters stale heartbeats using `timestamp >= server.start_time`
   - Updates `connected_nodes` dict
   - Caches responses for retrieval

4. **shutdown()**
   - Sends `master_shutdown` command to all nodes (graceful)
   - Waits 0.5s for message delivery
   - Closes sockets and terminates context

**Thread Safety:**
- `_response_lock`: Protects response cache from concurrent access

**Why Filter Stale Heartbeats?**
When master restarts, remotes might send heartbeats they queued before reconnection. Without filtering, master would think the remote is ready before it actually finishes reconnecting, causing timeouts when sending commands.

---

#### NetworkClient (Remote Side)

**Purpose**: Connect to master and respond to commands

**Key Components:**

```python
class NetworkClient:
    def __init__(self, master_host, command_port, response_port, node_id):
        # SUB socket - receives commands from master
        self.sub_socket = zmq.SUB

        # REQ socket - sends responses/heartbeats to master
        self.req_socket = zmq.REQ

        # Command handlers registry
        self.command_handlers = {}  # {command_type: handler_func}

        # Thread synchronization locks
        self._req_lock = threading.Lock()        # Protects REQ socket send-recv
        self._reconnect_lock = threading.Lock()  # Protects socket recreation
```

**Key Methods:**

1. **process_commands(timeout)** (called by main loop)
   - Polls SUB socket for incoming commands
   - Checks if command is for this node (`target == node_id` or `"all"`)
   - Calls registered handler and sends response
   - **Acquires `_reconnect_lock`** to prevent using sockets during reconnection

2. **_send_response(command_type, result, timeout)**
   - Sends response on REQ socket
   - **Acquires `_req_lock`** to ensure strict send-recv alternation
   - REQ sockets MUST alternate: send → recv → send → recv...
   - Waits for ACK from master

3. **_reconnect()** (called when connection lost)
   - **Acquires `_reconnect_lock`** to prevent concurrent reconnections
   - Closes old broken sockets
   - Creates fresh SUB and REQ sockets
   - Tests connection by sending heartbeat
   - Retries every 2 seconds until master comes back
   - See [Reconnection System](#reconnection-system) for details

4. **_send_heartbeats()** (runs in background thread)
   - Sends heartbeat every 5 seconds
   - **Checks `_reconnect_lock`** before sending (skips if reconnection in progress)
   - Triggers reconnection on failure
   - See [Heartbeat Mechanism](#heartbeat-mechanism) for details

**Why Two Locks?**

1. **`_req_lock`**: REQ sockets have strict send-recv state machine. If two threads both call `send()`, the second will fail. This lock serializes all REQ socket operations.

2. **`_reconnect_lock`**: When reconnecting, we close/recreate sockets. If another thread tries to use the socket during this time, we get segmentation faults. This lock prevents any socket access during reconnection.

**Why Not Just One Lock?**
- `_send_response()` is called frequently (every command, every heartbeat)
- If we used `_reconnect_lock` for all sends, it would be held too often
- By separating concerns, heartbeat can skip gracefully when reconnection is in progress

---

### interface_proxy.py - Remote Interface Control

Provides high-level proxy objects that make remote interfaces appear local.

#### InterfaceProxy (Master Side)

**Purpose**: Transparently forward method calls to remote interface

**How It Works:**

```python
# On master, create proxy that looks like a real Interface
camera_proxy = InterfaceProxy(
    Interface,                    # Class to run remotely
    remote_host="192.168.1.100",
    remote_setup_conf_idx=0,
    node_id="rpi_camera_1"
)

# Initialize with session metadata
camera_proxy.init_local(exp, beh)

# Now call methods as if it were local - they're forwarded to remote!
camera_proxy.start_recording()  # Actually runs on Raspberry Pi
camera_proxy.stop_recording()
```

**Key Components:**

```python
class InterfaceProxy:
    def __init__(self, interface_class, remote_host, ...):
        # Create NetworkServer for communication
        self.server = NetworkServer(command_port, response_port)

        # Register event handler for events from remote
        self.server.register_event_handler("log_event", self._handle_log_event)

    def init_local(self, exp, beh):
        # Wait for remote to connect (fresh heartbeat)
        # Send init_interface command WITH session metadata
        # Remote will create interface with correct animal_id, session, etc.

    def __getattr__(self, name):
        # Intercept ALL method calls and forward to remote
        def remote_method(*args, **kwargs):
            self.server.send_command("call_method", {
                "method": name,
                "args": args,
                "kwargs": kwargs
            }, target_node=self.node_id)

            response = self.server.get_response(...)
            return response["return_value"]

        return remote_method
```

**Why Wait for Fresh Heartbeat?**
When reconnecting, the remote's old heartbeat might arrive before reconnection completes. By filtering stale heartbeats (`timestamp >= server.start_time`), we ensure the remote is truly ready before sending initialization commands.

**Why Check `server.running`?**
When experiment ends, `interface.stop_cameras()` is called. This calls `camera_proxy.release()` and `camera_proxy.cleanup()`. But if the server already shut down, these commands would fail with "Socket operation on non-socket". By checking `server.running`, we skip gracefully.

---

#### RemoteInterfaceNode (Remote Side)

**Purpose**: Run the actual interface and respond to master commands

**Key Components:**

```python
class RemoteInterfaceNode:
    def __init__(self, master_host, node_id):
        # Connect to master
        self.client = NetworkClient(master_host, ...)

        # Register command handlers
        self.client.register_handler("init_interface", self._handle_init_interface)
        self.client.register_handler("call_method", self._handle_call_method)
        self.client.register_handler("cleanup", self._handle_cleanup)

    def _handle_init_interface(self, data):
        # Extract session metadata (animal_id, session, start_time)
        # Create logger and set metadata BEFORE creating interface
        # Create interface (this initializes camera with correct metadata)
        self.interface = interface_class(exp=proxy, beh=proxy)

    def _handle_call_method(self, data):
        # Execute method on real interface
        method = getattr(self.interface, data["method"])
        result = method(*data["args"], **data["kwargs"])
        return {"status": "success", "return_value": result}

    def run(self):
        # Main loop - process commands forever
        while True:
            self.client.process_commands(timeout=0.1)
```

**Why Session Metadata Before Interface Creation?**
Cameras need `animal_id`, `session`, and `start_time` in their constructor to create the correct video filename. If we create the interface first, the camera gets wrong/missing metadata. By setting logger metadata first, the interface constructor sees correct values.

---

#### BehaviorProxy

**Purpose**: Forward hardware events from remote back to master

**Example Flow:**

```
Remote: Button pressed → interface.beh.log_activity({"port": 1, ...})
                      ↓
             BehaviorProxy intercepts
                      ↓
        Sends log_event to master via REQ socket
                      ↓
Master: NetworkServer receives log_event
                      ↓
        Calls InterfaceProxy._handle_log_event()
                      ↓
        Forwards to real behavior.log_activity()
                      ↓
        Event logged in database ✓
```

**Why Use REQ Socket for Events?**
Events must be reliably delivered. REQ socket blocks until ACK received, ensuring master got the event before remote continues.

---

### interface.py - Interface Integration

Main interface class that supports both local and remote cameras.

#### Key Methods:

```python
class Interface:
    def setup_cameras(self):
        """Setup both local and remote cameras."""
        self._setup_local_camera()   # Camera on same computer
        self._setup_remote_cameras() # Cameras on remote computers

    def _setup_remote_cameras(self):
        # Read remote camera configs from database
        # For each remote:
        #   - Create InterfaceProxy
        #   - Call init_local(exp, beh) to send session metadata
        #   - Remote creates interface+camera with correct metadata

    def stop_cameras(self):
        # Stop local camera
        if self.camera:
            self.camera.stop_recording()
            self.camera.release()

        # Stop remote cameras (with server.running check)
        for camera_proxy in self.remote_cameras:
            camera_proxy.release()  # Checks server.running internally
            camera_proxy.cleanup()  # Checks server.running internally
```

**Remote Camera Configuration (Database):**

```sql
-- Table: SetupConfiguration.RemoteCamera
setup_conf_idx | remote_host     | node_id         | command_port | response_port
0              | 192.168.1.100   | rpi_camera_1    | 5557         | 5558
0              | 192.168.1.101   | rpi_camera_2    | 5559         | 5560
```

---

## Communication Flow

### Session Start (Normal Flow)

```
1. Master starts experiment
   └─> interface.setup_cameras()
       └─> _setup_remote_cameras()
           └─> InterfaceProxy(...) created
               └─> NetworkServer started on ports 5557/5558

2. Remote node (already running) detects master
   └─> NetworkClient sends initial heartbeat
       └─> Master receives heartbeat (timestamp >= start_time)
           └─> Adds node to connected_nodes dict

3. Master waits for remote to be ready
   └─> init_local(exp, beh) called
       └─> Polls connected_nodes until node appears
           └─> Sends init_interface command with session metadata

4. Remote receives init_interface
   └─> Creates logger with animal_id, session, start_time
       └─> Creates interface (camera gets correct metadata)
           └─> Returns "initialized" response

5. Master receives response
   └─> Camera proxy ready for method calls
       └─> Experiment continues normally

6. During session
   ├─> Master calls camera_proxy.start_recording()
   │   └─> Forwarded as call_method command
   │       └─> Remote executes camera.start_recording()
   │           └─> Returns success response
   │
   └─> Remote sends heartbeat every 5 seconds
       └─> Master updates connected_nodes timestamp
```

### Session End & Restart (Reconnection Flow)

```
1. Master ends session
   └─> interface.cleanup()
       └─> interface.stop_cameras()
           ├─> camera_proxy.release() (checks server.running)
           └─> camera_proxy.cleanup() (checks server.running)

2. Master shuts down server
   └─> server.shutdown()
       ├─> Sends "master_shutdown" command to all nodes
       ├─> Waits 0.5s for delivery
       └─> Closes sockets

3. Remote receives "master_shutdown"
   └─> _handle_master_shutdown() sets flag
       └─> Heartbeat thread triggers immediate reconnection

4. Remote enters reconnection loop
   └─> _reconnect() called
       ├─> Acquires _reconnect_lock (blocks all socket access)
       ├─> Closes old broken sockets
       ├─> Creates fresh sockets
       ├─> Tests connection with heartbeat
       │   └─> Fails because master not running yet
       ├─> Waits 2 seconds
       └─> Retries...

5. Master starts new session (30 seconds later)
   └─> New NetworkServer created
       └─> server.start_time = time.time() (filters old heartbeats)

6. Remote retry succeeds
   └─> Heartbeat ACK received from new server
       ├─> Releases _reconnect_lock
       └─> Heartbeat sent with timestamp = now

7. Master receives fresh heartbeat
   └─> Checks timestamp >= server.start_time ✓
       └─> Adds node to connected_nodes

8. Master sends init_interface for new session
   └─> Remote receives command
       ├─> Cleans up old interface
       ├─> Creates new interface with new session metadata
       └─> Returns "initialized"

9. New session continues normally
```

---

## Threading & Synchronization

### Threads in the System

**Master Side (per remote camera):**
1. **Main Thread**: Runs experiment loop, calls camera methods
2. **Heartbeat Monitor Thread**: Receives heartbeats/responses on REP socket

**Remote Side:**
1. **Main Thread**: Runs `process_commands()` loop, executes interface methods
2. **Heartbeat Thread**: Sends heartbeat every 5 seconds on REQ socket

### Locks & Their Purposes

#### `_req_lock` (NetworkClient)

**Purpose**: Serialize access to REQ socket

**Why Needed:**
- REQ sockets have strict state machine: send → recv → send → recv
- If two threads both call `send()`, second will fail with "Operation cannot be accomplished in current state"
- Example race condition:
  ```
  Thread 1 (main): Sends response for "call_method"
  Thread 2 (heartbeat): Sends heartbeat
  Thread 1: Tries to recv ACK ← FAILS! Socket expects Thread 2's recv first
  ```

**Used By:**
- `_send_response()` - wraps entire send-recv cycle with lock
- `_reconnect()` - holds lock while recreating REQ socket

**Lock Scope:**
```python
with self._req_lock:
    self.req_socket.send_json(message)
    ack = self.req_socket.recv_json()  # Must receive before releasing
```

---

#### `_reconnect_lock` (NetworkClient)

**Purpose**: Prevent socket access during reconnection

**Why Needed:**
- When reconnecting, we close old sockets and create new ones
- If another thread tries to use socket during this time: **segmentation fault**
- Example crash scenario:
  ```
  Thread 1 (reconnect): Closes req_socket
  Thread 2 (heartbeat): Calls req_socket.send_json() ← SEGFAULT!
  Thread 1 (reconnect): Creates new req_socket
  ```

**Used By:**
- `_reconnect()` - holds lock for entire reconnection (socket close + recreation)
- `process_commands()` - **tries** to acquire (skips if locked)
- `_send_heartbeats()` - **tries** to acquire (skips if locked)

**Non-Blocking Pattern:**
```python
# Don't wait if reconnection in progress - just skip this iteration
if not self._reconnect_lock.acquire(blocking=False):
    time.sleep(1.0)
    continue  # Try again next iteration

try:
    # Use sockets safely
    ...
finally:
    self._reconnect_lock.release()
```

**Why Non-Blocking?**
If we used `with self._reconnect_lock:` (blocking), the heartbeat thread would freeze waiting for reconnection to finish. By using `acquire(blocking=False)`, we allow threads to skip gracefully and retry later.

---

#### `_response_lock` (NetworkServer)

**Purpose**: Protect response cache from concurrent access

**Why Needed:**
- Heartbeat monitor thread writes to `_response_cache`
- Main thread reads from `_response_cache` via `get_response()`
- Without lock: race condition on dict modification

**Used By:**
- `wait_for_response()` - writes to cache after receiving response
- `get_response()` - reads from cache when polling for response

---

### Lock Acquisition Order

**Critical Rule**: Always acquire locks in same order to prevent deadlock

**Current Order:**
```
_reconnect_lock (outer) → _req_lock (inner)
```

**Example in `_reconnect()`:**
```python
with self._reconnect_lock:        # Acquire outer lock first
    with self._req_lock:           # Then inner lock
        self.req_socket.close()
        self.req_socket = create_new_socket()
```

**Why This Order?**
- Reconnection needs exclusive access to everything (no commands, no heartbeats)
- REQ lock is more granular (just socket operations)
- If reversed, could deadlock:
  ```
  Thread 1: Holds _req_lock, waits for _reconnect_lock
  Thread 2: Holds _reconnect_lock, waits for _req_lock ← DEADLOCK!
  ```

---

## Reconnection System

### When Reconnection Triggers

1. **Graceful Shutdown** (fast path)
   - Master sends `master_shutdown` command before closing
   - Remote receives notification immediately
   - Triggers reconnection without waiting for heartbeat failures

2. **Ungraceful Shutdown** (slow path)
   - Master crashes or network cable unplugged
   - Remote detects via heartbeat failures
   - After one failed heartbeat, triggers reconnection

### Reconnection Process

```python
def _reconnect(self):
    with self._reconnect_lock:  # Block all socket access
        log.warning("Lost connection to master")

        attempt = 0
        while self.running:
            attempt += 1
            try:
                # Close old broken sockets
                with self._req_lock:  # Protect socket closure
                    self.sub_socket.close()
                    self.req_socket.close()

                    # Create fresh sockets
                    self.sub_socket = self.context.socket(zmq.SUB)
                    self.sub_socket.connect(f"tcp://{master_host}:{command_port}")

                    self.req_socket = self.context.socket(zmq.REQ)
                    self.req_socket.connect(f"tcp://{master_host}:{response_port}")

                    # Test connection (verify master is listening)
                    self.req_socket.send_json({
                        "node_id": self.node_id,
                        "command_type": "heartbeat",
                        "result": {"status": "reconnected"},
                        "timestamp": time.time()
                    })
                    self.req_socket.setsockopt(zmq.RCVTIMEO, 2000)
                    ack = self.req_socket.recv_json()

                # Success!
                log.info(f"✅ Reconnected after {attempt} attempts!")
                break

            except Exception as e:
                log.debug(f"Attempt #{attempt} failed: {e}")
                time.sleep(2)  # Wait before retry

        # Lock released here - process_commands() can resume
```

**Why Test Connection?**
ZMQ `connect()` succeeds even if no server is listening (ZMQ uses async connect). By sending a heartbeat and waiting for ACK, we verify master is actually running.

**Why Hold Both Locks?**
- `_reconnect_lock`: Prevents other threads from accessing sockets during recreation
- `_req_lock`: Ensures heartbeat thread doesn't try to use REQ socket while we close/recreate it

---

### Stale Heartbeat Filtering

**Problem:**
When remote reconnects, it might have a heartbeat queued from before reconnection. If master sees this heartbeat, it thinks remote is ready. But remote is actually still inside `_reconnect()` with locks held!

**Timeline of Race Condition:**
```
15:00:20 - Remote queues heartbeat in ZMQ buffer
15:00:21 - Master shuts down
15:00:21 - Remote enters _reconnect() (holds _reconnect_lock)
15:00:26 - Remote reconnects successfully (still holds lock)
15:00:26 - Master receives stale heartbeat from buffer
15:00:26 - Master thinks remote is ready, sends init_interface
15:00:26 - Remote still in _reconnect(), can't process command
15:00:56 - Master timeout error!
```

**Solution: Timestamp Filtering**

```python
# Server tracks start time
class NetworkServer:
    def __init__(self):
        self.start_time = time.time()

# Heartbeat monitor filters old heartbeats
def _monitor_heartbeats(self):
    response = self.wait_for_response()
    timestamp = response.get("timestamp", 0)

    if timestamp < self.start_time:
        log.debug("Ignoring stale heartbeat")
        continue  # Don't add to connected_nodes

    # This is a fresh heartbeat from after reconnection
    self.connected_nodes[node_id] = time.time()
```

**Why This Works:**
- Each server instance has a unique `start_time`
- Heartbeats sent before reconnection have `timestamp < start_time`
- Only heartbeats sent AFTER server started are accepted
- Guarantees remote has finished reconnecting before we think it's ready

---

## Heartbeat Mechanism

### Purpose

1. **Connection Monitoring**: Detect when master/remote goes offline
2. **Node Registration**: Master knows which remotes are connected
3. **Keepalive**: Prevent firewalls from closing idle TCP connections

### Heartbeat Flow

```
Remote (every 5 seconds):
    └─> _send_heartbeats() thread
        ├─> Checks _reconnect_lock (skip if locked)
        ├─> Sends heartbeat via _send_response()
        │   └─> Message: {"node_id": "...", "command_type": "heartbeat",
        │                  "timestamp": time.time()}
        └─> Waits for ACK

Master (continuous):
    └─> _monitor_heartbeats() thread
        ├─> Receives heartbeat on REP socket
        ├─> Checks timestamp >= server.start_time (filter stale)
        ├─> Updates connected_nodes[node_id] = time.time()
        ├─> Sends ACK back to remote
        └─> Caches in _response_cache
```

### Why Check `_reconnect_lock` Before Sending?

**Problem Without Check:**
```
Thread 1 (_reconnect): Holds _reconnect_lock
                    ├─> Closes req_socket
                    └─> Creates new req_socket

Thread 2 (heartbeat): Tries to send heartbeat
                    └─> _send_response() acquires _req_lock
                        └─> req_socket.send_json() ← ERROR!
                            Socket is closed or in invalid state!
```

**Solution:**
```python
def _send_heartbeats(self):
    while self.running:
        # Skip if reconnection in progress
        if not self._reconnect_lock.acquire(blocking=False):
            time.sleep(1.0)
            continue

        try:
            self._send_response("heartbeat", {"status": "alive"})
        except Exception as e:
            # Release before calling _reconnect (it needs same lock)
            self._reconnect_lock.release()
            released_early = True
            self._reconnect()
        finally:
            if not released_early:
                self._reconnect_lock.release()

        time.sleep(5.0)
```

**Key Points:**
- `acquire(blocking=False)`: Returns immediately if lock is held
- If reconnection is in progress, heartbeat waits 1 second and retries
- Prevents accessing sockets while they're being recreated
- Avoids "Resource temporarily unavailable" errors

---

### Heartbeat Timeout Detection

```python
def _monitor_heartbeats(self):
    while self.running:
        # ... receive heartbeats ...

        # Check for dead nodes
        now = time.time()
        dead_nodes = [
            nid for nid, last_time in self.connected_nodes.items()
            if now - last_time > 10.0  # 10 second timeout
        ]

        for node_id in dead_nodes:
            log.warning(f"Node '{node_id}' heartbeat timeout")
            del self.connected_nodes[node_id]
```

**Why 10 Seconds?**
- Heartbeats sent every 5 seconds
- 10 seconds = 2 missed heartbeats
- Balances fast detection vs. false positives from network jitter

---

## Session Lifecycle

### Complete Session Example

```python
# ============ Remote Node (Raspberry Pi) ============
# Started once, runs forever

from ethopy.utils.interface_proxy import RemoteInterfaceNode

node = RemoteInterfaceNode(
    master_host="192.168.1.50",  # Master computer IP
    node_id="rpi_camera_1",
    command_port=5557,
    response_port=5558
)

node.run()  # Blocks forever, processing commands
            # Waits for master to connect...
            # Reconnects automatically when master restarts


# ============ Master Computer ============
# Session 1 (10:00 AM)

from ethopy.core.experiment import Experiment

exp = Experiment()
exp.start()
# ├─> interface.setup_cameras()
# │   └─> InterfaceProxy created, waits for remote
# │       └─> Remote sends heartbeat (fresh, timestamp >= server.start_time)
# │           └─> init_interface sent with animal_id=123, session=1
# │               └─> Remote creates camera with correct metadata
# │
# ├─> Trial 1
# │   ├─> camera_proxy.start_recording() → forwarded to remote
# │   └─> camera_proxy.stop_recording() → forwarded to remote
# │
# └─> exp.cleanup()
#     └─> interface.stop_cameras()
#         └─> camera_proxy.cleanup() (checks server.running)
#             └─> server.shutdown() sends "master_shutdown"
#                 └─> Remote receives, triggers reconnection


# --- 30 second gap (master offline) ---
# Remote: Reconnecting... (waiting for master)


# Session 2 (10:01 AM)

exp = Experiment()
exp.start()
# ├─> New NetworkServer created (new start_time)
# │   └─> Remote reconnects (sends fresh heartbeat with new timestamp)
# │       └─> Stale heartbeats filtered out
# │           └─> init_interface sent with animal_id=123, session=2
# │               └─> Remote cleans up old camera, creates new one
# │
# ├─> Trial 1
# └─> exp.cleanup()
```

---

## Common Issues & Solutions

### Issue 1: "Response timeout for rpi_camera_1/init_interface"

**Symptoms:**
- Master waits 30 seconds then times out
- Remote shows "✅ Reconnected" but doesn't respond to commands

**Root Cause:**
Master sent command while remote was still reconnecting (race condition).

**Solutions:**
1. ✅ **Stale heartbeat filtering** - Master only accepts heartbeats with `timestamp >= server.start_time`
2. ✅ **Wait for fresh heartbeat** - `init_local()` polls `connected_nodes` before sending commands
3. ✅ **Lock coordination** - `process_commands()` skips if `_reconnect_lock` held

### Issue 2: "Socket operation on non-socket"

**Symptoms:**
- Error during cleanup: `camera_proxy.release()` or `cleanup()` fails
- Traceback shows `send_command()` called after server shutdown

**Root Cause:**
`interface.stop_cameras()` called after `server.shutdown()`.

**Solution:**
```python
def remote_method(*args, **kwargs):
    # Check if server still running
    if not self.server.running:
        log.warning(f"Server shut down, skipping {name}()")
        return None

    self.server.send_command(...)
```

### Issue 3: Segmentation Fault

**Symptoms:**
- Python crashes with "Segmentation fault (core dumped)"
- Happens during reconnection

**Root Cause:**
Two threads accessing sockets concurrently during close/recreation.

**Solution:**
```python
# In process_commands() and _send_heartbeats()
if not self._reconnect_lock.acquire(blocking=False):
    time.sleep(0.1)
    return False  # Skip this iteration
```

### Issue 4: "Heartbeat failed: Resource temporarily unavailable"

**Symptoms:**
- Heartbeat errors during reconnection
- Triggers endless reconnection loop

**Root Cause:**
Heartbeat thread tried to send while `_reconnect()` was recreating sockets.

**Solution:**
```python
def _send_heartbeats(self):
    # Check lock before sending
    if not self._reconnect_lock.acquire(blocking=False):
        time.sleep(1.0)
        continue  # Skip this heartbeat
```

---

## Best Practices

### 1. Start Remote Nodes First
Always start remote nodes before master. Remotes wait indefinitely, master times out.

### 2. Use Unique Node IDs
Each remote needs a unique `node_id` to prevent command routing confusion.

### 3. Don't Block in Command Handlers
Command handlers run on main thread. Long operations block `process_commands()` loop.

**Bad:**
```python
def _handle_call_method(self, data):
    time.sleep(60)  # Blocks heartbeats!
    return result
```

**Good:**
```python
def _handle_call_method(self, data):
    # Run long operation in background
    thread = threading.Thread(target=long_operation)
    thread.start()
    return {"status": "started"}
```

### 4. Include Session Metadata in init_interface
Always send `animal_id`, `session`, `start_time` so remote creates correct filenames.

### 5. Cleanup Between Sessions
Call `camera_proxy.cleanup()` between sessions to destroy old interface, then `init_local()` for new session with new metadata.

### 6. Graceful Shutdown
Always call `server.shutdown()` to send graceful shutdown notification. Allows fast reconnection.

---

## Architecture Decisions

### Why ZMQ Instead of Raw Sockets?
- **Pattern Matching**: PUB/SUB and REQ/REP patterns built-in
- **Automatic Reconnection**: ZMQ handles TCP reconnection internally
- **Message Framing**: JSON messages automatically delimited
- **No Serialization Needed**: `send_json()` handles marshalling

### Why REQ/REP for Responses?
- **Guaranteed Delivery**: Blocks until ACK received
- **Ordering**: Strict send-recv alternation ensures order
- **Backpressure**: If master slow, remote blocks (prevents queue buildup)

### Why PUB/SUB for Commands?
- **Broadcast**: Single command reaches all nodes
- **Fire-and-Forget**: Master doesn't wait for command delivery
- **Scalability**: Adding nodes doesn't increase master load

### Why Separate Locks?
- **Performance**: `_req_lock` held briefly (just send-recv), `_reconnect_lock` held longer (entire reconnection)
- **Granularity**: Heartbeat can skip gracefully during reconnection
- **Deadlock Prevention**: Clear lock hierarchy prevents deadlock

---

## Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Monitor Heartbeats
```python
# In NetworkServer._monitor_heartbeats()
log.info(f"Heartbeat from {node_id}, timestamp={timestamp}")
```

### Check Connected Nodes
```python
# In master code
print(f"Connected nodes: {camera_proxy.server.connected_nodes}")
```

### Trace Command Flow
```python
# In InterfaceProxy
log.info(f"MASTER -> REMOTE: {command_type}")

# In RemoteInterfaceNode
log.info(f"REMOTE <- MASTER: {command_type}")
```

### Test Reconnection
```python
# Start remote, start master, let it run
# Kill master (Ctrl+C)
# Wait 30 seconds
# Start master again
# Should reconnect automatically
```

---

## Summary

The distributed networking system enables:
- **Persistence**: Remotes run for days, waiting for master
- **Robustness**: Automatic reconnection across sessions
- **Thread Safety**: Locks prevent concurrent socket access
- **Transparency**: Remote interfaces appear local via proxy
- **Synchronization**: Session metadata distributed to all nodes

**Key Components:**
- **NetworkServer/Client**: Low-level ZMQ communication
- **InterfaceProxy**: High-level remote method forwarding
- **RemoteInterfaceNode**: Runs actual interface on remote
- **Locks**: `_req_lock` (REQ socket), `_reconnect_lock` (reconnection)
- **Heartbeats**: Connection monitoring + stale filtering
- **Reconnection**: Graceful shutdown + automatic retry

**Critical Mechanisms:**
- **Stale heartbeat filtering** prevents race conditions
- **Lock coordination** prevents segfaults
- **Graceful shutdown** enables fast reconnection
- **Session metadata** ensures correct camera filenames