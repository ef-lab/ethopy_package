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
3. **Thread Safety**: Split socket ownership — each thread owns its socket exclusively, no locks needed
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
        self.nodes = {}  # {node_id: last_heartbeat_time}

        # Cache responses for retrieval by command handlers
        self._response_cache = {}  # {(node_id, cmd_type, request_id): response}
```

**Key Methods:**

1. **send_command(command_type, data, target_node)** → returns `request_id`
   - Broadcasts command to specific node or all nodes
   - Returns a unique `request_id` to pass to `get_response()`

2. **get_response(node_id, command_type, request_id, timeout)**
   - Polls response cache populated by monitor thread
   - Blocks until response arrives or timeout
   - Returns response dict or None

3. **_monitor_loop()** (runs in background thread)
   - Continuously receives heartbeats/responses on REP socket
   - Filters stale heartbeats using `timestamp >= server.start_time`
   - Updates `nodes` dict and caches responses

4. **shutdown()**
   - Sends `master_shutdown` command to all nodes (graceful)
   - Waits 0.5s for message delivery
   - Closes sockets and terminates context

**Why Filter Stale Heartbeats?**
When master restarts, remotes might send heartbeats queued before reconnection. Without filtering, master would think the remote is ready before it actually finishes reconnecting, causing timeouts when sending commands.

---

#### NetworkClient (Remote Side)

**Purpose**: Connect to master and respond to commands

**Key Components:**

```python
class NetworkClient:
    def __init__(self, master_host, command_port, response_port, node_id):
        # SUB socket - receives commands from master (owned by MAIN thread)
        self.sub_socket = zmq.SUB

        # REQ socket - sends responses/heartbeats to master (owned by OUTBOX thread)
        self.req_socket = zmq.REQ

        # Command handlers registry
        self.command_handlers = {}  # {command_type: handler_func}

        # Outbox queue: main thread queues messages, outbox thread sends them
        self._outbox = queue.Queue()
```

**Threading Model (split socket ownership):**

The client uses two threads with **exclusive socket ownership** to eliminate lock contention:

- **Main thread** owns the `sub_socket`: calls `process_commands()`, receives commands, queues responses into `_outbox`
- **Outbox thread** owns the `req_socket`: drains the outbox queue, sends heartbeats when idle, handles reconnection

Each socket is touched by exactly one thread — no locks needed.

**Key Methods:**

1. **process_commands(timeout)** (main thread)
   - Polls SUB socket for incoming commands
   - Checks if command is for this node (`target == node_id` or `"all"`)
   - Calls registered handler, puts response into `_outbox`
   - Manages SUB socket lifecycle during reconnection (closes/recreates its own socket)

2. **send(msg_type, data, wait=True)** (main thread → outbox)
   - Puts `OutboxMessage` into `_outbox` queue
   - If `wait=True`, blocks until outbox thread signals `done_event`

3. **_outbox_loop()** (outbox thread)
   - Drains outbox queue, sends each message via REQ socket
   - Sends heartbeat when queue is empty (every `HEARTBEAT_INTERVAL=2s`)
   - On send failure, triggers reconnection by setting state to `RECONNECTING`

4. **_reconnect()** (outbox thread only)
   - Closes and recreates the REQ socket (safe — outbox thread is sole owner)
   - Re-discovers master from candidate IPs
   - Main thread detects `RECONNECTING` state in `process_commands()` and recreates SUB socket independently

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
    remote_host="xxx.xxx.x.xxx",
    remote_setup_conf_idx=0,
    node_id="rpi_camera_1"
)

# Initialize with session metadata
camera_proxy.init_local(exp, beh)

# Now call methods as if it were local - they're forwarded to remote!
camera_proxy.start_recording()  # Actually runs on Raspberry Pi
camera_proxy.stop_recording()
```

**Why Wait for sync_ready in init_local?**
`init_local()` calls `add_node()` before sending session metadata. This ensures the remote's SUB socket is fully connected and ready to receive commands before `init_interface` is sent — preventing a race condition where the command arrives before the remote is listening.

**Why Check `server.running`?**
When experiment ends, `interface.release()` is called. If the server already shut down, commands would fail with "Socket operation on non-socket". By checking `server.running`, we skip gracefully.

---

#### RemoteInterfaceNode (Remote Side)

**Purpose**: Run the actual interface and respond to master commands

**Key Components:**

```python
class RemoteInterfaceNode:
    def __init__(self, master_host, node_id):
        # Connect to master (blocks until acknowledged)
        self.client = NetworkClient(master_host, ...)
        self.client.register_handler("init_interface", self._handle_init_interface)
        self.client.register_handler("call_method", self._handle_call_method)
        self.client.register_handler("cleanup", self._handle_cleanup)
        self.client.connect()

    def _handle_init_interface(self, data):
        # Extract session metadata (animal_id, session, start_time)
        # Create logger and set metadata BEFORE creating interface
        # Create interface (this initializes camera with correct metadata)
        self.interface = interface_class(exp=proxy, beh=proxy)

    def _handle_call_method(self, data):
        method = getattr(self.interface, data["method"])
        result = method(*data["args"], **data["kwargs"])
        return {"status": "success", "return_value": result}

    def run(self):
        while True:
            self.client.process_commands(timeout=0.1)
```

**Why Session Metadata Before Interface Creation?**
Cameras need `animal_id`, `session`, and `start_time` in their constructor to create the correct video filename. By setting logger metadata first, the interface constructor sees correct values.

---

#### RemoteBehaviorProxy and RemoteExperimentProxy

**Purpose**: Stub objects passed to the real interface on the remote

A real interface expects `exp` and `beh` objects in its constructor. On the remote there is no real experiment or behavior running, so two lightweight stubs fill that role:

- **RemoteExperimentProxy** — holds session metadata (animal_id, session, setup_conf_idx) sent from the master.
- **RemoteBehaviorProxy** — intercepts `beh.log_activity()` calls from the interface (e.g. button presses) and forwards them to the master's real behavior object via the network.

---

### interface.py - Interface Integration

Main interface class that supports both local and remote cameras.

#### Key Methods:

```python
class Interface:
    def setup_cameras(self):
        """Setup both local and remote cameras (idempotent — skips if already done)."""
        self._setup_local_camera()
        self._setup_remote_cameras()

    def _setup_remote_cameras(self):
        # Read remote camera configs from database
        # For each remote:
        #   - Create InterfaceProxy
        #   - Call init_local(exp, beh) to send session metadata
        #   - Remote creates interface+camera with correct metadata

    def release(self):
        """Release all hardware resources (local and remote cameras)."""
        if self.camera:
            self.camera.stop_recording()
            self.camera.release()

        for camera_proxy in self.remote_cameras:
            camera_proxy.release()   # checks server.running internally
            camera_proxy.cleanup()   # checks server.running internally
```

**Remote Camera Configuration (Database):**

```sql
-- Table: SetupConfiguration.RemoteCamera
setup_conf_idx | remote_host     | node_id         | command_port | response_port
0              | xxx.xxx.x.xxx   | rpi_camera_1    | 5557         | 5558
0              | xxx.xxx.x.xxx   | rpi_camera_2    | 5559         | 5560
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

2. Remote node (already running) sends sync_ready
   └─> Master receives sync_ready (timestamp >= start_time)
       └─> Adds node to nodes dict

3. Master calls init_local(exp, beh)
   └─> add_node() waits for sync_ready confirmation
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
   └─> Remote sends heartbeat every 2 seconds
       └─> Master updates nodes timestamp
```

### Session End & Restart (Reconnection Flow)

```
1. Master ends session
   └─> interface.release()
       ├─> camera_proxy.release() (checks server.running)
       └─> camera_proxy.cleanup() (checks server.running)

2. Master shuts down server
   └─> server.shutdown()
       ├─> Sends "master_shutdown" command to all nodes
       ├─> Waits 0.5s for delivery
       └─> Closes sockets

3. Remote receives "master_shutdown"
   └─> _on_master_shutdown() sets RECONNECTING state
       └─> Outbox thread begins reconnection loop

4. Remote enters reconnection loop (outbox thread)
   └─> _reconnect() called
       ├─> Closes REQ socket
       ├─> Probes master candidates
       │   └─> Fails because master not running yet
       ├─> Waits RECONNECT_DELAY seconds
       └─> Retries...

5. Master starts new session (30 seconds later)
   └─> New NetworkServer created
       └─> server.start_time = time.time() (filters old heartbeats)

6. Remote retry succeeds
   └─> REQ socket reconnects to new master
       └─> Main thread recreates SUB socket
           └─> Sends sync_ready with timestamp = now

7. Master receives fresh sync_ready
   └─> Checks timestamp >= server.start_time ✓
       └─> Adds node to nodes dict

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
2. **Monitor Thread**: Receives heartbeats/responses on REP socket, updates `nodes`

**Remote Side:**
1. **Main Thread**: Runs `process_commands()` loop, owns SUB socket
2. **Outbox Thread**: Owns REQ socket, sends queued responses and heartbeats, handles reconnection

### Split Socket Ownership

Thread safety is achieved by giving each thread **exclusive ownership** of its socket:

```
Main Thread                          Outbox Thread
───────────                          ─────────────
Owns: sub_socket                     Owns: req_socket
  • process_commands()                 • _outbox_loop()
  • Receives commands                  • Sends responses
  • Queues responses → _outbox         • Sends heartbeats (when idle)
  • Recreates sub_socket               • Handles reconnection
    on reconnect                       • Recreates req_socket
                                         on reconnect
```

No locks needed — cross-thread communication happens only via the `_outbox` queue and the `state` property (protected by a lightweight `_state_lock`).

---

## Reconnection System

### When Reconnection Triggers

1. **Graceful Shutdown** (fast path)
   - Master sends `master_shutdown` command before closing
   - Remote receives notification immediately
   - Outbox thread enters reconnection loop without waiting for heartbeat failures

2. **Ungraceful Shutdown** (slow path)
   - Master crashes or network cable unplugged
   - Outbox thread detects send failure → sets state to `RECONNECTING`

### Reconnection Process

```
Outbox thread detects failure or receives master_shutdown
    └─> state = RECONNECTING

Main thread detects RECONNECTING in process_commands()
    └─> Closes SUB socket
        └─> Waits for outbox thread to reconnect

Outbox thread (_reconnect):
    └─> Closes REQ socket
        └─> Probes master candidates in parallel
            ├─> Fails → waits RECONNECT_DELAY, retries
            └─> Succeeds → creates new REQ socket
                └─> state = CONNECTED

Main thread resumes:
    └─> Creates new SUB socket
        └─> Sends sync_ready to master
            └─> Ready for new session
```

### Stale Heartbeat Filtering

When the remote reconnects after a master restart, heartbeats sent before reconnection may still be in the ZMQ buffer. The server filters these by comparing each message's `timestamp` against `server.start_time` — only messages sent after the server started are accepted.

---

## Heartbeat Mechanism

### Purpose

1. **Connection Monitoring**: Detect when master/remote goes offline
2. **Node Registration**: Master knows which remotes are connected
3. **Keepalive**: Prevent firewalls from closing idle TCP connections

### Heartbeat Flow

```
Remote outbox thread (every 2 seconds when queue is empty):
    └─> _outbox_loop() sends heartbeat via REQ socket
        └─> Waits for ACK

Master monitor thread (continuous):
    └─> _monitor_loop() receives message on REP socket
        ├─> Checks timestamp >= server.start_time (filter stale)
        ├─> If sync_ready: adds node to nodes dict
        ├─> If heartbeat: updates nodes[node_id] timestamp
        └─> Sends ACK back to remote
```

### Heartbeat Timeout Detection

```python
# Dead node timeout = 5.0s (no heartbeat → disconnected)
dead = [nid for nid, last in self.nodes.items() if now - last > 5.0]
for node_id in dead:
    log.warning(f"Node '{node_id}' disconnected (no heartbeat response)")
    del self.nodes[node_id]
```

---

## Common Issues & Solutions

### Issue 1: "Response timeout for rpi_camera_1/init_interface"

**Symptoms:**
- Master waits then times out
- Remote shows "Reconnected" but doesn't respond to commands

**Root Cause:**
Master sent command while remote was still reconnecting (race condition).

**Solutions:**
1. ✅ **Stale heartbeat filtering** — Master only accepts messages with `timestamp >= server.start_time`
2. ✅ **add_node() in init_local()** — Waits for sync_ready before sending any commands

### Issue 2: "Socket operation on non-socket"

**Symptoms:**
- Error during cleanup: `camera_proxy.release()` or `cleanup()` fails
- Traceback shows `send_command()` called after server shutdown

**Root Cause:**
`interface.release()` called after `server.shutdown()`.

**Solution:**
```python
def remote_method(*args, **kwargs):
    if not self.server.running:
        log.warning(f"Server shut down, skipping {name}()")
        return None
    self.server.send_command(...)
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

## Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Connected Nodes
```python
print(f"Connected nodes: {server.nodes}")
print(f"Node IPs: {server.node_ips}")
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
- **Thread Safety**: Split socket ownership eliminates lock contention
- **Transparency**: Remote interfaces appear local via proxy
- **Synchronization**: Session metadata distributed to all nodes

**Key Components:**
- **NetworkServer/Client**: Low-level ZMQ communication
- **InterfaceProxy**: High-level remote method forwarding
- **RemoteInterfaceNode**: Runs actual interface on remote
- **Split socket ownership**: Main thread owns SUB, outbox thread owns REQ
- **Heartbeats**: Connection monitoring + stale filtering
- **Reconnection**: Graceful shutdown + automatic retry

**Critical Mechanisms:**
- **Stale heartbeat filtering** prevents race conditions on reconnect
- **Split socket ownership** prevents cross-thread socket crashes
- **Graceful shutdown** enables fast reconnection
- **Session metadata** ensures correct camera filenames
