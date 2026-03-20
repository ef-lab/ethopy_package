# Running Interfaces on Remote Computers

This guide explains how to run ethopy interfaces (cameras, RPPorts, etc.) on remote computers while the experiment runs on a master computer.

**For low-level network details** (threading, heartbeats, reconnection, ZMQ patterns): See [network_module_guide.md](network_module_guide.md)

---

## Overview

### What This Does

Run cameras/sensors on Raspberry Pis while the experiment runs on a master computer. The `InterfaceProxy` makes remote interfaces appear local—you call methods normally and they execute on the remote.

### When to Use

- Multi-camera setups across different physical locations
- Remote sensors on Raspberry Pis
- Separating recording hardware from experiment control

### Key Classes

| Class | Runs On | Purpose |
|-------|---------|---------|
| `InterfaceProxy` | Master | Forwards method calls to remote |
| `RemoteInterfaceNode` | Remote | Executes methods on real interface |

---

## Quick Start

### 1. On Remote (Raspberry Pi)

Create a script that runs continuously:

```python
# camera_node.py
from ethopy.utils.interface_proxy import RemoteInterfaceNode

node = RemoteInterfaceNode(
    master_host="xxx.xxx.x.10",      # Master computer IP
    node_id="rpi_camera_1",          # Unique identifier
    command_port=5557,
    response_port=5558
)

node.run()  # Blocks forever, reconnects automatically
```

Start it before running experiments:
```bash
python camera_node.py
```

### 2. On Master (Main Computer)

In your experiment, create a proxy:

```python
from ethopy.utils.interface_proxy import InterfaceProxy
from ethopy.interfaces.RPPorts import RPPorts

# Create proxy that looks like a local interface
camera_proxy = InterfaceProxy(
    RPPorts,                          # Interface class to run remotely
    remote_host="xxx.xxx.x.100",      # Raspberry Pi IP
    remote_setup_conf_idx=0,          # Setup config index on remote
    node_id="rpi_camera_1",           # Must match remote's node_id
    command_port=5557,
    response_port=5558
)

# Initialize with session metadata (animal_id, session, start_time)
camera_proxy.init_local(exp, beh)

# Use like a local interface!
camera_proxy.start_recording()
# ... run experiment ...
camera_proxy.stop_recording()

# Between sessions: cleanup interface
camera_proxy.cleanup()
```

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                    Master Computer                           │
│                                                              │
│  Experiment                InterfaceProxy                    │
│  camera_proxy.start() ──▶  Forwards as "call_method"         │
│                            command via NetworkServer         │
└──────────────────────────────────┬───────────────────────────┘
                                   │ ZMQ over TCP/IP
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                 Remote Computer (Raspberry Pi)               │
│                                                              │
│  RemoteInterfaceNode          Real Interface                 │
│  Receives command ──────────▶ camera.start_recording()       │
│  Returns result ◀────────────                                │
└──────────────────────────────────────────────────────────────┘
```

1. **Master** calls `camera_proxy.start_recording()`
2. **InterfaceProxy** intercepts the call and sends a `call_method` command
3. **RemoteInterfaceNode** receives command and calls `camera.start_recording()`
4. **Result** flows back to master as if it were a local call

---

## InterfaceProxy API

### Constructor

```python
InterfaceProxy(
    interface_class,        # Class to run remotely (e.g., RPPorts)
    remote_host,            # IP address of remote computer
    remote_setup_conf_idx,  # Setup configuration index on remote
    node_id=None,           # Unique identifier (default: remote_{class_name})
    command_port=5557,      # Port for sending commands
    response_port=5558      # Port for receiving responses
)
```

### Methods

| Method | Description |
|--------|-------------|
| `init_local(exp, beh)` | Initialize remote with session metadata. Blocks until remote is ready. |
| `cleanup()` | Cleanup remote interface. Resets initialization guard. Server stays alive for next session. |
| `shutdown()` | Close network connections. Call when completely done. |
| `set_timeout_policy(method_name, policy)` | Set per-method timeout behavior (see Timeout Policies). |
| `<any_method>(*args, **kwargs)` | Forwarded to remote interface automatically. |

### Session Metadata

`init_local()` sends these values to the remote:
- `animal_id` - From logger's trial_key
- `session` - From logger's trial_key
- `start_time` - From logger_timer (for synchronized timestamps)
- `setup` - Setup identifier

This ensures remote cameras create files with correct names like `animal123_session5_camera.mp4`.

---

## RemoteInterfaceNode API

### Constructor

```python
RemoteInterfaceNode(
    master_host,            # IP or list of IPs (see Multi-Master Support)
    node_id=None,           # Unique identifier (default: hostname)
    command_port=5557,      # Port for receiving commands
    response_port=5558      # Port for sending responses
)
```

### Methods

| Method | Description |
|--------|-------------|
| `run()` | Main loop. Blocks forever processing commands. Reconnects automatically. |

### Automatic Behaviors

The node handles these automatically:
- **Reconnection**: When master restarts, remote reconnects and waits for new session
- **Session changes**: When `init_interface` arrives, cleans up old interface and creates new one
- **Disconnect cleanup**: When connection lost, immediately cleans up interface (stops camera, transfers files)

---

## Timeout Policies

By default, any remote method that doesn't respond within 5 seconds raises `TimeoutError` and crashes the experiment. You can configure per-method behavior with `set_timeout_policy()`.

### Policies

| Policy | Behavior |
|--------|----------|
| `"crash"` | Raise `TimeoutError` (default) |
| `"ignore"` | Return `None` silently, log a warning |
| `("retry", n)` | Retry up to `n` times, then crash |

### Example

```python
# Non-critical sensor poll — ignore if remote is slow
camera_proxy.set_timeout_policy("in_position", "ignore")

# Critical recording command — retry 3 times before giving up
camera_proxy.set_timeout_policy("start_recording", ("retry", 3))

# Use defaults (crash) for everything else
```

### When to Use Each

- **`"ignore"`** — Polling methods called in the experiment hot-loop (e.g., `in_position`, `sync_out`). A missed response just means "no event this cycle".
- **`("retry", n)`** — State-changing commands that must succeed (start/stop recording). Transient network glitches shouldn't fail the session.
- **`"crash"`** (default) — Initialization and critical infrastructure calls where silent failure would corrupt data.

---

## Event Forwarding

Remote events (button presses, licks) are forwarded to the master's `beh.log_activity()`.

```
Remote: Button pressed
         ↓
       interface.beh.log_activity({"port": 1, "type": "button"})
         ↓
       RemoteBehaviorProxy intercepts
         ↓
       Sends to master via network
         ↓
Master: beh.log_activity() called
         ↓
       Event logged in database ✓
```

**Timestamps are synchronized**: The remote uses the master's `start_time` to calculate relative timestamps. The remote and the master should have synchronized time clocks(PTP synchronization).

---

## Multi-Master Support

A remote node can connect to **multiple possible masters**—useful when you have multiple lab computers.

### Configuration Formats

```python
# Single master (default)
node = RemoteInterfaceNode(master_host="xxx.xxx.x.10", ...)

# Multiple specific masters
node = RemoteInterfaceNode(
    master_host=["xxx.xxx.x.10", "xxx.xxx.x.20", "xxx.xxx.x.30"],
    ...
)

# IP range (DHCP networks)
node = RemoteInterfaceNode(
    master_host="xxx.xxx.x.10-30",  # Scans IPs 10-30
    ...
)
```

### How Discovery Works

1. Remote probes all candidate IPs **in parallel** (~2-3 seconds)
2. Connects to **first master that responds**
3. On reconnection, re-discovers (no caching)

For implementation details, see [network_module_guide.md](network_module_guide.md#multi-master-configuration).

---

## Session Lifecycle

### Complete Session Example

```python
# ============ REMOTE (start first, runs forever) ============
from ethopy.utils.interface_proxy import RemoteInterfaceNode

node = RemoteInterfaceNode(
    master_host="xxx.xxx.x.50",
    node_id="rpi_camera_1",
    command_port=5557,
    response_port=5558
)
node.run()  # Waits for master...


# ============ MASTER: Session 1 ============
exp = Experiment()
exp.start()
# → InterfaceProxy waits for remote
# → Remote sends heartbeat (connects)
# → init_interface sent with animal_id=123, session=1
# → Remote creates camera with correct metadata

# Run trials...
camera_proxy.start_recording()
# ...
camera_proxy.stop_recording()

exp.cleanup()
# → camera_proxy.cleanup() destroys interface
# → server.shutdown() notifies remote
# → Remote enters reconnection mode


# ============ 30 second gap ============
# Remote: "Reconnecting... waiting for master"


# ============ MASTER: Session 2 ============
exp = Experiment()
exp.start()
# → New server with new start_time
# → Remote reconnects (sends fresh heartbeat)
# → init_interface sent with animal_id=123, session=2
# → Remote creates NEW camera with new metadata

# ...and so on
```

### Key Points

- **Remote runs forever**: Start once, leave running
- **Master creates new server each session**: Fresh `start_time` filters stale heartbeats
- **Session metadata before interface**: Ensures correct video filenames
- **cleanup() vs shutdown()**: `cleanup()` between sessions, `shutdown()` when done

---

## Best Practices

### 1. Start Remote Nodes First

Remotes wait indefinitely for masters. Masters timeout if remote isn't ready.

### 2. Use Unique Node IDs

Each remote needs a unique `node_id` matching both sides:
```python
# Remote
node = RemoteInterfaceNode(node_id="rpi_camera_1", ...)

# Master
proxy = InterfaceProxy(node_id="rpi_camera_1", ...)  # Must match!
```

### 3. Include Session Metadata

Always call `init_local(exp, beh)` before using the proxy. This sends the animal_id, session, and start_time that an interface needs.

### 4. Use `cleanup()` between sessions, `shutdown()` when done

```python
# End of session — cleans up remote interface, server stays alive
camera_proxy.cleanup()

# Next session — re-initializes with new metadata
camera_proxy.init_local(exp, beh)

# Completely done — closes network connections
camera_proxy.shutdown()
```

### 5. Handle Disconnections

```python
from ethopy.utils.network import NodeDisconnectedError

try:
    req_id = server.send_command("call_method", {}, target_node="rpi_camera_1")
    response = server.get_response("rpi_camera_1", "call_method", req_id, timeout=5.0)
except NodeDisconnectedError:
    print("Camera disconnected - handle gracefully")
```

---

## Network Communication

For low-level implementation details check the [network_module_guide.md](network_module_guide.md)