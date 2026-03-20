# Network Module Guide

A general-purpose ZMQ networking library for distributed Python applications.

## Overview

The `network.py` module provides a **master-remote communication pattern** using ZeroMQ. It can be used:

- **With ethopy**: Coordinate cameras/sensors on Raspberry Pis
- **Standalone**: Any distributed Python application needing command-response communication

## Quick Start

Test locally with 2 terminals on the same computer:

### Terminal 1: Start the Worker (Client) First

```python
# worker.py
from ethopy.utils.network import NetworkClient

client = NetworkClient(
    master_host="localhost",
    node_id="worker",
    command_port=5555,
    response_port=5556
)

def handle_ping(data):
    print("Got ping!")
    return {"response": "pong"}

client.register_handler("ping", handle_ping)
client.connect()  # Blocks until master acknowledges

print("Worker ready!")
try:
    while True:
        client.process_commands(timeout=0.1)
except KeyboardInterrupt:
    client.shutdown()
```

### Terminal 2: Start the Master (Server)

```python
# master.py
from ethopy.utils.network import NetworkServer

server = NetworkServer(command_port=5555, response_port=5556)

print("Waiting for worker...")
server.add_node("worker", timeout=15.0)
print("Worker connected!")

# Send command and get response
req_id = server.send_command("ping", {}, target_node="worker")
response = server.get_response("worker", "ping", req_id, timeout=5.0)
print(f"Response: {response['result']['response']}")  # "pong"

server.shutdown()
```

**To test:**
```bash
# Terminal 1 (start first)
python worker.py

# Terminal 2
python master.py
```

## Capabilities

| Feature | Description |
|---------|-------------|
| **Broadcast commands** | Send to all remotes or target specific node |
| **Request-response** | Get responses from remotes with timeout |
| **Automatic reconnection** | Remotes stay in standby and reconnect when master restarts |
| **Multi-master discovery** | Remote can connect to any of multiple masters |
| **Heartbeat monitoring** | Detect dead nodes automatically |

## Restrictions

| Limitation | Description |
|------------|-------------|
| **One master** | Single server coordinates all remotes (no master-to-master) |
| **TCP only** | Requires network connectivity (no IPC/inproc) |
| **JSON serialization** | Data must be JSON-serializable |
| **No message queuing** | Commands not persisted if remote offline |
| **Single response** | One response per command (no streaming) |
| **Blocking handlers** | Long operations block command processing |

## Multi-Master Support

Remotes can connect to any of multiple possible masters:

```python
# Localhost (for testing on same machine)
client = NetworkClient(master_host="localhost", ...)

# Single IP
client = NetworkClient(master_host="192.168.1.10", ...)

# List of IPs (probed in parallel)
client = NetworkClient(master_host=["192.168.1.10", "192.168.1.20"], ...)

# IP range (auto-expands)
client = NetworkClient(master_host="192.168.1.10-30", ...)
```

**Discovery algorithm:**
1. Probe all candidates in parallel
2. First master to respond wins
3. Re-discover on every reconnection (no caching)

## Event Handlers

For async events that don't follow request-response pattern:

```python
# Master: register handler
def handle_sensor_event(data):
    print(f"Sensor {data['id']} triggered: {data['value']}")

server.register_event_handler("sensor_event", handle_sensor_event)

# Remote: send event
client.send("sensor_event", {"id": 1, "value": 3.14}, wait=False)
```

## Reconnection

When connection is lost, the remote:

1. Detects failure (heartbeat timeout or socket error)
2. Calls disconnect callback (cleanup resources)
3. Re-discovers master from candidates
4. Re-establishes connection

**Log throttling:** During reconnection, logs are throttled to avoid flooding:
- First attempt: logged immediately
- Then: logged once per hour with attempt count

```python
# Register cleanup callback
def on_disconnect():
    print("Connection lost - cleaning up")
    camera.stop()

client.on_disconnect(on_disconnect)
```

## Heartbeat & Timeout Timing

```
┌─────────────────────────────────────────────────────────────┐
│                     NetworkClient                           │
│                                                             │
│   HEARTBEAT_INTERVAL = 2.0s   (sends heartbeat every 2s)    │
│   SEND_TIMEOUT = 2.0s         (wait for ACK)                │
│   RECONNECT_DELAY = 5.0s      (delay between retries)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     NetworkServer                           │
│                                                             │
│   Dead node timeout = 5.0s    (no heartbeat → disconnected) │
└─────────────────────────────────────────────────────────────┘

Timeline when worker dies:
  t=0s  Worker sends heartbeat
  t=2s  Worker dies (no heartbeat sent)
  t=5s  Server detects: "Node 'worker' disconnected (no heartbeat response)"
```

## Node Disconnection Handling

When a node disconnects, the server detects it via heartbeat timeout and:
1. Logs: `"Node 'worker' disconnected (no heartbeat response)"`
2. Removes node from `server.nodes`
3. Clears cached responses for that node
4. Calls any registered disconnect handlers

**If you try to communicate with a disconnected node:**

```python
from ethopy.utils.network import NetworkServer, NodeDisconnectedError

server = NetworkServer(command_port=5555, response_port=5556)
server.add_node("worker", timeout=15.0)

# ... worker disconnects ...

# This raises NodeDisconnectedError instead of returning None
try:
    req_id = server.send_command("ping", {}, target_node="worker")
    response = server.get_response("worker", "ping", req_id, timeout=5.0)
except NodeDisconnectedError as e:
    print(f"Worker lost: {e}")
    # Handle gracefully - cleanup, notify user, etc.
```

**Register a disconnect callback on the server (optional):**

```python
def on_worker_disconnect(node_id):
    print(f"Node {node_id} disconnected!")
    # Custom cleanup, save state, alert user, etc.

server.on_node_disconnect(on_worker_disconnect)
```

## Connection States

```python
from ethopy.utils.network import ConnectionState

# States
ConnectionState.DISCONNECTED  # Initial state
ConnectionState.CONNECTING    # Finding master
ConnectionState.CONNECTED     # Ready for commands
ConnectionState.RECONNECTING  # Lost connection, recovering
```

## Waiting for Nodes

```python
# Recommended - informative error on mismatch
server.add_node("camera_1", timeout=15.0)

# If wrong node connects, error shows what actually connected:
# TimeoutError: Expected node 'camera_1' but connected node(s):
#   'camera_test' (IP: 192.168.1.50).
#   Hint: Remote client defaults to socket.gethostname() if node_id not set.
```

**IP tracking:** The server tracks each node's IP address in `server.node_ips`:
```python
print(server.node_ips)  # {'camera_1': '192.168.1.50', 'camera_2': '192.168.1.51'}
```

## Error Handling

| Error | Cause | Handling |
|-------|-------|----------|
| `NodeDisconnectedError` | Node disconnected, can't communicate | Catch and handle gracefully |
| `TimeoutError` | Master not found or node_id mismatch | Check node_id matches on both sides |
| `zmq.Again` | Receive timeout | Normal, continue loop |
| `zmq.ZMQError` | Socket error | Trigger reconnection |

```python
from ethopy.utils.network import NetworkServer, NodeDisconnectedError

# Handle disconnection gracefully
try:
    req_id = server.send_command("ping", {}, target_node="worker")
    response = server.get_response("worker", "ping", req_id, timeout=5.0)
    print(f"Got: {response['result']}")
except NodeDisconnectedError:
    print("Worker disconnected - stopping")
    server.shutdown()
```

## Best Practices

### 1. Start Workers First
```bash
# Terminal 1 (worker) - start first
python example_worker.py

# Terminal 2 (master) - start after
python example_master.py
```

### 2. Use Unique Node IDs
```python
# Good - unique per device
client = NetworkClient(node_id="rpi_camera_1", ...)

# Also good - omit to use hostname (must match on master side)
client = NetworkClient(...)  # uses socket.gethostname()

# Bad - duplicates cause routing issues
client = NetworkClient(node_id="camera", ...)
```

**Note:** If `node_id` is omitted, it defaults to `socket.gethostname()`.
The master must use the same `node_id` when calling `add_node()` or `send_command()`.

### 3. Don't Block Handlers
```python
# Bad - blocks command loop
def handle_train(data):
    train_model(epochs=1000)  # Takes 2 hours!
    return {"status": "done"}

# Good - run in background
def handle_train(data):
    thread = threading.Thread(target=train_model, args=(1000,))
    thread.start()
    return {"status": "started"}
```

### 4. Handle Disconnections
```python
from ethopy.utils.network import NodeDisconnectedError

try:
    req_id = server.send_command("command", {}, target_node="worker")
    response = server.get_response("worker", "command", req_id, timeout=5.0)
except NodeDisconnectedError:
    print("Worker lost - handle gracefully")
```

### 5. Always Cleanup
```python
try:
    # Your code
    pass
finally:
    client.shutdown()  # or server.shutdown()
```

## Complete Example

An interactive example you can run with 2 terminals on localhost:

### Terminal 1: example_worker.py

```python
"""Simple worker - demonstrates the master/worker communication flow.

Start this first, then run example_master.py
"""
import logging

from ethopy.utils.network import NetworkClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def handle_echo(data):
    """Echo back whatever master sends."""
    print(f"  Received: {data['text']}")
    return {"echo": data["text"]}


def handle_ping(data):
    """Respond to ping."""
    print("  Got ping!")
    return {"response": "pong"}


client = NetworkClient(
    master_host="localhost",
    node_id="worker",
    command_port=5555,
    response_port=5556
)

client.register_handler("echo", handle_echo)
client.register_handler("ping", handle_ping)

client.connect()

try:
    while True:
        client.process_commands(timeout=0.1)
except KeyboardInterrupt:
    client.shutdown()
```

### Terminal 2: example_master.py

```python
"""Interactive master - send commands to worker from user input.

Start example_worker.py first, then run this.
"""
import logging

from ethopy.utils.network import NetworkServer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

server = NetworkServer(command_port=5555, response_port=5556)
server.add_node("worker", timeout=15.0)

print("Commands:")
print("  ping        - ping the worker")
print("  echo <text> - send text to worker")
print("  quit        - exit\n")

try:
    while True:
        user_input = input("> ").strip()

        if not user_input:
            continue

        if user_input == "quit":
            break

        elif user_input == "ping":
            req_id = server.send_command("ping", {}, target_node="worker")
            response = server.get_response("worker", "ping", req_id, timeout=5.0)
            print(f"  Worker response: {response['result']['response']}\n")

        elif user_input.startswith("echo "):
            text = user_input[5:]
            req_id = server.send_command("echo", {"text": text}, target_node="worker")
            response = server.get_response("worker", "echo", req_id, timeout=5.0)
            print(f"  Worker echoed: {response['result']['echo']}\n")

        else:
            print("  Unknown command\n")

except KeyboardInterrupt:
    print("\n")

server.shutdown()
```

**To test:**
```bash
# Terminal 1 (start first)
python example_worker.py

# Terminal 2
python example_master.py
```

**Expected interaction:**
```
# Terminal 2 (master)
> ping
  Worker response: pong

> echo hello world
  Worker echoed: hello world

> quit

# Terminal 1 (worker) output
  Got ping!
  Received: hello world
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Master (NetworkServer)                  │
│                                                             │
│   PUB socket ──────────────────────┐                        │
│   (broadcasts commands)            │                        │
│                                    ▼                        │
│   REP socket ◄─── responses ───────┼────────────────────    │
│   (receives responses)             │                        │
└────────────────────────────────────┼────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Remote 1            │  │ Remote 2            │  │ Remote 3            │
│ (NetworkClient)     │  │ (NetworkClient)     │  │ (NetworkClient)     │
│                     │  │                     │  │                     │
│ SUB ◄── commands    │  │ SUB ◄── commands    │  │ SUB ◄── commands    │
│ REQ ──► responses   │  │ REQ ──► responses   │  │ REQ ──► responses   │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```
