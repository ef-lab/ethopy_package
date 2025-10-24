# ProcessProxy: Complete System Overview

**Purpose:** Run stimulus presentation in separate processes for stability, isolation, and future network distribution
**Status:** Production-ready for local multiprocess execution

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is ProcessProxy?](#what-is-processproxy)
3. [Architecture Overview](#architecture-overview)
4. [How It Works](#how-it-works)
5. [Key Features](#key-features)
6. [Communication System](#communication-system)
7. [Using ProcessProxy](#using-processproxy)
8. [Cleanup and Resource Management](#cleanup-and-resource-management)
9. [Best Practices](#best-practices)
10. [Network Migration Path](#network-migration-path)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Usage

```python
from ethopy.utils.process_proxy import ProcessProxy
from ethopy.stimuli.grating import Grating

# Only change needed: wrap stimulus class
stimulus = ProcessProxy(Grating)  # Instead of: stimulus = Grating()

# Everything else stays the same
stimulus.init(exp)
stimulus.prepare(condition)
stimulus.present()
```

### With Context Manager (Recommended)

```python
with ProcessProxy(Grating) as stimulus:
    stimulus.init(exp)
    stimulus.prepare(condition)
    stimulus.present()
# Automatic cleanup on exit
```

---

## What is ProcessProxy?

ProcessProxy is a **transparent wrapper** that allows any stimulus to run in a separate process without changing the stimulus code.

### Problem It Solves

**Stability Issues:**
- Rendering engine crashes (Panda3D, PsychoPy) terminate entire experiment
- GPU driver failures affect the whole process
- Resource conflicts when running multiple rendering engines

**Performance Issues:**
- Heavy rendering blocks experimental timing
- Database operations during rendering cause jitter
- Memory leaks accumulate over long sessions

**Solution:**
- Run each stimulus in isolated worker process
- Crashes only affect that stimulus, not the experiment
- Clean separation of concerns: main handles data, worker handles rendering

---

## Architecture Overview

### The Big Picture

```
Main Process                           Worker Process
┌─────────────────────┐               ┌──────────────────────┐
│  Experiment Control │               │  Stimulus Rendering  │
│  ┌───────────────┐  │               │  ┌────────────────┐  │
│  │ ProcessProxy  │  │               │  │ Real Stimulus  │  │
│  │ (Intercepts   │  │               │  │ Instance       │  │
│  │  method calls)│  │               │  │                │  │
│  └───────┬───────┘  │               │  └────────▲───────┘  │
│          │          │               │           │          │
│          ▼          │               │           │          │
│  ┌───────────────┐  │               │  ┌────────────────┐  │
│  │ Request Queue │──┼──────────────►│  │ Request Queue  │  │
│  └───────────────┘  │  Serialized   │  └────────┬───────┘  │
│          ▲          │  Messages     │           │          │
│          │          │               │           ▼          │
│  ┌───────────────┐  │               │  Execute Method      │
│  │Response Queue │◄─┼───────────────┼  ┌────────────────┐  │
│  └───────────────┘  │               │  │Response Queue  │  │
│                     │               │  └────────────────┘  │
│  ┌───────────────┐  │               │  ┌────────────────┐  │
│  │ Database      │  │               │  │ Callback       │  │
│  │ Logger        │◄─┼───────────────┼──│ Proxies        │  │
│  │ Hardware I/O  │  │  Callbacks    │  │ (logger, exp)  │  │
│  └───────────────┘  │               │  └────────────────┘  │
└─────────────────────┘               └──────────────────────┘
```

### Key Components

1. **ProcessProxy (Main):** Intercepts method calls and routes to appropriate location
2. **Worker Process:** Executes actual stimulus methods
3. **Request/Response Queues:** Bidirectional communication channels
4. **Callback Proxies:** Allow worker to call main process functions (logger, hardware)
5. **Local Instance:** Handles database operations in main process

---

## How It Works

### 1. Initialization

```python
# User code
stimulus = ProcessProxy(Grating)
```

**What happens:**
1. Create request/response queues
2. Spawn worker process
3. Worker creates actual Grating() instance
4. Worker enters event loop waiting for commands

### 2. Method Routing

ProcessProxy intelligently routes methods based on their purpose:

```python
# Database/query methods → Main Process
stimulus.make_conditions()  # Needs database access

# Rendering methods → Worker Process
stimulus.setup()           # Initialize display
stimulus.present()         # Render frames

# Metadata → Local (no proxy)
stimulus.name()           # Just returns class attribute
```

### 3. Communication Flow

**Example: `stimulus.present()`**

```
TIME  MAIN PROCESS                           WORKER PROCESS
──────────────────────────────────────────────────────────────
T0    stimulus.present()
      └─> request_queue.put('present') ────►
                                             receive 'present'
T1    Wait for response...                   execute present()
                                             ├─> render graphics
                                             └─> logger.log(...)
                                                 └─> send callback
T2    ◄────────────────────────────────────  response_queue.put(
      Receive callback                           '__CALLBACK_LOG__')
      └─> execute logger.log()
      Continue waiting...
                                             complete present()
T3    ◄────────────────────────────────────  response_queue.put(
      Receive response                           '__RESPONSE__')
      └─> return to user code
```

### 4. Callback System

Worker needs to call functions in main process (database, hardware). Can't pass these objects directly (not picklable).

**Solution: Callback Proxies**

```python
# Worker code (appears normal)
self.logger.log("StimCondition.Trial", data)

# What actually happens:
1. CallbackProxyLogger intercepts the call
2. Sends message to main: ('__CALLBACK_LOG__', args, kwargs)
3. Main receives and executes on real logger
4. Worker continues (fire-and-forget)
```

**Why This Works:**
- No need to pickle logger/exp objects
- Single source of truth (one logger in main)
- Real functionality (actually logs to database)
- No state synchronization issues

### 5. Configuration Transfer

Worker needs initial state from main process:

```
MAIN PROCESS                           WORKER PROCESS
────────────                           ──────────────
1. stimulus.init(exp)
   └─> Create local instance
   └─> Set logger, exp, monitor

2. stimulus.setup()
   └─> Extract config
       (all picklable attributes)

3. ───────[Config Dict]──────────────► 4. Receive config
                                          └─> Apply attributes
                                          └─> Create callback proxies
                                          └─> Call instance.setup()
```

**Config includes:**
- Monitor settings (resolution, refresh rate)
- Display parameters (px_per_deg, photodiode)
- Timing info (session start time for sync)
- All picklable stimulus attributes

**Not transferred:**
- logger (replaced with CallbackProxyLogger)
- exp (replaced with CallbackProxyExp)
- Database connections
- File handles

---

## Key Features

### 1. Process Isolation

**Benefit:** Stimulus crashes don't affect experiment

```python
# If Panda3D crashes in worker...
with ProcessProxy(PandaStimulus) as stimulus:
    stimulus.present()  # Crash here!
# Exception raised, but main process alive
# Can continue experiment with different stimulus
```

### 2. Transparent API

**Benefit:** Existing code works unchanged

```python
# Before (in-process)
stimulus = Grating()

# After (multiprocess)
stimulus = ProcessProxy(Grating)  # Only change!

# Everything else identical
stimulus.init(exp)
stimulus.setup()
stimulus.present()
```

### 3. Timer Synchronization

**Problem:** Worker spawns later than session start → timestamp offset

**Solution:** Share session start time

```python
# Main process
config['_session_start_time'] = logger.logger_timer.start_time

# Worker process
instance.logger.set_session_start_time(config['_session_start_time'])
```

Now both processes report same `elapsed_time()`

### 4. Robust Cleanup

**Multiple cleanup triggers:**
1. Context manager `__exit__`
2. Explicit `shutdown_worker()` call
3. `atexit` handler (automatic)
4. Signal handler (Ctrl+C)

**Escalating shutdown strategy:**
1. Send `__STOP__` command (wait 5s)
2. Send SIGTERM signal (wait 2s)
3. Send SIGKILL (immediate)

### 5. Comprehensive Logging

**Worker logs:**
```
[WORKER-12345] INFO - Worker process started for Grating
[WORKER-12345] INFO - Process ID: 12345
[WORKER-12345] INFO - Creating instance...
[WORKER-12345] INFO - Executing: prepare(condition)
[WORKER-12345] INFO - Worker cleanup completed
```

**Main logs:**
```
ethopy.proxy.Grating - INFO - Started worker process PID: 12345
ethopy.proxy.Grating - INFO - Initiating graceful shutdown
ethopy.proxy.Grating - INFO - Worker stopped successfully
```

---

## Communication System

### Queue Architecture

**Two queues for bidirectional communication:**

```python
request_queue:  Main → Worker (commands)
response_queue: Worker → Main (results + callbacks)
```

### Message Types

#### Request Queue (Main → Worker)

| Message | Purpose | Example |
|---------|---------|---------|
| `(method, args, kwargs)` | Execute method | `('present', (), {})` |
| `('__GET_ATTR__', attr)` | Get attribute | `('__GET_ATTR__', 'in_operation')` |
| `('__INIT_WORKER__', attrs)` | Initialize | `('__INIT_WORKER__', config)` |
| `('__STOP__', (), {})` | Shutdown | `('__STOP__', (), {})` |

#### Response Queue (Worker → Main)

| Message | Purpose | Example |
|---------|---------|---------|
| `('__RESPONSE__', status, result)` | Method result | `('__RESPONSE__', 'success', None)` |
| `('__CALLBACK_LOG__', args, kwargs)` | Log to database | `('__CALLBACK_LOG__', ('Trial', data), {})` |
| `('__CALLBACK_SYNC_OUT__', value)` | Hardware trigger | `('__CALLBACK_SYNC_OUT__', True)` |

### Special Method Handling

```python
# DIRECT_METHODS - class-level access (no proxy)
{'name', 'required_fields', 'default_key', 'cond_tables'}

# LOCAL_METHODS - execute in main process
{'make_conditions'}  # Needs database access

# Everything else → worker process
{'setup', 'prepare', 'present', 'start', 'stop', ...}
```

### Queue Lifecycle

```
1. Created in Main Process
   └─> request_queue = Queue()
   └─> response_queue = Queue()

2. Passed to Worker Process
   └─> Process(target=worker, args=(request_queue, response_queue))

3. Communication
   ├─> Main: request_queue.put(command)
   └─> Worker: request_queue.get() → execute → response_queue.put(result)

4. Cleanup
   └─> request_queue.close()
   └─> response_queue.close()
```

---

## Using ProcessProxy

### Basic Pattern

```python
from ethopy.utils.process_proxy import ProcessProxy
from ethopy.stimuli.grating import Grating

# Create proxy
stimulus = ProcessProxy(Grating)

# Initialize
stimulus.init(exp)

# Use normally
stimulus.prepare(condition)
stimulus.present()
stimulus.stop()

# Cleanup (automatic via atexit, or manual)
stimulus.shutdown_worker()
```

### Recommended Pattern (Context Manager)

```python
with ProcessProxy(Grating) as stimulus:
    stimulus.init(exp)

    for condition in conditions:
        stimulus.prepare(condition)
        stimulus.start()

        while stimulus.in_operation:
            stimulus.present()

        stimulus.stop()
# Automatic cleanup guaranteed
```

### Handling Multiple Stimuli

```python
# All cleaned up automatically
with ProcessProxy(Grating) as grating, \
     ProcessProxy(MovieStim) as movie:

    grating.init(exp)
    movie.init(exp)

    # Use both stimuli
    grating.present()
    movie.present()
# Both workers stopped
```

### Writing Compatible Stimuli

Your stimulus must follow these rules:

#### ✅ DO:

**1. Separate initialization from construction**
```python
class MyStimulus(Stimulus):
    def __init__(self):
        # Minimal setup only
        super().__init__()

    def init(self, exp):
        # Heavy initialization here
        self.logger = exp.logger
        self.setup_display()
```

**2. Use picklable data types**
```python
# Good
condition = {'theta': 0, 'duration': 1000}

# Bad
condition = {'callback': lambda x: x}  # Not picklable
```

**3. Return picklable results**
```python
def prepare(self, condition):
    # Good
    return {'status': 'ready'}

    # Bad
    return pygame.Surface((100, 100))  # Not picklable
```

#### ❌ DON'T:

1. Don't pass logger/exp to `__init__`
2. Don't use threading in worker
3. Don't access main process state from worker
4. Don't assume shared memory

---

## Cleanup and Resource Management

### Automatic Cleanup

ProcessProxy automatically cleans up via multiple mechanisms:

```python
# Method 1: Context manager
with ProcessProxy(Grating) as stimulus:
    stimulus.present()
# Cleanup happens here

# Method 2: atexit handler
stimulus = ProcessProxy(Grating)
stimulus.present()
# Cleanup happens when Python exits

# Method 3: Explicit
stimulus.shutdown_worker()
```

### Ctrl+C Handling

```python
stimulus = ProcessProxy(Grating)

try:
    while True:
        stimulus.present()
except KeyboardInterrupt:
    print("Interrupted")
    # Cleanup happens automatically via atexit
```

**What happens:**
1. User presses Ctrl+C
2. Main receives SIGINT → raises KeyboardInterrupt
3. Worker receives SIGINT → sets shutdown flag
4. Both processes exit gracefully
5. Resources released properly

### Signal Handling

**Worker process handles signals:**
```python
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination
```

When signal received:
1. Set `shutdown_requested` flag
2. Exit event loop
3. Call cleanup functions
4. Close resources

### Implementing Cleanup in Stimuli

```python
class MyStimulus(Stimulus):
    def exit(self):
        """MUST implement to release resources"""

        # Close display
        if hasattr(self, 'Presenter'):
            self.Presenter.quit()

        # Close files
        if hasattr(self, 'file'):
            self.file.close()

        # Stop threads
        if hasattr(self, 'thread'):
            self.thread.stop()
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with ProcessProxy(Grating) as stimulus:
    stimulus.present()

# Acceptable (if context manager not practical)
stimulus = ProcessProxy(Grating)
try:
    stimulus.present()
finally:
    stimulus.shutdown_worker()
```

### 2. Implement exit() Method

```python
class MyStimulus(Stimulus):
    def exit(self):
        """Release all resources"""
        if self.Presenter:
            self.Presenter.quit()
        # Close files, stop threads, release locks
```

### 3. Don't Block Event Loop

```python
# Bad: Never returns to event loop
def present(self):
    while True:
        display()

# Good: Returns after each frame
def present(self):
    display_one_frame()
```

### 4. Optimize Performance

```python
# Slow: Many queue round-trips
for cond in conditions:
    stimulus.prepare(cond)  # N round-trips

# Fast: Batch when possible
def prepare_multiple(self, conditions):
    for cond in conditions:
        self._prepare_one(cond)
# One round-trip

# Slow: Check attribute in tight loop
while stimulus.in_operation:  # Queue call each time!
    stimulus.present()

# Fast: Time-based loop
start = time.time()
while (time.time() - start) < duration:
    stimulus.present()
```

### 5. Log Important Events

```python
import logging
logger = logging.getLogger(__name__)

def my_method(self):
    logger.info("Starting operation")
    try:
        do_something()
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        raise
```

### 6. Test Both Modes

```python
import pytest

@pytest.fixture(params=['inprocess', 'multiprocess'])
def stimulus(request):
    if request.param == 'inprocess':
        return MyStimulus()
    else:
        return ProcessProxy(MyStimulus)

def test_prepare(stimulus):
    # Works for both!
    stimulus.prepare(condition)
    assert stimulus.state == 'ready'
```

---

## Network Migration Path

ProcessProxy is designed for **easy network extension**.

### Current: Local Multiprocessing

```python
from multiprocessing import Queue

# In ProcessProxy.__init__
self.request_queue = Queue()
self.response_queue = Queue()
```

### Future: Network Sockets

**Step 1: Create network communication layer**

```python
import zmq
import pickle

class NetworkComm:
    def __init__(self, host, port, mode='client'):
        self.context = zmq.Context()

        if mode == 'client':
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{host}:{port}")
        else:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{port}")

    def put(self, data):
        serialized = pickle.dumps(data)
        self.socket.send(serialized)

    def get(self):
        serialized = self.socket.recv()
        return pickle.loads(serialized)
```

**Step 2: Create NetworkProxy**

```python
class NetworkProxy(ProcessProxy):
    def __init__(self, target_class, host='192.168.1.100', port=5555):
        # Connect to remote worker
        self.request_queue = NetworkComm(host, port, mode='client')
        self.response_queue = self.request_queue

        # Rest identical to ProcessProxy!
        self._target_class = target_class
        # ...
```

**Step 3: Remote worker server**

```python
# On stimulus computer
def network_worker(target_class, host='0.0.0.0', port=5555):
    comm = NetworkComm(host, port, mode='server')
    instance = target_class()

    while True:
        method_name, args, kwargs = comm.get()

        if method_name == "__STOP__":
            break

        method = getattr(instance, method_name)
        result = method(*args, **kwargs)

        comm.put(("success", result))

# Run worker
network_worker(Grating, port=5555)
```

**Step 4: Use NetworkProxy**

```python
# On experiment computer
stimulus = NetworkProxy(Grating, host='192.168.1.100', port=5555)

# Use exactly like ProcessProxy!
stimulus.init(exp)
stimulus.prepare(condition)
stimulus.present()
```

### What Enables This Migration

1. **Queue abstraction:** Not tied to multiprocessing
2. **Serializable protocol:** Already using pickle
3. **Stateless communication:** No shared memory
4. **Callback architecture:** Works over network

**Estimated migration effort:** 6-9 weeks for production network version

---

## Troubleshooting

### Worker Not Stopping

**Symptom:** Worker hangs during shutdown

**Solutions:**
1. Check logs for errors
2. Verify `exit()` method doesn't block
3. Check for infinite loops in stimulus
4. Ensure pygame closes properly

**Debug:**
```python
import logging
logging.getLogger('ethopy.proxy').setLevel(logging.DEBUG)
logging.getLogger('ethopy.worker').setLevel(logging.DEBUG)
```

### Queue Timeout Errors

**Symptom:** `queue.Empty` exceptions

**Causes:**
- Worker crashed
- Method taking too long
- Event loop blocked

**Fix:**
```python
# Check worker status
if not stimulus.process.is_alive():
    print(f"Worker died: exit code {stimulus.process.exitcode}")

# Increase timeout for slow operations
# (modify in process_proxy.py)
```

### Performance Issues

**Symptom:** Slow method calls

**Cause:** Queue overhead accumulates

**Optimization:**
```python
# Reduce attribute checks in loops
while stimulus.in_operation:  # Slow
    pass

start = time.time()
while (time.time() - start) < duration:  # Fast
    pass

# Batch operations
stimulus.prepare_multiple(conditions)  # Fast
for c in conditions:
    stimulus.prepare(c)  # Slow
```

### Pickling Errors

**Symptom:** `TypeError: cannot pickle ...`

**Cause:** Non-picklable object in arguments/return value

**Fix:**
```python
# Bad
stimulus.prepare({'callback': lambda x: x})  # Can't pickle lambda

# Good
stimulus.prepare({'value': 42})  # Primitive types OK
```

### Memory Leaks

**Symptom:** Memory grows over time

**Cause:** Resources not released

**Fix:**
```python
class MyStimulus(Stimulus):
    def exit(self):
        # Clean up ALL resources
        self.Presenter.quit()
        self.close_files()
        self.stop_threads()
```

---

## Summary

### What ProcessProxy Provides

- ✅ **Process isolation:** Crashes don't kill experiment
- ✅ **Transparent API:** One-line change to use
- ✅ **Callback system:** Worker accesses database/hardware via main
- ✅ **Timer synchronization:** Accurate timestamps across processes
- ✅ **Robust cleanup:** Multiple automatic cleanup mechanisms
- ✅ **Network-ready architecture:** Foundation for distributed rendering

### When to Use

**Use ProcessProxy when:**
- Stimulus uses unstable libraries (Panda3D, custom OpenGL)
- Need resource isolation
- Want to prevent crashes from affecting experiment
- Planning future network distribution

**Don't use when:**
- Simple, stable stimuli (native pygame)
- Performance critical (overhead ~1ms per call)
- Stimulus needs shared memory

### Migration Summary

```
Current:    ProcessProxy(Stimulus)     [same machine, local process]
              ↓
Future:     NetworkProxy(Stimulus)     [TCP/IP, remote machine]
              ↓
            CloudProxy(Stimulus)        [cloud deployment]
```

**Key Takeaway:** ProcessProxy provides a clean abstraction that makes distributed computing accessible without changing stimulus code!

---

## Related Documentation

For deeper technical details, see:

- **ProcessProxy_Architecture.md** - Complete implementation guide
- **ProcessProxy_Callback_Architecture.md** - Why we can't pass logger/exp directly
- **ProcessProxy_Queue_Flow.md** - Detailed queue communication patterns
- **ProcessProxy_Cleanup_And_Logging.md** - Cleanup mechanisms and logging
- **ProcessProxy_Technical_Report.md** - Engineering analysis and alternatives

---

**Document Version:** 1.0
**Last Updated:** 2025-10-24
**Status:** Production
