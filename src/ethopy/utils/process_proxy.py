"""Process proxy for running objects in separate processes.

This module provides a ProcessProxy class that allows running stimulus objects
in separate processes while maintaining transparent access to their methods.
Communication between processes uses callback-based proxies to handle unpicklable
objects like database connections and hardware interfaces.
"""
from multiprocessing import Process, Queue
from typing import Any, Dict, Tuple, Type, Union
import importlib
import os
import signal
import atexit
import logging
import pickle
import queue

# Module logger
logger = logging.getLogger(__name__)


class CallbackProxyLogger:
    """Logger proxy that sends callbacks to main process via response_queue.

    This allows the worker process to call logger methods, which are executed
    on the real logger in the main process via callback messages.

    Attributes:
        response_queue: Queue for sending callbacks to main process
        is_pi: Flag indicating if running on Raspberry Pi
        logger_timer: Timer instance running in worker process
        _source_path: Cached source path from main process logger
    """

    def __init__(self, response_queue: Queue) -> None:
        """Initialize callback proxy logger.

        Args:
            response_queue: Queue for sending callbacks to main process
        """
        self.response_queue = response_queue
        self.is_pi: bool = False
        from ethopy.utils.timer import Timer
        self.logger_timer = Timer()  # Real timer runs in worker
        self._source_path: str = ""  # Set from config

    def set_session_start_time(self, start_time: float) -> None:
        """Set session start time from main process to synchronize timing.

        This ensures elapsed_time() measurements in worker are relative to
        the same zero point as the main process, providing accurate timestamps
        in database logs.

        Args:
            start_time: Absolute time.time() when session started in main process
        """
        import time
        self.logger_timer.start_time = start_time
        current_elapsed = self.logger_timer.elapsed_time()
        logger.debug(
            f"Worker timer synchronized - "
            f"session_start={start_time:.3f}, "
            f"now={time.time():.3f}, "
            f"elapsed={current_elapsed}ms"
        )

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Send log request to main process via callback.

        Args:
            *args: Positional arguments to pass to logger.log()
            **kwargs: Keyword arguments to pass to logger.log()
        """
        try:
            self.response_queue.put_nowait(('__CALLBACK_LOG__', args, kwargs))
        except queue.Full:
            logger.warning(
                "Response queue full - skipping log callback. "
                "Main process may not be processing callbacks fast enough."
            )
        except Exception as e:
            logger.error(f"Failed to send logger.log() callback: {e}")

    def get(self, *args: Any, **kwargs: Any) -> None:
        """Not supported in worker - use pre-fetched data instead.

        Args:
            *args: Ignored
            **kwargs: Ignored

        Raises:
            NotImplementedError: Always raised as get() is not available in worker
        """
        raise NotImplementedError(
            "logger.get() is not available in worker process. "
            "Data should be pre-fetched and passed via config or conditions."
        )

    def createDataset(self, *args: Any, **kwargs: Any) -> None:
        """Not supported in worker.

        Args:
            *args: Ignored
            **kwargs: Ignored

        Raises:
            NotImplementedError: Always raised as createDataset() is not available
        """
        raise NotImplementedError("createDataset() not available in worker process")

    @property
    def source_path(self) -> str:
        """Return cached source_path (set during config transfer).

        Returns:
            str: Cached source path from main process logger
        """
        return self._source_path


class CallbackProxyExp:
    """Exp proxy that sends callbacks to main process for interface operations.

    This allows the worker to trigger hardware events (sync_out) which are
    executed on the real interface in the main process.

    Attributes:
        logger: CallbackProxyLogger instance
        interface: CallbackInterface instance for hardware operations
    """

    def __init__(self, response_queue: Queue, logger: CallbackProxyLogger) -> None:
        """Initialize callback proxy exp.

        Args:
            response_queue: Queue for sending callbacks to main process
            logger: CallbackProxyLogger instance to attach
        """
        self.logger = logger

        class CallbackInterface:
            """Interface proxy for hardware operations."""

            def __init__(self, response_queue: Queue) -> None:
                """Initialize callback interface.

                Args:
                    response_queue: Queue for sending callbacks to main process
                """
                self.response_queue = response_queue

            def sync_out(self, value: bool) -> None:
                """Send hardware trigger to main process via callback.

                Args:
                    value: Trigger value (True/False)
                """
                try:
                    self.response_queue.put_nowait(('__CALLBACK_SYNC_OUT__', (value,), {}))
                except queue.Full:
                    logger.warning(
                        f"Response queue full - skipping sync_out({value}) callback. "
                        "Main process may not be processing callbacks fast enough."
                    )
                except Exception as e:
                    logger.error(f"Failed to send sync_out() callback: {e}")

        self.interface = CallbackInterface(response_queue)


# Setup logging for worker processes
def setup_worker_logging(process_name: str) -> logging.Logger:
    """Setup logging for worker process using ethopy's logging system.

    Args:
        process_name: Name of the process (typically the stimulus class name)

    Returns:
        logging.Logger: Configured logger for the worker process
    """
    from ethopy.utils.ethopy_logging import setup_logging

    # Initialize ethopy logging in the worker process
    setup_logging(console=True, log_level="INFO")

    # Return a logger for this specific worker
    return logging.getLogger(f'ethopy.worker.{process_name}')


def _proxy_worker(
    target_class: Union[str, Type],
    request_queue: Queue,
    response_queue: Queue,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> None:
    """Worker function running in separate process.

    This function runs in a separate process and handles:
    - Creating the stimulus instance
    - Processing commands from the main process
    - Sending responses and callbacks back to main process
    - Graceful shutdown on signals

    Args:
        target_class: Class to instantiate (either class object or module.Class string)
        request_queue: Queue for receiving commands from main process
        response_queue: Queue for sending responses/callbacks to main process
        args: Positional arguments for target_class constructor
        kwargs: Keyword arguments for target_class constructor
    """
    # Setup logging for this worker
    class_name = target_class if isinstance(target_class, str) else target_class.__name__
    logger = setup_worker_logging(class_name)

    logger.info("=" * 60)
    logger.info(f"Worker process started for {class_name}")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Parent PID: {os.getppid()}")
    logger.info("=" * 60)

    # Setup signal handlers for graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received signal {sig_name}, shutting down gracefully...")
        shutdown_requested = True

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

    # Cleanup function
    def cleanup():
        logger.debug("Worker cleanup: closing resources...")
        try:
            if hasattr(instance, 'exit'):
                instance.exit()
            logger.debug("Worker cleanup completed")
        except Exception as e:
            # Ignore expected errors for pygame-based stimuli that were never initialized
            error_msg = str(e).lower()
            if 'video system not initialized' in error_msg or 'pygame' in error_msg:
                logger.debug(f"Skipping cleanup (expected): {e}")
            else:
                logger.error(f"Error during cleanup: {e}")

    atexit.register(cleanup)

    # If target_class is a string, import it
    if isinstance(target_class, str):
        module_path, class_name = target_class.rsplit('.', 1)
        logger.debug(f"Importing {class_name} from {module_path}")
        module = importlib.import_module(module_path)
        target_class = getattr(module, class_name)

    logger.debug(f"Creating instance of {target_class.__name__}")
    try:
        instance = target_class(*args, **kwargs)
        logger.debug("Instance created successfully")
    except Exception as e:
        logger.error(f"Failed to create instance: {e}", exc_info=True)
        try:
            response_queue.put_nowait(("__RESPONSE__", "error", f"Failed to create instance: {e}"))
        except queue.Full:
            logger.warning("Response queue full - could not send instance creation error")
        return

    call_count = {}  # Track method calls for debugging

    logger.debug("Entering event loop, waiting for commands...")

    while True:
        # Check if shutdown requested
        if shutdown_requested:
            logger.debug("Shutdown requested, exiting event loop")
            break
        try:
            method_name, method_args, method_kwargs = request_queue.get(timeout=0.1)
        except Exception as e:
            # Timeout or empty queue - continue waiting
            logger.debug(f"Request queue empty or timed out: {e}")
            continue

        try:
            if method_name == "__STOP__":
                break

            # Handle getting attributes (not calling methods)
            if method_name == "__GET_ATTR__":
                attr_name = method_args[0]
                attr_value = getattr(instance, attr_name)
                try:
                    response_queue.put_nowait(("__RESPONSE__", "success", attr_value))
                except queue.Full:
                    logger.warning(f"Response queue full - could not send __GET_ATTR__ result for {attr_name}")
                continue

            # Special handling for setup() - create callback proxies
            if method_name == 'setup':
                config = method_args[0] if method_args else {}

                # Restore state from config
                for key, value in config.items():
                    if not key.startswith('_'):  # Skip internal keys
                        setattr(instance, key, value)

                # Create callback-based proxies (not stubs!)
                instance.logger = CallbackProxyLogger(response_queue)
                instance.logger._source_path = config.get('_logger_source_path', '')

                # Synchronize timer with main process
                if '_session_start_time' in config:
                    instance.logger.set_session_start_time(config['_session_start_time'])
                else:
                    logger.warning(
                        "No session start time in config - worker timer not synchronized. "
                        "Timestamps may be offset from main process."
                    )

                instance.exp = CallbackProxyExp(response_queue, instance.logger)

                # Call setup() without config parameter
                method = getattr(instance, method_name)
                result = method(config=None)  # Pass None to skip config handling in setup()

                try:
                    response_queue.put_nowait(("__RESPONSE__", "success", result))
                except queue.Full:
                    logger.warning("Response queue full - could not send setup() result")
                continue

            # Track method calls
            call_count[method_name] = call_count.get(method_name, 0) + 1
            call_num = call_count[method_name]

            # Log important method calls
            if method_name in ('prepare', 'start', 'stop', 'init'):
                logger.debug(f"Calling {method_name}() [call #{call_num}]")

            method = getattr(instance, method_name)
            result = method(*method_args, **method_kwargs)

            try:
                response_queue.put_nowait(("__RESPONSE__", "success", result))
            except queue.Full:
                logger.warning(f"Response queue full - could not send {method_name}() result")

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

            # Check if this is a fatal error that should terminate worker
            should_terminate = False

            # Check for display/rendering system shutdown
            error_str = str(e).lower()
            if isinstance(e, RuntimeError) and 'display closed' in error_str:
                should_terminate = True

            if should_terminate:
                logger.warning(f"Fatal error detected: {type(e).__name__}: {e}. Shutting down worker.")
                try:
                    response_queue.put_nowait(("__RESPONSE__", "error", f"Worker terminated: {error_msg}"))
                except queue.Full:
                    pass
                break  # Exit worker loop

            logger.error(f"Worker error: {error_msg}")
            try:
                response_queue.put_nowait(("__RESPONSE__", "error", error_msg))
            except queue.Full:
                logger.warning("Response queue full - could not send error response")
            # Don't break - allow recovery for non-fatal errors


class ProcessProxy:
    """Proxy that runs the target class in a separate process.

    This class provides transparent access to stimulus objects running in separate
    processes. It handles:
    - Spawning worker process
    - Routing method calls to appropriate location (local vs worker)
    - Managing callback-based communication for unpicklable objects
    - Extracting and transferring configuration to worker

    Attributes:
        request_queue: Queue for sending commands to worker
        response_queue: Queue for receiving responses/callbacks from worker
        process: Worker process handle
        _target_class: Original class to be proxied
        _local_instance: Local instance for database/callback operations
        _pending_attrs: Dictionary storing attribute modifications made before init()

    Class Attributes:
        DIRECT_METHODS: Methods called on class without going through worker
        LOCAL_METHODS: Methods called on local instance (need database access)
        LOCAL_ATTRIBUTES: Attributes accessed from local instance
    """

    # Methods and attributes that should be called directly on the class, not through the worker process
    DIRECT_METHODS = {'name', '__class__', '__dict__', 'required_fields', 'default_key', 'cond_tables'}

    # Methods and attributes that should be accessed on the local instance (not forwarded to worker)
    LOCAL_METHODS = {'make_conditions'}
    LOCAL_ATTRIBUTES = {'logger', 'exp', 'fill_colors'}

    def __init__(self, target_class: Union[str, Type], *args: Any, **kwargs: Any) -> None:
        """Initialize ProcessProxy and spawn worker process.

        Args:
            target_class: Class to run in worker process (class or 'module.Class' string)
            *args: Positional arguments for target_class constructor
            **kwargs: Keyword arguments for target_class constructor
        """
        self.request_queue = Queue()
        self.response_queue = Queue()

        # Store the original class for direct method calls
        self._target_class = target_class
        self._local_instance = None  # Will be created after init() is called
        self._pending_attrs = {}  # Store attribute modifications before init()

        # Convert class to string path for pickling
        target_class_path = target_class if isinstance(target_class, str) else f"{target_class.__module__}.{target_class.__name__}"

        # Pass queues as arguments instead of relying on self
        self.process = Process(
            target=_proxy_worker,
            args=(target_class_path, self.request_queue, self.response_queue, args, kwargs),
        )
        self.process.start()

    def name(self) -> str:
        """Get the name of the wrapped class without going through the worker process.

        Returns:
            str: Name of the stimulus class
        """
        if isinstance(self._target_class, str):
            return self._target_class.rsplit('.', 1)[1]
        return self._target_class().name()

    def init(self, exp) -> None:
        """Initialize local instance with experiment data.

        This creates a local instance that will be used for methods requiring
        database/logger access (like make_conditions). The configuration will
        be automatically extracted and sent to the worker when setup() is called.

        Args:
            exp: Experiment object containing logger and configuration
        """
        # Create a local instance for database/condition methods
        if isinstance(self._target_class, str):
            module_path, class_name = self._target_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            target_class = getattr(module, class_name)
        else:
            target_class = self._target_class

        self._local_instance = target_class()

        # Call the actual init() method on local instance
        # This sets up logger, exp, monitor, and computes derived values like px_per_deg
        self._local_instance.init(exp)

        # Apply any pending attribute modifications that were made before init()
        for attr_name, attr_obj in self._pending_attrs.items():
            if hasattr(self._local_instance, attr_name):
                # Get the real attribute from local instance
                real_attr = getattr(self._local_instance, attr_name)
                # Apply all stored modifications to it
                for method_name, calls in attr_obj._pending_calls.items():
                    for args, kwargs in calls:
                        method = getattr(real_attr, method_name)
                        method(*args, **kwargs)
                logger.debug(f"Applied {len(attr_obj._pending_calls)} pending modifications to {attr_name}")

        return None

    def _extract_config(self) -> dict:
        """Extract picklable configuration from local instance to send to worker.

        This automatically discovers and serializes all picklable attributes,
        excluding known unpicklable objects and private attributes.

        Returns:
            dict: Configuration dictionary with all picklable attributes
        """
        config = {}

        if self._local_instance is None:
            return config

        # Add logger.source_path and session start time for worker
        if hasattr(self._local_instance, 'logger'):
            try:
                config['_logger_source_path'] = self._local_instance.logger.source_path

                # Add session start time for timer synchronization
                if hasattr(self._local_instance.logger, 'logger_timer'):
                    config['_session_start_time'] = self._local_instance.logger.logger_timer.start_time
                    logger.debug(f"Extracted session start time: {config['_session_start_time']:.3f}")
            except Exception as e:
                logger.error(f"Failed to extract logger info: {e}")

        # Extract all picklable attributes
        for attr, value in self._local_instance.__dict__.items():
            # Skip private/protected attributes
            if attr.startswith('_'):
                continue

            # Skip known unpicklable objects
            if attr in {'logger', 'exp', 'Presenter'}:
                continue

            # Test if picklable
            try:
                pickle.dumps(value)
                config[attr] = value
            except Exception as e:
                # Skip unpicklable attributes (pygame objects, database connections, etc.)
                logger.debug(f"Skipping unpicklable attribute: {attr}, \n{e}")

        return config

    def _create_pending_attribute_proxy(self, attr_name: str) -> Any:
        """Create a proxy for attributes accessed before init().

        This allows code like `panda_obj.fill_colors.set({...})` before init() is called.
        Method calls are stored and replayed on the real attribute when init() creates
        the local instance.

        Args:
            attr_name: Name of the attribute being accessed

        Returns:
            PendingAttributeProxy: Proxy object that records method calls
        """
        # Check if we already have a pending proxy for this attribute
        if attr_name in self._pending_attrs:
            return self._pending_attrs[attr_name]

        # Create new pending proxy
        class PendingAttributeProxy:
            """Proxy that stores method calls for later replay."""
            def __init__(self):
                self._pending_calls = {}  # {method_name: [(args, kwargs), ...]}

            def __getattr__(self, method_name):
                """Return a callable that stores the method call."""
                def method_wrapper(*args, **kwargs):
                    # Store this method call for later
                    if method_name not in self._pending_calls:
                        self._pending_calls[method_name] = []
                    self._pending_calls[method_name].append((args, kwargs))
                    logger.debug(f"Stored pending call: {attr_name}.{method_name}(*{args}, **{kwargs})")
                    # Return self to allow method chaining
                    return self
                return method_wrapper

        proxy = PendingAttributeProxy()
        self._pending_attrs[attr_name] = proxy
        return proxy

    def __getattr__(self, name):
        """Intercept method calls and forward to the process"""
        # Avoid infinite recursion for internal attributes
        if name in ("request_queue", "response_queue", "process", "_target_class", "_local_instance", "_pending_attrs"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # For direct methods, call them on the local instance if available
        if name in self.DIRECT_METHODS:
            if self._local_instance is not None:
                return getattr(self._local_instance, name)
            else:
                # Fallback to creating a temporary instance
                if isinstance(self._target_class, str):
                    module_path, class_name = self._target_class.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    target_class = getattr(module, class_name)
                else:
                    target_class = self._target_class
                return getattr(target_class(), name)

        # Special handling for setup() - inject config automatically
        if name == 'setup':
            def setup_wrapper(*args, **kwargs):
                """Wrapper that extracts config and injects it into setup() call"""
                # Extract picklable state from local instance
                config = self._extract_config()

                # Override any user-provided config with our extracted config
                # This ensures worker gets all state from main process
                try:
                    self.request_queue.put(('setup', (config,), {}), timeout=1.0)
                except queue.Full:
                    raise RuntimeError("Worker request queue full - worker not responding")
                return self._handle_response()

            return setup_wrapper

        # For local methods that need logger access, call them on local instance
        if name in self.LOCAL_METHODS and self._local_instance is not None:
            return getattr(self._local_instance, name)

        # For local attributes (logger, exp, fill_colors), return from local instance if available
        if name in self.LOCAL_ATTRIBUTES:
            if self._local_instance is not None:
                return getattr(self._local_instance, name)
            else:
                # Before init(), return a pending proxy that stores modifications
                return self._create_pending_attribute_proxy(name)

        # Try to get the attribute from the worker process first
        # This handles both attributes and methods
        class AttributeProxy:
            def __init__(proxy_self, attr_name):
                proxy_self.attr_name = attr_name

            def __call__(proxy_self, *args, **kwargs):
                # Called as a method
                try:
                    self.request_queue.put((proxy_self.attr_name, args, kwargs), timeout=1.0)
                except queue.Full:
                    raise RuntimeError(f"Worker request queue full - worker not responding to {proxy_self.attr_name}()")
                return self._handle_response()

            def __getitem__(proxy_self, key):
                # Called with subscript like curr_cond["key"]
                # First get the attribute value
                try:
                    self.request_queue.put(("__GET_ATTR__", (proxy_self.attr_name,), {}), timeout=1.0)
                except queue.Full:
                    raise RuntimeError(f"Worker request queue full - worker not responding to __GET_ATTR__({proxy_self.attr_name})")
                result = self._handle_response()
                return result[key]

            def __bool__(proxy_self):
                # Called when used in boolean context (if, while, etc.)
                try:
                    self.request_queue.put(("__GET_ATTR__", (proxy_self.attr_name,), {}), timeout=1.0)
                except queue.Full:
                    raise RuntimeError(f"Worker request queue full - worker not responding to __GET_ATTR__({proxy_self.attr_name})")
                result = self._handle_response()
                return bool(result)

        return AttributeProxy(name)

    def _handle_response(self) -> Any:
        """Handle responses from worker, processing callbacks while waiting.

        This method waits for a response from the worker, but also processes
        any callback requests (like logger.log() or sync_out()) that arrive
        while waiting. Callbacks are executed on the main process objects.

        Returns:
            Any: Result from the worker process method execution

        Raises:
            Exception: If worker returns an error status
            RuntimeError: If worker process dies unexpectedly
            TimeoutError: If no response received (handled internally)
        """
        import time
        wait_start = time.time()

        while True:
            # Check if worker process is still alive
            if not self.process.is_alive():
                elapsed = time.time() - wait_start
                raise RuntimeError(
                    f"Worker process died unexpectedly (exit code: {self.process.exitcode}). "
                    f"No response received after {elapsed:.2f}s"
                )

            try:
                response = self.response_queue.get(timeout=0.1)

                # Validate response format
                if not isinstance(response, tuple) or len(response) < 2:
                    logger.debug(f"Invalid response format: {response}")
                    continue

                msg_type = response[0]

                # Handle callback: logger.log()
                if msg_type == '__CALLBACK_LOG__':
                    args, kwargs = response[1], response[2]
                    try:
                        self._local_instance.logger.log(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error executing logger.log callback: {e}")
                    continue  # Keep waiting for actual response

                # Handle callback: exp.interface.sync_out()
                elif msg_type == '__CALLBACK_SYNC_OUT__':
                    value = response[1][0]  # (value,) tuple
                    try:
                        self._local_instance.exp.interface.sync_out(value)
                    except Exception as e:
                        logger.error(f"Error executing sync_out callback: {e}")
                    continue  # Keep waiting for actual response

                # Handle normal response
                elif msg_type == '__RESPONSE__':
                    status, result = response[1], response[2]
                    if status == 'error':
                        raise Exception(result)
                    return result

                # Handle legacy response format (backward compatibility)
                elif msg_type in ('success', 'error'):
                    if msg_type == 'error':
                        raise Exception(response[1])
                    return response[1]

                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                    continue

            except queue.Empty:
                # No message yet, loop and check again
                continue

            except Exception as e:
                if "Worker process did not respond" not in str(e):
                    raise

    def shutdown_worker(self, timeout: float = 5.0) -> None:
        """Stop the worker process - call this to terminate the process.

        Attempts graceful shutdown first, then escalates to terminate and kill
        if worker doesn't respond.

        Args:
            timeout: Maximum time to wait for graceful shutdown before forcing termination
        """
        try:
            self.request_queue.put(("__STOP__", (), {}), timeout=1.0)
        except Exception as e:
            logger.warning(f"Failed to send STOP signal to worker, forcing termination: {e}")

        self.process.join(timeout=timeout)

        if self.process.is_alive():
            logger.warning(f"Worker process did not stop within {timeout}s, forcing termination")
            self.process.terminate()
            self.process.join(timeout=2.0)

            if self.process.is_alive():
                logger.error("Worker process did not respond to termination, killing")
                self.process.kill()
                self.process.join()

    def terminate(self) -> None:
        """Alias for shutdown_worker().

        Provided for compatibility with Process interface.
        """
        self.shutdown_worker()

    def exit(self) -> None:
        """Exit and cleanup worker process - called by experiment.stop().

        This is the standard cleanup method called during normal experiment flow.
        """
        logger.debug("ProcessProxy exit() called, shutting down worker")
        self.shutdown_worker()

    def __del__(self) -> None:
        """Destructor to ensure worker process is cleaned up.

        Called automatically when ProcessProxy is garbage collected.
        Provides last-resort cleanup in case explicit shutdown wasn't called.
        """
        if hasattr(self, 'process') and self.process.is_alive():
            logger.debug("ProcessProxy destructor called, cleaning up worker")
            try:
                self.shutdown_worker(timeout=2.0)
            except Exception as e:
                logger.error(f"Error during ProcessProxy cleanup: {e}")
