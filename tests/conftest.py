"""Common fixtures for ethopy tests.

This module provides fixtures that can be used across multiple test files
to ensure consistent test setup and resource management.

All tests should use these fixtures to prevent database connections,
provide consistent mock objects, and avoid thread hangs.
"""

import os
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture(scope="module")
def patch_imports():
    """Patch imports to prevent database connections and initialization.
    
    This fixture patches key modules and functions that would normally establish
    database connections, allowing tests to run without requiring an actual
    database connection.
    
    Use this fixture when you need to import ethopy modules safely.
    """
    mocks = {
        'datajoint': MagicMock(),
        'datajoint.config': MagicMock(),
    }
    
    # Create a mock connection and virtual modules that can be used by the Logger
    mock_modules = {
        'experiment': MagicMock(),
        'stimulus': MagicMock(),
        'behavior': MagicMock(),
        'interface': MagicMock(),
        'recording': MagicMock(),
        'mice': MagicMock(),
    }

    mock_conn = MagicMock()
    mock_conn.is_connected = True
    
    with patch.dict(sys.modules, mocks), \
         patch('pathlib.Path.home'), \
         patch('socket.gethostname', return_value="test_host"), \
         patch('ethopy.utils.helper_functions.create_virtual_modules', return_value=(mock_modules, mock_conn)):
        # Now we can safely import ethopy modules
        yield


@pytest.fixture(scope="function")
def patch_logger_log_setup_info():
    """Patch Logger._log_setup_info to prevent blocking during tests.
    
    This fixture specifically patches the _log_setup_info method that causes
    tests to hang due to blocking queue operations.
    
    Use this fixture when creating Logger instances in tests.
    """
    with patch('ethopy.core.logger.Logger._log_setup_info'):
        yield


@pytest.fixture(scope="function")
def mock_logger():
    """Create a standard mock Logger for use in tests.
    
    This fixture provides a consistent mock Logger with common methods 
    already mocked for convenience.
    
    Use this fixture when you need a Logger but don't need to test its internals.
    """
    logger = Mock()
    
    # Setup standard mock methods
    logger.get_table_keys.side_effect = lambda schema, table, key_type=None: {
        ("experiment", "Condition", None): {"id", "cond_hash", "value"},
        ("experiment", "Condition", "primary"): {"id", "cond_hash"},
        ("experiment", "TestTable", None): {"test_id", "cond_hash", "value"},
        ("experiment", "TestTable", "primary"): {"test_id", "cond_hash"},
    }.get((schema, table, key_type), set())
    
    logger.thread_end = MagicMock()
    logger.logger_timer = MagicMock()
    logger.logger_timer.elapsed_time.return_value = 1000
    
    return logger


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for tests that need to write files.
    
    Returns:
        str: Path to a temporary directory that will be cleaned up after the test.
    """
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Clean up created directory after test
    if os.path.exists(temp_path):
        for root, dirs, files in os.walk(temp_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_path)


@pytest.fixture(scope="function")
def real_logger_with_mocks(request, temp_dir):
    """Create a real Logger instance with critical dependencies mocked.
    
    This fixture allows testing real Logger functionality while avoiding
    database connections and providing proper cleanup of threads.
    
    Use this fixture when you need to test actual Logger functionality
    but don't want tests to hang or depend on external resources.
    """
    # Import here to avoid database connections before patching
    with patch('ethopy.core.logger._set_connection'), \
         patch('ethopy.core.logger.rgetattr'), \
         patch('ethopy.utils.helper_functions.rgetattr'), \
         patch('ethopy.core.logger.experiment'), \
         patch('ethopy.core.logger.public_conn'), \
         patch('ethopy.core.logger.Logger._log_setup_info'), \
         patch('os.makedirs', return_value=None), \
         patch('os.path.isdir', return_value=True):
        
        from ethopy.core.logger import Logger
        
        # Create logger with paths set to temp directory
        logger = Logger()
        logger.source_path = temp_dir + "/"
        logger.target_path = temp_dir + "/target/"
        
        # Add finalizer to clean up threads
        def cleanup():
            if hasattr(logger, 'thread_end'):
                logger.thread_end.set()
            
            if hasattr(logger, 'inserter_thread') and logger.inserter_thread.is_alive():
                logger.inserter_thread.join(timeout=1)
            if hasattr(logger, 'update_thread') and logger.update_thread.is_alive():
                logger.update_thread.join(timeout=1)
            
            if hasattr(logger, 'datasets'):
                for name, dataset in list(logger.datasets.items()):
                    if hasattr(dataset, 'exit'):
                        dataset.exit()
        
        request.addfinalizer(cleanup)
        
        yield logger