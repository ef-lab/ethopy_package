from queue import PriorityQueue
from unittest.mock import MagicMock, patch

import pytest

from ethopy.core.logger import Logger


# Basic fixtures for mocking dependencies
@pytest.fixture
def mock_dj(mocker):
    """Mock DataJoint and provide a mock connection."""
    mock_dj = mocker.MagicMock()
    mock_conn = mocker.MagicMock()
    mock_conn.is_connected = True
    mock_dj.Connection.return_value = mock_conn
    return mock_dj


@pytest.fixture
def mock_virtual_modules(mocker):
    """Create mock virtual modules for testing."""
    mock_modules = {
        "experiment": MagicMock(),
        "stimulus": MagicMock(),
        "behavior": MagicMock(),
        "interface": MagicMock(),
        "recording": MagicMock(),
        "mice": MagicMock(),
    }
    mock_conn = MagicMock()
    return mocker.patch(
        "ethopy.utils.helper_functions.create_virtual_modules",
        return_value=(mock_modules, mock_conn),
    )


@pytest.fixture
def basic_logger(mock_dj, mock_virtual_modules):
    """Create a basic logger instance with mocked dependencies."""
    with patch("socket.gethostname", return_value="test_host"):
        logger = Logger()
        yield logger
        logger.cleanup()


# Basic initialization tests
def test_logger_initialization(basic_logger):
    """Test basic initialization of Logger."""
    assert basic_logger.setup == "test_host"
    assert basic_logger.manual_run is False
    assert basic_logger.setup_status == "ready"
    assert isinstance(basic_logger.queue, PriorityQueue)
    assert basic_logger.total_reward == 0


def test_logger_thread_initialization(basic_logger):
    """Test that threads are properly initialized."""
    assert basic_logger.inserter_thread.is_alive()
    assert basic_logger.update_thread.is_alive()
    assert not basic_logger.thread_end.is_set()


# Queue operation tests
def test_put_basic_operation(basic_logger):
    """Test basic put operation without blocking."""
    test_data = {
        "table": "TestTable",
        "tuple": {"test_key": "test_value"},
        "priority": 1,
    }

    basic_logger.put(**test_data)
    # Get the item that was just put in the queue
    item = basic_logger.queue.get_nowait()
    assert item.table == "TestTable"
    assert item.tuple == {"test_key": "test_value"}
    assert item.priority == 1


def test_prioritized_queue_order(basic_logger):
    """Test that items are retrieved from queue in priority order."""
    # Add items with different priorities
    items = [
        {"table": "Table1", "tuple": {"key": 1}, "priority": 3},
        {"table": "Table2", "tuple": {"key": 2}, "priority": 1},
        {"table": "Table3", "tuple": {"key": 3}, "priority": 2},
    ]

    for item in items:
        basic_logger.put(**item)

    # Verify items come out in priority order
    first = basic_logger.queue.get_nowait()
    second = basic_logger.queue.get_nowait()
    third = basic_logger.queue.get_nowait()

    assert first.priority == 1
    assert second.priority == 2
    assert third.priority == 3


# def test_inserter_thread_processes_items(basic_logger):
#     """Test that inserter thread processes items in queue."""
#     # Mock the _insert_item method
#     with patch.object(basic_logger, '_insert_item') as mock_insert:
#         # Add test item
#         test_data = {
#             'table': 'TestTable',
#             'tuple': {'test_key': 'test_value'},
#             'priority': 1,
#             'block': False
#         }
#         basic_logger.put(**test_data)

#         # Give thread time to process
#         time.sleep(0.1)

#         # Verify item was processed
#         assert mock_insert.called


# def test_inserter_thread_handles_errors(basic_logger):
#     """Test that inserter thread properly handles errors."""
#     # Mock _insert_item to raise an exception
#     with patch.object(basic_logger, '_insert_item', side_effect=Exception('Test error')):
#         test_data = {
#             'table': 'TestTable',
#             'tuple': {'test_key': 'test_value'},
#             'priority': 1
#         }
#         basic_logger.put(**test_data)

#         # Give thread time to process and retry
#         time.sleep(2)

#         # Verify item was requeued with higher priority
#         item = basic_logger.queue.get_nowait()
#         assert item.priority > 1
#         assert item.error is True


# # Cleanup tests
# def test_cleanup_waits_for_queue(basic_logger):
#     """Test that cleanup waits for queue to empty."""
#     # Add some items to the queue
#     for i in range(3):
#         basic_logger.put(table="TestTable", tuple={"key": i}, priority=i)

#     # Start cleanup
#     cleanup_thread = threading.Thread(target=basic_logger.cleanup)
#     cleanup_thread.start()

#     # Verify cleanup waits
#     assert not basic_logger.queue.empty()

#     # Empty the queue
#     while not basic_logger.queue.empty():
#         basic_logger.queue.get()

#     # Wait for cleanup to finish
#     cleanup_thread.join(timeout=1)
#     assert not cleanup_thread.is_alive()


def test_create_dataset(basic_logger, tmp_path):
    """Test dataset creation with temporary directory."""
    # Mock paths to use temporary directory
    basic_logger.source_path = str(tmp_path)
    basic_logger.target_path = str(tmp_path / "target")

    # Create a dataset
    dataset = basic_logger.createDataset(
        dataset_name="test_dataset",
        dataset_type=float,
        filename="test.h5",
        db_log=False,
    )

    assert "test.h5" in basic_logger.datasets
    assert dataset is basic_logger.datasets["test.h5"]
    basic_logger.datasets["test.h5"].exit()


# Trial management tests
def test_update_trial_idx(basic_logger):
    """Test updating trial index."""
    basic_logger.update_trial_idx(5)
    assert basic_logger.trial_key["trial_idx"] == 5


def test_update_trial_idx_with_thread_exception(basic_logger):
    """Test that update_trial_idx raises thread exceptions."""
    basic_logger.thread_exception = Exception("Test thread error")
    with pytest.raises(Exception, match="Thread exception occurred:"):
        basic_logger.update_trial_idx(5)
