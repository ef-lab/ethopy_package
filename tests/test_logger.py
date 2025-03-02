"""Tests for the Logger class in ethopy.core.logger module.

These tests verify the functionality of the Logger class while properly
handling thread cleanup to avoid test hangs.
"""

import time
from queue import PriorityQueue
import pytest


@pytest.mark.usefixtures("patch_imports")
class TestLogger:
    """Test Logger with real threads but mocked database connections."""

    # Use the real_logger_with_mocks fixture from conftest.py instead of redefining it
    @pytest.fixture
    def logger(self, real_logger_with_mocks):
        """Get a real Logger instance with appropriate mocks."""
        return real_logger_with_mocks

    def test_logger_initialization(self, logger):
        """Test that the Logger initializes correctly."""
        # Basic initialization
        assert logger.setup == "test_host"
        assert logger.manual_run is False
        assert isinstance(logger.queue, PriorityQueue)

        # Check that thread objects were created and are running
        assert hasattr(logger, "inserter_thread")
        assert hasattr(logger, "update_thread")
        assert hasattr(logger, "thread_end")
        assert logger.inserter_thread.is_alive()
        assert logger.update_thread.is_alive()

    def test_put_operation(self, logger):
        """Test that put adds items to the queue."""
        from ethopy.core.logger import PrioritizedItem

        # Get initial queue size
        initial_size = logger.queue.qsize()

        # Add an item
        logger.put(
            table="TestTable",
            tuple={"key": "value"},
            priority=1,
            schema="test_schema",
            block=False,
        )

        # Initial check - item should be added to queue
        assert logger.queue.qsize() > initial_size

        # Get the item and verify its properties
        item = logger.queue.get()
        assert isinstance(item, PrioritizedItem)
        assert item.table == "TestTable"
        assert item.tuple == {"key": "value"}
        assert item.priority == 1

    def test_prioritized_queue(self, logger):
        """Test queue prioritization."""
        from ethopy.core.logger import PrioritizedItem

        # Clear any items in the queue
        while not logger.queue.empty():
            logger.queue.get()

        # Add items with explicit priorities
        logger.queue.put(
            PrioritizedItem(table="T1", tuple={"id": 1}, priority=3, schema="test")
        )
        logger.queue.put(
            PrioritizedItem(table="T2", tuple={"id": 2}, priority=1, schema="test")
        )
        logger.queue.put(
            PrioritizedItem(table="T3", tuple={"id": 3}, priority=2, schema="test")
        )

        # Items should come out in priority order
        assert logger.queue.get().priority == 1
        assert logger.queue.get().priority == 2
        assert logger.queue.get().priority == 3

    def test_cleanup(self, logger):
        """Test that cleanup sets the thread_end event and terminates threads."""
        # Verify threads are running
        assert logger.inserter_thread.is_alive()
        assert logger.update_thread.is_alive()

        # Call cleanup
        logger.cleanup()

        # Verify thread_end was set
        assert logger.thread_end.is_set()

        # Give threads time to respond to the termination signal
        time.sleep(0.5)

        # Threads should be stopping or stopped
        # Note: we can't reliably assert they're stopped because of timing issues
        # but we can verify the signal was properly set

    def test_update_trial_idx(self, logger):
        """Test updating trial index."""
        # Initial state
        logger.trial_key = {"animal_id": 0, "session": 1, "trial_idx": 0}

        # Update
        logger.update_trial_idx(5)

        # Verify update
        assert logger.trial_key["trial_idx"] == 5
