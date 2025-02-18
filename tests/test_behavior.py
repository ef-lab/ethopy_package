import pytest
from unittest.mock import Mock
from freezegun import freeze_time

from ethopy.core.behavior import Behavior, BehActivity


@pytest.fixture
def behavior():
    """Create a behavior instance with mocked dependencies."""
    beh = Behavior()
    beh.exp = Mock()
    beh.exp.in_operation = True
    beh.logger = Mock()
    beh.logger.logger_timer = Mock()
    beh.logger.logger_timer.elapsed_time.return_value = 1000
    beh.logger.trial_key = {}  # Empty dict for trial key
    beh.interface = Mock()
    beh.params = {}
    return beh


def test_log_activity_lick(behavior):
    """Test logging a lick activity."""
    activity_key = {"type": "Lick", "port": 2, "time": 1500}

    behavior.log_activity(activity_key)

    # Verify lick was recorded
    assert behavior.last_lick.port == 2
    assert behavior.last_lick.time == 1500
    assert behavior.licked_port == 2


def test_is_licking_no_lick(behavior):
    """Test is_licking when no lick has occurred."""
    behavior.last_lick = BehActivity()
    assert behavior.is_licking(since=0) == 0


def test_is_licking_with_lick(behavior):
    """Test is_licking when lick has occurred."""
    # First check with a lick
    behavior.last_lick = BehActivity(port=3, time=1000)
    assert behavior.is_licking(since=500) == 3

    # Test clearing after check
    behavior.last_lick = BehActivity(port=3, time=1000)  # Reset lick
    assert behavior.is_licking(since=500, clear=True) == 3
    assert behavior.is_licking(since=500) == 0  # Should be cleared


def test_is_licking_reward_port(behavior):
    """Test is_licking with reward port check."""
    behavior.last_lick = BehActivity(port=3, time=1000, reward=True)
    assert behavior.is_licking(since=500, reward=True) == 3

    # Test non-reward port
    behavior.last_lick = BehActivity(port=3, time=1000, reward=False)
    assert behavior.is_licking(since=500, reward=True) == 0


def test_update_history(behavior):
    """Test updating choice and reward history."""
    behavior.update_history(choice=1, reward=0.1, punish=0)

    assert behavior.choice_history == [1]
    assert behavior.reward_history == [0.1]
    assert behavior.punish_history == [0]

    # Test automatic choice from response
    behavior.response = BehActivity(port=2, time=1000)
    behavior.update_history(reward=0.2)

    assert behavior.choice_history == [1, 2]
    assert behavior.reward_history == [0.1, 0.2]


def test_is_hydrated(behavior):
    """Test hydration check functionality."""
    behavior.logger.total_reward = 5.0

    # Test with specific reward amount
    assert behavior.is_hydrated(rew=4.0) is True
    assert behavior.is_hydrated(rew=6.0) is False

    # Test with params max_reward
    behavior.params["max_reward"] = 4.0
    assert behavior.is_hydrated() is True

    behavior.params["max_reward"] = 6.0
    assert behavior.is_hydrated() is False

    # Test with no max_reward set
    behavior.params["max_reward"] = None
    assert behavior.is_hydrated() is False


def test_is_sleep_time(behavior):
    """Test sleep time checking functionality."""
    # Set sleep period from 10 PM to 6 AM
    behavior.logger.setup_info = {
        "start_time": "06:00:00",  # Wake time
        "stop_time": "22:00:00",  # Sleep time
    }

    with freeze_time("2024-02-18 14:30:00"):
        # 2 PM should not be sleep time
        assert behavior.is_sleep_time() is False

    with freeze_time("2024-02-18 23:00:00"):
        # 11 PM should be sleep time
        assert behavior.is_sleep_time() is True
