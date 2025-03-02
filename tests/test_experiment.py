"""Tests for the Experiment class in ethopy.core.experiment module.

These tests verify the functionality of the Experiment class while avoiding
database connections and thread issues that could cause tests to hang.
"""

import logging
import pytest
from unittest.mock import Mock, patch

# Use the shared fixtures from conftest.py for consistent test setup


@pytest.mark.usefixtures("patch_imports")
class TestExperiment:
    """Test experiment module functionality with mocked dependencies."""

    # Use the mock_logger fixture from conftest.py instead of defining it here

    @pytest.fixture
    def experiment(self, mock_logger):
        """Create an experiment instance with mocked logger."""
        with patch('ethopy.core.logger.Logger._log_setup_info'):
            from ethopy.core.experiment import ExperimentClass
            
            exp = ExperimentClass()
            exp.logger = mock_logger
            
            yield exp

    def test_log_conditions_empty_conditions(self, experiment):
        """Test handling of empty conditions list."""
        result = experiment.log_conditions([])
        assert result == [], "Empty conditions should return empty list"
        experiment.logger.put.assert_not_called()

    def test_log_conditions_single_condition_default_table(self, experiment):
        """Test logging single condition to default table."""
        condition = {"id": 1, "value": "test"}
        _ = experiment.log_conditions([condition])

        # Verify the logger was called correctly
        experiment.logger.put.assert_called_once()
        call_args = experiment.logger.put.call_args[1]
        assert call_args["table"] == "Condition"
        assert call_args["schema"] == "experiment"
        assert call_args["priority"] == 2

        # Verify hash was added to condition
        assert "cond_hash" in call_args["tuple"]

    def test_log_conditions_multiple_tables(self, experiment):
        """Test logging condition to multiple tables."""
        condition = {"id": 1, "test_id": 2, "value": "test"}
        tables = ["Condition", "TestTable"]

        experiment.log_conditions([condition], condition_tables=tables)

        # Verify logger was called for each table
        assert experiment.logger.put.call_count == 2

        # Verify priorities were incremented
        calls = experiment.logger.put.call_args_list
        assert calls[0][1]["priority"] == 2
        assert calls[1][1]["priority"] == 3
        assert calls[0][1]["table"] == "Condition"
        assert calls[1][1]["table"] == "TestTable"

    def test_log_conditions_iterable_primary_key(self, experiment):
        """Test handling of iterable primary key values."""
        condition = {"id": [1, 2, 3], "value": ["a", "b", "c"], "cond_hash": "hash123"}

        experiment.log_conditions([condition])

        # Verify logger was called for each item in the iterable
        assert experiment.logger.put.call_count == 3

        # Verify correct values were used for each call
        calls = experiment.logger.put.call_args_list
        expected_values = [
            {"id": 1, "value": "a", "cond_hash": "OzBJwTCvHJ3YRj4o2sgFdQ=="},
            {"id": 2, "value": "b", "cond_hash": "OzBJwTCvHJ3YRj4o2sgFdQ=="},
            {"id": 3, "value": "c", "cond_hash": "OzBJwTCvHJ3YRj4o2sgFdQ=="},
        ]

        for call, expected in zip(calls, expected_values):
            assert call[1]["tuple"] == expected

    def test_log_conditions_missing_required_fields(self, experiment, caplog):
        """Test handling of conditions missing required fields."""
        condition = {"id": 1}  # Missing 'value' field

        with caplog.at_level(logging.WARNING):
            experiment.log_conditions([condition])

        assert "Missing keys" in caplog.text
        experiment.logger.put.assert_not_called()

    def test_log_conditions_custom_hash_field(self, experiment):
        """Test using custom hash field name."""
        condition = {"id": 1, "value": "test"}
        _ = experiment.log_conditions(
            [condition], condition_tables=["CustomTable"], hash_field="custom_hash"
        )

        call_args = experiment.logger.put.call_args[1]
        assert "custom_hash" in call_args["tuple"]
        assert "cond_hash" not in call_args["tuple"]

    def test_log_conditions_hash_generation(self, experiment):
        """Test that hash is generated correctly from condition fields."""
        # Import inside the function to ensure mocking is in place
        from ethopy.utils.helper_functions import make_hash

        condition = {"id": 1, "value": "test"}
        _ = experiment.log_conditions([condition])

        # Get the hash from the logged condition
        call_args = experiment.logger.put.call_args[1]
        generated_hash = call_args["tuple"]["cond_hash"]

        # Calculate expected hash
        expected_hash = make_hash({"id": 1, "value": "test"})

        assert generated_hash == expected_hash

    def test_is_stopped(self, mock_logger):
        """Test is_stopped() behavior under different conditions."""
        # Import inside the function to ensure mocking is in place
        from ethopy.core.experiment import ExperimentClass

        exp = ExperimentClass()
        exp.logger = mock_logger
        exp.in_operation = True

        # Test when quit is False
        exp.quit = False
        assert not exp.is_stopped()
        assert exp.in_operation

        # Test when quit is True
        exp.quit = True
        exp.logger.setup_status = "running"
        assert exp.is_stopped()
        assert not exp.in_operation
        exp.logger.update_setup_info.assert_called_once_with({"status": "stop"})

        # Test when setup_status is already "stop"
        exp.logger.reset_mock()
        exp.logger.setup_status = "stop"
        assert exp.is_stopped()
        exp.logger.update_setup_info.assert_not_called()

    def test_stim_init(self):
        """Test stimulus initialization."""
        # Import inside the function to ensure mocking is in place
        from ethopy.core.experiment import ExperimentClass
        from ethopy.core.stimulus import Stimulus

        exp = ExperimentClass()
        mock_stim_class = Mock(spec=Stimulus)
        mock_stim_class.name.return_value = "test_stim"
        stims = {}

        # Test initializing new stimulus
        result = exp._stim_init(mock_stim_class, stims)
        assert "test_stim" in result
        mock_stim_class.init.assert_called_once_with(exp)

        # Test reusing existing stimulus
        mock_stim_class.reset_mock()
        result = exp._stim_init(mock_stim_class, result)
        mock_stim_class.init.assert_not_called()

    def test_get_keys_from_dict(self):
        """Test dictionary key extraction."""
        # Import inside the function to ensure mocking is in place
        with patch('ethopy.core.logger.Logger._log_setup_info'):
            from ethopy.core.experiment import ExperimentClass

            exp = ExperimentClass()
            test_dict = {"a": 1, "b": 2, "c": 3}

            # Test extracting existing keys
            result = exp.get_keys_from_dict(test_dict, ["a", "b"])
            assert result == {"a": 1, "b": 2}

            # Test with non-existent keys
            result = exp.get_keys_from_dict(test_dict, ["a", "d"])
            assert result == {"a": 1}

            # Test with empty keys list
            result = exp.get_keys_from_dict(test_dict, [])
            assert result == {}

    def test_validate_condition(self):
        """Test condition validation."""
        # Import inside the function to ensure mocking is in place
        with patch('ethopy.core.logger.Logger._log_setup_info'):
            from ethopy.core.experiment import ExperimentClass

            exp = ExperimentClass()
            exp.required_fields = ["field1", "field2"]

            # Test valid condition
            valid_condition = {"field1": "value1", "field2": "value2"}
            exp.validate_condition(valid_condition)  # Should not raise

            # Test invalid condition
            invalid_condition = {"field1": "value1"}
            with pytest.raises(ValueError) as exc_info:
                exp.validate_condition(invalid_condition)
            assert "Missing experiment required fields" in str(exc_info.value)

    def test_push_conditions(self):
        """Test condition pushing and initialization."""
        # Import inside the function to ensure mocking is in place
        with patch('ethopy.core.logger.Logger._log_setup_info'):
            from ethopy.core.experiment import ExperimentClass

            exp = ExperimentClass()
            exp.params = {}
            conditions = [
                {"difficulty": 1, "response_port": "left"},
                {"difficulty": 2, "response_port": "right"},
            ]

            with patch("numpy.random.choice") as mock_choice:
                mock_choice.return_value = conditions[0]
                exp.push_conditions(conditions)

                assert exp.conditions == conditions
                assert exp.cur_block == 1
                assert exp.curr_cond == conditions[0]
                assert len(exp.un_choices) > 0
                assert len(exp.blocks) == len(conditions)

    def test_prepare_trial(self):
        """Test trial preparation."""
        # Import inside the function to ensure mocking is in place
        from ethopy.core.experiment import ExperimentClass

        exp = ExperimentClass()
        exp.curr_cond = {
            "trial_selection": "random",
            "stimulus_class": "test_stim",
            "cond_hash": "123",
        }
        exp.conditions = [exp.curr_cond]

        # Setup mock logger
        exp.logger = Mock()
        exp.logger.logger_timer.elapsed_time.return_value = 1000
        exp.logger.thread_end.is_set.return_value = False
        exp.logger.log.return_value = True

        # Setup mock stimulus
        mock_stim = Mock()
        exp.stims = {"test_stim": mock_stim}
        exp.stim = mock_stim

        exp.prepare_trial()
        assert exp.curr_trial == 1
        assert exp.trial_start == 1000
        exp.logger.update_trial_idx.assert_called_once_with(1)
        exp.logger.log.assert_called_once()
        assert exp.in_operation

    def test_prepare_trial_quit_condition(self):
        """Test trial preparation when should quit."""
        # Import inside the function to ensure mocking is in place
        with patch('ethopy.core.logger.Logger._log_setup_info'):
            from ethopy.core.experiment import ExperimentClass

            exp = ExperimentClass()
            exp.curr_cond = {"trial_selection": "random"}
            exp.conditions = [exp.curr_cond]
            exp.logger = Mock()

            exp.prepare_trial()

            assert exp.quit
            assert exp.logger.update_trial_idx.call_count == 0
