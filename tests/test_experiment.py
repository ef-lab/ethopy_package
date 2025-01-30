import logging
from unittest.mock import Mock, patch

import pytest

from ethopy.core.experiment import ExperimentClass
from ethopy.utils.helper_functions import factorize, make_hash


@pytest.fixture
def mock_logger():
    logger = Mock()
    # Setup mock methods
    logger.get_table_keys.side_effect = lambda schema, table, key_type=None: {
        ("experiment", "Condition", None): {"id", "cond_hash", "value"},
        ("experiment", "Condition", "primary"): {"id", "cond_hash"},
        ("experiment", "TestTable", None): {"test_id", "cond_hash", "value"},
        ("experiment", "TestTable", "primary"): {"test_id", "cond_hash"},
        ("experiment", "TestCustom", None): {"id", "custom_hash", "value"},
    }.get((schema, table, key_type), set())
    return logger


@pytest.fixture
def experiment(mock_logger):
    exp = ExperimentClass()
    exp.logger = mock_logger
    return exp


def test_log_conditions_empty_conditions(experiment):
    """Test handling of empty conditions list"""
    result = experiment.log_conditions([])
    assert result == [], "Empty conditions should return empty list"
    experiment.logger.put.assert_not_called()


def test_log_conditions_single_condition_default_table(experiment):
    """Test logging single condition to default table"""
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


def test_log_conditions_multiple_tables(experiment):
    """Test logging condition to multiple tables"""
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


def test_log_conditions_iterable_primary_key(experiment):
    """Test handling of iterable primary key values"""
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
        print("call[1]['tuple'] ", call[1]["tuple"])
        assert call[1]["tuple"] == expected


def test_log_conditions_missing_required_fields(experiment, caplog):
    """Test handling of conditions missing required fields"""
    condition = {"id": 1}  # Missing 'value' field

    with caplog.at_level(logging.WARNING):
        experiment.log_conditions([condition])

    assert "Missing keys" in caplog.text
    experiment.logger.put.assert_not_called()


def test_log_conditions_custom_hash_field(experiment):
    """Test using custom hash field name"""
    condition = {"id": 1, "value": "test"}
    _ = experiment.log_conditions(
        [condition], condition_tables=["CustomTable"], hash_field="custom_hash"
    )

    call_args = experiment.logger.put.call_args[1]
    assert "custom_hash" in call_args["tuple"]
    assert "cond_hash" not in call_args["tuple"]


def test_log_conditions_hash_generation(experiment):
    """Test that hash is generated correctly from condition fields"""
    condition = {"id": 1, "value": "test"}
    _ = experiment.log_conditions([condition])

    # Get the hash from the logged condition
    call_args = experiment.logger.put.call_args[1]
    generated_hash = call_args["tuple"]["cond_hash"]

    # Calculate expected hash
    expected_hash = make_hash({"id": 1, "value": "test"})

    assert generated_hash == expected_hash
