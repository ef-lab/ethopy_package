"""Tests for helpers in ethopy.utils.task_helper_funcs.

expand_condition_rows is a pure function (no database), so these tests import
and call it directly.
"""

import numpy as np
import pytest

from ethopy.utils.task_helper_funcs import expand_condition_rows


class TestExpandConditionRows:
    """Turn one condition into the table rows it describes."""

    def test_scalar_only_returns_condition_unchanged(self):
        """No sequence anywhere -> the single condition, untouched."""
        condition = {"cond_hash": "h", "difficulty": 3, "reward": 5}
        rows = expand_condition_rows(
            condition, {"cond_hash", "difficulty", "reward"}, ["difficulty"]
        )
        assert rows == [condition]

    def test_single_sequence_primary_key_expands_and_repeats_scalars(self):
        """A list primary key -> one row per element; scalar fields repeat."""
        condition = {"cond_hash": "h", "port": [1, 2, 3], "reward": 5}
        rows = expand_condition_rows(
            condition, {"cond_hash", "port", "reward"}, ["port"]
        )
        assert rows == [
            {"cond_hash": "h", "port": 1, "reward": 5},
            {"cond_hash": "h", "port": 2, "reward": 5},
            {"cond_hash": "h", "port": 3, "reward": 5},
        ]

    def test_parallel_sequences_split_by_index(self):
        """Two equal-length sequence keys split together, element by element."""
        condition = {"cond_hash": "h", "port": [1, 2], "loc_x": [0.1, 0.2]}
        rows = expand_condition_rows(
            condition, {"cond_hash", "port", "loc_x"}, ["port", "loc_x"]
        )
        assert rows == [
            {"cond_hash": "h", "port": 1, "loc_x": 0.1},
            {"cond_hash": "h", "port": 2, "loc_x": 0.2},
        ]

    def test_scalar_primary_key_first_still_finds_the_sequence(self):
        """Expansion triggers on any primary key, not just the first one."""
        condition = {"cond_hash": "h", "resp_port": 7, "loc_x": [0.1, 0.2, 0.3]}
        rows = expand_condition_rows(
            condition, {"cond_hash", "resp_port", "loc_x"}, ["resp_port", "loc_x"]
        )
        assert [r["loc_x"] for r in rows] == [0.1, 0.2, 0.3]
        assert all(r["resp_port"] == 7 for r in rows)

    def test_sequence_in_non_primary_field_is_not_expanded(self):
        """A sequence outside the primary key stays whole (no duplicate keys)."""
        condition = {"cond_hash": "h", "label": "a", "blob": [1, 2, 3]}
        rows = expand_condition_rows(
            condition, {"cond_hash", "label", "blob"}, ["label"]
        )
        assert rows == [condition]

    def test_string_primary_key_is_a_single_value(self):
        """Strings are not sequences here -> not expanded per character."""
        condition = {"cond_hash": "h", "stim_type": "grating", "reward": 5}
        rows = expand_condition_rows(
            condition, {"cond_hash", "stim_type", "reward"}, ["stim_type"]
        )
        assert rows == [condition]

    def test_numpy_array_primary_key_expands(self):
        """A numpy array primary key expands like a list."""
        condition = {"cond_hash": "h", "port": np.array([1, 2, 3])}
        rows = expand_condition_rows(condition, {"cond_hash", "port"}, ["port"])
        assert [r["port"] for r in rows] == [1, 2, 3]

    def test_numpy_scalar_field_is_repeated_not_indexed(self):
        """A numpy scalar is a single value, repeated across the expanded rows."""
        condition = {"cond_hash": "h", "port": [1, 2], "seed": np.int64(42)}
        rows = expand_condition_rows(
            condition, {"cond_hash", "port", "seed"}, ["port"]
        )
        assert [r["seed"] for r in rows] == [np.int64(42), np.int64(42)]

    def test_tuple_primary_key_expands(self):
        """Tuples count as sequences too."""
        condition = {"cond_hash": "h", "port": (1, 2)}
        rows = expand_condition_rows(condition, {"cond_hash", "port"}, ["port"])
        assert [r["port"] for r in rows] == [1, 2]

    def test_tuple_secondary_field_is_split_like_a_list(self):
        """A tuple is split element-by-element, even as a secondary field.

        Note: this differs from factorize(), which keeps tuples as one composite
        value.
        """
        condition = {"cond_hash": "h", "port": [1, 2], "coord": (5, 6)}
        rows = expand_condition_rows(
            condition, {"cond_hash", "port", "coord"}, ["port"]
        )
        assert [r["coord"] for r in rows] == [5, 6]

    def test_empty_sequence_expands_to_no_rows(self):
        """An empty sequence primary key produces zero rows (silently)."""
        condition = {"cond_hash": "h", "port": []}
        rows = expand_condition_rows(condition, {"cond_hash", "port"}, ["port"])
        assert rows == []

    def test_mismatched_sequence_lengths_raise(self):
        """Unequal sequence lengths raise a clear error naming the fields."""
        condition = {"cond_hash": "h", "port": [1, 2, 3], "loc_x": [0.1, 0.2]}
        with pytest.raises(ValueError, match="unequal length"):
            expand_condition_rows(
                condition, {"cond_hash", "port", "loc_x"}, ["port", "loc_x"]
            )
