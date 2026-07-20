from typing import Any, Dict, List

import numpy as np

# Field values that split a single condition into several table rows.
# Strings and numpy scalars are deliberately excluded: they are single values.
_SEQUENCE_TYPES = (list, tuple, np.ndarray)


def expand_condition_rows(
    condition: Dict[str, Any], fields: set, core: List[str]
) -> List[Dict[str, Any]]:
    """Turn one condition into the list of table rows it describes.

    A condition usually maps to a single row. When a primary key holds a
    sequence (list/tuple/array) it instead describes several rows at once, for
    example one row per response port, all sharing the same ``cond_hash``.
    Every sequence field is then split in parallel by index, and every scalar
    field is repeated in each row.

    Expansion is triggered only by a sequence in a *primary* key (``core``): the
    rows must differ in their primary key to be distinct, so a sequence in a
    non-primary field alone is left untouched (it would create duplicate keys).

    Args:
        condition: The condition; already holds every name in ``fields``.
        fields: All column names of the target table.
        core: The non-hash primary key names of the target table.

    Returns:
        One dict per row, a single-element list when there is nothing to expand.

    Raises:
        ValueError: if the sequence fields do not all share the same length.
    """
    def is_sequence(value: Any) -> bool:
        return isinstance(value, _SEQUENCE_TYPES)

    if not any(is_sequence(condition[k]) for k in core):
        return [condition]

    lengths = {k: len(condition[k]) for k in fields if is_sequence(condition[k])}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Condition has sequence fields of unequal length: {lengths}. "
            "All sequence-valued fields in one condition must share one length."
        )

    n_rows = next(iter(lengths.values()))
    return [
        {k: condition[k][idx] if is_sequence(condition[k]) else condition[k]
         for k in fields}
        for idx in range(n_rows)
    ]


def get_parameters(_class):
    """Create a dictionary with required fields set to '...' and default values included.

    Args:
        _class (class): A class object to extract required fields and default values.

    Returns:
        dict: A dictionary containing all keys with required fields set to '...'
        and defaults included.

    """
    required_fields = _class.required_fields
    default_key = _class.default_key
    parameters = {key: "..." for key in required_fields}  # Required fields with '...'
    parameters.update(default_key)  # Merge with default keys
    return parameters


def format_params_print(parameters):
    # Pretty print while preserving np.array format
    formatted_string = "{\n"
    for key, value in parameters.items():
        if isinstance(value, np.ndarray):
            formatted_value = (
                f"np.array({repr(value.tolist())})"  # Keep np.array format
            )
        elif isinstance(value, tuple) and any(isinstance(v, np.ndarray) for v in value):
            formatted_value = f"({', '.join(f'np.array(...)' if isinstance(v, np.ndarray) else repr(v) for v in value)})"
        else:
            formatted_value = repr(value)

        formatted_string += f"    '{key}': {formatted_value},\n"
    formatted_string += "}"
    return formatted_string


if __name__ == "__main__":
    from ethopy.behaviors.multi_port import MultiPort
    from ethopy.experiments.match_port import Experiment
    from ethopy.stimuli.grating import Grating

    parameters_gr = get_parameters(Grating())
    parameters_exp = get_parameters(Experiment())
    parameters_mp = get_parameters(MultiPort())
    print(
        "All default and required parameters\nneeded for Grating, MatchPort and MultiPort:\n",
        format_params_print({**parameters_gr, **parameters_exp, **parameters_mp}),
    )
