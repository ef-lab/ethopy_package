import numpy as np


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
