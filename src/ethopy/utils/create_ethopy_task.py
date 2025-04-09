"""Ethopy Task Template Generator Script.

This script prompts the user for specific ethopy module paths and class names
(for Experiment, Behavior, and Stimulus components) and generates a
Python (.py) file that serves as a template for an ethopy experiment task,
following a predefined structure.
"""

import datetime
import textwrap

from ethopy.utils.task_helper_funcs import format_params_print, get_parameters


def generate_ethopy_template():
    print("--- Ethopy Task Template Generator ---")
    print("Please provide the details for the ethopy modules you want to use.")

    # --- Get Module Information from User ---
    # Get Experiment details
    exp_module_path = input(
        "1. Enter the experiment module path relative to ethopy "
        "(e.g., experiments.match_port): "
    )
    if not exp_module_path:
        exp_module_path = "experiments.match_port"
    exp_class_name = input(
        f"   Enter the Experiment class name from {exp_module_path} "
        "(e.g., Experiment): "
    )
    if not exp_class_name:
        exp_class_name = "Experiment"
    try:
        exec(f"from ethopy.{exp_module_path} import {exp_class_name}")
    except Exception as e:
        print(f"❌ Error importing {exp_class_name} from {exp_module_path}: {e}")
        return

    exp_params = format_params_print(get_parameters(eval(f"{exp_class_name}()")))
    # Get Behavior details
    beh_module_path = input(
        "2. Enter the behavior module path relative to ethopy "
        "(e.g., behaviors.multi_port): "
    )
    if not beh_module_path:
        beh_module_path = "behaviors.multi_port"
    beh_class_name = input(
        f"   Enter the Behavior class name from {beh_module_path} "
        "(e.g., MultiPort): "
    )
    if not beh_class_name:
        beh_class_name = "MultiPort"
    try:
        exec(f"from ethopy.{beh_module_path} import {beh_class_name}")
    except Exception as e:
        print(f"❌ Error importing {beh_class_name} from {beh_module_path}: {e}")
        return

    beh_params = format_params_print(get_parameters(eval(f"{beh_class_name}()")))

    # Get Stimulus details
    stim_module_path = input(
        "3. Enter the stimulus module path relative to ethopy "
        "(e.g., stimuli.grating): "
    )
    if not stim_module_path:
        stim_module_path = "stimuli.grating"
    stim_class_name = input(
        f"   Enter the Stimulus class name from {stim_module_path} "
        "(e.g., Grating): "
    )
    if not stim_class_name:
        stim_class_name = "Grating"
    try:
        exec(f"from ethopy.{stim_module_path} import {stim_class_name}")
    except Exception as e:
        print(f"❌ Error importing {stim_class_name} from {stim_module_path}: {e}")
        return

    stim_params = format_params_print(get_parameters(eval(f"{stim_class_name}()")))

    # Get output filename
    default_filename = (
        f"task_{stim_module_path.split('.')[-1]}_{datetime.date.today()}.py"
    )
    output_filename = input(
        "4. Enter the desired output filename (press Enter for default:"
        f" {default_filename}): "
    )
    if not output_filename:
        output_filename = default_filename
    if not output_filename.endswith(".py"):
        output_filename += ".py"

    print("\nGenerating template...")
    # --- Define the Template String ---
    # Use textwrap.dedent to handle indentation nicely in the multi-line f-string
    template = textwrap.dedent(f"""\
# --- Ethopy Module Imports ---
# Ensure these paths correctly point to your ethopy installation structure.
from ethopy.{exp_module_path} import {exp_class_name}
from ethopy.{beh_module_path} import {beh_class_name}
from ethopy.{stim_module_path} import {stim_class_name}


# --- Session Parameters (Typically Fixed) ---
# These parameters generally define the overall session constraints.
session_params = {{
    "max_reward": 3000,      # Maximum total reward, default is based on experiment type
    "min_reward": 30,        # Minimum total reward, default is based on experiment type
    "setup_conf_idx": 0,     # Index for setup configuration, default is 0
    # Add any other relevant session-wide parameters here
    # hydrate_delay: int # delay of hydration in minutes after session ends, default is based on experiment type
    # user_name:  "bot" # name of user running the experiment default is "bot"
    # start_time: "" # session start time if not defined, session will start based on control table
    # stop_time: "" # session stop time if not defined, session will stop based on control table
}}

# --- Experiment Initialization and Setup ---
exp = {exp_class_name}()  # Instantiate the experiment controller
exp.setup(logger, {beh_class_name}, session_params)  # Pass the actual imported class

# --- Trial Conditions Definitions ---
# Define the parameters that can vary from trial to trial.
# Split into categories for clarity.

# 1. Experiment Control Conditions: How the trial progresses.
experiment_conditions = {exp_params}

# 2. Behavior Conditions: Related to animal's actions and reinforcement.
behavior_conditions = {beh_params}

# 3. Stimulus Conditions: Parameters for the sensory stimulus presented.
stimulus_conditions = {stim_params}

# --- Combine Conditions for Trial Generation ---
# Merge the dictionaries to create a base set of conditions for a single trial type.
all_conditions = {{
    **experiment_conditions,
    **behavior_conditions,
    **stimulus_conditions
}}

# --- Generate the List of Trial Conditions ---
conditions = []  # Initialize an empty list to hold all trial dictionaries
stim_instance = {stim_class_name}()

# Default: Generate one condition based on the `all_conditions` dict defined above.
conditions += exp.make_conditions(stim_class=stim_instance, conditions=all_conditions)
# --- Push Conditions and Start Experiment ---
exp.push_conditions(conditions)  # Load the trial sequence into the experiment
exp.start()  # Begin the trial sequence execution
    """)

    # --- Write Template to File ---
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(template)
        print(f"\n✅ Successfully generated template file: '{output_filename}'")
        print("\n--- Next Steps ---")
        print(f"1. 👉 Open '{output_filename}' in your editor.")
        print(
            "2. 📝 **Customize the Conditions"
        )

    except IOError as e:
        print(f"\n❌ Error writing file '{output_filename}': {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    generate_ethopy_template()
