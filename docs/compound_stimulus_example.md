# Creating a Compound Stimulus

A **compound** (or multimodal) stimulus presents more than one modality in the same trial, a tone together with a grating, a sound together with a 3D object, and stores the parameters of every modality in the same trial condition.

This guide shows how to build one, and what to be careful about. It assumes you have read [Creating a Custom Stimulus](dot_stimulus_example.md) first.

## How compound stimuli work in EthoPy

There is no container that holds several stimulus objects. The state machine drives exactly **one** stimulus instance per trial:

```python
# ethopy/core/experiment.py
self.stim = self.stims[self.curr_cond["stimulus_class"]]
```

So a compound stimulus is a *single class* that:

1. **Declares several condition tables** in `cond_tables`, one per modality.
2. **Merges the parameter contracts** (`required_fields`, `default_key`) of the modalities it combines.
3. **Drives all modalities from its own lifecycle methods** (`start`, `present`, `stop`, `exit`).

In practice you subclass the *dominant* modality, the one with the heavy machinery, usually the visual one, and add the second modality on top. The second modality's class is typically **not** instantiated; you reuse its condition table and call the interface directly.

!!! note "Compound stimulus vs. stimulus periods"
    A compound stimulus presents **several modalities at once**, in one trial.

    If instead you want the **same** stimulus class presented with **different parameters at different points** of a trial, you don't need a compound stimulus, use stimulus periods:

    ```python
    conditions += exp.make_conditions(
        stim_class=Panda(),
        conditions={**block.dict(), **key},
        stim_periods=["Cue", "Response"],
    )
    ```

    Each period is logged separately in `StimCondition.Trial.period`.

## A minimal example

We combine an auditory `Tones` stimulus with the built-in visual `Grating`.

### 1. The component stimulus

`Tones` is an ordinary stimulus with its own table, nothing compound about it yet:

```python
# ~/.ethopy/ethopy_plugins/stimuli/tones.py
import datajoint as dj

from ethopy.core.logger import stimulus
from ethopy.core.stimulus import Stimulus


@stimulus.schema
class Tones(Stimulus, dj.Manual):
    definition = """
    # This class handles the presentation of Tones
    -> stimulus.StimCondition
    ---
    tone_duration    : int     # tone duration (ms)
    tone_frequency   : int     # tone frequency (hz)
    tone_volume      : int     # tone volume (percent)
    tone_pulse_freq  : float   # frequency of tone pulses (hz)
    """

    def __init__(self):
        super().__init__()
        self.cond_tables = ["Tones"]
        self.required_fields = ["tone_duration", "tone_frequency"]
        self.default_key = {"tone_volume": 50, "tone_pulse_freq": 0}
```

After adding a new condition table, create it in the database:

```bash
ethopy-setup-schema
```

### 2. The compound stimulus

```python
# ~/.ethopy/ethopy_plugins/stimuli/tones_grating.py
from ethopy.stimuli.grating import Grating


class TonesGrating(Grating):
    """Presents a Grating and a Tone in the same trial."""

    def __init__(self):
        super().__init__()  # installs Grating's cond_tables / required_fields / default_key

        # 1. add the second modality's condition table
        self.cond_tables += ["Tones"]

        # 2. merge the parameter contract
        self.required_fields += ["tone_duration", "tone_frequency"]
        self.default_key.update({"tone_volume": 50, "tone_pulse_freq": 0})

        # 3. per-modality state
        self.sound_in_operation = False
        self.grating_in_operation = False

    def start(self):
        self.sound_in_operation = True
        self.grating_in_operation = True
        self.exp.interface.give_sound(
            self.curr_cond["tone_frequency"],
            self.curr_cond["tone_volume"],
            self.curr_cond["tone_pulse_freq"],
        )
        super().start()  # logs the start time and starts the timer

    def present(self):
        elapsed = self.timer.elapsed_time()

        if elapsed > self.curr_cond["tone_duration"] and self.sound_in_operation:
            self.exp.interface.stop_sound()
            self.sound_in_operation = False

        if elapsed > self.curr_cond["duration"] and self.grating_in_operation:
            self.grating_in_operation = False

        # the trial ends only when BOTH modalities are done
        if not self.sound_in_operation and not self.grating_in_operation:
            self.in_operation = False
        elif self.grating_in_operation:
            super().present()

    def stop(self):
        super().stop()  # Grating.stop() -> fill, log_stop, close movie
        self.exp.interface.stop_sound()

    def exit(self):
        self.exp.interface.stop_sound()
        super().exit()
```

Building the parameter contract *additively* on top of `super().__init__()` is deliberate, see [Copy the parent's whole contract](#copy-the-parents-whole-contract).

### 3. The task

Nothing special is required in the task file, parameters of both modalities go into the same condition dictionary:

```python
from ethopy.behaviors.multi_port import MultiPort
from ethopy.experiments.match_port import Experiment
from ethopy.stimuli.tones_grating import TonesGrating

exp = Experiment()
exp.setup(logger, MultiPort, {"setup_conf_idx": 0, "max_reward": 3000})

key = {
    # Tones
    "tone_duration": 3000,
    "tone_frequency": 40000,
    "tone_volume": 50,
    # Grating
    "duration": 3000,
    "contrast": 80,
    "spatial_freq": 0.05,
    # trial control
    "trial_duration": 5000,
    "reward_amount": 8,
}

conditions = []
block = exp.Block(difficulty=1, next_up=1, next_down=1, trial_selection="staircase")
for port, theta in {1: 0, 2: 90}.items():
    conditions += exp.make_conditions(
        stim_class=TonesGrating(),
        conditions={**block.dict(), **key, "theta": theta,
                    "reward_port": port, "response_port": port},
    )

exp.push_conditions(conditions)
exp.start()
```

## The three things you must get right

### Condition tables

`cond_tables` lists the DataJoint tables in the `stimulus` schema that store this stimulus's parameters. `Stimulus.make_conditions` writes to all of them:

```python
conditions = self.exp.log_conditions(
    conditions,
    schema="stimulus",
    hash_field="stim_hash",
    condition_tables=["StimCondition"] + self.cond_tables,
)
```

The `stim_hash` is computed over the **union of the fields of every listed table**. A compound stimulus therefore produces **one hash per condition**, with **one row in each component table** under that same hash.

### Required fields and defaults

`required_fields` and `default_key` are not only validation. Together they are the **filter** deciding which task parameters reach the stimulus:

```python
# ethopy/core/experiment.py
stim_dict = self.get_keys_from_dict(conditions, get_parameters(stim_class).keys())
```

where `get_parameters()` returns `required_fields ∪ default_key.keys()`. A key that appears in neither is **not** part of the stimulus condition: it is reported as an unused parameter and is never written to a condition table, so it does not contribute to the `stim_hash` either. It is still visible in `curr_cond` at run time, which makes this easy to miss — the stimulus behaves as intended during the session, and the parameter is simply absent when you come back to analyse the data.

- `required_fields` — must be supplied by the task; `make_conditions` asserts on them.
- `default_key` — filled in when the task omits them.

### Class naming

The class name is stored as `stimulus_class` in the `Condition` table and is the key into `exp.stims`, so it must be **unique across the stimuli used in one session**.

Follow the plugin conventions for the module: a snake_case file under `stimuli/`,
imported as `ethopy.stimuli.<module>`.

## What to be careful about

### Set the three attributes inside `__init__`

`Stimulus.__init__` resets `cond_tables`, `required_fields` and `default_key` to empty. Declaring them as **class attributes** therefore does nothing — the instance attributes created by `super().__init__()` shadow them:

```python
class MyStimulus(Stimulus, dj.Manual):
    cond_tables = ["MyStimulus"]      # WRONG - silently ignored

    def __init__(self):
        super().__init__()            # resets cond_tables to []
```

The failure is silent and nasty: with empty `cond_tables` the hash is computed over zero fields, so **every condition gets the same `stim_hash`**, and no parameters are stored. Always assign after `super().__init__()`:

```python
    def __init__(self):
        super().__init__()
        self.cond_tables = ["MyStimulus"]     # correct
```

### Copy the parent's whole contract

When you subclass a stimulus, `super().__init__()` already installs the parent's `cond_tables`, `required_fields` and `default_key`. **Extend** them rather than reassigning:

```python
self.cond_tables += ["Tones"]                   # keeps Grating's tables
self.default_key.update({"tone_volume": 50})    # keeps Grating's defaults
```

If you reassign instead, you must repeat every parent entry by hand and the class will silently stop accepting any parameter the parent adds later.

Note that `cond_tables` includes **part tables**. A stimulus built on `Panda` must carry all of them:

```python
self.cond_tables = ["Tones", "Panda", "Panda.Object",
                    "Panda.Environment", "Panda.Light", "Panda.Movie"]
```

### A table with missing fields is skipped, not reported as an error

If a condition does not contain every field of a listed table, `log_conditions` skips that table with a warning and carries on:

```
WARNING Skipping Tones, Missing keys:{'tone_pulse_freq'}
```

The trial still runs and still gets a `stim_hash` but one modality's parameters are missing from the database. Watch for this warning on the first run of a new compound stimulus. It usually means a field is in the table definition but in neither `required_fields` nor `default_key`.

### Do not give the compound class its own table unless it adds parameters

A compound class that only aggregates existing tables needs **no** `@stimulus.schema` decorator and no `definition`. If you decorate a subclass that has no `definition` of its own, it inherits the parent's and DataJoint declares a second, permanently empty table.

Add a table only if the *combination* introduces genuinely new parameters, an audiovisual onset asynchrony, say — and then add it to `cond_tables` alongside the others.

### Log the trial exactly once

`log_stop()` writes the `StimCondition.Trial` row and toggles the sync signal (`sync_out(False)`). Calling it more than once per trial fires the sync output twice, the duplicate insert is dropped by the logger, so nothing errors.

The trap is calling it *and* delegating to a parent that calls it too:

```python
    def stop(self):
        self.log_stop()        # once here...
        self.exp.interface.stop_sound()
        super().stop()         # ...and again inside the parent's stop()
```

Pick one: either call `super().stop()` and let the parent log, or handle the whole stop yourself. The same applies to `log_start()` via `super().start()`.

### Decide which modality ends the trial

`self.in_operation` is what the state machine polls to decide the trial is over:

```python
# ethopy/experiments/passive.py
elif not self.stim.in_operation:   # timed out
    return "InterTrial"
```

With several modalities running at different durations, the recommended pattern is a flag per modality, with the shared `in_operation` cleared only once **all** of them have finished, as in the example above. The trial then lasts as long as the longest modality, and each stops at its own duration.

The alternative is to let one modality own the clock and not override `present()` at all. That is simpler, but be aware of the consequence: the other modality's duration parameter is still stored in its condition table while having **no effect** on presentation, it will look meaningful during analysis and not be. If you take this route, document it, and consider leaving the unused duration out of the condition.

### Remember to create the tables

A new component table only exists in the database after:

```bash
ethopy-setup-schema
```

## Checklist

Before running a new compound stimulus:

- [ ] `cond_tables`, `required_fields` and `default_key` are set **inside `__init__`**,
      after `super().__init__()`.
- [ ] `cond_tables` covers every modality, including the parent's part tables.
- [ ] Every field of every listed table is in `required_fields` or `default_key`.
- [ ] The class has no `@stimulus.schema` decorator, unless it defines new parameters.
- [ ] `start()` and `stop()` log exactly once.
- [ ] `in_operation` clears only when every modality is done.
- [ ] `exit()` tears down every modality (screen *and* sound).
- [ ] `ethopy-setup-schema` has been run.

After the first session, verify in the database that each trial produced **one** `stim_hash` with **one row in each component table**, and that no `Skipping <table>, Missing keys` warning appeared in the log.
