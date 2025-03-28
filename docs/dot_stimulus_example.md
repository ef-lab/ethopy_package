# Creating a Custom Stimulus: Dot Example

This guide explains how to create a custom stimulus in Ethopy by examining the `dot.py` example, which implements a simple dot stimulus for visual experiments.

## Overview of the Dot Stimulus

The Dot stimulus displays a configurable dot (rectangular or oval) on the screen at specified coordinates and sizes. This type of stimulus is commonly used in:

- Visual attention studies
- Simple detection tasks
- Eye movement tracking experiments
- Timing response experiments

## Defining the Stimulus Database Table

Each stimulus requires defining a database table to store its parameters. Here's how the Dot stimulus defines its table:

```python
@stimulus.schema
class Dot(Stimulus, dj.Manual):
    definition = """
    # This class handles the presentation of area mapping Bar stimulus
    -> stimulus.StimCondition
    ---
    bg_level              : tinyblob  # 0-255 baground color
    dot_level             : tinyblob  # 0-255 dot color
    dot_x                 : float  # (fraction of monitor width, 0 for center, from -0.5 to 0.5) position of dot on x axis
    dot_y                 : float  # (fraction of monitor width, 0 for center) position of dot on y axis
    dot_xsize             : float  # fraction of monitor width, width of dots
    dot_ysize             : float # fraction of monitor width, height of dots
    dot_shape             : enum('rect','oval') # shape of the dot
    dot_time              : float # (sec) time of each dot persists
    """
```

This table definition:
- Uses the `@stimulus.schema` decorator to associate with the database
- Inherits from both `Stimulus` (base class) and `dj.Manual` (DataJoint table class)
- Defines a foreign key relationship with `stimulus.StimCondition` (parent table)
- Specifies parameters for the dot's appearance:
  - Background and dot colors
  - Position (x, y coordinates as fractions of screen width)
  - Size (width and height as fractions of screen width)
  - Shape (rectangular or oval)
  - Duration (how long the dot persists)

## Implementing the Stimulus Class

The Dot class implements several key methods to control the stimulus lifecycle:

```python
def __init__(self):
    super().__init__()
    self.cond_tables = ['Dot']
    self.required_fields = ['dot_x', 'dot_y', 'dot_xsize', 'dot_ysize', 'dot_time']
    self.default_key = {'bg_level': 1,
                        'dot_level': 0,  # degrees
                        'dot_shape': 'rect'}
```

### 1. `__init__()` method

This initializes the stimulus and defines:

- `self.cond_tables`: List of condition tables this stimulus uses (just 'Dot' in this case)
- `self.required_fields`: Parameters that must be provided for this stimulus
- `self.default_key`: Default values for optional parameters

### 2. `prepare()` method

```python
def prepare(self, curr_cond):
    self.curr_cond = curr_cond
    self.fill_colors.background = self.curr_cond['bg_level']
    self.Presenter.set_background_color(self.curr_cond['bg_level'])
    width = self.monitor.resolution_x
    height = self.monitor.resolution_y
    x_start = self.curr_cond['dot_x'] * 2
    y_start = self.curr_cond['dot_y'] * 2 * width/height
    self.rect = (x_start - self.curr_cond['dot_xsize'],
                 y_start - self.curr_cond['dot_ysize']*width/height,
                 x_start + self.curr_cond['dot_xsize'],
                 y_start + self.curr_cond['dot_ysize']*width/height)
```

This method:
1. Stores the current condition parameters
2. Sets the background color
3. Calculates the dot's position and size based on:
   - Monitor resolution (to maintain aspect ratio)
   - Condition parameters for position and size
   - Conversion from normalized coordinates to actual screen coordinates
4. Creates a rectangle tuple (left, top, right, bottom) for drawing

### 3. `start()` method

```python
def start(self):
    super().start()
    self.Presenter.draw_rect(self.rect, self.curr_cond['dot_level'])
```

This method:
1. Calls the parent class's start method (which initializes timing)
2. Draws the rectangle using the precalculated coordinates and the specified color

### 4. `present()` method

```python
def present(self):
    if self.timer.elapsed_time() > self.curr_cond['dot_time']*1000:
        self.in_operation = False
        self.Presenter.fill(self.fill_colors.background)
```

This method:
1. Checks if the dot's display time has elapsed (converting seconds to milliseconds)
2. If the time has elapsed:
   - Marks the stimulus as no longer in operation
   - Fills the screen with the background color (removing the dot)

### 5. `stop()` and `exit()` methods

```python
def stop(self):
    self.log_stop()
    self.Presenter.fill(self.fill_colors.background)
    self.in_operation = False

def exit(self):
    self.Presenter.fill(self.fill_colors.background)
    super().exit()
```

These methods handle cleanup:
- `stop()`: Called when the stimulus needs to be stopped during operation
  - Logs the stop event
  - Clears the screen
  - Marks the stimulus as no longer in operation

- `exit()`: Called when the experiment is ending
  - Clears the screen
  - Calls the parent class's exit method for additional cleanup

## Stimulus Lifecycle

The dot stimulus follows this lifecycle:

1. **Initialization**: Set up parameters and default values
2. **Preparation**: Calculate positioning based on current conditions
3. **Start**: Draw the dot on the screen
4. **Present**: Monitor timing and remove the dot when its time expires
5. **Stop/Exit**: Clean up resources and reset the display

## How to Create Your Own Stimulus

To create your own stimulus:

1. **Define a database table** with appropriate parameters for your stimulus
2. **Create a stimulus class** that inherits from the base `Stimulus` class
3. **Specify required fields and defaults** in the `__init__` method
4. **Implement preparation logic** to set up your stimulus based on conditions
5. **Create presentation methods** that control how the stimulus appears:
   - `start()`: Initial display
   - `present()`: Continuous updates (if needed)
   - `stop()` and `exit()`: Cleanup

Your stimulus should handle:
- **Positioning** on the screen (considering aspect ratio)
- **Timing** of presentation
- **Transitions** stimulus conditions between states
- **Cleanup** to ensure the display returns to a neutral state

By following this pattern, you can create diverse visual stimuli for behavioral experiments, from simple dots and shapes to complex moving patterns or interactive elements.