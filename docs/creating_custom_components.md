# Creating Custom Components in Ethopy

This guide provides an introduction to extending Ethopy with your own custom components. Ethopy's modular design allows you to create specialized experiments by implementing three core component types:

1. **Experiments**: Define the overall experimental flow and state transitions
2. **Stimuli**: Create visual, auditory, or other sensory presentations
3. **Behaviors**: Handle animal interactions and responses

## Core Architecture

Ethopy uses a modular architecture where each component has specific responsibilities:


- **Experiments** manage the overall flow of your task using a state machine
- **Stimuli** control what is presented to the subject
- **Behaviors** track and validate responses
- **Interfaces** communicate with hardware
- **Loggers** record data to the database

All components integrate with the DataJoint database system to store parameters and results.

## Example Components

We provide three detailed examples to help you understand how to create your own components:

### 1. [Match Port Experiment](match_port_example.md)

The Match Port experiment implements a 2-Alternative Forced Choice (2AFC) task where animals need to choose the correct port based on stimuli. This example demonstrates:

- Creating a complex state machine with multiple states
- Implementing state transitions based on animal behavior
- Managing adaptive difficulty through staircase methods
- Handling reward, punishment, and intertrial periods

### 2. [Dot Stimulus](dot_stimulus_example.md)

The Dot stimulus provides a simple visual element that can be displayed at different positions and sizes. This example shows:

- Defining stimulus parameters in a DataJoint table
- Calculating screen positions based on monitor resolution
- Managing the lifecycle of visual elements
- Implementing timing-based presentation

### 3. [MultiPort Behavior](multi_port_behavior_example.md)

The MultiPort behavior handles interactions with multiple response ports. This example illustrates:

- Tracking which ports an animal interacts with
- Validating responses based on experimental conditions
- Managing reward delivery at specific ports
- Recording response history and outcomes

## Component Integration

These three component types work together to create a complete experimental setup:

1. The **Experiment** defines the sequence of states (e.g., ready → trial → reward)
2. The **Stimulus** determines what the animal sees or hears in each state
3. The **Behavior** handler tracks and validates the animal's responses

For example, in a typical trial:
- The experiment enters the "Trial" state
- The stimulus presents a visual cue
- The behavior handler detects when the animal responds
- The experiment transitions to "Reward" or "Punish" based on the response
- The cycle continues to the next trial

## Creating Your Own Components

To create your own custom components:

1. **Start with the examples**: Use the provided examples as templates
2. **Understand the lifecycle**: Each component type has specific initialization, operation, and cleanup methods
3. **Define database tables**: Create appropriate DataJoint tables for your parameters
4. **Implement required methods**: Each component type has essential methods that must be implemented
5. **Test incrementally**: Start with simple implementations and add complexity gradually

The detailed documentation for each example provides step-by-step guidance for implementing your own versions.

## Next Steps

Explore each example in detail:

- [Experiment](match_port_example.md) for state machine implementation
- [Stimulus](dot_stimulus_example.md) for visual stimulus creation
- [Behavior](multi_port_behavior_example.md) for response handling

These examples provide a foundation for understanding how to extend Ethopy with custom components tailored to your specific experimental needs.