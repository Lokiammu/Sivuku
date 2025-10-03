# EnvTorch: Agentic Execution Environments

A unified framework for CodeAct environments that supports both agent execution and RL training, built on Gym/Gymnasium APIs with PyTorch/HuggingFace integration patterns.

## Overview

EnvTorch provides a standard for agentic execution environments following the CodeAct paradigm, where actions are arbitrary Python code that can chain multiple tool calls. The framework bridges traditional RL environments with modern agent capabilities.

### Key Features

- **CodeAct Execution**: Actions are Python code strings executed in persistent contexts
- **State Persistence**: Variables and functions persist across steps within episodes
- **Tool Integration**: MCP (Model Context Protocol) support for external capabilities
- **RL Compatibility**: Transform system for reward computation and training
- **Error Handling**: Exceptions become observations for agent learning
- **Clean APIs**: Minimal, opinionous design following KISS principles

## Quick Start

```python
from src import create_codeact_env, CodeAction

# Create environment
env = create_codeact_env()
obs = env.reset()

# Execute Python code
action = CodeAction(code="""
x = 10
y = 20
result = x * y
print(f"Result: {result}")
result  # Return value
""")

obs = env.step(action)
print(f"Output: {obs.execution_result.stdout}")
print(f"Return: {obs.execution_result.return_value}")
```

## Core Components

### Actions and Observations

```python
# Actions contain arbitrary Python code
action = CodeAction(code="math.sqrt(16)")

# Observations include execution results
obs = env.step(action)
print(obs.execution_result.return_value)  # 4.0
print(obs.execution_result.success)       # True
print(obs.execution_result.stdout)        # Any print output
```

### Tool Integration

```python
from src import create_mcp_environment

# Environment with MCP tools
env = create_mcp_environment()
obs = env.reset()

# Tools available as Python objects
action = CodeAction(code="""
content = "Hello, world!"
file_write("/tmp/hello.txt", content)
result = file_read("/tmp/hello.txt")
print(f"File contents: {result}")
""")

obs = env.step(action)
```

### RL Training with Transforms

```python
from src import create_math_env_transform

# Environment that rewards correct math solutions
transform = create_math_env_transform(expected_answer=42)
env = create_codeact_env()
env.transform = transform

# Agent gets rewarded for correct answers
action = CodeAction(code="21 * 2")  # Correct answer
obs = env.step(action)
print(obs.reward)  # 1.0 (success) + quality bonuses
```

## Architecture

### Type System
- `Action` / `CodeAction`: Base and concrete action types
- `Observation` / `CodeObservation`: Base and concrete observation types
- `State` / `CodeState`: Environment state with execution context
- `ExecutionResult`: Detailed code execution results

### Core Classes
- `Environment`: Base class following Gym API
- `CodeActEnvironment`: Main environment for code execution
- `Transform`: Base class for observation modification
- `ToolRegistry`: Manages available tools and functions

### Transform Examples
- `CodeSafetyTransform`: Penalizes unsafe code patterns
- `MathProblemTransform`: Rewards correct numerical answers
- `CodeQualityTransform`: Evaluates code quality metrics
- `CompositeTransform`: Combines multiple transforms

## File Structure

```
src/
├── types.py          # Core type definitions
├── interfaces.py     # Abstract base classes
├── environment.py    # Main CodeAct environment
├── transforms.py     # Transform implementations
├── mcp.py           # MCP integration
└── __init__.py      # Clean exports
```

## Usage Patterns

### Agent Exploration
```python
env = create_codeact_env()
obs = env.reset()

# Multi-step problem solving
action1 = CodeAction(code="data = [1, 2, 3, 4, 5]")
obs = env.step(action1)

action2 = CodeAction(code="mean = sum(data) / len(data); mean")
obs = env.step(action2)  # Uses persistent data from step 1
```

### RL Training Loop
```python
# Create environment with reward function
transform = create_safe_env_transform()
env = create_codeact_env()
env.transform = transform

for episode in range(100):
    obs = env.reset()
    action = generate_action()  # From your policy
    obs = env.step(action)

    reward = obs.reward  # Computed by transforms
    # Update policy based on reward
```

### Hybrid Agent + RL
```python
# Phase 1: Agent exploration
env = create_codeact_env()
# Agent explores different solution approaches

# Phase 2: RL optimization
env.transform = optimization_transform
# Train to optimize based on exploration insights
```

## Design Principles

- **KISS Approach**: Minimal, opinionated design
- **Single Way**: One clear way to accomplish tasks
- **Pythonic**: Follows PyTorch/HuggingFace patterns
- **No Inline Comments**: Code should be self-explanatory
- **Functional Composition**: Private functions explain complex logic

## Testing

Run the test suite:
```bash
python test_unified.py
```

Run examples:
```bash
python example.py
```

## Requirements

See `requirements.txt` for dependencies. Core requirements:
- Python 3.9+
- PyTorch 2.0+
- HuggingFace datasets

## License

BSD 3-Clause License (see LICENSE file)