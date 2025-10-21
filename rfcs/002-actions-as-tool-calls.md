# RFC: Support multiple tool calls via Action wrapper abstraction

**Status**: In Review
**Created**: 10/15/2025
**Authors**: @Darktex, @pankit-eng
**RFC ID**: 002

## Summary

This RFC proposes treating environment actions as tool calls, introducing a standardized pattern where each action represents a discrete, named operation with typed parameters. This approach aligns OpenEnv with modern LLM agent frameworks while maintaining type safety and providing better introspection capabilities for agent training and debugging.

Instead of arbitrary `Action` subclasses with domain-specific fields, actions would follow a tool-call pattern with a `tool_name` and structured `parameters`, making the framework more composable and easier to integrate with tool-using agents.

## Motivation

### Problem Statement

Current action design in OpenEnv treats actions as dataclasses:

```python
@dataclass
class CodeAction(Action):
    code: str

@dataclass
class BashAction(Action):
    command: str
    cwd: Optional[str] = None
```

This approach has several limitations:

1. **Lack of Introspection**: No standard way to discover what actions an environment supports
2. **LLM Integration Friction**: Modern LLM agents use tool-calling patterns with JSON schemas, requiring translation layers
3. **Inconsistent Patterns**: Each environment invents its own action structure without standardization

### Goals

1. **Standardize Action Structure**: Define a consistent pattern for representing actions as tool calls
2. **Enable Tool Discovery**: Provide APIs to introspect available tools in an environment
3. **Improve LLM Integration**: Native compatibility with tool-calling patterns used by Claude, GPT-4, and other models
4. **Maintain Type Safety**: Preserve strong typing while adopting the tool-call pattern
5. **Support Multi-Tool Environments**: Enable environments that expose multiple tools naturally

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Agent/RL Code                        │
│                                                         │
│  # Tool discovery                                       │
│  tools = env.tools()                                    │
│  # -> [ToolDefinition(name="execute_code", ...)]        │
│                                                         │
│  # Execute tool call                                    │
│  action = ToolCallAction(                               │
│      tool_name="execute_code",                          │
│      parameters={"code": "print('Hello')"}              │
│  )                                                      │
│  observation = env.step(action)                         │
└─────────────────────────────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────┐
│              Environment (Docker Container)             │
│                                                         │
│  class PythonCodeActEnv(Environment):                   │
│                                                         │
│      @tool("execute_code")                              │
│      def execute_code(self, code: str) -> CodeResult:   │
│          return self._executor.run(code)                │
│                                                         │
│      def step(self, action: ToolCallAction):            │
│          tool_fn = self._get_tool(action.tool_name)     │
│          result = tool_fn(**action.parameters)          │
│          return self._make_observation(result)          │
└─────────────────────────────────────────────────────────┘
```

### Core Abstractions

#### 1. ToolCallAction

```python
from typing import Any, Dict
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class ToolCallAction(Action):
    """Action representing a tool call with name and parameters.

    This is the standard action type for tool-based environments.
    Environments can support multiple tools by dispatching based on tool_name.
    """

    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
```

#### 2. ToolDefinition

```python
from typing import Any, Callable, Dict, List
from dataclasses import dataclass

@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # JSON Schema type: "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None

@dataclass
class ToolDefinition:
    """Specification of a tool that can be called in an environment.

    This follows the format used by Claude, OpenAI, and other LLM providers
    for function calling, making it easy to pass directly to model APIs.
    """

    name: str
    description: str
    parameters: List[ToolParameter]

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }
```

#### 3. Enhanced Environment Interface

```python
from typing import List, Optional

class Environment(ABC):
    """Base class for all environment servers."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Take a step in the environment."""
        pass

    @property
    @abstractmethod
    def state(self) -> State:
        """Get current environment state."""
        pass

    def tools(self) -> List[ToolDefinition]:
        """Return list of available tools in this environment.

        For backward compatibility, environments that don't implement
        tool-based actions can return an empty list.
        """
        return []
```

### Key Design Decisions

#### Decision 1: Unified Action Type vs. Per-Tool Action Classes

**Chosen Approach**: Use a single `ToolCallAction` class with `tool_name` and `parameters` fields rather than creating separate action classes per tool.

**Rationale**:
- **Simplicity**: Single action type is easier to understand and work with
- **Flexibility**: Adding new tools doesn't require new action classes
- **LLM Compatibility**: Matches the structure used by for MCP tool calling
- **Type Safety**: JSON Schema validation can still enforce parameter types
- **Composability**: Multi-tool environments work naturally

**Trade-offs**:
- Advantages:
  - Less boilerplate (no action class per tool)
  - Natural support for dynamic tool sets
- Disadvantages:
  - Tool Parameters are `Dict[str, Any]` instead of strongly-typed fields

#### Decision 2: Tool Discovery via `tools()` Method

**Chosen Approach**: Add a `tools()` method to the `Environment` base class that returns `List[ToolDefinition]`.

**Rationale**:
- **Introspection**: Agents can discover what actions are available
- **LLM Integration**: Tool definitions can be passed directly to LLM APIs
- **Documentation**: Self-documenting environments via decorator pattern for declaring tools.


## Examples

### Example 1: Simple Single-Tool Environment

```python
from core.env_server import Environment, Observation, State, ToolCallAction
from core.tools import PyExecutor

class PythonCodeActEnv(Environment):
    """Environment for executing Python code via tool calls."""

    def __init__(self):
        self._executor = PyExecutor()
        self._state = CodeState()

    @tool("execute_code", "Execute Python code and return stdout, stderr, and exit code")
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code.

        Args:
            code: Python code to execute

        Returns:
            Dict with stdout, stderr, and exit_code keys
        """
        result = self._executor.run(code)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }

    def reset(self) -> Observation:
        self._state = CodeState(episode_id=str(uuid.uuid4()))
        return CodeObservation(stdout="", stderr="", exit_code=0)

    def step(self, action: Action) -> Observation:
        if not isinstance(action, ToolCallAction):
            raise ValueError(f"Expected ToolCallAction, got {type(action)}")

        # Dispatch to tool method
        if action.tool_name == "execute_code":
            result = self.execute_code(**action.parameters)
            reward = 1 if result["exit_code"] == 0 else -1
            self._state.step_count += 1
            return CodeObservation(reward=reward, **result)
        else:
            raise ValueError(f"Unknown tool: {action.tool_name}")

    @property
    def state(self) -> State:
        return self._state
```


### Example 2: Client-Side Usage with LLM

```python
from anthropic import Anthropic
from envs.coding_env import CodingEnv

# Initialize environment
env = CodingEnv.from_docker_image("coding-env:latest")

# Get available tools
tools = env.tools()  # Returns List[ToolDefinition]

# Convert to Claude's tool format
claude_tools = [tool.to_json_schema() for tool in tools]

# Initialize Claude client
client = Anthropic()

# Agent loop
observation = env.reset()
messages = [{"role": "user", "content": "Calculate fibonacci(10)"}]

while not observation.done:
    # Get model response with tools
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        tools=claude_tools,
    )

    # If model wants to use a tool
    if response.stop_reason == "tool_use":
        tool_use = response.content[0]

        # Create action from tool call
        action = ToolCallAction(
            tool_name=tool_use.name,
            parameters=tool_use.input,
            tool_call_id=tool_use.id,
        )

        # Execute in environment
        observation = env.step(action)

        # Add tool result to messages
        messages.append({
            "role": "assistant",
            "content": response.content,
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(observation),
            }],
        })
        print(observation.reward)
    else:
        break

env.close()
```

## References

- [Anthropic Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- RFC 001: OpenEnv Framework Specification
