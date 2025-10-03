# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for CodeActEnvironment class.
"""

import pytest
from unittest.mock import Mock, patch
import uuid

from src.environment import CodeActEnvironment, create_codeact_env
from src.types import CodeAction, CodeObservation, CodeState
from src.interfaces import Transform


class TestCodeActEnvironment:
    """Test the CodeActEnvironment class."""

    def test_environment_initialization(self):
        """Test basic environment initialization."""
        env = CodeActEnvironment()

        assert isinstance(env.tool_registry, object)
        assert isinstance(env.executor, object)
        assert isinstance(env._state, CodeState)
        assert env.transform is None

    def test_environment_with_tools(self):
        """Test environment initialization with tools."""
        tools = {'math': __import__('math'), 'test_func': lambda x: x * 2}
        env = CodeActEnvironment(tools=tools)

        # Tools should be registered
        assert 'math' in env.tool_registry.get_all()
        assert 'test_func' in env.tool_registry.get_all()

    def test_environment_with_transform(self):
        """Test environment initialization with transform."""
        mock_transform = Mock(spec=Transform)
        env = CodeActEnvironment(transform=mock_transform)

        assert env.transform is mock_transform

    def test_reset_method(self):
        """Test environment reset functionality."""
        env = CodeActEnvironment()

        obs = env.reset()

        # Should return CodeObservation
        assert isinstance(obs, CodeObservation)

        # Should have episode ID
        assert 'episode_id' in obs.metadata
        assert obs.metadata['episode_id'] is not None

        # State should be reset
        assert env._state.step_count == 0
        assert env._state.episode_id == obs.metadata['episode_id']

    def test_reset_generates_unique_episodes(self):
        """Test that reset generates unique episode IDs."""
        env = CodeActEnvironment()

        obs1 = env.reset()
        obs2 = env.reset()

        assert obs1.metadata['episode_id'] != obs2.metadata['episode_id']

    def test_step_with_valid_action(self):
        """Test stepping with a valid CodeAction."""
        env = CodeActEnvironment()
        env.reset()

        action = CodeAction(code="2 + 3")
        obs = env.step(action)

        assert isinstance(obs, CodeObservation)
        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 5
        assert env._state.step_count == 1

    def test_step_with_invalid_action_type(self):
        """Test stepping with invalid action type raises error."""
        env = CodeActEnvironment()
        env.reset()

        with pytest.raises(ValueError, match="Expected CodeAction"):
            env.step("invalid action")

    def test_step_updates_history(self):
        """Test that step updates action and result history."""
        env = CodeActEnvironment()
        env.reset()

        action = CodeAction(code="x = 42")
        obs = env.step(action)

        assert len(env._state.action_history) == 1
        assert len(env._state.result_history) == 1
        assert env._state.action_history[0] is action
        assert env._state.result_history[0] is obs.execution_result

    def test_step_metadata_includes_code(self):
        """Test that step metadata includes the executed code."""
        env = CodeActEnvironment()
        env.reset()

        code = "print('test')"
        action = CodeAction(code=code)
        obs = env.step(action)

        assert obs.metadata['last_code'] == code

    def test_state_property(self):
        """Test the state property returns current state."""
        env = CodeActEnvironment()
        env.reset()

        # Execute some code to modify state
        env.step(CodeAction(code="x = 42"))

        state = env.state

        assert isinstance(state, CodeState)
        assert state.step_count == 1
        assert 'x' in state.execution_globals

    def test_add_tool_method(self):
        """Test adding tools at runtime."""
        env = CodeActEnvironment()

        def new_tool(x):
            return x ** 2

        env.add_tool("square", new_tool)

        # Tool should be registered
        assert env.tool_registry.get("square") is new_tool

        # Tool should be available in executor
        env.reset()
        obs = env.step(CodeAction(code="square(5)"))
        assert obs.execution_result.return_value == 25

    def test_persistent_state_across_steps(self):
        """Test that variables persist across steps in same episode."""
        env = CodeActEnvironment()
        env.reset()

        # Step 1: Create variable
        obs1 = env.step(CodeAction(code="x = 10"))
        assert obs1.execution_result.success is True

        # Step 2: Use variable
        obs2 = env.step(CodeAction(code="x * 2"))
        assert obs2.execution_result.success is True
        assert obs2.execution_result.return_value == 20

    def test_state_isolation_across_episodes(self):
        """Test that state is isolated across different episodes."""
        env = CodeActEnvironment()

        # Episode 1
        env.reset()
        env.step(CodeAction(code="episode1_var = 42"))

        # Episode 2
        env.reset()
        obs = env.step(CodeAction(code="episode1_var"))

        # Variable from episode 1 should not exist
        assert obs.execution_result.success is False
        assert obs.execution_result.exception_type == "NameError"

    def test_tools_available_after_reset(self):
        """Test that tools remain available after reset."""
        tools = {'test_tool': lambda x: x + 1}
        env = CodeActEnvironment(tools=tools)

        # First episode
        obs1 = env.reset()
        assert 'test_tool' in obs1.available_tools

        result1 = env.step(CodeAction(code="test_tool(5)"))
        assert result1.execution_result.return_value == 6

        # Second episode
        obs2 = env.reset()
        assert 'test_tool' in obs2.available_tools

        result2 = env.step(CodeAction(code="test_tool(10)"))
        assert result2.execution_result.return_value == 11

    def test_transform_application(self):
        """Test that transforms are applied to observations."""
        mock_transform = Mock(spec=Transform)
        mock_transform.return_value = Mock(spec=CodeObservation)

        env = CodeActEnvironment(transform=mock_transform)
        env.reset()

        action = CodeAction(code="2 + 2")
        obs = env.step(action)

        # Transform should have been called
        mock_transform.assert_called()
        # Observation should be the transformed one
        assert obs is mock_transform.return_value

    def test_available_tools_in_observation(self):
        """Test that available tools are included in observations."""
        tools = {'math': __import__('math'), 'json': __import__('json')}
        env = CodeActEnvironment(tools=tools)

        obs = env.reset()
        available_tools = obs.available_tools

        assert 'math' in available_tools
        assert 'json' in available_tools

    def test_action_metadata_preservation(self):
        """Test that action metadata is preserved in observations."""
        env = CodeActEnvironment()
        env.reset()

        metadata = {'source': 'test', 'priority': 'high'}
        action = CodeAction(code="42", metadata=metadata)
        obs = env.step(action)

        assert obs.metadata['source'] == 'test'
        assert obs.metadata['priority'] == 'high'

    def test_error_handling_in_step(self):
        """Test error handling during step execution."""
        env = CodeActEnvironment()
        env.reset()

        action = CodeAction(code="undefined_variable")
        obs = env.step(action)

        assert obs.execution_result.success is False
        assert obs.execution_result.exception_type == "NameError"
        assert obs.execution_result.return_value is None

        # Environment should still be functional
        assert env._state.step_count == 1

    @pytest.mark.edge_case
    def test_extremely_long_code(self):
        """Test handling of very long code strings."""
        env = CodeActEnvironment()
        env.reset()

        # Generate long code string
        long_code = "\n".join([f"var_{i} = {i}" for i in range(1000)])
        long_code += "\nsum([" + ", ".join([f"var_{i}" for i in range(1000)]) + "])"

        action = CodeAction(code=long_code)
        obs = env.step(action)

        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == sum(range(1000))

    @pytest.mark.edge_case
    def test_deeply_nested_structures(self):
        """Test handling of deeply nested data structures."""
        env = CodeActEnvironment()
        env.reset()

        code = """
def create_nested_dict(depth):
    if depth <= 0:
        return "leaf"
    return {"level": depth, "nested": create_nested_dict(depth - 1)}

create_nested_dict(10)
"""
        action = CodeAction(code=code)
        obs = env.step(action)

        assert obs.execution_result.success is True
        result = obs.execution_result.return_value
        assert result["level"] == 10
        assert result["nested"]["level"] == 9


class TestCreateCodeactEnv:
    """Test the create_codeact_env factory function."""

    def test_create_basic_env(self):
        """Test creating environment with default tools."""
        env = create_codeact_env()

        assert isinstance(env, CodeActEnvironment)

        # Check that standard tools are available
        obs = env.reset()
        standard_tools = ['math', 'random', 'json', 're', 'datetime', 'print']

        for tool in standard_tools:
            assert tool in obs.available_tools

    def test_create_env_with_additional_tools(self):
        """Test creating environment with additional tools."""
        custom_tools = {'custom': lambda x: x * 10}
        env = create_codeact_env(tools=custom_tools)

        obs = env.reset()

        # Should have both standard and custom tools
        assert 'math' in obs.available_tools  # Standard tool
        assert 'custom' in obs.available_tools  # Custom tool

    def test_create_env_tool_functionality(self):
        """Test that tools in created environment actually work."""
        env = create_codeact_env()
        env.reset()

        # Test math module
        obs1 = env.step(CodeAction(code="math.sqrt(16)"))
        assert obs1.execution_result.return_value == 4.0

        # Test json module
        obs2 = env.step(CodeAction(code='json.dumps({"key": "value"})'))
        assert obs2.execution_result.return_value == '{"key": "value"}'

        # Test print function
        obs3 = env.step(CodeAction(code="print('hello')"))
        assert "hello" in obs3.execution_result.stdout