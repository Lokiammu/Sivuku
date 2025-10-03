# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for type definitions in types.py
"""

import pytest
from unittest.mock import Mock

from src.types import (
    Action,
    CodeAction,
    ExecutionResult,
    Observation,
    CodeObservation,
    State,
    CodeState,
)


class TestAction:
    """Test the base Action class."""

    def test_action_initialization(self):
        """Test Action can be initialized with metadata."""
        action = Action(metadata={"key": "value"})
        assert action.metadata == {"key": "value"}

    def test_action_default_metadata(self):
        """Test Action has empty metadata by default."""
        action = Action()
        assert action.metadata == {}

    def test_action_metadata_isolation(self):
        """Test that different Action instances have separate metadata."""
        action1 = Action()
        action2 = Action()
        action1.metadata["key"] = "value"
        assert action2.metadata == {}


class TestCodeAction:
    """Test the CodeAction class."""

    def test_codeaction_initialization(self):
        """Test CodeAction initialization with code."""
        action = CodeAction(code="print('hello')")
        assert action.code == "print('hello')"
        assert action.metadata == {}

    def test_codeaction_with_metadata(self):
        """Test CodeAction with both code and metadata."""
        action = CodeAction(
            code="x = 42",
            metadata={"source": "test"}
        )
        assert action.code == "x = 42"
        assert action.metadata == {"source": "test"}

    def test_codeaction_empty_code_error(self):
        """Test that empty code raises ValueError."""
        with pytest.raises(ValueError, match="code is required"):
            CodeAction(code="")

    def test_codeaction_whitespace_only_error(self):
        """Test that whitespace-only code raises ValueError."""
        with pytest.raises(ValueError, match="code is required"):
            CodeAction(code="   \n  \t  ")

    def test_codeaction_valid_whitespace(self):
        """Test that code with meaningful content and whitespace is valid."""
        action = CodeAction(code="  x = 42  \n")
        assert action.code == "  x = 42  \n"


class TestExecutionResult:
    """Test the ExecutionResult class."""

    def test_default_execution_result(self):
        """Test default ExecutionResult values."""
        result = ExecutionResult()
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.return_value is None
        assert result.exception is None
        assert result.exception_type is None
        assert result.exception_message == ""
        assert result.traceback_str == ""
        assert result.execution_time_ms == 0.0
        assert result.success is True

    def test_execution_result_with_values(self):
        """Test ExecutionResult with custom values."""
        result = ExecutionResult(
            stdout="Hello",
            stderr="Warning",
            return_value=42,
            execution_time_ms=1.5,
            success=False
        )
        assert result.stdout == "Hello"
        assert result.stderr == "Warning"
        assert result.return_value == 42
        assert result.execution_time_ms == 1.5
        assert result.success is False

    def test_from_exception_classmethod(self):
        """Test ExecutionResult.from_exception factory method."""
        exc = ValueError("Test error")
        result = ExecutionResult.from_exception(
            exc, stdout="some output", stderr="some error"
        )

        assert result.stdout == "some output"
        assert result.stderr == "some error"
        assert result.exception is exc
        assert result.exception_type == "ValueError"
        assert result.exception_message == "Test error"
        assert result.success is False
        assert "ValueError: Test error" in result.traceback_str

    def test_from_success_classmethod(self):
        """Test ExecutionResult.from_success factory method."""
        result = ExecutionResult.from_success(
            return_value=42,
            stdout="output",
            stderr="",
            execution_time_ms=2.5
        )

        assert result.return_value == 42
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.execution_time_ms == 2.5
        assert result.success is True
        assert result.exception is None

    def test_from_exception_default_outputs(self):
        """Test from_exception with default stdout/stderr."""
        exc = RuntimeError("Test")
        result = ExecutionResult.from_exception(exc)

        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exception_type == "RuntimeError"


class TestObservation:
    """Test the base Observation class."""

    def test_observation_defaults(self):
        """Test default Observation values."""
        obs = Observation()
        assert obs.done is False
        assert obs.reward is None
        assert obs.metadata == {}

    def test_observation_with_values(self):
        """Test Observation with custom values."""
        obs = Observation(
            done=True,
            reward=1.0,
            metadata={"step": 5}
        )
        assert obs.done is True
        assert obs.reward == 1.0
        assert obs.metadata == {"step": 5}

    def test_observation_reward_types(self):
        """Test different reward types."""
        # Boolean reward
        obs1 = Observation(reward=True)
        assert obs1.reward is True

        # Integer reward
        obs2 = Observation(reward=42)
        assert obs2.reward == 42

        # Float reward
        obs3 = Observation(reward=3.14)
        assert obs3.reward == 3.14


class TestCodeObservation:
    """Test the CodeObservation class."""

    def test_codeobservation_defaults(self):
        """Test default CodeObservation values."""
        obs = CodeObservation()
        assert obs.done is False
        assert obs.reward is None
        assert obs.metadata == {}
        assert isinstance(obs.execution_result, ExecutionResult)
        assert obs.available_tools == []

    def test_codeobservation_with_execution_result(self):
        """Test CodeObservation with custom ExecutionResult."""
        exec_result = ExecutionResult(
            stdout="test output",
            return_value=42,
            success=True
        )
        obs = CodeObservation(
            execution_result=exec_result,
            available_tools=["math", "json"]
        )

        assert obs.execution_result is exec_result
        assert obs.available_tools == ["math", "json"]

    def test_codeobservation_inheritance(self):
        """Test that CodeObservation inherits from Observation."""
        obs = CodeObservation(done=True, reward=5.0)
        assert obs.done is True
        assert obs.reward == 5.0


class TestState:
    """Test the base State class."""

    def test_state_defaults(self):
        """Test default State values."""
        state = State()
        assert state.episode_id is None
        assert state.step_count == 0
        assert state.metadata == {}

    def test_state_with_values(self):
        """Test State with custom values."""
        state = State(
            episode_id="test-123",
            step_count=5,
            metadata={"test": True}
        )
        assert state.episode_id == "test-123"
        assert state.step_count == 5
        assert state.metadata == {"test": True}


class TestCodeState:
    """Test the CodeState class."""

    def test_codestate_defaults(self):
        """Test default CodeState values."""
        state = CodeState()
        assert state.episode_id is None
        assert state.step_count == 0
        assert state.metadata == {}
        assert isinstance(state.execution_globals, dict)
        assert state.action_history == []
        assert state.result_history == []

    def test_codestate_post_init(self):
        """Test CodeState.__post_init__ sets up builtins."""
        state = CodeState()
        assert '__builtins__' in state.execution_globals

    def test_codestate_with_custom_globals(self):
        """Test CodeState with pre-existing globals."""
        custom_globals = {'custom_var': 42, '__builtins__': __builtins__}
        state = CodeState(execution_globals=custom_globals)

        # Should not modify existing globals
        assert state.execution_globals is custom_globals
        assert state.execution_globals['custom_var'] == 42

    def test_codestate_history_isolation(self):
        """Test that different CodeState instances have separate histories."""
        state1 = CodeState()
        state2 = CodeState()

        action = CodeAction(code="test")
        state1.action_history.append(action)

        assert len(state1.action_history) == 1
        assert len(state2.action_history) == 0

    def test_codestate_inheritance(self):
        """Test that CodeState inherits from State."""
        state = CodeState(episode_id="test", step_count=3)
        assert state.episode_id == "test"
        assert state.step_count == 3