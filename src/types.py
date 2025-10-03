# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import traceback


# Type aliases
Scalar = Union[int, float, bool]


@dataclass(kw_only=True)
class Action:
    """Base class for all environment actions."""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class CodeAction(Action):
    """Action containing Python code to execute in a CodeAct environment."""
    code: str

    def __post_init__(self):
        if not self.code or not self.code.strip():
            raise ValueError("code is required and cannot be empty")


@dataclass
class ExecutionResult:
    """Result of executing Python code."""
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    exception: Optional[Exception] = None
    exception_type: Optional[str] = None
    exception_message: str = ""
    traceback_str: str = ""
    execution_time_ms: float = 0.0
    success: bool = True

    @classmethod
    def from_exception(
        cls, exc: Exception, stdout: str = "", stderr: str = ""
    ) -> "ExecutionResult":
        return cls(
            stdout=stdout,
            stderr=stderr,
            exception=exc,
            exception_type=exc.__class__.__name__,
            exception_message=str(exc),
            traceback_str=traceback.format_exc(),
            success=False
        )

    @classmethod
    def from_success(
        cls,
        return_value: Any = None,
        stdout: str = "",
        stderr: str = "",
        execution_time_ms: float = 0.0
    ) -> "ExecutionResult":
        return cls(
            stdout=stdout,
            stderr=stderr,
            return_value=return_value,
            execution_time_ms=execution_time_ms,
            success=True
        )


@dataclass(kw_only=True)
class Observation:
    """Base class for all environment observations."""
    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class CodeObservation(Observation):
    """Observation from CodeAct environment execution."""
    execution_result: ExecutionResult = field(default_factory=ExecutionResult)
    available_tools: List[str] = field(default_factory=list)


@dataclass
class State:
    """Base class for environment state."""
    episode_id: Optional[str] = None
    step_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeState(State):
    """State for CodeAct environment with persistent execution context."""
    execution_globals: Dict[str, Any] = field(default_factory=dict)
    action_history: List[CodeAction] = field(default_factory=list)
    result_history: List[ExecutionResult] = field(default_factory=list)

    def __post_init__(self):
        if not self.execution_globals:
            self.execution_globals = {'__builtins__': __builtins__}
