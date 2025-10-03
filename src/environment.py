# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional

from .interfaces import Environment, Transform, ToolRegistry
from .types import (
    Action,
    CodeAction,
    CodeObservation,
    CodeState,
    ExecutionResult,
)


class PythonExecutor:
    """Executes Python code in a persistent namespace with output capture."""

    def __init__(self, initial_globals: Optional[Dict[str, Any]] = None):
        self.globals = initial_globals or {'__builtins__': __builtins__}

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code and capture all outputs."""
        start_time = time.perf_counter()

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(
                stderr_capture
            ):
                lines = code.strip().split('\n')
                last_line = lines[-1].strip() if lines else ''

                if len(lines) == 1:
                    # Single line - try as expression first
                    try:
                        compiled = compile(code, '<codeact>', 'eval')
                        result = eval(compiled, self.globals)
                        return ExecutionResult.from_success(
                            return_value=result,
                            stdout=stdout_capture.getvalue(),
                            stderr=stderr_capture.getvalue(),
                            execution_time_ms=(
                                time.perf_counter() - start_time
                            ) * 1000
                        )
                    except SyntaxError:
                        pass

                # Multi-line or failed expression - handle statements with
                # optional expression
                result = None
                keywords = [
                    'def ', 'class ', 'if ', 'for ', 'while ', 'with ',
                    'try:', 'import ', 'from '
                ]
                if last_line and not any(
                    last_line.startswith(kw) for kw in keywords
                ):
                    try:
                        statements = '\n'.join(lines[:-1])
                        if statements:
                            exec(
                                compile(statements, '<codeact>', 'exec'),
                                self.globals
                            )
                        result = eval(
                            compile(last_line, '<codeact>', 'eval'),
                            self.globals
                        )
                    except (SyntaxError, ValueError):
                        exec(compile(code, '<codeact>', 'exec'), self.globals)
                else:
                    exec(compile(code, '<codeact>', 'exec'), self.globals)

                elapsed_time = (time.perf_counter() - start_time) * 1000
                return ExecutionResult.from_success(
                    return_value=result,
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue(),
                    execution_time_ms=elapsed_time
                )

        except Exception as exc:
            return ExecutionResult.from_exception(
                exc=exc,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )

    def reset(self):
        """Reset execution environment."""
        self.globals = {'__builtins__': __builtins__}

    def add_tool(self, name: str, tool_obj: Any):
        """Add a tool to the execution environment."""
        self.globals[name] = tool_obj

    def get_available_names(self) -> List[str]:
        """Get available names in the execution context."""
        return [
            name for name in self.globals.keys() if not name.startswith('_')
        ]


class CodeActEnvironment(Environment):
    """Environment for executing arbitrary Python code actions.

    This environment allows agents to execute Python code that can:
    - Perform computations
    - Call tools/functions made available in the environment
    - Maintain state across steps within an episode
    - Chain multiple operations in a single action

    Args:
        tools: Dictionary of tools to make available
        transform: Optional transform to apply to observations
    """

    def __init__(
        self,
        tools: Optional[Dict[str, Any]] = None,
        transform: Optional[Transform] = None,
    ):
        super().__init__(transform=transform)

        self.tool_registry = ToolRegistry()
        self.executor = PythonExecutor()
        self._state = CodeState()

        # Register tools
        if tools:
            for name, tool in tools.items():
                self.tool_registry.register(name, tool)
                self.executor.add_tool(name, tool)

    def reset(self) -> CodeObservation:
        """Reset the environment and start a new episode."""
        episode_id = str(uuid.uuid4())

        # Reset executor and state
        self.executor.reset()
        self._state = CodeState(episode_id=episode_id)

        # Re-add tools after reset
        for name, tool in self.tool_registry.get_all().items():
            self.executor.add_tool(name, tool)

        observation = CodeObservation(
            available_tools=self.tool_registry.get_names(),
            metadata={'episode_id': episode_id}
        )

        return self._apply_transform(observation)

    def step(self, action: Action) -> CodeObservation:
        """Execute a code action and return results."""
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        # Update state
        self._state.step_count += 1
        self._state.action_history.append(action)

        # Execute code
        execution_result = self.executor.execute(action.code)
        self._state.result_history.append(execution_result)

        # Create observation
        observation = CodeObservation(
            execution_result=execution_result,
            available_tools=self.executor.get_available_names(),
            metadata={
                'episode_id': self._state.episode_id,
                'step_count': self._state.step_count,
                'last_code': action.code,
                **action.metadata
            }
        )

        return self._apply_transform(observation)

    @property
    def state(self) -> CodeState:
        """Get current environment state."""
        self._state.execution_globals = self.executor.globals.copy()
        return self._state

    def add_tool(self, name: str, tool_obj: Any):
        """Add a tool at runtime."""
        self.tool_registry.register(name, tool_obj)
        self.executor.add_tool(name, tool_obj)


def create_codeact_env(
    tools: Optional[Dict[str, Any]] = None
) -> CodeActEnvironment:
    """Create a CodeAct environment with standard Python libraries."""
    import math
    import random
    import json
    import re
    from datetime import datetime

    standard_tools = {
        'math': math,
        'random': random,
        'json': json,
        're': re,
        'datetime': datetime,
        'print': print,
    }

    if tools:
        standard_tools.update(tools)

    return CodeActEnvironment(tools=standard_tools)
