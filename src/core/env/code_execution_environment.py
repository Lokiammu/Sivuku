# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import uuid
from typing import Any, Dict, Literal

from ..docker.docker_executor import DockerExecutor
from .interfaces import Environment, Transform
from .types import CodeAction, CodeObservation, CodeState, Action, Observation, State


class CodeExecutionEnvironment(Environment):
    """Environment for executing Python code actions using Docker."""

    def __init__(
        self,
        transform: Transform | None = None,
        docker_image: str = "python:3.11-slim",
        timeout_seconds: int = 30
    ):
        super().__init__(transform)
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.executor = DockerExecutor(docker_image, timeout_seconds)
        self._state = CodeState()

    def reset(self) -> Observation:
        """Reset environment and start fresh Docker session."""
        # Stop any existing session
        self.executor.stop_session()

        # Initialize fresh state
        self._state = CodeState(
            episode_id=str(uuid.uuid4()),
            step_count=0
        )

        # Start new Docker session
        try:
            self.executor.start_session()
        except Exception as e:
            # Fail hard as requested
            raise RuntimeError(f"Failed to start Docker session: {e}")

        # Return initial observation
        observation = CodeObservation(
            execution_result=None,
            available_tools=[]  # TODO: populate from MCP registry
        )

        return self._apply_transform(observation)

    def step(self, action: Action) -> Observation:
        """Execute code action and return observation."""
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        # Execute the code
        execution_result = self.executor.execute_code(action.code)

        # Update state
        self._state.step_count += 1
        self._state.action_history.append(action)
        self._state.result_history.append(execution_result)

        # Create observation
        observation = CodeObservation(
            execution_result=execution_result,
            available_tools=[]  # TODO: populate from MCP registry
        )

        return self._apply_transform(observation)

    def render(self, mode: Literal["human", "raw", "ansi"] = "human") -> Any:
        """Render current environment state."""
        try:
            variables = self.executor.get_variable_dump()
        except Exception as e:
            variables = {"error": f"Failed to get variables: {e}"}

        render_data = {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "variables": variables,
            "last_result": self._state.result_history[-1] if self._state.result_history else None
        }

        if mode == "raw":
            return render_data
        elif mode == "ansi":
            return self._render_ansi(render_data)
        else:  # mode == "human"
            return self._render_human(render_data)

    def close(self) -> None:
        """Close environment and clean up Docker container."""
        self.executor.stop_session()

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state

    def _render_human(self, data: Dict[str, Any]) -> str:
        """Render in human-readable format."""
        lines = []
        lines.append(f"=== Code Environment (Episode: {data['episode_id'][:8]}...) ===")
        lines.append(f"Steps: {data['step_count']}")

        if data.get("last_result"):
            result = data["last_result"]
            lines.append(f"Last execution: {'✓ Success' if result.success else '✗ Failed'}")
            if result.stdout:
                lines.append(f"Output: {result.stdout[:100]}...")
            if not result.success and result.exception_message:
                lines.append(f"Error: {result.exception_message}")

        lines.append("\n--- Variables ---")
        variables = data.get("variables", {})
        if "error" in variables:
            lines.append(f"Error getting variables: {variables['error']}")
        else:
            for name, value in sorted(variables.items()):
                lines.append(f"{name}: {value}")

        return "\n".join(lines)

    def _render_ansi(self, data: Dict[str, Any]) -> str:
        """Render in ANSI terminal format with colors."""
        lines = []

        # ANSI color codes
        BLUE = "\033[34m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        lines.append(f"{BOLD}{BLUE}=== Code Environment ==={RESET}")
        lines.append(f"Episode: {data['episode_id'][:8]}...")
        lines.append(f"Steps: {YELLOW}{data['step_count']}{RESET}")

        if data.get("last_result"):
            result = data["last_result"]
            status_color = GREEN if result.success else RED
            status_text = "Success" if result.success else "Failed"
            lines.append(f"Last execution: {status_color}{status_text}{RESET}")

            if result.stdout:
                lines.append(f"Output: {result.stdout[:100]}...")
            if not result.success and result.exception_message:
                lines.append(f"{RED}Error: {result.exception_message}{RESET}")

        lines.append(f"\n{BOLD}--- Variables ---{RESET}")
        variables = data.get("variables", {})
        if "error" in variables:
            lines.append(f"{RED}Error getting variables: {variables['error']}{RESET}")
        else:
            for name, value in sorted(variables.items()):
                lines.append(f"{YELLOW}{name}{RESET}: {value}")

        return "\n".join(lines)