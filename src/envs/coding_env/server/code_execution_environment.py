# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import uuid

# from ..docker.docker_executor import DockerExecutor
from core.env_server import Action, Environment, Observation, Transform

from ..models import CodeAction, CodeObservation, CodeState
from .transforms import create_safe_coding_transform


class CodeExecutionEnvironment(Environment):
    """Environment for executing Python code actions using Docker."""

    def __init__(
        self,
        transform: Transform | None = None,
        docker_image: str = "python:3.11-slim",
        timeout_seconds: int = 30,
    ):
        super().__init__(transform)
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.executor = None  # TODO: Fix this
        self._state = CodeState()

    def reset(self) -> Observation:
        """Reset environment and start fresh Docker session."""

        # Initialize fresh state
        self._state = CodeState(episode_id=str(uuid.uuid4()), step_count=0)

        # Return initial observation
        # TODO: replace with actual result from the executor
        observation = CodeObservation(
            stderr="",
            stdout="",
            exit_code=0,
        )

        return self._apply_transform(observation)

    def step(self, action: Action) -> Observation:
        """Execute code action and return observation."""
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        # Execute the code
        # Update state
        self._state.step_count += 1

        # Create observation
        # TODO: replace with actual result from the executor
        observation = CodeObservation(
            stderr="",
            stdout="",
            exit_code=0,
        )

        return self._apply_transform(observation)

    def close(self) -> None:
        """Close environment and clean up executor."""
        pass

    @property
    def state(self) -> CodeState:
        """Get current environment state."""
        return self._state


class CodingEnv(CodeExecutionEnvironment):
    """Environment for learning to code with safety and quality evaluation.

    This environment extends the base CodeExecutionEnvironment with transforms
    that evaluate code safety and quality, making it suitable for training
    agents to write safe, high-quality code.
    """

    def __init__(
        self,
        transform: Transform | None = None,
        docker_image: str = "python:3.11-slim",
        timeout_seconds: int = 30,
        use_safety_transforms: bool = True,
    ):
        # If no transform provided but safety transforms requested, use default
        if transform is None and use_safety_transforms:
            transform = create_safe_coding_transform()

        super().__init__(
            transform=transform,
            docker_image=docker_image,
            timeout_seconds=timeout_seconds,
        )

    def step(self, action: Action) -> Observation:
        """Override step to add code to observation metadata for transforms."""
        # Store the code in metadata so transforms can access it
        if hasattr(action, "code"):
            # Execute the step
            observation = super().step(action)
            # Re-apply transforms now that metadata is populated
            return self._apply_transform(observation)
        else:
            return super().step(action)
