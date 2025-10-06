# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CodingEnv: A concrete environment implementation for learning to code."""

from ...core.env.code_execution_environment import CodeExecutionEnvironment
from ...core.env.interfaces import Transform
from ...core.env.types import Action, Observation
from .transforms import create_safe_coding_transform


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
        use_safety_transforms: bool = True
    ):
        # If no transform provided but safety transforms requested, use default
        if transform is None and use_safety_transforms:
            transform = create_safe_coding_transform()

        super().__init__(
            transform=transform,
            docker_image=docker_image,
            timeout_seconds=timeout_seconds
        )

    def step(self, action: Action) -> Observation:
        """Override step to add code to observation metadata for transforms."""
        # Store the code in metadata so transforms can access it
        if hasattr(action, 'code'):
            # Execute the step
            observation = super().step(action)
            # Add code to metadata for transforms
            observation.metadata['last_code'] = action.code
            # Re-apply transforms now that metadata is populated
            return self._apply_transform(observation)
        else:
            return super().step(action)