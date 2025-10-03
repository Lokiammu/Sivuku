# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any

from .types import Action, Observation, State


class Transform(ABC):
    """Transform observations to add rewards, metrics, or other modifications.

    Transforms follow the TorchRL pattern where they take an observation
    and return a (potentially modified) observation. This allows for
    flexible reward computation and observation augmentation.
    """

    @abstractmethod
    def __call__(self, observation: Observation) -> Observation:
        """Transform an observation.

        Args:
            observation: The input observation

        Returns:
            The transformed observation
        """
        pass


class Environment(ABC):
    """Base class for all environments following Gym/Gymnasium API.

    Args:
        transform: Optional transform to apply to observations
    """

    def __init__(self, transform: Transform | None = None):
        self.transform = transform

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
        """Get the current environment state."""
        pass

    def _apply_transform(self, observation: Observation) -> Observation:
        """Apply transform if one is provided."""
        if self.transform is not None:
            return self.transform(observation)
        return observation


class Tool(ABC):
    """Base class for tools that can be used in code execution."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        pass


class ToolRegistry:
    """Registry for managing tools available to code execution."""

    def __init__(self):
        self._tools: dict[str, Any] = {}

    def register(self, name: str, tool: Any):
        """Register a tool with a name."""
        self._tools[name] = tool

    def get(self, name: str) -> Any | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> dict[str, Any]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_names(self) -> list[str]:
        """Get all tool names."""
        return list(self._tools.keys())
