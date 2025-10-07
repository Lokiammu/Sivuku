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
    """Base class for all environment servers following Gym/Gymnasium API.

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
