# Abstract base classes for EnvTorch
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .types import StepResult

# Generic type variables
ActT = TypeVar("ActT")  # Type for the action sent to the environment
ObsT = TypeVar("ObsT")  # Type for the observation returned by the environment


class BaseEnv(ABC, Generic[ActT, ObsT]):
    """
    Abstract base class for all environments.

    Each environment must implement:
      - reset(): to initialize or reinitialize environment state
      - step(action): to execute an action and return results
      - close(): to release resources (containers, sessions, etc.)
    """

    @abstractmethod
    def reset(self) -> ObsT:
        """
        Resets the environment to its initial state.

        Returns:
            ObsT: The initial observation after resetting.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActT) -> StepResult[ObsT]:
        """
        Performs one logical step in the environment.

        Args:
            action (ActT): The action to perform in the environment.

        Returns:
            StepResult[ObsT]: The resulting observation, reward, done flag, and info.
        """
        raise NotImplementedError
