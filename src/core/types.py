# Type definitions for EnvTorch
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

# Generic type for observations
ObsT = TypeVar("ObsT")  # TypeVar for typehinting in IDEs


@dataclass
class StepResult(Generic[ObsT]):
    """
    Represents the result of one environment step.

    Attributes:
        observation: The environment's observation after the action.
        reward: Scalar reward for this step (optional).
        done: Whether the episode is finished.
        info: Additional metadata (e.g. debug info, latency, etc.).
    """

    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    info: Optional[dict[str, Any]] = None
