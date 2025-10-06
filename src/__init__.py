# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EnvTorch: Standardized agentic execution environments."""

# Core interfaces and types
from .core.env import (
    Environment, Transform, Tool, ToolRegistry,
    Action, CodeAction, Observation, CodeObservation,
    State, CodeState, ExecutionResult,
    CompositeTransform, NullTransform,
    CodeExecutionEnvironment
)

# Docker execution
from .core.docker import DockerExecutor

# Environment implementations
from .envs import CodingEnv

__version__ = "0.1.0"

__all__ = [
    # Core interfaces
    "Environment", "Transform", "Tool", "ToolRegistry",

    # Types
    "Action", "CodeAction", "Observation", "CodeObservation",
    "State", "CodeState", "ExecutionResult",

    # Base transforms
    "CompositeTransform", "NullTransform",

    # Base environment implementation
    "CodeExecutionEnvironment",

    # Execution engines
    "DockerExecutor",

    # Concrete environment implementations
    "CodingEnv",
]
