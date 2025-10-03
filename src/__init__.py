# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
EnvTorch: Agentic Execution Environments

A unified framework for CodeAct environments that supports both agent execution
and RL training through transforms, built on Gym/Gymnasium APIs.
"""

# Core types
from .types import (
    Action,
    CodeAction,
    CodeObservation,
    CodeState,
    ExecutionResult,
    Observation,
    Scalar,
    State,
)

# Core interfaces
from .interfaces import (
    Environment,
    Tool,
    ToolRegistry,
    Transform,
)

# Environment implementation
from .environment import (
    CodeActEnvironment,
    PythonExecutor,
    create_codeact_env,
)

# Transform implementations
from .transforms import (
    CodeQualityTransform,
    CodeSafetyTransform,
    CompositeTransform,
    MathProblemTransform,
    TaskCompletionTransform,
    create_math_env_transform,
    create_safe_env_transform,
)

# MCP integration
from .mcp import (
    MCPClient,
    create_mcp_environment,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Action",
    "CodeAction",
    "CodeObservation",
    "CodeState",
    "ExecutionResult",
    "Observation",
    "Scalar",
    "State",

    # Interfaces
    "Environment",
    "Tool",
    "ToolRegistry",
    "Transform",

    # Core environment
    "CodeActEnvironment",
    "PythonExecutor",
    "create_codeact_env",

    # Transforms
    "CodeQualityTransform",
    "CodeSafetyTransform",
    "CompositeTransform",
    "MathProblemTransform",
    "TaskCompletionTransform",
    "create_math_env_transform",
    "create_safe_env_transform",

    # MCP
    "MCPClient",
    "create_mcp_environment",
]
