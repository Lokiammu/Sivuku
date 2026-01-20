# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
envs/kernrl/models.py
---------------------
Action/Observation/State types for the kernrl GPU kernel optimization environment.
"""

from __future__ import annotations

from typing import Optional
from openenv.core.env_server.interfaces import Action, Observation, State


class KernelAction(Action):
    """
    Represents a kernel code submission.
    """
    code: str  # The CUDA/Triton kernel code


class KernelObservation(Observation):
    """
    Observation returned after evaluating a kernel submission.
    """
    problem_id: str
    problem_description: str
    reference_code: str
    gpu_info: str
    turn: int
    max_turns: int
    feedback: str = ""
    # Evaluation results
    compilation_success: bool = False
    compilation_error: Optional[str] = None
    correctness_pass: Optional[bool] = None
    max_diff: Optional[float] = None
    speedup: Optional[float] = None


class KernelState(State):
    """
    State for the kernrl environment.
    """
    problem_id: Optional[str] = None
    turn: int = 0
    max_turns: int = 10
    best_speedup: float = 0.0
    solved: bool = False
