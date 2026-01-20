# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
kernrl Client
-------------
Client-side wrapper for the kernrl GPU kernel optimization environment server.

This client maintains a persistent connection to the environment server,
enabling efficient multi-step interactions for kernel optimization.

Usage:
    from openenv.envs.kernrl import kernrl_env, KernelAction

    env = kernrl_env(base_url="http://localhost:8000")
    obs = env.reset(problem_id="L1_23_Softmax")

    action = KernelAction(code='''
    import torch
    import triton
    ...
    ''')
    result = env.step(action)
    print(f"Speedup: {result.observation.speedup}x")
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import KernelAction, KernelObservation, KernelState


class kernrl_env(EnvClient[KernelAction, KernelObservation, KernelState]):
    """
    Client for the kernrl GPU kernel optimization environment.

    Agents submit CUDA/Triton kernel code and receive feedback including:
    - Compilation status and errors
    - Correctness against reference implementation
    - Speedup compared to PyTorch baseline
    - Profiling data from NSight Systems/Compute
    """

    def _step_payload(self, action: KernelAction) -> dict:
        """Shape expected by the server's /step endpoint."""
        return {
            "code": action.code,
        }

    def _parse_result(self, payload: dict) -> StepResult[KernelObservation]:
        """Parse server response into StepResult."""
        obs_data = payload["observation"]
        obs = KernelObservation(
            problem_id=obs_data.get("problem_id", ""),
            problem_description=obs_data.get("problem_description", ""),
            reference_code=obs_data.get("reference_code", ""),
            gpu_info=obs_data.get("gpu_info", ""),
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 10),
            feedback=obs_data.get("feedback", ""),
            compilation_success=obs_data.get("compilation_success", False),
            compilation_error=obs_data.get("compilation_error"),
            correctness_pass=obs_data.get("correctness_pass"),
            max_diff=obs_data.get("max_diff"),
            speedup=obs_data.get("speedup"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> KernelState:
        """Parse server response into KernelState."""
        return KernelState(
            problem_id=payload.get("problem_id"),
            turn=payload.get("turn", 0),
            max_turns=payload.get("max_turns", 10),
            best_speedup=payload.get("best_speedup", 0.0),
            solved=payload.get("solved", False),
        )
