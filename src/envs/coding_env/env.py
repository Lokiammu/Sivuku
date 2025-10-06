"""
envs/coding_env/env.py
--------------------------------
Concrete environment implementation using the core BaseEnv.
POC implementation runs code locally via subprocess that can be changed later.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from core.base import BaseEnv
from core.types import StepResult

from .models import CodeAction, CodeObservation


class CodingEnv(BaseEnv[CodeAction, CodeObservation]):
    """
    Minimal Coding Environment.

    POC behavior:
      - reset(): returns a fresh, empty observation (no persistent state).
      - step(action): runs Python code with `python -c` and returns stdout/stderr/exit_code.

    Future swap:
      Replace _run_code_locally() with a call to your Docker/gateway backend without
      changing the public API.
    """

    def __init__(
        self,
        default_timeout_s: float = 10.0,
        python_executable: str = "python",
    ):
        """
        Args:
            default_timeout_s: Max seconds to allow code execution before timing out.
            python_executable: Interpreter to run (e.g., "python3", a venv path, etc.).
        """
        self._default_timeout_s = float(default_timeout_s)
        self._python = python_executable

    # --- BaseEnv interface ---

    def reset(self) -> CodeObservation:
        # No state to clear in this POC; return an initial observation.
        return CodeObservation(stdout="", stderr="", exit_code=0)

    def step(self, action: CodeAction) -> StepResult[CodeObservation]:
        if not isinstance(action, CodeAction):
            raise TypeError(f"Expected CodeAction, got {type(action)!r}")

        # TODO: replace dummy response with the call to the code executor inside the container
        obs, timed_out = CodeObservation(stderr="", stdout="", exit_code=0), False

        # Simple reward heuristic: success and no stderr -> 1.0 else 0.0
        reward: Optional[float] = (
            1.0 if (obs.exit_code == 0 and not obs.stderr) else 0.0
        )

        info = {
            "timed_out": timed_out,
            "interpreter": self._python,
        }

        return StepResult(
            observation=obs,
            reward=reward,
            done=False,  # Coding env is not episodic by default
        )
