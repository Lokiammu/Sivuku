"""
envs/coding_env/models.py
--------------------------------
Action/Observation types for the Coding environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CodeAction:
    """
    Represents a single code execution request.
    """

    code: str
    # Optional: future fields like 'lint': bool, 'timeout_s': float, etc.


@dataclass
class CodeObservation:
    """
    Result of executing code in the environment.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
