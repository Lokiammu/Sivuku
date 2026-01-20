# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU Kernel Optimization Environment.

Server-side environment for evaluating CUDA/Triton kernels against
PyTorch reference implementations.
"""

import os
import uuid
import random
from pathlib import Path
from typing import Optional

from openenv.core.env_server.interfaces import Action, Environment, Observation

from ..models import KernelAction, KernelObservation, KernelState
from .evaluator import LocalGPUEvaluator


class Problem:
    """A kernel optimization problem."""
    def __init__(self, id: str, level: int, name: str, description: str, reference_code: str):
        self.id = id
        self.level = level
        self.name = name
        self.description = description
        self.reference_code = reference_code


class KernelOptEnv(Environment):
    """
    GPU Kernel Optimization Environment.

    Agents submit CUDA/Triton kernel code and receive feedback including:
    - Compilation status and errors
    - Correctness against reference implementation
    - Speedup compared to PyTorch baseline
    - Profiling data from NSight Systems/Compute

    Requires local GPU with CUDA toolkit for full profiling support.
    """

    def __init__(
        self,
        problems_dir: Optional[str] = None,
        max_turns: int = 10,
        gpu: str = "cuda:0",
        levels: Optional[list[int]] = None,
        atol: float = 0.05,
        rtol: float = 0.02,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        enable_nsys: bool = True,
        enable_ncu: bool = False,
        timeout: int = 60,
    ):
        self.problems_dir = Path(problems_dir) if problems_dir else self._default_problems_dir()
        self.max_turns = max_turns
        self.gpu = gpu
        self.levels = levels or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Create evaluator
        self.evaluator = LocalGPUEvaluator(
            device=gpu,
            atol=atol,
            rtol=rtol,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            enable_nsys=enable_nsys,
            enable_ncu=enable_ncu,
            timeout=timeout,
        )

        # Load problems
        self.problems = self._load_problems()

        # Episode state
        self._state = KernelState()
        self._current_problem: Optional[Problem] = None
        self._feedbacks: list[str] = []

    def _default_problems_dir(self) -> Path:
        """Default to problems directory relative to package."""
        env_dir = os.environ.get("KERNRL_PROBLEMS_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                return p

        # Check relative to this file
        pkg_problems = Path(__file__).parent.parent / "problems"
        if pkg_problems.exists():
            return pkg_problems

        raise FileNotFoundError(
            "No problems directory found. Set KERNRL_PROBLEMS_DIR or "
            "ensure 'problems/' exists in the package directory."
        )

    def _load_problems(self) -> list[Problem]:
        """Load all problems from the problems directory."""
        problems = []

        for level in self.levels:
            level_dir = self.problems_dir / f"level{level}"
            if not level_dir.exists():
                continue

            for problem_file in sorted(level_dir.glob("*.py")):
                if problem_file.name.startswith("_"):
                    continue

                code = problem_file.read_text()
                name = problem_file.stem

                problems.append(Problem(
                    id=f"L{level}_{name}",
                    level=level,
                    name=name,
                    description=self._make_description(code, level),
                    reference_code=code,
                ))

        return problems

    def _make_description(self, code: str, level: int) -> str:
        """Create the problem description shown to the agent."""
        return f"""# GPU Kernel Optimization Task

## Objective
Write an optimized GPU kernel (using Triton or CUDA) that computes the same result
as the reference PyTorch implementation below, but faster.

## Reference Implementation
```python
{code}
```

## Requirements
1. Your kernel must produce the same output as the reference (atol={self.evaluator.atol}, rtol={self.evaluator.rtol})
2. Your kernel should be faster than the PyTorch baseline
3. You may use Triton (preferred) or raw CUDA

## Output Format
Provide a complete Python file with:
- A `Model` class with the same interface as the reference
- The `Model.forward()` method should use your optimized kernel
- Include any necessary imports (torch, triton, etc.)

## GPU Target
Device: {self.gpu}
"""

    def _get_gpu_info(self) -> str:
        """Get GPU info string."""
        try:
            import torch
            if torch.cuda.is_available():
                idx = int(self.gpu.split(":")[-1]) if ":" in self.gpu else 0
                name = torch.cuda.get_device_name(idx)
                mem = torch.cuda.get_device_properties(idx).total_memory / 1e9
                return f"{name} ({mem:.1f} GB)"
        except:
            pass
        return f"GPU: {self.gpu}"

    def reset(self, problem_id: Optional[str] = None) -> Observation:
        """
        Reset environment and start a new episode.

        Args:
            problem_id: Specific problem to use, or None for random selection

        Returns:
            Initial observation with problem description
        """
        if problem_id:
            self._current_problem = next(
                (p for p in self.problems if p.id == problem_id),
                None
            )
            if not self._current_problem:
                # Try partial match
                self._current_problem = next(
                    (p for p in self.problems if problem_id in p.id),
                    None
                )
            if not self._current_problem:
                raise ValueError(f"Problem {problem_id} not found")
        else:
            self._current_problem = random.choice(self.problems)

        self._state = KernelState(
            episode_id=str(uuid.uuid4()),
            problem_id=self._current_problem.id,
            turn=0,
            max_turns=self.max_turns,
            best_speedup=0.0,
            solved=False,
        )
        self._feedbacks = []

        return KernelObservation(
            problem_id=self._current_problem.id,
            problem_description=self._current_problem.description,
            reference_code=self._current_problem.reference_code,
            gpu_info=self._get_gpu_info(),
            turn=0,
            max_turns=self.max_turns,
            feedback="",
            compilation_success=True,
        )

    def step(self, action: Action) -> Observation:
        """
        Execute kernel code and return evaluation results.

        Args:
            action: KernelAction containing the kernel code

        Returns:
            KernelObservation with evaluation results
        """
        if not isinstance(action, KernelAction):
            raise ValueError(f"Expected KernelAction, got {type(action)}")

        if self._current_problem is None:
            raise RuntimeError("Must call reset() before step()")

        self._state.turn += 1

        # Evaluate the kernel
        eval_result = self.evaluator.evaluate(
            solution_code=action.code,
            reference_code=self._current_problem.reference_code,
            problem_id=self._current_problem.id,
            step=self._state.turn,
        )

        # Generate feedback
        feedback = eval_result.to_agent_feedback()
        self._feedbacks.append(feedback)

        # Update state
        if eval_result.benchmark and eval_result.benchmark.speedup > self._state.best_speedup:
            self._state.best_speedup = eval_result.benchmark.speedup

        if (eval_result.correctness and eval_result.correctness.correct and
            eval_result.benchmark and eval_result.benchmark.speedup > 1.05):
            self._state.solved = True

        return KernelObservation(
            problem_id=self._current_problem.id,
            problem_description=self._current_problem.description,
            reference_code=self._current_problem.reference_code,
            gpu_info=self._get_gpu_info(),
            turn=self._state.turn,
            max_turns=self.max_turns,
            feedback=feedback,
            compilation_success=eval_result.compilation.success,
            compilation_error=eval_result.compilation.error,
            correctness_pass=eval_result.correctness.correct if eval_result.correctness else None,
            max_diff=eval_result.correctness.max_diff if eval_result.correctness else None,
            speedup=eval_result.benchmark.speedup if eval_result.benchmark else None,
        )

    @property
    def state(self) -> KernelState:
        """Get current environment state."""
        return self._state

    @property
    def done(self) -> bool:
        """Check if episode is done."""
        return self._state.turn >= self.max_turns or self._state.solved

    @property
    def reward(self) -> float:
        """Get reward for current state."""
        # Reward is computed by evaluator and included in eval_result
        return 0.0  # Placeholder - actual reward comes from eval_result

    def list_problems(self) -> list[str]:
        """List all available problem IDs."""
        return [p.id for p in self.problems]

    @property
    def num_problems(self) -> int:
        return len(self.problems)
