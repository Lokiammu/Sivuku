# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transform implementations for CodeAct environments.

Transforms enable RL training by adding rewards, safety checks, and quality
metrics to observations based on code execution results.
"""

import re
import ast
from typing import Union

from .interfaces import Transform
from .types import CodeObservation, Observation


class CodeSafetyTransform(Transform):
    """Evaluates code safety and assigns penalties for dangerous patterns."""

    def __init__(self, penalty: float = -1.0):
        self.penalty = penalty
        self.dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'eval\(',
            r'exec\(',
            r'__import__',
            r'open\(',
        ]

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        if 'last_code' in observation.metadata:
            code = observation.metadata['last_code']
            for pattern in self.dangerous_patterns:
                if re.search(pattern, code):
                    observation.reward = self.penalty
                    observation.metadata['safety_violation'] = pattern
                    break
            else:
                if observation.reward is None:
                    observation.reward = 0.0

        return observation


class TaskCompletionTransform(Transform):
    """Rewards successful task completion based on execution results."""

    def __init__(self,
                 success_reward: float = 1.0,
                 failure_reward: float = 0.0,
                 error_penalty: float = -0.5):
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.error_penalty = error_penalty

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        if not observation.execution_result.success:
            observation.reward = self.error_penalty
        else:
            observation.reward = self._evaluate_completion(observation)

        return observation

    def _evaluate_completion(self, observation: CodeObservation) -> float:
        """Override this method for task-specific completion criteria."""
        return (
            self.success_reward if observation.execution_result.success
            else self.failure_reward
        )


class MathProblemTransform(TaskCompletionTransform):
    """Rewards correct numerical answers for mathematical problems."""

    def __init__(
        self,
        expected_answer: Union[int, float],
        tolerance: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.expected_answer = expected_answer
        self.tolerance = tolerance

    def _evaluate_completion(self, observation: CodeObservation) -> float:
        result = observation.execution_result.return_value
        if result is None:
            return self.failure_reward

        try:
            numeric_result = float(result)
            if abs(numeric_result - self.expected_answer) <= self.tolerance:
                return self.success_reward
            else:
                return self.failure_reward
        except (ValueError, TypeError):
            return self.failure_reward


class CodeQualityTransform(Transform):
    """Evaluates and rewards code quality metrics."""

    def __init__(self,
                 concise_bonus: float = 0.1,
                 max_length_threshold: int = 100,
                 syntax_penalty: float = -0.2):
        self.concise_bonus = concise_bonus
        self.max_length_threshold = max_length_threshold
        self.syntax_penalty = syntax_penalty

    def __call__(self, observation: Observation) -> Observation:
        if not isinstance(observation, CodeObservation):
            return observation

        quality_score = 0.0

        if 'last_code' in observation.metadata:
            code = observation.metadata['last_code']

            # Reward concise code
            if len(code.strip()) <= self.max_length_threshold:
                quality_score += self.concise_bonus

            # Check syntax (redundant but useful for quality assessment)
            try:
                ast.parse(code)
            except SyntaxError:
                quality_score += self.syntax_penalty

        # Add to existing reward
        if observation.reward is None:
            observation.reward = quality_score
        else:
            observation.reward += quality_score

        return observation


class CompositeTransform(Transform):
    """Combines multiple transforms into a single transform."""

    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, observation: Observation) -> Observation:
        for transform in self.transforms:
            observation = transform(observation)
        return observation


def create_math_env_transform(
    expected_answer: Union[int, float], tolerance: float = 1e-6
) -> CompositeTransform:
    """Create a composite transform for mathematical problem solving."""
    return CompositeTransform([
        MathProblemTransform(
            expected_answer=expected_answer, tolerance=tolerance
        ),
        CodeSafetyTransform(),
        CodeQualityTransform()
    ])


def create_safe_env_transform() -> CompositeTransform:
    """Create a transform focused on code safety and quality."""
    return CompositeTransform([
        CodeSafetyTransform(),
        CodeQualityTransform(),
        TaskCompletionTransform()
    ])
