# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for transform classes.
"""

import pytest
from unittest.mock import Mock

from src.transforms import (
    CodeSafetyTransform,
    TaskCompletionTransform,
    MathProblemTransform,
    CodeQualityTransform,
    CompositeTransform,
    create_math_env_transform,
    create_safe_env_transform,
)
from src.types import CodeObservation, ExecutionResult, Observation


class TestCodeSafetyTransform:
    """Test the CodeSafetyTransform class."""

    def test_initialization(self):
        """Test CodeSafetyTransform initialization."""
        transform = CodeSafetyTransform()
        assert transform.penalty == -1.0
        assert len(transform.dangerous_patterns) > 0

    def test_initialization_with_custom_penalty(self):
        """Test initialization with custom penalty."""
        transform = CodeSafetyTransform(penalty=-5.0)
        assert transform.penalty == -5.0

    def test_safe_code_no_penalty(self):
        """Test that safe code gets no penalty."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(metadata={'last_code': 'x = 2 + 2'})

        result = transform(obs)

        assert result.reward == 0.0
        assert 'safety_violation' not in result.metadata

    def test_dangerous_import_os(self):
        """Test detection of dangerous os import."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(metadata={'last_code': 'import os\nos.system("rm -rf /")'})

        result = transform(obs)

        assert result.reward == -1.0
        assert result.metadata['safety_violation'] == r'import\s+os'

    def test_dangerous_subprocess(self):
        """Test detection of dangerous subprocess import."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(metadata={'last_code': 'import subprocess\nsubprocess.run("ls")'})

        result = transform(obs)

        assert result.reward == -1.0
        assert result.metadata['safety_violation'] == r'import\s+subprocess'

    def test_dangerous_eval(self):
        """Test detection of dangerous eval usage."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(metadata={'last_code': 'eval("print(42)")'})

        result = transform(obs)

        assert result.reward == -1.0
        assert result.metadata['safety_violation'] == r'eval\('

    def test_dangerous_exec(self):
        """Test detection of dangerous exec usage."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(metadata={'last_code': 'exec("x = 42")'})

        result = transform(obs)

        assert result.reward == -1.0
        assert result.metadata['safety_violation'] == r'exec\('

    def test_dangerous_open(self):
        """Test detection of file operations."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(metadata={'last_code': 'open("/etc/passwd", "r")'})

        result = transform(obs)

        assert result.reward == -1.0
        assert result.metadata['safety_violation'] == r'open\('

    def test_non_codeobservation_passthrough(self):
        """Test that non-CodeObservation objects pass through unchanged."""
        transform = CodeSafetyTransform()
        obs = Observation(reward=5.0)

        result = transform(obs)

        assert result is obs
        assert result.reward == 5.0

    def test_missing_last_code_metadata(self):
        """Test handling when last_code metadata is missing."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(metadata={})

        result = transform(obs)

        # Should not modify observation
        assert result is obs
        assert result.reward is None

    def test_existing_reward_preserved_for_safe_code(self):
        """Test that existing reward is not overwritten for safe code."""
        transform = CodeSafetyTransform()
        obs = CodeObservation(
            reward=2.0,
            metadata={'last_code': 'x = 42'}
        )

        result = transform(obs)

        # Should not overwrite existing reward for safe code
        assert result.reward == 2.0


class TestTaskCompletionTransform:
    """Test the TaskCompletionTransform class."""

    def test_initialization(self):
        """Test TaskCompletionTransform initialization."""
        transform = TaskCompletionTransform()
        assert transform.success_reward == 1.0
        assert transform.failure_reward == 0.0
        assert transform.error_penalty == -0.5

    def test_initialization_with_custom_values(self):
        """Test initialization with custom values."""
        transform = TaskCompletionTransform(
            success_reward=2.0,
            failure_reward=-1.0,
            error_penalty=-2.0
        )
        assert transform.success_reward == 2.0
        assert transform.failure_reward == -1.0
        assert transform.error_penalty == -2.0

    def test_successful_execution_reward(self):
        """Test reward for successful execution."""
        transform = TaskCompletionTransform()
        obs = CodeObservation(
            execution_result=ExecutionResult(success=True)
        )

        result = transform(obs)

        assert result.reward == 1.0  # Default success reward

    def test_failed_execution_penalty(self):
        """Test penalty for failed execution."""
        transform = TaskCompletionTransform()
        obs = CodeObservation(
            execution_result=ExecutionResult(success=False)
        )

        result = transform(obs)

        assert result.reward == -0.5  # Default error penalty

    def test_non_codeobservation_passthrough(self):
        """Test that non-CodeObservation objects pass through."""
        transform = TaskCompletionTransform()
        obs = Observation()

        result = transform(obs)

        assert result is obs


class TestMathProblemTransform:
    """Test the MathProblemTransform class."""

    def test_initialization(self):
        """Test MathProblemTransform initialization."""
        transform = MathProblemTransform(expected_answer=42)
        assert transform.expected_answer == 42
        assert transform.tolerance == 1e-6

    def test_correct_answer_reward(self):
        """Test reward for correct numerical answer."""
        transform = MathProblemTransform(expected_answer=42)
        obs = CodeObservation(
            execution_result=ExecutionResult(
                success=True,
                return_value=42
            )
        )

        result = transform(obs)

        assert result.reward == 1.0

    def test_correct_answer_with_tolerance(self):
        """Test reward for answer within tolerance."""
        transform = MathProblemTransform(expected_answer=42, tolerance=0.1)
        obs = CodeObservation(
            execution_result=ExecutionResult(
                success=True,
                return_value=42.05  # Within tolerance
            )
        )

        result = transform(obs)

        assert result.reward == 1.0

    def test_incorrect_answer_no_reward(self):
        """Test no reward for incorrect answer."""
        transform = MathProblemTransform(expected_answer=42)
        obs = CodeObservation(
            execution_result=ExecutionResult(
                success=True,
                return_value=43  # Wrong answer
            )
        )

        result = transform(obs)

        assert result.reward == 0.0

    def test_no_return_value(self):
        """Test handling when there's no return value."""
        transform = MathProblemTransform(expected_answer=42)
        obs = CodeObservation(
            execution_result=ExecutionResult(
                success=True,
                return_value=None
            )
        )

        result = transform(obs)

        assert result.reward == 0.0

    def test_non_numeric_return_value(self):
        """Test handling of non-numeric return values."""
        transform = MathProblemTransform(expected_answer=42)
        obs = CodeObservation(
            execution_result=ExecutionResult(
                success=True,
                return_value="not a number"
            )
        )

        result = transform(obs)

        assert result.reward == 0.0

    def test_float_expected_answer(self):
        """Test with float expected answer."""
        transform = MathProblemTransform(expected_answer=3.14159, tolerance=0.001)
        obs = CodeObservation(
            execution_result=ExecutionResult(
                success=True,
                return_value=3.141  # Close but not exact
            )
        )

        result = transform(obs)

        assert result.reward == 1.0

    def test_execution_failure_penalty(self):
        """Test that execution failure gets penalty."""
        transform = MathProblemTransform(expected_answer=42)
        obs = CodeObservation(
            execution_result=ExecutionResult(success=False)
        )

        result = transform(obs)

        assert result.reward == -0.5  # From parent class error penalty


class TestCodeQualityTransform:
    """Test the CodeQualityTransform class."""

    def test_initialization(self):
        """Test CodeQualityTransform initialization."""
        transform = CodeQualityTransform()
        assert transform.concise_bonus == 0.1
        assert transform.max_length_threshold == 100
        assert transform.syntax_penalty == -0.2

    def test_concise_code_bonus(self):
        """Test bonus for concise code."""
        transform = CodeQualityTransform()
        obs = CodeObservation(
            metadata={'last_code': 'x = 42'}  # Short code
        )

        result = transform(obs)

        assert result.reward == 0.1

    def test_long_code_no_bonus(self):
        """Test no bonus for long code."""
        transform = CodeQualityTransform(max_length_threshold=10)
        obs = CodeObservation(
            metadata={'last_code': 'x = 42\ny = 84\nz = x + y'}  # Long code
        )

        result = transform(obs)

        assert result.reward == 0.0  # No concise bonus

    def test_syntax_error_penalty(self):
        """Test penalty for syntax errors in code quality check."""
        transform = CodeQualityTransform()
        obs = CodeObservation(
            metadata={'last_code': 'invalid syntax here'}
        )

        result = transform(obs)

        # Should get syntax penalty but not concise bonus
        assert result.reward == -0.2

    def test_reward_accumulation(self):
        """Test that quality scores accumulate with existing rewards."""
        transform = CodeQualityTransform()
        obs = CodeObservation(
            reward=1.0,  # Existing reward
            metadata={'last_code': 'x = 42'}  # Short, valid code
        )

        result = transform(obs)

        assert result.reward == 1.1  # 1.0 + 0.1 concise bonus

    def test_missing_code_metadata(self):
        """Test handling when code metadata is missing."""
        transform = CodeQualityTransform()
        obs = CodeObservation(metadata={})

        result = transform(obs)

        assert result.reward is None  # No change


class TestCompositeTransform:
    """Test the CompositeTransform class."""

    def test_initialization(self):
        """Test CompositeTransform initialization."""
        transforms = [
            CodeSafetyTransform(),
            CodeQualityTransform()
        ]
        composite = CompositeTransform(transforms)
        assert composite.transforms == transforms

    def test_sequential_application(self):
        """Test that transforms are applied sequentially."""
        # Create mock transforms that modify reward
        mock1 = Mock()
        mock1.return_value = Mock(reward=1.0)

        mock2 = Mock()
        mock2.return_value = Mock(reward=2.0)

        composite = CompositeTransform([mock1, mock2])
        obs = CodeObservation()

        result = composite(obs)

        # First transform should be called with original observation
        mock1.assert_called_once_with(obs)
        # Second transform should be called with result of first
        mock2.assert_called_once_with(mock1.return_value)
        # Final result should be from second transform
        assert result is mock2.return_value

    def test_empty_transforms_list(self):
        """Test composite with empty transforms list."""
        composite = CompositeTransform([])
        obs = CodeObservation()

        result = composite(obs)

        assert result is obs

    def test_real_transforms_combination(self):
        """Test combination of real transforms."""
        composite = CompositeTransform([
            CodeSafetyTransform(penalty=-1.0),
            CodeQualityTransform(concise_bonus=0.5)
        ])

        # Safe, concise code
        obs = CodeObservation(metadata={'last_code': 'x = 42'})
        result = composite(obs)

        assert result.reward == 0.5  # Only quality bonus, no safety penalty

        # Unsafe code
        obs2 = CodeObservation(metadata={'last_code': 'import os'})
        result2 = composite(obs2)

        assert result2.reward == -0.5  # Safety penalty + quality bonus


class TestTransformFactoryFunctions:
    """Test transform factory functions."""

    def test_create_math_env_transform(self):
        """Test create_math_env_transform factory."""
        transform = create_math_env_transform(expected_answer=42, tolerance=0.1)

        assert isinstance(transform, CompositeTransform)
        assert len(transform.transforms) == 3

        # Test that it works
        obs = CodeObservation(
            execution_result=ExecutionResult(success=True, return_value=42),
            metadata={'last_code': 'x = 42'}
        )
        result = transform(obs)

        # Should get math reward + quality bonus
        assert result.reward > 1.0

    def test_create_safe_env_transform(self):
        """Test create_safe_env_transform factory."""
        transform = create_safe_env_transform()

        assert isinstance(transform, CompositeTransform)
        assert len(transform.transforms) == 3

        # Test safety detection
        obs = CodeObservation(metadata={'last_code': 'import os'})
        result = transform(obs)

        # Should detect safety violation
        assert result.reward < 0

    def test_create_math_env_transform_defaults(self):
        """Test create_math_env_transform with default tolerance."""
        transform = create_math_env_transform(expected_answer=3.14159)

        obs = CodeObservation(
            execution_result=ExecutionResult(success=True, return_value=3.14159),
            metadata={'last_code': 'math.pi'}
        )
        result = transform(obs)

        assert result.reward >= 1.0  # Should get reward for correct answer