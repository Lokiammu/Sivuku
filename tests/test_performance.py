# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Performance tests for EnvTorch components.

These tests verify performance characteristics and identify bottlenecks.
"""

import pytest
import time
from statistics import mean, stdev

from src import (
    create_codeact_env,
    create_mcp_environment,
    CodeAction,
    PythonExecutor,
    MathProblemTransform,
)


class TestPerformanceBasics:
    """Test basic performance characteristics."""

    @pytest.mark.performance
    def test_environment_reset_performance(self):
        """Test environment reset performance."""
        env = create_codeact_env()

        # Warm up
        for _ in range(5):
            env.reset()

        # Measure reset times
        reset_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            env.reset()
            reset_times.append(time.perf_counter() - start_time)

        avg_time = mean(reset_times)
        max_time = max(reset_times)

        # Reset should be fast (< 1ms on average)
        assert avg_time < 0.001, f"Average reset time {avg_time:.4f}s too slow"
        assert max_time < 0.01, f"Max reset time {max_time:.4f}s too slow"

    @pytest.mark.performance
    def test_simple_code_execution_performance(self):
        """Test performance of simple code execution."""
        env = create_codeact_env()
        env.reset()

        # Test simple arithmetic
        simple_codes = [
            "2 + 2",
            "10 * 5",
            "100 / 4",
            "2 ** 8",
            "sum([1, 2, 3, 4, 5])"
        ]

        execution_times = []
        for code in simple_codes * 20:  # 100 total executions
            action = CodeAction(code=code)
            start_time = time.perf_counter()
            obs = env.step(action)
            execution_times.append(time.perf_counter() - start_time)
            assert obs.execution_result.success is True

        avg_time = mean(execution_times)
        assert avg_time < 0.001, f"Average execution time {avg_time:.4f}s too slow"

    @pytest.mark.performance
    def test_variable_persistence_performance(self):
        """Test performance when maintaining large state."""
        env = create_codeact_env()
        env.reset()

        # Create increasingly large state
        step_times = []
        for i in range(1, 21):  # 20 steps
            code = f"var_{i} = list(range({i * 100}))"
            action = CodeAction(code=code)

            start_time = time.perf_counter()
            obs = env.step(action)
            step_times.append(time.perf_counter() - start_time)

            assert obs.execution_result.success is True

        # Performance should not degrade significantly
        early_avg = mean(step_times[:5])
        late_avg = mean(step_times[-5:])

        # Later steps shouldn't be more than 3x slower than early steps
        assert late_avg < early_avg * 3, "Performance degraded significantly with state growth"

    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_data_processing(self):
        """Test performance with large data structures."""
        env = create_codeact_env()
        env.reset()

        code = '''
# Create large dataset
import random
random.seed(42)

large_data = []
for i in range(10000):
    large_data.append({
        "id": i,
        "value": random.random(),
        "category": random.choice(["A", "B", "C"])
    })

# Process data
category_sums = {"A": 0, "B": 0, "C": 0}
for item in large_data:
    category_sums[item["category"]] += item["value"]

len(large_data)
'''

        start_time = time.perf_counter()
        obs = env.step(CodeAction(code=code))
        execution_time = time.perf_counter() - start_time

        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 10000

        # Should complete within reasonable time (< 1 second)
        assert execution_time < 1.0, f"Large data processing took {execution_time:.2f}s"


class TestMemoryUsage:
    """Test memory usage characteristics."""

    @pytest.mark.performance
    def test_executor_memory_isolation(self):
        """Test that executors don't leak memory between resets."""
        executor = PythonExecutor()

        # Create large objects and reset multiple times
        for _ in range(10):
            # Create large object
            result = executor.execute("large_list = list(range(100000))")
            assert result.success is True

            # Reset should clean up memory
            executor.reset()

            # Check that object is gone
            result = executor.execute("large_list")
            assert result.success is False
            assert result.exception_type == "NameError"

    @pytest.mark.performance
    def test_environment_state_cleanup(self):
        """Test that environment properly cleans up between episodes."""
        env = create_codeact_env()

        for episode in range(5):
            obs = env.reset()

            # Create some state
            env.step(CodeAction(code=f"episode_{episode}_data = list(range(1000))"))

            # Reset and verify cleanup
            obs = env.reset()

            # Previous episode data should be gone
            obs = env.step(CodeAction(code=f"episode_{episode}_data"))
            assert obs.execution_result.success is False


class TestScalabilityLimits:
    """Test scalability limits and edge cases."""

    @pytest.mark.performance
    @pytest.mark.edge_case
    def test_deep_recursion_handling(self):
        """Test handling of deep recursion."""
        env = create_codeact_env()
        env.reset()

        # Test with moderate recursion depth
        code = '''
def recursive_sum(n):
    if n <= 0:
        return 0
    return n + recursive_sum(n - 1)

# Test with depth that shouldn't hit Python's recursion limit
result = recursive_sum(500)
result
'''
        obs = env.step(CodeAction(code=code))

        assert obs.execution_result.success is True
        # Sum of 1 to 500 = 500 * 501 / 2 = 125250
        assert obs.execution_result.return_value == 125250

    @pytest.mark.performance
    @pytest.mark.edge_case
    @pytest.mark.slow
    def test_long_running_computation(self):
        """Test handling of longer computations."""
        env = create_codeact_env()
        env.reset()

        code = '''
# Compute prime numbers up to 1000
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 1000) if is_prime(n)]
len(primes)
'''

        start_time = time.perf_counter()
        obs = env.step(CodeAction(code=code))
        execution_time = time.perf_counter() - start_time

        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 168  # Number of primes < 1000

        # Should complete in reasonable time
        assert execution_time < 5.0, f"Prime computation took {execution_time:.2f}s"

    @pytest.mark.performance
    def test_many_small_steps_performance(self):
        """Test performance with many small execution steps."""
        env = create_codeact_env()
        env.reset()

        step_times = []
        total_start = time.perf_counter()

        # Execute many small steps
        for i in range(100):
            action = CodeAction(code=f"x_{i} = {i} * 2")

            step_start = time.perf_counter()
            obs = env.step(action)
            step_times.append(time.perf_counter() - step_start)

            assert obs.execution_result.success is True

        total_time = time.perf_counter() - total_start

        # Check that performance remains consistent
        early_avg = mean(step_times[:20])
        late_avg = mean(step_times[-20:])

        assert late_avg < early_avg * 2, "Step performance degraded significantly"
        assert total_time < 1.0, f"100 small steps took {total_time:.2f}s"


class TestTransformPerformance:
    """Test performance of transform operations."""

    @pytest.mark.performance
    def test_transform_application_performance(self):
        """Test performance of transform applications."""
        transform = MathProblemTransform(expected_answer=42)

        # Create many observations to transform
        observations = []
        for i in range(1000):
            from src.types import CodeObservation, ExecutionResult
            obs = CodeObservation(
                execution_result=ExecutionResult(
                    success=True,
                    return_value=42 if i % 2 == 0 else 43
                )
            )
            observations.append(obs)

        # Time transform application
        start_time = time.perf_counter()
        for obs in observations:
            result = transform(obs)
            assert result.reward in [0.0, 1.0]

        transform_time = time.perf_counter() - start_time

        # Transforms should be very fast
        assert transform_time < 0.1, f"Transform application took {transform_time:.3f}s"

    @pytest.mark.performance
    def test_composite_transform_performance(self):
        """Test performance of composite transforms."""
        from src.transforms import CompositeTransform, CodeSafetyTransform, CodeQualityTransform

        composite = CompositeTransform([
            MathProblemTransform(expected_answer=42),
            CodeSafetyTransform(),
            CodeQualityTransform()
        ])

        # Test with realistic observations
        observations = []
        codes = ["x = 42", "import os", "print('hello')", "2 + 2"]
        for i in range(250):  # 1000 total (250 * 4)
            from src.types import CodeObservation, ExecutionResult
            obs = CodeObservation(
                execution_result=ExecutionResult(success=True, return_value=42),
                metadata={'last_code': codes[i % len(codes)]}
            )
            observations.append(obs)

        start_time = time.perf_counter()
        for obs in observations:
            result = composite(obs)

        composite_time = time.perf_counter() - start_time

        # Composite transforms should still be fast
        assert composite_time < 0.5, f"Composite transform took {composite_time:.3f}s"


class TestMCPPerformance:
    """Test performance of MCP tool operations."""

    @pytest.mark.performance
    def test_mcp_tool_invocation_performance(self):
        """Test performance of MCP tool invocations."""
        env = create_mcp_environment()
        env.reset()

        # Test calculator tool performance
        calculator_times = []
        for i in range(100):
            code = f'calculator("{i} + {i * 2}")'
            action = CodeAction(code=code)

            start_time = time.perf_counter()
            obs = env.step(action)
            calculator_times.append(time.perf_counter() - start_time)

            assert obs.execution_result.success is True

        avg_calc_time = mean(calculator_times)
        assert avg_calc_time < 0.01, f"Average calculator time {avg_calc_time:.4f}s too slow"

    @pytest.mark.performance
    def test_file_operations_performance(self, temp_dir):
        """Test performance of file operations."""
        env = create_mcp_environment()
        env.reset()

        import os
        test_file = os.path.join(temp_dir, "perf_test.txt")

        # Test repeated file operations
        file_times = []
        for i in range(50):
            code = f'''
content = "Test content {i} " * 100
file_write("{test_file}", content)
read_content = file_read("{test_file}")
len(read_content)
'''
            action = CodeAction(code=code)

            start_time = time.perf_counter()
            obs = env.step(action)
            file_times.append(time.perf_counter() - start_time)

            assert obs.execution_result.success is True

        avg_file_time = mean(file_times)
        # File operations are inherently slower, but should be reasonable
        assert avg_file_time < 0.1, f"Average file operation time {avg_file_time:.4f}s too slow"


@pytest.mark.performance
class TestConcurrencySimulation:
    """Simulate concurrent usage patterns."""

    def test_multiple_environment_isolation(self):
        """Test that multiple environments don't interfere with each other."""
        envs = [create_codeact_env() for _ in range(5)]

        # Reset all environments
        for env in envs:
            env.reset()

        # Execute different code in each environment simultaneously
        results = []
        for i, env in enumerate(envs):
            code = f"unique_var_{i} = {i * 100}"
            obs = env.step(CodeAction(code=code))
            results.append(obs.execution_result.success)

        # All should succeed
        assert all(results)

        # Verify isolation - each env should only have its own variable
        for i, env in enumerate(envs):
            # Should have own variable
            obs = env.step(CodeAction(code=f"unique_var_{i}"))
            assert obs.execution_result.success is True
            assert obs.execution_result.return_value == i * 100

            # Should not have other variables
            other_i = (i + 1) % len(envs)
            obs = env.step(CodeAction(code=f"unique_var_{other_i}"))
            assert obs.execution_result.success is False