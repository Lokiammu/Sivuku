# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions and helpers for EnvTorch tests.
"""

import time
import tempfile
import os
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from src import CodeAction, CodeActEnvironment, create_codeact_env


class TestTimer:
    """Context manager for timing test operations."""

    def __init__(self, max_time: float = None):
        self.max_time = max_time
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        if self.max_time is not None and self.elapsed > self.max_time:
            raise AssertionError(
                f"Operation took {self.elapsed:.3f}s, "
                f"expected < {self.max_time:.3f}s"
            )


@contextmanager
def temporary_files(file_contents: Dict[str, str]):
    """Context manager that creates temporary files with given contents.

    Args:
        file_contents: Dict mapping filenames to their contents

    Yields:
        Dict mapping filenames to their full paths
    """
    temp_dir = tempfile.mkdtemp()
    file_paths = {}

    try:
        for filename, content in file_contents.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            file_paths[filename] = file_path

        yield file_paths

    finally:
        # Clean up
        for file_path in file_paths.values():
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


class CodeSequence:
    """Helper class for building and executing sequences of code actions."""

    def __init__(self, env: Optional[CodeActEnvironment] = None):
        self.env = env or create_codeact_env()
        self.actions: List[CodeAction] = []
        self.results: List[Any] = []

    def add(self, code: str, metadata: Optional[Dict[str, Any]] = None) -> 'CodeSequence':
        """Add a code action to the sequence."""
        action = CodeAction(code=code, metadata=metadata or {})
        self.actions.append(action)
        return self

    def execute(self, reset_first: bool = True) -> List[Any]:
        """Execute all actions in sequence and return results."""
        if reset_first:
            self.env.reset()

        self.results = []
        for action in self.actions:
            obs = self.env.step(action)
            self.results.append(obs)

        return self.results

    def assert_all_success(self) -> 'CodeSequence':
        """Assert that all executed actions succeeded."""
        for i, result in enumerate(self.results):
            assert result.execution_result.success, (
                f"Action {i} failed: {result.execution_result.exception_message}"
            )
        return self

    def assert_return_values(self, expected_values: List[Any]) -> 'CodeSequence':
        """Assert that return values match expected values."""
        assert len(self.results) == len(expected_values), (
            f"Expected {len(expected_values)} results, got {len(self.results)}"
        )

        for i, (result, expected) in enumerate(zip(self.results, expected_values)):
            actual = result.execution_result.return_value
            assert actual == expected, (
                f"Action {i}: expected {expected}, got {actual}"
            )
        return self

    def get_return_values(self) -> List[Any]:
        """Get return values from all executed actions."""
        return [result.execution_result.return_value for result in self.results]


def create_math_test_cases() -> List[Dict[str, Any]]:
    """Create a set of mathematical test cases for validation."""
    return [
        {
            "code": "2 + 2",
            "expected": 4,
            "description": "Simple addition"
        },
        {
            "code": "10 * 5",
            "expected": 50,
            "description": "Multiplication"
        },
        {
            "code": "import math\nmath.sqrt(16)",
            "expected": 4.0,
            "description": "Square root using math module"
        },
        {
            "code": "sum(range(1, 11))",
            "expected": 55,
            "description": "Sum of numbers 1-10"
        },
        {
            "code": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

factorial(5)
""",
            "expected": 120,
            "description": "Recursive factorial"
        },
        {
            "code": "[x**2 for x in range(5)]",
            "expected": [0, 1, 4, 9, 16],
            "description": "List comprehension with squares"
        }
    ]


def create_error_test_cases() -> List[Dict[str, Any]]:
    """Create a set of error cases for validation."""
    return [
        {
            "code": "1 / 0",
            "expected_error": "ZeroDivisionError",
            "description": "Division by zero"
        },
        {
            "code": "undefined_variable",
            "expected_error": "NameError",
            "description": "Undefined variable"
        },
        {
            "code": "'string' + 42",
            "expected_error": "TypeError",
            "description": "Type mismatch"
        },
        {
            "code": "[1, 2, 3][10]",
            "expected_error": "IndexError",
            "description": "List index out of range"
        },
        {
            "code": "invalid syntax here",
            "expected_error": "SyntaxError",
            "description": "Syntax error"
        }
    ]


class PerformanceProfiler:
    """Simple profiler for tracking test performance."""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}

    def measure(self, name: str):
        """Context manager for measuring operation time."""
        return self._MeasureContext(self, name)

    class _MeasureContext:
        def __init__(self, profiler: 'PerformanceProfiler', name: str):
            self.profiler = profiler
            self.name = name

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.perf_counter() - self.start_time
            if self.name not in self.profiler.measurements:
                self.profiler.measurements[self.name] = []
            self.profiler.measurements[self.name].append(elapsed)

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a measurement."""
        if name not in self.measurements:
            return {}

        times = self.measurements[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
        }

    def report(self) -> str:
        """Generate a performance report."""
        lines = ["Performance Report:", "=" * 50]

        for name in sorted(self.measurements.keys()):
            stats = self.get_stats(name)
            lines.append(f"{name}:")
            lines.append(f"  Count: {stats['count']}")
            lines.append(f"  Total: {stats['total']:.4f}s")
            lines.append(f"  Mean:  {stats['mean']:.4f}s")
            lines.append(f"  Min:   {stats['min']:.4f}s")
            lines.append(f"  Max:   {stats['max']:.4f}s")
            lines.append("")

        return "\n".join(lines)


def assert_execution_success(obs, message: str = "Execution should succeed"):
    """Helper to assert that code execution was successful."""
    assert obs.execution_result.success, (
        f"{message}. Error: {obs.execution_result.exception_type} - "
        f"{obs.execution_result.exception_message}"
    )


def assert_execution_failure(obs, expected_error: str = None,
                           message: str = "Execution should fail"):
    """Helper to assert that code execution failed as expected."""
    assert not obs.execution_result.success, f"{message}"

    if expected_error:
        assert obs.execution_result.exception_type == expected_error, (
            f"Expected {expected_error}, got {obs.execution_result.exception_type}"
        )


def run_test_suite(test_cases: List[Dict[str, Any]], env: CodeActEnvironment = None) -> Dict[str, Any]:
    """Run a suite of test cases and return summary results."""
    if env is None:
        env = create_codeact_env()

    results = {
        'passed': 0,
        'failed': 0,
        'errors': [],
        'details': []
    }

    for i, test_case in enumerate(test_cases):
        try:
            env.reset()
            obs = env.step(CodeAction(code=test_case['code']))

            # Check if this is an error test case
            if 'expected_error' in test_case:
                if obs.execution_result.success:
                    results['failed'] += 1
                    results['errors'].append(f"Test {i}: Expected error but got success")
                elif obs.execution_result.exception_type == test_case['expected_error']:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(
                        f"Test {i}: Expected {test_case['expected_error']}, "
                        f"got {obs.execution_result.exception_type}"
                    )
            else:
                # Success test case
                if not obs.execution_result.success:
                    results['failed'] += 1
                    results['errors'].append(
                        f"Test {i}: Execution failed: {obs.execution_result.exception_message}"
                    )
                elif obs.execution_result.return_value == test_case['expected']:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(
                        f"Test {i}: Expected {test_case['expected']}, "
                        f"got {obs.execution_result.return_value}"
                    )

            results['details'].append({
                'test_case': i,
                'description': test_case.get('description', 'No description'),
                'success': obs.execution_result.success,
                'return_value': obs.execution_result.return_value,
                'error': obs.execution_result.exception_type
            })

        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Test {i}: Unexpected error: {e}")

    return results


class MockTool:
    """Mock tool for testing tool integration."""

    def __init__(self, return_value: Any = "mock_result", should_raise: Exception = None):
        self.return_value = return_value
        self.should_raise = should_raise
        self.call_count = 0
        self.call_args = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.call_args.append((args, kwargs))

        if self.should_raise:
            raise self.should_raise

        return self.return_value

    def reset(self):
        """Reset call tracking."""
        self.call_count = 0
        self.call_args = []


def create_test_environment_with_mock_tools(tools: Dict[str, MockTool] = None) -> CodeActEnvironment:
    """Create a test environment with mock tools."""
    if tools is None:
        tools = {
            'mock_tool': MockTool(),
            'failing_tool': MockTool(should_raise=ValueError("Mock error")),
            'counter_tool': MockTool(return_value=42)
        }

    return CodeActEnvironment(tools=tools)


# Constants for common test values
TEST_TIMEOUT = 5.0  # Maximum time for most tests
PERFORMANCE_TIMEOUT = 1.0  # Maximum time for performance-sensitive tests
SLOW_TEST_TIMEOUT = 30.0  # Maximum time for slow tests

# Common test data
SAMPLE_JSON_DATA = {
    "string": "test",
    "number": 42,
    "boolean": True,
    "null": None,
    "array": [1, 2, 3],
    "object": {"nested": "value"}
}

SAMPLE_CSV_DATA = """name,age,score
Alice,25,95
Bob,30,87
Charlie,28,92"""