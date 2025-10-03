# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for PythonExecutor class.
"""

import pytest
import math
from unittest.mock import patch

from src.environment import PythonExecutor
from src.types import ExecutionResult


class TestPythonExecutor:
    """Test the PythonExecutor class."""

    def test_executor_initialization(self):
        """Test PythonExecutor initialization."""
        executor = PythonExecutor()
        assert '__builtins__' in executor.globals
        assert len(executor.globals) == 1  # Only builtins by default

    def test_executor_with_initial_globals(self):
        """Test PythonExecutor with initial globals."""
        initial = {'test_var': 42, '__builtins__': __builtins__}
        executor = PythonExecutor(initial_globals=initial)
        assert executor.globals is initial
        assert executor.globals['test_var'] == 42

    def test_simple_expression_execution(self):
        """Test executing a simple expression."""
        executor = PythonExecutor()
        result = executor.execute("2 + 3")

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.return_value == 5
        assert result.exception is None

    def test_variable_assignment(self):
        """Test variable assignment and persistence."""
        executor = PythonExecutor()

        # Assign variable
        result1 = executor.execute("x = 42")
        assert result1.success is True
        assert result1.return_value is None

        # Use variable
        result2 = executor.execute("x * 2")
        assert result2.success is True
        assert result2.return_value == 84

    def test_function_definition_and_call(self):
        """Test defining and calling functions."""
        executor = PythonExecutor()

        # Define function
        result1 = executor.execute("""
def square(x):
    return x * x
""")
        assert result1.success is True

        # Call function
        result2 = executor.execute("square(5)")
        assert result2.success is True
        assert result2.return_value == 25

    def test_print_output_capture(self):
        """Test capturing print output."""
        executor = PythonExecutor()
        result = executor.execute("print('Hello, World!')")

        assert result.success is True
        assert "Hello, World!" in result.stdout

    def test_stderr_capture(self):
        """Test capturing stderr output."""
        executor = PythonExecutor()
        result = executor.execute("""
import sys
print('Error message', file=sys.stderr)
""")

        assert result.success is True
        assert "Error message" in result.stderr

    def test_exception_handling(self):
        """Test exception handling and capture."""
        executor = PythonExecutor()
        result = executor.execute("1 / 0")

        assert result.success is False
        assert result.exception_type == "ZeroDivisionError"
        assert "division by zero" in result.exception_message
        assert result.traceback_str != ""
        assert isinstance(result.exception, ZeroDivisionError)

    def test_syntax_error_handling(self):
        """Test syntax error handling."""
        executor = PythonExecutor()
        result = executor.execute("invalid syntax here")

        assert result.success is False
        assert result.exception_type == "SyntaxError"
        assert isinstance(result.exception, SyntaxError)

    def test_multi_line_execution(self):
        """Test executing multi-line code."""
        executor = PythonExecutor()
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"5! = {result}")
result
"""
        result = executor.execute(code)

        assert result.success is True
        assert result.return_value == 120
        assert "5! = 120" in result.stdout

    def test_expression_vs_statement_detection(self):
        """Test proper detection of expressions vs statements."""
        executor = PythonExecutor()

        # Pure expression
        result1 = executor.execute("2 + 3")
        assert result1.return_value == 5

        # Statement with expression on last line
        result2 = executor.execute("x = 10\nx * 2")
        assert result2.return_value == 20

        # Pure statement
        result3 = executor.execute("y = 42")
        assert result3.return_value is None

    def test_import_statements(self):
        """Test import statement execution."""
        executor = PythonExecutor()
        result = executor.execute("import math\nmath.sqrt(16)")

        assert result.success is True
        assert result.return_value == 4.0

    def test_execution_time_measurement(self):
        """Test that execution time is measured."""
        executor = PythonExecutor()
        result = executor.execute("sum(range(100))")

        assert result.success is True
        assert result.execution_time_ms >= 0
        assert isinstance(result.execution_time_ms, float)

    def test_add_tool_method(self):
        """Test adding tools to executor namespace."""
        executor = PythonExecutor()

        def custom_tool(x):
            return x * 10

        executor.add_tool("mytool", custom_tool)
        result = executor.execute("mytool(5)")

        assert result.success is True
        assert result.return_value == 50

    def test_get_available_names(self):
        """Test getting available names from globals."""
        executor = PythonExecutor()

        # Initially should only have non-private names
        names = executor.get_available_names()
        assert all(not name.startswith('_') for name in names)

        # Add some variables and tools
        executor.execute("x = 42")
        executor.add_tool("tool1", lambda: None)

        names = executor.get_available_names()
        assert "x" in names
        assert "tool1" in names
        assert "__builtins__" not in names  # Private names excluded

    def test_reset_method(self):
        """Test executor reset functionality."""
        executor = PythonExecutor()

        # Add some state
        executor.execute("x = 42")
        executor.add_tool("tool", lambda: None)

        # Verify state exists
        result = executor.execute("x")
        assert result.return_value == 42

        # Reset and verify clean state
        executor.reset()
        result = executor.execute("x")
        assert result.success is False
        assert result.exception_type == "NameError"

    def test_complex_data_structures(self):
        """Test handling complex data structures."""
        executor = PythonExecutor()
        code = """
data = {
    'list': [1, 2, 3],
    'dict': {'nested': True},
    'tuple': (4, 5, 6)
}
data
"""
        result = executor.execute(code)

        assert result.success is True
        expected = {
            'list': [1, 2, 3],
            'dict': {'nested': True},
            'tuple': (4, 5, 6)
        }
        assert result.return_value == expected

    def test_exception_with_output(self):
        """Test exception after some successful output."""
        executor = PythonExecutor()
        code = """
print("Before error")
x = 5
print(f"x = {x}")
raise ValueError("Test error")
"""
        result = executor.execute(code)

        assert result.success is False
        assert "Before error" in result.stdout
        assert "x = 5" in result.stdout
        assert result.exception_type == "ValueError"

    def test_keyword_detection(self):
        """Test proper handling of Python keywords in last line."""
        executor = PythonExecutor()

        # Test various keywords that should not be treated as expressions
        keywords_to_test = [
            "def test(): pass",
            "class Test: pass",
            "if True: pass",
            "for i in range(1): pass",
            "while False: pass",
            "with open('/dev/null') as f: pass",
            "try: pass\nexcept: pass",
            "import os",
            "from math import sqrt"
        ]

        for code in keywords_to_test:
            result = executor.execute(code)
            # Should execute as statement, not try to evaluate as expression
            assert result.success is True
            assert result.return_value is None

    def test_long_output_handling(self):
        """Test handling of long output strings."""
        executor = PythonExecutor()
        code = """
for i in range(100):
    print(f"Line {i}")
"completed"
"""
        result = executor.execute(code)

        assert result.success is True
        assert result.return_value == "completed"
        assert len(result.stdout.split('\n')) >= 100

    @pytest.mark.edge_case
    def test_recursive_execution(self):
        """Test deeply recursive code execution."""
        executor = PythonExecutor()
        code = """
def deep_recursion(n, acc=0):
    if n <= 0:
        return acc
    return deep_recursion(n-1, acc + n)

# Use moderate depth to avoid stack overflow in tests
deep_recursion(50)
"""
        result = executor.execute(code)

        assert result.success is True
        assert result.return_value == sum(range(1, 51))

    @pytest.mark.edge_case
    def test_memory_intensive_operation(self):
        """Test memory-intensive operations."""
        executor = PythonExecutor()
        code = """
# Create a reasonably large list
big_list = list(range(10000))
len(big_list)
"""
        result = executor.execute(code)

        assert result.success is True
        assert result.return_value == 10000