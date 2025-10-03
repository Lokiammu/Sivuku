# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Edge case tests for EnvTorch components.

These tests verify robustness against unusual inputs and boundary conditions.
"""

import pytest
import sys
from unittest.mock import patch, Mock

from src import (
    create_codeact_env,
    create_mcp_environment,
    CodeAction,
    PythonExecutor,
    CodeActEnvironment,
)
from src.types import ExecutionResult


class TestInputEdgeCases:
    """Test edge cases in input handling."""

    @pytest.mark.edge_case
    def test_empty_and_whitespace_code(self):
        """Test handling of empty and whitespace-only code."""
        env = create_codeact_env()
        env.reset()

        # Empty string should raise error during CodeAction creation
        with pytest.raises(ValueError):
            CodeAction(code="")

        # Whitespace only should raise error
        with pytest.raises(ValueError):
            CodeAction(code="   \n\t  ")

        # Single space should raise error
        with pytest.raises(ValueError):
            CodeAction(code=" ")

    @pytest.mark.edge_case
    def test_extremely_long_code(self):
        """Test handling of very long code strings."""
        env = create_codeact_env()
        env.reset()

        # Generate very long code (1MB)
        long_assignment = "x = " + "1" * (1024 * 1024 - 4)
        action = CodeAction(code=long_assignment)

        obs = env.step(action)
        assert obs.execution_result.success is True

    @pytest.mark.edge_case
    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in code."""
        env = create_codeact_env()
        env.reset()

        # Unicode in string literals
        unicode_code = '''
message = "Hello 🌍 世界 🚀"
emoji_count = len([c for c in message if ord(c) > 127])
print(f"Unicode message: {message}")
print(f"Non-ASCII characters: {emoji_count}")
emoji_count
'''
        obs = env.step(CodeAction(code=unicode_code))
        assert obs.execution_result.success is True
        assert obs.execution_result.return_value > 0

        # Unicode in variable names (Python 3 supports this)
        unicode_var_code = '''
π = 3.14159
résultat = π * 2
résultat
'''
        obs = env.step(CodeAction(code=unicode_var_code))
        assert obs.execution_result.success is True
        assert abs(obs.execution_result.return_value - 6.28318) < 0.001

    @pytest.mark.edge_case
    def test_nested_quotes_and_escapes(self):
        """Test handling of complex string literals."""
        env = create_codeact_env()
        env.reset()

        complex_string_code = r'''
# Test various quote combinations
single = 'He said "Hello"'
double = "She replied 'Hi there!'"
triple_single = """This has 'single' and "double" quotes"""
triple_double = '''This also has "double" and 'single' quotes'''
escaped = "Line 1\nLine 2\tTabbed\\"Quoted\\""

all_strings = [single, double, triple_single, triple_double, escaped]
len(all_strings)
'''
        obs = env.step(CodeAction(code=complex_string_code))
        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 5


class TestExecutionEdgeCases:
    """Test edge cases in code execution."""

    @pytest.mark.edge_case
    def test_infinite_loop_detection(self):
        """Test that infinite loops can be interrupted (if timeout implemented)."""
        env = create_codeact_env()
        env.reset()

        # This would be an infinite loop, but we don't have timeout yet
        # So we test a long-running but finite loop instead
        long_loop_code = '''
count = 0
for i in range(1000000):
    count += 1
    if count >= 100000:  # Break early to avoid actual infinite loop
        break
count
'''
        obs = env.step(CodeAction(code=long_loop_code))
        assert obs.execution_result.success is True

    @pytest.mark.edge_case
    def test_memory_intensive_operations(self):
        """Test handling of memory-intensive operations."""
        env = create_codeact_env()
        env.reset()

        # Large list creation
        large_list_code = '''
# Create a large list (but not too large to avoid OOM in tests)
big_list = list(range(1000000))
len(big_list)
'''
        obs = env.step(CodeAction(code=large_list_code))
        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 1000000

        # Nested data structures
        nested_code = '''
# Create nested structure
nested = {}
for i in range(100):
    nested[f"key_{i}"] = {f"subkey_{j}": j for j in range(100)}

total_keys = sum(len(v) for v in nested.values())
total_keys
'''
        obs = env.step(CodeAction(code=nested_code))
        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 10000

    @pytest.mark.edge_case
    def test_recursion_depth_limits(self):
        """Test behavior near Python's recursion limits."""
        env = create_codeact_env()
        env.reset()

        # Test with deep but safe recursion
        safe_recursion_code = '''
import sys
original_limit = sys.getrecursionlimit()

def safe_recursive(n, acc=0):
    if n <= 0:
        return acc
    return safe_recursive(n - 1, acc + n)

# Use a safe depth (well below typical limits)
result = safe_recursive(500)
result
'''
        obs = env.step(CodeAction(code=safe_recursion_code))
        assert obs.execution_result.success is True

        # Test recursion limit exceeded
        deep_recursion_code = '''
def deep_recursive(n):
    if n <= 0:
        return 0
    return deep_recursive(n - 1) + 1

# This should hit recursion limit
try:
    result = deep_recursive(10000)
except RecursionError:
    result = "RecursionError caught"

result
'''
        obs = env.step(CodeAction(code=deep_recursion_code))
        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == "RecursionError caught"

    @pytest.mark.edge_case
    def test_system_exit_handling(self):
        """Test handling of sys.exit() calls."""
        env = create_codeact_env()
        env.reset()

        exit_code = '''
import sys
try:
    sys.exit(42)
except SystemExit as e:
    exit_code = e.code

exit_code
'''
        obs = env.step(CodeAction(code=exit_code))
        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 42


class TestErrorHandlingEdgeCases:
    """Test edge cases in error handling."""

    @pytest.mark.edge_case
    def test_multiple_exception_types(self):
        """Test handling of various exception types."""
        env = create_codeact_env()
        env.reset()

        exception_tests = [
            ("ZeroDivisionError", "1 / 0"),
            ("NameError", "undefined_variable"),
            ("TypeError", "'string' + 42"),
            ("IndexError", "[1, 2, 3][10]"),
            ("KeyError", "{}['nonexistent']"),
            ("ValueError", "int('not_a_number')"),
            ("AttributeError", "'string'.nonexistent_method()"),
        ]

        for expected_type, code in exception_tests:
            obs = env.step(CodeAction(code=code))
            assert obs.execution_result.success is False
            assert obs.execution_result.exception_type == expected_type

    @pytest.mark.edge_case
    def test_syntax_error_variations(self):
        """Test various syntax error conditions."""
        env = create_codeact_env()
        env.reset()

        syntax_errors = [
            "if True",  # Missing colon
            "def func()",  # Missing colon
            "for i in range(10)",  # Missing colon
            "print('unclosed string",  # Unclosed string
            "print(unclosed_paren",  # Unclosed parenthesis
            "invalid syntax here",  # Invalid tokens
            "1 + + 2",  # Double operator
        ]

        for bad_code in syntax_errors:
            obs = env.step(CodeAction(code=bad_code))
            assert obs.execution_result.success is False
            assert obs.execution_result.exception_type == "SyntaxError"

    @pytest.mark.edge_case
    def test_indentation_errors(self):
        """Test various indentation error conditions."""
        env = create_codeact_env()
        env.reset()

        indentation_errors = [
            "if True:\nprint('no indent')",  # Missing indentation
            "    print('unexpected indent')",  # Unexpected indentation
            "if True:\n    print('good')\n  print('bad indent')",  # Mixed indentation
        ]

        for bad_code in indentation_errors:
            obs = env.step(CodeAction(code=bad_code))
            assert obs.execution_result.success is False
            assert obs.execution_result.exception_type == "IndentationError"

    @pytest.mark.edge_case
    def test_import_errors(self):
        """Test various import error conditions."""
        env = create_codeact_env()
        env.reset()

        # Non-existent module
        obs1 = env.step(CodeAction(code="import nonexistent_module"))
        assert obs1.execution_result.success is False
        assert obs1.execution_result.exception_type == "ModuleNotFoundError"

        # Non-existent attribute from existing module
        obs2 = env.step(CodeAction(code="from math import nonexistent_function"))
        assert obs2.execution_result.success is False
        assert obs2.execution_result.exception_type == "ImportError"


class TestStateManagementEdgeCases:
    """Test edge cases in state management."""

    @pytest.mark.edge_case
    def test_variable_shadowing_and_scoping(self):
        """Test complex variable scoping scenarios."""
        env = create_codeact_env()
        env.reset()

        scoping_code = '''
# Global variable
x = "global"

def outer():
    x = "outer"

    def inner():
        x = "inner"
        return x

    return inner(), x

result = outer()
global_x = x
(inner_result, outer_result, global_x)
'''
        obs = env.step(CodeAction(code=scoping_code))
        assert obs.execution_result.success is True
        inner_result, outer_result, global_x = obs.execution_result.return_value
        assert inner_result == "inner"
        assert outer_result == "outer"
        assert global_x == "global"

    @pytest.mark.edge_case
    def test_namespace_pollution_protection(self):
        """Test protection against namespace pollution."""
        env = create_codeact_env()
        env.reset()

        # Override built-in function
        obs1 = env.step(CodeAction(code='''
# Override built-in
len = lambda x: "overridden"
result1 = len([1, 2, 3])
result1
'''))
        assert obs1.execution_result.success is True
        assert obs1.execution_result.return_value == "overridden"

        # Reset and check that built-in is restored
        obs = env.reset()

        obs2 = env.step(CodeAction(code='''
result2 = len([1, 2, 3])
result2
'''))
        assert obs2.execution_result.success is True
        assert obs2.execution_result.return_value == 3

    @pytest.mark.edge_case
    def test_complex_object_persistence(self):
        """Test persistence of complex objects across steps."""
        env = create_codeact_env()
        env.reset()

        # Create complex object
        obs1 = env.step(CodeAction(code='''
class ComplexObject:
    def __init__(self):
        self.data = {"nested": {"deep": [1, 2, 3]}}
        self.methods_called = []

    def method1(self):
        self.methods_called.append("method1")
        return "method1_result"

    def method2(self, arg):
        self.methods_called.append(f"method2_{arg}")
        return f"method2_result_{arg}"

complex_obj = ComplexObject()
complex_obj.method1()
'''))
        assert obs1.execution_result.success is True

        # Use complex object in next step
        obs2 = env.step(CodeAction(code='''
result = complex_obj.method2("test")
call_history = complex_obj.methods_called
(result, call_history)
'''))
        assert obs2.execution_result.success is True
        result, call_history = obs2.execution_result.return_value
        assert result == "method2_result_test"
        assert call_history == ["method1", "method2_test"]


class TestTransformEdgeCases:
    """Test edge cases in transform applications."""

    @pytest.mark.edge_case
    def test_transform_with_none_values(self):
        """Test transform handling of None and missing values."""
        from src.transforms import MathProblemTransform
        from src.types import CodeObservation, ExecutionResult

        transform = MathProblemTransform(expected_answer=42)

        # Observation with None return value
        obs1 = CodeObservation(
            execution_result=ExecutionResult(success=True, return_value=None)
        )
        result1 = transform(obs1)
        assert result1.reward == 0.0

        # Observation with failed execution
        obs2 = CodeObservation(
            execution_result=ExecutionResult(success=False, return_value=None)
        )
        result2 = transform(obs2)
        assert result2.reward == -0.5  # Error penalty

    @pytest.mark.edge_case
    def test_transform_with_unusual_return_types(self):
        """Test transform handling of unusual return value types."""
        from src.transforms import MathProblemTransform
        from src.types import CodeObservation, ExecutionResult

        transform = MathProblemTransform(expected_answer=42)

        unusual_values = [
            ("string", "not_a_number"),
            ("list", [42]),
            ("dict", {"answer": 42}),
            ("complex", 42 + 0j),
            ("bool", True),
        ]

        for value_type, value in unusual_values:
            obs = CodeObservation(
                execution_result=ExecutionResult(success=True, return_value=value)
            )
            result = transform(obs)

            if value_type == "complex" and value == 42 + 0j:
                assert result.reward == 1.0  # Complex number with right value
            elif value_type == "bool" and value is True:
                # True can be converted to 1, not 42
                assert result.reward == 0.0
            else:
                assert result.reward == 0.0  # Can't convert to expected number

    @pytest.mark.edge_case
    def test_safety_transform_edge_cases(self):
        """Test edge cases in safety transform pattern matching."""
        from src.transforms import CodeSafetyTransform
        from src.types import CodeObservation

        transform = CodeSafetyTransform()

        # Edge cases in pattern matching
        edge_cases = [
            ("# import os  # commented out", False),  # Commented import
            ("text = 'import os in string'", False),  # Import in string
            ("import_os = 'variable name'", False),  # Variable with import in name
            ("import os", True),  # Actual dangerous import
            ("from os import path", True),  # From import
            ("    import os  ", True),  # With whitespace
        ]

        for code, should_detect in edge_cases:
            obs = CodeObservation(metadata={'last_code': code})
            result = transform(obs)

            if should_detect:
                assert result.reward < 0, f"Should detect danger in: {code}"
                assert 'safety_violation' in result.metadata
            else:
                assert result.reward >= 0, f"Should not detect danger in: {code}"
                assert 'safety_violation' not in result.metadata


class TestMCPEdgeCases:
    """Test edge cases in MCP tool integration."""

    @pytest.mark.edge_case
    def test_mcp_tool_error_conditions(self):
        """Test MCP tools under error conditions."""
        env = create_mcp_environment()
        env.reset()

        # Test calculator with invalid expressions
        calc_errors = [
            '1 / 0',  # Division by zero should be caught by eval
            '2 ** 10000000',  # Very large number
            'invalid_function()',  # Unknown function
            '',  # Empty expression
            '2 +',  # Incomplete expression
        ]

        for expr in calc_errors:
            obs = env.step(CodeAction(code=f'calculator("{expr}")'))
            assert obs.execution_result.success is True
            result = obs.execution_result.return_value
            assert isinstance(result, str) and "Error" in result

    @pytest.mark.edge_case
    def test_file_operations_edge_cases(self, temp_dir):
        """Test file operations with edge cases."""
        env = create_mcp_environment()
        env.reset()

        import os

        # Test with various problematic paths
        edge_cases = [
            ("empty_file", ""),  # Empty content
            ("unicode_file", "Hello 🌍 Unicode! 你好"),  # Unicode content
            ("large_file", "X" * 10000),  # Large content
            ("multiline_file", "Line 1\nLine 2\nLine 3\n"),  # Multilines
        ]

        for filename, content in edge_cases:
            filepath = os.path.join(temp_dir, filename)

            code = f'''
# Write and read back
write_result = file_write("{filepath}", """{content}""")
read_result = file_read("{filepath}")
matches = read_result == """{content}"""
(len(read_result), matches)
'''
            obs = env.step(CodeAction(code=code))
            assert obs.execution_result.success is True

            length, matches = obs.execution_result.return_value
            assert length == len(content)
            assert matches is True

    @pytest.mark.edge_case
    def test_web_search_unusual_queries(self):
        """Test web search with unusual query strings."""
        env = create_mcp_environment()
        env.reset()

        unusual_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "🔍 emoji search 🚀",  # With emojis
            "query with\nnewlines",  # With newlines
            "very " * 100 + "long query",  # Very long query
            "special!@#$%^&*()chars",  # Special characters
        ]

        for query in unusual_queries:
            obs = env.step(CodeAction(code=f'web_search("{query}")'))
            assert obs.execution_result.success is True
            result = obs.execution_result.return_value
            assert isinstance(result, str)
            assert query in result or "Mock search results" in result


class TestConcurrencyEdgeCases:
    """Test edge cases that might arise in concurrent scenarios."""

    @pytest.mark.edge_case
    def test_rapid_reset_and_step_cycles(self):
        """Test rapid cycling between reset and step operations."""
        env = create_codeact_env()

        # Rapidly cycle reset/step operations
        for i in range(100):
            obs = env.reset()
            obs = env.step(CodeAction(code=f"x = {i}"))
            assert obs.execution_result.success is True
            assert obs.execution_result.return_value is None

            # Immediately access the variable
            obs = env.step(CodeAction(code="x"))
            assert obs.execution_result.success is True
            assert obs.execution_result.return_value == i

    @pytest.mark.edge_case
    def test_environment_isolation_stress_test(self):
        """Stress test environment isolation."""
        envs = [create_codeact_env() for _ in range(10)]

        # Each environment gets unique state
        for i, env in enumerate(envs):
            env.reset()
            env.step(CodeAction(code=f"env_id = {i}"))

        # Cross-check that environments don't interfere
        for i, env in enumerate(envs):
            obs = env.step(CodeAction(code="env_id"))
            assert obs.execution_result.success is True
            assert obs.execution_result.return_value == i

            # Try to access other environment's variables (should fail)
            other_id = (i + 1) % len(envs)
            obs = env.step(CodeAction(code=f"other_env_{other_id} = 'should not exist'"))
            obs = env.step(CodeAction(code=f"other_env_{other_id}"))
            # This should succeed because we just created it
            assert obs.execution_result.success is True

        # But other environments shouldn't see it
        for j, env in enumerate(envs):
            if j != i:
                obs = env.step(CodeAction(code=f"other_env_{other_id}"))
                assert obs.execution_result.success is False