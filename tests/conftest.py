# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pytest configuration and shared fixtures for EnvTorch tests.
"""

import sys
import tempfile
import os
from pathlib import Path

import pytest

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src import (
    CodeActEnvironment,
    CodeAction,
    PythonExecutor,
    create_codeact_env,
    create_mcp_environment,
    CodeSafetyTransform,
    MathProblemTransform,
    CodeQualityTransform,
    CompositeTransform,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def basic_env():
    """Create a basic CodeAct environment for testing."""
    return create_codeact_env()


@pytest.fixture
def mcp_env():
    """Create a CodeAct environment with MCP tools."""
    return create_mcp_environment()


@pytest.fixture
def executor():
    """Create a fresh Python executor."""
    return PythonExecutor()


@pytest.fixture
def sample_actions():
    """Sample CodeAction objects for testing."""
    return {
        'simple_math': CodeAction(code="2 + 2"),
        'variable_assignment': CodeAction(code="x = 42"),
        'function_call': CodeAction(code="print('hello')"),
        'expression_return': CodeAction(code="x = 10\nx * 2"),
        'syntax_error': CodeAction(code="invalid syntax here"),
        'runtime_error': CodeAction(code="1 / 0"),
        'import_statement': CodeAction(code="import math\nmath.sqrt(16)"),
        'multi_line': CodeAction(code="""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

factorial(5)
"""),
    }


@pytest.fixture
def sample_transforms():
    """Sample transform objects for testing."""
    return {
        'safety': CodeSafetyTransform(),
        'math_42': MathProblemTransform(expected_answer=42),
        'quality': CodeQualityTransform(),
        'composite': CompositeTransform([
            CodeSafetyTransform(),
            CodeQualityTransform()
        ])
    }


@pytest.fixture
def test_files(temp_dir):
    """Create sample files for file operation testing."""
    files = {}

    # Text file
    text_file = os.path.join(temp_dir, "test.txt")
    with open(text_file, 'w') as f:
        f.write("Hello, World!")
    files['text'] = text_file

    # JSON file
    json_file = os.path.join(temp_dir, "data.json")
    with open(json_file, 'w') as f:
        f.write('{"key": "value", "number": 42}')
    files['json'] = json_file

    return files


@pytest.fixture(autouse=True)
def cleanup_globals():
    """Ensure test isolation by cleaning up any global state."""
    yield
    # Any cleanup code can go here
    pass


# Test markers
pytest_markers = [
    "unit: Unit tests for individual components",
    "integration: Integration tests for full workflows",
    "performance: Performance and benchmark tests",
    "edge_case: Edge case and error condition tests",
    "slow: Tests that take longer to run",
]

for marker in pytest_markers:
    pytest.register_assert_rewrite(marker.split(':')[0])