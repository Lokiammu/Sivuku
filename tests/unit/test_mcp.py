# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for MCP integration classes.
"""

import pytest
from unittest.mock import patch, mock_open

from src.mcp import MCPClient, create_mcp_environment
from src.environment import CodeActEnvironment
from src.types import CodeAction


class TestMCPClient:
    """Test the MCPClient class."""

    def test_initialization(self):
        """Test MCPClient initialization."""
        client = MCPClient()
        assert client.server_config == {}
        assert isinstance(client.tools, dict)
        assert len(client.tools) > 0  # Should have mock tools

    def test_initialization_with_config(self):
        """Test MCPClient initialization with config."""
        config = {"server_url": "http://localhost:8080"}
        client = MCPClient(server_config=config)
        assert client.server_config == config

    def test_mock_tools_available(self):
        """Test that mock tools are properly set up."""
        client = MCPClient()
        tools = client.get_tools()

        expected_tools = ['file_read', 'file_write', 'web_search', 'calculator']
        for tool in expected_tools:
            assert tool in tools
            assert callable(tools[tool])

    def test_get_tool_names(self):
        """Test getting tool names."""
        client = MCPClient()
        names = client.get_tool_names()

        assert isinstance(names, list)
        assert 'file_read' in names
        assert 'file_write' in names
        assert 'calculator' in names

    def test_file_read_tool(self):
        """Test the mock file_read tool."""
        client = MCPClient()
        file_read = client.tools['file_read']

        # Mock successful file read
        with patch("builtins.open", mock_open(read_data="test content")):
            result = file_read("/test/path.txt")
            assert result == "test content"

        # Mock file read error
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            result = file_read("/nonexistent/file.txt")
            assert "Error reading file" in result

    def test_file_write_tool(self):
        """Test the mock file_write tool."""
        client = MCPClient()
        file_write = client.tools['file_write']

        # Mock successful file write
        with patch("builtins.open", mock_open()) as mock_file:
            result = file_write("/test/path.txt", "test content")
            assert "Successfully wrote to" in result
            mock_file.assert_called_once_with("/test/path.txt", 'w')

        # Mock file write error
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = file_write("/restricted/file.txt", "content")
            assert "Error writing file" in result

    def test_calculator_tool(self):
        """Test the mock calculator tool."""
        client = MCPClient()
        calculator = client.tools['calculator']

        # Test valid expressions
        assert calculator("2 + 3") == 5
        assert calculator("10 / 2") == 5.0
        assert calculator("2 * (3 + 4)") == 14

        # Test invalid characters
        result = calculator("import os")
        assert "Invalid characters" in result

        # Test invalid expression
        result = calculator("2 + ")
        assert "Error:" in result

    def test_web_search_tool(self):
        """Test the mock web_search tool."""
        client = MCPClient()
        web_search = client.tools['web_search']

        result = web_search("python programming")
        assert "Mock search results for: python programming" in result

    def test_get_tools_returns_copy(self):
        """Test that get_tools returns a copy, not the original."""
        client = MCPClient()
        tools1 = client.get_tools()
        tools2 = client.get_tools()

        # Should be equal but not the same object
        assert tools1 == tools2
        assert tools1 is not tools2

        # Modifying one shouldn't affect the other
        tools1['new_tool'] = lambda: None
        assert 'new_tool' not in tools2


class TestCreateMCPEnvironment:
    """Test the create_mcp_environment factory function."""

    def test_create_basic_mcp_environment(self):
        """Test creating a basic MCP environment."""
        env = create_mcp_environment()

        assert isinstance(env, CodeActEnvironment)

        # Should have MCP tools available
        obs = env.reset()
        mcp_tools = ['file_read', 'file_write', 'web_search', 'calculator']

        for tool in mcp_tools:
            assert tool in obs.available_tools

    def test_create_mcp_environment_with_config(self):
        """Test creating MCP environment with custom config."""
        config = {"server_url": "http://localhost:8080"}
        env = create_mcp_environment(mcp_config=config)

        assert isinstance(env, CodeActEnvironment)
        # Environment should be created successfully with config
        obs = env.reset()
        assert len(obs.available_tools) > 0

    def test_create_mcp_environment_with_additional_tools(self):
        """Test creating MCP environment with additional tools."""
        additional_tools = {'custom_tool': lambda x: x * 2}
        env = create_mcp_environment(additional_tools=additional_tools)

        obs = env.reset()

        # Should have both MCP and additional tools
        assert 'file_read' in obs.available_tools  # MCP tool
        assert 'custom_tool' in obs.available_tools  # Additional tool

    def test_mcp_tools_functionality(self):
        """Test that MCP tools work in the environment."""
        env = create_mcp_environment()
        env.reset()

        # Test calculator tool
        obs1 = env.step(CodeAction(code='calculator("2 + 3")'))
        assert obs1.execution_result.success is True
        assert obs1.execution_result.return_value == 5

        # Test web search tool
        obs2 = env.step(CodeAction(code='web_search("test query")'))
        assert obs2.execution_result.success is True
        assert "Mock search results" in str(obs2.execution_result.return_value)

    def test_mcp_file_operations(self):
        """Test MCP file operations in environment."""
        env = create_mcp_environment()
        env.reset()

        # Test file write and read
        code = '''
write_result = file_write("/tmp/mcp_test.txt", "Hello MCP!")
read_result = file_read("/tmp/mcp_test.txt")
print(f"Write: {write_result}")
print(f"Read: {read_result}")
read_result
'''
        obs = env.step(CodeAction(code=code))

        assert obs.execution_result.success is True
        assert "Hello MCP!" in obs.execution_result.return_value

    def test_mcp_calculator_integration(self):
        """Test MCP calculator integration with Python code."""
        env = create_mcp_environment()
        env.reset()

        code = '''
# Use calculator for complex computation
x = calculator("2 * 3 + 4")
y = calculator("10 / 2")
result = x + y
print(f"Calculator results: x={x}, y={y}, total={result}")
result
'''
        obs = env.step(code=CodeAction(code=code))

        assert obs.execution_result.success is True
        assert obs.execution_result.return_value == 15  # (2*3+4) + (10/2) = 10 + 5 = 15
        assert "Calculator results" in obs.execution_result.stdout

    @pytest.mark.integration
    def test_mcp_environment_state_persistence(self):
        """Test that MCP tools work with persistent state."""
        env = create_mcp_environment()
        env.reset()

        # Step 1: Calculate and store values
        obs1 = env.step(CodeAction(code='''
val1 = calculator("5 * 6")
val2 = calculator("100 / 4")
print(f"Calculated: val1={val1}, val2={val2}")
'''))
        assert obs1.execution_result.success is True

        # Step 2: Use stored values
        obs2 = env.step(CodeAction(code='''
total = val1 + val2
print(f"Total: {total}")
total
'''))
        assert obs2.execution_result.success is True
        assert obs2.execution_result.return_value == 55  # 30 + 25

    @pytest.mark.edge_case
    def test_mcp_tool_error_handling(self):
        """Test error handling in MCP tools."""
        env = create_mcp_environment()
        env.reset()

        # Test calculator with invalid input
        obs1 = env.step(CodeAction(code='calculator("invalid expression")'))
        assert obs1.execution_result.success is True
        result = obs1.execution_result.return_value
        assert "Error:" in result

        # Test file operations with invalid paths
        obs2 = env.step(CodeAction(code='file_read("/nonexistent/path")'))
        assert obs2.execution_result.success is True
        result = obs2.execution_result.return_value
        assert "Error reading file" in result

    def test_mcp_environment_isolation(self):
        """Test that different MCP environments are isolated."""
        env1 = create_mcp_environment()
        env2 = create_mcp_environment()

        env1.reset()
        env2.reset()

        # Modify state in env1
        env1.step(CodeAction(code="env1_var = 42"))

        # Check that env2 doesn't have env1's variable
        obs = env2.step(CodeAction(code="env1_var"))
        assert obs.execution_result.success is False
        assert obs.execution_result.exception_type == "NameError"