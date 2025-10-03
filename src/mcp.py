# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Context Protocol (MCP) integration for CodeAct environments.

This module provides tools and clients for integrating with MCP servers,
making external capabilities available to code execution environments.
"""

from typing import Any, Dict, List, Optional, Union

from .environment import CodeActEnvironment


class MCPClient:
    """Client for Model Context Protocol servers.

    For now this provides mock tools. In production this would:
    - Connect to MCP servers via JSON-RPC
    - Handle tool discovery and invocation
    - Manage async communication
    """

    def __init__(self, server_config: Optional[Dict[str, Any]] = None):
        self.server_config = server_config or {}
        self.tools: Dict[str, Any] = {}
        self._setup_mock_tools()

    def _setup_mock_tools(self):
        """Set up mock tools for development."""

        def file_read(path: str) -> str:
            try:
                with open(path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        def file_write(path: str, content: str) -> str:
            try:
                with open(path, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {path}"
            except Exception as e:
                return f"Error writing file: {e}"

        def web_search(query: str) -> str:
            return f"Mock search results for: {query}"

        def calculator(expression: str) -> Union[float, str]:
            try:
                # Simple safe calculator
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "Error: Invalid characters"
                return eval(expression)
            except Exception as e:
                return f"Error: {e}"

        self.tools.update({
            'file_read': file_read,
            'file_write': file_write,
            'web_search': web_search,
            'calculator': calculator,
        })

    def get_tools(self) -> Dict[str, Any]:
        """Get all available tools."""
        return self.tools.copy()

    def get_tool_names(self) -> List[str]:
        """Get names of available tools."""
        return list(self.tools.keys())


def create_mcp_environment(
    mcp_config: Optional[Dict[str, Any]] = None,
    additional_tools: Optional[Dict[str, Any]] = None
) -> CodeActEnvironment:
    """Create a CodeAct environment with MCP tool integration."""

    # Get MCP tools
    client = MCPClient(mcp_config)
    mcp_tools = client.get_tools()

    # Combine with additional tools
    all_tools = {**mcp_tools}
    if additional_tools:
        all_tools.update(additional_tools)

    return CodeActEnvironment(tools=all_tools)
