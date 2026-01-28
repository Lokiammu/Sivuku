"""
MCP integration test: Verify that /mcp tools/list returns all registered tools.
Checks registry consistency between MCP_TOOLS_LIST and TOOL_HANDLERS.
"""

import sys, os, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from handlers import mcp_handler
from handlers.tool_handlers import MCP_TOOLS_LIST, TOOL_HANDLERS


# ------------------------------------------------------------------------------
# FastAPI app mounting /mcp
# ------------------------------------------------------------------------------

app = FastAPI()

@app.post("/mcp")
async def mcp_entry(request: Request):
    """Forward JSON-RPC request to MCP handler."""
    return await mcp_handler.handle_mcp_request(request)

client = TestClient(app)


# ------------------------------------------------------------------------------
# Test 1: Direct call to /mcp tools/list
# ------------------------------------------------------------------------------

def test_mcp_tools_list_endpoint():
    """ Verify that /mcp tools/list returns JSON-RPC result with all tools."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "params": {},
        "id": 1
    }
    headers = {"x-database-id": "test-db-001"}  # optional header
    response = client.post("/mcp", json=payload, headers=headers)

    assert response.status_code == 200, "Expected HTTP 200 from /mcp"
    body = response.json()
    assert "result" in body, f"Missing JSON-RPC result: {body}"

    result = body["result"]
    assert "tools" in result, f"Expected 'tools' key in result, got: {result}"
    tool_count = len(result["tools"])
    print(f"\n📦 /mcp tools/list returned {tool_count} tools")

    assert tool_count == len(MCP_TOOLS_LIST), (
        f"Mismatch between registered tool list ({len(MCP_TOOLS_LIST)}) and returned count ({tool_count})"
    )


# ------------------------------------------------------------------------------
# Test 2: Internal registry consistency
# ------------------------------------------------------------------------------

def test_tool_registry_consistency():
    """ Verify that every tool name in MCP_TOOLS_LIST has a handler."""
    tool_names = {t["name"] for t in MCP_TOOLS_LIST}
    missing = [name for name in tool_names if name not in TOOL_HANDLERS]

    print(f"\n {len(tool_names)} tools in MCP_TOOLS_LIST")
    print(f" {len(TOOL_HANDLERS)} handlers registered")

    assert not missing, f"Missing handlers for: {missing}"
    assert len(tool_names) >= 10, "Expected at least 10 registered tools"
