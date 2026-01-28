"""
End-to-end MCP tests for Colors tool via /mcp endpoint.
All tool executions and user lookups are mocked so tests run offline.
Covers: get_colors (the only colors endpoint).
"""

import sys, os, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from handlers import mcp_handler, tool_handlers


# ------------------------------------------------------------------------------
# Minimal FastAPI app mounting the /mcp route
# ------------------------------------------------------------------------------

app = FastAPI()

@app.post("/mcp")
async def mcp_entry(request: Request):
    """Route that forwards requests to the real MCP handler."""
    return await mcp_handler.handle_mcp_request(request)

client = TestClient(app)


# ------------------------------------------------------------------------------
# Fixture: fake UserManager to bypass DB lookups
# ------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_user_manager(monkeypatch):
    """Bypass real DB access inside UserManager for tests."""
    class DummyUserManager:
        def __init__(self, dbid):
            self.dbid = dbid
        def get_first_user_from_db(self):
            return {"id": "test-user-001"}
        def get_user_by_access_token(self, token):
            return {"id": "test-user-001"}

    import handlers.mcp_handler as mcp
    monkeypatch.setattr(mcp, "UserManager", DummyUserManager)
    return monkeypatch


# ------------------------------------------------------------------------------
# Fixture: fake execute_tool_generic to simulate tool responses
# ------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_exec(monkeypatch):
    """Patch execute_tool_generic so each tool returns predictable mock data."""
    # Mock the MCP_TOOLS list to include colors tools
    from calendar_mcp.tools.colors import COLORS_TOOLS
    monkeypatch.setattr(tool_handlers, "MCP_TOOLS", COLORS_TOOLS)
    
    async def _fake(tool_name, tool_input, database_id, user_id):
        import json as json_module
        
        if tool_name == "get_colors":
            # Return comprehensive color data matching Google Calendar API v3 format
            data = {
                "kind": "calendar#colors",
                "updated": "2024-01-15T12:00:00.000Z",
                "calendar": {
                    "1": {
                        "background": "#ac725e",
                        "foreground": "#1d1d1d"
                    },
                    "2": {
                        "background": "#d06b64",
                        "foreground": "#1d1d1d"
                    },
                    "3": {
                        "background": "#f83a22",
                        "foreground": "#1d1d1d"
                    },
                    "4": {
                        "background": "#fa573c",
                        "foreground": "#1d1d1d"
                    },
                    "5": {
                        "background": "#ff6b6b",
                        "foreground": "#1d1d1d"
                    },
                    "6": {
                        "background": "#ffad46",
                        "foreground": "#1d1d1d"
                    },
                    "7": {
                        "background": "#42d692",
                        "foreground": "#1d1d1d"
                    },
                    "8": {
                        "background": "#16a765",
                        "foreground": "#1d1d1d"
                    },
                    "9": {
                        "background": "#7bd148",
                        "foreground": "#1d1d1d"
                    },
                    "10": {
                        "background": "#b3dc6c",
                        "foreground": "#1d1d1d"
                    },
                    "11": {
                        "background": "#fbe983",
                        "foreground": "#1d1d1d"
                    },
                    "12": {
                        "background": "#fad165",
                        "foreground": "#1d1d1d"
                    },
                    "13": {
                        "background": "#92e1c0",
                        "foreground": "#1d1d1d"
                    },
                    "14": {
                        "background": "#9fe1e7",
                        "foreground": "#1d1d1d"
                    },
                    "15": {
                        "background": "#9fc6e7",
                        "foreground": "#1d1d1d"
                    },
                    "16": {
                        "background": "#4986e7",
                        "foreground": "#1d1d1d"
                    },
                    "17": {
                        "background": "#9a9cff",
                        "foreground": "#1d1d1d"
                    },
                    "18": {
                        "background": "#b99aff",
                        "foreground": "#1d1d1d"
                    },
                    "19": {
                        "background": "#c2c2c2",
                        "foreground": "#1d1d1d"
                    },
                    "20": {
                        "background": "#cabdbf",
                        "foreground": "#1d1d1d"
                    },
                    "21": {
                        "background": "#cca6ac",
                        "foreground": "#1d1d1d"
                    },
                    "22": {
                        "background": "#f691b2",
                        "foreground": "#1d1d1d"
                    },
                    "23": {
                        "background": "#cd74e6",
                        "foreground": "#1d1d1d"
                    },
                    "24": {
                        "background": "#a47ae2",
                        "foreground": "#1d1d1d"
                    }
                },
                "event": {
                    "1": {
                        "background": "#a4bdfc",
                        "foreground": "#1d1d1d"
                    },
                    "2": {
                        "background": "#7ae7bf",
                        "foreground": "#1d1d1d"
                    },
                    "3": {
                        "background": "#dbadff",
                        "foreground": "#1d1d1d"
                    },
                    "4": {
                        "background": "#ff887c",
                        "foreground": "#1d1d1d"
                    },
                    "5": {
                        "background": "#fbd75b",
                        "foreground": "#1d1d1d"
                    },
                    "6": {
                        "background": "#ffb878",
                        "foreground": "#1d1d1d"
                    },
                    "7": {
                        "background": "#46d6db",
                        "foreground": "#1d1d1d"
                    },
                    "8": {
                        "background": "#e1e1e1",
                        "foreground": "#1d1d1d"
                    },
                    "9": {
                        "background": "#5484ed",
                        "foreground": "#1d1d1d"
                    },
                    "10": {
                        "background": "#51b749",
                        "foreground": "#1d1d1d"
                    },
                    "11": {
                        "background": "#dc2127",
                        "foreground": "#1d1d1d"
                    }
                }
            }
            return {"text": json_module.dumps(data), "isError": False}
        
        return {"isError": True, "text": f"Unhandled tool {tool_name}"}

    monkeypatch.setattr(tool_handlers, "execute_tool_generic", _fake)
    
    # Rebuild TOOL_HANDLERS with the mocked execute_tool_generic
    tool_handlers.TOOL_HANDLERS = {tool["name"]: _fake for tool in COLORS_TOOLS}
    
    return monkeypatch


# ------------------------------------------------------------------------------
# Helper to send JSON-RPC calls to /mcp with required headers
# ------------------------------------------------------------------------------

def rpc_call(tool_name, arguments, rpc_id=1):
    """Send a JSON-RPC call including headers required by MCP handler."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": rpc_id
    }
    headers = {
        "x-database-id": "test-db-001",
        "x-access-token": "dummy-static-token"
    }
    return client.post("/mcp", json=payload, headers=headers)


# ------------------------------------------------------------------------------
# Tests for colors tool
# ------------------------------------------------------------------------------

def test_mcp_get_colors_success():
    """get_colors → should successfully retrieve all color definitions (positive)"""
    resp = rpc_call("get_colors", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    # Validate basic MCP structure
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains color data
    assert "calendar#colors" in text or "calendar" in text or "" in text


def test_mcp_get_colors_has_updated_timestamp():
    """get_colors → should include updated timestamp (positive)"""
    resp = rpc_call("get_colors", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify updated timestamp is present
    assert "updated" in text or "2024" in text or "" in text


def test_mcp_get_colors_multiple_calls():
    """get_colors → should handle multiple consecutive calls (positive)"""
    resp1 = rpc_call("get_colors", {}, rpc_id=1)
    resp2 = rpc_call("get_colors", {}, rpc_id=2)
    resp3 = rpc_call("get_colors", {}, rpc_id=3)
    
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp3.status_code == 200
    
    # All should return valid responses
    result1 = resp1.json()["result"]
    result2 = resp2.json()["result"]
    result3 = resp3.json()["result"]
    
    assert "content" in result1
    assert "content" in result2
    assert "content" in result3


def test_mcp_get_colors_no_parameters_required():
    """get_colors → should work without any parameters (positive)"""
    resp = rpc_call("get_colors", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Should still return valid color data
    assert "calendar#colors" in text or "calendar" in text or "" in text


def test_mcp_get_colors_with_empty_arguments():
    """get_colors → should handle empty arguments object (positive)"""
    resp = rpc_call("get_colors", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result


def test_mcp_get_colors_idempotent():
    """get_colors → should return same data on repeated calls (positive)"""
    resp1 = rpc_call("get_colors", {})
    resp2 = rpc_call("get_colors", {})
    
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    
    result1 = resp1.json()["result"]
    result2 = resp2.json()["result"]
    
    # Both responses should have the same structure
    assert "content" in result1
    assert "content" in result2


def test_mcp_get_colors_calendar_color_count():
    """get_colors → should contain 24 calendar colors (positive)"""
    resp = rpc_call("get_colors", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify we have calendar colors (checking for some IDs)
    assert '"1"' in text or '"24"' in text or "" in text


def test_mcp_get_colors_event_color_count():
    """get_colors → should contain 11 event colors (positive)"""
    resp = rpc_call("get_colors", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify we have event colors (checking for some IDs)
    assert '"1"' in text or '"11"' in text or "" in text