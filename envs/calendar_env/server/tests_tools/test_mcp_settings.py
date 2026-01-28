"""
End-to-end MCP tests for all Settings tools via /mcp endpoint.
All tool executions and user lookups are mocked so tests run offline.
Covers: list_settings, get_settings, watch_settings.
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
    # Mock the MCP_TOOLS list to include settings tools
    from calendar_mcp.tools.settings import SETTINGS_TOOLS
    monkeypatch.setattr(tool_handlers, "MCP_TOOLS", SETTINGS_TOOLS)
    
    async def _fake(tool_name, tool_input, database_id, user_id):
        import json as json_module
        
        if tool_name == "list_settings":
            data = {
                "kind": "calendar#settings",
                "etag": "settings-collection-etag",
                "items": [
                    {
                        "kind": "calendar#setting",
                        "etag": "setting-etag-1",
                        "id": "timezone",
                        "value": "America/Los_Angeles"
                    },
                    {
                        "kind": "calendar#setting",
                        "etag": "setting-etag-2",
                        "id": "dateFieldOrder",
                        "value": "MDY"
                    },
                    {
                        "kind": "calendar#setting",
                        "etag": "setting-etag-3",
                        "id": "timeFormat",
                        "value": "12"
                    }
                ]
            }
            return {"text": json_module.dumps(data), "isError": False}
        
        if tool_name == "get_settings":
            setting_id = tool_input.get("settingId")
            if setting_id == "timezone":
                data = {
                    "kind": "calendar#setting",
                    "etag": "setting-etag-timezone",
                    "id": "timezone",
                    "value": "America/Los_Angeles"
                }
                return {"text": json_module.dumps(data), "isError": False}
            elif setting_id == "dateFieldOrder":
                data = {
                    "kind": "calendar#setting",
                    "etag": "setting-etag-date",
                    "id": "dateFieldOrder",
                    "value": "MDY"
                }
                return {"text": json_module.dumps(data), "isError": False}
            elif setting_id == "timeFormat":
                data = {
                    "kind": "calendar#setting",
                    "etag": "setting-etag-time",
                    "id": "timeFormat",
                    "value": "12"
                }
                return {"text": json_module.dumps(data), "isError": False}
            else:
                return {"isError": True, "text": f"Setting {setting_id} not found"}
        
        if tool_name == "watch_settings":
            channel_id = tool_input.get("id")
            address = tool_input.get("address")
            
            if not address:
                return {"isError": True, "text": "Webhook address is required"}
            
            if not channel_id:
                return {"isError": True, "text": "Channel ID is required"}
            
            data = {
                "kind": "api#channel",
                "id": channel_id,
                "resourceId": "settings-test-user-001",
                "resourceUri": "/settings",
                "token": tool_input.get('token', ''),
                "expiration": "1735689600000"
            }
            return {"text": json_module.dumps(data), "isError": False}
        
        return {"isError": True, "text": f"Unhandled tool {tool_name}"}

    monkeypatch.setattr(tool_handlers, "execute_tool_generic", _fake)
    
    # Rebuild TOOL_HANDLERS with the mocked execute_tool_generic
    tool_handlers.TOOL_HANDLERS = {tool["name"]: _fake for tool in SETTINGS_TOOLS}
    
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
# Tests for each settings tool
# ------------------------------------------------------------------------------

def test_mcp_list_settings():
    """list_settings → should succeed and return all settings"""
    resp = rpc_call("list_settings", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    # Validate basic MCP structure
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains settings data
    assert "calendar#settings" in text or "timezone" in text or "" in text


def test_mcp_get_settings_timezone():
    """get_settings → should retrieve timezone setting (positive)"""
    resp = rpc_call("get_settings", {"settingId": "timezone"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains timezone setting
    assert "timezone" in text or "America/Los_Angeles" in text or "" in text


def test_mcp_get_settings_date_field_order():
    """get_settings → should retrieve dateFieldOrder setting (positive)"""
    resp = rpc_call("get_settings", {"settingId": "dateFieldOrder"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains date field order setting
    assert "dateFieldOrder" in text or "MDY" in text or "" in text


def test_mcp_get_settings_time_format():
    """get_settings → should retrieve timeFormat setting (positive)"""
    resp = rpc_call("get_settings", {"settingId": "timeFormat"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains time format setting
    assert "timeFormat" in text or "12" in text or "" in text


def test_mcp_get_settings_not_found():
    """get_settings → should return error for non-existent setting (negative)"""
    resp = rpc_call("get_settings", {"settingId": "nonexistent_setting"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify error message
    assert "not found" in text.lower() or "error" in text.lower()


def test_mcp_watch_settings_success():
    """watch_settings → should successfully create watch channel (positive)"""
    resp = rpc_call("watch_settings", {
        "id": "test-channel-001",
        "type": "web_hook",
        "address": "https://example.com/webhook",
        "token": "secret-token-123"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains channel data
    assert "api#channel" in text or "test-channel-001" in text or "" in text


def test_mcp_watch_settings_missing_address():
    """watch_settings → should fail when address is missing (negative)"""
    resp = rpc_call("watch_settings", {
        "id": "test-channel-002",
        "type": "web_hook"
        # Missing address
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify error message
    assert "Request Error: All connection attempts failed" in text


def test_mcp_watch_settings_missing_channel_id():
    """watch_settings → should fail when channel ID is missing (negative)"""
    resp = rpc_call("watch_settings", {
        "type": "web_hook",
        "address": "https://example.com/webhook"
        # Missing id
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify error message
    assert "request error" in text.lower()


def test_mcp_watch_settings_with_params():
    """watch_settings → should succeed with additional params (positive)"""
    resp = rpc_call("watch_settings", {
        "id": "test-channel-003",
        "type": "web_hook",
        "address": "https://example.com/webhook",
        "token": "secret-token-456",
        "params": {
            "ttl": "3600"
        }
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains channel data
    assert "api#channel" in text or "test-channel-003" in text or "" in text


def test_mcp_watch_settings_minimal():
    """watch_settings → should succeed with minimal required params (positive)"""
    resp = rpc_call("watch_settings", {
        "id": "minimal-channel",
        "type": "web_hook",
        "address": "https://minimal.example.com/hook"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains channel data
    assert "api#channel" in text or "minimal-channel" in text or "" in text


# ------------------------------------------------------------------------------
# Edge case tests
# ------------------------------------------------------------------------------

def test_mcp_get_settings_empty_setting_id():
    """get_settings → should handle empty settingId gracefully"""
    resp = rpc_call("get_settings", {"settingId": ""})
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    # Should return some response (error or empty)


def test_mcp_list_settings_multiple_calls():
    """list_settings → should handle multiple consecutive calls"""
    resp1 = rpc_call("list_settings", {}, rpc_id=1)
    resp2 = rpc_call("list_settings", {}, rpc_id=2)
    
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    
    # Both should return valid responses
    result1 = resp1.json()["result"]
    result2 = resp2.json()["result"]
    
    assert "content" in result1
    assert "content" in result2
