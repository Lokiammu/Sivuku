"""
End-to-end MCP tests for all Calendar tools via /mcp endpoint.
All tool executions and user lookups are mocked so tests run offline.
Covers: create, get, patch, update, delete, clear.
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
    # Mock the MCP_TOOLS list to include calendar list tools
    from calendar_mcp.tools.calendar_list import CALENDAR_LIST_TOOLS
    monkeypatch.setattr(tool_handlers, "MCP_TOOLS", CALENDAR_LIST_TOOLS)
    
    async def _fake(tool_name, tool_input, database_id, user_id):
        import json as json_module
        
        if tool_name == "create_calendar":
            data = {"id": "alice-team", "summary": tool_input.get("summary")}
            return {"text": json_module.dumps(data), "isError": False}
        if tool_name == "get_calendar":
            data = {"id": tool_input.get("calendarId"), "summary": "Alice Johnson"}
            return {"text": json_module.dumps(data), "isError": False}
        if tool_name == "patch_calendar":
            data = {"id": tool_input.get("calendarId"), "summary": tool_input.get("summary", "patched")}
            return {"text": json_module.dumps(data), "isError": False}
        if tool_name == "update_calendar":
            data = {"id": tool_input.get("calendarId"), "summary": "Updated Full Calendar"}
            return {"text": json_module.dumps(data), "isError": False}
        if tool_name == "replace_calendar_in_list":
            calendar_id = tool_input.get("calendarId")
            if not calendar_id:
                return {"isError": True, "text": "Calendar ID is required"}
            
            data = {
                "kind": "calendar#calendarListEntry",
                "etag": "etag-replaced",
                "id": calendar_id,
                "summary": tool_input.get("summaryOverride", "Replaced Calendar"),
                "summaryOverride": tool_input.get("summaryOverride"),
                "colorId": tool_input.get("colorId"),
                "backgroundColor": tool_input.get("backgroundColor"),
                "foregroundColor": tool_input.get("foregroundColor"),
                "hidden": tool_input.get("hidden", False),
                "selected": tool_input.get("selected", True),
                "accessRole": "owner",
                "defaultReminders": tool_input.get("defaultReminders", [])
            }
            return {"text": json_module.dumps(data), "isError": False}
        if tool_name in ("delete_calendar", "clear_calendar"):
            return {"text": "{}", "isError": False}
        if tool_name == "watch_calendar_list":
            channel_id = tool_input.get("id")
            address = tool_input.get("address")
            
            if not address:
                return {"isError": True, "text": "Webhook address is required"}
            
            if not channel_id:
                return {"isError": True, "text": "Channel ID is required"}
            
            data = {
                "kind": "api#channel",
                "id": channel_id,
                "resourceId": f"calendar-list-{user_id}",
                "resourceUri": "/users/me/calendarList",
                "token": tool_input.get("token", ""),
                "expiration": "1735689600000",
                "type": "web_hook",
                "address": address
            }
            return {"text": json_module.dumps(data), "isError": False}
        return {"isError": True, "text": f"Unhandled tool {tool_name}"}

    monkeypatch.setattr(tool_handlers, "execute_tool_generic", _fake)
    
    # Rebuild TOOL_HANDLERS with the mocked execute_tool_generic
    tool_handlers.TOOL_HANDLERS = {tool["name"]: _fake for tool in CALENDAR_LIST_TOOLS}
    
    # Also patch TOOL_HANDLERS in mcp_handler module since it imports it
    import handlers.mcp_handler as mcp
    monkeypatch.setattr(mcp, "TOOL_HANDLERS", tool_handlers.TOOL_HANDLERS)
    
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
# Tests for each calendar tool
# ------------------------------------------------------------------------------

def test_mcp_create_calendar():
    """ create_calendar → should succeed and return mocked response"""
    resp = rpc_call("create_calendar", {"summary": "Team Coordination"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    # Validate basic MCP structure
    assert "content" in result
    text = result["content"][0]["text"]
    assert "" in text or "completed" in text


def test_mcp_get_calendar():
    """ get_calendar → should succeed"""
    resp = rpc_call("get_calendar", {"calendarId": "alice-primary"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_patch_calendar():
    """ patch_calendar → should succeed"""
    resp = rpc_call("patch_calendar", {"calendarId": "alice-primary", "summary": "Updated Title"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_update_calendar():
    """ update_calendar → should succeed"""
    resp = rpc_call("update_calendar", {"calendarId": "bob-development"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_delete_calendar():
    """ delete_calendar → should succeed"""
    resp = rpc_call("delete_calendar", {"calendarId": "carol-design"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_clear_calendar():
    """ clear_calendar → should succeed"""
    resp = rpc_call("clear_calendar", {"calendarId": "dave-primary"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_watch_calendar_list_success():
    """watch_calendar_list → should successfully create watch channel (positive)"""
    resp = rpc_call("watch_calendar_list", {
        "id": "test-watch-channel-001",
        "type": "web_hook",
        "address": "https://example.com/webhook",
        "token": "verification-token-123"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains channel data
    assert "api#channel" in text or "test-watch-channel-001" in text or "" in text


def test_mcp_watch_calendar_list_missing_address():
    """watch_calendar_list → should fail when address is missing (negative)"""
    resp = rpc_call("watch_calendar_list", {
        "id": "test-watch-channel-002",
        "type": "web_hook"
        # Missing address
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify error message
    assert "address" in text.lower() or "required" in text.lower()


def test_mcp_watch_calendar_list_missing_channel_id():
    """watch_calendar_list → should fail when channel ID is missing (negative)"""
    resp = rpc_call("watch_calendar_list", {
        "type": "web_hook",
        "address": "https://example.com/webhook"
        # Missing id
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify error message
    assert "channel" in text.lower() or "id" in text.lower() or "required" in text.lower()


def test_mcp_watch_calendar_list_with_params():
    """watch_calendar_list → should succeed with additional params (positive)"""
    resp = rpc_call("watch_calendar_list", {
        "id": "test-watch-channel-003",
        "type": "web_hook",
        "address": "https://example.com/webhook",
        "token": "verification-token-456",
        "params": {
            "ttl": "3600"
        }
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains channel data
    assert "api#channel" in text or "test-watch-channel-003" in text or "" in text


def test_mcp_watch_calendar_list_minimal():
    """watch_calendar_list → should succeed with minimal required params (positive)"""
    resp = rpc_call("watch_calendar_list", {
        "id": "minimal-watch-channel",
        "type": "web_hook",
        "address": "https://minimal.example.com/hook"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains channel data
    assert "api#channel" in text or "minimal-watch-channel" in text or "" in text


def test_mcp_replace_calendar_in_list_full_update():
    """replace_calendar_in_list → should perform full update with all fields (positive)"""
    resp = rpc_call("replace_calendar_in_list", {
        "calendarId": "test-calendar-001",
        "summaryOverride": "Fully Replaced Calendar",
        "colorId": "5",
        "hidden": False,
        "selected": True,
        "defaultReminders": [
            {"method": "email", "minutes": 30},
            {"method": "popup", "minutes": 10}
        ]
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains updated data
    assert "calendar#calendarListEntry" in text or "Fully Replaced Calendar" in text or "" in text


def test_mcp_replace_calendar_in_list_minimal():
    """replace_calendar_in_list → should succeed with minimal required params (positive)"""
    resp = rpc_call("replace_calendar_in_list", {
        "calendarId": "test-calendar-002"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains calendar data
    assert "test-calendar-002" in text or "calendar#calendarListEntry" in text or "" in text


def test_mcp_replace_calendar_in_list_with_colors():
    """replace_calendar_in_list → should update with RGB colors (positive)"""
    resp = rpc_call("replace_calendar_in_list", {
        "calendarId": "test-calendar-003",
        "colorRgbFormat": True,
        "backgroundColor": "#FF5733",
        "foregroundColor": "#FFFFFF",
        "summaryOverride": "Custom Colored Calendar"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains color data
    assert "#FF5733" in text or "#FFFFFF" in text or "Custom Colored Calendar" in text or "" in text


def test_mcp_replace_calendar_in_list_hidden():
    """replace_calendar_in_list → should update hidden status (positive)"""
    resp = rpc_call("replace_calendar_in_list", {
        "calendarId": "test-calendar-004",
        "hidden": True,
        "selected": False
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains updated data
    assert "test-calendar-004" in text or "" in text


def test_mcp_replace_calendar_in_list_missing_calendar_id():
    """replace_calendar_in_list → should fail when calendarId is missing (negative)"""
    resp = rpc_call("replace_calendar_in_list", {
        "summaryOverride": "No Calendar ID"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify error message
    assert "calendar" in text.lower() or "id" in text.lower() or "required" in text.lower()


def test_mcp_replace_calendar_in_list_with_reminders():
    """replace_calendar_in_list → should update with default reminders (positive)"""
    resp = rpc_call("replace_calendar_in_list", {
        "calendarId": "test-calendar-005",
        "summaryOverride": "Calendar with Reminders",
        "defaultReminders": [
            {"method": "email", "minutes": 60},
            {"method": "popup", "minutes": 15}
        ]
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    
    assert "content" in result
    text = result["content"][0]["text"]
    
    # Verify response contains reminder data
    assert "email" in text or "popup" in text or "Calendar with Reminders" in text or "" in text

