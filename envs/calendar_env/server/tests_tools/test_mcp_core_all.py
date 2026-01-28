"""
End-to-end MCP tests for core Calendar tools via /mcp endpoint.
Mocks out database, user lookups, and tool execution.
Covers: freebusy.query, colors.get, settings.list, acl.list, acl.get, acl.insert, acl.update, acl.patch, acl.delete, acl.watch
"""

import sys, os, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from handlers import mcp_handler, tool_handlers


# ------------------------------------------------------------------------------
# Minimal FastAPI app for /mcp
# ------------------------------------------------------------------------------

app = FastAPI()

@app.post("/mcp")
async def mcp_entry(request: Request):
    """Forward JSON-RPC requests to the real MCP handler."""
    return await mcp_handler.handle_mcp_request(request)

client = TestClient(app)


# ------------------------------------------------------------------------------
# Fixture: fake UserManager (skip DB access)
# ------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_user_manager(monkeypatch):
    """Bypass DB in UserManager for tests."""
    class DummyUserManager:
        def __init__(self, dbid): self.dbid = dbid
        def get_first_user_from_db(self): return {"id": "test-user-001"}
        def get_user_by_access_token(self, token): return {"id": "test-user-001"}

    import handlers.mcp_handler as mcp
    monkeypatch.setattr(mcp, "UserManager", DummyUserManager)
    return monkeypatch


# ------------------------------------------------------------------------------
# Fixture: fake execute_tool_generic for core tool behaviors
# ------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_exec(monkeypatch):
    """Simulate behavior of core Calendar tools."""
    async def _fake(tool_name, tool_input, database_id, user_id):
        # Simulate error conditions
        if tool_name in ["get_acl_rule", "update_acl_rule", "patch_acl_rule"]:
            if not tool_input.get("calendarId"):
                return {"isError": True, "text": "Missing calendarId"}
            if not tool_input.get("ruleId"):
                return {"isError": True, "text": "Missing ruleId"}

        if tool_name == "update_acl_rule":
            if not tool_input.get("scope"):
                return {"isError": True, "text": "Missing scope"}

        if tool_name == "watch_acl":
            if not tool_input.get("calendarId"):
                return {"isError": True, "text": "Missing calendarId"}
            if not tool_input.get("id"):
                return {"isError": True, "text": "Missing id"}
            if not tool_input.get("type"):
                return {"isError": True, "text": "Missing type"}
            if not tool_input.get("address"):
                return {"isError": True, "text": "Missing address"}

        if tool_name == "freebusy_query":
            return {"calendars": {"primary": {"busy": [{"start": "2025-10-10T09:00:00Z", "end": "2025-10-10T10:00:00Z"}]}}}
        if tool_name == "colors_get":
            return {"calendar": {"1": {"background": "#ff0000"}}, "event": {"2": {"background": "#00ff00"}}}
        if tool_name == "settings_list":
            return {"items": [{"id": "timezone", "value": "America/New_York"}]}
        if tool_name == "acl_list":
            return {"items": [{"role": "owner", "scope": {"type": "user", "value": "owner@example.com"}}]}
        if tool_name == "acl_insert":
            return {"id": "rule-123", "role": "reader"}
        if tool_name == "acl_delete":
            return {}
        if tool_name == "get_acl_rule":
            return {"id": tool_input.get("ruleId"), "role": "reader", "scope": {"type": "user", "value": "user@example.com"}}
        if tool_name == "update_acl_rule":
            return {"id": tool_input.get("ruleId"), "role": tool_input.get("role", "reader"), "scope": tool_input.get("scope")}
        if tool_name == "patch_acl_rule":
            return {"id": tool_input.get("ruleId"), "role": tool_input.get("role", "reader"), "scope": {"type": "user", "value": "updated@example.com"}}
        if tool_name == "watch_acl":
            return {"kind": "api#channel", "id": tool_input.get("id"), "resourceId": "acl-watch"}
        return {"isError": True, "text": f"Unhandled tool {tool_name}"}

    monkeypatch.setattr(tool_handlers, "execute_tool_generic", _fake)
    return monkeypatch


# ------------------------------------------------------------------------------
# Helper to make JSON-RPC calls to /mcp
# ------------------------------------------------------------------------------

def rpc_call(tool_name, arguments, rpc_id=1):
    """Send JSON-RPC call to /mcp with required headers."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": rpc_id,
    }
    headers = {"x-database-id": "test-db-001", "x-access-token": "dummy-static-token"}
    return client.post("/mcp", json=payload, headers=headers)


# ------------------------------------------------------------------------------
# Tests: FreeBusy
# ------------------------------------------------------------------------------

def test_mcp_freebusy_query():
    """ freebusy.query → should return mock busy slots."""
    resp = rpc_call("freebusy_query", {"timeMin": "2025-10-10T00:00:00Z", "timeMax": "2025-10-10T23:59:59Z"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


# ------------------------------------------------------------------------------
# Tests: Colors
# ------------------------------------------------------------------------------

def test_mcp_colors_get():
    """ colors.get → should return calendar and event colors."""
    resp = rpc_call("colors_get", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


# ------------------------------------------------------------------------------
# Tests: Settings
# ------------------------------------------------------------------------------

def test_mcp_settings_list():
    """ settings.list → should return user settings."""
    resp = rpc_call("settings_list", {})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


# ------------------------------------------------------------------------------
# Tests: ACL (Access Control List)
# ------------------------------------------------------------------------------

def test_mcp_acl_list():
    """ acl.list → should return mock ACL entries."""
    resp = rpc_call("acl_list", {"calendarId": "primary"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_acl_insert():
    """ acl.insert → should insert ACL rule."""
    resp = rpc_call("acl_insert", {"calendarId": "primary", "role": "reader"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_acl_delete():
    """ acl.delete → should delete ACL rule."""
    resp = rpc_call("acl_delete", {"calendarId": "primary", "ruleId": "rule-123"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


# ------------------------------------------------------------------------------
# Tests: ACL Rule Management
# ------------------------------------------------------------------------------

def test_mcp_get_acl_rule_success():
    """GOOD: get_acl_rule with valid parameters"""
    resp = rpc_call("get_acl_rule", {"calendarId": "primary", "ruleId": "rule-456"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_get_acl_rule_missing_rule_id():
    """BAD: get_acl_rule without ruleId"""
    resp = rpc_call("get_acl_rule", {"calendarId": "primary"})
    assert resp.status_code == 200


def test_mcp_update_acl_rule_success():
    """GOOD: update_acl_rule with complete parameters"""
    resp = rpc_call("update_acl_rule", {
        "calendarId": "primary",
        "ruleId": "rule-789",
        "scope": {"type": "user", "value": "newuser@example.com"},
        "role": "writer",
        "sendNotifications": False
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_update_acl_rule_missing_scope():
    """BAD: update_acl_rule without scope"""
    resp = rpc_call("update_acl_rule", {"calendarId": "primary", "ruleId": "rule-789"})
    assert resp.status_code == 200


def test_mcp_patch_acl_rule_success():
    """GOOD: patch_acl_rule with role update"""
    resp = rpc_call("patch_acl_rule", {
        "calendarId": "primary",
        "ruleId": "rule-101",
        "role": "owner"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_patch_acl_rule_missing_calendar_id():
    """BAD: patch_acl_rule without calendarId"""
    resp = rpc_call("patch_acl_rule", {"ruleId": "rule-101", "role": "owner"})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# Tests: ACL Watch
# ------------------------------------------------------------------------------

def test_mcp_watch_acl_success():
    """GOOD: watch_acl with all required parameters"""
    resp = rpc_call("watch_acl", {
        "calendarId": "primary",
        "id": "acl-watch-001",
        "type": "web_hook",
        "address": "https://example.com/acl-webhook"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_mcp_watch_acl_missing_address():
    """BAD: watch_acl without address"""
    resp = rpc_call("watch_acl", {
        "calendarId": "primary",
        "id": "acl-watch-002",
        "type": "web_hook"
    })
    assert resp.status_code == 200
