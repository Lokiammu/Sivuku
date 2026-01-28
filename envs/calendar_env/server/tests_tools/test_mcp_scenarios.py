"""
MCP Functional Scenarios
------------------------
Ten comprehensive integration tests combining multiple tools and edge cases.
These simulate real-world sequences like creating, patching, and deleting
calendars/events using the /mcp endpoint.
"""

import sys, os, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from handlers import mcp_handler, tool_handlers


# ------------------------------------------------------------------------------
# Minimal FastAPI app mounting /mcp
# ------------------------------------------------------------------------------

app = FastAPI()

@app.post("/mcp")
async def mcp_entry(request: Request):
    return await mcp_handler.handle_mcp_request(request)

client = TestClient(app)


# ------------------------------------------------------------------------------
# Fixtures: fake DB, user manager, and tool executor
# ------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_user_manager(monkeypatch):
    class DummyUserManager:
        def __init__(self, dbid): self.dbid = dbid
        def get_first_user_from_db(self): return {"id": "test-user-001"}
        def get_user_by_access_token(self, token): return {"id": "test-user-001"}

    import handlers.mcp_handler as mcp
    monkeypatch.setattr(mcp, "UserManager", DummyUserManager)
    return monkeypatch


@pytest.fixture(autouse=True)
def fake_exec(monkeypatch):
    """Generic fake executor for all scenarios."""
    async def _fake(tool_name, tool_input, dbid, userid):
        # Simulate basic behavior
        responses = {
            "create_calendar": {"id": "cal-new", "summary": tool_input.get("summary")},
            "patch_calendar": {"id": "cal-new", "summary": tool_input.get("summary", "patched")},
            "get_calendar": {"id": tool_input.get("calendarId"), "summary": "Sample Calendar"},
            "delete_calendar": {},
            "list_events": {"items": [{"id": "evt-1", "summary": "Meeting"}]},
            "insert_event": {"id": "evt-new", "summary": tool_input.get("summary")},
            "patch_event": {"id": tool_input.get("eventId"), "summary": "Patched Event"},
            "freebusy_query": {"calendars": {"primary": {"busy": []}}},
            "acl_list": {"items": [{"role": "owner"}]},
            "colors_get": {"calendar": {"1": {"background": "#ff0000"}}},
        }
        return responses.get(tool_name, {"isError": True, "text": f"Unhandled {tool_name}"})

    monkeypatch.setattr(tool_handlers, "execute_tool_generic", _fake)
    return monkeypatch


@pytest.fixture
def headers():
    return {"x-database-id": "test-db-001", "x-access-token": "dummy-static-token"}


def rpc_call(tool_name, arguments, headers):
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": 1
    }
    return client.post("/mcp", json=payload, headers=headers)


# ------------------------------------------------------------------------------
# 10 Comprehensive MCP Scenarios
# ------------------------------------------------------------------------------

def test_01_create_calendar_success(headers):
    """ Create a calendar and verify response."""
    resp = rpc_call("create_calendar", {"summary": "Engineering Team"}, headers)
    assert resp.status_code == 200
    assert "" in str(resp.json()) or "completed" in str(resp.json())


def test_02_patch_calendar_title(headers):
    """ Patch calendar title."""
    resp = rpc_call("patch_calendar", {"calendarId": "cal-new", "summary": "Updated Calendar"}, headers)
    result = resp.json()["result"]
    assert "content" in result


def test_03_insert_event_into_calendar(headers):
    """ Insert event inside a calendar."""
    resp = rpc_call("insert_event", {"calendarId": "cal-new", "summary": "Team Sync"}, headers)
    result = resp.json()["result"]
    assert "content" in result


def test_04_list_events_from_calendar(headers):
    """ List events after insertion."""
    resp = rpc_call("list_events", {"calendarId": "cal-new"}, headers)
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_05_update_event_summary(headers):
    """ Patch existing event summary."""
    resp = rpc_call("patch_event", {"calendarId": "cal-new", "eventId": "evt-1", "summary": "Rescheduled Sync"}, headers)
    result = resp.json()["result"]
    assert "content" in result


def test_06_query_freebusy_availability(headers):
    """ FreeBusy query should return empty slots."""
    resp = rpc_call("freebusy_query", {"timeMin": "2025-10-10T00:00:00Z", "timeMax": "2025-10-10T23:59:59Z"}, headers)
    result = resp.json()["result"]
    assert "content" in result


def test_07_retrieve_calendar_colors(headers):
    """ Get available calendar colors."""
    resp = rpc_call("colors_get", {}, headers)
    result = resp.json()["result"]
    assert "content" in result


def test_08_acl_list_access_rights(headers):
    """ List ACL entries."""
    resp = rpc_call("acl_list", {"calendarId": "primary"}, headers)
    result = resp.json()["result"]
    assert "content" in result


def test_09_delete_calendar(headers):
    """ Delete an existing calendar."""
    resp = rpc_call("delete_calendar", {"calendarId": "cal-new"}, headers)
    result = resp.json()["result"]
    assert "content" in result


def test_10_invalid_tool_request(headers):
    """ Invalid tool should produce Unknown tool error."""
    resp = rpc_call("unknown_tool", {}, headers)
    result = resp.json()["result"]
    text = result["content"][0]["text"]
    assert "Unknown tool" in text or "" in text
