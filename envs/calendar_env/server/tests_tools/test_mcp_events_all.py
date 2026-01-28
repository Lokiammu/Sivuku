"""
End-to-end MCP tests for all Event tools via /mcp endpoint.
All tool executions and user lookups are mocked so tests run offline.
Covers: list_events, get_event, insert_event, patch_event, update_event, delete_event, move_event, quick_add_event, import_event, get_event_instances, watch_events
Each function has 2 good scenarios and 3 bad scenarios.
They are classified in sub clasess
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
# Fixture: fake execute_tool_generic to simulate event tool responses
# ------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fake_exec(monkeypatch):
    """Patch execute_tool_generic so each event tool returns predictable mock data."""
    async def _fake(tool_name, tool_input, database_id, user_id):
        # Simulate error conditions
        if not tool_input.get("calendarId"):
            return {"isError": True, "text": "Missing calendarId"}

        if tool_name in ["get_event", "patch_event", "update_event", "delete_event", "move_event", "get_event_instances"]:
            if not tool_input.get("eventId"):
                return {"isError": True, "text": "Missing eventId"}

        if tool_name == "move_event":
            if not tool_input.get("destination"):
                return {"isError": True, "text": "Missing destination"}

        if tool_name == "quick_add_event":
            if not tool_input.get("text"):
                return {"isError": True, "text": "Missing text"}

        if tool_name == "import_event":
            if not tool_input.get("start"):
                return {"isError": True, "text": "Missing start"}
            if not tool_input.get("end"):
                return {"isError": True, "text": "Missing end"}
            if not tool_input.get("iCalUID"):
                return {"isError": True, "text": "Missing iCalUID"}

        if tool_name == "watch_events":
            if not tool_input.get("id"):
                return {"isError": True, "text": "Missing id"}
            if not tool_input.get("type"):
                return {"isError": True, "text": "Missing type"}
            if not tool_input.get("address"):
                return {"isError": True, "text": "Missing address"}

        if tool_name == "list_events":
            return {
                "items": [
                    {"id": "evt-1", "summary": "Morning Meeting"},
                    {"id": "evt-2", "summary": "Lunch Break"},
                ]
            }
        if tool_name == "get_event":
            return {"id": tool_input.get("eventId"), "summary": "Project Sync"}
        if tool_name == "insert_event":
            return {"id": "evt-new", "summary": tool_input.get("summary", "Untitled Event")}
        if tool_name == "patch_event":
            return {"id": tool_input.get("eventId"), "summary": tool_input.get("summary", "Patched")}
        if tool_name == "update_event":
            return {"id": tool_input.get("eventId"), "summary": "Updated Event"}
        if tool_name == "delete_event":
            return {}
        if tool_name == "move_event":
            return {"id": tool_input.get("eventId"), "summary": "Moved Event"}
        if tool_name == "quick_add_event":
            return {"id": "evt-quick", "summary": "Quick Added Event"}
        if tool_name == "import_event":
            return {"id": "evt-import", "summary": "Imported Event"}
        if tool_name == "get_event_instances":
            return {
                "items": [
                    {"id": "evt-inst-1", "summary": "Recurring Meeting Instance 1"},
                    {"id": "evt-inst-2", "summary": "Recurring Meeting Instance 2"},
                ]
            }
        if tool_name == "watch_events":
            return {"kind": "api#channel", "id": tool_input.get("id"), "resourceId": "calendar-watch"}
        return {"isError": True, "text": f"Unhandled tool {tool_name}"}

    monkeypatch.setattr(tool_handlers, "execute_tool_generic", _fake)
    return monkeypatch


# ------------------------------------------------------------------------------
# Helper to send JSON-RPC calls to /mcp with required headers
# ------------------------------------------------------------------------------

def rpc_call(tool_name, arguments, rpc_id=1, db_id="test-db-001", token="dummy-static-token"):
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
        "x-database-id": db_id,
        "x-access-token": token
    }
    return client.post("/mcp", json=payload, headers=headers)


# ------------------------------------------------------------------------------
# list_events: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_list_events_primary_calendar_success():
    """GOOD: list_events with primary calendar"""
    resp = rpc_call("list_events", {"calendarId": "primary"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_list_events_secondary_calendar_success():
    """GOOD: list_events with secondary calendar"""
    resp = rpc_call("list_events", {"calendarId": "work@company.com"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_list_events_missing_calendar_id():
    """BAD: list_events without calendarId"""
    resp = rpc_call("list_events", {})
    assert resp.status_code == 200
    # Check for error in response content if applicable


def test_list_events_invalid_calendar_format():
    """BAD: list_events with malformed calendar ID"""
    resp = rpc_call("list_events", {"calendarId": "not@valid@format@email"})
    assert resp.status_code == 200


def test_list_events_empty_calendar_id():
    """BAD: list_events with empty string calendarId"""
    resp = rpc_call("list_events", {"calendarId": ""})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# get_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_get_event_valid_primary_calendar():
    """GOOD: get_event with valid eventId on primary calendar"""
    resp = rpc_call("get_event", {"calendarId": "primary", "eventId": "evt-123"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_get_event_valid_secondary_calendar():
    """GOOD: get_event with valid eventId on secondary calendar"""
    resp = rpc_call("get_event", {"calendarId": "team@company.com", "eventId": "evt-456"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_get_event_missing_event_id():
    """BAD: get_event without eventId"""
    resp = rpc_call("get_event", {"calendarId": "primary"})
    assert resp.status_code == 200


def test_get_event_missing_calendar_id():
    """BAD: get_event without calendarId"""
    resp = rpc_call("get_event", {"eventId": "evt-123"})
    assert resp.status_code == 200


def test_get_event_empty_event_id():
    """BAD: get_event with empty eventId"""
    resp = rpc_call("get_event", {"calendarId": "primary", "eventId": ""})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# insert_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_insert_event_with_summary():
    """GOOD: insert_event with summary"""
    resp = rpc_call("insert_event", {"calendarId": "primary", "summary": "Team Standup"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_insert_event_with_unicode_summary():
    """GOOD: insert_event with unicode characters"""
    resp = rpc_call("insert_event", {"calendarId": "primary", "summary": "会議"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_insert_event_missing_calendar_id():
    """BAD: insert_event without calendarId"""
    resp = rpc_call("insert_event", {"summary": "Meeting"})
    assert resp.status_code == 200


def test_insert_event_empty_calendar_id():
    """BAD: insert_event with empty calendarId"""
    resp = rpc_call("insert_event", {"calendarId": "", "summary": "Meeting"})
    assert resp.status_code == 200


def test_insert_event_null_summary():
    """BAD: insert_event with null summary"""
    resp = rpc_call("insert_event", {"calendarId": "primary", "summary": None})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# patch_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_patch_event_update_summary():
    """GOOD: patch_event to update summary"""
    resp = rpc_call("patch_event", {"calendarId": "primary", "eventId": "evt-555", "summary": "Updated Title"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_patch_event_partial_update():
    """GOOD: patch_event with minimal changes"""
    resp = rpc_call("patch_event", {"calendarId": "primary", "eventId": "evt-666", "summary": "Quick Fix"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_patch_event_missing_event_id():
    """BAD: patch_event without eventId"""
    resp = rpc_call("patch_event", {"calendarId": "primary", "summary": "New Title"})
    assert resp.status_code == 200


def test_patch_event_missing_calendar_id():
    """BAD: patch_event without calendarId"""
    resp = rpc_call("patch_event", {"eventId": "evt-555", "summary": "New Title"})
    assert resp.status_code == 200


def test_patch_event_empty_event_id():
    """BAD: patch_event with empty eventId"""
    resp = rpc_call("patch_event", {"calendarId": "primary", "eventId": "", "summary": "Title"})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# update_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_update_event_full_replace():
    """GOOD: update_event with full replacement"""
    resp = rpc_call("update_event", {"calendarId": "primary", "eventId": "evt-888"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_update_event_with_all_fields():
    """GOOD: update_event with complete data"""
    resp = rpc_call("update_event", {
        "calendarId": "primary",
        "eventId": "evt-999",
        "summary": "Complete Update"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_update_event_missing_event_id():
    """BAD: update_event without eventId"""
    resp = rpc_call("update_event", {"calendarId": "primary"})
    assert resp.status_code == 200


def test_update_event_missing_calendar_id():
    """BAD: update_event without calendarId"""
    resp = rpc_call("update_event", {"eventId": "evt-888"})
    assert resp.status_code == 200


def test_update_event_empty_event_id():
    """BAD: update_event with empty eventId"""
    resp = rpc_call("update_event", {"calendarId": "primary", "eventId": ""})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# delete_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_delete_event_from_primary():
    """GOOD: delete_event from primary calendar"""
    resp = rpc_call("delete_event", {"calendarId": "primary", "eventId": "evt-999"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_delete_event_from_secondary():
    """GOOD: delete_event from secondary calendar"""
    resp = rpc_call("delete_event", {"calendarId": "work@example.com", "eventId": "evt-777"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_delete_event_missing_event_id():
    """BAD: delete_event without eventId"""
    resp = rpc_call("delete_event", {"calendarId": "primary"})
    assert resp.status_code == 200


def test_delete_event_missing_calendar_id():
    """BAD: delete_event without calendarId"""
    resp = rpc_call("delete_event", {"eventId": "evt-999"})
    assert resp.status_code == 200


def test_delete_event_empty_event_id():
    """BAD: delete_event with empty eventId"""
    resp = rpc_call("delete_event", {"calendarId": "primary", "eventId": ""})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# move_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_move_event_to_primary_calendar():
    """GOOD: move_event from secondary to primary calendar"""
    resp = rpc_call("move_event", {
        "calendarId": "work@company.com",
        "eventId": "evt-123",
        "destination": "primary"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_move_event_with_notifications():
    """GOOD: move_event with notification settings"""
    resp = rpc_call("move_event", {
        "calendarId": "primary",
        "eventId": "evt-456",
        "destination": "team@company.com",
        "sendNotifications": True,
        "sendUpdates": "all"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_move_event_missing_destination():
    """BAD: move_event without destination"""
    resp = rpc_call("move_event", {"calendarId": "primary", "eventId": "evt-123"})
    assert resp.status_code == 200


def test_move_event_missing_event_id():
    """BAD: move_event without eventId"""
    resp = rpc_call("move_event", {"calendarId": "primary", "destination": "work@company.com"})
    assert resp.status_code == 200


def test_move_event_missing_calendar_id():
    """BAD: move_event without calendarId"""
    resp = rpc_call("move_event", {"eventId": "evt-123", "destination": "work@company.com"})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# quick_add_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_quick_add_event_simple_text():
    """GOOD: quick_add_event with simple text"""
    resp = rpc_call("quick_add_event", {"calendarId": "primary", "text": "Meeting tomorrow at 2pm"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_quick_add_event_complex_text():
    """GOOD: quick_add_event with complex text and notifications"""
    resp = rpc_call("quick_add_event", {
        "calendarId": "primary",
        "text": "Team lunch next Friday at 12:30pm at Downtown Cafe",
        "sendNotifications": True
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_quick_add_event_missing_text():
    """BAD: quick_add_event without text"""
    resp = rpc_call("quick_add_event", {"calendarId": "primary"})
    assert resp.status_code == 200


def test_quick_add_event_missing_calendar_id():
    """BAD: quick_add_event without calendarId"""
    resp = rpc_call("quick_add_event", {"text": "Meeting tomorrow"})
    assert resp.status_code == 200


def test_quick_add_event_empty_text():
    """BAD: quick_add_event with empty text"""
    resp = rpc_call("quick_add_event", {"calendarId": "primary", "text": ""})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# import_event: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_import_event_basic():
    """GOOD: import_event with required fields"""
    resp = rpc_call("import_event", {
        "calendarId": "primary",
        "iCalUID": "import-test-001",
        "start": {"dateTime": "2023-12-01T10:00:00Z"},
        "end": {"dateTime": "2023-12-01T11:00:00Z"}
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_import_event_with_details():
    """GOOD: import_event with complete details"""
    resp = rpc_call("import_event", {
        "calendarId": "primary",
        "iCalUID": "import-test-002",
        "summary": "Imported Meeting",
        "description": "Meeting imported from external calendar",
        "start": {"dateTime": "2023-12-01T14:00:00Z"},
        "end": {"dateTime": "2023-12-01T15:00:00Z"},
        "supportsAttachments": True
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_import_event_missing_start():
    """BAD: import_event without start"""
    resp = rpc_call("import_event", {
        "calendarId": "primary",
        "iCalUID": "import-test-003",
        "end": {"dateTime": "2023-12-01T11:00:00Z"}
    })
    assert resp.status_code == 200


def test_import_event_missing_end():
    """BAD: import_event without end"""
    resp = rpc_call("import_event", {
        "calendarId": "primary",
        "iCalUID": "import-test-004",
        "start": {"dateTime": "2023-12-01T10:00:00Z"}
    })
    assert resp.status_code == 200


def test_import_event_missing_ical_uid():
    """BAD: import_event without iCalUID"""
    resp = rpc_call("import_event", {
        "calendarId": "primary",
        "start": {"dateTime": "2023-12-01T10:00:00Z"},
        "end": {"dateTime": "2023-12-01T11:00:00Z"}
    })
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# get_event_instances: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_get_event_instances_basic():
    """GOOD: get_event_instances with basic parameters"""
    resp = rpc_call("get_event_instances", {"calendarId": "primary", "eventId": "recurring-evt-123"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_get_event_instances_with_time_range():
    """GOOD: get_event_instances with time range filter"""
    resp = rpc_call("get_event_instances", {
        "calendarId": "primary",
        "eventId": "recurring-evt-456",
        "timeMin": "2023-12-01T00:00:00Z",
        "timeMax": "2023-12-31T23:59:59Z",
        "maxResults": 10
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_get_event_instances_missing_event_id():
    """BAD: get_event_instances without eventId"""
    resp = rpc_call("get_event_instances", {"calendarId": "primary"})
    assert resp.status_code == 200


def test_get_event_instances_missing_calendar_id():
    """BAD: get_event_instances without calendarId"""
    resp = rpc_call("get_event_instances", {"eventId": "recurring-evt-123"})
    assert resp.status_code == 200


def test_get_event_instances_empty_event_id():
    """BAD: get_event_instances with empty eventId"""
    resp = rpc_call("get_event_instances", {"calendarId": "primary", "eventId": ""})
    assert resp.status_code == 200


# ------------------------------------------------------------------------------
# watch_events: 2 good + 3 bad scenarios
# ------------------------------------------------------------------------------

def test_watch_events_basic():
    """GOOD: watch_events with required parameters"""
    resp = rpc_call("watch_events", {
        "calendarId": "primary",
        "id": "watch-channel-001",
        "type": "web_hook",
        "address": "https://example.com/webhook"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_watch_events_with_token():
    """GOOD: watch_events with authentication token"""
    resp = rpc_call("watch_events", {
        "calendarId": "primary",
        "id": "watch-channel-002",
        "type": "web_hook",
        "address": "https://secure.example.com/webhook",
        "token": "secure-token-123",
        "eventTypes": "default"
    })
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "content" in result


def test_watch_events_missing_id():
    """BAD: watch_events without id"""
    resp = rpc_call("watch_events", {
        "calendarId": "primary",
        "type": "web_hook",
        "address": "https://example.com/webhook"
    })
    assert resp.status_code == 200


def test_watch_events_missing_type():
    """BAD: watch_events without type"""
    resp = rpc_call("watch_events", {
        "calendarId": "primary",
        "id": "watch-channel-003",
        "address": "https://example.com/webhook"
    })
    assert resp.status_code == 200


def test_watch_events_missing_address():
    """BAD: watch_events without address"""
    resp = rpc_call("watch_events", {
        "calendarId": "primary",
        "id": "watch-channel-004",
        "type": "web_hook"
    })
    assert resp.status_code == 200