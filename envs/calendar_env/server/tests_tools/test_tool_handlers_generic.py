"""
Tests for execute_tool_generic() from handlers.tool_handlers.
These mock httpx and router inspection to validate behavior without live APIs.
"""

import pytest
import json as json_module
import httpx
from handlers import tool_handlers


# --------------------------------------------------------------------------
# Common async helper
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def call_execute(tool_name="create_calendar", arguments=None, status=200, json_data=None, method="POST"):
    """Helper to invoke execute_tool_generic() with mocked httpx."""
    arguments = arguments or {}

    # async-compatible request mock
    async def mock_request(self, m, url, headers=None, json=None, params=None):
        class MockResponse:
            def __init__(self):
                self.status_code = status
                self._json = json_data
                self.text = json_module.dumps(json_data) if json_data else ""
            def json(self): return self._json
        return MockResponse()

    # Patch network + endpoint discovery
    tool_handlers.httpx.AsyncClient.request = mock_request
    tool_handlers.get_api_endpoint_for_tool = lambda name: (method, f"/mock/{name}")
    tool_handlers.log_tool_response = lambda *a, **kw: None

    result = await tool_handlers.execute_tool_generic(
        tool_name, arguments, database_id="test-db-001", user_id="test-user-001"
    )
    return result


# --------------------------------------------------------------------------
# 1. Successful POST
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_success_post():
    data = {"id": "123", "summary": "OK"}
    result = await call_execute(json_data=data)
    assert not result["isError"], f"Unexpected error: {result['text']}"
    assert '"id": "123"' in result["text"]


# --------------------------------------------------------------------------
# 2. Successful GET
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_success_get():
    async def mock_get(self, url, headers=None, params=None):
        class R:
            status_code = 200
            def json(self): return {"ping": "pong"}
            text = json_module.dumps({"ping": "pong"})
        return R()

    # Mock tool configuration
    original_tools = getattr(tool_handlers, 'MCP_TOOLS', [])
    tool_handlers.MCP_TOOLS = [{"name": "get_calendar_list", "description": "test"}]
    
    try:
        tool_handlers.httpx.AsyncClient.get = mock_get
        tool_handlers.get_api_endpoint_for_tool = lambda name: ("GET", "/mock/path")
        tool_handlers.log_tool_response = lambda *a, **kw: None

        result = await tool_handlers.execute_tool_generic("get_calendar_list", {}, "test-db-001", "u1")
        assert not result["isError"], f"Unexpected error: {result['text']}"
        assert "pong" in result["text"]
    finally:
        # Restore original tools
        tool_handlers.MCP_TOOLS = original_tools


# --------------------------------------------------------------------------
# 3. 204 (DELETE)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_delete_204():
    async def mock_del(self, url, headers=None, params=None):
        class R: status_code = 204; text = ""
        return R()
    tool_handlers.httpx.AsyncClient.delete = mock_del
    tool_handlers.get_api_endpoint_for_tool = lambda name: ("DELETE", "/mock/path")
    tool_handlers.log_tool_response = lambda *a, **kw: None

    result = await tool_handlers.execute_tool_generic("delete_calendar", {}, "test-db-001", "u1")
    assert result["status_code"] == 204


# --------------------------------------------------------------------------
# 4. 404 friendly error
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_404_friendly_error():
    async def mock_get(self, url, headers=None, params=None):
        class R:
            status_code = 404
            def json(self): return {"detail": "Calendar not found"}
            text = ""
        return R()

    # Mock tool configuration
    original_tools = getattr(tool_handlers, 'MCP_TOOLS', [])
    tool_handlers.MCP_TOOLS = [{"name": "get_calendar", "description": "test"}]
    
    try:
        tool_handlers.httpx.AsyncClient.get = mock_get
        tool_handlers.get_api_endpoint_for_tool = lambda name: ("GET", "/calendars/{calendarId}")
        tool_handlers.log_tool_response = lambda *a, **kw: None

        result = await tool_handlers.execute_tool_generic("get_calendar", {"calendarId": "bad"}, "test-db-001", "user1")
        assert result["isError"]
        assert "not found" in result["text"].lower()
    finally:
        # Restore original tools
        tool_handlers.MCP_TOOLS = original_tools


# --------------------------------------------------------------------------
# 5. Missing path parameter
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_missing_param(monkeypatch):
    monkeypatch.setattr(tool_handlers, "get_api_endpoint_for_tool", lambda n: ("GET", "/calendars/{calendarId}"))
    tool_handlers.log_tool_response = lambda *a, **kw: None
    result = await tool_handlers.execute_tool_generic("get_calendar", {}, "test-db-001", "user1")
    assert result["isError"]
    assert "The parameter 'calendarId' is required" in result["text"]


# --------------------------------------------------------------------------
# 6. Validation error (Pydantic detail list)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_validation_error(monkeypatch):
    async def mock_request(self, m, url, headers=None, json=None, params=None):
        class R:
            status_code = 422
            def json(self):
                return {"detail": [{"loc": ["body", "summary"], "msg": "field required"}]}
            text = ""
        return R()

    monkeypatch.setattr(tool_handlers.httpx.AsyncClient, "request", mock_request)
    monkeypatch.setattr(tool_handlers, "get_api_endpoint_for_tool", lambda n: ("POST", "/mock"))
    tool_handlers.log_tool_response = lambda *a, **kw: None

    result = await tool_handlers.execute_tool_generic("create_calendar", {}, "test-db-001", "user1")
    assert result["isError"]
    assert "Validation" in result["text"]


# --------------------------------------------------------------------------
# 7. Non-JSON error body
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_non_json_error(monkeypatch):
    async def mock_request(self, m, url, headers=None, json=None, params=None):
        class R:
            status_code = 500
            text = "Internal Error"
            def json(self): raise ValueError("Not JSON")
        return R()
    monkeypatch.setattr(tool_handlers.httpx.AsyncClient, "request", mock_request)
    monkeypatch.setattr(tool_handlers, "get_api_endpoint_for_tool", lambda n: ("POST", "/mock"))
    tool_handlers.log_tool_response = lambda *a, **kw: None

    result = await tool_handlers.execute_tool_generic("create_calendar", {}, "test-db-001", "user1")
    assert result["isError"]
    assert "HTTP 500" in result["text"]


# --------------------------------------------------------------------------
# 8. Request Exception
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_request_exception(monkeypatch):
    async def fail_request(*a, **kw): raise httpx.RequestError("Connection refused")
    monkeypatch.setattr(tool_handlers.httpx.AsyncClient, "request", fail_request)
    monkeypatch.setattr(tool_handlers, "get_api_endpoint_for_tool", lambda n: ("POST", "/mock"))
    tool_handlers.log_tool_response = lambda *a, **kw: None

    result = await tool_handlers.execute_tool_generic("create_calendar", {}, "test-db-001", "user1")
    assert result["isError"]
    assert "Request Error" in result["text"]


# --------------------------------------------------------------------------
# 9. Unexpected Exception
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_unexpected_exception(monkeypatch):
    monkeypatch.setattr(tool_handlers, "get_api_endpoint_for_tool", lambda n: 1/0)
    tool_handlers.log_tool_response = lambda *a, **kw: None
    result = await tool_handlers.execute_tool_generic("create_calendar", {}, "test-db-001", "user1")
    assert result["isError"]
    assert "Unexpected" in result["text"]


# --------------------------------------------------------------------------
# 10. API endpoint not found
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_endpoint_not_found(monkeypatch):
    monkeypatch.setattr(tool_handlers, "get_api_endpoint_for_tool", lambda n: None)
    tool_handlers.log_tool_response = lambda *a, **kw: None
    result = await tool_handlers.execute_tool_generic("unknown_tool", {}, "test-db-001", "user1")
    assert result["isError"]
    assert "API endpoint not found" in result["text"] or "Tool configuration not found" in result["text"]
