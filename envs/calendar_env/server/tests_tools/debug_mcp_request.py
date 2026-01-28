import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from handlers import mcp_handler


app = FastAPI()

@app.post("/mcp")
async def mcp_entry(request: Request):
    data = await request.json()
    print("\n=== RAW JSON BODY RECEIVED BY FASTAPI ===")
    print(data)
    result = await mcp_handler.handle_mcp_request(request)
    print("\n=== RESULT FROM HANDLER ===")
    print(result)
    return result


if __name__ == "__main__":
    client = TestClient(app)

    # Try variant 1: database_id at the same level as "name"
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "create_calendar",
            "database_id": "test-db-001",
            "user_id": "test-user-001",
            "arguments": {"summary": "Team Coordination"}
        },
        "id": 1
    }

    r = client.post("/mcp", json=payload)
    print("\n=== HTTP STATUS:", r.status_code, "===")
    print(r.json())
