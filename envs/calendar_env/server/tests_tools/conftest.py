# calendar/tests_tools/conftest.py
import importlib
import importlib.util
import os
import sqlite3
import types
from pathlib import Path
import pytest

# ---- How we load the tools registry ----
TOOLS_MODULE_ENV = "CAL_TOOLS_MODULE"   # optional
TOOLS_FILE_ENV   = "CAL_TOOLS_PATH"     # preferred for our adapter

def _import_module_from_path(path: str, name: str = "calendar_tools_adapter") -> types.ModuleType:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"{TOOLS_FILE_ENV} points to missing file: {p}")
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture(scope="session")
def tools_root_module() -> types.ModuleType:
    """
    Prefer an explicit adapter module if provided via env vars.
    Falls back to the legacy package only if neither env var is set.
    """
    file_path = os.getenv(TOOLS_FILE_ENV)
    if file_path:
        return _import_module_from_path(file_path)

    mod_name = os.getenv(TOOLS_MODULE_ENV)
    if mod_name:
        return importlib.import_module(mod_name)

    # legacy fallback (not recommended because of stdlib 'calendar' collision)
    return importlib.import_module("calendar_mcp.tools")

@pytest.fixture(scope="session")
def tool_registry(tools_root_module):
    """
    Expect the adapter to expose a TOOLS dict {name: handler}.
    """
    reg = getattr(tools_root_module, "TOOLS", None)
    assert isinstance(reg, dict) and reg, "Adapter did not expose any tools via TOOLS"
    print(f"\n[conftest] tool registry size: {len(reg)}")
    return reg

# ---- Convenience fixtures the tests expect ----
@pytest.fixture(scope="session")
def all_tool_names(tool_registry):
    return sorted(tool_registry.keys())

@pytest.fixture()
def tool_call(tool_registry):
    def _invoke(name: str, params: dict):
        handler = tool_registry[name]  # let KeyError surface if test asked for wrong name
        try:
            res = handler(params or {})
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], bool):
                return res[0], res[1]
            return True, res
        except Exception as e:
            return False, {"error": str(e)}
    return _invoke

@pytest.fixture()
def pick_tool(all_tool_names):
    lowered = [n.lower() for n in all_tool_names]
    def _pick(*hints: str) -> str:
        for h in hints:
            if h in all_tool_names:
                return h
        for h in hints:
            hl = h.lower()
            if hl in lowered:
                return all_tool_names[lowered.index(hl)]
        for h in hints:
            hl = h.lower()
            for i, n in enumerate(lowered):
                if hl in n:
                    return all_tool_names[i]
        raise KeyError(f"No tool matches {hints}. Sample: {all_tool_names[:10]}")
    return _pick

# ---- Minimal DB seed so tools can run ----
@pytest.fixture(scope="session")
def sample_sql() -> str:
    return """
    PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, email TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS calendars (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT);
    CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, calendar_id TEXT NOT NULL, summary TEXT, start TEXT, end TEXT);
    INSERT OR IGNORE INTO users(user_id, email) VALUES ('alice_manager','alice.manager@techcorp.com');
    INSERT OR IGNORE INTO calendars(id, user_id, summary) VALUES ('alice-primary','alice_manager','Alice Primary'),
                                                                 ('alice-projects','alice_manager','Alice Projects');
    INSERT OR IGNORE INTO events(id, calendar_id, summary, start, end)
      VALUES ('ev1','alice-projects','Kickoff','2025-10-07T10:00:00Z','2025-10-07T11:00:00Z');
    """

@pytest.fixture()
def seeded_db(tmp_path, sample_sql):
    db_dir = tmp_path / "mcp_databases"
    db_dir.mkdir(exist_ok=True)
    dbid = "pytestdb"
    db_path = db_dir / f"{dbid}.sqlite"
    con = sqlite3.connect(db_path)
    try:
        con.executescript(sample_sql)
        con.commit()
    finally:
        con.close()
    return {
        "database_id": dbid,
        "user_id": "alice_manager",
        "email": "alice.manager@techcorp.com",
        "primary_calendar_id": "alice-primary",
        "some_calendar_id": "alice-projects",
        "db_path": str(db_path),
    }
