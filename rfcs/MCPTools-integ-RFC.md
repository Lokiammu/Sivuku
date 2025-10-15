# RFC: MCP Tools Integration for OpenEnv

**Status**: Request for Comments
**Created**: OCtober 2025
**Authors**: EnvTorch Contributors
**Related**: OpenEnv-0.1-RFC.md

## Summary

This RFC proposes integrating Model Context Protocol (MCP) tools into OpenEnv environments, enabling agents and RL frameworks to discover and use external tools (file systems, APIs, databases, etc.) through a standardized interface. MCP tools will be surfaced via environment manifests, discoverable through standard APIs, and routable from within containerized environments.

## Motivation

### Problem Statement

AI agents and RL systems often need to interact with external tools and data sources:
- File system operations (read, write, search)
- API calls (web search, weather, databases)
- Code execution environments with tool access
- Multi-tool orchestration for complex tasks

Current challenges:
- **Discovery Problem**: Agents don't know what tools are available
- **Isolation Issues**: Tools running in containers need secure access to external resources
- **Configuration Complexity**: Tool setup varies across environments
- **Type Safety**: No standardized schemas for tool inputs/outputs

### Goals

1. **Standardized Tool Interface**: Use MCP protocol for consistent tool integration
2. **Easy Discovery**: Agents can query available tools via environment Gymnasium based APIs
3. **Declarative Configuration**: Tools defined in YAML manifests
4. **Container-Safe**: Tools work within Docker isolation boundaries
5. **Type-Safe**: Tool schemas validated at runtime
6. **Extensible**: Easy to add new MCP tool servers

## Background: Model Context Protocol (MCP)

MCP is a protocol that standardizes how AI assistants connect to data sources and tools:

- **MCP Server**: Provides tools/resources via standard protocol
- **MCP Client**: Discovers and invokes tools from servers
- **Tool Schema**: JSON Schema definition of inputs/outputs
- **Transport**: Typically stdio or HTTP

Example MCP tool:
```json
{
  "name": "read_file",
  "description": "Read contents of a file",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {"type": "string"}
    },
    "required": ["path"]
  }
}
```

### Types of MCP Tools Supported

OpenEnv will support three categories of MCP tools:

#### 1. External Backend Tools
Tools that call to another backend service or API.

**Examples**:
- Web search (Brave, Google, Bing)
- GitHub API (create issues, PRs, search repos)
- Slack/Discord bots
- Weather APIs
- Database queries (Postgres, MongoDB)

**Characteristics**:
- Require API keys/credentials
- Make external network calls
- May have rate limits

**Configuration Example**:
```yaml
- name: github
  type: mcp
  mcp_server:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    transport: stdio
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
```

#### 2. Simulated Backend Tools
Tools that simulate backend services for testing/training without real API calls.

**Examples**:
- Mock Airbnb booking system
- Mock Google Calendar
- Mock payment gateway
- Mock email service
- Simulated REST APIs

**Characteristics**:
- No external dependencies
- Fast and deterministic
- Perfect for training RL agents
- Can be reset to initial state
- Useful for benchmarking

**Configuration Example**:
```yaml
- name: airbnb_simulator
  type: mcp
  mcp_server:
    command: "python"
    args: ["-m", "mcp_simulators.airbnb"]
    transport: stdio
    env:
      SIMULATOR_SEED: "42"
```

#### 3. Self-Contained Service Tools
Tools that provide real services directly within the container.

**Examples**:
- Filesystem operations (read, write, search)
- SQLite database
- Local git operations
- Image processing
- PDF parsing

**Characteristics**:
- No external dependencies
- Fast execution
- Work offline
- Data stays within container
- Stateful across environment steps

**Configuration Example**:
```yaml
- name: filesystem
  type: mcp
  mcp_server:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    transport: stdio
```

## Design

### Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Agent / RL Framework                     │
│                                                             │
│  1. Query /tools → Get available tools                     │
│  2. Execute action with tool_name + tool_args              │
└────────────────┬───────────────────────────────────────────┘
                 │ HTTP
                 │
┌────────────────▼───────────────────────────────────────────┐
│              Environment Server (in Container)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Environment                                         │  │
│  │    - Reads tools.yaml manifest                       │  │
│  │    - Routes tool calls to MCP clients                │  │
│  │    - Returns tool results in observations            │  │
│  └──────────────┬───────────────────────────────────────┘  │
│                 │                                           │
│  ┌──────────────▼───────────────────────────────────────┐  │
│  │  MCP Client Manager                                  │  │
│  │    - Manages connections to MCP servers              │  │
│  │    - Caches tool schemas                             │  │
│  │    - Handles tool invocation                         │  │
│  └──────────────┬───────────────────────────────────────┘  │
└─────────────────┼──────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌────▼─────┐  ┌───▼────┐
│ MCP    │  │ MCP      │  │ MCP    │
│ Server │  │ Server   │  │ Server │
│ (Files)│  │ (GitHub) │  │ (DB)   │
└────────┘  └──────────┘  └────────┘
```

### 1. Surfacing MCP Tools via Manifest

#### Environment Tools Manifest (`tools.yaml`)

Each environment declares available tools in a YAML manifest:

```yaml
# src/envs/my_env/tools.yaml
version: "1.0"
tools:
  - name: filesystem
    type: mcp
    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
      transport: stdio
    enabled: true

  - name: github
    type: mcp
    mcp_server:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-github"]
      transport: stdio
      env:
        GITHUB_TOKEN: "${GITHUB_TOKEN}"
    enabled: true

  - name: brave_search
    type: mcp
    mcp_server:
      url: "http://mcp-brave-search:8080"
      transport: http
    enabled: false  # Disabled by default
```

**Manifest Fields**:
- `name`: Unique identifier for the tool set
- `type`: Always "mcp" for MCP tools
- `mcp_server.command`: Executable to run (for stdio transport)
- `mcp_server.args`: Arguments to pass
- `mcp_server.url`: HTTP endpoint (for http transport)
- `mcp_server.env`: Environment variables (with secret interpolation)
- `enabled`: Whether to load this tool set

#### Loading Manifest in Environment

```python
# server/my_environment.py
import yaml
from pathlib import Path
from core.tools.mcp import MCPClientManager

class MyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = MyState()

        # Load tools manifest
        tools_config = self._load_tools_manifest()
        self._mcp_manager = MCPClientManager(tools_config)

    def _load_tools_manifest(self) -> dict:
        manifest_path = Path(__file__).parent.parent / "tools.yaml"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return yaml.safe_load(f)
        return {"tools": []}
```

### 2. Tool Discoverability

#### New HTTP Endpoint: `/tools`

Environments expose available tools via a dedicated endpoint:

```python
# core/env_server/http_server.py

@app.get("/tools")
async def get_tools() -> Dict[str, Any]:
    """Get available tools in this environment."""
    tools = env.get_available_tools()
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in tools
        ]
    }
```

**Response Example**:
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read contents of a file",
      "inputSchema": {
        "type": "object",
        "properties": {
          "path": {"type": "string"}
        },
        "required": ["path"]
      }
    }
  ]
}
```

### 3. Routing Tool Calls from Environment

#### Action Model Extension

```python
@dataclass
class UnifiedAction(Action):
    """Unified action supporting both code and tools."""
    action_type: str  # "code" or "tool"
    code: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
```

#### Environment Step Logic

```python
def step(self, action: UnifiedAction) -> MyObservation:
    if action.action_type == "code":
        result = self._executor.run(action.code)
        return MyObservation(stdout=result.stdout, ...)

    elif action.action_type == "tool":
        tool_result = self._mcp_manager.call_tool(
            tool_name=action.tool_name,
            arguments=action.tool_args
        )
        return MyObservation(stdout=str(tool_result.content), ...)
```

### 4. Environment Logic for Tool Support

```python
class ToolEnabledEnvironment(Environment):
    """Base class for environments that support MCP tools."""

    def __init__(self, tools_manifest: str | None = None):
        super().__init__()
        if tools_manifest:
            tools_config = self._load_tools_manifest(tools_manifest)
            self._mcp_manager = MCPClientManager(tools_config)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        if self._mcp_manager:
            return self._mcp_manager.get_available_tools()
        return []
```

### 5. Docker Runtime and Packaging

#### Dockerfile with MCP Support

```dockerfile
FROM envtorch-base:latest

# Install Node.js for MCP servers
RUN apt-get update && apt-get install -y nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# Install Python MCP SDK
RUN pip install --no-cache-dir mcp

# Copy environment code
COPY src/core/ /app/src/core/
COPY src/envs/my_env/ /app/src/envs/my_env/

# Set up workspace for filesystem tools
RUN mkdir -p /workspace && chmod 777 /workspace

CMD ["uvicorn", "envs.my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Volume Mounts for Tool Access

```python
# Starting container with workspace mount
client = CodingEnv.from_docker_image(
    "coding-env:latest",
    workspace_dir=os.getcwd(),  # Mount current directory
    env_vars={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}
)
```

## Open Questions & Feedback Requested

### 1. Tool Discovery Timing
**Question**: When should tools be discovered - at startup or lazy loaded?

**Feedback Needed**: Which approach fits your use case?

### 2. Tool Security
**Question**: How should we handle tool permissions and secrets?

**Options**:
- Allowlist of permitted tools
- Role-based access control
- Audit logging

**Feedback Needed**: What security measures are essential?

### 3. Tool Schema Validation
**Question**: Client-side, server-side, or both?

**Feedback Needed**: Where should validation happen?

### 4. Multi-Tool Composition
**Question**: Should environments support automatic tool chaining?

**Feedback Needed**: Is this in scope?

## Implementation Plan

### Phase 1: Core MCP Integration
- [ ] Add MCP Python SDK dependency
- [ ] Implement `MCPClientManager`
- [ ] Add `ToolEnabledEnvironment` base class
- [ ] Implement `/tools` HTTP endpoint
- [ ] Create tools.yaml manifest schema

### Phase 2: Container Support
- [ ] Update Dockerfile to include Node.js
- [ ] Add volume mount support
- [ ] Test with filesystem MCP server

### Phase 3: Example Environments
- [ ] Update `CodingEnv` to support tools
- [ ] Add example tools.yaml
- [ ] Create documentation

## Conclusion

MCP tools integration will enable EnvTorch environments to provide standardized tool access to AI agents and RL frameworks. The proposed design uses declarative YAML manifests, provides automatic discovery via `/tools` endpoint, and maintains container isolation through careful volume mounting and environment variable passing.

**Key areas for feedback**:
1. Tool discovery and initialization timing
2. Security model for tool access
3. Multi-tool composition support
4. Docker runtime requirements

Please share your thoughts via GitHub Issues or Discussions!
