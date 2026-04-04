# AGENTS.md — OpenEnv Unified Agent Context

> Read this file first. It is the single source of truth for any AI agent working
> in this repository. It consolidates README, PRINCIPLES, INVARIANTS, PATTERNS,
> REPO_WALKTHROUGH, and TESTING_STRATEGY into one document.

---

## What is OpenEnv?

**OpenEnv** (package: `openenv-core`, v0.2.3) is a Meta/PyTorch framework for creating,
deploying, and using isolated execution environments for **agentic Reinforcement Learning
training**. It provides a Gymnasium-style API (`reset`, `step`, `state`) over WebSocket +
Docker so RL frameworks (TRL, torchforge, SkyRL, Unsloth, ART, Oumi) can train LLMs against
diverse simulated environments.

**Key idea**: An environment is a Docker container exposing a FastAPI WebSocket server.
Clients connect via a Python `EnvClient` subclass. Agents interact via MCP tools.

---

## Architecture

```
Training Loop (RL framework)
        │  WebSocket (reset / step / state)
        ▼
┌────────────────────────────┐
│  FastAPI + WebSocket       │  ← HTTPEnvServer (src/openenv/core/env_server/)
│  ┌──────────────────────┐  │
│  │  MyEnvironment        │  │  ← Environment[ActT, ObsT, StateT] subclass
│  │  (reset / step /     │  │
│  │   state / rubric)    │  │
│  └──────────────────────┘  │
│  MCP endpoint (/mcp)        │  ← MCPEnvironment (via FastMCP)
└────────────────────────────┘
        ▲
        │  Docker (isolated container)
        │
EnvClient[ActT, ObsT, StateT]   ← client-side Python class
MCPToolClient                    ← for MCP-only environments (e.g. echo_env)
```

**Two distinct API boundaries** (critical — see Invariants):

| Boundary | Protocol | Used by |
|----------|----------|---------|
| Gym-like (`reset/step/state`) | WebSocket | Training orchestration only |
| Agent tools | MCP (`/mcp`) | The agent being trained |

---

## Repository Map

```
OpenEnv/
├── src/openenv/               # Core library (installed as `openenv-core`)
│   ├── core/
│   │   ├── env_server/
│   │   │   ├── interfaces.py      # Environment ABC (the base class to subclass)
│   │   │   ├── http_server.py     # HTTPEnvServer — FastAPI + WebSocket wrapper
│   │   │   ├── mcp_environment.py # MCPEnvironment — adds FastMCP tools to env
│   │   │   ├── types.py           # Action, Observation, State, wire types
│   │   │   ├── serialization.py   # Pydantic serialization helpers
│   │   │   └── web_interface.py   # Debug UI (ENABLE_WEB_INTERFACE=true)
│   │   ├── containers/runtime/    # LocalDockerProvider, DaytonaProvider, UVProvider
│   │   ├── env_client.py          # EnvClient ABC (base class for env clients)
│   │   ├── mcp_client.py          # MCPToolClient (for MCP-only environments)
│   │   ├── generic_client.py      # GenericEnvClient (dict-based, untyped)
│   │   ├── sync_client.py         # SyncEnvClient (.sync() wrapper)
│   │   ├── rubrics/               # Rubric base + LLM judge + trajectory rubric
│   │   ├── evals/                 # Evaluation harness
│   │   └── tools/                 # Reusable tools (python executor, git client)
│   ├── auto/                      # Auto-discovery (_discovery.py, auto_env.py)
│   └── cli/                       # `openenv` CLI (init, push, serve, build, validate)
│       └── templates/openenv_env/ # Scaffold template for `openenv init`
│
├── envs/                      # 30 environments (not installed, on PYTHONPATH)
│   ├── echo_env/              # *** REFERENCE IMPLEMENTATION — read this first ***
│   ├── coding_env/            # Python code execution (smolagents)
│   ├── chess_env/             # Chess with configurable opponents
│   ├── atari_env/             # Atari via Gymnasium ALE
│   ├── textarena_env/         # Text-based games (TextArena)
│   └── ...                    # 25 more environments
│
├── tests/
│   ├── core/                  # Core library unit + integration tests
│   │   ├── test_evals/        # Evaluation harness tests
│   │   ├── test_mcp/          # MCP client/server tests
│   │   └── test_rubrics/      # Rubric tests
│   ├── envs/                  # Per-environment tests
│   ├── test_cli/              # CLI command tests
│   └── scripts/               # Script utility tests
│
├── examples/                  # 45 runnable example scripts
├── docs/                      # Sphinx documentation
├── rfcs/                      # Architectural decision records (001–005)
├── scripts/                   # HF deployment / PR review utilities
├── tutorial/                  # GPU Mode lecture materials + examples
│
├── AGENTS.md                  # ← You are here
├── CLAUDE.md                  # Claude Code configuration and skill index
├── pyproject.toml             # Package config (openenv-core, v0.2.3)
└── .claude/                   # Skills, agents, hooks for Claude Code
```

---

## Every Environment Follows This Structure

```
envs/my_env/
├── __init__.py          # exports: MyAction, MyObservation, MyEnv
├── models.py            # Pydantic: MyAction, MyObservation, MyState (shared client+server)
├── client.py            # class MyEnv(MCPToolClient) or (EnvClient[...])
├── openenv.yaml         # manifest: name, type, runtime, port
├── pyproject.toml       # environment-specific dependencies
├── README.md
└── server/
    ├── my_environment.py  # class MyEnvironment(MCPEnvironment or Environment[...])
    ├── app.py             # create_app(env, MyAction, MyObservation)
    ├── requirements.txt   # Docker deps
    └── Dockerfile
```

Scaffold a new one: `openenv init my_env`

---

## Core Patterns

### Implementing an Environment (server-side)

```python
# server/my_environment.py
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP
from uuid import uuid4

class MyEnvironment(MCPEnvironment):
    def __init__(self):
        mcp = FastMCP("my_env")

        @mcp.tool
        def do_something(param: str) -> str:
            """Tool description for the agent."""
            return f"result: {param}"

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return Observation(done=False, reward=0.0, metadata={"status": "ready"})

    def step(self, action: Action, **kwargs) -> Observation:
        self._state.step_count += 1
        return super().step(action, **kwargs)

    @property
    def state(self) -> State:
        return self._state
```

### Typed Environment (non-MCP)

```python
# models.py
from pydantic import BaseModel
from openenv.core.env_server.types import Action, Observation, State

class MyAction(Action):
    command: str

class MyObservation(Observation):
    result: str

# server/my_environment.py
from openenv.core.env_server.interfaces import Environment

class MyEnvironment(Environment[MyAction, MyObservation, State]):
    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:
        return MyObservation(result="ready", done=False, reward=0.0)

    def step(self, action: MyAction, **kwargs) -> MyObservation:
        return MyObservation(result=action.command, done=False, reward=1.0)

    @property
    def state(self) -> State:
        return self._state
```

### Client

```python
# client.py
from openenv.core.mcp_client import MCPToolClient

class MyEnv(MCPToolClient):
    pass  # inherits: list_tools(), call_tool(), reset(), step()
```

### App Setup

```python
# server/app.py
from openenv.core.env_server import create_app
from .my_environment import MyEnvironment

env = MyEnvironment()
app = create_app(env)
```

---

## Non-Negotiable Invariants

These must NEVER be violated. If a change would break one, stop and flag it.

1. **Agents cannot reset.** `reset()` is infrastructure-only. Never expose it via MCP tools.
2. **Dual API boundary.** WebSocket = training orchestration. MCP = agent tools. Never mix.
3. **Rewards inside environment.** Compute rewards in `step()`, not externally.
4. **Client-server separation.** `client.py` must never import from `server/`. Shared types live in `models.py`.
5. **Gym API signatures are frozen:**
   - `reset(seed?, episode_id?) -> Observation`
   - `step(action) -> Observation`
   - `state -> State` (property)
6. **All wire types are Pydantic models** — JSON-serializable.
7. **One env = one trajectory.** No multiplexing. For batches, stack multiple container instances.

---

## Error Handling Pattern

Return errors in observations — don't raise exceptions to the client.

```python
def step(self, action: MyAction, **kwargs) -> MyObservation:
    try:
        result = self._execute(action)
        return MyObservation(result=result, done=False, reward=1.0)
    except InvalidAction as e:
        return MyObservation(result="", error=str(e), done=False, reward=0.0)
    except FatalError as e:
        return MyObservation(result="", error=str(e), done=True, reward=0.0)
```

---

## Design Principles (from RFC 000)

1. Minimize lifecycle deltas — training/evals/production use identical interfaces
2. Minimize human-agent divergence — tools for humans work for agents
3. Be hands-on — ready-to-use implementations, not specs
4. Design for LLMs — context-efficient, in-distribution behavior

**Trade-offs deliberately made:**
- Flexibility traded for simplicity (one canonical way to build envs)
- Performance traded for isolation (Docker overhead accepted for reproducibility)
- Cutting-edge traded for stability (FastAPI over experimental frameworks)

---

## Build & Test Commands

```bash
# Install
pip install -e .          # or: uv pip install -e .
uv sync --all-extras      # install everything including dev/docs deps

# Run tests
PYTHONPATH=src:envs uv run pytest tests/ -v --tb=short
PYTHONPATH=src:envs uv run pytest tests/envs/test_chess_environment.py -v  # single file
bash .claude/hooks/test.sh  # same but excludes envs needing special setup

# Lint / format
uv run usort check src/ tests/
uv run ruff format src/ tests/ --check
uv run ruff check src/ tests/
# Auto-fix:
uv run usort format src/ tests/ && uv run ruff format src/ tests/

# Docker
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t echo-env:latest -f envs/echo_env/server/Dockerfile .

# CLI
openenv init my_env        # scaffold new environment
openenv push               # deploy to HuggingFace Spaces
openenv serve              # run server locally
openenv build              # build Docker image
openenv validate           # check openenv.yaml config
```

---

## Testing Conventions

```
tests/
├── core/          # Unit + integration tests for src/openenv/core/
├── envs/          # Per-environment integration tests
├── test_cli/      # CLI command tests
└── scripts/       # Tests for scripts/
```

```python
# Async test (pytest-asyncio configured in pyproject.toml)
async def test_reset_works(env):
    obs = await env.reset()
    assert obs.done is False

# Parametrized
@pytest.mark.parametrize("msg,expected", [("hi", "hi"), ("", "")])
def test_echo(msg, expected):
    assert transform(msg) == expected

# Docker / network tests use markers
@pytest.mark.docker
async def test_container_starts():
    ...
```

**High-signal tests cover:** Pydantic validation, WebSocket protocol, reward computation,
state machine transitions, error recovery.

**Skip:** Testing Python builtins, mocking so heavily no real behavior is tested.

---

## Development Workflow (TDD)

```
/work-on-issue #42     Start from GitHub issue
      ↓
/write-tests           Write failing tests (Red)
      ↓
/implement             Make tests pass (Green)
      ↓
/update-docs           Fix stale docs
      ↓
/simplify              Refactor (optional)
      ↓
/pre-submit-pr         Validate before PR
```

Skills (run inline): `alignment-review`, `pre-submit-pr`, `rfc-check`, `generate-openenv-env`

Agents (isolated): `alignment-reviewer`, `env-validator`, `openenv-architect`, `tester`, `implementer`

---

## Key RFCs (Architectural Decisions)

| RFC | Topic | Key Decision |
|-----|-------|-------------|
| 001 | Abstractions | Two-interface model (WebSocket + MCP), agents cannot reset |
| 002 | Env Spec | Rewards inside environment, one env = one trajectory |
| 003 | MCP Support | MCP as universal agent-environment standard |
| 004 | Rubrics | Delayed rewards via trajectory-based scoring |
| 005 | Agentic Harness | Integration with RL training frameworks |

---

## Common Pitfalls

- **Client importing server code** → violation of client-server separation
- **Exposing reset() via MCP** → agents must not control simulation
- **Computing rewards outside env** → violates RFC 002
- **Multiplexing trajectories** → not supported; stack container instances instead
- **Modifying Gym API signatures** → requires major version bump
- **Heavy deps in root pyproject.toml** → each env has its own pyproject.toml
