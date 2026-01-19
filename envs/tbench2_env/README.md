---
title: TB2 Environment Server
emoji: "🧪"
colorFrom: "#0F766E"
colorTo: "#22C55E"
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - terminal-bench-2
  - spaces
---

# TB2 Environment (Spaces-Compatible)

OpenEnv wrapper for Terminal-Bench 2 tasks that runs **locally inside the server container** (no Docker-in-Docker). This is designed for Hugging Face Spaces, where a Docker daemon is not available. It can execute commands and run task tests with pytest in the task directory.

> Note: TB2 tasks often specify a Docker image in `task.toml`. This local mode ignores those images, so tasks that rely on custom images may fail. Use a Docker-backed runner on a machine with Docker for full fidelity.

## Quick Start

```python
from tbench2_env import Tbench2Env, Tbench2Action

env = Tbench2Env(base_url="http://localhost:8000")
result = env.reset(task_id="headless-terminal")
print(result.observation.instruction)

result = env.step(Tbench2Action(action_type="exec", command="ls -la"))
print(result.observation.output)

result = env.step(Tbench2Action(action_type="evaluate"))
print(result.reward, result.done)

env.close()
```

## Action Types

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `exec` | Run a shell command | `command`, optionally `block`, `session_id` |
| `write` | Send input to a running session | `session_id`, `command` |
| `view` | Read pending output | `session_id` |
| `wait` | Wait for output | `session_id`, optionally `wait_seconds` |
| `kill` | Terminate a running session | `session_id` |
| `write_file` | Write content to a file | `file_path`, `content` |
| `evaluate` | Run pytest tests, return reward | (none) |
| `close` | Stop and cleanup | (none) |

## Session IDs (Streaming Processes)

`session_id` is **only** required when you start a non-blocking process and want to interact with it (`write`, `view`, `wait`, `kill`). For plain `exec` commands, you can omit it.

Example (Python):
```python
# Start a long-running process
env.step(Tbench2Action(action_type="exec", command="python -i", block=False, session_id="sess1"))

# Send input to it
env.step(Tbench2Action(action_type="write", session_id="sess1", command="print(2+2)\n"))

# Read its output
env.step(Tbench2Action(action_type="view", session_id="sess1"))
```

## Environment Variables

- `TB2_TASKS_DIR`: Path to a local Terminal-Bench-2 repo checkout (optional).
- `TB2_OUTPUT_DIR`: Directory to write session logs (default: `/tmp/tbench2_env_runs`).
- `TB2_CACHE_DIR`: Where to download/extract the TB2 repo if `TB2_TASKS_DIR` is unset.
- `TB2_REPO_URL`: Repo zip URL for auto-download (default: `https://github.com/laude-institute/terminal-bench-2/archive/refs/heads/main.zip`).

## Reward

Binary reward on `evaluate` action:
- `1.0` - All pytest tests pass (exit code 0)
- `0.0` - Tests fail (non-zero exit code)

Intermediate steps return `reward=None`.

## Running Locally

```bash
# Auto-download from GitHub (default)
python -m tbench2_env.server.app

# Use a local repo checkout
TB2_TASKS_DIR=/path/to/terminal-bench-2 python -m tbench2_env.server.app
```

## Project Structure

```
tbench2_env/
├── __init__.py              # Module exports (Tbench2Env, Tbench2Action, etc.)
├── README.md                # This file
├── client.py                # Tbench2Env client implementation
├── models.py                # Tbench2Action, Tbench2Observation, Tbench2State
├── openenv.yaml             # OpenEnv configuration
├── pyproject.toml           # Package dependencies
└── server/
    ├── __init__.py          # Server exports
    ├── app.py               # FastAPI application
    ├── tbench2_env_environment.py  # Core environment logic
    └── Dockerfile           # Container image definition
```
