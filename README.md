# EnvTorch: Agentic Execution Environments

An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style simple APIs.

## Overview

EnvTorch provides a standard for interacting with agentic execution environments via simple Gymnasium style APIs - step(), reset(), state(). Users of agentic execution environments can interact with the environment during RL training loops using these simple APIs. In addition to making it easier for researchers and RL framework writers, we also provide tools for environment creators making it easier for them to create richer environments and make them available over familar protocols like HTTP and packaged using canonical technologies like docker. Environment creators can use the EnvTorch framework to create environments that are isolated, secure, and easy to deploy and use.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Client Application                    │
│  ┌────────────────┐              ┌──────────────────┐   │
│  │  EchoEnv       │              │  CodingEnv       │   │
│  │  (HTTPEnvClient)│              │  (HTTPEnvClient) │   │
│  └────────┬───────┘              └────────┬─────────┘   │
└───────────┼───────────────────────────────┼─────────────┘
            │ HTTP                           │ HTTP
            │ (reset, step, state)           │
┌───────────▼───────────────────────────────▼─────────────┐
│              Docker Containers (Isolated)                │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ FastAPI Server       │    │ FastAPI Server       │   │
│  │   EchoEnvironment    │    │ PythonCodeActEnv     │   │
│  │ (Environment base)   │    │ (Environment base)   │   │
│  └──────────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Environment (Server-Side)
Base class for implementing environment logic:
- **`reset()`**: Initialize a new episode, returns initial `Observation`
- **`step(action)`**: Execute an `Action`, returns resulting `Observation`
- **`state()`**: Access episode metadata (`State` with episode_id, step_count, etc.)

#### 2. HTTPEnvClient (Client-Side)
Base class for HTTP communication:
- Handles HTTP requests to environment server
- Contains a utility to spin up a docker container locally for the corresponding environment
- Type-safe action/observation parsing

#### 3. Container Providers
Manage container deployment:
- `LocalDockerProvider`: Run containers on local Docker daemon
- `KubernetesProvider`: Deploy to K8s clusters (future)

#### 4. Models
Type-safe data structures:
- `Action`: Base class for environment actions
- `Observation`: Base class for environment observations
- `State`: Episode state tracking
- `StepResult`: Combines observation, reward, done flag

## Design Principles

1. **Separation of Concerns**: Clear client-server boundaries
2. **Type Safety**: Strongly-typed actions, observations, and state
3. **Container Isolation**: Each environment runs in its own container
4. **Simple APIs**: Minimal, intuitive interfaces

## Quick Start

### Using the Echo Environment(Example)

```python
from envs.echo_env import EchoAction, EchoEnv

# Automatically start container and connect
client = EchoEnv.from_docker_image("echo-env:latest")

# Reset the environment
result = client.reset()
print(result.observation.echoed_message)  # "Echo environment ready!"

# Send messages
result = client.step(EchoAction(message="Hello, World!"))
print(result.observation.echoed_message)  # "Hello, World!"
print(result.reward)  # 1.3 (based on message length)

# Cleanup
client.close()  # Stops and removes container
```

## Requirements

- Python 3.11+
- Docker Desktop or Docker Engine
- FastAPI >= 0.104.0
- Uvicorn >= 0.24.0
- Requests >= 2.25.0
- smolagents (for coding environment)

## Example Environments

### Echo Environment
A simple environment that echoes back messages with metadata. Perfect for:
- Testing the HTTP server infrastructure
- Learning the framework basics
- Verifying container deployment

See: [`src/envs/echo_env/README.md`](src/envs/echo_env/README.md)

### Coding Environment
Executes arbitrary Python code in a sandboxed environment. Features:
- Safe code execution using smolagents
- Capture stdout, stderr, and exit codes
- Persistent execution context within episodes
- Error handling with detailed messages

See: [`src/envs/coding_env/README.md`](src/envs/coding_env/README.md)


## Building Your Own Environment

### 1. Define Models

```python
# models.py
from dataclasses import dataclass
from core.env_server import Action, Observation, State

@dataclass
class MyAction(Action):
    """Your custom action."""
    command: str
    parameters: dict

@dataclass
class MyObservation(Observation):
    """Your custom observation."""
    result: str
    success: bool

@dataclass
class MyState(State):
    """Custom state fields."""
    custom_field: int = 0
```

### 2. Implement Environment

```python
# server/my_environment.py
from core.env_server import Environment
from ..models import MyAction, MyObservation, MyState

class MyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = MyState()

    def reset(self) -> MyObservation:
        self._state = MyState(episode_id=str(uuid.uuid4()))
        return MyObservation(result="Ready", success=True)

    def step(self, action: MyAction) -> MyObservation:
        # Implement your logic here
        self._state.step_count += 1
        result = self._execute_command(action.command)
        return MyObservation(result=result, success=True)

    @property
    def state(self) -> MyState:
        return self._state
```

### 3. Create FastAPI Server

```python
# server/app.py
from core.env_server import create_fastapi_app
from ..models import MyAction, MyObservation
from .my_environment import MyEnvironment

env = MyEnvironment()
app = create_fastapi_app(env, MyAction, MyObservation)
```

### 4. Create Dockerfile

```dockerfile
FROM envtorch-base:latest

# Install any additional dependencies
RUN pip install --no-cache-dir your-dependencies

# Copy environment code
COPY src/core/ /app/src/core/
COPY src/envs/my_env/ /app/src/envs/my_env/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "envs.my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. Implement Client

```python
# client.py
from core.http_env_client import HTTPEnvClient
from core.types import StepResult
from .models import MyAction, MyObservation, MyState

class MyEnv(HTTPEnvClient[MyAction, MyObservation]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"command": action.command, "parameters": action.parameters}

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs = MyObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MyState:
        return MyState(**payload)
```


## Building and Running

### 1. Build Base Image

```bash
docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .
```

### 2. Build Environment Images

```bash
# Echo environment
docker build -t echo-env:latest -f src/envs/echo_env/server/Dockerfile .

# Coding environment
docker build -t coding-env:latest -f src/envs/coding_env/server/Dockerfile .
```

### 3. Run Examples

```bash
# Test echo environment
python3 examples/local_echo_env.py

# Test coding environment
python3 examples/local_coding_env.py
```

## API Reference

### HTTPEnvClient Methods

```python
# Create from Docker image (automatic container management)
client = MyEnv.from_docker_image("my-env:latest")

# Connect to existing server
client = MyEnv(base_url="http://localhost:8000")

# Environment operations
result = client.reset()                    # Reset environment
result = client.step(action)               # Execute action
state = client.state()                     # Get current state
client.close()                             # Cleanup resources
```


## License

BSD 3-Clause License (see LICENSE file)
