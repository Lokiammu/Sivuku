# OpenEnv Core

Core components for OpenEnv - a framework for building HTTP-based agentic environments.

## Overview

`openenv-core` provides the foundational building blocks for creating and interacting with containerized environments over HTTP. It enables you to build agent environments that can be deployed as Docker containers and accessed via a simple HTTP API.

## Features

- **HTTPEnvClient**: Generic HTTP client for interacting with remote environments
- **HTTPEnvServer**: FastAPI-based server wrapper for exposing environments over HTTP
- **Container Providers**: Pluggable architecture for running containers (Docker, Kubernetes, etc.)
- **Type System**: Strongly-typed Action/Observation/State interfaces
- **Web Interface**: Optional web UI for interacting with environments

## Installation

```bash
pip install openenv-core
```

For development:
```bash
pip install openenv-core[dev]
```

## Quick Start

### Creating an Environment Client

```python
from openenv_core import HTTPEnvClient, StepResult
from dataclasses import dataclass

@dataclass
class MyAction:
    text: str

@dataclass
class MyObservation:
    response: str

class MyEnvClient(HTTPEnvClient[MyAction, MyObservation]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"text": action.text}

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs_data = payload["observation"]
        return StepResult(
            observation=MyObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> Any:
        return payload

# Use with Docker
env = MyEnvClient.from_docker_image("my-env:latest")
result = env.reset()
step_result = env.step(MyAction(text="hello"))
env.close()
```

### Creating an Environment Server

```python
from openenv_core.env_server import Environment, HTTPEnvServer, create_app
from dataclasses import dataclass

@dataclass
class MyAction:
    text: str

@dataclass
class MyObservation:
    response: str
    reward: float = 0.0
    done: bool = False

class MyEnvironment(Environment):
    def reset(self) -> MyObservation:
        return MyObservation(response="Ready")

    def step(self, action: MyAction) -> MyObservation:
        return MyObservation(
            response=f"Echo: {action.text}",
            reward=1.0,
            done=False
        )

# Create FastAPI app
env = MyEnvironment()
app = create_app(env, MyAction, MyObservation)

# Run with: uvicorn module:app --host 0.0.0.0 --port 8000
```

## Container Providers

OpenEnv Core supports multiple container providers:

### Local Docker Provider

```python
from openenv_core.containers.runtime import LocalDockerProvider

provider = LocalDockerProvider()
base_url = provider.start_container("my-env:latest")
provider.wait_for_ready(base_url)
# Use environment...
provider.stop_container()
```

### Kubernetes Provider (Coming Soon)

```python
from openenv_core.containers.runtime import KubernetesProvider

provider = KubernetesProvider(namespace="envs")
base_url = provider.start_container("my-env:latest")
# Use environment...
provider.stop_container()
```

## Architecture

OpenEnv Core follows a client-server architecture:

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│                 │◄─────────────────────►│                 │
│  HTTPEnvClient  │   /reset, /step       │  HTTPEnvServer  │
│                 │   /state, /health     │                 │
└─────────────────┘                       └─────────────────┘
        │                                          │
        │                                          │
        ▼                                          ▼
┌─────────────────┐                       ┌─────────────────┐
│ Container       │                       │  Environment    │
│ Provider        │                       │  Implementation │
└─────────────────┘                       └─────────────────┘
```

## API Reference

### HTTPEnvClient

Base class for environment clients with these abstract methods:

- `_step_payload(action)`: Convert action to JSON
- `_parse_result(payload)`: Parse response to StepResult
- `_parse_state(payload)`: Parse state response

### HTTPEnvServer

Server wrapper with these methods:

- `register_routes(app)`: Register endpoints on FastAPI app
- `_deserialize_action(data)`: Convert JSON to Action
- `_serialize_observation(obs)`: Convert Observation to JSON

### Environment Interface

Base interface for environment implementations:

- `reset()`: Reset environment and return initial observation
- `step(action)`: Execute action and return observation
- `state`: Property returning current environment state

## License

This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see the main OpenEnv repository for contribution guidelines.

## Links

- **Homepage**: https://github.com/facebookresearch/OpenEnv
- **Documentation**: https://github.com/facebookresearch/OpenEnv/blob/main/README.md
- **Bug Tracker**: https://github.com/facebookresearch/OpenEnv/issues
