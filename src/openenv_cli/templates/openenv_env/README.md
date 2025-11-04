---
title: __ENV_TITLE_NAME__ Environment Server
emoji: 🔊
colorFrom: '#00C9FF'
colorTo: '#1B2845'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# __ENV_TITLE_NAME__ Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the __ENV_TITLE_NAME__ environment is through the `__ENV_CLASS_NAME__Env` class:

```python
from __ENV_NAME__ import __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Env

try:
    # Create environment from Docker image
    __ENV_NAME__env = __ENV_CLASS_NAME__Env.from_docker_image("__ENV_NAME__-env:latest")

    # Reset
    result = __ENV_NAME__env.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = __ENV_NAME__env.step(__ENV_CLASS_NAME__Action(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    __ENV_NAME__env.close()
```

That's it! The `__ENV_CLASS_NAME__Env.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t __ENV_NAME__-env:latest -f server/Dockerfile .
```

## Environment Details

### Action
**__ENV_CLASS_NAME__Action**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**__ENV_CLASS_NAME__Observation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a __ENV_TITLE_NAME__ environment server running, you can connect directly:

```python
from __ENV_NAME__ import __ENV_CLASS_NAME__Env

# Connect to existing server
__ENV_NAME__env = __ENV_CLASS_NAME__Env(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = __ENV_NAME__env.reset()
result = __ENV_NAME__env.step(__ENV_CLASS_NAME__Action(message="Hello!"))
```

Note: When connecting to an existing server, `__ENV_NAME__env.close()` will NOT stop the server.

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/__ENV_NAME___environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
__ENV_NAME__/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── client.py              # __ENV_CLASS_NAME__Env client implementation
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── __ENV_NAME___environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    ├── Dockerfile         # Container image definition
    └── requirements.txt  # Python dependencies
```
