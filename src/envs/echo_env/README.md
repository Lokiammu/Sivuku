# Echo Environment HTTP Server

A simple test environment that echoes back messages. Perfect for testing the HTTP server infrastructure.

## Overview

The Echo environment is a minimal environment implementation designed to test the HTTP server infrastructure. It simply echoes back any message it receives along with some basic metadata.

## Files

```
server/
├── __init__.py            # Module exports
├── echo_environment.py    # EchoEnvironment implementation
├── app.py                 # FastAPI application
├── test_echo_env.py       # Direct environment tests
└── README.md              # This file
```

## Testing the Environment

### Direct Test (No HTTP Server)

Test the environment directly without starting the HTTP server:

```bash
# From the server directory
python3 test_echo_env.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

## Running the HTTP Server

### Prerequisites
```bash
pip install fastapi uvicorn
```

### Start the server
```bash
# From the src directory
cd /path/to/envtorch/src
uvicorn envs.echo_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

## API Endpoints

- `POST /reset` - Reset the environment
- `POST /step` - Send a message to be echoed
- `GET /state` - Get current environment state
- `GET /health` - Health check

## Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Reset
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'

# Echo a message
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "message": "Hello, Echo!"
    }
  }'

# Get state
curl http://localhost:8000/state
```

## Environment Details

### Action
- **EchoAction**: Contains a `message` field (string)

### Observation
- **EchoObservation**:
  - `echoed_message` (str) - The message echoed back
  - `message_length` (int) - Length of the message
  - `reward` (float) - Reward based on message length (length * 0.1)
  - `done` (bool) - Always False for echo environment
  - `metadata` (dict) - Additional info like step count

### State
- `episode_id` (str) - UUID for the current episode
- `step_count` (int) - Number of steps taken

## Implementation

The HTTP server is created with a single line:

```python
from core.env_server import create_fastapi_app
from envs.echo_env.server import EchoEnvironment
from envs.echo_env.models import EchoAction, EchoObservation

env = EchoEnvironment()
app = create_fastapi_app(env, EchoAction, EchoObservation)
```

That's it! The `create_fastapi_app` helper automatically:
- Creates all HTTP endpoints
- Handles serialization/deserialization
- Manages request/response formatting
- Provides health checks
