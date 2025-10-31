---
title: WebArena Environment Server
emoji: 🌐
colorFrom: '#00C9FF'
colorTo: '#1B2845'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - webarena
  - web-agents
  - browser-automation
---

# WebArena Environment

WebArena is a realistic web environment for building autonomous agents that can interact with web interfaces. This environment provides browser-based interaction where agents can perform complex tasks on websites including shopping, forums, content management systems, GitLab, Wikipedia, and more.

## Features

- **Browser-Based Interaction**: Powered by Playwright for realistic web browsing
- **Multiple Observation Types**: Support for accessibility trees and HTML observations
- **Rich Action Space**: Click, type, navigate, scroll, form interactions, and more
- **Configurable Tasks**: JSON-based task configuration with evaluation metrics
- **Realistic Environments**: Test agents on real web applications
- **Gymnasium Compatible**: Follows the standard Gymnasium API

## Quick Start

### Using Docker

```python
from envs.webarena_env import WebArenaEnv, WebArenaAction

# Create environment from Docker image
env = WebArenaEnv.from_docker_image(
    "ghcr.io/openenv/webarena-env:latest",
    environment={
        "SHOPPING": "http://your-server:7770",
        "SHOPPING_ADMIN": "http://your-server:7780/admin",
        "REDDIT": "http://your-server:9999",
        "GITLAB": "http://your-server:8023",
        "MAP": "http://your-server:3000",
        "WIKIPEDIA": "http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "HOMEPAGE": "http://your-server:4399"
    }
)

# Reset the environment
result = env.reset()
print(f"Task: {result.observation.text[:200]}")

# Take actions
action = WebArenaAction(action_str="click [123]")
result = env.step(action)
print(f"Success: {result.observation.success}")
print(f"URL: {result.observation.url}")

# Clean up
env.close()
```

### Direct HTTP Connection

```python
from envs.webarena_env import WebArenaEnv, WebArenaAction

# Connect to a running server
env = WebArenaEnv(base_url="http://localhost:8000")

# Reset and interact
result = env.reset()
action = WebArenaAction(action_str="goto [http://example.com]")
result = env.step(action)

env.close()
```

## Building the Docker Image

### Prerequisites

1. **WebArena Websites**: You need to set up the WebArena websites first. See the [WebArena documentation](https://github.com/web-arena-x/webarena/tree/main/environment_docker) for details on setting up the backend environments.

2. **Base Image**: Build the OpenEnv base image first:

```bash
# From the OpenEnv repository root
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
```

### Build the WebArena Environment

```bash
# From the OpenEnv repository root
docker build -t webarena-env:latest -f src/envs/webarena_env/server/Dockerfile .
```

### Run the Server

```bash
docker run -p 8000:8000 \
  -e SHOPPING="http://your-server:7770" \
  -e SHOPPING_ADMIN="http://your-server:7780/admin" \
  -e REDDIT="http://your-server:9999" \
  -e GITLAB="http://your-server:8023" \
  -e MAP="http://your-server:3000" \
  -e WIKIPEDIA="http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
  -e HOMEPAGE="http://your-server:4399" \
  webarena-env:latest
```

## Environment Details

### Action

Actions in WebArena are specified using a string format:

```python
from envs.webarena_env import WebArenaAction

# Click on an element (identified by ID in the accessibility tree)
action = WebArenaAction(action_str="click [123]")

# Type text into an element
action = WebArenaAction(action_str="type [45] [hello world]")

# Navigate to a URL
action = WebArenaAction(action_str="goto [http://example.com]")

# Scroll the page
action = WebArenaAction(action_str="scroll [down]")

# Select an option from a dropdown
action = WebArenaAction(action_str="select_option [67] [Option 1]")

# Go back
action = WebArenaAction(action_str="go_back []")

# Go forward
action = WebArenaAction(action_str="go_forward []")

# Stop the episode
action = WebArenaAction(action_str="stop []")
```

### Observation

Observations contain the current state of the web page:

```python
result = env.step(action)
obs = result.observation

# Text representation (accessibility tree or HTML)
print(obs.text)

# Current URL
print(obs.url)

# Action execution status
print(obs.success)  # True if action succeeded
print(obs.fail_error)  # Error message if failed

# Episode status
print(obs.done)  # True if episode ended
print(obs.reward)  # Reward for the action
```

### State

The environment state tracks progress through a task:

```python
state = env.state()

print(f"Episode ID: {state.episode_id}")
print(f"Steps: {state.step_count}")
print(f"Task: {state.task_id}")
print(f"Intent: {state.intent}")
print(f"URL: {state.current_url}")
print(f"Terminated: {state.terminated}")
```

## Configuration

Environment variables:

- `WEBARENA_HEADLESS`: Run browser in headless mode (default: `true`)
- `WEBARENA_OBSERVATION_TYPE`: Observation type - `accessibility_tree` or `html` (default: `accessibility_tree`)
- `WEBARENA_VIEWPORT_WIDTH`: Browser viewport width (default: `1280`)
- `WEBARENA_VIEWPORT_HEIGHT`: Browser viewport height (default: `720`)
- `WEBARENA_CONFIG_DIR`: Directory containing task config files (default: `/app/config_files`)

Required website URLs:
- `SHOPPING`: Shopping website URL
- `SHOPPING_ADMIN`: Shopping admin panel URL
- `REDDIT`: Reddit-like forum URL
- `GITLAB`: GitLab instance URL
- `MAP`: Map service URL
- `WIKIPEDIA`: Wikipedia instance URL
- `HOMEPAGE`: Homepage URL

## Advanced Usage

### Working with Tasks

WebArena uses JSON configuration files to define tasks. Each task specifies:
- Starting URL
- Task intent/instruction
- Evaluation criteria

Example config file structure:
```json
{
  "task_id": "1",
  "start_url": "http://example.com",
  "intent": "Find and purchase the cheapest laptop",
  "require_login": true,
  "storage_state": ".auth/shopping_state.json"
}
```

Mount a config directory when running the container:
```bash
docker run -p 8000:8000 \
  -v /path/to/config_files:/app/config_files \
  -e SHOPPING="..." \
  webarena-env:latest
```

### Extracting Element IDs

The accessibility tree observation shows element IDs in brackets:

```
[4] RootWebArea 'Example Page'
    [12] link 'Home'
    [28] button 'Submit'
    [45] textbox 'Search' required: False
```

To interact with an element, use its ID:
```python
action = WebArenaAction(action_str="click [28]")  # Click the Submit button
action = WebArenaAction(action_str="type [45] [search query]")  # Type in the textbox
```

### Observation Types

#### Accessibility Tree (Recommended)
- Structured representation of the page
- Shows interactive elements with IDs
- More compact than HTML
- Better for action grounding

#### HTML
- Full HTML source of the page
- More detailed but larger
- Useful for understanding page structure

## Development & Testing

### Running Tests

```bash
# From the OpenEnv repository root
pytest tests/envs/test_webarena_env.py
```

### Local Development

```bash
# Install in development mode
cd /home/hamidnazeri/OpenEnv
pip install -e .

# Run the server locally
cd src/envs/webarena_env/server
export WEBARENA_PATH=/home/hamidnazeri/webarena
export SHOPPING="http://your-server:7770"
# ... set other environment variables
python app.py
```

## Project Structure

```
webarena_env/
├── __init__.py              # Module exports
├── models.py                # Action, Observation, State dataclasses
├── client.py                # HTTPEnvClient implementation
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── webarena_environment.py  # Environment implementation
    ├── Dockerfile           # Container specification
    └── requirements.txt     # Python dependencies
```

## References

- [WebArena Paper](https://arxiv.org/abs/2307.13854)
- [WebArena GitHub](https://github.com/web-arena-x/webarena)
- [WebArena Website](https://webarena.dev/)
- [OpenEnv Documentation](https://github.com/openenv/openenv)

## Citation

If you use WebArena in your research, please cite:

```bibtex
@article{zhou2023webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and Xu, Frank F and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Bisk, Yonatan and Fried, Daniel and Alon, Uri and others},
  journal={arXiv preprint arXiv:2307.13854},
  year={2023}
}
```

## License

WebArena is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/web-arena-x/webarena/blob/main/LICENSE) file for details.
