
# BrowserGym Environment

BrowserGym is a unified framework for web-based agent tasks that provides access to multiple benchmarks under a single Gymnasium-compatible API. This integration brings the complete training-to-evaluation pipeline for web agents into OpenEnv.

## Why BrowserGym?

**Complete Pipeline**: Train on MiniWoB++ → Evaluate on WebArena/VisualWebArena
- **MiniWoB++**: 100+ simple tasks for training (works immediately, no setup!)
- **WebArena**: 812 realistic tasks for evaluation (requires backend setup)
- **VisualWebArena**: Visual navigation tasks
- **WorkArena**: Enterprise automation tasks

**Key Advantage**: MiniWoB tasks work out-of-the-box with no external infrastructure needed!

## Quick Start - Training (MiniWoB)

### No Setup Required! 🎉

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment for MiniWoB training task
env = BrowserGymEnv.from_docker_image(
    "ghcr.io/openenv/browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",  # or "click-button", "click-dialog", etc.
    }
)

# Train your agent!
for episode in range(1000):
    result = env.reset()
    print(f"Goal: {result.observation.goal}")

    done = False
    while not done:
        # Your agent decides what to do
        action_str = agent.get_action(result.observation.text)
        action = BrowserGymAction(action_str=action_str)

        result = env.step(action)
        done = result.done

        print(f"Reward: {result.reward}")

env.close()
```

### Available MiniWoB Tasks (100+ total)

Popular training tasks:
- **click-test**: Click on a specific button
- **click-button**: Click buttons with specific text
- **click-dialog**: Click buttons in dialogs
- **click-checkboxes**: Select specific checkboxes
- **click-link**: Click on links
- **enter-text**: Type text into input fields
- **navigate-tree**: Navigate through tree structures
- **search-engine**: Use a search interface
- **use-autocomplete**: Interact with autocomplete
- **login-user**: Fill in login forms

And many more! See [MiniWoB++ documentation](https://github.com/Farama-Foundation/miniwob-plusplus) for the full list.

## Evaluation (WebArena)

### Prerequisites

WebArena requires setting up backend infrastructure. See the [WebArena documentation](https://github.com/web-arena-x/webarena/tree/main/environment_docker).

### Usage

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment for WebArena evaluation
env = BrowserGymEnv.from_docker_image(
    "ghcr.io/openenv/browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",  # Task ID
        # WebArena backend URLs (required)
        "SHOPPING": "http://your-server:7770",
        "SHOPPING_ADMIN": "http://your-server:7780/admin",
        "REDDIT": "http://your-server:9999",
        "GITLAB": "http://your-server:8023",
        "MAP": "http://your-server:3000",
        "WIKIPEDIA": "http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "HOMEPAGE": "http://your-server:4399",
    }
)

# Evaluate your trained agent
result = env.reset()
while not result.done:
    action_str = agent.get_action(result.observation)
    action = BrowserGymAction(action_str=action_str)
    result = env.step(action)

print(f"Success: {result.reward}")
env.close()
```

## Building the Docker Image

### Prerequisites

1. **Base Image**: Build the OpenEnv base image first:

```bash
# From the OpenEnv repository root
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
```

### Build the BrowserGym Environment

```bash
# From the OpenEnv repository root
docker build -t browsergym-env:latest -f src/envs/browsergym_env/server/Dockerfile .
```

### Run the Server

#### For MiniWoB (Training):

```bash
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="miniwob" \
  -e BROWSERGYM_TASK_NAME="click-test" \
  browsergym-env:latest
```

#### For WebArena (Evaluation):

```bash
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="webarena" \
  -e BROWSERGYM_TASK_NAME="0" \
  -e SHOPPING="http://your-server:7770" \
  -e SHOPPING_ADMIN="http://your-server:7780/admin" \
  -e REDDIT="http://your-server:9999" \
  -e GITLAB="http://your-server:8023" \
  -e MAP="http://your-server:3000" \
  -e WIKIPEDIA="http://your-server:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
  -e HOMEPAGE="http://your-server:4399" \
  browsergym-env:latest
```

## Environment Details

### Action

Actions in BrowserGym are natural language strings that describe browser operations:

```python
from envs.browsergym_env import BrowserGymAction

# Click actions
action = BrowserGymAction(action_str="click('Submit button')")
action = BrowserGymAction(action_str="click('element_id_123')")

# Type actions
action = BrowserGymAction(action_str="fill('username', 'john@example.com')")
action = BrowserGymAction(action_str="fill('password', 'secret123')")

# Navigate actions
action = BrowserGymAction(action_str="goto('https://example.com')")

# Keyboard actions
action = BrowserGymAction(action_str="press('Enter')")
action = BrowserGymAction(action_str="press('Tab')")

# Scroll actions
action = BrowserGymAction(action_str="scroll('down')")
```

### Observation

Observations contain multiple modalities:

```python
result = env.step(action)
obs = result.observation

# Text observations
print(obs.text)          # Primary text representation (AXTree or DOM)
print(obs.axtree_txt)    # Accessibility tree
print(obs.pruned_html)   # Pruned HTML (interactive elements only)

# Page metadata
print(obs.url)           # Current URL
print(obs.goal)          # Task goal/instruction

# Visual (if enabled)
if obs.screenshot is not None:
    print(obs.screenshot.shape)  # [height, width, channels]

# Error handling
if obs.last_action_error:
    print(f"Action failed: {obs.error}")

# Episode status
print(obs.done)          # True if episode ended
print(obs.reward)        # Reward for the step
```

### State

The environment state tracks progress:

```python
state = env.state()

print(f"Benchmark: {state.benchmark}")     # 'miniwob', 'webarena', etc.
print(f"Task: {state.task_name}")          # Task name/ID
print(f"Episode: {state.episode_id}")      # Unique episode ID
print(f"Steps: {state.step_count}")        # Number of steps taken
print(f"Total Reward: {state.cum_reward}") # Cumulative reward
print(f"Goal: {state.goal}")               # Task instruction
print(f"URL: {state.current_url}")         # Current page URL
```

## Configuration

Environment variables:

### Common Settings
- `BROWSERGYM_BENCHMARK`: Benchmark to use (`miniwob`, `webarena`, `visualwebarena`, `workarena`)
- `BROWSERGYM_TASK_NAME`: Specific task name (optional, will use first available if not set)
- `BROWSERGYM_HEADLESS`: Run browser in headless mode (default: `true`)
- `BROWSERGYM_VIEWPORT_WIDTH`: Browser viewport width (default: `1280`)
- `BROWSERGYM_VIEWPORT_HEIGHT`: Browser viewport height (default: `720`)
- `BROWSERGYM_TIMEOUT`: Action timeout in milliseconds (default: `10000`)

### WebArena-Specific (only needed for WebArena benchmark)
- `SHOPPING`: Shopping website URL
- `SHOPPING_ADMIN`: Shopping admin panel URL
- `REDDIT`: Reddit-like forum URL
- `GITLAB`: GitLab instance URL
- `MAP`: Map service URL
- `WIKIPEDIA`: Wikipedia instance URL
- `HOMEPAGE`: Homepage URL

## Supported Benchmarks

### 1. MiniWoB++ (Training) ✅ Recommended for Training

- **100+ tasks** ranging from simple (click buttons) to complex (form filling, navigation)
- **Fast**: Instant resets, quick episodes
- **Randomized**: Task variations for generalization
- **No setup**: Works out-of-the-box
- **Dense rewards**: Immediate feedback for learning

**Use Case**: Train agents on fundamental web navigation skills

### 2. WebArena (Evaluation) 📊 Benchmark

- **812 realistic tasks** across 6 websites
- **Complex**: Multi-step reasoning, real web interfaces
- **Requires setup**: Need to run 7 backend services
- **Sparse rewards**: Binary success/failure
- **Evaluation-focused**: Test real-world performance

**Use Case**: Evaluate agents on realistic web tasks

### 3. VisualWebArena (Evaluation) 👁️ Visual Benchmark

- **910 tasks** requiring visual understanding
- **Multimodal**: Both text and visual observations
- **Requires setup**: Similar to WebArena
- **Challenging**: Requires visual reasoning

**Use Case**: Test visual web navigation capabilities

### 4. WorkArena (Evaluation) 💼 Enterprise Benchmark

- **Enterprise tasks**: CRM, project management, etc.
- **Realistic workflows**: Real enterprise software
- **Requires setup**: Enterprise software instances

**Use Case**: Evaluate on business automation tasks

## Typical Training Pipeline

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Stage 1: Train on MiniWoB (simple tasks, fast)
train_env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-button",
    }
)

# Train your agent (RL, imitation learning, etc.)
agent.train(train_env, num_episodes=10000)
train_env.close()

# Stage 2: Evaluate on WebArena (complex tasks, realistic)
eval_env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",
        # ... WebArena URLs
    }
)

# Test performance
success_rate = agent.evaluate(eval_env, num_tasks=812)
print(f"WebArena Success Rate: {success_rate:.2%}")
eval_env.close()
```

## Development & Testing

### Running Tests

```bash
# From the OpenEnv repository root
pytest tests/envs/test_browsergym_env.py
```

### Local Development

```bash
# Install in development mode
cd /path/to/OpenEnv
pip install -e .

# Install BrowserGym
pip install browsergym browsergym-miniwob browsergym-webarena

# Run the server locally
cd src/envs/browsergym_env/server
export BROWSERGYM_BENCHMARK=miniwob
export BROWSERGYM_TASK_NAME=click-test
python app.py
```

## Project Structure

```
browsergym_env/
├── __init__.py              # Module exports
├── models.py                # Action, Observation, State dataclasses
├── client.py                # HTTPEnvClient implementation
├── README.md                # This file
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── browsergym_environment.py  # Environment implementation
    ├── Dockerfile           # Container specification
    └── requirements.txt     # Python dependencies
```

## References

- [BrowserGym GitHub](https://github.com/ServiceNow/BrowserGym)
- [MiniWoB++ Paper](https://arxiv.org/abs/1802.08802)
- [WebArena Paper](https://arxiv.org/abs/2307.13854)
- [WebArena Website](https://webarena.dev/)
- [VisualWebArena Paper](https://jykoh.com/vwa)
- [OpenEnv Documentation](https://github.com/openenv/openenv)
