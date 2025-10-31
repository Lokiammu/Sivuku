# Testing Guide for BrowserGym Environment

This guide provides comprehensive instructions for testing the BrowserGym integration in OpenEnv.

## Quick Start

```bash
# From repository root
cd /home/hamidnazeri/OpenEnv

# Run manual tests (no dependencies needed for basic tests)
python test_browsergym_manual.py

# Run automated tests with pytest
pytest tests/envs/test_browsergym_models.py -v
pytest tests/envs/test_browsergym_environment.py -v
```

## Test Levels

### Level 1: Model Tests (No Dependencies)

These tests verify that the data models work correctly and require no external dependencies.

```bash
pytest tests/envs/test_browsergym_models.py -v
```

**What it tests:**
- ✅ BrowserGymAction creation and metadata
- ✅ BrowserGymObservation with all fields
- ✅ BrowserGymState for different benchmarks
- ✅ Default values and error handling

**Requirements:** None (pure Python)

### Level 2: Local Environment Tests (Requires BrowserGym)

These tests verify that the environment wrapper works correctly with BrowserGym installed locally.

```bash
# Install dependencies first
pip install browsergym browsergym-miniwob playwright
playwright install chromium

# Run manual test
python test_browsergym_manual.py

# Or run pytest tests (requires running server)
pytest tests/envs/test_browsergym_environment.py -v
```

**What it tests:**
- ✅ Environment creation and initialization
- ✅ Reset functionality
- ✅ Step execution
- ✅ State tracking
- ✅ Error handling
- ✅ Multiple episodes

**Requirements:**
- `browsergym>=0.2.0`
- `browsergym-miniwob>=0.2.0`
- `playwright>=1.40.0`
- Chromium browser

### Level 3: HTTP Server Tests (Integration Tests)

These tests verify the full HTTP server integration including FastAPI endpoints and client-server communication.

```bash
# Run integration tests
pytest tests/envs/test_browsergym_environment.py -v
```

**What it tests:**
- ✅ Server startup and health checks
- ✅ HTTP endpoints (/reset, /step, /state, /health)
- ✅ Client-server communication
- ✅ Request/response serialization
- ✅ Concurrent requests
- ✅ Error propagation

**Requirements:**
- All Level 2 requirements
- `gunicorn` (for server)
- `uvicorn[standard]` (worker)

### Level 4: Docker Tests (Full Integration)

These tests verify the complete Docker containerization and deployment.

```bash
# Build the base image
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build the BrowserGym environment image
docker build -t browsergym-env:latest -f src/envs/browsergym_env/server/Dockerfile .

# Test MiniWoB (no backend needed)
docker run -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="miniwob" \
  -e BROWSERGYM_TASK_NAME="click-test" \
  browsergym-env:latest

# In another terminal, test the running server
python -c "
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction
env = BrowserGymEnv(base_url='http://localhost:8000')
result = env.reset()
print(f'Goal: {result.observation.goal}')
action = BrowserGymAction(action_str=\"click('button')\")
result = env.step(action)
print(f'Reward: {result.reward}')
env.close()
"
```

**What it tests:**
- ✅ Docker image builds successfully
- ✅ Container starts and runs
- ✅ Health checks work
- ✅ MiniWoB tasks work in container
- ✅ Client can connect from host

**Requirements:**
- Docker installed
- Sufficient memory (2GB+ recommended)

## Manual Testing Scenarios

### Scenario 1: Basic MiniWoB Task

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment
env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
    }
)

# Test reset
result = env.reset()
print(f"✅ Reset works: {result.observation.goal}")

# Test step
action = BrowserGymAction(action_str="click('Submit')")
result = env.step(action)
print(f"✅ Step works: reward={result.reward}, done={result.done}")

# Test state
state = env.state()
print(f"✅ State works: {state.step_count} steps, benchmark={state.benchmark}")

env.close()
print("✅ All basic tests passed!")
```

### Scenario 2: Multiple Task Types

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

tasks = ["click-test", "click-button", "enter-text", "click-checkboxes"]

for task in tasks:
    print(f"\nTesting task: {task}")
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        environment={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": task,
        }
    )

    result = env.reset()
    print(f"  Goal: {result.observation.goal[:50]}...")

    # Take a generic action
    action = BrowserGymAction(action_str="noop()")
    result = env.step(action)
    print(f"  Step reward: {result.reward}")

    env.close()
    print(f"  ✅ {task} works!")
```

### Scenario 3: Error Handling

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
    }
)

env.reset()

# Test invalid action
action = BrowserGymAction(action_str="invalid_syntax_here")
result = env.step(action)
print(f"Error handled: {result.observation.error or 'No error'}")
print(f"Observation still returned: {len(result.observation.text)} chars")

env.close()
print("✅ Error handling works!")
```

### Scenario 4: Multiple Episodes

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    environment={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
    }
)

episode_ids = []

for episode in range(3):
    result = env.reset()
    state = env.state()
    episode_ids.append(state.episode_id)
    print(f"Episode {episode + 1}: {state.episode_id}")

# Verify all episodes have unique IDs
assert len(set(episode_ids)) == 3, "Episode IDs should be unique!"
print("✅ Multiple episodes work!")

env.close()
```

## Common Issues and Solutions

### Issue 1: "BrowserGym not installed"

**Error:**
```
ImportError: No module named 'browsergym'
```

**Solution:**
```bash
pip install browsergym browsergym-miniwob
```

### Issue 2: "Playwright browsers not installed"

**Error:**
```
playwright._impl._api_types.Error: Executable doesn't exist at ...
```

**Solution:**
```bash
playwright install chromium
```

### Issue 3: "Server not starting"

**Error:**
```
Server did not become healthy in time
```

**Solutions:**
1. Check if port is already in use: `lsof -i :8000`
2. Check server logs for errors
3. Verify BrowserGym is installed: `pip list | grep browsergym`
4. Try running server manually:
   ```bash
   cd src/envs/browsergym_env/server
   BROWSERGYM_BENCHMARK=miniwob python app.py
   ```

### Issue 4: "Docker image build fails"

**Error:**
```
ERROR: failed to solve: process "/bin/sh -c playwright install chromium" did not complete successfully
```

**Solutions:**
1. Increase Docker memory limit (2GB minimum)
2. Check internet connection (downloads browser binaries)
3. Try building with `--progress=plain` to see detailed logs:
   ```bash
   docker build --progress=plain -t browsergym-env:latest -f src/envs/browsergym_env/server/Dockerfile .
   ```

## Performance Benchmarks

Expected performance on a standard machine (2022+ laptop, 8GB RAM):

| Operation | Time | Notes |
|-----------|------|-------|
| Import models | <100ms | Pure Python |
| Create environment | 2-5s | First time (browser startup) |
| Reset episode | 500ms-2s | Loads page |
| Execute step | 200ms-1s | Depends on action |
| State query | <50ms | Local operation |
| Close environment | <500ms | Cleanup |

## CI/CD Testing

For GitHub Actions or CI/CD pipelines:

```yaml
- name: Install dependencies
  run: |
    pip install -e .
    pip install browsergym browsergym-miniwob playwright pytest
    playwright install chromium

- name: Run model tests
  run: pytest tests/envs/test_browsergym_models.py -v

- name: Run environment tests
  run: pytest tests/envs/test_browsergym_environment.py -v
  env:
    BROWSERGYM_HEADLESS: "true"

- name: Run manual tests
  run: python test_browsergym_manual.py
```

## Test Coverage Goals

- ✅ Model creation and validation: **100%**
- ✅ Environment initialization: **100%**
- ✅ Reset functionality: **100%**
- ✅ Step execution: **95%** (some edge cases may vary by BrowserGym version)
- ✅ State tracking: **100%**
- ✅ Error handling: **90%** (depends on BrowserGym error types)
- ✅ HTTP endpoints: **100%**
- ✅ Client-server communication: **100%**

## Next Steps

After verifying these tests pass:

1. **Add to CI/CD**: Include tests in `.github/workflows/`
2. **Performance Testing**: Measure latency and throughput
3. **WebArena Testing**: Add tests for WebArena tasks (requires backend setup)
4. **Visual Testing**: Add tests for screenshot observations
5. **Multi-Benchmark**: Test with VisualWebArena and WorkArena

## Resources

- [BrowserGym Documentation](https://github.com/ServiceNow/BrowserGym)
- [MiniWoB++ Tasks](https://github.com/Farama-Foundation/miniwob-plusplus)
- [Playwright Documentation](https://playwright.dev/python/)
- [OpenEnv Testing Guide](../../tests/README.md)
