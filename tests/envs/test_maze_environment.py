"""Unit tests for OpenSpiel environment server."""

import os
import shutil
import sys
import subprocess
import socket
import time
import requests
import pytest

# Add the project root to the path for envs imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from envs.maze_env.client import MazeEnv
from envs.maze_env.models import MazeAction

# Skip all tests if gunicorn is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("gunicorn") is None, reason="gunicorn not installed"
)


@pytest.fixture(scope="module")
def server():
    """Starts the Maze environment server as a background process."""
    # Define paths for subprocess environment
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    SRC_PATH = os.path.join(ROOT_DIR, "src")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        PORT = sock.getsockname()[1]
    localhost = f"http://localhost:{PORT}"

    print(f"\n--- Starting Maze server on port {PORT} ---")

    server_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([ROOT_DIR, SRC_PATH]),
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
        "BROWSERGYM_HEADLESS": "true",
    }

    gunicorn_command = [
        "gunicorn",
        "-w",
        "1",  # Single worker for testing
        "-k",
        "uvicorn.workers.UvicornWorker",
        "-b",
        f"0.0.0.0:{PORT}",
        "envs.maze_env.server.app:app",
    ]

    server_process = subprocess.Popen(
        gunicorn_command,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to become healthy
    print("\n--- Waiting for server to become healthy... ---")
    is_healthy = False
    for i in range(12):
        try:
            response = requests.get(f"{localhost}/health", timeout=5)
            if response.status_code == 200:
                is_healthy = True
                print("✅ Server is running and healthy!")
                break
        except requests.exceptions.RequestException:
            print(f"Attempt {i + 1}/12: Server not ready, waiting 10 seconds...")
            time.sleep(10)

    if not is_healthy:
        print("❌ Server did not become healthy in time. Aborting.")
        print("\n--- Server Logs ---")
        stdout, stderr = server_process.communicate(timeout=5)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        try:
            server_process.kill()
        except ProcessLookupError:
            # The process is already dead; nothing to clean up.
            pass
        pytest.skip("Server failed to start - OpenSpiel may not be installed")

    yield localhost

    # Cleanup
    print("\n--- Cleaning up server ---")
    try:
        server_process.kill()
        print("✅ Server process killed")
    except ProcessLookupError:
        print("✅ Server process was already killed")


def test_health_endpoint(server):
    """Test that the health endpoint works."""
    response = requests.get(f"{server}/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_reset(server):
    """Test that reset() returns a valid observation."""
    env = MazeEnv(base_url=server)
    result = env.reset()

    assert result.observation is not None
    assert hasattr(result.observation, "legal_actions")
    assert hasattr(result.observation, "current_position")
    assert hasattr(result.observation, "previous_position")
    assert result.observation.done is False
    env.close()


def test_reset_multiple_times(server):
    """Test that reset() can be called multiple times."""
    env = MazeEnv(base_url=server)

    result1 = env.reset()
    result2 = env.reset()

    # Both should be valid observations
    assert result1.observation is not None
    assert result2.observation is not None

    # Episode IDs should be different (new episodes)
    state1 = env.state()
    env.reset()
    state2 = env.state()
    assert state1.episode_id != state2.episode_id
    env.close()


def test_step(server):
    """Test that step() returns a valid result."""
    env = MazeEnv(base_url=server)
    result = env.reset()

    # Take a simple action
    action = MazeAction(action=result.observation.legal_actions[0])
    result = env.step(action)

    assert result.observation is not None
    assert isinstance(result.reward, (int, float)) or result.reward is None
    assert isinstance(result.done, bool)
    env.close()


def test_step_multiple_times(server):
    """Test that step() can be called multiple times."""
    env = MazeEnv(base_url=server)
    env.reset()

    # Take multiple actions
    action1 = MazeAction(action=0)
    result1 = env.step(action1)

    action2 = MazeAction(action=1)
    result2 = env.step(action2)

    # Both should be valid
    assert result1.observation is not None
    assert result2.observation is not None
    env.close()


def test_state_endpoint(server):
    """Test that the state endpoint returns valid state."""
    env = MazeEnv(base_url=server)
    env.reset()

    state = env.state()

    assert state is not None
    assert hasattr(state, "current_position")
    assert hasattr(state, "exit_cell")
    assert hasattr(state, "status")
    env.close()


def test_step_count_increments(server):
    """Test that step count increments correctly."""
    env = MazeEnv(base_url=server)
    env.reset()

    state1 = env.state()
    assert state1.step_count == 0

    action = MazeAction(action=0)
    env.step(action)

    state2 = env.state()
    assert state2.step_count == 1

    env.step(action)

    state3 = env.state()
    assert state3.step_count == 2
    env.close()


def test_action_with_metadata(server):
    """Test that actions with metadata work."""
    env = MazeEnv(base_url=server)
    env.reset()

    action = MazeAction(action=0, metadata={"test": "value", "number": 42})
    result = env.step(action)

    assert result.observation is not None
    env.close()
