#tests/envs/test_dipg_environment.py
import os
import sys
import subprocess
import time
import requests
import pytest

from envs.dipg_safety_env.client import DIPGSafetyEnv
from envs.dipg_safety_env.models import DIPGAction


@pytest.fixture(scope="module")
def server():
    """Starts the environment server as a background process."""
    # --- Define Absolute Paths & Port ---
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SRC_PATH = os.path.join(ROOT_DIR, "src")
    DATASET_SOURCE_PATH = os.path.abspath("harmonic_reasoner_dataset_structured.jsonl")
    PORT = 8009

    # --- Download the dataset ---
    subprocess.run([sys.executable, "scripts/download_dataset.py"], check=True)

    # --- Launch the Server using Gunicorn ---
    localhost = f"http://localhost:{PORT}"
    print(f"--- Starting DIPGSafetyEnv server with Gunicorn on port {PORT} ---")

    server_env = {
        **os.environ,
        "PYTHONPATH": SRC_PATH,
        "DIPG_DATASET_PATH": DATASET_SOURCE_PATH,
    }

    gunicorn_command = [
        "gunicorn",
        "-w", "4",
        "-k", "uvicorn.workers.UvicornWorker",
        "-b", f"0.0.0.0:{PORT}",
        "envs.dipg_safety_env.server.app:app",
    ]
    openenv_process = subprocess.Popen(
        gunicorn_command,
        env=server_env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    # --- Wait and Verify ---
    print("\n--- Waiting for server to become healthy... ---")
    is_healthy = False
    for i in range(12):
        try:
            response = requests.get(f"{localhost}/health", timeout=5)
            if response.status_code == 200 and "healthy" in response.text:
                is_healthy = True
                print("✅ Server is running and healthy!")
                break
        except requests.exceptions.RequestException:
            print(f"Attempt {i+1}/12: Server not ready, waiting 10 seconds...")
            time.sleep(10)

    if not is_healthy:
        print("❌ Server did not become healthy in time. Aborting.")
        print("\n--- Server Logs ---")
        print(openenv_process.stderr.read())
        try:
            openenv_process.kill()
        except ProcessLookupError:
            pass
        raise RuntimeError("Server failed to start.")

    yield localhost

    # --- Clean up ---
    print("\n--- Cleaning up ---")
    try:
        openenv_process.kill()
        print("✅ Server process killed.")
    except ProcessLookupError:
        print("✅ Server process was already killed.")

def test_reset(server):
    """Test that reset() returns a valid observation."""
    env = DIPGSafetyEnv(base_url=server, timeout=300)
    obs1 = env.reset()
    obs2 = env.reset()
    assert obs1.question != obs2.question