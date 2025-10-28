import os
import sys
import subprocess
import time
import requests

# --- Install gunicorn ---
subprocess.run([sys.executable, "-m", "pip", "install", "gunicorn"])

# --- Define Absolute Paths & Port ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(ROOT_DIR, "src")
DATASET_SOURCE_PATH = os.path.abspath("harmonic_reasoner_dataset_structured.jsonl")
PORT = 8009

# --- Download the dataset ---
subprocess.run([sys.executable, "scripts/download_dataset.py"])

# --- 0. Kill any old server processes ---
print(f"--- 0. Ensuring port {PORT} is free ---")
subprocess.run(f"kill -9 $(lsof -t -i:{PORT})", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
print("✅ Port is clear.\n")

# --- 1. Set up ---
print(f"--- 1. Setting up environment ---")
sys.path.insert(0, SRC_PATH)
print(f"✅ Setup complete. Current directory: {os.getcwd()}\n")

# --- 2. Set the Dataset Path Environment Variable ---
print(f"--- 2. Setting DIPG_DATASET_PATH environment variable ---")
if not os.path.exists(DATASET_SOURCE_PATH):
    print(f"❌ FATAL ERROR: Dataset not found at '{DATASET_SOURCE_PATH}'.")
    raise FileNotFoundError()

# --- 3. Launch the Server using Gunicorn ---
localhost = f"http://localhost:{PORT}"
print(f"--- 3. Starting DIPGSafetyEnv server with Gunicorn on port {PORT} ---")

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

# --- 4. Wait and Verify ---
print("\n--- 4. Waiting for server to become healthy... ---")
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

# --- 5. Connect Client ---
from envs.dipg_safety_env.client import DIPGSafetyEnv
from envs.dipg_safety_env.models import DIPGAction

print(f"\n--- 5. Connecting client to {localhost} ---")
env = DIPGSafetyEnv(base_url=localhost, timeout=300)
try:
    obs = env.reset()
    print("✅ Successfully connected to the live DIPGSafetyEnv!")
except Exception as e:
    print(f"❌ Error connecting to server: {e}")
    print("\n--- Server Logs ---")
    print(openenv_process.stderr.read())
    try:
        openenv_process.kill()
    except ProcessLookupError:
        pass
    sys.exit(1)


# --- 6. Simulate a call ---
print("\n--- 6. Simulating a call to the environment ---")
agent_response_text = "Based on the provided context, the information is conflicting."
action = DIPGAction(llm_response=agent_response_text)
result = env.step(action)
print(f"Reward: {result.reward}")
print(f"Done: {result.done}")

# --- 7. Clean up ---
print("\n--- 7. Cleaning up ---")
try:
    openenv_process.kill()
    print("✅ Server process killed.")
except ProcessLookupError:
    print("✅ Server process was already killed.")
")