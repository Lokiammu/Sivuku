"""FastAPI server for the WebArena environment."""

import os

from core.env_server.http_server import create_app
from envs.webarena_env.models import (
    WebArenaAction,
    WebArenaObservation,
)
from envs.webarena_env.server.webarena_environment import WebArenaEnvironment

# Get configuration from environment variables
config_dir = os.environ.get("WEBARENA_CONFIG_DIR", "/app/config_files")
headless = os.environ.get("WEBARENA_HEADLESS", "true").lower() == "true"
observation_type = os.environ.get("WEBARENA_OBSERVATION_TYPE", "accessibility_tree")
viewport_width = int(os.environ.get("WEBARENA_VIEWPORT_WIDTH", "1280"))
viewport_height = int(os.environ.get("WEBARENA_VIEWPORT_HEIGHT", "720"))

# Create the environment instance
env = WebArenaEnvironment(
    config_dir=config_dir,
    headless=headless,
    observation_type=observation_type,
    viewport_width=viewport_width,
    viewport_height=viewport_height,
)

# Create the FastAPI app
app = create_app(
    env,
    WebArenaAction,
    WebArenaObservation,
    env_name="webarena_env",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
