"""WebArena Environment for OpenEnv.

WebArena is a realistic web environment for building autonomous agents that
can interact with web interfaces. It provides a browser-based environment
where agents can perform tasks on websites including shopping, forums,
content management systems, and more.

Key features:
- Browser-based interaction using Playwright
- Support for accessibility tree and HTML observations
- Various action types (click, type, navigate, scroll, etc.)
- Configurable tasks with evaluation metrics
- Realistic web environments for testing agents

Example:
    ```python
    from envs.webarena_env import WebArenaEnv, WebArenaAction

    # Create environment
    env = WebArenaEnv.from_docker_image("webarena-env:latest")

    # Reset and interact
    result = env.reset()
    action = WebArenaAction(action_str="click [123]")
    result = env.step(action)

    # Clean up
    env.close()
    ```
"""

from .client import WebArenaEnv
from .models import WebArenaAction, WebArenaObservation, WebArenaState

__all__ = [
    "WebArenaEnv",
    "WebArenaAction",
    "WebArenaObservation",
    "WebArenaState",
]
