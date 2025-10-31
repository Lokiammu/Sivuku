"""HTTP client for the WebArena environment."""

from typing import Any, Dict

from core.http_env_client import HTTPEnvClient, StepResult
from envs.webarena_env.models import (
    WebArenaAction,
    WebArenaObservation,
    WebArenaState,
)


class WebArenaEnv(HTTPEnvClient[WebArenaAction, WebArenaObservation]):
    """Client for interacting with the WebArena environment over HTTP.

    Example usage:
        ```python
        from envs.webarena_env import WebArenaEnv, WebArenaAction

        # Create environment from Docker image
        env = WebArenaEnv.from_docker_image("webarena-env:latest")

        # Reset the environment with a specific task config
        result = env.reset()
        print(f"Initial observation: {result.observation.text[:100]}")

        # Take actions
        action = WebArenaAction(action_str="click [123]")
        result = env.step(action)
        print(f"After click: {result.observation.url}")

        # Navigate to a URL
        action = WebArenaAction(action_str="goto [http://example.com]")
        result = env.step(action)

        # Type text
        action = WebArenaAction(action_str="type [45] [hello world]")
        result = env.step(action)

        # Stop the episode
        action = WebArenaAction(action_str="stop []")
        result = env.step(action)
        assert result.done

        # Get current state
        state = env.state()
        print(f"Task: {state.intent}")
        print(f"Steps: {state.step_count}")

        # Clean up
        env.close()
        ```
    """

    def _step_payload(self, action: WebArenaAction) -> Dict[str, Any]:
        """Convert a WebArenaAction to the JSON payload for the server."""
        return {
            "action_str": action.action_str,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[WebArenaObservation]:
        """Parse the server response into a StepResult."""
        obs_data = payload.get("observation", {})

        observation = WebArenaObservation(
            text=obs_data.get("text", ""),
            url=obs_data.get("url", ""),
            success=obs_data.get("success", True),
            fail_error=obs_data.get("fail_error", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
            observation_metadata=obs_data.get("observation_metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> WebArenaState:
        """Parse the server state response into a WebArenaState object."""
        return WebArenaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            config_file=payload.get("config_file"),
            task_id=payload.get("task_id"),
            intent=payload.get("intent", ""),
            current_url=payload.get("current_url", ""),
            terminated=payload.get("terminated", False),
        )
