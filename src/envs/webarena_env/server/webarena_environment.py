"""WebArena Environment implementation for OpenEnv.

This module wraps the WebArena browser environment to provide a compatible
interface with OpenEnv's Environment ABC.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Add the webarena directory to the path to import browser_env
WEBARENA_PATH = os.environ.get("WEBARENA_PATH", "/home/hamidnazeri/webarena")
if WEBARENA_PATH not in sys.path:
    sys.path.insert(0, WEBARENA_PATH)

from browser_env import ScriptBrowserEnv, create_id_based_action

from core.env_server.interfaces import Environment
from envs.webarena_env.models import (
    WebArenaAction,
    WebArenaObservation,
    WebArenaState,
)


class WebArenaEnvironment(Environment):
    """WebArena environment wrapper for OpenEnv.

    This environment wraps the WebArena browser environment to provide web-based
    task evaluation for autonomous agents.
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        headless: bool = True,
        observation_type: str = "accessibility_tree",
        viewport_width: int = 1280,
        viewport_height: int = 720,
    ):
        """Initialize the WebArena environment.

        Args:
            config_dir: Directory containing task config files (*.json)
            headless: Whether to run browser in headless mode
            observation_type: Type of observation ('accessibility_tree' or 'html')
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
        """
        super().__init__()

        self.config_dir = Path(config_dir) if config_dir else None
        self.headless = headless
        self.observation_type = observation_type
        self.viewport_size = {"width": viewport_width, "height": viewport_height}

        # Initialize the browser environment
        self.browser_env = ScriptBrowserEnv(
            headless=self.headless,
            observation_type=self.observation_type,
            current_viewport_only=True,
            viewport_size=self.viewport_size,
            slow_mo=100 if not headless else 0,
        )

        # State tracking
        self._state = WebArenaState(
            episode_id=str(uuid4()),
            step_count=0,
        )
        self._current_config_file: Optional[str] = None
        self._last_obs_text: str = ""
        self._last_url: str = ""

    def reset(
        self,
        config_file: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> WebArenaObservation:
        """Reset the environment with a specific task configuration.

        Args:
            config_file: Path to the task config JSON file
            task_id: Optional task ID (extracted from config if not provided)

        Returns:
            Initial observation for the task
        """
        # Generate new episode ID
        self._state = WebArenaState(
            episode_id=str(uuid4()),
            step_count=0,
        )

        # Determine config file to use
        if config_file:
            self._current_config_file = config_file
        elif self.config_dir and self.config_dir.exists():
            # Use the first config file in the directory if none specified
            config_files = list(self.config_dir.glob("*.json"))
            if config_files:
                self._current_config_file = str(config_files[0])

        # Load config to extract metadata
        if self._current_config_file:
            config_path = Path(self._current_config_file)
            if config_path.exists():
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    self._state.intent = config_data.get("intent", "")
                    self._state.task_id = (
                        task_id or config_data.get("task_id") or config_path.stem
                    )

        # Reset the browser environment
        reset_options = {}
        if self._current_config_file:
            reset_options["config_file"] = self._current_config_file

        obs, info = self.browser_env.reset(options=reset_options)

        # Extract observation details
        self._last_obs_text = obs.get("text", "")
        self._last_url = info.get("page", {}).url if hasattr(info.get("page", {}), "url") else ""
        self._state.current_url = self._last_url
        self._state.config_file = self._current_config_file
        self._state.terminated = False

        return WebArenaObservation(
            text=self._last_obs_text,
            url=self._last_url,
            success=True,
            done=False,
            reward=0.0,
            observation_metadata=info.get("observation_metadata", {}),
        )

    def step(self, action: WebArenaAction) -> WebArenaObservation:
        """Execute an action in the environment.

        Args:
            action: The action to execute

        Returns:
            Observation after executing the action
        """
        self._state.step_count += 1

        # Parse the action string and create WebArena action
        action_str = action.action_str.strip()

        # Check for termination action
        if action_str.startswith("stop"):
            self._state.terminated = True
            return WebArenaObservation(
                text=self._last_obs_text,
                url=self._last_url,
                success=True,
                done=True,
                reward=0.0,
            )

        # Execute the action in the browser environment
        try:
            # Create the action using WebArena's action creation
            browser_action = create_id_based_action(action_str)

            # Execute the action
            obs, reward, terminated, truncated, info = self.browser_env.step(
                browser_action
            )

            # Extract observation details
            self._last_obs_text = obs.get("text", "")
            page_info = info.get("page", {})
            self._last_url = page_info.url if hasattr(page_info, "url") else self._last_url
            self._state.current_url = self._last_url

            success = info.get("fail_error", "") == ""
            fail_error = info.get("fail_error", "")

            # Update state
            self._state.terminated = terminated or truncated

            return WebArenaObservation(
                text=self._last_obs_text,
                url=self._last_url,
                success=success,
                fail_error=fail_error,
                done=self._state.terminated,
                reward=float(reward),
                observation_metadata=info.get("observation_metadata", {}),
            )

        except Exception as e:
            # Handle action execution errors
            error_msg = str(e)
            return WebArenaObservation(
                text=self._last_obs_text,
                url=self._last_url,
                success=False,
                fail_error=error_msg,
                done=False,
                reward=0.0,
            )

    @property
    def state(self) -> WebArenaState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""
        if hasattr(self, "browser_env"):
            self.browser_env.close()
