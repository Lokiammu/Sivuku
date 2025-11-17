# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test 123 Environment HTTP Client.

This module provides the client for connecting to a Test 123 Environment server
over HTTP.
"""

from typing import Any, Dict

from openenv_core.client_types import StepResult
from openenv_core.env_server.types import State
from openenv_core.http_env_client import HTTPEnvClient

from .models import Test123Action, Test123Observation


class Test123Env(HTTPEnvClient[Test123Action, Test123Observation]):
    """
    HTTP client for the Test 123 Environment.

    This client connects to a Test123Environment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = Test123Env(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.echoed_message)
        >>>
        >>> # Send a message
        >>> result = client.step(Test123Action(message="Hello!"))
        >>> print(result.observation.echoed_message)
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = Test123Env.from_docker_image("test_123-env:latest")
        >>> result = client.reset()
        >>> result = client.step(Test123Action(message="Test"))
    """

    def _step_payload(self, action: Test123Action) -> Dict:
        """
        Convert Test123Action to JSON payload for step request.

        Args:
            action: Test123Action instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[Test123Observation]:
        """
        Parse server response into StepResult[Test123Observation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with Test123Observation
        """
        obs_data = payload.get("observation", {})
        observation = Test123Observation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
