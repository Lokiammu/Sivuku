# src/envs/dipg_safety_env/client.py
"""
Client implementation for the custom DIPGSafetyEnv.

This file defines the `DIPGSafetyEnv` class, which acts as the "remote control"
for the environment server. Its primary job is to handle the HTTP communication:
  1.  It takes Python objects (like an Action) from the agent's code.
  2.  It converts them into JSON to send to the server.
  3.  It receives JSON responses from the server.
  4.  It parses that JSON back into useful Python objects (like Observations and Rewards).
"""

from core.http_env_client import HTTPEnvClient, StepResult
from .models import DIPGAction, DIPGObservation, DIPGState


class DIPGSafetyEnv(HTTPEnvClient[DIPGAction, DIPGObservation]):
    """
    Client for interacting with the `DIPGSafetyEnv` server.

    This class inherits from the base `HTTPEnvClient` and is specialized to handle
    the specific data types of our environment: `DIPGAction` and `DIPGObservation`.
    """

    def _step_payload(self, action: DIPGAction) -> dict:
        """
        Formats the `DIPGAction` object into a JSON-serializable dictionary.
        
        This dictionary becomes the body of the HTTP POST request sent to the
        server's `/step` endpoint.

        Args:
            action: The `DIPGAction` object containing the model's response.

        Returns:
            A dictionary to be sent as the JSON request body.
        """
        return {"llm_response": action.llm_response}

    def _parse_result(self, payload: dict) -> StepResult[DIPGObservation]:
        """
        Parses the JSON payload from the server's response into a `StepResult`.

        This method contains critical logic to handle a known inconsistency between
        the data structures returned by the server's `/reset` and `/step` endpoints.

        Args:
            payload: The raw dictionary parsed from the server's JSON response.

        Returns:
            A structured `StepResult` object containing the observation, reward, and done status.
        """
        # The server's response contains an 'observation' key.
        obs_data = payload.get("observation", {})

        # ROBUSTNESS FIX: The server's /step endpoint returns a double-nested
        # observation `{'observation': {'observation': {...}}}` while the /reset
        # endpoint returns a single-nested one `{'observation': {...}}`.
        # This code checks for the double-nesting and handles both cases gracefully.
        if "observation" in obs_data:
            # If it's double-nested (from /step), go one level deeper.
            actual_obs_data = obs_data["observation"]
        else:
            # If it's single-nested (from /reset), use the data directly.
            actual_obs_data = obs_data
        
        # Create the DIPGObservation object from the correctly identified data.
        obs = DIPGObservation(**actual_obs_data)
        
        # Assemble the final StepResult object for the agent.
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DIPGState:
        """
        Parses the JSON payload from the server's `/state` endpoint into a `DIPGState` object.
        
        Args:
            payload: The raw dictionary parsed from the server's JSON response.
            
        Returns:
            A structured `DIPGState` object.
        """
        return DIPGState(**payload)