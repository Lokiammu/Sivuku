"""
CodingEnv
---------
Client-side wrapper for the Coding environment server.
Talks HTTP to a single base_url exposing: /reset and /step.

- users instantiate CodingEnv with a base_url provided by the higher-level
  vector/orchestration layer.
- Environment authors ship the Docker image that serves the HTTP API.

(Seeds, episode IDs, request IDs, capabilities can be added later in the payloads.)
"""

from __future__ import annotations

from typing import Optional

from core.base_env_client import HTTPEnvClient
from core.types import StepResult

from ..models import CodeAction, CodeObservation


class CodingEnv(HTTPEnvClient[CodeAction, CodeObservation]):
    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 15.0,
    ):
        super().__init__(
            base_url=base_url,
            request_timeout_s=request_timeout_s,
        )

    # --- HTTPEnvClient abstract hooks ---

    def _step_payload(self, action: CodeAction) -> dict:
        # Shape expected by the server's /step endpoint under "action"
        return {
            "code": action.code,
        }

    def _parse_result(self, payload: dict) -> StepResult[CodeObservation]:
        # Expecting: { "observation": {...}, "reward": <float|null>, "done": <bool>, "info": {...} }
        obs = CodeObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )
