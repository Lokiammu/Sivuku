"""
core/runner_env.py
Minimal HTTP-based environment client.
- Talks to a single env worker exposing: POST /reset, POST /step

Future hooks (commented below) for:
- episode_id, seed on reset
- request_id on step
- custom headers (auth/trace)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

import requests

from .base import BaseEnv
from .types import StepResult

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")


class HTTPEnvClient(BaseEnv[ActT, ObsT], Generic[ActT, ObsT]):
    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 15.0,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = float(request_timeout_s)
        self._http = requests.Session()
        self._headers = default_headers or {}

    @abstractmethod
    def _step_payload(self, action: ActT) -> dict:
        """Convert an Action object to the JSON body expected by the env server."""
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, payload: dict) -> StepResult[ObsT]:
        """Convert a JSON response from the env server to StepResult[ObsT]."""
        raise NotImplementedError

    # ---------- BaseEnv ----------
    def reset(self) -> ObsT:
        body: Dict[str, Any] = {}
        # TODO: later:
        # body["seed"] = seed
        # body["episode_id"] = episode_id
        r = self._http.post(
            f"{self._base}/reset",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_result(r.json()).observation

    def step(self, action: ActT) -> StepResult[ObsT]:
        body: Dict[str, Any] = {
            "action": self._step_payload(action),
            "timeout_s": int(self._timeout),
        }
        # TODO: later:
        # body["request_id"] = str(uuid.uuid4())
        # body["episode_id"] = current_episode_id
        r = self._http.post(
            f"{self._base}/step",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_result(r.json())

    def close(self) -> None:
        # nothing to close; higher-level libraries own lifecycles of the endpoints
        pass
