"""Providers for launching Hugging Face Spaces via ``uv run``."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import requests

from .providers import ContainerProvider


def _poll_health(health_url: str, timeout_s: float) -> None:
    """Poll a health endpoint until it returns HTTP 200 or times out."""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            timeout = max(0.0001, min(deadline - time.time(), 2.0))
            response = requests.get(health_url, timeout=timeout)
            if response.status_code == 200:
                return
        except requests.RequestException:
            continue

        time.sleep(0.5)

    raise TimeoutError(f"Server did not become ready within {timeout_s:.1f} seconds")


def _create_uv_command(
    repo_id: str,
    host: str,
    port: int,
    reload: bool,
    project_url: Optional[str] = None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--isolated",
        "--with",
        project_url or f"git+https://huggingface.co/spaces/{repo_id}",
        "--",
        "server",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")
    return command


@dataclass
class UVProvider(ContainerProvider):
    """ContainerProvider implementation backed by ``uv run``."""

    repo_id: str
    host: str = "0.0.0.0"
    port: Optional[int] = None
    reload: bool = False
    project_url: Optional[str] = None
    connect_host: Optional[str] = None
    extra_env: Optional[Dict[str, str]] = None
    context_timeout_s: float = 60.0

    _process: subprocess.Popen | None = field(init=False, default=None)
    _base_url: str | None = field(init=False, default=None)

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **_: Dict[str, str],
    ) -> str:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("UVProvider is already running")

        self.repo_id = image or self.repo_id

        bind_port = port or self.port or self._find_free_port()

        command = _create_uv_command(
            self.repo_id,
            self.host,
            bind_port,
            self.reload,
            project_url=self.project_url,
        )

        env = os.environ.copy()
        if self.extra_env:
            env.update(self.extra_env)
        if env_vars:
            env.update(env_vars)

        try:
            self._process = subprocess.Popen(command, env=env)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`uv` executable not found. Install uv from "
                "https://github.com/astral-sh/uv and ensure it is on PATH."
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to launch `uv run`: {exc}") from exc

        client_host = self.connect_host or (
            "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        )
        self._base_url = f"http://{client_host}:{bind_port}"
        self.port = bind_port
        return self._base_url

    def wait_for_ready(self, base_url: str, timeout_s: float = 60.0) -> None:
        if self._process and self._process.poll() is not None:
            code = self._process.returncode
            raise RuntimeError(f"uv process exited prematurely with code {code}")

        _poll_health(f"{base_url}/health", timeout_s)

    def stop_container(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)

        self._process = None
        self._base_url = None

    def start(self) -> str:
        return self.start_container(self.repo_id, port=self.port)

    def stop(self) -> None:
        self.stop_container()

    def wait_for_ready_default(self, timeout_s: float | None = None) -> None:
        if self._base_url is None:
            raise RuntimeError("UVProvider has not been started")
        self.wait_for_ready(
            self._base_url,
            timeout_s or self.context_timeout_s,
        )

    def close(self) -> None:
        self.stop_container()

    def __enter__(self) -> "UVProvider":
        if self._base_url is None:
            base_url = self.start_container(self.repo_id, port=self.port)
            self.wait_for_ready(base_url, timeout_s=self.context_timeout_s)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_container()

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.listen(1)
            return sock.getsockname()[1]

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("UVProvider has not been started")
        return self._base_url
