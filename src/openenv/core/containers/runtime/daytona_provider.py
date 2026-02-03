# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Daytona container provider for running OpenEnv environments in Daytona cloud sandboxes.

Requires the ``daytona`` SDK: ``pip install daytona>=0.10``
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

from .providers import ContainerProvider

# Daytona sandboxes use Docker images for the filesystem but do NOT execute
# the image's CMD/ENTRYPOINT.  The provider discovers the server command from
# ``openenv.yaml`` inside the sandbox and starts it via ``process.exec``.


class DaytonaProvider(ContainerProvider):
    """
    Container provider that runs environments in Daytona cloud sandboxes.

    Daytona sandboxes use Docker images for the filesystem (installed
    packages, copied files) but do **not** automatically run the image's
    CMD.  The provider starts the server process via
    ``sandbox.process.exec`` after creation.

    The server command is resolved in order:

    1. Explicit ``cmd`` from the constructor or ``**kwargs``.
    2. Auto-discovered from ``openenv.yaml`` inside the sandbox.
    3. ``ValueError`` if neither is available.

    Example:
        >>> provider = DaytonaProvider(api_key="your-key")
        >>> base_url = provider.start_container("lovrepesut/openenv-connect4:latest")
        >>> provider.wait_for_ready(base_url)
        >>> # Use environment via base_url
        >>> provider.stop_container()
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        public: bool = False,
        resources: Optional[Any] = None,
        auto_stop_interval: int = 15,
        target: Optional[str] = None,
        on_snapshot_create_logs: Optional[Callable[[str], None]] = None,
        cmd: Optional[str] = None,
    ):
        """
        Args:
            api_key: Daytona API key. Falls back to ``DAYTONA_API_KEY`` env var.
            public: If True, the sandbox preview is publicly accessible.
            resources: Optional ``daytona.Resources`` instance for CPU/memory.
            auto_stop_interval: Minutes of inactivity before auto-stop (0 disables).
            target: Daytona target region (e.g. "us").
            on_snapshot_create_logs: Callback for snapshot build log lines.
            cmd: Shell command to start the server inside the sandbox.
                If ``None``, auto-discovered from ``openenv.yaml`` inside
                the sandbox after creation.
        """
        from daytona import Daytona, DaytonaConfig

        config_kwargs: Dict[str, Any] = {}
        resolved_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if resolved_key:
            config_kwargs["api_key"] = resolved_key
        if target:
            config_kwargs["target"] = target

        self._daytona = Daytona(DaytonaConfig(**config_kwargs))
        self._public = public
        self._resources = resources
        self._auto_stop_interval = auto_stop_interval
        self._on_snapshot_create_logs = on_snapshot_create_logs
        self._cmd = cmd
        self._sandbox: Any = None
        self._preview_url: Optional[str] = None
        self._preview_token: Optional[str] = None

    def _discover_server_cmd(self, sandbox: Any, port: int = 8000) -> str:
        """Discover the server command from ``openenv.yaml`` inside *sandbox*.

        Finds the file, reads the ``app`` field, and constructs a command
        of the form ``cd <env_root> && uvicorn <app> --host 0.0.0.0 --port <port>``.

        Raises:
            ValueError: If ``openenv.yaml`` is not found or lacks an ``app`` field.
        """
        yaml_path = self._find_openenv_yaml(sandbox)
        if yaml_path is None:
            raise ValueError(
                "Could not find openenv.yaml inside the sandbox. "
                "Pass an explicit cmd= to DaytonaProvider or start_container()."
            )

        cat_resp = sandbox.process.exec(f"cat {yaml_path}", timeout=10)
        content = cat_resp.result if hasattr(cat_resp, "result") else str(cat_resp)
        app = self._parse_app_field(content)
        if app is None:
            raise ValueError(
                f"openenv.yaml at {yaml_path} does not contain an 'app' field. "
                "Pass an explicit cmd= to DaytonaProvider or start_container()."
            )

        # The directory containing openenv.yaml is the env root
        env_root = yaml_path.rsplit("/", 1)[0]
        return f"cd {env_root} && uvicorn {app} --host 0.0.0.0 --port {port}"

    def _find_openenv_yaml(self, sandbox: Any) -> Optional[str]:
        """Locate ``openenv.yaml`` inside the sandbox.

        Tries the modern layout path ``/app/env/openenv.yaml`` first,
        then falls back to a ``find`` command for the old layout.
        """
        # Fast path: modern Dockerfile layout
        resp = sandbox.process.exec(
            "test -f /app/env/openenv.yaml && echo found", timeout=10
        )
        out = resp.result if hasattr(resp, "result") else str(resp)
        if "found" in (out or ""):
            return "/app/env/openenv.yaml"

        # Fallback: search for it (redirect stderr so error messages
        # like "No such file or directory" don't get mistaken for paths).
        resp = sandbox.process.exec(
            "find /app -name openenv.yaml -maxdepth 4 -print -quit 2>/dev/null",
            timeout=10,
        )
        path = (resp.result if hasattr(resp, "result") else str(resp) or "").strip()
        if path and path.startswith("/"):
            return path

        return None

    @staticmethod
    def _parse_app_field(yaml_content: str) -> Optional[str]:
        """Extract the ``app`` value from raw openenv.yaml content.

        Uses simple line-by-line parsing to avoid a PyYAML dependency.
        """
        for line in yaml_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("app:"):
                value = stripped[len("app:") :].strip()
                # Strip optional quotes
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                return value if value else None
        return None

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a Daytona sandbox from a Docker image or snapshot.

        After sandbox creation the provider starts the server process
        (Daytona sandboxes do not execute the image CMD automatically).

        The command is resolved in order:

        1. Explicit ``cmd`` passed to the constructor.
        2. ``cmd`` key in ``**kwargs`` (popped before forwarding).
        3. Auto-discovered from ``openenv.yaml`` inside the sandbox.

        Args:
            image: Docker image name (e.g. "echo-env:latest") or
                   "snapshot:<name>" to create from a pre-built snapshot.
            port: Must be ``None`` or ``8000``. Daytona exposes port 8000
                  via its preview proxy; other ports raise ``ValueError``.
            env_vars: Environment variables forwarded to the sandbox.
            **kwargs: ``cmd`` (str) to override the server command;
                remaining kwargs passed through to ``Daytona.create()``.

        Returns:
            HTTPS preview URL for the sandbox (base_url).
        """
        if port is not None and port != 8000:
            raise ValueError(
                f"DaytonaProvider only supports port 8000 (got {port}). "
                "The Daytona preview proxy routes to port 8000 inside the sandbox."
            )

        # Resolve the server command (may be None; discovery happens after
        # sandbox creation when we can inspect the filesystem).
        cmd = kwargs.pop("cmd", None) or self._cmd

        # Build creation params
        create_kwargs: Dict[str, Any] = {}
        if env_vars:
            create_kwargs["env_vars"] = env_vars
        if self._public:
            create_kwargs["public"] = True
        if self._auto_stop_interval != 15:
            create_kwargs["auto_stop_interval"] = self._auto_stop_interval

        if image.startswith("snapshot:"):
            from daytona import CreateSandboxFromSnapshotParams

            snapshot_name = image[len("snapshot:") :]
            params = CreateSandboxFromSnapshotParams(
                snapshot=snapshot_name, **create_kwargs
            )
        else:
            from daytona import CreateSandboxFromImageParams

            img_kwargs: Dict[str, Any] = {"image": image, **create_kwargs}
            if self._resources is not None:
                img_kwargs["resources"] = self._resources
            params = CreateSandboxFromImageParams(**img_kwargs)

        # Create sandbox
        extra: Dict[str, Any] = dict(kwargs)
        if self._on_snapshot_create_logs is not None:
            extra["on_snapshot_create_logs"] = self._on_snapshot_create_logs

        self._sandbox = self._daytona.create(params, **extra)

        # Discover server command from openenv.yaml if not explicitly set.
        if cmd is None:
            cmd = self._discover_server_cmd(self._sandbox)

        # Start the server process.  Daytona sandboxes do NOT run the
        # Docker image CMD, so we exec it ourselves in the background.
        # Wrap in bash -c so compound commands (cd ... && uvicorn ...)
        # are handled correctly by nohup.
        self._sandbox.process.exec(
            f'nohup bash -c "{cmd}" > /tmp/openenv-server.log 2>&1 &',
            timeout=10,
        )

        # Get preview link for port 8000
        preview_info = self._sandbox.get_preview_link(8000)
        self._preview_url = preview_info.url
        self._preview_token = preview_info.token

        return self._preview_url

    def stop_container(self) -> None:
        """Delete the Daytona sandbox."""
        if self._sandbox is None:
            return

        try:
            self._daytona.delete(self._sandbox)
        finally:
            self._sandbox = None
            self._preview_url = None
            self._preview_token = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """
        Poll the /health endpoint until the sandbox is ready.

        Uses a longer default timeout (120s) than Docker providers because
        Daytona sandboxes may have cold-start latency.

        Args:
            base_url: Preview URL returned by ``start_container()``.
            timeout_s: Maximum seconds to wait.

        Raises:
            TimeoutError: If the sandbox doesn't become ready in time.
        """
        import requests

        health_url = f"{base_url}/health"
        headers = self.get_connect_headers()

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0, headers=headers)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1.0)

        raise TimeoutError(
            f"Daytona sandbox at {base_url} did not become ready within {timeout_s}s"
        )

    def get_connect_headers(self) -> Dict[str, str]:
        """Return the Daytona preview token header when the sandbox is private."""
        if not self._public and self._preview_token:
            return {"x-daytona-preview-token": self._preview_token}
        return {}
