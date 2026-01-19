# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TB2 environment server implementation (Spaces-compatible local mode)."""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import tomllib
from openenv.core.env_server.interfaces import Environment


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from tbench2_env.models import Tbench2Action, Tbench2Observation, Tbench2State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import Tbench2Action, Tbench2Observation, Tbench2State

_CAMEL_IMPORT_ERROR: Exception | None = None


def _require_terminal_toolkit() -> Any:
    global _CAMEL_IMPORT_ERROR
    if _CAMEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "camel-ai (TerminalToolkit) is required for TB2. "
            "Install from PyPI or from the CAMEL repo."
        ) from _CAMEL_IMPORT_ERROR

    try:
        from camel.toolkits import TerminalToolkit
    except Exception as exc:  # pragma: no cover
        _CAMEL_IMPORT_ERROR = exc
        raise RuntimeError(
            "camel-ai (TerminalToolkit) is required for TB2. "
            "Install from PyPI or from the CAMEL repo."
        ) from exc

    return TerminalToolkit


def _download_tb2_repo(cache_dir: Path) -> Path:
    repo_url = os.getenv(
        "TB2_REPO_URL",
        "https://github.com/laude-institute/terminal-bench-2/archive/refs/heads/main.zip",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "terminal-bench-2.zip"

    if not archive_path.exists():
        urllib.request.urlretrieve(repo_url, archive_path)

    with zipfile.ZipFile(archive_path) as zf:
        root = zf.namelist()[0].split("/")[0]
        extract_dir = cache_dir / root
        if not extract_dir.exists():
            zf.extractall(cache_dir)

    return extract_dir


def _read_instruction(task_dir: Path) -> str:
    instruction_path = task_dir / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text(encoding="utf-8")
    return ""


def _read_timeout(task_dir: Path, fallback: float) -> float:
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return fallback
    try:
        data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    verifier = data.get("verifier", {})
    return float(verifier.get("timeout_sec", fallback))


class Tbench2Environment(Environment[Tbench2Action, Tbench2Observation, Tbench2State]):
    """OpenEnv wrapper around Terminal-Bench 2 tasks (local execution)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks_dir: str | None = None,
        output_dir: str | None = None,
        command_timeout_s: float = 20.0,
        safe_mode: bool = False,
    ) -> None:
        super().__init__()
        self.tasks_dir = tasks_dir or os.getenv("TB2_TASKS_DIR", "")
        self.output_dir = Path(output_dir or os.getenv("TB2_OUTPUT_DIR", "/tmp/tbench2_env_runs"))
        self.command_timeout_s = command_timeout_s
        self.safe_mode = safe_mode

        self._state = Tbench2State()
        self._task_dir: Path | None = None
        self._terminal_toolkit = None
        self._instruction = ""

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        del seed

        TerminalToolkit = _require_terminal_toolkit()

        task_id = kwargs.get("task_id") or kwargs.get("task_name")
        task_path = kwargs.get("task_path") or kwargs.get("path")

        task_dir = self._resolve_task_path(task_id, task_path)
        resolved_task_id = task_id or task_dir.name

        self._instruction = _read_instruction(task_dir)
        self._task_dir = task_dir

        trial_name = f"{resolved_task_id}.{episode_id or uuid4().hex}"
        session_logs_dir = self.output_dir / trial_name / "terminal_toolkit_session_logs"
        session_logs_dir.mkdir(parents=True, exist_ok=True)

        self._terminal_toolkit = TerminalToolkit(
            timeout=self.command_timeout_s,
            working_directory=str(task_dir),
            use_docker_backend=False,
            session_logs_dir=session_logs_dir,
            safe_mode=self.safe_mode,
        )

        self._state = Tbench2State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=resolved_task_id,
            task_path=str(task_dir),
            terminal_ready=True,
        )

        return Tbench2Observation(
            instruction=self._instruction,
            output="",
            success=True,
            error="",
            task_id=resolved_task_id,
            task_path=str(task_dir),
            session_id=None,
            action_type="reset",
            info={},
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: Tbench2Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Tbench2Observation:
        del timeout_s, kwargs

        if not isinstance(action, Tbench2Action):
            raise TypeError(f"Expected Tbench2Action, got {type(action)}")

        if self._terminal_toolkit is None or self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.last_command = action.command

        output = ""
        error = ""
        success = True
        reward = None
        done = False
        info: dict[str, Any] = {}
        session_id = action.session_id or "tb2-session"

        try:
            if action.action_type == "exec":
                output = self._terminal_toolkit.shell_exec(
                    command=action.command,
                    block=action.block,
                    id=session_id,
                )
            elif action.action_type == "write":
                self._ensure_session_id(action.session_id, action.action_type)
                output = self._terminal_toolkit.shell_write_to_process(
                    id=action.session_id,
                    command=action.command,
                )
            elif action.action_type == "view":
                self._ensure_session_id(action.session_id, action.action_type)
                output = self._terminal_toolkit.shell_view(id=action.session_id)
            elif action.action_type == "wait":
                self._ensure_session_id(action.session_id, action.action_type)
                wait_seconds = action.wait_seconds or 0.0
                output = self._terminal_toolkit.shell_wait(
                    id=action.session_id,
                    wait_seconds=wait_seconds,
                )
            elif action.action_type == "kill":
                self._ensure_session_id(action.session_id, action.action_type)
                self._terminal_toolkit.shell_kill_process(id=action.session_id)
                output = f"Killed session {action.session_id}"
            elif action.action_type == "write_file":
                self._terminal_toolkit.shell_write_content_to_file(
                    content=action.content,
                    file_path=action.file_path,
                )
                output = f"Wrote content to {action.file_path}"
            elif action.action_type == "evaluate":
                output, reward, info = self._evaluate_task()
                done = True
            elif action.action_type == "close":
                self.close()
                output = "Closed TB2 environment."
                done = True
            else:
                raise ValueError(f"Unsupported action_type: {action.action_type}")
        except Exception as exc:  # pragma: no cover
            success = False
            error = str(exc)

        self._state.last_output = output
        self._state.session_id = session_id

        return Tbench2Observation(
            instruction=self._instruction,
            output=output,
            success=success,
            error=error,
            task_id=self._state.task_id,
            task_path=self._state.task_path,
            session_id=session_id,
            action_type=action.action_type,
            info=info,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> Tbench2State:
        return self._state

    def close(self) -> None:
        self._terminal_toolkit = None
        self._task_dir = None
        self._instruction = ""

    def _resolve_task_path(
        self, task_id: str | None, task_path: str | None
    ) -> Path:
        if task_path:
            resolved = Path(task_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Task path not found: {resolved}")
            return resolved

        if not task_id:
            raise ValueError("Provide task_id or task_path to reset TB2 environment.")

        if not self.tasks_dir:
            cache_dir = Path(os.getenv("TB2_CACHE_DIR", str(self.output_dir / "repo_cache")))
            repo_dir = _download_tb2_repo(cache_dir)
            resolved = repo_dir / task_id
        else:
            resolved = Path(self.tasks_dir).expanduser().resolve() / task_id

        if not resolved.exists():
            raise FileNotFoundError(f"Task path not found: {resolved}")
        return resolved

    def _ensure_session_id(self, session_id: str | None, action_type: str) -> None:
        if not session_id:
            raise ValueError(f"session_id is required for action_type='{action_type}'")

    def _evaluate_task(self) -> tuple[str, float, dict[str, Any]]:
        if self._task_dir is None:
            raise RuntimeError("TB2 environment not initialized. Call reset() first.")
        if self._terminal_toolkit is None:
            raise RuntimeError("Terminal toolkit not initialized.")

        timeout_s = _read_timeout(self._task_dir, fallback=900.0)
        tests_dir = self._task_dir / "tests"
        cmd = (
            f"cd {self._task_dir} && "
            f"python -m pytest -q {tests_dir} -rA; "
            f"echo __TB2_EXIT_CODE__:$?"
        )
        output = self._terminal_toolkit.shell_exec(
            id="tb2-tests",
            command=cmd,
            block=True,
        )

        exit_code = 1
        marker = "__TB2_EXIT_CODE__"
        for line in output.splitlines()[::-1]:
            if marker in line:
                try:
                    exit_code = int(line.split(":", 1)[1].strip())
                except Exception:
                    exit_code = 1
                break

        reward = 1.0 if exit_code == 0 else 0.0
        info = {"tests_passed": exit_code == 0, "exit_code": exit_code}
        return output, reward, info
