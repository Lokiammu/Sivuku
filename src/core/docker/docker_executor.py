# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import time
from typing import Any

from ..env.types import ExecutionResult


class DockerExecutor:
    """Simple Docker-based Python code executor with persistent session."""

    def __init__(self, image: str = "python:3.11-slim", timeout_seconds: int = 30):
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.container_id: str | None = None
        self._process: subprocess.Popen | None = None

    def start_session(self) -> None:
        """Start new Docker container with persistent Python session."""
        if self.container_id:
            self.stop_session()

        # Run interactive Python in container
        cmd = [
            "docker", "run", "--rm", "-i",
            "--memory=512m",
            "--cpus=1.0",
            "--network=host",  # For MCP integration later
            self.image,
            "python", "-u", "-c",
            "import sys; [exec(input()) for _ in iter(int, 1)]"
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )

            # Get container ID for potential cleanup
            # This is a bit hacky but works for the simple approach
            time.sleep(0.1)  # Give container time to start

        except Exception as e:
            raise RuntimeError(f"Failed to start Docker container: {e}")

    def execute_code(self, code: str) -> ExecutionResult:
        """Send code to running container and get results."""
        if not self._process:
            raise RuntimeError("No active session. Call start_session() first.")

        start_time = time.time()

        try:
            # Wrap user code to capture output and results
            wrapped_code = self._wrap_code_for_execution(code)

            # Send code to container
            self._process.stdin.write(wrapped_code + "\n")
            self._process.stdin.flush()

            # Read result with timeout
            stdout_lines = []
            stderr_lines = []

            # Simple timeout mechanism - not perfect but works for MVP
            end_time = start_time + self.timeout_seconds
            result_captured = False

            while time.time() < end_time and not result_captured:
                if self._process.poll() is not None:
                    # Process died
                    break

                # Try to read a line with short timeout
                try:
                    # This is simplified - in production we'd use select/threading
                    self._process.stdout.settimeout(0.1)
                    line = self._process.stdout.readline()
                    if line:
                        if line.startswith("__ENVTORCH_RESULT__"):
                            result_captured = True
                            result_json = line[len("__ENVTORCH_RESULT__"):].strip()
                            break
                        else:
                            stdout_lines.append(line.rstrip())
                except:
                    time.sleep(0.01)

            if not result_captured:
                return ExecutionResult.from_exception(
                    TimeoutError(f"Code execution timed out after {self.timeout_seconds}s"),
                    stdout="\n".join(stdout_lines),
                    stderr="\n".join(stderr_lines)
                )

            # Parse result
            try:
                result_data = json.loads(result_json)
                execution_time_ms = (time.time() - start_time) * 1000

                if result_data.get("success", True):
                    return ExecutionResult.from_success(
                        return_value=result_data.get("return_value"),
                        stdout=result_data.get("stdout", ""),
                        stderr=result_data.get("stderr", ""),
                        execution_time_ms=execution_time_ms
                    )
                else:
                    # Reconstruct exception from data
                    exc_type = result_data.get("exception_type", "Exception")
                    exc_message = result_data.get("exception_message", "")

                    # Create a generic exception for now
                    exc = Exception(f"{exc_type}: {exc_message}")
                    result = ExecutionResult.from_exception(
                        exc,
                        stdout=result_data.get("stdout", ""),
                        stderr=result_data.get("stderr", "")
                    )
                    result.traceback_str = result_data.get("traceback", "")
                    result.execution_time_ms = execution_time_ms
                    return result

            except json.JSONDecodeError as e:
                return ExecutionResult.from_exception(
                    RuntimeError(f"Failed to parse execution result: {e}"),
                    stdout="\n".join(stdout_lines)
                )

        except Exception as e:
            return ExecutionResult.from_exception(e)

    def get_variable_dump(self) -> dict[str, Any]:
        """Get all variables for render() - send globals() inspection command."""
        if not self._process:
            raise RuntimeError("No active session. Call start_session() first.")

        dump_code = '''
import json
result = {}
for name, value in globals().items():
    if not name.startswith('_') and name not in ['json']:
        try:
            # Try to get a readable representation
            if hasattr(value, '__dict__') and not callable(value):
                result[name] = f"<{type(value).__name__}: {str(value)[:100]}>"
            else:
                result[name] = repr(value)[:200]  # Limit length
        except:
            result[name] = f"<{type(value).__name__} object>"
print("__ENVTORCH_DUMP__" + json.dumps(result))
'''

        # Execute the dump code
        exec_result = self.execute_code(dump_code)
        if not exec_result.success:
            return {"error": "Failed to dump variables", "details": exec_result.exception_message}

        # Extract dump from stdout
        for line in exec_result.stdout.split('\n'):
            if line.startswith("__ENVTORCH_DUMP__"):
                try:
                    return json.loads(line[len("__ENVTORCH_DUMP__"):])
                except json.JSONDecodeError:
                    pass

        return {"error": "No variable dump found in output"}

    def stop_session(self) -> None:
        """Kill the container process."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            finally:
                self._process = None
                self.container_id = None

    def _wrap_code_for_execution(self, code: str) -> str:
        """Wrap user code to capture results and exceptions."""
        return f'''
import sys
import json
import traceback
from io import StringIO

# Capture stdout/stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

result = {{"success": True, "return_value": None, "stdout": "", "stderr": ""}}

try:
    # Execute user code
    exec_result = None
    exec("""{code}""")
    result["return_value"] = exec_result if 'exec_result' in locals() else None
except Exception as e:
    result["success"] = False
    result["exception_type"] = e.__class__.__name__
    result["exception_message"] = str(e)
    result["traceback"] = traceback.format_exc()
finally:
    # Restore stdout/stderr and capture output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

# Send result back
print("__ENVTORCH_RESULT__" + json.dumps(result, default=str))
'''

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_session()