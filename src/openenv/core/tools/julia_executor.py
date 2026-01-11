# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local Julia Executor.

This module provides a Julia code executor that runs Julia code in a subprocess.
It handles:
- Code execution with configurable timeouts
- Capturing stdout/stderr
- Process pool support for better performance
- Graceful error handling

Key features:
- Uses subprocess.run for reliable code execution
- Supports process pooling for reduced Julia startup overhead
- Configurable execution timeout
- Structured logging for operational visibility
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from multiprocessing import Manager
from typing import Optional

from openenv.core.env_server.types import CodeExecResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Global process pool (initialized lazily)
_pool: Optional[ProcessPoolExecutor] = None
_pool_size: int = 0
_pool_timeout: int = 120
_manager: Optional[Manager] = None


def _execute_julia_code(code: str, timeout: int = 120) -> dict:
    """
    Execute Julia code in a subprocess.

    This function is designed to be called from a process pool.

    Args:
        code: Julia code to execute
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with stdout, stderr, and exit_code
    """
    try:
        # Check if Julia is available
        julia_path = shutil.which("julia")
        if julia_path is None:
            return {
                "stdout": "",
                "stderr": "Julia not found in PATH. Please install Julia.",
                "exit_code": 127,
            }

        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jl", delete=False
        ) as tmp_file:
            tmp_file.write(code)
            tmp_file.flush()
            tmp_path = tmp_file.name

        try:
            # Execute Julia code
            result = subprocess.run(
                [julia_path, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timeout after {timeout}s",
                "exit_code": 124,  # Standard timeout exit code
            }

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "stdout": "",
            "stderr": f"Execution error: {e}\n{tb}",
            "exit_code": 1,
        }


class JuliaExecutor:
    """Julia code executor with optional process pooling.

    The executor runs Julia code in subprocesses, capturing stdout/stderr
    and exit codes. It supports process pooling for better performance
    under concurrent load.

    Example:
        >>> executor = JuliaExecutor()
        >>> result = executor.run('println("Hello, Julia!")')
        >>> print(result.stdout)  # "Hello, Julia!\\n"
        >>> print(result.exit_code)  # 0
    """

    def __init__(
        self,
        use_process_pool: bool = True,
        timeout: int = 120,
    ):
        """
        Initialize Julia executor.

        Args:
            use_process_pool: Use process pool for execution (better performance)
            timeout: Default execution timeout in seconds
        """
        self._use_process_pool = use_process_pool
        self._timeout = timeout

    def run(self, code: str, timeout: Optional[int] = None) -> CodeExecResult:
        """
        Execute Julia code and return the result.

        Args:
            code: Julia code to execute
            timeout: Execution timeout in seconds (overrides default)

        Returns:
            CodeExecResult with stdout, stderr, and exit_code
        """
        if timeout is None:
            timeout = self._timeout

        try:
            if self._use_process_pool and _pool is not None:
                # Use process pool
                try:
                    future = _pool.submit(_execute_julia_code, code, timeout)
                    result = future.result(timeout=timeout + 30)  # Extra buffer
                except TimeoutError:
                    return CodeExecResult(
                        stdout="",
                        stderr=f"Process pool timeout after {timeout + 30}s",
                        exit_code=124,
                    )
            else:
                # Direct execution
                result = _execute_julia_code(code, timeout)

            return CodeExecResult(
                stdout=result["stdout"],
                stderr=result["stderr"],
                exit_code=result["exit_code"],
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("JuliaExecutor raised an exception during run")
            return CodeExecResult(stdout="", stderr=tb, exit_code=1)

    @staticmethod
    def enable_process_pool(size: int = 4, timeout: int = 120) -> bool:
        """
        Enable process pool for Julia execution.

        Args:
            size: Number of workers in the pool
            timeout: Default execution timeout

        Returns:
            True if pool was enabled successfully
        """
        global _pool, _pool_size, _pool_timeout

        try:
            if _pool is not None:
                # Pool already exists
                return True

            _pool = ProcessPoolExecutor(
                max_workers=size,
            )
            _pool_size = size
            _pool_timeout = timeout

            logger.info(f"Julia process pool enabled with {size} workers")
            return True

        except Exception as e:
            logger.error(f"Failed to enable Julia process pool: {e}")
            return False

    @staticmethod
    def shutdown_pool() -> None:
        """Shutdown the process pool."""
        global _pool

        if _pool is not None:
            try:
                _pool.shutdown(wait=True)
                logger.info("Julia process pool shut down")
            except Exception as e:
                logger.error(f"Error shutting down Julia process pool: {e}")
            finally:
                _pool = None

    @staticmethod
    def get_pool_metrics() -> dict:
        """
        Get metrics about the process pool.

        Returns:
            Dictionary with pool metrics
        """
        global _pool, _pool_size

        if _pool is None:
            return {
                "enabled": False,
                "pool_size": 0,
                "available_workers": 0,
            }

        return {
            "enabled": True,
            "pool_size": _pool_size,
            "timeout": _pool_timeout,
            # Note: ProcessPoolExecutor doesn't expose worker availability
            "available_workers": _pool_size,
        }
