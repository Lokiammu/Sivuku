# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv serve command.

This module provides the 'openenv serve' command to start environment servers
without requiring Docker containers. Supports direct Python execution for
notebooks, development, and cluster environments.
"""

import sys
from pathlib import Path
from typing import Optional

import typer


def serve(
    env_name: str = typer.Argument(
        ..., help="Name of the environment to serve (e.g., 'echo_env')"
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server to"),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes"
    ),
    reload: bool = typer.Option(
        False, "--reload", "-r", help="Enable auto-reload for development"
    ),
) -> None:
    """
    Start an OpenEnv environment server directly without Docker.

    This command allows you to run environment servers in various contexts:
    - Local development with fast iteration
    - Jupyter notebooks and Colab
    - HPC clusters without container support
    - Cloud deployments with custom configurations

    Examples:
        # Start echo_env on default port
        openenv serve echo_env

        # Start with custom configuration
        openenv serve echo_env --host 0.0.0.0 --port 8080 --workers 4

        # Development mode with auto-reload
        openenv serve echo_env --reload

        # From any directory (auto-discovers environment)
        cd /path/to/project
        openenv serve my_custom_env
    """
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "Error: uvicorn is required to run the server. Install it with:\n"
            "    pip install uvicorn\n"
            "or:\n"
            "    uv pip install uvicorn",
            err=True,
        )
        raise typer.Exit(1)

    # Normalize environment name (remove _env suffix if present)
    if env_name.endswith("_env"):
        base_name = env_name
    else:
        base_name = f"{env_name}_env"

    # Try to locate the environment
    env_path = _find_environment(base_name)
    if not env_path:
        typer.echo(
            f"Error: Environment '{env_name}' not found.\n"
            f"Searched for: src/envs/{base_name}\n\n"
            f"Make sure you're in the OpenEnv root directory or the environment exists.",
            err=True,
        )
        raise typer.Exit(1)

    # Check if server/app.py exists
    server_app = env_path / "server" / "app.py"
    if not server_app.exists():
        typer.echo(
            f"Error: Server application not found at {server_app}\n"
            f"Environment structure may be invalid.",
            err=True,
        )
        raise typer.Exit(1)

    # Construct the app module path
    # If we're in src/envs/{env_name}_env, the module is envs.{env_name}_env.server.app:app
    app_module = f"envs.{base_name}.server.app:app"

    typer.echo(f"Starting {env_name} server...")
    typer.echo(f"  Host: {host}")
    typer.echo(f"  Port: {port}")
    typer.echo(f"  Workers: {workers}")
    if reload:
        typer.echo("  Reload: enabled")
    typer.echo(f"  Module: {app_module}")
    typer.echo()

    # Ensure src/ is in the Python path
    src_path = Path.cwd() / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        # Start the server using uvicorn
        uvicorn.run(
            app_module,
            host=host,
            port=port,
            workers=workers
            if not reload
            else 1,  # reload doesn't work with multiple workers
            reload=reload,
            log_level="info",
        )
    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1)


def _find_environment(env_name: str) -> Optional[Path]:
    """
    Find the environment directory.

    Searches in the following locations:
    1. Current directory (if it's an environment)
    2. src/envs/{env_name}
    3. ../src/envs/{env_name} (if running from an env directory)

    Args:
        env_name: Name of the environment (e.g., 'echo_env')

    Returns:
        Path to the environment directory, or None if not found
    """
    cwd = Path.cwd()

    # Check if current directory is the environment
    if (cwd / "server" / "app.py").exists() and cwd.name == env_name:
        return cwd

    # Check src/envs/{env_name}
    env_path = cwd / "src" / "envs" / env_name
    if env_path.exists() and env_path.is_dir():
        return env_path

    # Check ../src/envs/{env_name} (if running from within an env)
    parent_env_path = cwd.parent / "src" / "envs" / env_name
    if parent_env_path.exists() and parent_env_path.is_dir():
        return parent_env_path

    # Check ../../{env_name} (if running from src/envs/{env_name}/server)
    if "server" in cwd.parts:
        grandparent_env_path = cwd.parent
        if (grandparent_env_path / "server" / "app.py").exists():
            return grandparent_env_path

    return None
