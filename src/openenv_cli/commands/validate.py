# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv validate command.

This module provides the 'openenv validate' command to check if environments
are properly configured for multi-mode deployment.
"""

from pathlib import Path

import typer

from openenv_cli._validation import (
    format_validation_report,
    get_deployment_modes,
    validate_multi_mode_deployment,
)


def validate(
    env_name: str = typer.Argument(
        ..., help="Name of the environment to validate (e.g., 'echo_env')"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed information"
    ),
) -> None:
    """
    Validate an environment for multi-mode deployment readiness.

    This command checks if an environment is properly configured to support
    all deployment modes:
    - Docker deployment
    - Direct execution via openenv serve
    - uv run server
    - python -m module execution

    Examples:
        # Validate echo_env
        openenv validate echo_env

        # Validate with detailed output
        openenv validate echo_env --verbose
    """
    # Normalize environment name
    if env_name.endswith("_env"):
        base_name = env_name
    else:
        base_name = f"{env_name}_env"

    # Find environment path
    cwd = Path.cwd()
    env_path = cwd / "src" / "envs" / base_name

    if not env_path.exists():
        typer.echo(f"Error: Environment '{env_name}' not found at {env_path}", err=True)
        raise typer.Exit(1)

    # Run validation
    is_valid, issues = validate_multi_mode_deployment(env_path)

    # Show validation report
    report = format_validation_report(env_name, is_valid, issues)
    typer.echo(report)

    # Show deployment modes if verbose
    if verbose:
        typer.echo("\nSupported deployment modes:")
        modes = get_deployment_modes(env_path)
        for mode, supported in modes.items():
            status = "[YES]" if supported else "[NO]"
            typer.echo(f"  {status} {mode}")

        if is_valid:
            typer.echo("\nUsage examples:")
            typer.echo(f"  openenv serve {env_name}")
            typer.echo(f"  cd src/envs/{base_name} && uv run server")
            typer.echo(f"  python -m envs.{base_name}.server.app")

    if not is_valid:
        raise typer.Exit(1)
