# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build Docker images for OpenEnv environments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from .._cli_utils import console

app = typer.Typer(help="Build Docker images for OpenEnv environments")


def _run_command(
    cmd: list[str],
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors."""
    console.print(f"[bold cyan]Running:[/bold cyan] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            console.print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error running command:[/bold red] {e}", file=sys.stderr)
        if e.stderr:
            console.print(e.stderr, file=sys.stderr)
        if check:
            raise typer.Exit(1) from e
        return e


def _build_docker_image(
    env_path: Path,
    tag: str | None = None,
    context_path: Path | None = None,
    dockerfile: Path | None = None,
    build_args: dict[str, str] | None = None,
    no_cache: bool = False,
) -> bool:
    """Build Docker image for the environment."""
    # Determine context and Dockerfile paths
    if context_path is None:
        context_path = env_path / "server"

    if dockerfile is None:
        dockerfile = context_path / "Dockerfile"

    if not dockerfile.exists():
        console.print(
            f"[bold red]Error:[/bold red] Dockerfile not found at {dockerfile}",
            file=sys.stderr,
        )
        return False

    # Generate tag if not provided
    if tag is None:
        env_name = env_path.name
        if env_name.endswith("_env"):
            env_name = env_name[:-4]
        tag = f"openenv-{env_name}"

    console.print(f"[bold cyan]Building Docker image:[/bold cyan] {tag}")
    console.print(f"[bold cyan]Context:[/bold cyan] {context_path}")
    console.print(f"[bold cyan]Dockerfile:[/bold cyan] {dockerfile}")

    # Build Docker command
    cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile)]

    if no_cache:
        cmd.append("--no-cache")

    if build_args:
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

    cmd.append(str(context_path))

    result = _run_command(cmd, check=False)
    return result.returncode == 0


def _push_docker_image(tag: str, registry: str | None = None) -> bool:
    """Push Docker image to registry."""
    if registry:
        full_tag = f"{registry}/{tag}"
        console.print(f"[bold cyan]Tagging image as {full_tag}[/bold cyan]")
        _run_command(["docker", "tag", tag, full_tag])
        tag = full_tag

    console.print(f"[bold cyan]Pushing image:[/bold cyan] {tag}")
    result = _run_command(["docker", "push", tag], check=False)
    return result.returncode == 0


@app.command()
def build(
    env_path: Annotated[
        str,
        typer.Argument(help="Path to the environment directory"),
    ],
    tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            "-t",
            help="Docker image tag (default: openenv-<env_name>)",
        ),
    ] = None,
    context: Annotated[
        str | None,
        typer.Option(
            "--context",
            "-c",
            help="Build context path (default: <env_path>/server)",
        ),
    ] = None,
    dockerfile: Annotated[
        str | None,
        typer.Option(
            "--dockerfile",
            "-f",
            help="Path to Dockerfile (default: <context>/Dockerfile)",
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Build without using cache",
        ),
    ] = False,
    build_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--build-arg",
            help="Build arguments (can be used multiple times, format: KEY=VALUE)",
        ),
    ] = None,
    push: Annotated[
        bool,
        typer.Option(
            "--push",
            help="Push image to registry after building",
        ),
    ] = False,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry",
            help="Registry to push to (e.g., docker.io/username)",
        ),
    ] = None,
) -> None:
    """
    Build Docker images for OpenEnv environments.

    This command builds Docker images using the environment's pyproject.toml
    and uv for dependency management.

    Examples:
        # Build echo_env with default settings
        $ openenv build src/envs/echo_env

        # Build with custom tag
        $ openenv build src/envs/echo_env -t my-custom-tag

        # Build and push to registry
        $ openenv build src/envs/echo_env --push --registry myregistry.io

        # Build without cache
        $ openenv build src/envs/echo_env --no-cache

        # Build with custom build arguments
        $ openenv build src/envs/echo_env --build-arg VERSION=1.0 --build-arg ENV=prod
    """
    # Validate environment path
    env_path_obj = Path(env_path)
    if not env_path_obj.exists():
        console.print(
            f"[bold red]Error:[/bold red] Environment path does not exist: {env_path_obj}",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    if not env_path_obj.is_dir():
        console.print(
            f"[bold red]Error:[/bold red] Environment path is not a directory: {env_path_obj}",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    console.print(f"[bold]Building Docker image for:[/bold] {env_path_obj.name}")
    console.print("=" * 60)

    # Parse build args
    build_args = {}
    if build_arg:
        for arg in build_arg:
            if "=" in arg:
                key, value = arg.split("=", 1)
                build_args[key] = value
            else:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Invalid build arg format: {arg}",
                    file=sys.stderr,
                )

    # Convert string paths to Path objects
    context_path_obj = Path(context) if context else None
    dockerfile_path_obj = Path(dockerfile) if dockerfile else None

    # Build Docker image
    success = _build_docker_image(
        env_path=env_path_obj,
        tag=tag,
        context_path=context_path_obj,
        dockerfile=dockerfile_path_obj,
        build_args=build_args if build_args else None,
        no_cache=no_cache,
    )

    if not success:
        console.print("[bold red]✗ Docker build failed[/bold red]", file=sys.stderr)
        raise typer.Exit(1)

    console.print("[bold green]✓ Docker build successful[/bold green]")

    # Push if requested
    if push:
        console.print()
        tag_to_push = tag or f"openenv-{env_path_obj.name.replace('_env', '')}"
        success = _push_docker_image(tag_to_push, registry)
        if not success:
            console.print("[bold red]✗ Docker push failed[/bold red]", file=sys.stderr)
            raise typer.Exit(1)
        console.print("[bold green]✓ Docker push successful[/bold green]")

    console.print("\n[bold green]Done![/bold green]")
