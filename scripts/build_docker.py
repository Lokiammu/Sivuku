#!/usr/bin/env python3
"""
Cross-platform Docker build script for OpenEnv environments.

This script:
1. Generates requirements.txt from pyproject.toml (if it exists)
2. Builds the Docker image for the environment
3. Optionally pushes to a registry
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}", file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        if check:
            sys.exit(1)
        return e


def generate_requirements(env_path: Path, output_path: Path | None = None) -> bool:
    """Generate requirements.txt from pyproject.toml."""
    pyproject = env_path / "pyproject.toml"

    if not pyproject.exists():
        print(f"No pyproject.toml found in {env_path}, skipping requirements generation")
        return False

    print(f"Generating requirements.txt from {pyproject}")

    # Use the generate_requirements.py script
    script_dir = Path(__file__).parent
    generate_script = script_dir / "generate_requirements.py"

    cmd = [sys.executable, str(generate_script), str(env_path)]
    if output_path:
        cmd.extend(["-o", str(output_path)])

    result = run_command(cmd, check=False)
    return result.returncode == 0


def build_docker_image(
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
        print(f"Error: Dockerfile not found at {dockerfile}", file=sys.stderr)
        return False

    # Generate tag if not provided
    if tag is None:
        env_name = env_path.name
        if env_name.endswith("_env"):
            env_name = env_name[:-4]
        tag = f"openenv-{env_name}"

    print(f"Building Docker image: {tag}")
    print(f"Context: {context_path}")
    print(f"Dockerfile: {dockerfile}")

    # Build Docker command
    cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile)]

    if no_cache:
        cmd.append("--no-cache")

    if build_args:
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])

    cmd.append(str(context_path))

    result = run_command(cmd, check=False)
    return result.returncode == 0


def push_docker_image(tag: str, registry: str | None = None) -> bool:
    """Push Docker image to registry."""
    if registry:
        full_tag = f"{registry}/{tag}"
        print(f"Tagging image as {full_tag}")
        run_command(["docker", "tag", tag, full_tag])
        tag = full_tag

    print(f"Pushing image: {tag}")
    result = run_command(["docker", "push", tag], check=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Build Docker images for OpenEnv environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build echo_env with default settings
  python scripts/build_docker.py src/envs/echo_env

  # Build with custom tag
  python scripts/build_docker.py src/envs/echo_env -t my-custom-tag

  # Build and push to registry
  python scripts/build_docker.py src/envs/echo_env --push --registry myregistry.io

  # Build without cache
  python scripts/build_docker.py src/envs/echo_env --no-cache

  # Skip requirements generation
  python scripts/build_docker.py src/envs/echo_env --no-generate-requirements
        """,
    )

    parser.add_argument("env_path", type=Path, help="Path to the environment directory")

    parser.add_argument("-t", "--tag", help="Docker image tag (default: openenv-<env_name>)")

    parser.add_argument("-c", "--context", type=Path, help="Build context path (default: <env_path>/server)")

    parser.add_argument("-f", "--dockerfile", type=Path, help="Path to Dockerfile (default: <context>/Dockerfile)")

    parser.add_argument(
        "--no-generate-requirements",
        action="store_true",
        help="Skip generating requirements.txt from pyproject.toml",
    )

    parser.add_argument("--no-cache", action="store_true", help="Build without using cache")

    parser.add_argument("--build-arg", action="append", help="Build arguments (can be used multiple times)")

    parser.add_argument("--push", action="store_true", help="Push image to registry after building")

    parser.add_argument("--registry", help="Registry to push to (e.g., docker.io/username)")

    args = parser.parse_args()

    # Validate environment path
    if not args.env_path.exists():
        print(f"Error: Environment path does not exist: {args.env_path}", file=sys.stderr)
        sys.exit(1)

    if not args.env_path.is_dir():
        print(f"Error: Environment path is not a directory: {args.env_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Building Docker image for: {args.env_path.name}")
    print("=" * 60)

    # Generate requirements.txt if needed
    if not args.no_generate_requirements:
        success = generate_requirements(args.env_path)
        if success:
            print("✓ Requirements generated successfully")
        print()

    # Parse build args
    build_args = {}
    if args.build_arg:
        for arg in args.build_arg:
            if "=" in arg:
                key, value = arg.split("=", 1)
                build_args[key] = value
            else:
                print(f"Warning: Invalid build arg format: {arg}", file=sys.stderr)

    # Build Docker image
    success = build_docker_image(
        env_path=args.env_path,
        tag=args.tag,
        context_path=args.context,
        dockerfile=args.dockerfile,
        build_args=build_args if build_args else None,
        no_cache=args.no_cache,
    )

    if not success:
        print("✗ Docker build failed", file=sys.stderr)
        sys.exit(1)

    print("✓ Docker build successful")

    # Push if requested
    if args.push:
        print()
        tag = args.tag or f"openenv-{args.env_path.name.replace('_env', '')}"
        success = push_docker_image(tag, args.registry)
        if not success:
            print("✗ Docker push failed", file=sys.stderr)
            sys.exit(1)
        print("✓ Docker push successful")

    print("\nDone!")


if __name__ == "__main__":
    main()
