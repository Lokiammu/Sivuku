#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to generate requirements.txt from pyproject.toml for Docker builds.

This script uses uv to compile dependencies from pyproject.toml into a locked
requirements.txt file suitable for Docker builds.

Usage:
    python scripts/generate_requirements.py <path_to_env_dir>
    python scripts/generate_requirements.py src/envs/echo_env
"""

import argparse
import subprocess
import sys
from pathlib import Path


def generate_requirements(env_path: Path, output_path: Path = None) -> bool:
    """
    Generate requirements.txt from pyproject.toml using uv.

    Args:
        env_path: Path to the environment directory containing pyproject.toml
        output_path: Optional custom output path for requirements.txt

    Returns:
        True if successful, False otherwise
    """
    pyproject_path = env_path / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found at {pyproject_path}", file=sys.stderr)
        return False

    # Default output location
    if output_path is None:
        output_path = env_path / "server" / "requirements.txt"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating requirements.txt from {pyproject_path}")
    print(f"Output: {output_path}")

    try:
        # Use uv pip compile to generate requirements.txt
        cmd = [
            "uv",
            "pip",
            "compile",
            str(pyproject_path),
            "--output-file",
            str(output_path),
            "--no-header",  # Skip header comments for cleaner output
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stdout:
            print(result.stdout)

        print(f"✓ Successfully generated {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print("Error running uv pip compile:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False
    except FileNotFoundError:
        print(
            "Error: 'uv' command not found. Please install uv first:", file=sys.stderr
        )
        print("  pip install uv", file=sys.stderr)
        print("  or visit: https://github.com/astral-sh/uv", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate requirements.txt from pyproject.toml for Docker builds"
    )
    parser.add_argument(
        "env_path",
        type=Path,
        help="Path to environment directory containing pyproject.toml",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Custom output path for requirements.txt (default: <env_path>/server/requirements.txt)",
    )
    parser.add_argument(
        "--extras",
        nargs="+",
        help="Optional dependency groups to include (e.g., dev test)",
    )

    args = parser.parse_args()

    if not args.env_path.exists():
        print(f"Error: Directory not found: {args.env_path}", file=sys.stderr)
        sys.exit(1)

    success = generate_requirements(args.env_path, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
