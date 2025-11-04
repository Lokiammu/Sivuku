"""Convert an existing environment to OpenEnv-compatible structure."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, List, Tuple

import typer
from importlib import resources

from .._cli_utils import console


app = typer.Typer(help="Convert an existing environment to OpenEnv format")


def _is_git_repo(directory: Path) -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=str(directory), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _ensure_git_repo(directory: Path) -> None:
    if not _is_git_repo(directory):
        subprocess.run(["git", "init"], cwd=str(directory), check=True)


def _snake_to_pascal(snake_str: str) -> str:
    """Convert snake_case to PascalCase (e.g., 'coding_env' -> 'CodingEnv')."""
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _snake_to_title(snake_str: str) -> str:
    """Convert snake_case to Title Case (e.g., 'coding_env' -> 'Coding Env')."""
    return " ".join(word.capitalize() for word in snake_str.split("_"))


def _copy_template_file(template_pkg: str, template_rel_path: str, dest_path: Path, env_name: str) -> None:
    """Copy template file and replace placeholders with appropriate naming conventions."""
    base = resources.files(template_pkg)
    src = base.joinpath(template_rel_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    try:
        text = data.decode("utf-8")
        # Replace placeholders with appropriate naming conventions
        env_class_name = _snake_to_pascal(env_name)
        env_title_name = _snake_to_title(env_name)
        text = text.replace("__ENV_CLASS_NAME__", env_class_name)
        text = text.replace("__ENV_TITLE_NAME__", env_title_name)
        text = text.replace("__ENV_NAME__", env_name)  # Keep snake_case for module names, etc.
        dest_path.write_text(text)
    except UnicodeDecodeError:
        dest_path.write_bytes(data)


def _ensure_files(env_root: Path, env_name: str) -> Tuple[List[Path], List[Path]]:
    """
    Ensure required files exist. Returns (created_files, created_dirs).
    """
    created_files: List[Path] = []
    created_dirs: List[Path] = []

    template_pkg = "openenv_cli.templates.openenv_env"

    # Manifest
    manifest = env_root / "openenv.yaml"
    if not manifest.exists():
        _copy_template_file(template_pkg, "openenv.yaml", manifest, env_name)
        created_files.append(manifest)

    # README
    readme = env_root / "README.md"
    if not readme.exists():
        _copy_template_file(template_pkg, "README.md", readme, env_name)
        created_files.append(readme)

    # client.py and models.py
    client_py = env_root / "client.py"
    if not client_py.exists():
        _copy_template_file(template_pkg, "client.py", client_py, env_name)
        created_files.append(client_py)

    models_py = env_root / "models.py"
    if not models_py.exists():
        _copy_template_file(template_pkg, "models.py", models_py, env_name)
        created_files.append(models_py)

    # server tree
    server_dir = env_root / "server"
    if not server_dir.exists():
        server_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(server_dir)

    server_init = server_dir / "__init__.py"
    if not server_init.exists():
        _copy_template_file(template_pkg, "server/__init__.py", server_init, env_name)
        created_files.append(server_init)

    server_app = server_dir / "app.py"
    if not server_app.exists():
        _copy_template_file(template_pkg, "server/app.py", server_app, env_name)
        created_files.append(server_app)

    dockerfile = server_dir / "Dockerfile"
    if not dockerfile.exists():
        _copy_template_file(template_pkg, "server/Dockerfile", dockerfile, env_name)
        created_files.append(dockerfile)

    reqs = server_dir / "requirements.txt"
    if not reqs.exists():
        _copy_template_file(template_pkg, "server/requirements.txt", reqs, env_name)
        created_files.append(reqs)

    return created_files, created_dirs


def _stage_paths(env_root: Path, paths: List[Path]) -> None:
    if not paths:
        return
    rels = [str(p.relative_to(env_root)) for p in paths]
    subprocess.run(["git", "add", "-A", *rels], cwd=str(env_root), check=True)


def _unstage_all(env_root: Path) -> None:
    subprocess.run(["git", "reset"], cwd=str(env_root), check=True)


def _show_staged_diff(env_root: Path) -> None:
    subprocess.run(["git", "diff", "--staged"], cwd=str(env_root), check=True)


def _rollback_created(created_files: List[Path], created_dirs: List[Path]) -> None:
    # Remove files first, then empty dirs we created
    for f in created_files:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass
    for d in sorted(created_dirs, key=lambda p: len(str(p)), reverse=True):
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass


@app.command()
def convert(
    env_path: Annotated[
        str | None,
        typer.Option(
            "--env-path",
            help="Path to the environment root (defaults to current working directory)",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Accept changes without confirmation"),
    ] = False,
) -> None:
    """
    Convert the current directory (or specified path) to OpenEnv-compatible format.
    
    Unlike 'init', this command works directly on the current working directory without
    creating a subdirectory. It ensures all required OpenEnv files exist, stages changes,
    shows a git diff, and optionally commits or rolls back the changes.
    """
    # Work directly on the current directory or specified path (no subdirectory creation)
    env_root = Path(env_path).resolve() if env_path is not None else Path.cwd().resolve()
    if not env_root.exists() or not env_root.is_dir():
        raise typer.BadParameter(f"Environment path is invalid: {env_root}")

    env_name = env_root.name

    try:
        _ensure_git_repo(env_root)

        # Create missing files
        created_files, created_dirs = _ensure_files(env_root, env_name)

        if not created_files and not created_dirs:
            console.print("[bold green]Environment already OpenEnv-compatible. No changes needed.[/bold green]")
            return

        # Stage new files for preview
        _stage_paths(env_root, created_files)

        console.print("[bold cyan]Proposed changes:[/bold cyan]")
        _show_staged_diff(env_root)

        proceed = yes or typer.confirm("Apply these changes and create a commit?", default=True)
        if proceed:
            subprocess.run(["git", "commit", "-m", "openenv: convert environment to OpenEnv format"], cwd=str(env_root), check=True)
            console.print("[bold green]Conversion committed.[/bold green]")
        else:
            _unstage_all(env_root)
            _rollback_created(created_files, created_dirs)
            console.print("[bold yellow]Changes discarded.[/bold yellow]")

    except Exception as e:
        # Best-effort rollback of created files and unstage
        try:
            _unstage_all(env_root)
        except Exception:
            pass
        # Attempt to clean any obvious template additions if we can detect them
        # (No-op if none were created before the exception.)
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


