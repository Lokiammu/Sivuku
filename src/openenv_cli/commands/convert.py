"""Convert an existing OpenEnv environment to the standardized structure."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .._cli_utils import console

app = typer.Typer(help="Convert existing environments to standardized structure")


def _backup_directory(path: Path) -> Path:
    """Create a backup of the directory."""
    backup_path = path.parent / f"{path.name}.backup"
    counter = 1
    while backup_path.exists():
        backup_path = path.parent / f"{path.name}.backup.{counter}"
        counter += 1

    shutil.copytree(path, backup_path)
    return backup_path


def _parse_requirements_txt(requirements_file: Path) -> list[str]:
    """Parse requirements.txt and return list of dependencies."""
    if not requirements_file.exists():
        return []

    dependencies = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                # Remove inline comments
                if "#" in line:
                    line = line.split("#")[0].strip()
                dependencies.append(line)

    return dependencies


def _get_template_content(filename: str) -> str:
    """Get content from a template file.
    
    Raises:
        FileNotFoundError: If template file cannot be found
        ValueError: If filename is not a known template
    """
    # Get the path to the templates directory relative to this file
    cli_dir = Path(__file__).parent.parent
    template_dir = cli_dir / "templates" / "openenv_env"
    
    # Navigate to the file
    if filename == "pyproject.toml":
        template_file = template_dir / filename
    elif filename == "Dockerfile":
        template_file = template_dir / "server" / filename
    else:
        raise ValueError(f"Unknown template file: {filename}")
    
    if not template_file.exists():
        raise FileNotFoundError(
            f"Template file not found: {template_file}\n"
            f"Expected location: {template_file.absolute()}"
        )
        
    return template_file.read_text(encoding="utf-8")


def _generate_pyproject_from_template(
    env_name: str, env_title: str, dependencies: list[str]
) -> str:
    """Generate pyproject.toml from template with dependencies.
    
    Raises:
        FileNotFoundError: If template file cannot be found
    """
    template = _get_template_content("pyproject.toml")
    
    # Replace template variables
    content = template.replace("__ENV_NAME__", env_name)
    content = content.replace("__ENV_TITLE_NAME__", env_title)
    
    # Add additional dependencies if any
    if dependencies:
        # Find the dependencies section and add custom deps
        deps_comment = "    # Add your environment-specific dependencies here"
        if deps_comment in content:
            additional_deps = "\n".join(f'    "{dep}",' for dep in dependencies)
            content = content.replace(
                deps_comment,
                f"{deps_comment}\n{additional_deps}"
            )
    
    return content


def _generate_dockerfile_from_template(env_name: str) -> str:
    """Generate Dockerfile from template.
    
    Raises:
        FileNotFoundError: If template file cannot be found
    """
    template = _get_template_content("Dockerfile")
    
    # Replace template variables
    content = template.replace("__ENV_NAME__", env_name)
    
    return content


def _create_outputs_structure(env_dir: Path) -> None:
    """Create standard outputs directory structure."""
    outputs_dir = env_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (outputs_dir / "logs").mkdir(exist_ok=True)
    (outputs_dir / "evals").mkdir(exist_ok=True)

    # Create .gitignore for outputs
    gitignore_path = outputs_dir / ".gitignore"
    gitignore_path.write_text(
        "# Ignore all files in outputs directory\n*\n!.gitignore\n", encoding="utf-8"
    )


def _update_gitignore(env_dir: Path) -> None:
    """Update or create .gitignore with standard patterns."""
    gitignore_path = env_dir / ".gitignore"

    # Standard patterns
    patterns = [
        "# Python",
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "*.so",
        ".Python",
        "build/",
        "develop-eggs/",
        "dist/",
        "downloads/",
        "eggs/",
        ".eggs/",
        "lib/",
        "lib64/",
        "parts/",
        "sdist/",
        "var/",
        "wheels/",
        "*.egg-info/",
        ".installed.cfg",
        "*.egg",
        "",
        "# Virtual environments",
        "venv/",
        "ENV/",
        "env/",
        ".venv/",
        "",
        "# IDEs",
        ".vscode/",
        ".idea/",
        "*.swp",
        "*.swo",
        "*~",
        "",
        "# OpenEnv specific",
        "outputs/",
        "",
        "# OS",
        ".DS_Store",
        "Thumbs.db",
    ]

    if gitignore_path.exists():
        # Read existing content
        existing = gitignore_path.read_text(encoding="utf-8")
        # Only add patterns that don't exist
        new_patterns = [p for p in patterns if p not in existing]
        if new_patterns:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n# Added by openenv convert\n")
                f.write("\n".join(new_patterns))
                f.write("\n")
    else:
        gitignore_path.write_text("\n".join(patterns) + "\n", encoding="utf-8")


def _generate_uv_lock(env_dir: Path) -> bool:
    """Generate uv.lock from pyproject.toml using uv."""
    pyproject_path = env_dir / "pyproject.toml"

    if not pyproject_path.exists():
        return False

    try:
        cmd = [
            "uv",
            "lock",
            "--directory",
            str(env_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stdout:
            console.print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        console.print(
            f"[yellow]Warning: Could not generate uv.lock: {e.stderr}[/yellow]"
        )
        return False
    except FileNotFoundError:
        console.print(
            "[yellow]Warning: 'uv' not found. Install it to generate uv.lock[/yellow]"
        )
        return False


@app.command()
def convert(
    env_path: Annotated[
        str,
        typer.Argument(help="Path to the environment directory to convert"),
    ],
    backup: Annotated[
        bool,
        typer.Option("--backup/--no-backup", help="Create backup before conversion"),
    ] = True,
    generate_lock: Annotated[
        bool,
        typer.Option(
            "--generate-lock/--no-generate-lock",
            help="Generate uv.lock from pyproject.toml",
        ),
    ] = True,
) -> None:
    """
    Convert an existing OpenEnv environment to the standardized structure.

    This command:
    1. Creates a backup of the original structure (optional)
    2. Analyzes existing requirements.txt (if present)
    3. Generates pyproject.toml with dependencies
    4. Creates outputs/ directory structure
    5. Updates .gitignore
    6. Optionally generates uv.lock from pyproject.toml

    Example:
        $ openenv convert src/envs/echo_env
        $ openenv convert src/envs/atari_env --no-backup
        $ openenv convert ./my_env --no-generate-lock
    """
    env_dir = Path(env_path).resolve()

    # Validate environment directory
    if not env_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Directory not found: {env_dir}")
        raise typer.Exit(1)

    if not env_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Path is not a directory: {env_dir}")
        raise typer.Exit(1)

    env_name = env_dir.name
    if env_name.endswith("_env"):
        env_name = env_name[:-4]

    console.print(f"[bold cyan]Converting environment: {env_dir.name}[/bold cyan]")

    # Create backup
    if backup:
        try:
            backup_path = _backup_directory(env_dir)
            console.print(f"[green]✓[/green] Created backup at: {backup_path}")
        except Exception as e:
            console.print(f"[bold red]Error creating backup:[/bold red] {e}")
            raise typer.Exit(1) from e

    try:
        # Check if pyproject.toml already exists
        pyproject_path = env_dir / "pyproject.toml"
        if pyproject_path.exists():
            console.print(
                "[yellow]⚠[/yellow] pyproject.toml already exists, skipping generation"
            )
        else:
            # Look for requirements.txt
            requirements_file = env_dir / "server" / "requirements.txt"
            if not requirements_file.exists():
                requirements_file = env_dir / "requirements.txt"

            dependencies = []
            if requirements_file.exists():
                console.print(
                    f"[green]✓[/green] Found requirements.txt at: {requirements_file}"
                )
                dependencies = _parse_requirements_txt(requirements_file)
                console.print(
                    f"[green]✓[/green] Parsed {len(dependencies)} dependencies"
                )
            else:
                console.print(
                    "[yellow]⚠[/yellow] No requirements.txt found, creating minimal pyproject.toml"
                )

            # Generate pyproject.toml from template
            env_title = env_name.replace("_", " ").title()
            pyproject_content = _generate_pyproject_from_template(
                env_name, env_title, dependencies
            )
            pyproject_path.write_text(pyproject_content, encoding="utf-8")
            console.print("[green]✓[/green] Generated pyproject.toml from template")

        # Check if Dockerfile exists, and generate/update if needed
        server_dir = env_dir / "server"
        dockerfile_path = server_dir / "Dockerfile"
        
        if not dockerfile_path.exists():
            # Create server directory if it doesn't exist
            server_dir.mkdir(exist_ok=True)
            
            # Generate Dockerfile from template
            dockerfile_content = _generate_dockerfile_from_template(env_name)
            dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
            console.print("[green]✓[/green] Generated Dockerfile from template")
        else:
            console.print(
                "[yellow]⚠[/yellow] Dockerfile already exists, skipping generation"
            )

        # Create outputs directory structure
        _create_outputs_structure(env_dir)
        console.print("[green]✓[/green] Created outputs/ directory structure")

        # Update .gitignore
        _update_gitignore(env_dir)
        console.print("[green]✓[/green] Updated .gitignore")

        # Generate uv.lock if requested
        if generate_lock:
            console.print(
                "\n[bold]Generating uv.lock from pyproject.toml...[/bold]"
            )
            if _generate_uv_lock(env_dir):
                console.print("[green]✓[/green] Generated uv.lock")
            else:
                console.print(
                    "[yellow]⚠[/yellow] Could not generate uv.lock automatically"
                )
                console.print("    You can generate it manually with:")
                console.print(
                    f"    cd {env_dir} && uv lock"
                )

        console.print("\n[bold green]Conversion completed successfully![/bold green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Review the generated pyproject.toml and Dockerfile")
        console.print("  2. Test the environment to ensure it works correctly")
        console.print("  3. Build Docker image (from repo root):")
        console.print(f"     docker build -t {env_name}_env:latest -f src/envs/{env_name}_env/server/Dockerfile .")
        console.print(f"  4. Run: cd {env_dir} && uv sync")

    except Exception as e:
        console.print(f"\n[bold red]Error during conversion:[/bold red] {e}")
        if backup:
            console.print(
                f"[yellow]You can restore from backup at: {backup_path}[/yellow]"
            )
        raise typer.Exit(1) from e
