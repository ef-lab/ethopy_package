"""Database setup module for EthoPy.

Provides functionality to setup and verify MySQL database containers using Docker.
"""

import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple

import click
import datajoint as dj

from ethopy.utils.ethopy_logging import setup_logging

setup_logging()


def check_docker_status() -> Tuple[bool, str]:
    """Check if Docker daemon is running and accessible.

    Returns:
        Tuple of (is_running: bool, message: str)

    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero return code
        )

        if result.returncode == 0:
            return True, "Docker is running"

        return False, "Docker daemon is not running"

    except FileNotFoundError:
        return False, "Docker is not installed"
    except subprocess.CalledProcessError:
        return False, "Docker daemon is not running"
    except Exception as e:
        return False, f"Error checking Docker status: {str(e)}"


def check_mysql_container(
    container_name: str = "ethopy_sql_db",
) -> Tuple[bool, bool, str]:
    """Check if MySQL container exists and its status.

    Args:
        container_name: Name of the MySQL container

    Returns:
        Tuple of (exists: bool, is_running: bool, message: str)

    """
    try:
        # Check if container exists
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception as we're handling the output
        )

        exists = container_name in result.stdout

        if not exists:
            return False, False, "MySQL container does not exist"

        # Check if container is running
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"name={container_name}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception as we're handling the output
        )

        is_running = container_name in result.stdout

        if is_running:
            # Check port accessibility
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                is_port_open = s.connect_ex(("localhost", 3306)) == 0
                if is_port_open:
                    return True, True, "MySQL container is running and accessible"
                return (
                    True,
                    True,
                    "MySQL container is running but port 3306 is not accessible",
                )

        return True, False, "MySQL container exists but is not running"

    except Exception as e:
        return False, False, f"Error checking MySQL container status: {str(e)}"


def start_existing_container(
    container_name: str = "ethopy_sql_db", max_retries: int = 5
) -> bool:
    """Start an existing MySQL container.

    Args:
        container_name: Name of the MySQL container
        max_retries: Maximum number of attempts to verify container is running

    Returns:
        bool: True if container started successfully

    """
    try:
        # Check if container is running
        # container_name = find_mysql_container()
        subprocess.run(
            ["docker", "start", f"mysql-docker-{container_name}-1"],
            capture_output=True,
            check=True,  # We want to know if the start command fails
        )

        # Wait for container to be fully running
        for _ in range(max_retries):
            _, is_running, _ = check_mysql_container(container_name)
            if is_running:
                return True
            sleep(2)  # Wait 2 seconds before next check

        return False

    except subprocess.CalledProcessError:
        return False


@click.command()
@click.option(
    "--mysql-path", type=click.Path(), help="Path to store MySQL Docker files"
)
@click.option(
    "--container-name", default="ethopy_sql_db", help="Name for the MySQL container"
)
@click.help_option("-h", "--help")
def setup_dj_docker(mysql_path: Optional[str], container_name: str) -> None:
    """Initialize the database environment using Docker.

    This command sets up a MySQL database in Docker, configures it for use with
    ethopy, and prepares the initial environment.
    """
    # Check Docker status first
    docker_running, docker_message = check_docker_status()
    if not docker_running:
        raise click.ClickException(f"Docker check failed: {docker_message}")

    # Check existing container status
    exists, is_running, message = check_mysql_container(container_name)

    if exists:
        if is_running:
            click.echo(f"MySQL container is already running: {message}")
            return
        else:
            click.echo("Found existing MySQL container, attempting to start it...")
            if start_existing_container(container_name):
                click.echo("Successfully started existing MySQL container")
                return
            click.echo("Failed to start existing container")
            return

    try:
        # Determine MySQL setup directory
        if mysql_path:
            mysql_dir = Path(mysql_path)
        else:
            # Default to user's home directory
            mysql_dir = Path.home() / ".ethopy" / "mysql-docker"

        # Create directory for MySQL Docker setup
        mysql_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(str(mysql_dir))

        # MySQL only honours MYSQL_ROOT_PASSWORD when it initialises an empty
        # data directory, so a leftover one silently keeps its old credentials.
        data_dir = mysql_dir / f"data_{container_name}"
        if data_dir.exists():
            click.echo(
                f"WARNING: a MySQL data directory already exists at {data_dir}.\n"
                "MySQL will reuse it and IGNORE the password entered below - the "
                "root password stored in that directory stays in force.\n"
                "Enter that existing password, or press Ctrl-C and move the "
                "directory aside to start from a clean database."
            )

        # Get password securely using Click's password prompt
        mysql_password = click.prompt(
            "Enter the MySQL root password", hide_input=True, confirmation_prompt=True
        )

        docker_content = (
            f"services:\n"
            f"  {container_name}:\n"
            f"    image: datajoint/mysql:8\n"
            f"    environment:\n"
            f"    - MYSQL_ROOT_PASSWORD={mysql_password}\n"
            f"    ports:\n"
            f"    - '3306:3306'\n"
            f"    volumes:\n"
            f"    - ./data_{container_name}:/var/lib/mysql"
        )

        with open("docker-compose.yaml", "w") as f:
            f.write(docker_content)

        click.echo("Starting Docker container...")
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True,
            check=True,  # Important for container startup
        )

        # Verify the container started successfully
        for _ in range(5):  # Try 5 times
            _, is_running, message = check_mysql_container(container_name)
            if is_running:
                click.echo("MySQL container started successfully")
                return
            sleep(2)  # Wait 2 seconds before next check

        raise click.ClickException("Failed to verify MySQL container is running")

    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Command failed: {e.cmd}")
    except Exception as e:
        raise click.ClickException(f"Error during initialization: {str(e)}")


def check_db_connection() -> None:
    """Check if connection with db is established."""
    from ethopy import local_conf

    try:
        dj.config.update(local_conf.get("dj_local_conf"))
        dj.logger.setLevel(local_conf.get("dj_local_conf")["datajoint.loglevel"])
        _ = dj.conn()
    except Exception:
        logging.error("Failed to connect to database")
        raise Exception(f"Failed to connect to database {dj.config['database.host']}")

    logging.info(
        f"Connected to {dj.config['database.user']}@{dj.config['database.host']} !!"
    )


def _plugin_import_commands() -> List[Tuple[str, str]]:
    """Build the import commands for every registered user plugin.

    The wildcard imports used for the core package only cover modules shipped
    inside ethopy, so plugin modules have to be imported explicitly for their
    tables to be declared.

    Returns:
        List of (name, import command) tuples, one per plugin module.

    """
    from ethopy import plugin_manager

    commands = []
    for plugins in plugin_manager.list_plugins(include_core=False).values():
        for plugin in plugins:
            import_path = plugin["import_path"]
            commands.append((f"plugin/{import_path}", f"import {import_path}"))

    return commands


def _run_import(schema_name: str, cmd: str) -> Optional[str]:
    """Run a single import command in a subprocess to declare its tables.

    Args:
        schema_name: Name used to report the result to the user.
        cmd: Python import statement to execute.

    Returns:
        None on success, otherwise the formatted error message.

    """
    try:
        # Capture both stdout and stderr
        _ = subprocess.run(
            [sys.executable, "-c", cmd], check=True, capture_output=True, text=True
        )
        click.echo(f"Successfully created tables for: {schema_name}")
        return None

    except subprocess.CalledProcessError as e:
        return f"""
                    Failed to create schema: {schema_name}
                    Command: {cmd}
                    Error output: {e.stderr}
                    """


def createschema() -> None:
    """Create all required database schemas.

    This function:
    1. Verifies database connection
    2. Creates schemas in the correct order
    3. Provides detailed feedback for each step

    Raises:
        ClickException: If schema creation fails with detailed error message

    """
    check_db_connection()
    logging.info("Creating schemas and tables...")

    # Import commands in dependency order
    import_commands = [
        ("core/experiment", "from ethopy.core.experiment import *"),
        ("core/stimulus", "from ethopy.core.stimulus import *"),
        ("core/interface", "from ethopy.core.interface import *"),
        ("core/behavior", "from ethopy.core.behavior import *"),
        ("core/recordings", "from ethopy.core.recordings import *"),
        ("core/mice", "from ethopy.core.mice import *"),
        ("stimuli", "from ethopy.stimuli import *"),
        ("behaviors", "from ethopy.behaviors import *"),
        ("experiments", "from ethopy.experiments import *"),
        ("interfaces", "from ethopy.interfaces import *"),
    ]

    for schema_name, cmd in import_commands:
        error = _run_import(schema_name, cmd)
        if error:
            raise click.ClickException(error)

    # Plugins are optional and often need hardware specific dependencies, so a plugin
    # that fails to import only warns instead of aborting the whole setup.
    failed_plugins = []
    for schema_name, cmd in _plugin_import_commands():
        error = _run_import(schema_name, cmd)
        if error:
            logging.warning(error)
            failed_plugins.append(schema_name)

    if failed_plugins:
        click.echo(
            f"\nSkipped {len(failed_plugins)} plugin(s) that failed to import: "
            f"{', '.join(failed_plugins)}\n"
            "See the log for the full error of each one."
        )
