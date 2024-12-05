import logging
import os
import socket
import subprocess
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple

import click
import datajoint as dj


def check_docker_status() -> Tuple[bool, str]:
    """
    Check if Docker daemon is running and accessible.

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


def check_mysql_container(container_name: str = "mysql") -> Tuple[bool, bool, str]:
    """
    Check if MySQL container exists and its status.

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


def find_mysql_container() -> str:
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "'{{.Names}}'"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Split by newline to get list of container names
    containers = result.stdout.strip().split("\n")

    # Find first container with 'mysql' in the name
    mysql_container = next(
        (name.strip("'") for name in containers if "mysql" in name.lower()),
        "",  # Default value if no mysql container found
    )

    return mysql_container


def start_existing_container(
    container_name: str = "mysql", max_retries: int = 5
) -> bool:
    """
    Start an existing MySQL container.

    Args:
        container_name: Name of the MySQL container
        max_retries: Maximum number of attempts to verify container is running

    Returns:
        bool: True if container started successfully
    """
    try:
        # Check if container is running
        name = container_name = find_mysql_container()
        subprocess.run(
            ["docker", "start", name],
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
@click.option("--container-name", default="mysql", help="Name for the MySQL container")
def setup_dj_docker(mysql_path: Optional[str], container_name: str):
    """
    Initialize the database environment using Docker.

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
            click.echo(
                "Failed to start existing container, will attempt to create new one"
            )

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

        # Get password securely using Click's password prompt
        mysql_password = click.prompt(
            "Enter the MySQL root password", hide_input=True, confirmation_prompt=True
        )

        click.echo("Downloading Docker configuration...")
        subprocess.run(
            [
                "wget",
                "https://raw.githubusercontent.com/datajoint/mysql-docker/master/docker-compose.yaml",
            ],
            capture_output=True,
            check=True,  # Important for download to succeed
        )

        # Update docker-compose file with container name and password
        with open("docker-compose.yaml", "r") as f:
            content = f.read()
        content = content.replace(
            "MYSQL_ROOT_PASSWORD=simple", f"MYSQL_ROOT_PASSWORD={mysql_password}"
        )
        # Add container name if not default
        if container_name != "mysql":
            content = content.replace(
                "container_name: mysql", f"container_name: {container_name}"
            )
        with open("docker-compose.yaml", "w") as f:
            f.write(content)

        click.echo("Starting Docker container...")
        subprocess.run(
            ["docker-compose", "up", "-d"],
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


def get_import_commands() -> List[str]:
    """
    Get list of commands to create schemas in correct order.
    This preserves the original schema creation logic while making it accessible
    through the CLI interface.
    """
    return [
        "from ethopy.core.experiment import *",
        "from ethopy.core.stimulus import *",
        "from ethopy.core.behavior import *",
        "from ethopy.stimuli import *",
        "from ethopy.behaviors import *",
        "from ethopy.experiments import *",
    ]


def createschema():
    """
    Create all required database schemas.

    This command imports and initializes all schema definitions in the correct order,
    setting up the database structure needed by ethopy.
    """
    # Try to establish connection
    from ethopy import config_manager
    try:
        dj.config.update(config_manager.get_dict()['dj_local_conf'])
        dj.logger.setLevel(config_manager.db.loglevel)
        conn = dj.conn()
    except Exception:
        logging.error(f"Failed to connect to database")
        raise Exception(f"Failed to connect to database {dj.config['database.host']}")

    logging.info("Creating schemas and tables...")

    for cmd in get_import_commands():
        try:
            subprocess.run(["python", "-c", cmd], check=True)
            click.echo(f"Successfully executed: {cmd}")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Error executing {cmd}: {str(e)}")
