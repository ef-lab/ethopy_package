"""Command-line interface for EthoPy using Click."""

import logging
from pathlib import Path

import click

from ethopy.utils.config import ConfigurationManager
from ethopy.utils.ethopy_logging import LoggingManager
from ethopy.utils.task import resolve_task

# Create a single instance for the entire application
log_manager = LoggingManager("ethopy")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-p",
    "--task-path",
    type=str,
    help="Path to task file",
)
@click.option(
    "-i",
    "--task-id",
    type=int,
    help="Task ID from database",
)
@click.option("--debug", is_flag=True, help="Enable debug mode with verbose logging")
@click.option(
    "--log-console",
    is_flag=True,
    help="Enable console logging",
)
@click.version_option()
def main(task_path: Path, task_id: int, debug: bool, log_console: bool) -> None:
    """EthoPy - Behavioral Training Control System."""
    # Load configuration
    local_conf = ConfigurationManager()

    # Configure logging before anything else happens
    log_manager.configure(
        log_dir=local_conf.get("logging")["directory"],
        console=log_console,
        log_level=local_conf.get("logging")["level"],
        log_file=local_conf.get("logging")["filename"],
    )

    # Import run after logging is configured
    from ethopy.run import run

    try:
        task = resolve_task(task_path, task_id)
        run(task)
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        raise click.BadParameter(str(e))


if __name__ == "__main__":
    main()
