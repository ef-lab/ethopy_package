"""Command-line interface for EthoPy using Click."""

from pathlib import Path
import click
from ethopy.utils.config import ConfigurationManager
from ethopy.utils.logging import LoggingManager

# Create a single instance for the entire application
log_manager = LoggingManager("ethopy")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("-c", "--config", default=None, help="Path to configuration file")
@click.option("--debug", is_flag=True, help="Enable debug mode with verbose logging")
@click.option(
    "--log-console",
    is_flag=True,
    help="Enable console logging",
)
@click.version_option()
def main(config: Path, debug: bool, log_console: bool) -> None:
    """EthoPy - Behavioral Training Control System."""
    # Now load your config and continue
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

    if config:
        run(config)
    else:
        run()
