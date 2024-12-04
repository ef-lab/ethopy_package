"""
Command-line interface for EthoPy using Click.

This module provides a user-friendly CLI for running EthoPy experiments.
"""
from pathlib import Path

import click


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '-c', '--config',
    default=None,
    help='Path to configuration file',
)
@click.option(
    '--debug',
    is_flag=True,
    help='Enable debug mode with verbose logging'
)
@click.version_option()
def main(config: Path, debug: bool) -> None:
    """EthoPy - Behavioral Training Control System."""
    from ethopy.run import run
    if config:
        # Run specific config file
        run(config)
    else:
        # Run default run.py
        run()


if __name__ == "__main__":
    main()
