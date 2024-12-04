from os import environ

import datajoint as dj

from ethopy.utils.config import ConfigurationManager

__version__ = "0.0.1"

# Set environment variables
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

config_manager = ConfigurationManager()
# set the datajoint parameters
dj.config.update(config_manager.db.__dict__)
dj.logger.setLevel(dj.config["loglevel"])

__all__ = ["config_manager"]
