from os import environ

import datajoint as dj

from ethopy.plugin_manager import plugin_manager
from ethopy.utils.config import ConfigurationManager

__version__ = "0.0.1"

# Set environment variables
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

config_manager = ConfigurationManager()
# set the datajoint parameters
dj.config.update(config_manager.db.to_dict)
dj.logger.setLevel(config_manager.db.loglevel)
# Schema mappings
SCHEMATA = config_manager.schema.__dict__

# Initialize plugins
plugin_manager  # This will trigger plugin loading

__all__ = ["config_manager",
           "plugin_manager",
           "__version__",
           "SCHEMATA"]
