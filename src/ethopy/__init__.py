"""ethopy package initializer.

This module initializes the ethopy package by setting up environment variables,
reading configuration files, setting DataJoint parameters, and initializing plugins.

Attributes:
    __version__ (str): The version of the ethopy package.
    local_conf (ConfigurationManager): The configuration manager instance for reading
        local configuration(local_conf.json).
    SCHEMATA (dict): The schema mappings from the local configuration.
    plugin_manager (PluginManager): The plugin manager instance for managing plugins.

Environment Variables:
    PYGAME_HIDE_SUPPORT_PROMPT (str): Set to "1" to hide the Pygame support prompt.

Modules:
    ConfigurationManager: Manages configuration settings.
    PluginManager: Manages plugins for the ethopy package.

"""

from os import environ

import datajoint as dj

from ethopy.config import ConfigurationManager
from ethopy.plugin_manager import PluginManager

__version__ = "0.0.8.1"

# Set environment variables
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# read the local_conf file
local_conf = ConfigurationManager()

# set the datajoint parameters
dj.config.update(local_conf.get("dj_local_conf"))
dj.logger.setLevel(local_conf.get("dj_local_conf")["datajoint.loglevel"])
# Schema mappings
SCHEMATA = local_conf.get("SCHEMATA")

# Initialize plugins
plugin_manager = PluginManager(local_conf.get("plugin_path"))

__all__ = ["local_conf", "plugin_manager", "__version__", "SCHEMATA"]
