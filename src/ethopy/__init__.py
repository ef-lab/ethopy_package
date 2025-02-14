from os import environ

import datajoint as dj

from ethopy.plugin_manager import PluginManager
from ethopy.utils.config import ConfigurationManager

__version__ = "0.0.5"

# Set environment variables
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# read the local_conf file
local_conf = ConfigurationManager()

# set the datajoint parameters
dj.config.update(local_conf.get("dj_local_conf"))
dj.logger.setLevel(local_conf.get("dj_local_conf")['datajoint.loglevel'])
# Schema mappings
SCHEMATA = local_conf.get("SCHEMATA")

# Initialize plugins
plugin_manager = PluginManager(local_conf.get("plugin_path"))

__all__ = ["local_conf",
           "plugin_manager",
           "__version__",
           "SCHEMATA"]
