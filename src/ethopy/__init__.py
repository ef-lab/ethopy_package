from os import environ

import datajoint as dj

from ethopy.plugin_manager import plugin_manager
from ethopy.utils.config import ConfigurationManager

__version__ = "0.0.1"

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
plugin_manager  # This will trigger plugin loading

__all__ = ["local_conf",
           "plugin_manager",
           "__version__",
           "SCHEMATA"]
