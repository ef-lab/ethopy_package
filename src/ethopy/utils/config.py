import json
import logging
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ethopy.utils.exceptions import ConfigurationError
from ethopy.utils.logging import setup_logging


@dataclass
class DatabaseConfig:
    """Database configuration settings"""

    host: str = "127.0.0.1"
    user: str = "root"
    password: str = ""
    port: int = 3306
    reconnect: bool = True
    use_tls: bool = False
    loglevel: str = "WARNING"

    @property
    def to_dict(self) -> Dict[str, Any]:
        return {
            "database.host": self.host,
            "database.user": self.user,
            "database.password": self.password,
            "database.port": self.port,
            "database.reconnect": self.reconnect,
            "database.use_tls": self.use_tls,
            "datajoint.loglevel": self.loglevel,
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            attr_name = key.split(".")[-1]
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)


@dataclass
class SchemaConfig:
    """Schema configuration settings"""

    experiment: str = "lab_experiments"
    stimulus: str = "lab_stimuli"
    behavior: str = "lab_behavior"
    recording: str = "lab_recordings"
    mice: str = "lab_mice"

    @property
    def to_dict(self) -> Dict[str, str]:
        return {key: getattr(self, key) for key in self.__annotations__}

    def update_from_dict(self, data: Dict[str, str]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class PathConfig:
    """Path configuration settings with support for custom paths"""
    source_path: Path = Path.home() / "EthoPy_Files"
    target_path: Optional[Path] = None
    plugin_path: Optional[Path] = None
    custom_paths: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string paths to Path objects"""
        self.source_path = Path(self.source_path)
        if self.target_path:
            self.target_path = Path(self.target_path)

        if self.plugin_path:
            self.plugin_path = Path(self.plugin_path)

        for key, value in self.custom_paths.items():
            self.custom_paths[key] = Path(value)

    def add_path(self, name: str, path: Union[str, Path], create: bool = True) -> None:
        """Add a custom path to the configuration"""
        path_obj = Path(path)
        if create:
            path_obj.mkdir(parents=True, exist_ok=True)
        self.custom_paths[name] = path_obj

    def get_path(self, name: str) -> Optional[Path]:
        """Get a path by name"""
        return self.custom_paths.get(name)

    @property
    def to_dict(self) -> Dict[str, Any]:
        base_paths = {
            "source_path": str(self.source_path),
            "target_path": str(self.target_path) if self.target_path else None,
            "plugin_path": str(self.plugin_path) if self.plugin_path else None,
        }
        custom_paths = {
            f"path_{key}": str(value) for key, value in self.custom_paths.items()
        }
        return {**base_paths, **custom_paths}


class ConfigurationManager:
    """
    Configuration manager using consistent paths across all operating systems.
    All configurations are stored in the user's home directory under .ethopy
    """

    def __init__(self, import_config: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            import_config: Optional path to existing configuration file to import
        """
        setup_logging(console_log=True, name=__name__)
        self.logging = logging
        self.db = DatabaseConfig()
        self.schema = SchemaConfig()
        self.paths = PathConfig()
        self.custom_params: Dict[str, Any] = {}

        # Set up unified paths for all operating systems
        self._setup_paths()

        # Create configuration directory if it doesn't exist
        os.makedirs(self.CONFIG_DIR, exist_ok=True)

        # Import or load configuration
        if import_config:
            self.import_configuration(import_config)
        else:
            self._load_configuration()

    def _setup_paths(self) -> None:
        """
        Set up unified configuration paths for all operating systems.
        All paths are relative to user's home directory.
        """
        # Base configuration directory in user's home
        self.CONFIG_DIR = self._normalize_path("~/.ethopy")

        # Configuration file paths
        self.CONFIG_PATH = self.CONFIG_DIR / "local_conf.json"

        # Environment variable for overriding config location
        self.ENV_CONFIG_PATH = "ETHOPY_CONFIG"

    def _normalize_path(self, path: Union[str, Path]) -> Path:
        """
        Normalize path and resolve user home directory.
        """
        return Path(os.path.expanduser(str(path))).resolve()

    def _load_configuration(self) -> None:
        """Load configuration from file or use defaults"""
        config_path = os.getenv(self.ENV_CONFIG_PATH, self.CONFIG_PATH)
        config_path = self._normalize_path(config_path)

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self._apply_config(config)
                self.logging.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logging.error(f"Error loading config from {config_path}: {e}")
                raise ConfigurationError(f"Failed to load configuration: {e}")
        else:
            self.logging.info("No configuration file found. Using defaults.")

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration from dictionary to the current settings.

        Args:
            config: Dictionary containing configuration settings
        """
        # Update database configuration
        if "dj_local_conf" in config:
            self.db.update_from_dict(config["dj_local_conf"])

        # Update schema configuration
        if "SCHEMATA" in config:
            self.schema.update_from_dict(config["SCHEMATA"])

        for key, value in config.items():
            if key.startswith("path_"):
                # Handle custom paths (path_video, path_interface, etc.)
                path_name = key[5:]  # Remove 'path_' prefix
                self.paths.add_path(path_name, value, create=True)
            elif key in ["source_path", "target_path", "plugin_path"]:
                # Handle standard paths
                setattr(self.paths, key, Path(value))
            elif key not in ["dj_local_conf", "SCHEMATA"]:
                # Store any other parameters as custom parameters
                self.custom_params[key] = value

        self.logging.debug("Applied configuration settings successfully")

    def import_configuration(self, config_path: Union[str, Path]) -> None:
        """
        Import an existing configuration file.

        Args:
            config_path: Path to the configuration file to import
        """
        source_path = self._normalize_path(config_path)
        if not source_path.exists():
            raise ConfigurationError(f"Configuration file not found: {source_path}")

        try:
            # Verify JSON is valid
            with open(source_path, "r") as f:
                config = json.load(f)

            # Copy to config directory
            shutil.copy2(source_path, self.CONFIG_PATH)

            # Set environment variable
            os.environ[self.ENV_CONFIG_PATH] = str(self.CONFIG_PATH)

            self.logging.info(
                f"Imported configuration from {source_path} to {self.CONFIG_PATH}"
            )

            # Load the configuration
            self._load_configuration()

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error importing configuration: {e}")

    def save_configuration(self) -> None:
        """Save current configuration to file"""
        config = {
            "dj_local_conf": self.db.to_dict,
            "SCHEMATA": self.schema.to_dict,
            **self.paths.to_dict,
            **self.custom_params,
        }

        try:
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=4)
            self.logging.info(f"Saved configuration to {self.CONFIG_PATH}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def add_custom_param(self, name: str, value: Any) -> None:
        """Add a custom parameter"""
        self.custom_params[name] = value

    def get_custom_param(self, name: str, default: Any = None) -> Any:
        """Get a custom parameter value"""
        return self.custom_params.get(name, default)

    def get_dict(self) -> Dict[str, Any]:
        """
        Get the complete configuration as a dictionary.
        This includes database settings, schema settings, paths and custom parameters.

        Returns:
            Dict[str, Any]: Complete configuration dictionary
        """
        try:
            config = {
                "dj_local_conf": self.db.to_dict,
                "SCHEMATA": self.schema.to_dict,
                **self.paths.to_dict,
                **self.custom_params
            }
            return deepcopy(config)  # Return a deep copy to prevent accidental modifications
        except Exception as e:
            logging.error(f"Error creating configuration dictionary: {e}")
            raise ConfigurationError(f"Failed to create configuration dictionary: {e}")
        
    def display_config(self) -> None:
        """
        Display the current configuration and optionally save it to a file.

        Args:
            file_path: Optional path to save the configuration display
        """
        config_str = self.get_dict()

        # Always print to console
        print(f"Current Configuration (path: {self.CONFIG_PATH}):")
        print("---------------------")
        print(config_str)
        print("---------------------")

    def get(self, variable_name: str, default: Any = None) -> Optional[Any]:
        """
        Get any variable from the configuration.
        Returns None if the variable doesn't exist.

        Args:
            variable_name: Name of the variable to retrieve

        Returns:
            The value of the variable if it exists, None otherwise
        """
        # Check in paths first (including custom paths)
        if variable_name.startswith("path_"):
            return self.paths.get_path(variable_name[5:])  # Remove 'path_' prefix
        elif hasattr(self.paths, variable_name):
            return getattr(self.paths, variable_name)

        # Check in custom parameters
        if variable_name in self.custom_params:
            return self.custom_params[variable_name]

        # Check in database config
        if hasattr(self.db, variable_name):
            return getattr(self.db, variable_name)

        # Check in schema config
        if hasattr(self.schema, variable_name):
            return getattr(self.schema, variable_name)

        if default:
            return default
        return None
