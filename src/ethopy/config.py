"""Simple configuration management for ethopy package.

Provides a straightforward way to load, access, and save configuration settings
while maintaining compatibility with the original configuration structure.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigurationManager:
    """Configuration manager for ethopy package settings."""

    DEFAULT_FILE = Path.home() / ".ethopy" / "local_conf.json"
    DEFAULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_file: Optional path to config file. If not provided,
                        uses ~/.ethopy/local_conf.json

        """
        self.logging = logging.getLogger(__name__)
        self._config: Dict[str, Any] = {}

        # Set up config file path
        self.config_file = (
            Path(config_file)
            if config_file
            else Path.home() / ".ethopy" / "local_conf.json"
        )
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self._load_config()

        # Set up default values if not present
        self._set_defaults()

    def _set_defaults(self) -> None:
        """Set default values for essential settings if not present."""
        defaults = {
            "dj_local_conf": {
                "database.host": "127.0.0.1",
                "database.user": "root",
                "database.password": "",
                "database.port": 3306,
                "database.reconnect": True,
                "database.use_tls": False,
                "datajoint.loglevel": "WARNING",
            },
            "SCHEMATA": {
                "experiment": "lab_experiments",
                "stimulus": "lab_stimuli",
                "behavior": "lab_behavior",
                "interface": "lab_interface",
                "recording": "lab_recordings",
                "mice": "lab_mice",
            },
            "logging": {
                "level": "INFO",
                "directory": str(Path.home() / ".ethopy"),
                "filename": "ethopy.log",
                "max_size": 31457280,
                "backup_count": 5,
            },
            "source_path": str(Path.home() / "EthoPy_Files"),
            "target_path": "/",
            "plugin_path": str(Path.home() / ".ethopy/ethopy_plugins"),
        }

        # Update config with defaults for missing values only
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
            elif isinstance(value, dict) and isinstance(self._config[key], dict):
                # Deep update for nested dictionaries
                self._config[key].update(
                    {k: v for k, v in value.items() if k not in self._config[key]}
                )

    def _format_value(self, value: Any, indent: int = 0) -> str:
        """Format a configuration value for display."""
        indent_str = "  " * indent

        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = []
            for k, v in value.items():
                lines.append(f"{indent_str}{k}: {self._format_value(v, indent + 1)}")
            return "\n" + "\n".join(lines)

        if isinstance(value, str):
            return f"'{value}'"
        return str(value)

    def __str__(self) -> str:
        """Return a formatted string of all configuration settings."""
        lines = ["Configuration:", f"Config File: {self.config_file}", "-" * 50]

        for key, value in self._config.items():
            lines.append(f"{key}: {self._format_value(value, 1)}")

        return "\n".join(lines)

    def get_datajoint_config(self) -> Dict[str, Any]:
        """Get the DataJoint configuration as a dictionary.

        Returns:
            Dictionary containing all DataJoint configuration settings.

        """
        return self._config.get("dj_local_conf", {})

    def _load_config(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                self._config = json.loads(self.config_file.read_text())
                self.logging.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                self.logging.error(f"Error loading config: {e}")
                self._config = {}
        else:
            self.logging.info("No configuration file found, using defaults")
            self._config = {}

    def save(self) -> None:
        """Save current configuration to file."""
        try:
            if self.DEFAULT_FILE != self._config:
                self.DEFAULT_FILE.write_text(json.dumps(self._config, indent=4))
            self.config_file.write_text(json.dumps(self._config, indent=4))
            self.logging.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            self.logging.error(f"Error saving config: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'database.host', 'logging.level')
            default: Default value if key not found

        Returns:
            Configuration value or default if not found

        """
        # Handle top-level keys first
        if key in self._config:
            return self._config[key]

        # Handle nested keys
        try:
            # Check in dj_local_conf first
            if key in self._config.get("dj_local_conf", {}):
                return self._config["dj_local_conf"][key]

            # Then try normal dot notation
            value = self._config
            for part in key.split("."):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'database.host', 'logging.level')
            value: Value to set

        """
        if "." not in key:
            self._config[key] = value
            return

        # For nested keys
        parts = key.split(".")
        target = self._config

        # Navigate to the nested location
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values at once."""
        for key, value in updates.items():
            self.set(key, value)
