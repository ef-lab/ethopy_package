# ethopy/core/plugin_manager.py
import sys
import os
from pathlib import Path
import importlib
import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


@dataclass
class PluginInfo:
    """Information about a discovered plugin"""

    name: str
    path: str
    type: str  # 'standalone', 'core', or one of PLUGIN_CATEGORIES
    import_path: str  # Full import path (e.g., 'ethopy.mymodule')
    is_core: bool = False  # True if it's from main ethopy package


class PluginManager:
    """
    Plugin manager for ethopy that enables dynamic loading of user plugins.
    Handles both standalone modules and structured plugins with duplicate detection,
    including checks against the main ethopy package.
    """

    PLUGIN_CATEGORIES = ["behaviors", "experiments", "interfaces", "stimuli"]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._plugin_paths: Set[str] = set()
        self._plugins: Dict[str, PluginInfo] = {}  # import_path -> PluginInfo
        self._duplicates: Dict[str, List[str]] = (
            {}
        )  # import_path -> list of duplicate paths

        # Get ethopy's main package path
        self._ethopy_path = self._get_ethopy_path()

        # First scan core ethopy modules
        self._scan_core_modules()

        # Then load plugins from default locations
        self._setup_plugin_paths()
        self._initialize_plugin_imports()

    def _get_ethopy_path(self) -> Optional[str]:
        """Get the installation path of the main ethopy package"""
        try:
            import ethopy

            return os.path.dirname(ethopy.__file__)
        except ImportError:
            self.logger.warning("Could not find main ethopy package")
            return None

    def _scan_core_modules(self):
        """Scan the main ethopy package for modules"""
        if not self._ethopy_path:
            return

        def scan_package(package_path: str, package_name: str = "ethopy"):
            """Recursively scan a package for modules"""
            if not os.path.isdir(package_path):
                return

            # Check all items in the directory
            for item in os.listdir(package_path):
                item_path = os.path.join(package_path, item)

                # Skip special files/dirs
                if item.startswith("_"):
                    continue

                if item.endswith(".py"):
                    # Register Python module
                    module_name = item[:-3]
                    import_path = f"{package_name}.{module_name}"
                    self._register_plugin(
                        import_path=import_path,
                        plugin_path=item_path,
                        plugin_type="core",
                        is_core=True,
                    )

                elif os.path.isdir(item_path):
                    # If it's a category directory or has __init__.py, scan it
                    if item in self.PLUGIN_CATEGORIES or os.path.exists(
                        os.path.join(item_path, "__init__.py")
                    ):
                        scan_package(item_path, f"{package_name}.{item}")

        scan_package(self._ethopy_path)

    def _setup_plugin_paths(self):
        """Setup default plugin paths and from environment variable"""
        default_paths = [
            Path.home() / "ethopy_plugins",
            Path.cwd() / "ethopy_plugins",
        ]

        env_paths = [
            p for p in os.environ.get("ETHOPY_PLUGIN_PATH", "").split(",") if p
        ]

        # Add paths in order of precedence (later paths override earlier ones)
        for path in [*default_paths, *env_paths]:
            if path and str(path).strip():
                self.add_plugin_path(str(path))

    def _register_plugin(
        self,
        import_path: str,
        plugin_path: str,
        plugin_type: str,
        is_core: bool = False,
    ):
        """
        Register a plugin, handling duplicates with warnings.

        Args:
            import_path: Full import path (e.g., 'ethopy.mymodule')
            plugin_path: Path to the plugin file
            plugin_type: 'standalone', 'core', or category name
            is_core: Whether this is from the main ethopy package
        """
        name = import_path.split(".")[-1]

        if import_path in self._plugins:
            existing = self._plugins[import_path]

            # Handle core package conflicts
            if is_core and not existing.is_core:
                self.logger.warning(
                    f"Plugin '{import_path}' from {plugin_path} conflicts with core ethopy "
                    f"module. Core module will be used."
                )
                return
            elif existing.is_core and not is_core:
                self.logger.warning(
                    f"Plugin '{import_path}' from {plugin_path} conflicts with core ethopy "
                    f"module at {existing.path}. Core module will be used."
                )
                return

            # Handle plugin conflicts
            if import_path not in self._duplicates:
                self._duplicates[import_path] = [existing.path]
            self._duplicates[import_path].append(plugin_path)

            self.logger.warning(
                f"Duplicate plugin found for '{import_path}':\n"
                f"  Using:     {plugin_path}\n"
                f"  Ignoring:  {existing.path}"
            )

        # Register the plugin
        self._plugins[import_path] = PluginInfo(
            name=name,
            path=plugin_path,
            type=plugin_type,
            import_path=import_path,
            is_core=is_core,
        )

        # Register the plugin (overwriting any existing one)
        self._plugins[import_path] = PluginInfo(
            name=name, path=plugin_path, type=plugin_type, import_path=import_path
        )

    def add_plugin_path(self, path: str) -> None:
        """
        Add a new plugin directory to the system.

        Args:
            path: Directory path containing plugins
        """
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            # self.logger.warning(f"Plugin path not found: {path}")
            return

        if path not in self._plugin_paths:
            self._plugin_paths.add(path)

            # Add to Python's import path
            if path not in sys.path:
                sys.path.insert(0, path)

            self.logger.info(f"Added plugin path: {path}")

            # Scan for plugins
            self._scan_plugins(path)

            # Refresh import cache
            importlib.invalidate_caches()

    def _scan_plugins(self, path: str):
        """Scan directory for plugins and register them"""
        # Scan standalone modules
        for item in os.listdir(path):
            if item.endswith(".py") and not item.startswith("_"):
                module_name = item[:-3]
                import_path = f"ethopy.{module_name}"
                plugin_path = os.path.join(path, item)
                self._register_plugin(import_path, plugin_path, "standalone")

        # Scan category plugins
        for category in self.PLUGIN_CATEGORIES:
            category_path = os.path.join(path, category)
            if os.path.isdir(category_path):
                for item in os.listdir(category_path):
                    if item.endswith(".py") and not item.startswith("_"):
                        module_name = item[:-3]
                        import_path = f"ethopy.{category}.{module_name}"
                        plugin_path = os.path.join(category_path, item)
                        self._register_plugin(import_path, plugin_path, category)

    def _initialize_plugin_imports(self):
        """Initialize the plugin import hook system"""

        class EthopyPluginFinder:
            """Custom import finder for ethopy plugins"""

            def __init__(self, plugin_manager):
                self.plugin_manager = plugin_manager

            def find_spec(self, fullname: str, path=None, target=None):
                if not fullname.startswith("ethopy."):
                    return None

                # Check if this is a registered plugin
                plugin_info = self.plugin_manager._plugins.get(fullname)
                if plugin_info:
                    return importlib.util.spec_from_file_location(
                        fullname, plugin_info.path
                    )

                return None

        # Register our custom finder
        sys.meta_path.insert(0, EthopyPluginFinder(self))

    def list_plugins(
        self, show_duplicates: bool = False, include_core: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        List all available plugins.

        Args:
            show_duplicates: If True, include information about duplicate plugins
            include_core: If True, include core ethopy modules

        Returns:
            Dictionary with plugin categories and their plugins
        """
        result = {
            "standalone": [],
            **{category: [] for category in self.PLUGIN_CATEGORIES},
        }

        if include_core:
            result["core"] = []

        for plugin in self._plugins.values():
            # Skip core modules if not requested
            if not include_core and plugin.is_core:
                continue

            plugin_info = {
                "name": plugin.name,
                "path": plugin.path,
                "is_core": plugin.is_core,
            }

            if show_duplicates and plugin.import_path in self._duplicates:
                plugin_info["duplicates"] = self._duplicates[plugin.import_path]

            result[plugin.type].append(plugin_info)

        return result

    def get_plugin_info(self, import_path: str) -> PluginInfo:
        """Get information about a specific plugin"""
        return self._plugins.get(import_path)


# Global instance
plugin_manager = PluginManager()
