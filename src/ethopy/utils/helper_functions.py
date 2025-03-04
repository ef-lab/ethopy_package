import base64
import functools
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from getpass import getpass
from itertools import product
from typing import Any, Dict, List, Tuple
import subprocess
import importlib.metadata
import datajoint as dj
import numpy as np
from scipy import ndimage
import platform
import socket

log = logging.getLogger(__name__)


@dataclass
class FillColors:
    """Color configuration for different stimulus states.

    Attributes:
        start: Color for start state
        ready: Color for ready state
        reward: Color for reward state
        punish: Color for punish state
        background: Color for background

    """

    start: Tuple[int, int, int] = field(default=())
    ready: Tuple[int, int, int] = field(default=())
    reward: Tuple[int, int, int] = field(default=())
    punish: Tuple[int, int, int] = field(default=())
    background: Tuple[int, int, int] = (0, 0, 0)

    def set(self, dictionary: Dict[str, Any]) -> None:
        """Update color attributes from a dictionary.

        Args:
            dictionary: Dictionary containing color values to update.
                      Keys should match existing attributes.

        """
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"FillColors has no attribute '{key}'")

    def values(self):
        """Get all color values.

        Returns:
            A dictionary_values object containing all color values.

        """
        return self.__dict__.values()


def create_virtual_modules(schemata, create_tables=True, create_schema=True):
    try:
        if dj.config["database.password"] is None:
            dj.config["password"] = getpass(prompt="Please enter DataJoint password: ")
        # Create virtual modules
        _conn = dj.Connection(
            dj.config["database.host"],
            dj.config["database.user"],
            dj.config["database.password"],
            use_tls=dj.config["database.use_tls"],
        )
        virtual_modules = {}
        for name, schema in schemata.items():
            virtual_modules[name] = dj.create_virtual_module(
                name,
                schema,
                create_tables=create_tables,
                create_schema=create_schema,
                connection=_conn,
            )
        return virtual_modules, _conn
    except Exception as e:
        error_message = (
            f"Failed to connect to the database due "
            f"to an internet connection error: {e}"
        )
        logging.error("ERROR %s", error_message)
        raise Exception(error_message) from e


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols


def flat2curve(I, dist, mon_size, **kwargs):
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (phi, rho)

    def pol2cart(phi, rho):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    params = dict(
        {"center_x": 0, "center_y": 0, "method": "index"}, **kwargs
    )  # center_x, center_y points in normalized x coordinates from center

    # Shift the origin to the closest point of the image.
    nrows, ncols = np.shape(I)
    [yi, xi] = np.meshgrid(np.linspace(1, ncols, ncols), np.linspace(1, nrows, nrows))
    yt = yi - ncols / 2 + params["center_y"] * nrows - 0.5
    xt = xi - nrows / 2 - params["center_x"] * nrows - 0.5

    # Convert the Cartesian x- and y-coordinates to cylindrical angle (theta) and radius (r) coordinates
    [theta, r] = cart2pol(yt, xt)

    # Compute spherical radius
    diag = np.sqrt(sum(np.array(np.shape(I)) ** 2))  # diagonal in px
    dist_px = dist / 2.54 / mon_size * diag  # closest distance from the monitor in px
    phi = np.arctan(r / dist_px)

    h = np.cos(phi / 2) * dist_px
    r_new = 2 * np.sqrt(dist_px**2 - h**2)

    # Convert back to the Cartesian coordinate system. Shift the origin back to the upper-right corner of the image.
    [ut, vt] = pol2cart(theta, r_new)
    ui = ut + ncols / 2 - params["center_y"] * nrows
    vi = vt + nrows / 2 + params["center_x"] * nrows

    # Tranform image
    if params["method"] == "index":
        idx = (vi.astype(int), ui.astype(int))
        transform = lambda x: x[idx]
    elif params["method"] == "interp":
        transform = lambda x: ndimage.map_coordinates(
            x, [vi.ravel() - 0.5, ui.ravel() - 0.5], order=1, mode="nearest"
        ).reshape(x.shape)
    return (transform(I), transform)


def reverse_lookup(dictionary, target):
    return next(key for key, value in dictionary.items() if value == target)


def factorize(cond: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Factorizes conditions into individual combinations.

    This function takes a dictionary of conditions and generates all possible combinations
    of conditions, where each combination consists of one value for each key in the input
    dictionary.

    Args:
    - cond (Dict[str, Any]): A dictionary representing conditions.

    Returns:
    - List[Dict[str, Any]]: List of factorized conditions.

    Example:
    Suppose we have the following conditions:
    cond = {'param1': [1, 2], 'param2': [3, 4], 'param3': 'value', 'param4': (5, 6)}
    This function will generate the following combinations:
    [{'param1': 1, 'param2': 3, 'param3': 'value', 'param4': (5, 6)},
     {'param1': 1, 'param2': 4, 'param3': 'value', 'param4': (5, 6)},
     {'param1': 2, 'param2': 3, 'param3': 'value', 'param4': (5, 6)},
     {'param1': 2, 'param2': 4, 'param3': 'value', 'param4': (5, 6)}]

    """
    # Ensure all values are wrapped in lists
    values = [v if isinstance(v, list) else [v] for v in cond.values()]

    # Generate all combinations of conditions
    conds = []
    for combination in product(*values):
        # Create a dictionary representing each combination
        combined_cond = dict(zip(cond.keys(), combination))
        # Convert lists to tuples for immutability
        combined_cond = {
            k: tuple(v) if isinstance(v, list) else v for k, v in combined_cond.items()
        }
        conds.append(combined_cond)

    return conds


def make_hash(cond):
    def make_hashable(cond):
        if isinstance(cond, (tuple, list)):
            return tuple((make_hashable(e) for e in cond))
        if isinstance(cond, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in cond.items()))
        if isinstance(cond, (set, frozenset)):
            return tuple(sorted(make_hashable(e) for e in cond))
        return cond

    hasher = hashlib.md5()
    hasher.update(repr(make_hashable(cond)).encode())

    return base64.b64encode(hasher.digest()).decode()


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def iterable(v):
    return np.array([v]) if type(v) not in [np.array, np.ndarray, list, tuple] else v


class DictStruct:
    def __init__(self, dictionary):
        self.__dict__.update(**dictionary)

    def set(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def values(self):
        return self.__dict__.values()


def generate_conf_list(folder_path):
    contents = []
    files = os.listdir(folder_path)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, file_name in enumerate(files):
        contents.append([i, file_name, "", current_datetime])
    return contents


def convert_numeric_keys(data: Dict) -> Dict:
    """Recursively convert string keys to integers if they represent numbers.
    Converts only for specific fields: 'Odor', 'Liquid', 'Lick', 'Proximity', 'Sound'

    Args:
        data: Dictionary to convert
    Returns:
        Dictionary with converted keys
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Convert the value recursively if it's a dictionary
            if isinstance(value, dict):
                # For specific fields, convert their keys to integers
                if key in ["Odor", "Liquid", "Lick", "Proximity", "Sound"]:
                    new_dict[key] = {int(k): v for k, v in value.items()}
                else:
                    new_dict[key] = convert_numeric_keys(value)
            else:
                new_dict[key] = value
        return new_dict
    return data


def get_code_version_info(project_path: str = None) -> Dict[str, Any]:
    """Determine version information for a project directory.

    Checks git hash first, then PyPI version, then returns None.

    Args:
        project_path: Optional path to the project directory. If not provided,
            uses the current script's directory.

    Returns:
        dict: Contains source_type ('git', 'pypi', or None) and version information

    """
    from pathlib import Path

    if project_path is None:
        # project_path = os.path.dirname(os.path.abspath(__file__))
        project_path = Path(os.path.dirname(os.path.abspath(__file__))).parents[2]
        # two_folders_back = project_path.parents[1]
    result = {
        "project_path": os.path.abspath(project_path),
        "source_type": None,
        "version": None,
        "repository_url": '',
        "is_dirty": False,
    }

    # Check if it's a git repository
    git_dir = os.path.join(project_path, ".git")
    if os.path.isdir(git_dir):
        try:
            # Get the commit hash
            git_hash = (
                subprocess.check_output(
                    [
                        "git",
                        "--git-dir",
                        git_dir,
                        "--work-tree",
                        project_path,
                        "rev-parse",
                        "--short",
                        "HEAD",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )

            # Check if the repo has uncommitted changes
            git_status = (
                subprocess.check_output(
                    [
                        "git",
                        "--git-dir",
                        git_dir,
                        "--work-tree",
                        project_path,
                        "status",
                        "--porcelain",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
            is_dirty = len(git_status) > 0

            # Get the remote repository URL
            try:
                repo_url = (
                    subprocess.check_output(
                        [
                            "git",
                            "--git-dir",
                            git_dir,
                            "--work-tree",
                            project_path,
                            "config",
                            "--get",
                            "remote.origin.url",
                        ],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode("utf-8")
                    .strip()
                )
            except subprocess.CalledProcessError:
                repo_url = None

            result["source_type"] = "git"
            result["version"] = git_hash
            result["repository_url"] = repo_url
            result["is_dirty"] = is_dirty
            return result
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.debug(f"Not a git repository or git error: {e}")

    # Check if it's installed via PyPI
    try:
        # Try to get package metadata
        package_name = "ethopy"
        package_version = importlib.metadata.version(package_name)

        result["source_type"] = "pypi"
        result["version"] = package_version
        return result
    except (importlib.metadata.PackageNotFoundError, ImportError) as e:
        log.debug(f"Not a PyPI package or error: {e}")

    # If we get here, we couldn't determine version info
    return result


def get_environment_info() -> Dict[str, Any]:
    """Collect information about the system environment.

    Returns:
        dict: System environment information

    """
    cpu_info = "Unknown"
    memory_info = "Unknown"

    if platform.system() == "Linux":
        # Get CPU info on Linux
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_info = line.split(":", 1)[1].strip()
                            break
            except Exception as e:
                log.warning(f"Could not read CPU info: {e}")

        # Get memory info on Linux
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            memory_info = line.split(":", 1)[1].strip()
                            break
            except Exception as e:
                log.warning(f"Could not read memory info: {e}")

    elif platform.system() == "Darwin":  # macOS
        try:
            # Get CPU info on macOS
            cpu_info = (
                subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                .decode()
                .strip()
            )

            # Get memory info on macOS
            memory_info = (
                subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            )
            memory_info = f"{int(memory_info) // (1024**2)} kB"
        except Exception as e:
            log.warning(f"Could not read system info on macOS: {e}")

    elif platform.system() == "Windows":
        try:
            # Get CPU info on Windows
            cpu_info = (
                subprocess.check_output(["wmic", "cpu", "get", "Name"])
                .decode()
                .split("\n")[1]
                .strip()
            )

            # Get memory info on Windows
            memory_info = (
                subprocess.check_output(
                    ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"]
                )
                .decode()
                .split("\n")[1]
                .strip()
            )
            memory_info = f"{int(memory_info) // (1024**2)} kB"
        except Exception as e:
            log.warning(f"Could not read system info on Windows: {e}")

    return {
        "os_name": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "cpu_info": cpu_info,
        "memory_info": memory_info,
        "hostname": socket.gethostname(),
        "username": os.getlogin() if hasattr(os, "getlogin") else "unknown",
    }
