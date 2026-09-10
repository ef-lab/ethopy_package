# Installation

## Requirements

- Python >=3.8,<3.12
- Docker (for database setup)
- Dependencies: numpy, pandas, datajoint, pygame, pillow, and more (automatically installed)

## Installation Options

### Basic Installation

To install EthoPy with basic features, run:

```bash
pip install ethopy
```

This resolves the most recent versions allowed by EthoPy's declared dependency ranges. Those versions may be newer than anything that has been tested on a rig.

### Reproducible Installation (recommended for rigs)

`requirements-lock.txt` in the repository records the exact versions of every package, including transitive dependencies, from a Raspberry Pi rig that is known to run correctly. Install that environment first, then install EthoPy without letting it re-resolve:

```bash
pip install -r https://raw.githubusercontent.com/ef-lab/ethopy_package/main/requirements-lock.txt
pip install --no-deps ethopy
```

This works on every supported platform. The lock file uses environment markers rather than being tied to one operating system, and is verified to install from wheels on Python 3.9, 3.10 and 3.11, on both aarch64 (Raspberry Pi) and x86_64 (Linux, macOS, Windows). No pin requires a compiler.

The locked versions are refreshed by hand, so they will lag behind the newest releases. That is intentional. See [Dependency Management](contributing.md#dependency-management) for the refresh procedure.

#### Per-machine packages

The lock file covers only EthoPy's core dependencies. Hardware and analysis packages are excluded, because every one of them is a lazy import and the core runs without them. Install them on top of the lock file, only where they are needed:

```bash
# Raspberry Pi rigs
pip install RPi.GPIO pigpio pyserial picamera2 opencv-python

# DeepLabCut setups, such as the openfield workstation
pip install deeplabcut-live scikit-video
```

### Optional Features

For additional functionality:

```bash
# For development
pip install "ethopy[dev]"

# For documentation
pip install "ethopy[docs]"
```

### From Source

To install the latest development version:

```bash
pip install git+https://github.com/ef-lab/ethopy_package
```

For development installation:

```bash
git clone https://github.com/ef-lab/ethopy_package.git
cd ethopy_package
pip install -e ".[dev,docs]"
```

## Raspberry Pi Setup

For detailed Raspberry Pi setup instructions, including hardware-specific configurations and dependencies, please refer to our [Raspberry Pi Setup Guide](raspberry_pi.md).
