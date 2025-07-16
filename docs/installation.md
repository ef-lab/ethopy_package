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

This is the preferred method as it will install the most recent stable release.

### Optional Features

For additional functionality:

```bash
# For 3D object support
pip install "ethopy[obj]"

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
git clone https://github.com/ef-lab/ethopy_package/.git
cd ethopy
pip install -e ".[dev,docs]"
```

## Raspberry Pi Setup

For detailed Raspberry Pi setup instructions, including hardware-specific configurations and dependencies, please refer to our [Raspberry Pi Setup Guide](raspberry_pi.md).
