# Getting Started with EthoPy

This guide will walk you through the process of setting up and running your first experiment with EthoPy. After completing this guide, you'll have a solid understanding of how to configure and run basic behavioral experiments.

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher (but less than 3.12) installed
- MariaDB database
!!! note  Database setup
    We recommend using [Docker](https://www.docker.com/blog/getting-started-with-docker-desktop/) for setting up a new database.

## Step 1: Installation

### Setting Up a Virtual Environment

Before installing dependencies, it's recommended to use a virtual environment to keep your project isolated and manageable. Choose one of the following methods:

=== "Conda"
    ```sh
    conda create --name myenv python=3.x
    conda activate myenv
    ```
    [More details](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

=== "venv (pip)"
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
    [More details](https://docs.python.org/3/library/venv.html)

=== "uv"
    ```sh
    uv venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```
    [More details](https://github.com/astral-sh/uv)

Once activated, proceed with installation.


#### Install EthoPy Package

Choose the installation method that fits your needs:

=== "Basic Installation"
    Install with pip:
    ```sh
    pip install ethopy
    ```

=== "Latest Development Version"
    Install from source:
    ```sh
    pip install git+https://github.com/ef-lab/ethopy_package
    ```

=== "Development Installation"
    For contributing or modifying the package:
    ```sh
    git clone https://github.com/ef-lab/ethopy_package.git
    cd ethopy_package
    pip install -e ".[dev,docs]"
    ```

**Verify installation:**

   ```bash
   ethopy --version
   ```

## Step 2: Database Setup

EthoPy relies on a MariaDB database for experiment configuration and data logging. If there is not a database availabe, here is a quick setup of setting mysql database with docker:

1. **Start the database container:**

   ```bash
   ethopy-setup-djdocker
   ```
   alterative follow the instruction from datajoint [here](https://github.com/datajoint/mysql-docker)

   The default username is "root".

> **Note:** if ethopy-setup-djdocker not works try to setup the docker image based on the [datajoint instructions](https://github.com/datajoint/mysql-docker)


> **Note:** By default, Docker requires sudo because the Docker daemon runs as root.
This command adds your user to the docker group, so you can run Docker commands without sudo.
>
>```bash
>sudo usermod -aG docker $USER
>```
>
>restart your session (log out and back in) or run:
>```bash
>newgrp docker
>```
>

## Step 3: Configure ethopy

   Create a configuration file at path:
   
=== "Linux/macOS"
    `~/.ethopy/local_conf.json`

=== "Windows"
    `%USERPROFILE%\.ethopy\local_conf.json`
   dj_local_conf have all the parameters refers to [datajoint configuration](https://datajoint.com/docs/elements/element-miniscope/0.2/tutorials/01-Configure/):
   ```json
   {
       "dj_local_conf": {
           "database.host": "127.0.0.1",
           "database.user": "root",
           "database.password": ...,
           "database.port": 3306
       },
     "SCHEMATA": {
       "experiment": "lab_experiments",
       "stimulus": "lab_stimuli",
       "behavior": "lab_behavior",
       "interface": "lab_interface",
       "recording": "lab_recordings"
     }
   }
   ```

Check if connection with db is established with your local configuration.
   ```bash
   ethopy-db-connection
   ```

## Step 4: Create required schemas
   Create all required database schemas.

   ```bash
   ethopy-setup-schema
   ```
!!! tip Check database
    For verifing the schema/tables creation you can download [DBeaver](https://dbeaver.io/) which is a popular and versatile database tool.


### Step 5: Run your first experiment

```
ethopy --task-path grating_test.py --log-console
```
The option --task-path is for defining the path of the task. The [example tasks](https://github.com/ef-lab/ethopy_package/tree/main/src/ethopy/task) can run by the file name, for any other experiment you must define the full path.
Option --log-console is to enable the logging in terminal.


You can check all the options of ethopy by:
```bash
ethopy --help
```

## Example Tasks

Explore these sample tasks in the `ethopy/task/` directory:

1. **grating_test.py** - Grating stimulus presentation
2. **bar_test.py** - Moving bar stimulus
3. **dot_test.py** - Moving dot patterns

## Troubleshooting and Help

If you encounter issues, refer to the [Troubleshooting Guide](troubleshooting.md).

For specific questions, check the:
- [API Reference](API/logger.md) for detailed module documentation
- [GitHub Issues](https://github.com/ef-lab/ethopy_package/issues) for known problems

---

## Where to Go Next

Now that you have a basic understanding of EthoPy:

1. [Creating Custom Components](creating_custom_components.md)
2. How to [create a Task ](https://ef-lab.github.io/ethopy_package/task_setup/)
3. Explore the [Plugin System](plugin.md) to extend functionality
4. Dive deeper into [Local Configuration](local_conf.md) for advanced settings
5. Understand [setup configuration index](setup_configuration_idx.md)
6. Learn more about [Database Setup](database.md)
7. Study the [API Reference](API/logger.md) for detailed documentation
8. Check [Contributing](contributing.md) if you want to help improve EthoPy