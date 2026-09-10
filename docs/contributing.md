# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/ef-lab/ethopy_package/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with `bug` and
`help wanted` is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
`enhancement` and `help wanted` is open to whoever wants to implement it.

### Write Documentation

EthoPy could always use more documentation,
whether as part of the official EthoPy docs,
in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/ef-lab/ethopy_package/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Development Setup

Ready to contribute? Here's how to set up Ethopy for local development:

1. Fork the Ethopy repo on GitHub.

2. Clone your fork locally:
    ```bash
    git clone git@github.com:your_name_here/ethopy.git
    cd ethopy
    ```

3. Install development dependencies:
    ```bash
    pip install -e ".[dev,docs]"
    ```

4. Create a branch for local development:
    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

5. Make your changes locally. The project uses several tools to maintain code quality:
        - **ruff**: Code formatting
        - **isort**: Import sorting
        - **mypy**: Static type checking
        - **ruff**: Linting
        - **pytest**: Testing

6. Run the test suite and code quality checks:
    ```bash
    # Run tests with coverage
    pytest

    # Run linting
    ruff check src/ethopy
    ```

7. Build and check documentation locally:
    ```bash
    mkdocs serve
    ```
   Visit http://127.0.0.1:8000 to view the documentation.

8. Commit your changes and push your branch to GitHub:
    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

9. Submit a pull request through the GitHub website.

## Dependency Management

EthoPy constrains dependencies in two layers, and they do different jobs.

### `pyproject.toml`

Declares the core dependencies with loose ranges. Upper bounds are added only where a package has a track record of breaking releases, currently `datajoint`, `setuptools`, `numpy`, `pandas` and `scipy`. Do not cap every dependency. Blanket caps cause resolution conflicts for anyone installing EthoPy alongside other packages, and each one needs a release to lift.

When adding a new core dependency, add it without an upper bound unless you have evidence of a breakage, and refresh the lock file in the same pull request.

### `requirements-lock.txt`

One pinned version per package for the entire tree, including transitive dependencies, and what [Installation](installation.md) points every machine at. It is a single cross-platform file: environment markers cover the few packages that differ by operating system or Python version, so there is no per-OS variant to keep in sync.

Versions are chosen as the newest one actually running on a verified EthoPy machine that still supports the whole declared Python range, with the resolver enforcing mutual consistency. Hardware and analysis packages are excluded, since every such import in EthoPy is lazy and they are installed per machine.

To refresh it:

1. On each machine you care about, build a fresh virtual environment, install EthoPy without the lock file, run the test suite and a real experimental session, then capture `pip freeze --exclude-editable`. Do not skip the session, some breakages only appear at runtime with hardware attached.
2. Regenerate the cross-platform pin set, capped at the versions you just verified:
    ```bash
    uv pip compile pyproject.toml --universal --python-version <lowest you support> \
        --constraint <verified versions> -o requirements-lock.txt
    ```
3. Confirm it still resolves on every target before committing:
    ```bash
    uv pip compile requirements-lock.txt --python-version 3.9.2 \
        --python-platform aarch64-unknown-linux-gnu
    ```
   Repeat for each Python version and architecture you support. Any version drift means the pins are inconsistent.
4. Update the header comment with the source machines, Python versions and date.

### If nobody refreshes this

The lock file will age. Installs keep working, reproducing an increasingly old environment, and EthoPy falls behind the ecosystem. This is the intended failure mode and it is preferred to the alternative. An install that reproduces a two-year-old working environment is still a working install. An install that silently picks up an untested major release is not.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.md
3.  The pull request should work for Python 3.9 through 3.11, the range declared by
    `requires-python`. Check <https://github.com/ef-lab/ethopy_package/pulls> and make sure that the tests pass for all supported Python versions.
