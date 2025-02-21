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

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.md
3.  The pull request should work for Python 3.8 and later, and
    for PyPy. Check <https://github.com/ef-lab/ethopy_package/pulls> and make sure that the tests pass for all
    supported Python versions.
