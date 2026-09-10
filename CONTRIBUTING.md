# Contributing to ethopy

We welcome contributions to ethopy, a Python package for behavioral training! Whether you're interested in adding new analysis methods, improving documentation, or fixing bugs, your help is appreciated. Here's how you can contribute:

- Reporting bugs or usability issues
- Improving documentation and examples
- Adding new behavioral analysis features
- Enhancing existing modules
- Implementing new visualization methods
- Adding support for new data formats
- Optimizing performance

## Development Process

We use GitHub to host code, track issues and feature requests, and accept pull requests. Here's our development workflow:

1. Fork the repo and create your branch from `main`.
2. Set up your development environment:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```
3. Write your code and add tests:
   - Add unit tests for new features
   - Use the mocking pattern described in the Testing section below
4. Ensure code quality:
   - Run tests: `pytest -vv`
   - Check code style: `ruff check .`
   - Fix formatting issues: `ruff format .`
   - Type checking: `mypy src/ethopy` # TODO a lot of errors
5. Update documentation:
   - Add docstrings (Google format)
   - Update API documentation if needed
   - Include examples in docstrings if necessary
6. Submit your pull request

## Pull Request Process

1. Ensure your PR includes:
   - A clear description of the changes
   - Any updates to documentation
   - New or updated tests
   - Example usage if applicable
2. Link any related issues in the PR description
3. The PR will be reviewed by maintainers who may request changes
4. Once approved, your PR will be merged

## Code Style and Standards

We follow scientific Python coding standards to maintain consistency:

1. Code Style:
   - Follow PEP 8 guidelines
   - Use Google docstring format
   - Maximum line length: 88 characters
   - Use type hints for function signatures
   - Use snake_case for functions/variables, CamelCase for classes
   - Use double quotes for strings

## Dependency Management

EthoPy pins dependencies in two layers, and they do different jobs.

### `pyproject.toml`

Declares the core dependencies with loose ranges. Upper bounds are added only where a package has a track record of breaking releases, currently `datajoint`, `setuptools`, `numpy`, `pandas` and `scipy`. Do not add a cap to every dependency. Blanket caps cause resolution conflicts for anyone installing EthoPy alongside other packages, and each cap needs a release to lift.

When adding a new core dependency, add it without an upper bound unless you have evidence of a breakage, and refresh the lock file in the same pull request.

### `requirements-lock.txt`

One pinned version per package for the entire tree, including transitive dependencies. It is a single cross-platform file: environment markers cover the few packages that differ by operating system or Python version, so there is no per-OS variant to keep in sync.

Versions are chosen as the newest one actually running on a verified EthoPy machine that still supports the whole declared Python range, with the resolver enforcing mutual consistency. The point is to record what actually ran, not what a resolver believes should work.

Hardware and analysis packages are deliberately excluded. Every such import in EthoPy is lazy, so the core runs without them, and they are installed per machine.

To refresh it:

1. On each machine you care about, build a fresh virtual environment, install EthoPy without
   the lock file, run the test suite and a real experimental session, then capture
   `pip freeze --exclude-editable`. Do not skip the session; some breakages only appear at
   runtime with hardware attached.
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
   Repeat for each Python version and architecture you support. Any version drift means the
   pins are inconsistent.
4. Update the header comment with the source machines, Python versions and date.

### If nobody refreshes this

The lock file will age. Installs will keep working, reproducing an increasingly old
environment, and EthoPy will fall behind the ecosystem. This is the intended failure mode and
it is preferred to the alternative. An install that reproduces a two-year-old working
environment is still a working install; an install that silently picks up an untested major
release is not.

## Testing Guidelines

Ethopy has specific testing requirements due to its database connections:

1. **Database Mocking**: 
   - Tests must run without an actual database connection
   - Use the established mocking pattern for database and threading

2. **Testing Pattern**:
   - Use the `patch_imports` fixture pattern from existing tests
   - Import modules inside test functions, not at module level
   - Example:
   ```python
   import pytest
   import sys
   from unittest.mock import patch, MagicMock
   
   @pytest.fixture(scope="module")
   def patch_imports():
       """Patch imports to prevent database connections."""
       mocks = {'datajoint': MagicMock(), 'datajoint.config': MagicMock()}
       with patch.dict(sys.modules, mocks), patch('pathlib.Path.home'), patch('threading.Thread'):
           yield
           
   @pytest.mark.usefixtures("patch_imports")
   class TestYourModule:
       # Tests go here...
   ```

3. **Running Tests**:
  `pytest -vv`
   - For a single test: `pytest tests/test_behavior.py::TestBehavior::test_update_history -v`

4. **Troubleshooting**:
   - If tests hang, it usually means threading or database connection issues
   - Ensure all threads are mocked and database connections are properly patched
   - See existing test files for reference implementations

5. **CI Environment**:
   - GitHub Actions will run all tests with database mocking
   - Always ensure your tests pass in a CI environment with no database


## Reporting Issues

Report bugs and feature requests using GitHub's [Issue Tracker](https://github.com/ef-lab/ethopy_package/issues). When reporting bugs:

1. Use a clear and descriptive title
2. Describe the exact steps to reproduce the bug
3. Include example code and data if possible
4. Describe the expected behavior
5. Include system information:
   - ethopy version
   - Python version
   - Operating system
   - Relevant package versions (numpy, pandas, etc.)

## License

By contributing to ethopy, you agree that your contributions will be licensed under the same license as the project. Please contact the maintainers if you have any questions about licensing.