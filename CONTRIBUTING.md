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
   - Include integration tests if needed
   - Add example notebooks for new features
4. Ensure code quality:
   - Run tests: `pytest`
   - Check code style: `ruff check .`
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


## Reporting Issues

Report bugs and feature requests using GitHub's [Issue Tracker](https://github.com/ethopy/issues). When reporting bugs:

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