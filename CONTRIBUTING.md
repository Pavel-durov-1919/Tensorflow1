# Contributing to AeroShield

First off, thank you for considering contributing to AeroShield! It's people like you that make AeroShield such a great tool.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally.
3. **Set up the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
   pip install -r requirements.txt
   ```
4. **Copy the `.env.example`** to `.env` and fill in any required local configuration.

## Development Workflow

- **Branching**: Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature`).
- **Coding Standards**:
  - Follow PEP 8 guidelines for Python code.
  - Maintain type hints where applicable.
  - Write clear and descriptive docstrings.
- **Testing**:
  - Write unit tests for new features using `pytest`.
  - Ensure all existing tests pass by running:
    ```bash
    PYTHONPATH="backend" pytest tests/
    ```

## Submitting Changes

1. **Commit your changes**: Provide clear, concise commit messages.
2. **Push to your fork** and submit a Pull Request.
3. **Review**: Maintainers will review your PR and provide feedback. We aim to review all PRs within 48 hours.

## Bug Reports and Feature Requests

Please use the issue tracker to report bugs or request features. When reporting a bug, include:
- A clear, descriptive title.
- Steps to reproduce the issue.
- Expected behavior vs. actual behavior.
- Relevant logs or error messages.

Thank you for contributing!
