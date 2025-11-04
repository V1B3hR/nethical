# Contributing to Nethical

Thank you for your interest in contributing to Nethical! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/nethical.git
   cd nethical
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/V1B3hR/nethical.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher (3.9-3.12 recommended)
- Git
- pip and virtualenv

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   # Install core dependencies
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install in editable mode
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   python -c "from nethical.core import IntegratedGovernance; print('Success!')"
   pytest tests/ -v
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Examples**: Create new example scripts
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Security**: Identify and fix security issues

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are open for community contributions
- Check the [roadmap.md](roadmap.md) for planned features

### Before You Start

1. **Check existing issues** to see if your idea is already being worked on
2. **Open an issue** to discuss significant changes before starting work
3. **Comment on the issue** to let others know you're working on it

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black formatter default)
- **Imports**: Organized using isort
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Code Quality Tools

Format and lint your code before submitting:

```bash
# Format code with Black
black nethical/ tests/ examples/

# Check code style with flake8
flake8 nethical/ tests/ examples/

# Type checking with mypy
mypy nethical/
```

### Code Organization

- **nethical/core/**: Core governance components
- **nethical/detectors/**: Safety and violation detectors
- **nethical/judges/**: Judgment and decision systems
- **nethical/monitors/**: Monitoring components
- **nethical/api/**: API interfaces
- **nethical/security/**: Authentication and security
- **nethical/mlops/**: ML operations and training
- **nethical/marketplace/**: Plugin marketplace
- **nethical/storage/**: Storage backends

### Naming Conventions

- **Classes**: PascalCase (e.g., `IntegratedGovernance`)
- **Functions/Methods**: snake_case (e.g., `process_action`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_VIOLATIONS`)
- **Private members**: Prefix with underscore (e.g., `_internal_method`)

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_governance.py

# Run with coverage
pytest tests/ --cov=nethical --cov-report=html

# Run tests for specific feature
pytest tests/ -k "anomaly"
```

### Writing Tests

1. **Test location**: Place tests in `tests/` matching the source structure
2. **Test naming**: Prefix test functions with `test_`
3. **Test organization**: Group related tests in classes
4. **Fixtures**: Use pytest fixtures for common setup
5. **Coverage**: Aim for >80% code coverage for new code

Example test structure:

```python
import pytest
from nethical.core import IntegratedGovernance

class TestIntegratedGovernance:
    @pytest.fixture
    def governance(self):
        return IntegratedGovernance(storage_dir="./test_data")
    
    def test_process_action_basic(self, governance):
        result = governance.process_action(
            agent_id="test_agent",
            action="test action",
            cohort="test"
        )
        assert result is not None
        assert "phase3" in result
```

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Test performance characteristics
- **Adversarial tests**: Test security and robustness

## Documentation

### Types of Documentation

1. **Code docstrings**: Document all public functions and classes
2. **README.md**: Project overview and quick start
3. **Implementation guides**: Detailed feature documentation in `docs/implementation/`
4. **API documentation**: Generated from docstrings
5. **Examples**: Working code examples in `examples/`

### Docstring Format

Use Google-style docstrings:

```python
def process_action(
    agent_id: str,
    action: str,
    cohort: str = "default"
) -> dict:
    """Process an agent action through the governance pipeline.
    
    Args:
        agent_id: Unique identifier for the agent
        action: Description of the action being performed
        cohort: Agent cohort for fairness sampling (default: "default")
    
    Returns:
        Dictionary containing governance results with keys:
        - phase3: Risk assessment results
        - phase4: Integrity and audit results
        - phase567: ML and anomaly detection results
        - phase89: Human feedback results
    
    Raises:
        ValueError: If agent_id or action is empty
        RuntimeError: If governance system is not initialized
    
    Example:
        >>> gov = IntegratedGovernance()
        >>> result = gov.process_action("agent_1", "User query", "production")
        >>> print(result["phase3"]["risk_score"])
        0.35
    """
    # Implementation
```

### Updating Documentation

- Update relevant docs when changing functionality
- Add examples for new features
- Update README.md for significant changes
- Keep CHANGELOG.md up to date

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Run linters**:
   ```bash
   black nethical/ tests/ examples/
   flake8 nethical/ tests/ examples/
   mypy nethical/
   ```

4. **Update documentation** as needed

### Creating a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin your-branch-name
   ```

2. **Create PR on GitHub** with:
   - Clear title describing the change
   - Description explaining what and why
   - Reference to related issues
   - Screenshots for UI changes
   - Test results

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security fix

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for changes
- [ ] Updated existing tests

## Documentation
- [ ] Updated docstrings
- [ ] Updated README.md
- [ ] Updated implementation guides
- [ ] Added examples

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] No new warnings generated
- [ ] CHANGELOG.md updated

## Related Issues
Fixes #(issue number)
```

### Review Process

1. **Automated checks**: CI/CD must pass
2. **Code review**: At least one maintainer approval required
3. **Testing**: All tests must pass
4. **Documentation**: Docs must be updated
5. **Security**: Security scans must pass

### After Approval

- Maintainers will merge your PR
- Your contribution will be acknowledged in CHANGELOG.md
- Consider helping others by reviewing PRs

## Reporting Issues

### Bug Reports

Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS, etc.)
- Error messages and stack traces
- Minimal code example

### Feature Requests

Include:
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach
- Willingness to contribute

### Security Issues

**Do not** open public issues for security vulnerabilities. Instead:
- Email security concerns to the maintainers
- See [SECURITY.md](SECURITY.md) for details

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions and discussions
- **Roadmap**: See [roadmap.md](roadmap.md) for planned features

### Getting Help

- Check [README.md](README.md) for getting started
- Review [documentation](docs/) for detailed guides
- Look at [examples](examples/) for working code
- Search [existing issues](https://github.com/V1B3hR/nethical/issues) for similar questions

### Recognition

Contributors are recognized through:
- CHANGELOG.md acknowledgments
- GitHub contributor list
- Project documentation credits

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

## Questions?

If you have questions about contributing:
1. Check this guide thoroughly
2. Review existing issues and PRs
3. Open a new issue with your question

Thank you for contributing to Nethical! 🔒
