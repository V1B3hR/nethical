# Ruff Linter Guide

This document describes Ruff integration in Nethical for ultra-fast Python linting and formatting.

## Overview

Ruff is an extremely fast Python linter and formatter, written in Rust. It replaces multiple tools:
- **Flake8** - Code linting
- **isort** - Import sorting
- **pyupgrade** - Python upgrade syntax
- **pylint** - Additional checks

**Benefits:**
- 10-100x faster than traditional linters
- Single tool for linting + formatting
- Drop-in replacement for existing tools
- Zero-config defaults

## Installation

Ruff is included in project dependencies:

```bash
# Via pip
pip install ruff

# Via pyproject.toml
pip install -e .
```

## Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 120

exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
]

select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "PERF", # performance anti-patterns
    "ASYNC",# async best practices
]

ignore = [
    "E501",   # Line too long (handled by formatter)
    "B008",   # Function call in defaults
]

fixable = ["ALL"]
show-fixes = true

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # Unused imports OK in __init__
"tests/*" = ["ARG001", "ARG002"]  # Unused args OK in tests

[tool.ruff.isort]
known-first-party = ["nethical"]

[tool.ruff.mccabe]
max-complexity = 15
```

## Usage

### Command Line

#### Check Code
```bash
# Check all files
ruff check .

# Check specific files
ruff check nethical/detectors/realtime/

# Check with GitHub Actions format
ruff check . --output-format=github
```

#### Auto-fix Issues
```bash
# Fix all auto-fixable issues
ruff check . --fix

# Preview fixes without applying
ruff check . --fix --diff
```

#### Format Code
```bash
# Format all files
ruff format .

# Check formatting without applying
ruff format --check .

# Format specific files
ruff format nethical/detectors/
```

### Pre-commit Hook

Automatically run Ruff before commits.

#### Setup
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

#### Configuration (`.pre-commit-config.yaml`)
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

#### Manual Run
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files
pre-commit run
```

### CI/CD Integration

Ruff is integrated in GitHub Actions (`.github/workflows/ci.yml`):

```yaml
- name: Run Ruff linter
  run: |
    pip install ruff
    ruff check . --output-format=github

- name: Run Ruff format check
  run: |
    ruff format --check .
```

### IDE Integration

#### VS Code
1. Install extension: `charliermarsh.ruff`
2. Add to `.vscode/settings.json`:
```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": true,
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm
1. Install Ruff plugin
2. Configure: Settings → Tools → Ruff
3. Enable "Format on save"

#### Vim/Neovim
```lua
-- With null-ls
local null_ls = require("null-ls")
null_ls.setup({
  sources = {
    null_ls.builtins.diagnostics.ruff,
    null_ls.builtins.formatting.ruff,
  },
})
```

## Rule Categories

### Enabled Rules

#### E/W - pycodestyle
Style errors and warnings (PEP 8).
```python
# E101 - Indentation contains mixed spaces and tabs
if True:
    pass  # Good: 4 spaces

# E711 - Comparison to None
if x is None:  # Good
if x == None:  # Bad
```

#### F - Pyflakes
Logic errors and unused code.
```python
# F401 - Unused import
import os  # Bad if not used

# F841 - Unused variable
x = 1  # Bad if x never used
```

#### I - isort
Import sorting and organization.
```python
# Good: Standard lib → Third party → First party
import os
import sys

import numpy as np
from cachetools import LRUCache

from nethical.detectors import BaseDetector
```

#### N - pep8-naming
Naming conventions.
```python
# N801 - Class name should use CapWords
class my_class:  # Bad
class MyClass:   # Good

# N806 - Variable in function should be lowercase
def func():
    MyVar = 1  # Bad
    my_var = 1  # Good
```

#### UP - pyupgrade
Modern Python syntax.
```python
# UP008 - Use super() instead of super(__class__, self)
super().__init__()  # Good
super(MyClass, self).__init__()  # Bad

# UP032 - Use f-strings
f"Hello {name}"  # Good
"Hello {}".format(name)  # Bad
```

#### B - flake8-bugbear
Common bugs and design problems.
```python
# B006 - Mutable default argument
def func(x=[]):  # Bad
def func(x=None):  # Good
    x = x or []

# B007 - Unused loop control variable
for i in range(10):  # Bad if i unused
for _ in range(10):  # Good
```

#### C4 - flake8-comprehensions
List/dict/set comprehension improvements.
```python
# C400 - Unnecessary list comprehension
list(x for x in y)  # Bad
[x for x in y]  # Good

# C416 - Unnecessary dict comprehension
dict((x, y) for x, y in z)  # Bad
{x: y for x, y in z}  # Good
```

#### SIM - flake8-simplify
Code simplification.
```python
# SIM102 - Use single if statement
if a:  # Bad
    if b:
        pass

if a and b:  # Good
    pass

# SIM105 - Use contextlib.suppress
try:  # Bad
    os.remove(file)
except FileNotFoundError:
    pass

from contextlib import suppress
with suppress(FileNotFoundError):  # Good
    os.remove(file)
```

#### PERF - Performance anti-patterns
Performance issues.
```python
# PERF401 - List comprehension instead of for loop with append
result = []
for x in items:  # Bad
    result.append(x * 2)

result = [x * 2 for x in items]  # Good

# PERF402 - Use list comprehension instead of for loop with extend
result = []
for x in items:  # Bad
    result.extend(x)

result = [y for x in items for y in x]  # Good
```

#### ASYNC - Async best practices
Async/await patterns.
```python
# ASYNC100 - Async function with no await
async def func():  # Bad
    return 1

def func():  # Good
    return 1

# ASYNC102 - Sync call in async function
async def func():
    time.sleep(1)  # Bad
    await asyncio.sleep(1)  # Good
```

### Ignored Rules

#### E501 - Line too long
Handled by formatter, not enforced during linting.

#### B008 - Function call in defaults
Allowed for dataclasses and Pydantic models.

```python
@dataclass
class Config:
    # OK: Allowed for dataclasses
    authorized_apis: Set[str] = field(default_factory=set)
```

## Common Issues

### Unused Imports in `__init__.py`

**Problem:**
```python
# __init__.py
from .detector import MyDetector  # F401: imported but unused
```

**Solution:**
Already configured to ignore F401 in `__init__.py`.

### Line Length

**Problem:**
```python
# Line exceeds 120 characters
def very_long_function_name_with_many_parameters(param1, param2, param3, param4, param5, param6):
    pass
```

**Solution:**
Use formatter to auto-fix:
```bash
ruff format .
```

Or break manually:
```python
def very_long_function_name_with_many_parameters(
    param1,
    param2,
    param3,
    param4,
    param5,
    param6,
):
    pass
```

### Import Sorting

**Problem:**
```python
from nethical.detectors import BaseDetector
import os
import numpy as np
```

**Solution:**
```bash
ruff check --select I --fix .
```

Result:
```python
import os

import numpy as np

from nethical.detectors import BaseDetector
```

### Type Annotations

Ruff doesn't enforce type annotations, but recommends them:

```python
# Good
async def detect(self, context: Dict[str, Any]) -> List[Violation]:
    pass

# OK, but not ideal
async def detect(self, context):
    pass
```

## Performance

Ruff is extremely fast:

```bash
# Benchmark on Nethical codebase
time ruff check .
# real    0m0.021s

time flake8 .
# real    0m1.243s

# Ruff is ~60x faster
```

## Migration from Flake8

### Replace Flake8

**Before:**
```toml
[tool.flake8]
max-line-length = 88
extend-ignore = E203, W503
```

**After:**
```toml
[tool.ruff]
line-length = 88
ignore = ["E203", "W503"]
```

### Replace isort

**Before:**
```toml
[tool.isort]
profile = "black"
known_first_party = ["nethical"]
```

**After:**
```toml
[tool.ruff]
select = ["I"]

[tool.ruff.isort]
known-first-party = ["nethical"]
```

### Update CI

**Before:**
```yaml
- run: flake8 .
- run: isort --check .
```

**After:**
```yaml
- run: ruff check .
- run: ruff format --check .
```

## Troubleshooting

### Ruff Not Found

```bash
# Ensure Ruff is installed
pip install ruff

# Check version
ruff --version
```

### Config Not Loaded

```bash
# Verify config file
cat pyproject.toml | grep -A 20 "tool.ruff"

# Force config file
ruff check --config pyproject.toml .
```

### Pre-commit Not Running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Test manually
pre-commit run --all-files
```

## Best Practices

1. **Run Ruff frequently:**
   ```bash
   ruff check . --fix && ruff format .
   ```

2. **Integrate in IDE:**
   Configure format-on-save in your editor.

3. **Use pre-commit:**
   Catch issues before committing.

4. **Review fixes:**
   Always review auto-fixes before committing.

5. **Keep config minimal:**
   Use defaults when possible.

6. **Document exceptions:**
   Add comments for `# noqa` overrides.

## Custom Rules

To add custom rules:

```toml
[tool.ruff]
# Add specific rules
select = [
    "E",
    "F",
    "YOUR_RULE_CODE",
]

# Or ignore specific rules
ignore = [
    "YOUR_RULE_CODE",
]
```

See [Ruff rules documentation](https://docs.astral.sh/ruff/rules/) for available rules.

## Support

- **Documentation:** https://docs.astral.sh/ruff/
- **Repository:** https://github.com/astral-sh/ruff
- **Discord:** https://discord.gg/astral-sh
- **Issues:** https://github.com/astral-sh/ruff/issues

## References

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ruff Rules](https://docs.astral.sh/ruff/rules/)
- [Ruff Settings](https://docs.astral.sh/ruff/settings/)
- [Ruff vs Flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8)
