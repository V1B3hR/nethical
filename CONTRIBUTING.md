# Contributing to Nethical

Welcome to **Nethical**! We are building the governance, security, and ethics layer for the age of autonomous AI.

Because Nethical governs safety-critical systems, we hold our codebase to rigorous standards of integrity, cryptographic verifiability, and resilience.

---

## 1. Contributor License Agreement (CLA) & DCO

To ensure clean chain-of-title and intellectual property safety for all downstream deployers and community members, Nethical requires all contributions to comply with the **Developer Certificate of Origin (DCO 1.1)** and our [Contributor License Agreement (CLA)](CLA.md).

### How to Sign Your Commits
Every commit must include a `Signed-off-by` line indicating your agreement with the DCO. Git does this automatically when you use the `-s` flag:

```bash
git commit -s -m "feat(governance): implement ISO 42001 continuous audit hook"
```

If you submit pull requests on behalf of a corporation or enterprise entity, please review the Corporate CLA section in [CLA.md](CLA.md) or contact `compliance@nethical.ai`.

---

## 2. The "Sapper Mindset"

Nethical is not a standard web framework. A defect or bypass here could allow a rogue AI agent to circumvent safety constraints in industrial, financial, or cyber operations.

When contributing code, adopt the **Sapper Mindset**:
1. **Assume Active Adversary:** What if an LLM or autonomous agent deliberately Crafts an evasion prompt or race condition to bypass your validation?
2. **Verify Everything:** Zero trust. Validate types, bounds, cryptographically anchored signatures, and invariants.
3. **Adversarial Self-Testing:** Run tests from `tests/adversarial/` and `tests/misuse/` against your changes before submitting.
4. **Fail Closed:** If a governance check fails or encounters an unhandled state, the default outcome must be safe lockdown, not silent permission.

---

## 3. Development Workflow

### Prerequisites
- Python 3.10+ (tested through 3.12)
- Git 2.30+

### Setup
```bash
git clone https://github.com/V1B3hR/nethical.git
cd nethical
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -e ".[dev,test]"
```

### Coding Standards
- **Formatting & Linting:** Code is formatted with `black`, checked with `flake8` and `ruff`.
- **Type Annotations:** Full type hints checked with `mypy`.
- **License Headers:** Every new Python source file must include an SPDX header:
  ```python
  # SPDX-License-Identifier: MIT
  # Copyright (c) 2025-2026 Nethical Contributors
  ```

### Running Tests
Before opening a pull request, verify that tests pass:
```bash
pytest tests/ -q
```
For fast sanity checks on core modules:
```bash
pytest tests/core/ tests/security/ tests/api/ -q
```

---

## 4. Pull Request Checklist

When submitting a Pull Request:
- [ ] Commits are signed off (`git commit -s`) per [CLA.md](CLA.md).
- [ ] New features or fixes include automated test coverage in `tests/`.
- [ ] No secrets, private keys, or credentials are included.
- [ ] Documentation is updated for any API or governance policy changes.
- [ ] Code adheres to the 25 Fundamental Laws ([docs/laws_and_policies/FUNDAMENTAL_LAWS.md](docs/laws_and_policies/FUNDAMENTAL_LAWS.md)).

Thank you for contributing to responsible and resilient AI safety!
