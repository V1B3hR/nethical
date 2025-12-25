# Pull Request

## Description

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

<!-- Mark with an 'x' the applicable options -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Security fix
- [ ] Performance improvement
- [ ] Refactoring
- [ ] CI/CD changes
- [ ] Dependency updates

## Testing

<!-- Describe the tests you ran to verify your changes -->

- [ ] Unit tests pass locally
- [ ] Integration tests pass locally
- [ ] Manual testing performed
- [ ] New tests added for new features

## Security Review Checklist

<!-- REQUIRED for all PRs. Mark with 'x' or 'N/A' if not applicable -->

### Secrets and Credentials
- [ ] No secrets, API keys, passwords, or credentials committed
- [ ] Verified with: `git log -p | grep -iE "password|secret|key|token|api_key"`
- [ ] All sensitive data uses environment variables or secret management

### Dependencies
- [ ] Dependencies updated to latest secure versions
- [ ] No new high/critical vulnerabilities introduced
- [ ] Ran: `pip-audit` or `safety check`
- [ ] Dependencies come from official sources only (PyPI)
- [ ] Hash verification present for production dependencies

### Input Validation
- [ ] All user inputs are validated
- [ ] No SQL injection vulnerabilities
- [ ] No command injection vulnerabilities
- [ ] No path traversal vulnerabilities
- [ ] XSS protection implemented where applicable

### Authentication & Authorization
- [ ] Authentication checks are present where required
- [ ] Authorization follows least privilege principle
- [ ] No hardcoded credentials or tokens
- [ ] Session management is secure

### Code Security
- [ ] Security linters pass (bandit, semgrep)
- [ ] No eval() or exec() usage without validation
- [ ] File operations use safe paths
- [ ] Proper error handling (no sensitive data in errors)

### Infrastructure & Configuration
- [ ] Docker configurations follow security best practices
- [ ] AppArmor/SELinux profiles updated if needed
- [ ] Network policies remain secure
- [ ] No unnecessary ports exposed

### Monitoring & Logging
- [ ] Security-relevant events are logged
- [ ] Logs don't contain sensitive data
- [ ] Monitoring/alerting configured for security events

## Documentation

<!-- Check all that apply -->

- [ ] README.md updated (if applicable)
- [ ] Security documentation updated (if applicable)
- [ ] API documentation updated (if applicable)
- [ ] Code comments added for complex logic
- [ ] CHANGELOG.md updated

## CI/CD Checks

<!-- These should pass automatically -->

- [ ] All CI/CD workflows pass
- [ ] Linters pass (black, flake8, mypy)
- [ ] Security scans pass (bandit, pip-audit, trivy)
- [ ] Test coverage maintained or improved
- [ ] No critical code scanning alerts

## Deployment Considerations

<!-- Check if applicable -->

- [ ] Database migrations included (if needed)
- [ ] Backward compatibility maintained
- [ ] Environment variables documented
- [ ] Rollback plan documented
- [ ] Performance impact assessed

## Security Impact Assessment

<!-- Rate the security impact of this PR -->

- [ ] **No security impact** - Documentation, tests, or internal refactoring only
- [ ] **Low impact** - Minor changes to non-critical components
- [ ] **Medium impact** - Changes to security-related code or configurations
- [ ] **High impact** - Changes to authentication, authorization, or cryptography
- [ ] **Critical impact** - Security vulnerability fix or major security feature

### If Medium/High/Critical Impact:

**Security Review Required By**: 
<!-- Tag security team members: @security-team -->

**Security Testing Performed**:
<!-- Describe security testing, penetration testing, or security audit performed -->

**Threat Model Updated**: 
- [ ] Yes
- [ ] No
- [ ] N/A

## Checklist

<!-- General PR checklist -->

- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Additional Notes

<!-- Any additional information that reviewers should know -->

---

## For Reviewers

### Security Review Focus Areas:

1. **Secrets Management**: Check for exposed credentials
2. **Input Validation**: Verify all inputs are sanitized
3. **Dependencies**: Review for vulnerable dependencies
4. **Authentication**: Verify auth mechanisms are secure
5. **Authorization**: Check least privilege is enforced
6. **Error Handling**: Ensure no sensitive data leaks
7. **Logging**: Verify appropriate logging without sensitive data

### Review Commands:

```bash
# Clone and checkout PR
gh pr checkout <PR_NUMBER>

# Run security scans
pip-audit -r requirements.txt
safety check
bandit -r nethical/ -ll

# Run tests
pytest tests/ -v

# Check for secrets
git log -p | grep -iE "password|secret|key|token|api_key"
```

### Approval Criteria:

- [ ] Code quality is acceptable
- [ ] Tests are sufficient and passing
- [ ] Documentation is adequate
- [ ] Security checklist is complete
- [ ] No unresolved security concerns
- [ ] Performance impact is acceptable

---

**By submitting this PR, I confirm that**:
- I have read and followed the [Contributing Guidelines](./CONTRIBUTING.md)
- I have reviewed the [Security Policy](./SECURITY.md)
- I understand this project's [Security Hardening Guide](./docs/SECURITY_HARDENING.md)
