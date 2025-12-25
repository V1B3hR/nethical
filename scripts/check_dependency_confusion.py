#!/usr/bin/env python3
"""
Dependency Confusion Detection Script

This script checks for potential dependency confusion attacks by:
1. Verifying package sources
2. Checking for typosquatting
3. Validating package names against known patterns
4. Detecting suspicious package sources

Usage:
    python scripts/check_dependency_confusion.py [requirements.txt]
"""

import sys
import re
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple


# Known legitimate packages (common dependencies)
KNOWN_PACKAGES = {
    'requests', 'urllib3', 'certifi', 'numpy', 'pandas', 'django', 
    'flask', 'pytest', 'fastapi', 'pydantic', 'sqlalchemy', 'celery',
    'redis', 'boto3', 'tensorflow', 'torch', 'scipy', 'matplotlib',
    'pillow', 'cryptography', 'bcrypt', 'jwt', 'black', 'mypy'
}

# Suspicious patterns in package names
SUSPICIOUS_PATTERNS = [
    r'.*[0O][0O].*',  # Zero/O confusion
    r'.*[1Il].*',      # One/I/l confusion
    r'.*-test$',       # Test packages
    r'.*-dev$',        # Dev packages
    r'^test-.*',       # Test prefix
]

# Company/internal package prefixes (customize for your org)
INTERNAL_PREFIXES = [
    'company-',
    'internal-',
    'private-',
    'corp-',
]


class DependencyChecker:
    """Check for dependency confusion vulnerabilities"""
    
    def __init__(self, requirements_file: str = 'requirements.txt'):
        self.requirements_file = Path(requirements_file)
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        
    def parse_requirements(self) -> List[Tuple[str, str]]:
        """Parse requirements file and return list of (package, version) tuples"""
        packages = []
        
        if not self.requirements_file.exists():
            print(f"❌ Error: {self.requirements_file} not found")
            sys.exit(1)
            
        with open(self.requirements_file) as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                    
                # Skip options
                if line.startswith('-'):
                    continue
                
                # Extract package name and version
                match = re.match(r'^([a-zA-Z0-9_-]+)[>=<~!]=?(.+)?', line)
                if match:
                    package = match.group(1).lower()
                    version = match.group(2) if match.group(2) else 'any'
                    packages.append((package, version))
                    
        return packages
    
    def check_external_sources(self) -> None:
        """Check for non-PyPI package sources"""
        with open(self.requirements_file) as f:
            content = f.read()
            
        # Check for external sources
        external_patterns = [
            (r'git\+https?://', 'Git repository'),
            (r'https?://[^/]+/[^/]+\.git', 'Direct Git URL'),
            (r'https?://(?!pypi\.org|files\.pythonhosted\.org)', 'Non-PyPI HTTP source'),
            (r'--index-url', 'Custom package index'),
            (r'--extra-index-url', 'Additional package index'),
        ]
        
        for pattern, description in external_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                self.warnings.append({
                    'type': 'external_source',
                    'severity': 'medium',
                    'description': f'{description} detected',
                    'details': matches,
                    'recommendation': 'Verify these sources are trusted and authorized'
                })
    
    def check_typosquatting(self, packages: List[Tuple[str, str]]) -> None:
        """Check for potential typosquatting"""
        for package, version in packages:
            # Check against known packages
            for known in KNOWN_PACKAGES:
                # Levenshtein distance approximation
                if self._similar_names(package, known) and package != known:
                    self.issues.append({
                        'type': 'typosquatting',
                        'severity': 'high',
                        'package': package,
                        'similar_to': known,
                        'description': f'Package "{package}" is similar to known package "{known}"',
                        'recommendation': f'Verify this is not a typosquatting attempt. Did you mean "{known}"?'
                    })
            
            # Check suspicious patterns
            for pattern in SUSPICIOUS_PATTERNS:
                if re.match(pattern, package):
                    self.warnings.append({
                        'type': 'suspicious_pattern',
                        'severity': 'low',
                        'package': package,
                        'pattern': pattern,
                        'description': f'Package "{package}" matches suspicious pattern',
                        'recommendation': 'Verify this package is legitimate'
                    })
    
    def check_internal_packages(self, packages: List[Tuple[str, str]]) -> None:
        """Check for internal package conflicts"""
        for package, version in packages:
            for prefix in INTERNAL_PREFIXES:
                if package.startswith(prefix):
                    self.warnings.append({
                        'type': 'internal_package',
                        'severity': 'medium',
                        'package': package,
                        'description': f'Package "{package}" appears to be an internal package',
                        'recommendation': 'Ensure private package index is configured to prevent dependency confusion'
                    })
    
    def check_package_existence(self, packages: List[Tuple[str, str]]) -> None:
        """Check if packages exist on PyPI"""
        try:
            for package, version in packages[:5]:  # Limit to avoid rate limiting
                result = subprocess.run(
                    ['pip', 'index', 'versions', package],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if 'No matching distribution found' in result.stderr:
                    self.issues.append({
                        'type': 'nonexistent_package',
                        'severity': 'critical',
                        'package': package,
                        'description': f'Package "{package}" not found on PyPI',
                        'recommendation': 'Verify package name is correct. This could be a private package or typo.'
                    })
        except subprocess.TimeoutExpired:
            self.warnings.append({
                'type': 'check_timeout',
                'severity': 'low',
                'description': 'Package existence check timed out',
                'recommendation': 'Manually verify packages exist on PyPI'
            })
        except Exception as e:
            self.warnings.append({
                'type': 'check_error',
                'severity': 'low',
                'description': f'Error checking package existence: {str(e)}',
                'recommendation': 'Manually verify packages'
            })
    
    def check_hash_verification(self) -> None:
        """Check if hash-based verification is used"""
        hashed_file = self.requirements_file.parent / 'requirements-hashed.txt'
        
        if not hashed_file.exists():
            self.warnings.append({
                'type': 'no_hash_verification',
                'severity': 'medium',
                'description': 'No requirements-hashed.txt found',
                'recommendation': 'Use hash-based verification for production: pip-compile --generate-hashes'
            })
        else:
            # Check if hashes are present
            with open(hashed_file) as f:
                content = f.read()
                if '--hash=sha256:' not in content:
                    self.warnings.append({
                        'type': 'invalid_hash_format',
                        'severity': 'high',
                        'description': 'requirements-hashed.txt exists but lacks proper hash verification',
                        'recommendation': 'Regenerate with: pip-compile --generate-hashes'
                    })
    
    def _similar_names(self, name1: str, name2: str) -> bool:
        """Check if two package names are similar (simple check)"""
        if len(name1) != len(name2):
            return False
            
        # Check if only 1-2 characters different
        diff_count = sum(c1 != c2 for c1, c2 in zip(name1, name2))
        return diff_count <= 2
    
    def run_checks(self) -> bool:
        """Run all checks and return True if issues found"""
        print("🔍 Checking for dependency confusion vulnerabilities...\n")
        
        # Parse requirements
        packages = self.parse_requirements()
        print(f"📦 Found {len(packages)} packages in {self.requirements_file}\n")
        
        # Run checks
        self.check_external_sources()
        self.check_typosquatting(packages)
        self.check_internal_packages(packages)
        self.check_hash_verification()
        # self.check_package_existence(packages)  # Can be slow
        
        # Report results
        self._report_results()
        
        return len(self.issues) > 0
    
    def _report_results(self) -> None:
        """Print results"""
        if not self.issues and not self.warnings:
            print("✅ No dependency confusion vulnerabilities detected!")
            return
        
        # Report issues (high severity)
        if self.issues:
            print("❌ ISSUES DETECTED:\n")
            for issue in self.issues:
                print(f"  🚨 [{issue['severity'].upper()}] {issue['type']}")
                print(f"     {issue['description']}")
                print(f"     → {issue['recommendation']}")
                if 'details' in issue:
                    print(f"     Details: {issue['details']}")
                print()
        
        # Report warnings (lower severity)
        if self.warnings:
            print("⚠️  WARNINGS:\n")
            for warning in self.warnings:
                print(f"  ⚠️  [{warning['severity'].upper()}] {warning['type']}")
                print(f"     {warning['description']}")
                print(f"     → {warning['recommendation']}")
                if 'details' in warning:
                    print(f"     Details: {warning['details']}")
                print()
        
        # Summary
        print("\n" + "="*60)
        print(f"Summary: {len(self.issues)} issues, {len(self.warnings)} warnings")
        print("="*60)


def main():
    """Main entry point"""
    requirements_file = sys.argv[1] if len(sys.argv) > 1 else 'requirements.txt'
    
    checker = DependencyChecker(requirements_file)
    has_issues = checker.run_checks()
    
    # Exit with error code if issues found
    sys.exit(1 if has_issues else 0)


if __name__ == '__main__':
    main()
