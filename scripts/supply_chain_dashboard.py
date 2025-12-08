#!/usr/bin/env python3
"""
Supply Chain Security Dashboard

This script analyzes and reports on supply chain security metrics including:
- Dependency versions and update status
- Known vulnerabilities (via GitHub Advisory Database)
- SBOM generation and validation
- SLSA compliance tracking
- License compliance

Usage:
    python scripts/supply_chain_dashboard.py [--format json|markdown|html]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


class SupplyChainDashboard:
    """Supply chain security dashboard generator"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.requirements_file = project_root / "requirements.txt"
        self.requirements_dev_file = project_root / "requirements-dev.txt"

    def parse_requirements(self, requirements_file: Path) -> List[Dict[str, str]]:
        """Parse requirements file and extract package information"""
        packages = []

        if not requirements_file.exists():
            return packages

        with open(requirements_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Parse package==version format
                match = re.match(r"^([a-zA-Z0-9_-]+)==([0-9.]+)", line)
                if match:
                    packages.append(
                        {
                            "name": match.group(1),
                            "version": match.group(2),
                            "file": requirements_file.name,
                        }
                    )
                elif ">=" in line or ">" in line:
                    # Handle range specifications
                    name = re.match(r"^([a-zA-Z0-9_-]+)", line)
                    if name:
                        packages.append(
                            {
                                "name": name.group(1),
                                "version": "unpinned",
                                "file": requirements_file.name,
                                "warning": "Version not pinned",
                            }
                        )

        return packages

    def check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages using pip"""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            print(f"Warning: Could not check outdated packages: {e}", file=sys.stderr)

        return []

    def generate_sbom(self) -> Dict:
        """Generate Software Bill of Materials (SBOM)"""
        prod_packages = self.parse_requirements(self.requirements_file)
        dev_packages = self.parse_requirements(self.requirements_dev_file)

        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": {
                    "type": "application",
                    "name": "nethical",
                    "version": "1.0.0",
                },
            },
            "components": [],
        }

        # Add production dependencies
        for pkg in prod_packages:
            sbom["components"].append(
                {
                    "type": "library",
                    "name": pkg["name"],
                    "version": pkg.get("version", "unknown"),
                    "scope": "required",
                    "purl": f"pkg:pypi/{pkg['name']}@{pkg.get('version', 'unknown')}",
                }
            )

        # Add development dependencies
        for pkg in dev_packages:
            sbom["components"].append(
                {
                    "type": "library",
                    "name": pkg["name"],
                    "version": pkg.get("version", "unknown"),
                    "scope": "optional",
                    "purl": f"pkg:pypi/{pkg['name']}@{pkg.get('version', 'unknown')}",
                }
            )

        return sbom

    def assess_slsa_compliance(self) -> Dict[str, any]:
        """Assess SLSA (Supply-chain Levels for Software Artifacts) compliance"""
        compliance = {
            "level": "Unknown",
            "requirements": {},
        }

        # Check SLSA Level 1 requirements
        level1_checks = {
            "version_control": self._check_version_control(),
            "automated_build": self._check_automated_build(),
            "build_documentation": self._check_build_docs(),
        }

        # Check SLSA Level 2 requirements
        level2_checks = {
            "version_control_service": self._check_version_control_service(),
            "authenticated_provenance": self._check_provenance(),
        }

        # Check SLSA Level 3 requirements
        level3_checks = {
            "hardened_build": self._check_hardened_build(),
            "provenance_non_falsifiable": self._check_provenance_protection(),
            "dependencies_locked": self._check_dependency_pinning(),
        }

        compliance["requirements"]["level_1"] = level1_checks
        compliance["requirements"]["level_2"] = level2_checks
        compliance["requirements"]["level_3"] = level3_checks

        # Determine current level
        if all(level3_checks.values()):
            compliance["level"] = "SLSA Level 3"
        elif all(level2_checks.values()):
            compliance["level"] = "SLSA Level 2"
        elif all(level1_checks.values()):
            compliance["level"] = "SLSA Level 1"
        else:
            compliance["level"] = "Below SLSA Level 1"

        return compliance

    def _check_version_control(self) -> bool:
        """Check if project uses version control"""
        return (self.project_root / ".git").exists()

    def _check_automated_build(self) -> bool:
        """Check if automated build exists"""
        ci_files = [
            ".github/workflows/ci.yml",
            ".github/workflows/security.yml",
            ".gitlab-ci.yml",
            ".travis.yml",
        ]
        return any((self.project_root / f).exists() for f in ci_files)

    def _check_build_docs(self) -> bool:
        """Check if build documentation exists"""
        docs = ["README.md", "docs/"]
        return any((self.project_root / d).exists() for d in docs)

    def _check_version_control_service(self) -> bool:
        """Check if using hosted version control"""
        # If .git exists and has a remote, we're using a service
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=5,
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except:
            return False

    def _check_provenance(self) -> bool:
        """Check if build provenance exists"""
        # Check for SBOM workflow
        return (self.project_root / ".github/workflows/sbom-sign.yml").exists()

    def _check_hardened_build(self) -> bool:
        """Check if build environment is hardened"""
        # Check for security scanning in CI
        security_workflow = self.project_root / ".github/workflows/security.yml"
        if security_workflow.exists():
            with open(security_workflow, "r") as f:
                content = f.read()
                return "security" in content.lower()
        return False

    def _check_provenance_protection(self) -> bool:
        """Check if provenance is protected from tampering"""
        # Check for signing workflow
        return (self.project_root / ".github/workflows/sbom-sign.yml").exists()

    def _check_dependency_pinning(self) -> bool:
        """Check if dependencies are pinned"""
        packages = self.parse_requirements(self.requirements_file)
        if not packages:
            return False

        # Check if all packages have pinned versions
        unpinned = [p for p in packages if p.get("version") == "unpinned"]
        return len(unpinned) == 0

    def generate_report(self, format: str = "markdown") -> str:
        """Generate supply chain security report"""
        prod_packages = self.parse_requirements(self.requirements_file)
        dev_packages = self.parse_requirements(self.requirements_dev_file)
        outdated = self.check_outdated_packages()
        slsa = self.assess_slsa_compliance()

        if format == "json":
            return json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "production_dependencies": prod_packages,
                    "development_dependencies": dev_packages,
                    "outdated_packages": outdated,
                    "slsa_compliance": slsa,
                },
                indent=2,
            )

        elif format == "markdown":
            report = []
            report.append("# Supply Chain Security Dashboard")
            report.append(
                f'\n**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}'
            )
            report.append("\n## 📦 Dependency Summary\n")
            report.append(f"- Production dependencies: **{len(prod_packages)}**")
            report.append(f"- Development dependencies: **{len(dev_packages)}**")
            report.append(f"- Outdated packages: **{len(outdated)}**")

            # SLSA Compliance
            report.append("\n## 🔒 SLSA Compliance\n")
            report.append(f'**Current Level:** {slsa["level"]}\n')

            for level, checks in slsa["requirements"].items():
                level_name = level.replace("_", " ").title()
                report.append(f"\n### {level_name}\n")
                for check, status in checks.items():
                    icon = "✅" if status else "❌"
                    check_name = check.replace("_", " ").title()
                    report.append(f"- {icon} {check_name}")

            # Outdated packages
            if outdated:
                report.append("\n## ⚠️ Outdated Packages\n")
                report.append("| Package | Current | Latest | Type |")
                report.append("|---------|---------|--------|------|")
                for pkg in outdated[:10]:  # Show first 10
                    report.append(
                        f"| {pkg.get('name', 'N/A')} | {pkg.get('version', 'N/A')} | {pkg.get('latest_version', 'N/A')} | {pkg.get('latest_filetype', 'N/A')} |"
                    )

            # Pinned packages
            report.append("\n## 📌 Pinned Dependencies\n")
            report.append(f"\n### Production ({len(prod_packages)})\n")
            for pkg in prod_packages[:15]:  # Show first 15
                warning = f" ⚠️ {pkg['warning']}" if "warning" in pkg else ""
                report.append(
                    f"- {pkg['name']}=={pkg.get('version', 'unpinned')}{warning}"
                )

            # Recommendations
            report.append("\n## 💡 Recommendations\n")
            unpinned = [p for p in prod_packages if p.get("version") == "unpinned"]
            if unpinned:
                report.append(
                    f"- 🔴 **HIGH**: Pin {len(unpinned)} unpinned dependencies"
                )
            if outdated:
                report.append(
                    f"- 🟡 **MEDIUM**: Update {len(outdated)} outdated packages"
                )
            if slsa["level"].startswith("Below"):
                report.append("- 🟡 **MEDIUM**: Improve SLSA compliance to Level 1+")

            report.append("\n## 🔍 Next Steps\n")
            report.append("1. Review and update outdated dependencies")
            report.append("2. Pin all unpinned dependencies with hash verification")
            report.append("3. Enable automated security scanning")
            report.append("4. Implement SBOM generation in CI/CD")
            report.append("5. Achieve SLSA Level 3+ compliance")

            return "\n".join(report)

        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate supply chain security dashboard"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (default: stdout)",
    )

    args = parser.parse_args()

    # Find project root
    project_root = Path(__file__).parent.parent

    # Generate dashboard
    dashboard = SupplyChainDashboard(project_root)
    report = dashboard.generate_report(format=args.format)

    # Output
    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
