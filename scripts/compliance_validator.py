#!/usr/bin/env python3
"""Compliance Validator CLI Tool for Nethical.

This script provides a command-line interface for validating compliance
against regulatory frameworks including GDPR, EU AI Act, CCPA, ISO 27001,
and NIST AI RMF.

Usage:
    python scripts/compliance_validator.py [OPTIONS] FRAMEWORK

Examples:
    # Validate all frameworks
    python scripts/compliance_validator.py all

    # Validate GDPR only
    python scripts/compliance_validator.py gdpr

    # Validate EU AI Act with output file
    python scripts/compliance_validator.py eu_ai_act --output reports/eu_ai_act.json

    # Validate with custom config file
    python scripts/compliance_validator.py all --config config/compliance.yaml

Adheres to the 25 Fundamental Laws:
- Law 15: Audit Compliance - Comprehensive compliance auditing
- Law 10: Reasoning Transparency - Clear validation results

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from nethical.compliance import (
    ComplianceValidator,
    ComplianceFramework,
    ComplianceReport,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def color_status(status: str) -> str:
    """Add color to status based on value.

    Args:
        status: Status string

    Returns:
        Colored status string
    """
    status_lower = status.lower()
    if status_lower == "compliant":
        return f"{Colors.GREEN}✓ COMPLIANT{Colors.ENDC}"
    elif status_lower == "partial":
        return f"{Colors.YELLOW}◐ PARTIAL{Colors.ENDC}"
    elif status_lower == "non_compliant":
        return f"{Colors.RED}✗ NON-COMPLIANT{Colors.ENDC}"
    else:
        return f"{Colors.CYAN}○ {status.upper()}{Colors.ENDC}"


def print_header(text: str) -> None:
    """Print a section header.

    Args:
        text: Header text
    """
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")


def print_section(text: str) -> None:
    """Print a section title.

    Args:
        text: Section text
    """
    print(f"\n{Colors.BOLD}{Colors.BLUE}--- {text} ---{Colors.ENDC}\n")


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file.

    Args:
        config_path: Path to config file (JSON or YAML)

    Returns:
        Configuration dictionary
    """
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s", config_path)
        return {}

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        if not YAML_AVAILABLE:
            logger.error("PyYAML not installed. Install with: pip install pyyaml")
            return {}
        return yaml.safe_load(content)
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        logger.warning("Unknown config format: %s", path.suffix)
        return {}


def print_report_summary(report: ComplianceReport) -> None:
    """Print compliance report summary to console.

    Args:
        report: ComplianceReport to print
    """
    print_header("NETHICAL COMPLIANCE VALIDATION REPORT")

    # Report metadata
    print(f"{Colors.CYAN}Report ID:{Colors.ENDC} {report.report_id}")
    print(f"{Colors.CYAN}Generated:{Colors.ENDC} {report.generated_at.isoformat()}")
    print(
        f"{Colors.CYAN}Frameworks:{Colors.ENDC} {', '.join(f.value for f in report.frameworks_validated)}"
    )

    # Overall status
    print_section("OVERALL STATUS")
    print(f"Status: {color_status(report.overall_status.value)}")

    # Score bar
    score = report.compliance_score
    bar_width = 40
    filled = int(bar_width * score / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    if score >= 80:
        score_color = Colors.GREEN
    elif score >= 60:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED

    print(f"Score: {score_color}[{bar}] {score:.1f}%{Colors.ENDC}")

    # Framework summaries
    if report.gdpr_summary:
        print_section("GDPR SUMMARY")
        summary = report.gdpr_summary
        print(f"  Total Checks: {summary.get('total_checks', 0)}")
        print(f"  Status: {summary.get('overall_status', 'N/A')}")
        print(f"  Score: {summary.get('compliance_score', 0):.1f}%")

    if report.eu_ai_act_summary:
        print_section("EU AI ACT SUMMARY")
        summary = report.eu_ai_act_summary
        print(f"  Risk Level: {summary.get('risk_level', 'N/A')}")
        print(f"  Articles Validated: {summary.get('total_articles_validated', 0)}")
        print(f"  Score: {summary.get('compliance_score', 0):.1f}%")
        print(
            f"  Certification Ready: {'Yes' if summary.get('certification_ready') else 'No'}"
        )

    # Validation results by framework
    print_section("VALIDATION RESULTS")

    current_framework = None
    for result in report.validation_results:
        if result.framework != current_framework:
            current_framework = result.framework
            print(f"\n{Colors.BOLD}{current_framework.value.upper()}{Colors.ENDC}")

        print(
            f"  [{result.check_id}] {result.check_name}: {color_status(result.status.value)}"
        )

        if result.gaps:
            for gap in result.gaps[:2]:  # Show first 2 gaps
                print(f"    {Colors.RED}• {gap}{Colors.ENDC}")

    # Recommendations
    if report.recommendations:
        print_section("RECOMMENDATIONS")
        for i, rec in enumerate(report.recommendations[:10], 1):  # Show first 10
            print(f"  {i}. {rec}")

        if len(report.recommendations) > 10:
            print(f"  ... and {len(report.recommendations) - 10} more")

    # Data residency summary
    if report.data_residency_summary:
        print_section("DATA RESIDENCY")
        summary = report.data_residency_summary
        violations = summary.get("total_violations", 0)
        if violations > 0:
            print(f"  {Colors.RED}Violations: {violations}{Colors.ENDC}")
        else:
            print(f"  {Colors.GREEN}No violations detected{Colors.ENDC}")

    print("\n" + "=" * 60)


def run_validation(
    framework: str,
    config: Dict[str, Any],
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Run compliance validation.

    Args:
        framework: Framework to validate
        config: Validation configuration
        output_path: Optional output file path
        verbose: Enable verbose output

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    # Map string to enum
    framework_map = {
        "all": ComplianceFramework.ALL,
        "gdpr": ComplianceFramework.GDPR,
        "ccpa": ComplianceFramework.CCPA,
        "eu_ai_act": ComplianceFramework.EU_AI_ACT,
        "iso_27001": ComplianceFramework.ISO_27001,
        "nist_ai_rmf": ComplianceFramework.NIST_AI_RMF,
    }

    framework_enum = framework_map.get(framework.lower())
    if not framework_enum:
        logger.error("Unknown framework: %s", framework)
        print(f"Valid frameworks: {', '.join(framework_map.keys())}")
        return 1

    # Initialize validator
    system_characteristics = config.get(
        "system_characteristics",
        {
            "critical_infrastructure": True,  # Enables high-risk AI evaluation
        },
    )

    validator = ComplianceValidator(system_characteristics=system_characteristics)

    # Run validation
    logger.info("Running compliance validation for: %s", framework)
    report = validator.validate(
        framework=framework_enum,
        configs=config,
    )

    # Print report
    print_report_summary(report)

    # Save report if output path specified
    if output_path:
        report.save(output_path)
        print(f"\n{Colors.GREEN}Report saved to: {output_path}{Colors.ENDC}")

    # Return exit code based on status
    if report.overall_status.value == "compliant":
        return 0
    elif report.overall_status.value == "partial":
        return 0  # Partial is acceptable for CI
    else:
        return 1


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Nethical Compliance Validator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s all                    # Validate all frameworks
  %(prog)s gdpr --verbose         # Validate GDPR with verbose output
  %(prog)s eu_ai_act -o report.json  # Validate EU AI Act, save report
  
Frameworks:
  all         - All frameworks
  gdpr        - EU General Data Protection Regulation
  ccpa        - California Consumer Privacy Act
  eu_ai_act   - EU AI Act (Regulation 2024/1689)
  iso_27001   - ISO 27001 Information Security
  nist_ai_rmf - NIST AI Risk Management Framework
        """,
    )

    parser.add_argument(
        "framework",
        type=str,
        nargs="?",
        default="all",
        help="Framework to validate (default: all)",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file (JSON or YAML)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save output report (JSON)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_config(args.config)

    try:
        return run_validation(
            framework=args.framework,
            config=config,
            output_path=args.output,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\nValidation cancelled by user.")
        return 130
    except Exception as e:
        logger.exception("Validation failed with error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
