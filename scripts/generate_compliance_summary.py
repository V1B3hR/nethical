"""Generate a compliance summary from the compliance report JSON."""

import json
import sys


def main():
    try:
        with open("reports/compliance/compliance_report.json") as f:
            report = json.load(f)

        print(f'**Report ID:** {report.get("report_id", "N/A")}')
        print(f'**Generated:** {report.get("generated_at", "N/A")}')
        print(f'**Overall Status:** {report.get("overall_status", "N/A")}')
        print(f'**Compliance Score:** {report.get("compliance_score", 0):.1f}%')
        print("")
        print("### Frameworks Validated")
        for fw in report.get("frameworks_validated", []):
            print(f"- {fw}")
        print("")

        # Count by status
        results = report.get("validation_results", [])
        compliant = sum(1 for r in results if r.get("status") == "compliant")
        partial = sum(1 for r in results if r.get("status") == "partial")
        non_compliant = sum(1 for r in results if r.get("status") == "non_compliant")

        print("### Results Summary")
        print(f"- \u2705 Compliant: {compliant}")
        print(f"- \u26a0\ufe0f Partial: {partial}")
        print(f"- \u274c Non-Compliant: {non_compliant}")

        if report.get("recommendations"):
            print("")
            print("### Top Recommendations")
            for i, rec in enumerate(report["recommendations"][:5], 1):
                print(f"{i}. {rec}")
    except Exception as e:
        print(f"Error reading report: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()
