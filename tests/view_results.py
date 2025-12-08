#!/usr/bin/env python3
"""
Results Viewer for Nethical Advanced Tests
==========================================

This script provides easy access to test results and reports.
"""

import os
import webbrowser
from pathlib import Path
import argparse
import json
from datetime import datetime


def find_latest_report():
    """Find the latest test report files."""
    results_dir = Path("tests/results")

    # Find latest HTML report
    html_reports = list(results_dir.glob("reports/test_report_*.html"))
    latest_html = max(html_reports, key=os.path.getctime) if html_reports else None

    # Find latest text summary
    text_summaries = list(results_dir.glob("summary/test_summary_*.txt"))
    latest_text = max(text_summaries, key=os.path.getctime) if text_summaries else None

    # Find latest JSON report
    json_reports = list(results_dir.glob("reports/full_report_*.json"))
    latest_json = max(json_reports, key=os.path.getctime) if json_reports else None

    return latest_html, latest_text, latest_json


def show_summary():
    """Show a quick summary of the latest test results."""
    _, latest_text, latest_json = find_latest_report()

    if latest_json and latest_json.exists():
        with open(latest_json, "r") as f:
            data = json.load(f)

        print("🔥 NETHICAL ADVANCED TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Timestamp: {data['timestamp']}")
        print(f"Duration: {data['duration']:.2f}s")
        print(f"Total Classes: {data['total_classes']}")
        print(f"Total Tests: {data['total_tests']}")
        print(f"Passed: {data['passed']}")
        print(f"Failed: {data['failed']}")
        print(
            f"Success Rate: {(data['passed'] / max(data['total_tests'], 1)) * 100:.2f}%"
        )
        print()

        print("TEST CLASS BREAKDOWN:")
        print("-" * 30)
        for test_class in data["test_classes"]:
            success_rate = (
                test_class["passed"] / max(test_class["total_tests"], 1)
            ) * 100
            status_icon = (
                "✅" if success_rate == 100 else "⚠️" if success_rate >= 50 else "❌"
            )
            print(
                f"{status_icon} {test_class['class_name']}: {test_class['passed']}/{test_class['total_tests']} ({success_rate:.1f}%)"
            )

    elif latest_text and latest_text.exists():
        print("📄 SHOWING TEXT SUMMARY:")
        print("=" * 50)
        with open(latest_text, "r") as f:
            content = f.read()
        print(content)

    else:
        print("❌ No test results found. Please run the tests first:")
        print("   python tests/run_advanced_tests.py")


def open_html_report():
    """Open the latest HTML report in the default browser."""
    latest_html, _, _ = find_latest_report()

    if latest_html and latest_html.exists():
        abs_path = latest_html.resolve()
        print(f"🌐 Opening HTML report: {abs_path}")
        webbrowser.open(f"file://{abs_path}")
    else:
        print("❌ No HTML report found. Please run the tests first:")
        print("   python tests/run_advanced_tests.py")


def list_available_reports():
    """List all available reports."""
    results_dir = Path("tests/results")

    print("📁 AVAILABLE REPORTS:")
    print("=" * 30)

    # HTML Reports
    html_reports = sorted(
        results_dir.glob("reports/test_report_*.html"),
        key=os.path.getctime,
        reverse=True,
    )
    if html_reports:
        print("\n🌐 HTML Reports:")
        for report in html_reports:
            mtime = datetime.fromtimestamp(os.path.getmtime(report))
            print(f"  - {report.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")

    # Text Summaries
    text_summaries = sorted(
        results_dir.glob("summary/test_summary_*.txt"),
        key=os.path.getctime,
        reverse=True,
    )
    if text_summaries:
        print("\n📄 Text Summaries:")
        for summary in text_summaries:
            mtime = datetime.fromtimestamp(os.path.getmtime(summary))
            print(f"  - {summary.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")

    # JSON Reports
    json_reports = sorted(
        results_dir.glob("reports/full_report_*.json"),
        key=os.path.getctime,
        reverse=True,
    )
    if json_reports:
        print("\n📊 JSON Reports:")
        for report in json_reports:
            mtime = datetime.fromtimestamp(os.path.getmtime(report))
            print(f"  - {report.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")

    # Individual Results
    individual_results = sorted(
        results_dir.glob("individual/*_detailed.json"),
        key=os.path.getctime,
        reverse=True,
    )
    if individual_results:
        print(f"\n🧪 Individual Test Class Results: {len(individual_results)} files")
        print("   Use --class <classname> to view specific class results")


def show_class_results(class_name):
    """Show results for a specific test class."""
    results_dir = Path("tests/results/individual")

    # Find the detailed JSON file for the class
    pattern = f"{class_name.lower()}_detailed.json"
    json_file = results_dir / pattern

    if not json_file.exists():
        # Try to find a partial match
        matches = list(results_dir.glob(f"*{class_name.lower()}*_detailed.json"))
        if matches:
            json_file = matches[0]
        else:
            print(f"❌ No results found for test class: {class_name}")
            print("\nAvailable test classes:")
            for file in results_dir.glob("*_detailed.json"):
                class_name_from_file = file.stem.replace("_detailed", "").replace(
                    "test", "Test"
                )
                print(f"  - {class_name_from_file}")
            return

    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"🧪 TEST CLASS RESULTS: {data['class_name']}")
    print("=" * 50)
    print(f"Total Tests: {data['total_tests']}")
    print(f"Passed: {data['passed']}")
    print(f"Failed: {data['failed']}")
    print(f"Duration: {data['duration']:.2f}s")
    print(f"Success Rate: {(data['passed'] / max(data['total_tests'], 1)) * 100:.2f}%")
    print()

    print("INDIVIDUAL TEST RESULTS:")
    print("-" * 30)

    for test in data["tests"]:
        status_icon = {
            "PASSED": "✅",
            "FAILED": "❌",
            "SKIPPED": "⏸️",
            "ERROR": "🔥",
        }.get(test["status"], "❓")
        print(f"{status_icon} {test['test_name']} ({test['status']})")

        if test["error_message"]:
            # Show first line of error message
            first_line = test["error_message"].split("\n")[0]
            print(f"   Error: {first_line}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="View Nethical test results")
    parser.add_argument(
        "--summary", "-s", action="store_true", help="Show test summary"
    )
    parser.add_argument(
        "--html", action="store_true", help="Open HTML report in browser"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available reports"
    )
    parser.add_argument(
        "--class", "-c", dest="class_name", help="Show results for specific test class"
    )

    args = parser.parse_args()

    if args.html:
        open_html_report()
    elif args.list:
        list_available_reports()
    elif args.class_name:
        show_class_results(args.class_name)
    elif args.summary:
        show_summary()
    else:
        # Default: show summary
        show_summary()


if __name__ == "__main__":
    main()
