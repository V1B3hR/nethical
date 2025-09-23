#!/usr/bin/env python3
"""
Advanced Test Runner for Nethical
================================

This script runs the comprehensive advanced test suite and organizes results
into structured reports saved in the tests/results folder.

Features:
- Runs all test classes in advancedtests.py
- Collects detailed results including timing, failures, and statistics
- Organizes results by test class/category
- Generates JSON, HTML, and text reports
- Saves individual test class results
"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest
import sys
import subprocess
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    test_class: str
    status: str  # PASSED, FAILED, SKIPPED, ERROR
    duration: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


@dataclass
class TestClassResult:
    """Results for a test class."""
    class_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    tests: List[TestResult]


@dataclass
class OverallResult:
    """Overall test run results."""
    timestamp: str
    total_classes: int
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    test_classes: List[TestClassResult]


class AdvancedTestRunner:
    """Advanced test runner with detailed reporting."""
    
    def __init__(self, test_file: str = "tests/advancedtests.py"):
        self.test_file = test_file
        self.results_dir = Path("tests/results")
        self.individual_dir = self.results_dir / "individual"
        self.summary_dir = self.results_dir / "summary"
        self.reports_dir = self.results_dir / "reports"
        
        # Ensure directories exist
        for dir_path in [self.individual_dir, self.summary_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def discover_test_classes(self) -> List[str]:
        """Discover test classes in the advanced test file."""
        test_classes = []
        try:
            with open(self.test_file, 'r') as f:
                content = f.read()
                
            # Look for test classes
            import re
            class_pattern = r'^class (Test\w+):'
            matches = re.findall(class_pattern, content, re.MULTILINE)
            test_classes = matches
            
        except Exception as e:
            print(f"Error discovering test classes: {e}")
        
        return test_classes
    
    def run_test_class(self, test_class: str) -> TestClassResult:
        """Run tests for a specific test class."""
        print(f"\n🧪 Running test class: {test_class}")
        print("=" * 60)
        
        start_time = time.time()
        test_results = []
        
        # Run pytest for the specific test class
        cmd = [
            sys.executable, "-m", "pytest",
            f"{self.test_file}::{test_class}",
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.individual_dir}/{test_class.lower()}_results.json"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per test class
            )
            
            duration = time.time() - start_time
            
            # Parse the JSON report if it exists
            json_report_file = self.individual_dir / f"{test_class.lower()}_results.json"
            if json_report_file.exists():
                try:
                    with open(json_report_file, 'r') as f:
                        json_data = json.load(f)
                    
                    # Extract test results from JSON
                    for test in json_data.get('tests', []):
                        test_name = test.get('nodeid', '').split('::')[-1]
                        status = test.get('outcome', 'UNKNOWN').upper()
                        test_duration = test.get('duration', 0.0)
                        
                        error_message = None
                        error_traceback = None
                        if status == 'FAILED':
                            if 'call' in test and 'longrepr' in test['call']:
                                error_message = test['call']['longrepr'][:500]  # Truncate long errors
                                error_traceback = test['call'].get('traceback', '')
                        
                        test_results.append(TestResult(
                            test_name=test_name,
                            test_class=test_class,
                            status=status,
                            duration=test_duration,
                            error_message=error_message,
                            error_traceback=error_traceback
                        ))
                
                except Exception as e:
                    print(f"Error parsing JSON report: {e}")
            
            # If we couldn't parse JSON, fall back to basic parsing
            if not test_results:
                # Parse basic output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if '::' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line):
                        parts = line.split()
                        if len(parts) >= 2:
                            test_name = parts[0].split('::')[-1]
                            status = parts[1]
                            test_results.append(TestResult(
                                test_name=test_name,
                                test_class=test_class,
                                status=status,
                                duration=0.0
                            ))
            
            # Calculate statistics
            passed = sum(1 for t in test_results if t.status == 'PASSED')
            failed = sum(1 for t in test_results if t.status == 'FAILED')
            skipped = sum(1 for t in test_results if t.status == 'SKIPPED')
            errors = sum(1 for t in test_results if t.status in ['ERROR', 'INTERRUPTED'])
            
            class_result = TestClassResult(
                class_name=test_class,
                total_tests=len(test_results),
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration=duration,
                tests=test_results
            )
            
            # Save individual class results
            self.save_class_results(class_result, result.stdout, result.stderr)
            
            # Print summary
            print(f"✅ {test_class} completed:")
            print(f"   Total: {len(test_results)}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}, Errors: {errors}")
            print(f"   Duration: {duration:.2f}s")
            
            return class_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {test_class} timed out after {duration:.2f}s")
            return TestClassResult(
                class_name=test_class,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=duration,
                tests=[TestResult(
                    test_name="timeout",
                    test_class=test_class,
                    status="ERROR",
                    duration=duration,
                    error_message="Test class timed out"
                )]
            )
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Error running {test_class}: {e}")
            return TestClassResult(
                class_name=test_class,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=duration,
                tests=[TestResult(
                    test_name="error",
                    test_class=test_class,
                    status="ERROR",
                    duration=duration,
                    error_message=str(e)
                )]
            )
    
    def save_class_results(self, class_result: TestClassResult, stdout: str, stderr: str):
        """Save individual test class results."""
        class_name = class_result.class_name.lower()
        
        # Save JSON results
        json_file = self.individual_dir / f"{class_name}_detailed.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(class_result), f, indent=2, default=str)
        
        # Save text output
        text_file = self.individual_dir / f"{class_name}_output.txt"
        with open(text_file, 'w') as f:
            f.write(f"Test Class: {class_result.class_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n")
            f.write("STDOUT:\n")
            f.write(stdout)
            f.write("\n" + "=" * 60 + "\n")
            f.write("STDERR:\n")
            f.write(stderr)
            f.write("\n")
    
    def generate_reports(self, overall_result: OverallResult):
        """Generate comprehensive reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        json_file = self.reports_dir / f"full_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(overall_result), f, indent=2, default=str)
        
        # Generate text summary
        text_file = self.summary_dir / f"test_summary_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write("NETHICAL ADVANCED TEST SUITE RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {overall_result.timestamp}\n")
            f.write(f"Total Duration: {overall_result.duration:.2f}s\n\n")
            
            f.write("OVERALL SUMMARY:\n")
            f.write(f"  Total Test Classes: {overall_result.total_classes}\n")
            f.write(f"  Total Tests: {overall_result.total_tests}\n")
            f.write(f"  Passed: {overall_result.passed}\n")
            f.write(f"  Failed: {overall_result.failed}\n")
            f.write(f"  Skipped: {overall_result.skipped}\n")
            f.write(f"  Errors: {overall_result.errors}\n")
            
            success_rate = (overall_result.passed / max(overall_result.total_tests, 1)) * 100
            f.write(f"  Success Rate: {success_rate:.2f}%\n\n")
            
            f.write("DETAILED RESULTS BY TEST CLASS:\n")
            f.write("-" * 50 + "\n")
            
            for class_result in overall_result.test_classes:
                f.write(f"\n{class_result.class_name}:\n")
                f.write(f"  Tests: {class_result.total_tests}\n")
                f.write(f"  Passed: {class_result.passed}\n")
                f.write(f"  Failed: {class_result.failed}\n")
                f.write(f"  Skipped: {class_result.skipped}\n")
                f.write(f"  Errors: {class_result.errors}\n")
                f.write(f"  Duration: {class_result.duration:.2f}s\n")
                
                if class_result.failed > 0:
                    f.write("  Failed Tests:\n")
                    for test in class_result.tests:
                        if test.status == 'FAILED':
                            f.write(f"    - {test.test_name}: {test.error_message or 'No details'}\n")
        
        # Generate HTML report
        html_file = self.reports_dir / f"test_report_{timestamp}.html"
        self.generate_html_report(overall_result, html_file)
        
        print(f"\n📊 Reports generated:")
        print(f"  JSON: {json_file}")
        print(f"  Text: {text_file}")
        print(f"  HTML: {html_file}")
    
    def generate_html_report(self, overall_result: OverallResult, html_file: Path):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Nethical Advanced Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .test-class {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .test-class-header {{ background-color: #f8f8f8; padding: 10px; font-weight: bold; }}
        .test-list {{ padding: 10px; }}
        .test-item {{ padding: 5px; border-bottom: 1px solid #eee; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .skipped {{ color: orange; }}
        .error {{ color: purple; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 Nethical Advanced Test Suite Results</h1>
        <p><strong>Execution Time:</strong> {overall_result.timestamp}</p>
        <p><strong>Total Duration:</strong> {overall_result.duration:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <div style="font-size: 24px;">{overall_result.total_tests}</div>
        </div>
        <div class="metric">
            <h3>Passed</h3>
            <div style="font-size: 24px; color: green;">{overall_result.passed}</div>
        </div>
        <div class="metric">
            <h3>Failed</h3>
            <div style="font-size: 24px; color: red;">{overall_result.failed}</div>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <div style="font-size: 24px;">{(overall_result.passed / max(overall_result.total_tests, 1)) * 100:.1f}%</div>
        </div>
    </div>
    
    <h2>Test Classes</h2>
"""
        
        for class_result in overall_result.test_classes:
            success_rate = (class_result.passed / max(class_result.total_tests, 1)) * 100
            html_content += f"""
    <div class="test-class">
        <div class="test-class-header">
            {class_result.class_name} - {class_result.total_tests} tests, {success_rate:.1f}% passed ({class_result.duration:.2f}s)
        </div>
        <div class="test-list">
"""
            for test in class_result.tests:
                status_class = test.status.lower()
                html_content += f"""
            <div class="test-item">
                <span class="{status_class}">●</span> {test.test_name} 
                <small>({test.status}, {test.duration:.2f}s)</small>
"""
                if test.error_message:
                    html_content += f"<br><small style='color: #666;'>{test.error_message[:200]}...</small>"
                html_content += "</div>\n"
            
            html_content += """
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def run_all_tests(self) -> OverallResult:
        """Run all advanced tests and collect results."""
        print("🔥 INITIATING NETHICAL ADVANCED TEST SUITE")
        print("=" * 60)
        
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Discover test classes
        test_classes = self.discover_test_classes()
        print(f"📋 Discovered {len(test_classes)} test classes:")
        for cls in test_classes:
            print(f"  - {cls}")
        
        if not test_classes:
            print("❌ No test classes found!")
            return OverallResult(
                timestamp=timestamp,
                total_classes=0,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                duration=0.0,
                test_classes=[]
            )
        
        # Run each test class
        class_results = []
        for test_class in test_classes:
            try:
                class_result = self.run_test_class(test_class)
                class_results.append(class_result)
            except Exception as e:
                print(f"❌ Critical error running {test_class}: {e}")
                class_results.append(TestClassResult(
                    class_name=test_class,
                    total_tests=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    errors=1,
                    duration=0.0,
                    tests=[TestResult(
                        test_name="critical_error",
                        test_class=test_class,
                        status="ERROR",
                        duration=0.0,
                        error_message=str(e)
                    )]
                ))
        
        duration = time.time() - start_time
        
        # Calculate overall statistics
        total_tests = sum(r.total_tests for r in class_results)
        passed = sum(r.passed for r in class_results)
        failed = sum(r.failed for r in class_results)
        skipped = sum(r.skipped for r in class_results)
        errors = sum(r.errors for r in class_results)
        
        overall_result = OverallResult(
            timestamp=timestamp,
            total_classes=len(test_classes),
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            test_classes=class_results
        )
        
        # Generate reports
        self.generate_reports(overall_result)
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🏁 FINAL RESULTS")
        print("=" * 60)
        print(f"Total Test Classes: {len(test_classes)}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Errors: {errors}")
        print(f"Success Rate: {(passed / max(total_tests, 1)) * 100:.2f}%")
        print(f"Total Duration: {duration:.2f}s")
        print("=" * 60)
        
        return overall_result


def main():
    """Main entry point."""
    runner = AdvancedTestRunner()
    result = runner.run_all_tests()
    
    # Exit with appropriate code
    if result.failed > 0 or result.errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()