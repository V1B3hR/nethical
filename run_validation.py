#!/usr/bin/env python3
"""
Nethical Validation Suite Runner

Runs comprehensive validation tests and generates validation.json artifact.
"""

import sys
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class ValidationRunner:
    """Run validation test suites"""
    
    def __init__(self, config_path: str = "validation_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "suites": {},
            "summary": {}
        }
    
    def load_config(self) -> Dict:
        """Load validation configuration"""
        if not self.config_path.exists():
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            return {}
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def run_test_suite(self, suite_name: str, test_path: str) -> Dict:
        """
        Run a test suite and capture results
        
        Args:
            suite_name: Name of the test suite
            test_path: Path to test file
            
        Returns:
            Test results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Running {suite_name}...")
        print(f"{'='*70}")
        
        cmd = [
            "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=validation_reports/{suite_name}_report.json"
        ]
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return {
                "suite": suite_name,
                "status": "passed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "duration_seconds": elapsed,
                "stdout": result.stdout[-2000:],  # Last 2000 chars
                "stderr": result.stderr[-2000:] if result.stderr else ""
            }
        
        except subprocess.TimeoutExpired:
            return {
                "suite": suite_name,
                "status": "timeout",
                "exit_code": -1,
                "duration_seconds": 600,
                "error": "Test suite exceeded 10 minute timeout"
            }
        
        except Exception as e:
            return {
                "suite": suite_name,
                "status": "error",
                "exit_code": -1,
                "error": str(e)
            }
    
    def run_all_suites(self, suites: List[str] = None) -> None:
        """
        Run all validation suites
        
        Args:
            suites: Optional list of specific suites to run
        """
        # Define available suites
        available_suites = {
            "ethics_benchmark": "tests/validation/test_ethics_benchmark.py",
            "drift_detection": "tests/validation/test_drift_detection.py",
            "performance": "tests/validation/test_performance_validation.py",
            "data_integrity": "tests/validation/test_data_integrity.py",
            "explainability": "tests/validation/test_explainability.py",
        }
        
        # Run specified suites or all
        suites_to_run = suites if suites else list(available_suites.keys())
        
        for suite_name in suites_to_run:
            if suite_name not in available_suites:
                print(f"Warning: Unknown suite '{suite_name}', skipping")
                continue
            
            test_path = available_suites[suite_name]
            result = self.run_test_suite(suite_name, test_path)
            self.results["suites"][suite_name] = result
    
    def generate_summary(self) -> None:
        """Generate validation summary"""
        total_suites = len(self.results["suites"])
        passed_suites = sum(1 for s in self.results["suites"].values() if s["status"] == "passed")
        failed_suites = sum(1 for s in self.results["suites"].values() if s["status"] == "failed")
        error_suites = sum(1 for s in self.results["suites"].values() if s["status"] == "error")
        
        total_duration = sum(
            s.get("duration_seconds", 0) 
            for s in self.results["suites"].values()
        )
        
        self.results["summary"] = {
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": failed_suites,
            "error_suites": error_suites,
            "success_rate": passed_suites / total_suites if total_suites > 0 else 0.0,
            "total_duration_seconds": total_duration,
            "overall_status": "passed" if failed_suites == 0 and error_suites == 0 else "failed"
        }
    
    def check_thresholds(self) -> Dict:
        """Check if validation meets defined thresholds"""
        thresholds = self.config.get("metrics", {})
        
        # This would check actual metric values from test reports
        # For now, we check if tests passed
        checks = {
            "ethics_benchmark": self.results["suites"].get("ethics_benchmark", {}).get("status") == "passed",
            "drift_detection": self.results["suites"].get("drift_detection", {}).get("status") == "passed",
            "performance": self.results["suites"].get("performance", {}).get("status") == "passed",
            "data_integrity": self.results["suites"].get("data_integrity", {}).get("status") == "passed",
            "explainability": self.results["suites"].get("explainability", {}).get("status") == "passed",
        }
        
        return {
            "checks": checks,
            "all_passed": all(checks.values()),
            "failed_checks": [k for k, v in checks.items() if not v]
        }
    
    def save_results(self, output_path: str = "validation_reports/validation.json") -> None:
        """Save validation results to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add threshold checks
        self.results["threshold_checks"] = self.check_thresholds()
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Validation results saved to: {output_file}")
        print(f"{'='*70}")
    
    def print_summary(self) -> None:
        """Print validation summary"""
        summary = self.results["summary"]
        threshold_checks = self.results.get("threshold_checks", {})
        
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Suites:       {summary['total_suites']}")
        print(f"Passed Suites:      {summary['passed_suites']}")
        print(f"Failed Suites:      {summary['failed_suites']}")
        print(f"Error Suites:       {summary['error_suites']}")
        print(f"Success Rate:       {summary['success_rate']:.1%}")
        print(f"Total Duration:     {summary['total_duration_seconds']:.1f}s")
        print(f"Overall Status:     {summary['overall_status'].upper()}")
        print(f"\nThreshold Checks:   {'PASSED' if threshold_checks.get('all_passed') else 'FAILED'}")
        
        if not threshold_checks.get('all_passed'):
            print(f"Failed Checks:      {', '.join(threshold_checks.get('failed_checks', []))}")
        
        print(f"{'='*70}\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Nethical Validation Suite")
    parser.add_argument(
        "--suites",
        nargs="+",
        help="Specific suites to run (default: all)",
        choices=["ethics_benchmark", "drift_detection", "performance", 
                 "data_integrity", "explainability"]
    )
    parser.add_argument(
        "--output",
        default="validation_reports/validation.json",
        help="Output file for validation results"
    )
    parser.add_argument(
        "--config",
        default="validation_config.yaml",
        help="Path to validation configuration file"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ValidationRunner(config_path=args.config)
    
    # Run suites
    runner.run_all_suites(suites=args.suites)
    
    # Generate summary
    runner.generate_summary()
    
    # Save results
    runner.save_results(output_path=args.output)
    
    # Print summary
    runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if runner.results["summary"]["overall_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
