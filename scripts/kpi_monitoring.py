#!/usr/bin/env python3
"""
KPI Monitoring & Automation Script
Nethical Platform - Phase 10A: Maintenance & Sustainability

This script automates the collection, analysis, and reporting of key performance
indicators (KPIs) for the Nethical platform, supporting continuous assurance and
long-term sustainability.

Usage:
    python scripts/kpi_monitoring.py --mode [collect|analyze|report|alert]
    
Examples:
    # Collect current KPI values
    python scripts/kpi_monitoring.py --mode collect
    
    # Analyze KPI trends
    python scripts/kpi_monitoring.py --mode analyze --period 30
    
    # Generate KPI report
    python scripts/kpi_monitoring.py --mode report --format json
    
    # Check for KPI threshold breaches and send alerts
    python scripts/kpi_monitoring.py --mode alert
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class KPIMonitor:
    """Automated KPI monitoring and reporting system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize KPI monitor with configuration."""
        self.config_path = config_path or str(project_root / "config" / "kpi_config.json")
        self.data_dir = project_root / "data" / "kpi"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.kpis = self._load_kpi_definitions()
        
    def _load_kpi_definitions(self) -> Dict:
        """Load KPI definitions from configuration."""
        # Default KPI definitions from maintenance_policy.md
        default_kpis = {
            "proof_coverage": {
                "name": "Proof Coverage",
                "description": "Percentage of critical properties with formal proofs",
                "target": 85.0,
                "threshold_warning": 88.0,
                "threshold_critical": 85.0,
                "unit": "percent",
                "frequency": "weekly",
                "category": "formal_assurance"
            },
            "admitted_critical_lemmas": {
                "name": "Admitted Critical Lemmas",
                "description": "Count of critical lemmas without complete proofs",
                "target": 0,
                "threshold_warning": 2,
                "threshold_critical": 0,
                "unit": "count",
                "frequency": "weekly",
                "category": "formal_assurance"
            },
            "determinism_violations": {
                "name": "Determinism Violations",
                "description": "Count of P-DET invariant violations",
                "target": 0,
                "threshold_warning": 1,
                "threshold_critical": 0,
                "unit": "count",
                "frequency": "continuous",
                "category": "correctness"
            },
            "fairness_sp_diff": {
                "name": "Fairness Statistical Parity Difference",
                "description": "Maximum SP difference across protected attributes",
                "target": 0.10,
                "threshold_warning": 0.12,
                "threshold_critical": 0.10,
                "unit": "ratio",
                "frequency": "monthly",
                "category": "fairness"
            },
            "appeal_resolution_time": {
                "name": "Appeal Resolution Median Time",
                "description": "Median time to resolve appeals (hours)",
                "target": 72.0,
                "threshold_warning": 96.0,
                "threshold_critical": 72.0,
                "unit": "hours",
                "frequency": "monthly",
                "category": "governance"
            },
            "system_uptime": {
                "name": "System Uptime",
                "description": "System availability (90-day rolling)",
                "target": 99.95,
                "threshold_warning": 99.90,
                "threshold_critical": 99.95,
                "unit": "percent",
                "frequency": "continuous",
                "category": "operational"
            },
            "mttr_critical": {
                "name": "Mean Time to Resolution (P0)",
                "description": "Average time to resolve critical incidents",
                "target": 30.0,
                "threshold_warning": 45.0,
                "threshold_critical": 30.0,
                "unit": "minutes",
                "frequency": "continuous",
                "category": "operational"
            },
            "security_vulnerabilities_critical": {
                "name": "Critical Security Vulnerabilities",
                "description": "Count of critical vulnerabilities (CVSS ≥9.0)",
                "target": 0,
                "threshold_warning": 1,
                "threshold_critical": 0,
                "unit": "count",
                "frequency": "daily",
                "category": "security"
            },
            "security_vulnerabilities_high": {
                "name": "High Security Vulnerabilities",
                "description": "Count of high severity vulnerabilities (CVSS 7.0-8.9)",
                "target": 0,
                "threshold_warning": 5,
                "threshold_critical": 0,
                "unit": "count",
                "frequency": "daily",
                "category": "security"
            },
            "test_pass_rate": {
                "name": "Test Pass Rate",
                "description": "Percentage of automated tests passing",
                "target": 99.0,
                "threshold_warning": 95.0,
                "threshold_critical": 99.0,
                "unit": "percent",
                "frequency": "continuous",
                "category": "quality"
            },
            "lineage_chain_verification": {
                "name": "Lineage Chain Verification",
                "description": "Percentage of policy lineage verifications succeeding",
                "target": 100.0,
                "threshold_warning": 99.5,
                "threshold_critical": 100.0,
                "unit": "percent",
                "frequency": "daily",
                "category": "governance"
            },
            "sbom_generation_success": {
                "name": "SBOM Generation Success",
                "description": "Percentage of releases with successful SBOM generation",
                "target": 100.0,
                "threshold_warning": 100.0,
                "threshold_critical": 100.0,
                "unit": "percent",
                "frequency": "per_release",
                "category": "supply_chain"
            },
            "technical_debt_critical": {
                "name": "Critical Technical Debt Items",
                "description": "Count of critical severity technical debt items",
                "target": 0,
                "threshold_warning": 3,
                "threshold_critical": 0,
                "unit": "count",
                "frequency": "weekly",
                "category": "maintenance"
            },
            "performance_p95_latency": {
                "name": "Decision Evaluation P95 Latency",
                "description": "95th percentile latency for decision evaluation (ms)",
                "target": 200.0,
                "threshold_warning": 250.0,
                "threshold_critical": 200.0,
                "unit": "milliseconds",
                "frequency": "continuous",
                "category": "performance"
            },
            "audit_log_integrity": {
                "name": "Audit Log Integrity",
                "description": "Percentage of audit log Merkle root verifications succeeding",
                "target": 100.0,
                "threshold_warning": 99.9,
                "threshold_critical": 100.0,
                "unit": "percent",
                "frequency": "daily",
                "category": "governance"
            }
        }
        
        # Try to load from config file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_kpis = json.load(f)
                    default_kpis.update(loaded_kpis)
            except Exception as e:
                print(f"Warning: Could not load KPI config from {self.config_path}: {e}")
                print("Using default KPI definitions")
        
        return default_kpis
    
    def collect_kpis(self) -> Dict:
        """Collect current KPI values from various sources."""
        timestamp = datetime.now().isoformat()
        
        kpi_values = {
            "timestamp": timestamp,
            "values": {}
        }
        
        # Collect each KPI
        # Note: In production, these would query actual data sources
        # For now, we use placeholder collection methods
        
        for kpi_id, kpi_def in self.kpis.items():
            try:
                value = self._collect_single_kpi(kpi_id, kpi_def)
                kpi_values["values"][kpi_id] = {
                    "value": value,
                    "target": kpi_def["target"],
                    "unit": kpi_def["unit"],
                    "status": self._evaluate_kpi_status(value, kpi_def)
                }
            except Exception as e:
                print(f"Error collecting KPI '{kpi_id}': {e}")
                kpi_values["values"][kpi_id] = {
                    "value": None,
                    "error": str(e),
                    "status": "unknown"
                }
        
        # Save collected data
        self._save_kpi_data(kpi_values)
        
        return kpi_values
    
    def _collect_single_kpi(self, kpi_id: str, kpi_def: Dict) -> float:
        """Collect a single KPI value."""
        # Placeholder implementations - in production, these would query actual systems
        
        if kpi_id == "proof_coverage":
            # Query formal verification coverage dashboard
            # For now, return a placeholder value
            return self._get_proof_coverage()
        
        elif kpi_id == "admitted_critical_lemmas":
            # Parse formal/debt_log.json
            return self._get_admitted_lemmas()
        
        elif kpi_id == "determinism_violations":
            # Query runtime probes
            return self._get_invariant_violations("P-DET")
        
        elif kpi_id == "fairness_sp_diff":
            # Query fairness dashboard
            return self._get_fairness_metric("sp_diff")
        
        elif kpi_id == "appeal_resolution_time":
            # Query appeals database
            return self._get_appeal_metrics()
        
        elif kpi_id == "system_uptime":
            # Query monitoring system (Prometheus/Grafana)
            return self._get_system_uptime()
        
        elif kpi_id == "mttr_critical":
            # Query incident tracking system
            return self._get_mttr("P0")
        
        elif kpi_id == "security_vulnerabilities_critical":
            # Query vulnerability scanner results
            return self._get_vulnerability_count("critical")
        
        elif kpi_id == "security_vulnerabilities_high":
            return self._get_vulnerability_count("high")
        
        elif kpi_id == "test_pass_rate":
            # Query CI/CD system
            return self._get_test_pass_rate()
        
        elif kpi_id == "lineage_chain_verification":
            return self._get_lineage_verification_rate()
        
        elif kpi_id == "sbom_generation_success":
            return self._get_sbom_success_rate()
        
        elif kpi_id == "technical_debt_critical":
            return self._get_technical_debt_count("critical")
        
        elif kpi_id == "performance_p95_latency":
            return self._get_performance_metric("decision_eval_p95")
        
        elif kpi_id == "audit_log_integrity":
            return self._get_audit_log_integrity()
        
        else:
            raise ValueError(f"Unknown KPI: {kpi_id}")
    
    # Placeholder data collection methods
    # In production, these would integrate with actual systems
    
    def _get_proof_coverage(self) -> float:
        """Get current proof coverage percentage."""
        # TODO: Integrate with formal verification coverage dashboard
        return 87.5  # Placeholder
    
    def _get_admitted_lemmas(self) -> int:
        """Get count of admitted critical lemmas."""
        debt_log_path = project_root / "formal" / "debt_log.json"
        if debt_log_path.exists():
            try:
                with open(debt_log_path, 'r') as f:
                    debt_log = json.load(f)
                    return len([d for d in debt_log.get("items", []) 
                               if d.get("severity") == "critical"])
            except Exception:
                pass
        return 0  # Placeholder
    
    def _get_invariant_violations(self, property_id: str) -> int:
        """Get count of invariant violations."""
        # TODO: Query runtime probes
        return 0  # Placeholder
    
    def _get_fairness_metric(self, metric: str) -> float:
        """Get fairness metric value."""
        # TODO: Query fairness dashboard API
        return 0.08  # Placeholder
    
    def _get_appeal_metrics(self) -> float:
        """Get median appeal resolution time in hours."""
        # TODO: Query appeals database
        return 48.0  # Placeholder
    
    def _get_system_uptime(self) -> float:
        """Get system uptime percentage (90-day rolling)."""
        # TODO: Query monitoring system
        return 99.97  # Placeholder
    
    def _get_mttr(self, severity: str) -> float:
        """Get mean time to resolution in minutes."""
        # TODO: Query incident tracking system
        return 25.0  # Placeholder
    
    def _get_vulnerability_count(self, severity: str) -> int:
        """Get count of vulnerabilities by severity."""
        # TODO: Query vulnerability scanner results
        return 0  # Placeholder
    
    def _get_test_pass_rate(self) -> float:
        """Get test pass rate percentage."""
        # TODO: Query CI/CD system
        return 99.5  # Placeholder
    
    def _get_lineage_verification_rate(self) -> float:
        """Get lineage chain verification success rate."""
        # TODO: Query lineage verification logs
        return 100.0  # Placeholder
    
    def _get_sbom_success_rate(self) -> float:
        """Get SBOM generation success rate."""
        # TODO: Query release pipeline logs
        return 100.0  # Placeholder
    
    def _get_technical_debt_count(self, severity: str) -> int:
        """Get technical debt item count by severity."""
        debt_register_path = project_root / "technical_debt_register.json"
        if debt_register_path.exists():
            try:
                with open(debt_register_path, 'r') as f:
                    debt_register = json.load(f)
                    return len([d for d in debt_register.get("items", [])
                               if d.get("severity") == severity])
            except Exception:
                pass
        return 0  # Placeholder
    
    def _get_performance_metric(self, metric: str) -> float:
        """Get performance metric value."""
        # TODO: Query monitoring system
        return 180.0  # Placeholder
    
    def _get_audit_log_integrity(self) -> float:
        """Get audit log integrity verification success rate."""
        # TODO: Query audit log verification results
        return 100.0  # Placeholder
    
    def _evaluate_kpi_status(self, value: float, kpi_def: Dict) -> str:
        """Evaluate KPI status based on thresholds."""
        if value is None:
            return "unknown"
        
        target = kpi_def["target"]
        warning = kpi_def["threshold_warning"]
        critical = kpi_def["threshold_critical"]
        
        # Determine if higher or lower is better based on target/thresholds
        if target <= critical:  # Lower is better (e.g., latency, violations)
            if value <= target:
                return "green"
            elif value <= warning:
                return "yellow"
            else:
                return "red"
        else:  # Higher is better (e.g., uptime, coverage)
            if value >= target:
                return "green"
            elif value >= warning:
                return "yellow"
            else:
                return "red"
    
    def _save_kpi_data(self, kpi_values: Dict) -> None:
        """Save KPI data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"kpi_data_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(kpi_values, f, indent=2)
        
        # Also save to latest.json for easy access
        latest_file = self.data_dir / "kpi_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(kpi_values, f, indent=2)
    
    def analyze_kpis(self, period_days: int = 30) -> Dict:
        """Analyze KPI trends over specified period."""
        # Load historical KPI data
        historical_data = self._load_historical_kpis(period_days)
        
        analysis = {
            "period_start": (datetime.now() - timedelta(days=period_days)).isoformat(),
            "period_end": datetime.now().isoformat(),
            "period_days": period_days,
            "trends": {}
        }
        
        for kpi_id in self.kpis.keys():
            kpi_history = [d["values"][kpi_id]["value"] 
                          for d in historical_data 
                          if kpi_id in d.get("values", {}) 
                          and d["values"][kpi_id].get("value") is not None]
            
            if len(kpi_history) >= 2:
                trend = self._calculate_trend(kpi_history)
                analysis["trends"][kpi_id] = {
                    "direction": trend["direction"],
                    "magnitude": trend["magnitude"],
                    "data_points": len(kpi_history),
                    "min": min(kpi_history),
                    "max": max(kpi_history),
                    "avg": sum(kpi_history) / len(kpi_history),
                    "current": kpi_history[-1]
                }
            else:
                analysis["trends"][kpi_id] = {
                    "direction": "insufficient_data",
                    "data_points": len(kpi_history)
                }
        
        return analysis
    
    def _load_historical_kpis(self, period_days: int) -> List[Dict]:
        """Load historical KPI data."""
        cutoff_date = datetime.now() - timedelta(days=period_days)
        historical_data = []
        
        for filename in sorted(self.data_dir.glob("kpi_data_*.json")):
            try:
                # Extract timestamp from filename
                timestamp_str = filename.stem.split("_", 2)[2]
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if file_date >= cutoff_date:
                    with open(filename, 'r') as f:
                        historical_data.append(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
        
        return historical_data
    
    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend direction and magnitude."""
        if len(values) < 2:
            return {"direction": "insufficient_data", "magnitude": 0}
        
        # Simple linear trend: compare first half to second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid
        second_half_avg = sum(values[mid:]) / (len(values) - mid)
        
        diff = second_half_avg - first_half_avg
        pct_change = (diff / first_half_avg * 100) if first_half_avg != 0 else 0
        
        if abs(pct_change) < 1.0:
            direction = "stable"
        elif pct_change > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            "direction": direction,
            "magnitude": abs(pct_change)
        }
    
    def generate_report(self, format: str = "json") -> str:
        """Generate KPI report in specified format."""
        # Collect current KPIs
        current_kpis = self.collect_kpis()
        
        # Analyze trends
        trends = self.analyze_kpis(period_days=30)
        
        report = {
            "report_date": datetime.now().isoformat(),
            "current_kpis": current_kpis,
            "trends_30_days": trends,
            "summary": self._generate_summary(current_kpis, trends)
        }
        
        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "text":
            return self._format_text_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_summary(self, current_kpis: Dict, trends: Dict) -> Dict:
        """Generate summary statistics."""
        values = current_kpis["values"]
        
        green_count = sum(1 for v in values.values() if v.get("status") == "green")
        yellow_count = sum(1 for v in values.values() if v.get("status") == "yellow")
        red_count = sum(1 for v in values.values() if v.get("status") == "red")
        
        return {
            "total_kpis": len(values),
            "green": green_count,
            "yellow": yellow_count,
            "red": red_count,
            "overall_health": "good" if red_count == 0 else "needs_attention",
            "critical_alerts": [kpi_id for kpi_id, v in values.items() 
                               if v.get("status") == "red"]
        }
    
    def _format_text_report(self, report: Dict) -> str:
        """Format report as human-readable text."""
        lines = []
        lines.append("=" * 80)
        lines.append("KPI MONITORING REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {report['report_date']}")
        lines.append("")
        
        summary = report['summary']
        lines.append(f"Overall Health: {summary['overall_health'].upper()}")
        lines.append(f"Total KPIs: {summary['total_kpis']}")
        lines.append(f"  Green (On Target): {summary['green']}")
        lines.append(f"  Yellow (Warning): {summary['yellow']}")
        lines.append(f"  Red (Critical): {summary['red']}")
        lines.append("")
        
        if summary['critical_alerts']:
            lines.append("CRITICAL ALERTS:")
            for kpi_id in summary['critical_alerts']:
                kpi_value = report['current_kpis']['values'][kpi_id]
                lines.append(f"  - {self.kpis[kpi_id]['name']}: "
                           f"{kpi_value['value']} {kpi_value['unit']} "
                           f"(Target: {kpi_value['target']} {kpi_value['unit']})")
            lines.append("")
        
        lines.append("CURRENT KPI VALUES:")
        lines.append("-" * 80)
        
        for kpi_id, kpi_value in report['current_kpis']['values'].items():
            status_symbol = {"green": "✓", "yellow": "⚠", "red": "✗"}.get(
                kpi_value.get("status"), "?")
            lines.append(f"{status_symbol} {self.kpis[kpi_id]['name']}: "
                        f"{kpi_value['value']} {kpi_value['unit']} "
                        f"(Target: {kpi_value['target']})")
        
        lines.append("")
        lines.append("30-DAY TRENDS:")
        lines.append("-" * 80)
        
        for kpi_id, trend in report['trends_30_days']['trends'].items():
            if trend.get('direction') != 'insufficient_data':
                trend_symbol = {"increasing": "↑", "decreasing": "↓", 
                               "stable": "→"}.get(trend['direction'], "?")
                lines.append(f"{trend_symbol} {self.kpis[kpi_id]['name']}: "
                           f"{trend['direction']} "
                           f"({trend['magnitude']:.1f}% change)")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def check_alerts(self) -> List[Dict]:
        """Check for KPI threshold breaches and generate alerts."""
        current_kpis = self.collect_kpis()
        alerts = []
        
        for kpi_id, kpi_value in current_kpis["values"].items():
            if kpi_value.get("status") == "red":
                alerts.append({
                    "severity": "critical",
                    "kpi_id": kpi_id,
                    "kpi_name": self.kpis[kpi_id]["name"],
                    "value": kpi_value["value"],
                    "target": kpi_value["target"],
                    "unit": kpi_value["unit"],
                    "message": f"{self.kpis[kpi_id]['name']} is {kpi_value['value']} "
                              f"{kpi_value['unit']}, exceeding critical threshold of "
                              f"{self.kpis[kpi_id]['threshold_critical']} "
                              f"{kpi_value['unit']}"
                })
            elif kpi_value.get("status") == "yellow":
                alerts.append({
                    "severity": "warning",
                    "kpi_id": kpi_id,
                    "kpi_name": self.kpis[kpi_id]["name"],
                    "value": kpi_value["value"],
                    "target": kpi_value["target"],
                    "unit": kpi_value["unit"],
                    "message": f"{self.kpis[kpi_id]['name']} is {kpi_value['value']} "
                              f"{kpi_value['unit']}, exceeding warning threshold of "
                              f"{self.kpis[kpi_id]['threshold_warning']} "
                              f"{kpi_value['unit']}"
                })
        
        if alerts:
            self._send_alerts(alerts)
        
        return alerts
    
    def _send_alerts(self, alerts: List[Dict]) -> None:
        """Send alerts via configured channels."""
        # TODO: Integrate with actual alerting systems (email, Slack, PagerDuty)
        print(f"\n{'='*80}")
        print(f"ALERTS GENERATED: {len(alerts)}")
        print(f"{'='*80}")
        for alert in alerts:
            print(f"[{alert['severity'].upper()}] {alert['message']}")
        print(f"{'='*80}\n")


def main():
    """Main entry point for KPI monitoring script."""
    parser = argparse.ArgumentParser(
        description="KPI Monitoring & Automation for Nethical Platform"
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "analyze", "report", "alert"],
        required=True,
        help="Operation mode"
    )
    parser.add_argument(
        "--period",
        type=int,
        default=30,
        help="Analysis period in days (for analyze mode)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Report format (for report mode)"
    )
    parser.add_argument(
        "--config",
        help="Path to KPI configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize KPI monitor
    monitor = KPIMonitor(config_path=args.config)
    
    try:
        if args.mode == "collect":
            print("Collecting KPI values...")
            kpis = monitor.collect_kpis()
            print(json.dumps(kpis, indent=2))
            print(f"\nKPI data saved to {monitor.data_dir}")
        
        elif args.mode == "analyze":
            print(f"Analyzing KPI trends ({args.period} days)...")
            analysis = monitor.analyze_kpis(period_days=args.period)
            print(json.dumps(analysis, indent=2))
        
        elif args.mode == "report":
            print("Generating KPI report...")
            report = monitor.generate_report(format=args.format)
            print(report)
        
        elif args.mode == "alert":
            print("Checking for KPI alerts...")
            alerts = monitor.check_alerts()
            if alerts:
                print(f"\n{len(alerts)} alert(s) generated")
            else:
                print("\nNo alerts - all KPIs within thresholds")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
