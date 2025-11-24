#!/usr/bin/env python3
"""
Vulnerability SLA Enforcement Script
Checks if vulnerabilities exceed SLA timelines:
- Critical: < 24 hours
- High: < 72 hours
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# SLA thresholds in hours
SLA_CRITICAL = 24
SLA_HIGH = 72

# ANSI colors
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
GREEN = '\033[0;32m'
BOLD = '\033[1m'
NC = '\033[0m'  # No Color


def parse_timestamp(ts_str: str) -> datetime:
    """Parse various timestamp formats to datetime."""
    formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    # If all formats fail, try ISO format
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except Exception as e:
        print(f"{RED}Error parsing timestamp '{ts_str}': {e}{NC}")
        return datetime.now(timezone.utc)


def check_trivy_format(data: dict) -> Tuple[List[Dict], str]:
    """Parse Trivy scan format."""
    vulnerabilities = []
    
    if 'Results' in data:
        for result in data.get('Results', []):
            for vuln in result.get('Vulnerabilities', []):
                vulnerabilities.append({
                    'id': vuln.get('VulnerabilityID', 'UNKNOWN'),
                    'severity': vuln.get('Severity', 'UNKNOWN'),
                    'published': vuln.get('PublishedDate', ''),
                    'title': vuln.get('Title', ''),
                    'package': vuln.get('PkgName', ''),
                    'installed_version': vuln.get('InstalledVersion', ''),
                    'fixed_version': vuln.get('FixedVersion', 'N/A'),
                    'description': vuln.get('Description', '')[:100],
                })
    
    return vulnerabilities, 'Trivy'


def check_npm_audit_format(data: dict) -> Tuple[List[Dict], str]:
    """Parse npm audit format."""
    vulnerabilities = []
    
    if 'vulnerabilities' in data:
        for pkg, vuln_data in data['vulnerabilities'].items():
            for vuln in vuln_data.get('via', []):
                if isinstance(vuln, dict):
                    vulnerabilities.append({
                        'id': vuln.get('source', 'UNKNOWN'),
                        'severity': vuln.get('severity', 'unknown').upper(),
                        'published': vuln.get('created', ''),
                        'title': vuln.get('title', ''),
                        'package': pkg,
                        'installed_version': vuln_data.get('version', ''),
                        'fixed_version': vuln_data.get('fixAvailable', {}).get('version', 'N/A'),
                        'description': vuln.get('overview', '')[:100],
                    })
    
    return vulnerabilities, 'NPM Audit'


def check_generic_format(data: dict) -> Tuple[List[Dict], str]:
    """Parse generic vulnerability format."""
    vulnerabilities = []
    
    if isinstance(data, list):
        for vuln in data:
            vulnerabilities.append({
                'id': vuln.get('id', 'UNKNOWN'),
                'severity': vuln.get('severity', 'UNKNOWN').upper(),
                'published': vuln.get('published', vuln.get('publishedDate', '')),
                'title': vuln.get('title', vuln.get('name', '')),
                'package': vuln.get('package', vuln.get('component', '')),
                'installed_version': vuln.get('version', ''),
                'fixed_version': vuln.get('fixed_version', 'N/A'),
                'description': vuln.get('description', '')[:100],
            })
    
    return vulnerabilities, 'Generic'


def parse_vulnerability_file(filepath: str) -> Tuple[List[Dict], str]:
    """Parse vulnerability scan results."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"{RED}Error: File '{filepath}' not found{NC}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{RED}Error: Invalid JSON in '{filepath}': {e}{NC}")
        sys.exit(1)
    
    # Try different formats, prioritizing by confidence
    # First check for Trivy format (most specific)
    if 'Results' in data and any('Vulnerabilities' in r for r in data.get('Results', [])):
        return check_trivy_format(data)
    
    # Check for npm audit format
    if 'vulnerabilities' in data and isinstance(data['vulnerabilities'], dict):
        return check_npm_audit_format(data)
    
    # Fall back to generic format
    if isinstance(data, list):
        return check_generic_format(data)
    
    return [], 'Unknown'


def check_sla(vulnerabilities: List[Dict]) -> Tuple[int, int, int]:
    """Check SLA compliance for vulnerabilities."""
    now = datetime.now(timezone.utc)
    
    breaches = []
    critical_breaches = 0
    high_breaches = 0
    
    print(f"\n{BOLD}Vulnerability SLA Analysis{NC}")
    print("=" * 80)
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"SLA thresholds: Critical < {SLA_CRITICAL}h, High < {SLA_HIGH}h\n")
    
    # Count by severity
    severity_count = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNKNOWN': 0}
    
    for vuln in vulnerabilities:
        severity = vuln['severity']
        severity_count[severity] = severity_count.get(severity, 0) + 1
        
        # Skip non-critical/high for SLA check
        if severity not in ['CRITICAL', 'HIGH']:
            continue
        
        published_str = vuln['published']
        if not published_str:
            print(f"{YELLOW}Warning: No publish date for {vuln['id']}, skipping SLA check{NC}")
            continue
        
        published = parse_timestamp(published_str)
        age = now - published
        age_hours = age.total_seconds() / 3600
        age_days = age.days
        
        # Check SLA breach
        is_breach = False
        if severity == 'CRITICAL' and age_hours > SLA_CRITICAL:
            is_breach = True
            critical_breaches += 1
        elif severity == 'HIGH' and age_hours > SLA_HIGH:
            is_breach = True
            high_breaches += 1
        
        if is_breach:
            breaches.append({
                'id': vuln['id'],
                'severity': severity,
                'age_hours': age_hours,
                'age_days': age_days,
                'package': vuln['package'],
                'fixed_version': vuln['fixed_version'],
                'title': vuln['title'],
            })
    
    # Print summary
    print(f"{BOLD}Vulnerability Count by Severity:{NC}")
    for severity, count in sorted(severity_count.items()):
        if count > 0:
            if severity == 'CRITICAL':
                print(f"  {RED}{severity}{NC}: {count}")
            elif severity == 'HIGH':
                print(f"  {YELLOW}{severity}{NC}: {count}")
            else:
                print(f"  {severity}: {count}")
    
    print(f"\n{BOLD}SLA Breach Analysis:{NC}")
    
    if not breaches:
        print(f"{GREEN}✓ No SLA breaches detected!{NC}")
        return 0, critical_breaches, high_breaches
    
    # Print breaches
    print(f"\n{RED}✗ Found {len(breaches)} SLA breach(es):{NC}\n")
    
    for breach in sorted(breaches, key=lambda x: x['age_hours'], reverse=True):
        print(f"{RED}{'=' * 80}{NC}")
        print(f"ID:       {breach['id']}")
        print(f"Severity: {RED if breach['severity'] == 'CRITICAL' else YELLOW}{breach['severity']}{NC}")
        print(f"Age:      {breach['age_days']} days ({breach['age_hours']:.1f} hours)")
        print(f"Package:  {breach['package']}")
        print(f"Fix:      {breach['fixed_version']}")
        print(f"Title:    {breach['title'][:70]}")
        
        # Calculate how much over SLA
        sla_limit = SLA_CRITICAL if breach['severity'] == 'CRITICAL' else SLA_HIGH
        overage = breach['age_hours'] - sla_limit
        print(f"Overage:  {RED}{overage:.1f} hours over SLA{NC}")
        print()
    
    return len(breaches), critical_breaches, high_breaches


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <vulnerability-scan-file.json>")
        print(f"\nChecks if vulnerabilities exceed SLA timelines:")
        print(f"  Critical: < {SLA_CRITICAL} hours")
        print(f"  High:     < {SLA_HIGH} hours")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print(f"{BOLD}Vulnerability SLA Checker{NC}")
    print(f"Scanning file: {filepath}\n")
    
    # Parse vulnerabilities
    vulnerabilities, format_name = parse_vulnerability_file(filepath)
    
    if not vulnerabilities:
        print(f"{YELLOW}Warning: No vulnerabilities found in the scan results{NC}")
        print(f"{GREEN}✓ SLA compliance check passed (no vulnerabilities){NC}")
        sys.exit(0)
    
    print(f"Detected format: {format_name}")
    print(f"Total vulnerabilities: {len(vulnerabilities)}")
    
    # Check SLA compliance
    total_breaches, critical_breaches, high_breaches = check_sla(vulnerabilities)
    
    # Print final summary
    print(f"\n{BOLD}{'=' * 80}{NC}")
    print(f"{BOLD}Final Summary:{NC}")
    print(f"  Total SLA breaches: {total_breaches}")
    print(f"  Critical breaches:  {critical_breaches}")
    print(f"  High breaches:      {high_breaches}")
    
    if total_breaches == 0:
        print(f"\n{GREEN}✓ SLA compliance check PASSED{NC}")
        sys.exit(0)
    else:
        print(f"\n{RED}✗ SLA compliance check FAILED{NC}")
        print(f"\n{BOLD}Action Required:{NC}")
        if critical_breaches > 0:
            print(f"  {RED}• {critical_breaches} CRITICAL vulnerabilities exceed 24h SLA{NC}")
        if high_breaches > 0:
            print(f"  {YELLOW}• {high_breaches} HIGH vulnerabilities exceed 72h SLA{NC}")
        print(f"\nPlease update dependencies or apply patches immediately.")
        sys.exit(1)


if __name__ == '__main__':
    main()
