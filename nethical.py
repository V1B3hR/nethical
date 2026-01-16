#!/usr/bin/env python3
"""
Nethical: AI Security Assessment & Reporting Tool
Author: V1B3hR

Description:
    Nethical is a safe, ethical, and strictly compliant entry point for generating advanced AI-powered security assessment reports from existing scan data.
    This tool DOES NOT perform any scanning, enumeration, or automated reconnaissance. It only analyzes scan results you supply, using AI to create structured, actionable reports.
    Intended for authorized use by professionals, researchers, and system owners. Misuse is strictly prohibited.

Usage:
    Run this script in an environment where you have permission to analyze scan data. You must have an OpenAI API key to generate AI reports.
"""

import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Optional
from colorama import Fore, Style, init

# --- AI DEPENDENCY ---
try:
    from openai import OpenAI
except ImportError:
    logging.error("Missing dependency 'openai'. Install via 'pip install openai'")

# Initialize colorama
init(autoreset=True)

# --- CONFIGURATION ---
class Config:
    def __init__(self):
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.max_tokens = 4000

config = Config()

# --- UTILITY FUNCTIONS ---

def print_colored(color, text):
    print(color + text + Style.RESET_ALL)

def print_banner():
    banner = f"""
{Fore.MAGENTA}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║                N E T H I C A L   ▸  AI REPORTER          ║
║         Security Assessment & Vulnerability Reporting     ║
║                       (No Scanning)                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)

def show_legal_disclaimer():
    print_colored(Fore.RED, """
╔════════════════════════════════════════════════════════════╗
║                    ⚠️  LEGAL WARNING ⚠️                    ║
╠════════════════════════════════════════════════════════════╣
║  Only analyze scan results you are authorized to process. ║
║  Unauthorized use is ILLEGAL and strictly prohibited.     ║
║                                                           ║
║  By continuing, you accept full legal responsibility.     ║
╚════════════════════════════════════════════════════════════╝
    """)
    confirm = input(Fore.YELLOW + "I have authorization to use and analyze these scans (yes/no): " + Style.RESET_ALL)
    if confirm.lower().strip() != 'yes':
        print_colored(Fore.RED, "[!] Authorization not confirmed. Exiting.")
        sys.exit(0)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and unsafe chars."""
    filename = os.path.basename(filename)
    return "".join(c if c.isalnum() or c in ['.', '-', '_'] else "_" for c in filename)

def read_file_safe(file_path: str) -> Optional[str]:
    """Safely read file with proper encoding handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='replace')
        except Exception as e:
            print_colored(Fore.YELLOW, f"[!] Could not read {file_path}: {e}")
            return None
    except FileNotFoundError:
        print_colored(Fore.YELLOW, f"[!] File not found: {file_path}")
        return None
    except Exception as e:
        print_colored(Fore.YELLOW, f"[!] Error reading {file_path}: {e}")
        return None

def log_report(target: str, files: list, status: str, results_dir: str):
    """Log report generation activity to history file"""
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'target': target,
        'type': 'ai_report',
        'scan_files': files,
        'status': status,
        'results_directory': results_dir
    }
    log_file = Path.home() / '.nethical_history.jsonl'
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print_colored(Fore.YELLOW, f"[!] Could not write to history: {e}")

def view_scan_history():
    """Display report history"""
    log_file = Path.home() / '.nethical_history.jsonl'
    if not log_file.exists():
        print_colored(Fore.YELLOW, "[!] No report history found.")
        return
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        if not lines:
            print_colored(Fore.YELLOW, "[!] No report history found.")
            return
        print_colored(Fore.CYAN, "\n" + "="*60)
        print_colored(Fore.CYAN, "REPORT HISTORY (Last 10 entries)")
        print_colored(Fore.CYAN, "="*60)
        for line in lines[-10:]:
            try:
                entry = json.loads(line)
                timestamp = entry.get('timestamp', 'N/A')
                target = entry.get('target', 'N/A')
                filelist = ', '.join(entry.get('scan_files', []))
                status = entry.get('status', 'N/A')
                status_color = Fore.GREEN if status == 'success' else Fore.RED
                print(f"{Fore.YELLOW}{timestamp}{Style.RESET_ALL} | {Fore.CYAN}{target}{Style.RESET_ALL} | {filelist} | {status_color}{status}{Style.RESET_ALL}")
            except json.JSONDecodeError:
                continue
        print_colored(Fore.CYAN, "="*60 + "\n")
    except Exception as e:
        print_colored(Fore.RED, f"[!] Error reading report history: {e}")

# --- AI REPORTING FUNCTION ---

def generate_ai_report(scan_data: str, api_key: str, target: str, scan_files: list) -> str:
    """Generate AI-powered security analysis report."""
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to initialize OpenAI client: {e}")

    prompt = f"""You are an expert penetration tester analyzing security scan results.

CONTEXT:
- Target: {target}
- Scan Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- Files Provided: {', '.join(scan_files)}

TASK: Generate a comprehensive CVSS-scored vulnerability report in Markdown format with the following structure:

## Executive Summary
Provide a 2-3 sentence high-level overview of the security posture and most critical findings.

## Critical Findings (CVSS 9.0-10.0)
List any critical vulnerabilities with:
- CVSS Score
- Vulnerability description
- Potential impact
- Immediate remediation steps

## High Priority Issues (CVSS 7.0-8.9)
List high-severity findings with same details as above.

## Medium Priority Issues (CVSS 4.0-6.9)
List medium-severity findings with same details as above.

## Informational Findings (CVSS 0.1-3.9)
List low-severity findings and informational items.

## Open Ports & Services Analysis
Detailed analysis of all discovered open ports, services, and versions. Highlight any outdated or vulnerable services.

## Web Application Findings
Summarize web server vulnerabilities, misconfigurations, and discovered directories/files.

## Attack Vectors
Describe realistic attack scenarios based on the findings.

## Prioritized Remediation Plan
Provide a numbered, prioritized list of specific, actionable remediation steps.

## Compliance Impact
Note any potential violations of PCI-DSS, HIPAA, GDPR, or other compliance frameworks (if applicable).

---
RAW SCAN DATA:
{scan_data}
---

Important: Base your analysis ONLY on the actual data provided. Do not make assumptions about vulnerabilities not present in the scan results. Assign accurate CVSS scores based on industry standards.
"""
    try:
        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": "You are an expert cybersecurity analyst specializing in penetration testing and vulnerability assessment. Generate accurate, actionable reports based solely on supplied scan data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=config.max_tokens
        )
        report = response.choices[0].message.content
        return report.strip()
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")

def export_to_html(markdown_file: str):
    """Convert markdown report to HTML (optional enhancement)"""
    try:
        import markdown
        with open(markdown_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Nethical Security Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .critical {{ color: #c0392b; font-weight: bold; }}
        .high {{ color: #e67e22; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
    </style>
</head>
<body>
{markdown.markdown(md_content, extensions=['extra', 'codehilite'])}
</body>
</html>
"""
        html_file = markdown_file.replace('.md', '.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print_colored(Fore.GREEN, f"[+] HTML report also saved to {html_file}")
    except ImportError:
        print_colored(Fore.YELLOW, "[!] 'markdown' package not installed. Skipping HTML export.")
    except Exception as e:
        print_colored(Fore.YELLOW, f"[!] Could not export to HTML: {e}")

# --- MENU FUNCTIONS ---

def main_menu():
    print_colored(Fore.MAGENTA, "\n" + "="*60)
    print_colored(Fore.YELLOW, "Select an action:")
    print("  1. Generate AI Security Report (from scan files)")
    print("  2. View Report History")
    print_colored(Fore.RED, "  0. Exit")
    print_colored(Fore.MAGENTA, "="*60)

# --- MAIN EXECUTION ---

def main():
    print_banner()
    show_legal_disclaimer()
    try:
        while True:
            main_menu()
            choice = input(Fore.CYAN + ">> Enter your choice: " + Style.RESET_ALL).strip()

            if choice == '1':
                print_colored(Fore.CYAN, "\n[*] AI Report Generator: Provide your scan result files (e.g., nmap, nikto, etc.)")
                scan_files = []
                while True:
                    scan_input = input(Fore.YELLOW + "[?] Enter scan result filename (or press Enter to finish): " + Style.RESET_ALL).strip()
                    if not scan_input:
                        break
                    clean_name = sanitize_filename(scan_input)
                    if not os.path.exists(clean_name):
                        print_colored(Fore.RED, f"[!] {clean_name} not found in current directory.")
                        continue
                    scan_files.append(clean_name)
                if not scan_files:
                    print_colored(Fore.RED, "[!] No scan files provided.")
                    continue

                target = input(Fore.YELLOW + "[?] Enter the target (domain or IP these scans refer to): " + Style.RESET_ALL).strip()
                if not target:
                    print_colored(Fore.RED, "[!] Target cannot be empty. Exiting.")
                    continue

                # Aggregate scan file contents
                full_scan_data = ""
                for file in scan_files:
                    content = read_file_safe(file)
                    if content:
                        full_scan_data += f"\n{'='*60}\n{file.upper()}\n{'='*60}\n{content}\n"
                    else:
                        print_colored(Fore.YELLOW, f"[!] Warning: {file} could not be read or is empty.")

                if not full_scan_data:
                    print_colored(Fore.RED, "[!] None of the scan files had readable content. Aborting report generation.")
                    continue

                # Get API key now if not set
                api_key = config.openai_api_key
                if not api_key:
                    api_key = input(Fore.YELLOW + "[?] Enter your OpenAI API key (or set OPENAI_API_KEY env variable): " + Style.RESET_ALL).strip()
                if not api_key:
                    print_colored(Fore.RED, "[!] API key is required to generate AI report.")
                    continue

                # Results directory (per report generation)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                safe_target = sanitize_filename(target)
                dir_name = f"{safe_target}_{timestamp}"
                try:
                    os.makedirs(dir_name, exist_ok=True)
                    os.chdir(dir_name)
                    print_colored(Fore.GREEN, f"[+] Report and artifacts will be saved in: {os.getcwd()}\n")
                except Exception as e:
                    print_colored(Fore.RED, f"[!] Could not create results directory: {e}")
                    continue

                # Generate report
                try:
                    print_colored(Fore.CYAN, f"[*] Contacting OpenAI (using model: {config.openai_model})...")
                    report_content = generate_ai_report(full_scan_data, api_key, target, scan_files)
                    report_filename = f'AI_Security_Report_{safe_target}_{timestamp}.md'
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        f.write(f"# Security Assessment Report\n\n")
                        f.write(f"**Target:** {target}\n\n")
                        f.write(f"**Scan Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
                        f.write(f"**Generated by:** Nethical AI Reporter\n\n")
                        f.write("---\n\n")
                        f.write(report_content)
                    print_colored(Fore.GREEN, f"\n[+] AI Security Report saved to: {report_filename}")
                    export_html = input(Fore.YELLOW + "[?] Export report to HTML as well? (y/n): " + Style.RESET_ALL).strip().lower()
                    if export_html == 'y':
                        export_to_html(report_filename)
                    log_report(target, scan_files, "success", os.getcwd())
                except Exception as e:
                    print_colored(Fore.RED, f"[!] AI report generation failed: {e}")
                    log_report(target, scan_files, "failed", os.getcwd())
                finally:
                    # Return to parent directory
                    try:
                        os.chdir("..")
                    except:
                        pass

            elif choice == '2':
                view_scan_history()

            elif choice == '0':
                print_colored(Fore.YELLOW, "\n[*] Exiting Nethical. Stay ethical, stay safe!")
                break

            else:
                print_colored(Fore.RED, "[!] Invalid choice. Please try again.")

    except KeyboardInterrupt:
        print_colored(Fore.YELLOW, "\n\n[*] User interrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print_colored(Fore.RED, f"\n[!] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
