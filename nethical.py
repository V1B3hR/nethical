# nethical.py
# Author: V1B3hR
# Description: A Python-based tool for automating reconnaissance tasks with an integrated AI reporting feature.
# Version: 2.0 - Enhanced Security & Features

import os
import sys
import subprocess
import datetime
import json
import re
import ipaddress
import time
from pathlib import Path
from colorama import Fore, Style, init
from typing import Optional, List, Dict

# --- AI DEPENDENCY ---
from openai import OpenAI

# Initialize colorama
init(autoreset=True)

# --- CONFIGURATION ---
class Config:
    """Configuration management for Nethical"""
    def __init__(self):
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.scan_delay = 1.0
        self.max_tokens = 4000
        self.default_wordlist = '/usr/share/wordlists/dirb/common.txt'
        
config = Config()

# --- UTILITY FUNCTIONS ---

def print_colored(color, text):
    """Prints text in a specified color."""
    print(color + text + Style.RESET_ALL)

def print_banner():
    """Displays the Nethical banner"""
    banner = f"""
{Fore.MAGENTA}╔══════════════════════════════════════════════════════════╗
║                                                          ║
║                      N E T H I C A L                     ║
║              Security Reconnaissance Toolkit             ║
║                       Version 2.0                        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)

def show_legal_disclaimer():
    """Display legal warning and get user confirmation"""
    print_colored(Fore.RED, """
╔════════════════════════════════════════════════════════════╗
║                    ⚠️  LEGAL WARNING ⚠️                    ║
╠════════════════════════════════════════════════════════════╣
║  Only scan systems you own or have explicit permission to ║
║  test. Unauthorized scanning is ILLEGAL and may result in: ║
║                                                            ║
║  • Criminal prosecution                                    ║
║  • Civil lawsuits                                          ║
║  • Network bans and ISP penalties                          ║
║                                                            ║
║  By continuing, you accept full legal responsibility.      ║
╚════════════════════════════════════════════════════════════╝
    """)
    confirm = input(Fore.YELLOW + "I have authorization to scan this target (yes/no): " + Style.RESET_ALL)
    if confirm.lower() != 'yes':
        print_colored(Fore.RED, "[!] Authorization not confirmed. Exiting.")
        sys.exit(0)

def validate_target(target: str) -> bool:
    """Validates if target is a valid IP address or domain name"""
    # Check if valid IP
    try:
        ipaddress.ip_address(target)
        return True
    except ValueError:
        pass
    
    # Check if valid domain (basic validation)
    domain_regex = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    if re.match(domain_regex, target):
        return True
    
    # Check if valid hostname (single word)
    hostname_regex = r'^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
    if re.match(hostname_regex, target):
        return True
    
    return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal"""
    return "".join(c if c.isalnum() or c in ['.', '-', '_'] else "_" for c in filename)

def check_tools() -> List[str]:
    """Checks if required tools are installed and returns list of missing tools"""
    print_colored(Fore.CYAN, "[*] Checking for required tools...")
    required_tools = ["nmap", "nikto", "dirb", "sublist3r"]
    missing_tools = []
    
    for tool in required_tools:
        try:
            result = subprocess.run(
                ["which", tool], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            if result.returncode != 0:
                missing_tools.append(tool)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing_tools.append(tool)
    
    if missing_tools:
        print_colored(Fore.RED, f"[!] Missing tools: {', '.join(missing_tools)}")
        print_colored(Fore.YELLOW, "[!] Install them to use all features. Continuing with available tools...")
        return missing_tools
    else:
        print_colored(Fore.GREEN, "[+] All required tools are present.")
        return []

def log_scan(target: str, scan_type: str, status: str, results_dir: str):
    """Log scan activity to history file"""
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'target': target,
        'scan_type': scan_type,
        'status': status,
        'results_directory': results_dir
    }
    
    log_file = Path.home() / '.nethical_history.jsonl'
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print_colored(Fore.YELLOW, f"[!] Could not write to scan history: {e}")

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
        return None
    except Exception as e:
        print_colored(Fore.YELLOW, f"[!] Error reading {file_path}: {e}")
        return None

# --- SCAN FUNCTIONS ---

def nmap_scan(target_ip: str, full_scan: bool = True) -> bool:
    """Perform Nmap scan with proper security"""
    scan_type = "Full" if full_scan else "Quick"
    print_colored(Fore.CYAN, f"[*] Starting Nmap {scan_type} Scan on {target_ip}...")
    print_colored(Fore.CYAN, "[*] This may take several minutes. Please wait...")
    
    # Build command as list (prevents shell injection)
    command = ["nmap", "-sV", "-oN", "nmap_results.txt", target_ip]
    if full_scan:
        command.insert(1, "-p-")
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        print_colored(Fore.GREEN, f"[+] Nmap {scan_type} Scan complete. Results saved to nmap_results.txt")
        log_scan(target_ip, f"nmap_{scan_type.lower()}", "success", os.getcwd())
        return True
    except subprocess.CalledProcessError as e:
        print_colored(Fore.RED, f"[!] Nmap scan failed: {e.stderr}")
        log_scan(target_ip, f"nmap_{scan_type.lower()}", "failed", os.getcwd())
        return False
    except subprocess.TimeoutExpired:
        print_colored(Fore.RED, "[!] Nmap scan timed out after 1 hour")
        log_scan(target_ip, f"nmap_{scan_type.lower()}", "timeout", os.getcwd())
        return False
    except FileNotFoundError:
        print_colored(Fore.RED, "[!] Nmap not found. Please install it first.")
        return False

def nikto_scan(target_ip: str) -> bool:
    """Perform Nikto web vulnerability scan"""
    print_colored(Fore.CYAN, f"[*] Starting Nikto Scan on {target_ip}...")
    print_colored(Fore.CYAN, "[*] This may take several minutes. Please wait...")
    
    command = ["nikto", "-h", target_ip, "-output", "nikto_results.txt"]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        print_colored(Fore.GREEN, "[+] Nikto Scan complete. Results saved to nikto_results.txt")
        log_scan(target_ip, "nikto", "success", os.getcwd())
        return True
    except subprocess.CalledProcessError as e:
        print_colored(Fore.RED, f"[!] Nikto scan failed: {e.stderr}")
        log_scan(target_ip, "nikto", "failed", os.getcwd())
        return False
    except subprocess.TimeoutExpired:
        print_colored(Fore.RED, "[!] Nikto scan timed out after 30 minutes")
        log_scan(target_ip, "nikto", "timeout", os.getcwd())
        return False
    except FileNotFoundError:
        print_colored(Fore.RED, "[!] Nikto not found. Please install it first.")
        return False

def dirb_scan(target_ip: str, wordlist_path: Optional[str] = None) -> bool:
    """Perform Dirb directory scan"""
    print_colored(Fore.CYAN, f"[*] Starting Dirb Scan on {target_ip}...")
    
    if not wordlist_path:
        wordlist_path = config.default_wordlist
        print_colored(Fore.YELLOW, f"[*] Using default wordlist: {wordlist_path}")
    
    if not os.path.exists(wordlist_path):
        print_colored(Fore.RED, f"[!] Wordlist not found at: {wordlist_path}")
        return False
    
    print_colored(Fore.CYAN, "[*] This may take several minutes. Please wait...")
    
    command = ["dirb", f"http://{target_ip}", wordlist_path, "-o", "dirb_results.txt"]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        print_colored(Fore.GREEN, "[+] Dirb Scan complete. Results saved to dirb_results.txt")
        log_scan(target_ip, "dirb", "success", os.getcwd())
        return True
    except subprocess.CalledProcessError as e:
        print_colored(Fore.RED, f"[!] Dirb scan failed: {e.stderr}")
        log_scan(target_ip, "dirb", "failed", os.getcwd())
        return False
    except subprocess.TimeoutExpired:
        print_colored(Fore.RED, "[!] Dirb scan timed out after 30 minutes")
        log_scan(target_ip, "dirb", "timeout", os.getcwd())
        return False
    except FileNotFoundError:
        print_colored(Fore.RED, "[!] Dirb not found. Please install it first.")
        return False

def sublist3r_scan(target: str) -> bool:
    """Perform Sublist3r subdomain enumeration"""
    print_colored(Fore.CYAN, f"[*] Starting Sublist3r Scan on {target}...")
    print_colored(Fore.CYAN, "[*] This may take several minutes. Please wait...")
    
    command = ["sublist3r", "-d", target, "-o", "sublist3r_results.txt"]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        print_colored(Fore.GREEN, "[+] Sublist3r Scan complete. Results saved to sublist3r_results.txt")
        log_scan(target, "sublist3r", "success", os.getcwd())
        return True
    except subprocess.CalledProcessError as e:
        print_colored(Fore.RED, f"[!] Sublist3r scan failed: {e.stderr}")
        log_scan(target, "sublist3r", "failed", os.getcwd())
        return False
    except subprocess.TimeoutExpired:
        print_colored(Fore.RED, "[!] Sublist3r scan timed out after 30 minutes")
        log_scan(target, "sublist3r", "timeout", os.getcwd())
        return False
    except FileNotFoundError:
        print_colored(Fore.RED, "[!] Sublist3r not found. Please install it first.")
        return False

# --- AI REPORTING FUNCTION ---

def generate_ai_report(scan_data: str, api_key: str, target: str) -> str:
    """Generate AI-powered security analysis report"""
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to initialize OpenAI client: {e}")

    prompt = f"""You are an expert penetration tester analyzing security scan results.

CONTEXT:
- Target: {target}
- Scan Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- Tools Used: Nmap, Nikto, Dirb, Sublist3r

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
                {"role": "system", "content": "You are an expert cybersecurity analyst specializing in penetration testing and vulnerability assessment. Generate accurate, actionable reports based strictly on provided scan data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, factual output
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
        
        with open(markdown_file, 'r') as f:
            md_content = f.read()
        
        html_content = f"""
<!DOCTYPE html>
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
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print_colored(Fore.GREEN, f"[+] HTML report also saved to {html_file}")
    except ImportError:
        print_colored(Fore.YELLOW, "[!] 'markdown' package not installed. Skipping HTML export.")
    except Exception as e:
        print_colored(Fore.YELLOW, f"[!] Could not export to HTML: {e}")

# --- MENU FUNCTIONS ---

def main_menu():
    """Display the main menu"""
    print_colored(Fore.MAGENTA, "\n" + "="*60)
    print_colored(Fore.YELLOW, "Select a scan to perform:")
    print("  1. Nmap Full Scan (All Ports)")
    print("  2. Nmap Quick Scan (Top 1000 Ports)")
    print("  3. Nikto Web Vulnerability Scan")
    print("  4. Dirb Web Directory Scan")
    print("  5. Sublist3r Subdomain Enumeration")
    print("  6. Run All Scans (Recommended)")
    print_colored(Fore.CYAN, "  8. Generate AI Security Report")
    print_colored(Fore.YELLOW, "  9. View Scan History")
    print_colored(Fore.RED, "  0. Exit")
    print_colored(Fore.MAGENTA, "="*60)

def view_scan_history():
    """Display scan history"""
    log_file = Path.home() / '.nethical_history.jsonl'
    
    if not log_file.exists():
        print_colored(Fore.YELLOW, "[!] No scan history found.")
        return
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print_colored(Fore.YELLOW, "[!] No scan history found.")
            return
        
        print_colored(Fore.CYAN, "\n" + "="*60)
        print_colored(Fore.CYAN, "SCAN HISTORY (Last 10 entries)")
        print_colored(Fore.CYAN, "="*60)
        
        for line in lines[-10:]:
            try:
                entry = json.loads(line)
                timestamp = entry.get('timestamp', 'N/A')
                target = entry.get('target', 'N/A')
                scan_type = entry.get('scan_type', 'N/A')
                status = entry.get('status', 'N/A')
                
                status_color = Fore.GREEN if status == 'success' else Fore.RED
                print(f"{Fore.YELLOW}{timestamp}{Style.RESET_ALL} | {Fore.CYAN}{target}{Style.RESET_ALL} | {scan_type} | {status_color}{status}{Style.RESET_ALL}")
            except json.JSONDecodeError:
                continue
        
        print_colored(Fore.CYAN, "="*60 + "\n")
    except Exception as e:
        print_colored(Fore.RED, f"[!] Error reading scan history: {e}")

# --- MAIN EXECUTION ---

def main():
    """Main execution function"""
    print_banner()
    
    # Check for required tools
    missing_tools = check_tools()
    
    # Show legal disclaimer
    show_legal_disclaimer()
    
    # Get target
    target = input(Fore.YELLOW + "\n[?] Enter the target domain or IP address: " + Style.RESET_ALL).strip()
    
    if not target:
        print_colored(Fore.RED, "[!] Target cannot be empty. Exiting.")
        sys.exit(1)
    
    if not validate_target(target):
        print_colored(Fore.RED, "[!] Invalid target format. Please enter a valid IP address or domain name.")
        sys.exit(1)
    
    target_ip = target
    
    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_target = sanitize_filename(target)
    dir_name = f"{safe_target}_{timestamp}"
    
    try:
        os.makedirs(dir_name, exist_ok=True)
        os.chdir(dir_name)
        print_colored(Fore.GREEN, f"[+] Results will be saved in: {os.getcwd()}\n")
    except Exception as e:
        print_colored(Fore.RED, f"[!] Could not create results directory: {e}")
        sys.exit(1)
    
    # Main loop
    try:
        while True:
            main_menu()
            choice = input(Fore.CYAN + ">> Enter your choice: " + Style.RESET_ALL).strip()

            if choice == '1':
                nmap_scan(target_ip, full_scan=True)
            
            elif choice == '2':
                nmap_scan(target_ip, full_scan=False)
            
            elif choice == '3':
                nikto_scan(target_ip)
            
            elif choice == '4':
                wordlist = input(Fore.YELLOW + f"[?] Enter wordlist path (press Enter for default: {config.default_wordlist}): " + Style.RESET_ALL).strip()
                dirb_scan(target_ip, wordlist if wordlist else None)
            
            elif choice == '5':
                sublist3r_scan(target)
            
            elif choice == '6':
                print_colored(Fore.CYAN, "\n[*] Running all available scans...")
                print_colored(Fore.YELLOW, "[*] This will take a significant amount of time. Please be patient.\n")
                
                if 'nmap' not in missing_tools:
                    nmap_scan(target_ip, full_scan=True)
                if 'nikto' not in missing_tools:
                    nikto_scan(target_ip)
                if 'dirb' not in missing_tools:
                    dirb_scan(target_ip)
                if 'sublist3r' not in missing_tools:
                    sublist3r_scan(target)
                
                print_colored(Fore.GREEN, "\n[+] All available scans completed.")
            
            elif choice == '8':
                print_colored(Fore.CYAN, "\n[*] Preparing to generate AI Security Report...")
                
                report_files = ['nmap_results.txt', 'nikto_results.txt', 'dirb_results.txt', 'sublist3r_results.txt']
                full_scan_data = ""
                
                for file in report_files:
                    content = read_file_safe(file)
                    if content:
                        full_scan_data += f"\n{'='*60}\n{file.upper()}\n{'='*60}\n{content}\n"
                
                if not full_scan_data:
                    print_colored(Fore.RED, "[!] No scan result files found. Please run a scan first.")
                    continue
                
                # Get API key
                api_key = config.openai_api_key
                if not api_key:
                    api_key = input(Fore.YELLOW + "[?] Enter your OpenAI API key (or set OPENAI_API_KEY environment variable): " + Style.RESET_ALL).strip()
                
                if not api_key:
                    print_colored(Fore.RED, "[!] API key is required to generate AI report.")
                    continue
                
                try:
                    print_colored(Fore.CYAN, f"[*] Contacting OpenAI (using model: {config.openai_model})...")
                    print_colored(Fore.CYAN, "[*] This may take 30-60 seconds...")
                    
                    ai_report = generate_ai_report(full_scan_data, api_key, target)
                    
                    report_filename = f'AI_Security_Report_{safe_target}_{timestamp}.md'
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        f.write(f"# Security Assessment Report\n\n")
                        f.write(f"**Target:** {target}\n\n")
                        f.write(f"**Scan Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
                        f.write(f"**Generated by:** Nethical v2.0\n\n")
                        f.write("---\n\n")
                        f.write(ai_report)
                    
                    print_colored(Fore.GREEN, f"\n[+] AI Security Report saved to: {report_filename}")
                    
                    # Optional HTML export
                    export_html = input(Fore.YELLOW + "[?] Export report to HTML as well? (y/n): " + Style.RESET_ALL).strip().lower()
                    if export_html == 'y':
                        export_to_html(report_filename)
                    
                    log_scan(target, "ai_report", "success", os.getcwd())
                
                except Exception as e:
                    print_colored(Fore.RED, f"[!] AI report generation failed: {e}")
                    log_scan(target, "ai_report", "failed", os.getcwd())
            
            elif choice == '9':
                view_scan_history()
            
            elif choice == '0':
                print_colored(Fore.YELLOW, "\n[*] Exiting Nethical. Stay safe and ethical!")
                break
            
            else:
                print_colored(Fore.RED, "[!] Invalid choice. Please try again.")

    except KeyboardInterrupt:
        print_colored(Fore.YELLOW, "\n\n[*] User interrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print_colored(Fore.RED, f"\n[!] Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Return to parent directory
        try:
            os.chdir("..")
        except:
            pass

if __name__ == "__main__":
    main()
```
