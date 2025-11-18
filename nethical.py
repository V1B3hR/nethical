# nethical.py
# Author: V1B3hR
# Description: A Python-based tool for automating reconnaissance tasks with an integrated AI reporting feature.

import os
import subprocess
import datetime
from colorama import Fore, Style, init

# --- NEW AI DEPENDENCY ---
# Make sure you have added 'openai' to your requirements.txt
from openai import OpenAI

# Initialize colorama
init(autoreset=True)

def print_colored(color, text):
    """Prints text in a specified color."""
    print(color + text + Style.RESET_ALL)

def check_tools():
    """Checks if all required tools are installed."""
    print_colored(Fore.CYAN, "[*] Checking for required tools...")
    required_tools = ["nmap", "nikto", "dirb", "sublist3r"]
    missing_tools = []
    for tool in required_tools:
        if subprocess.run(f"command -v {tool}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            missing_tools.append(tool)
    
    if missing_tools:
        print_colored(Fore.RED, f"[!] The following tools are not installed or not in PATH: {', '.join(missing_tools)}")
        exit(1)
    else:
        print_colored(Fore.GREEN, "[+] All required tools are present.")

def main_menu():
    """Displays the main menu."""
    print_colored(Fore.MAGENTA, "\n" + "="*40)
    print_colored(Fore.MAGENTA, " " * 15 + "NETHICAL")
    print_colored(Fore.MAGENTA, "="*40)
    print_colored(Fore.YELLOW, "Select a scan to perform:")
    print("1. Nmap Full Scan (All Ports)")
    print("2. Nmap Quick Scan (Top 1000 Ports)")
    print("3. Nikto Web Vulnerability Scan")
    print("4. Dirb Web Directory Scan")
    print("5. Sublist3r Subdomain Enumeration")
    print("6. Run All Scans (Recommended)")
    print_colored(Fore.CYAN, "8. [NEW!] Generate AI Summary Report")
    print_colored(Fore.RED, "9. Exit")
    print_colored(Fore.MAGENTA, "="*40)

# --- SCAN FUNCTIONS ---

def nmap_scan(target_ip, full_scan=True):
    scan_type = "Full" if full_scan else "Quick"
    ports = "-p-" if full_scan else ""
    print_colored(Fore.CYAN, f"[*] Starting Nmap {scan_type} Scan...")
    command = f"nmap {ports} -sV -oN nmap_results.txt {target_ip}"
    subprocess.run(command, shell=True)
    print_colored(Fore.GREEN, f"[+] Nmap {scan_type} Scan complete. Results saved to nmap_results.txt")

def nikto_scan(target_ip):
    print_colored(Fore.CYAN, "[*] Starting Nikto Scan...")
    command = f"nikto -h {target_ip} -output nikto_results.txt"
    subprocess.run(command, shell=True)
    print_colored(Fore.GREEN, "[+] Nikto Scan complete. Results saved to nikto_results.txt")

def dirb_scan(target_ip):
    print_colored(Fore.CYAN, "[*] Starting Dirb Scan...")
    wordlist_path = input(Fore.YELLOW + "[?] Enter the path to your wordlist (e.g., /usr/share/wordlists/dirb/common.txt): " + Style.RESET_ALL)
    if not os.path.exists(wordlist_path):
        print_colored(Fore.RED, "[!] Wordlist not found at the specified path.")
        return
    command = f"dirb http://{target_ip} {wordlist_path} -o dirb_results.txt"
    subprocess.run(command, shell=True)
    print_colored(Fore.GREEN, "[+] Dirb Scan complete. Results saved to dirb_results.txt")

def sublist3r_scan(target):
    print_colored(Fore.CYAN, "[*] Starting Sublist3r Scan...")
    command = f"sublist3r -d {target} -o sublist3r_results.txt"
    subprocess.run(command, shell=True)
    print_colored(Fore.GREEN, "[+] Sublist3r Scan complete. Results saved to sublist3r_results.txt")

# --- NEW AI REPORTING FUNCTION ---

def generate_ai_report(scan_data: str, api_key: str) -> str:
    """
    Analyzes raw security scan data using the OpenAI GPT model and generates
    a structured, human-readable report in Markdown format.
    """
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to initialize OpenAI client: {e}")

    prompt = f"""
    As an expert cybersecurity analyst, your task is to analyze the following raw security scan results.
    The results are from tools like Nmap, Nikto, and Dirb. Your analysis must be clear, concise, and actionable.

    Generate a report in Markdown format with the following strict structure:

    ### Executive Summary
    Provide a brief, high-level overview of the findings suitable for a manager. Highlight the most critical discoveries and the overall security posture.

    ### Open Ports Analysis (from Nmap)
    List all discovered open TCP/UDP ports. For each port, specify the service detected and its version, if available. Briefly note the security implication of any particularly sensitive open ports (e.g., SSH, RDP, Telnet).

    ### Web Server Vulnerabilities (from Nikto)
    Summarize any potential vulnerabilities, outdated software, or dangerous misconfigurations found by the web server scan. Group similar findings if possible.

    ### Discovered Web Directories & Files (from Dirb/Sublist3r)
    List any interesting directories, files, or subdomains that were discovered. Highlight any that appear sensitive or out of place (e.g., `/admin`, `/backup`, `.git`, `config.php.bak`).

    ### Prioritized Recommendations
    Provide a numbered list of concrete actions the system administrator should take to remediate the findings. Start with the most critical recommendations first.

    ---
    RAW SCAN DATA TO ANALYZE:
    ---
    {scan_data}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful cybersecurity analyst that generates structured reports in Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        report = response.choices[0].message.content
        return report.strip()
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    check_tools()
    
    target = input(Fore.YELLOW + "[?] Enter the target domain or IP address: " + Style.RESET_ALL)
    target_ip = target # Assuming IP for tools that need it, domain for others.
    
    # Create a unique directory for the target's results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{target}_{timestamp}"
    os.makedirs(dir_name)
    os.chdir(dir_name)
    print_colored(Fore.GREEN, f"[+] Results will be saved in: {os.getcwd()}")

    try:
        while True:
            main_menu()
            choice = input(Fore.CYAN + ">> Enter your choice: " + Style.RESET_ALL)

            if choice == '1':
                nmap_scan(target_ip, full_scan=True)
            elif choice == '2':
                nmap_scan(target_ip, full_scan=False)
            elif choice == '3':
                nikto_scan(target_ip)
            elif choice == '4':
                dirb_scan(target_ip)
            elif choice == '5':
                sublist3r_scan(target)
            elif choice == '6':
                print_colored(Fore.CYAN, "[*] Running all scans...")
                nmap_scan(target_ip, full_scan=True)
                nikto_scan(target_ip)
                dirb_scan(target_ip)
                sublist3r_scan(target)
                print_colored(Fore.GREEN, "[+] All scans completed.")
            
            # --- NEW AI REPORTING LOGIC ---
            elif choice == '8':
                print_colored(Fore.CYAN, "[*] Preparing to generate AI Summary Report...")
                
                report_files = ['nmap_results.txt', 'nikto_results.txt', 'dirb_results.txt', 'sublist3r_results.txt']
                full_scan_data = ""
                
                for file in report_files:
                    if os.path.exists(file):
                        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                            full_scan_data += f"--- {file.upper()} ---\n{f.read()}\n\n"

                if not full_scan_data:
                    print_colored(Fore.RED, "[!] No scan result files found in this directory to analyze. Please run a scan first.")
                    continue

                try:
                    api_key = input(Fore.YELLOW + "[?] Please enter your OpenAI API key: " + Style.RESET_ALL)
                    if not api_key:
                        print_colored(Fore.RED, "[!] API key cannot be empty.")
                        continue
                    
                    print_colored(Fore.CYAN, "[*] Contacting OpenAI... This may take a moment.")
                    ai_report = generate_ai_report(full_scan_data, api_key)
                    
                    report_filename = 'AI_Summary_Report.md'
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        f.write(ai_report)
                    
                    print_colored(Fore.GREEN, f"[+] AI Summary Report saved successfully to {report_filename}")

                except Exception as e:
                    print_colored(Fore.RED, f"[!] An error occurred during AI report generation: {e}")

            elif choice == '9':
                print_colored(Fore.YELLOW, "[*] Exiting Nethical. Goodbye!")
                break
            else:
                print_colored(Fore.RED, "[!] Invalid choice, please try again.")

    except KeyboardInterrupt:
        print_colored(Fore.YELLOW, "\n[*] User interrupted. Exiting...")
    finally:
        # Change back to the original directory
        os.chdir("..")
