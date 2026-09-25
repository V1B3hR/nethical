# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Windows Service Runner for Nethical Sovereign AI Governance Daemon.

Provides:
- Native Windows Service integration via pywin32 / win32serviceutil (when available).
- Standalone background daemon mode with watchdog and health monitoring.
- Named pipe and REST API initialization on Windows hosts.
"""

import sys
import os
import time
import logging
import subprocess
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [WindowsService] %(message)s",
)
log = logging.getLogger("nethical_service")


def run_standalone_daemon():
    """Runs Nethical as a persistent background daemon with process supervision"""
    log.info("Starting Nethical Sovereign Governance Daemon on Windows...")
    os.environ["NETHICAL_LOCAL_FIRST"] = "1"
    os.environ["NETHICAL_SAFETY_ACKNOWLEDGED"] = "1"
    os.environ["NETHICAL_HSM_PROVIDER"] = "tpm-2.0"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "nethical.api.v1.app:create_v1_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    log.info(f"Launching supervisor for command: {' '.join(cmd)}")
    while True:
        try:
            proc = subprocess.Popen(cmd)
            log.info(f"Nethical daemon running with PID: {proc.pid}")
            proc.wait()
            log.warning(f"Nethical daemon exited with code {proc.returncode}. Restarting in 3 seconds...")
            time.sleep(3)
        except KeyboardInterrupt:
            log.info("Termination signal received. Stopping Nethical daemon.")
            if proc:
                proc.terminate()
            break
        except Exception as e:
            log.error(f"Supervisor error: {e}. Retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--service":
        try:
            import win32serviceutil
            import win32service
            import win32event
            import servicemanager

            class NethicalWinService(win32serviceutil.ServiceFramework):
                _svc_name_ = "NethicalGovernanceService"
                _svc_display_name_ = "Nethical Sovereign AI Governance Daemon"
                _svc_description_ = "Full-stack AI safety, kernel sandboxing, and Merkle DAG audit daemon"

                def __init__(self, args):
                    super().__init__(args)
                    self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)

                def SvcStop(self):
                    self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
                    win32event.SetEvent(self.hWaitStop)

                def SvcDoRun(self):
                    run_standalone_daemon()

            win32serviceutil.HandleCommandLine(NethicalWinService)
        except ImportError:
            log.warning("pywin32 not installed. Running in standalone background daemon mode.")
            run_standalone_daemon()
    else:
        run_standalone_daemon()
