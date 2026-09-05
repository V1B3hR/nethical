"""Wieloplatformowy (Windows & Linux / Docker / Kubernetes) kanał IPC dla Ambasadora Błyskawicy.

Zapewnia sub-milisekundowy czas odpowiedzi rzędu mikrosekund:
- Na Windows: poprzez natywne Windows Named Pipes (_winapi)
- Na Linux / Docker / Datacentres: poprzez natywne gniazda UNIX Domain Sockets (socket.AF_UNIX)
- Zdalnie (opcjonalnie): poprzez bezpieczny strumień TCP/mTLS.
"""

import json
import os
import sys
import time
import uuid
import socket
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("nethical.ambassador.channel")

DEFAULT_WIN_PIPE = r"\\.\pipe\blyskawica_nethical_ambassador"
DEFAULT_UNIX_SOCK = "/tmp/blyskawica_nethical_ambassador.sock"

IS_WINDOWS = sys.platform == "win32"

try:
    import _winapi
    HAS_WINAPI = True
except ImportError:
    HAS_WINAPI = False


class AmbassadorChannel:
    """Natywny, wieloplatformowy klient IPC dla komunikacji Nethical <-> Błyskawica."""

    def __init__(self, pipe_path: Optional[str] = None, timeout_ms: int = 500):
        if pipe_path:
            self.ipc_path = pipe_path
        else:
            env_path = os.getenv("NETHICAL_AMBASSADOR_IPC_PATH")
            if env_path:
                self.ipc_path = env_path
            else:
                self.ipc_path = DEFAULT_WIN_PIPE if IS_WINDOWS else DEFAULT_UNIX_SOCK

        self.timeout_ms = timeout_ms

    def is_available(self) -> bool:
        """Sprawdza, czy kanał IPC jest aktywny w systemie (Windows Named Pipe lub Linux Socket)."""
        if IS_WINDOWS:
            if not HAS_WINAPI:
                return False
            try:
                handle = self._open_pipe_win()
                _winapi.CloseHandle(handle)
                return True
            except Exception:
                return False
        else:
            # Linux / Unix Domain Socket
            if not os.path.exists(self.ipc_path):
                return False
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(self.timeout_ms / 1000.0)
                    s.connect(self.ipc_path)
                    return True
            except Exception:
                return False

    def _open_pipe_win(self):
        """Otwiera Windows Named Pipe z obsługą WaitNamedPipe w przypadku zajętości."""
        try:
            return _winapi.CreateFile(
                self.ipc_path,
                _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
                0,
                _winapi.NULL,
                _winapi.OPEN_EXISTING,
                0,
                _winapi.NULL,
            )
        except OSError as e:
            if getattr(e, "winerror", None) == 231:  # ERROR_PIPE_BUSY
                try:
                    _winapi.WaitNamedPipe(self.ipc_path, self.timeout_ms)
                    return _winapi.CreateFile(
                        self.ipc_path,
                        _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
                        0,
                        _winapi.NULL,
                        _winapi.OPEN_EXISTING,
                        0,
                        _winapi.NULL,
                    )
                except Exception:
                    raise e
            raise e

    def send_command(
        self, command: str, payload: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], float]:
        """Wysyła polecenie do Błyskawicy przez IPC (Named Pipe na Windows lub Unix Socket na Linux) i odbiera odpowiedź."""
        req_id = str(uuid.uuid4())[:8]
        request_obj = {
            "id": req_id,
            "command": command,
            "payload": payload or {},
        }
        raw_req = (json.dumps(request_obj) + "\n").encode("utf-8")

        t_start = time.perf_counter()

        if IS_WINDOWS:
            return self._send_win(raw_req, t_start)
        else:
            return self._send_unix(raw_req, t_start)

    def _send_win(self, raw_req: bytes, t_start: float) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], float]:
        """Obsługa transmisji IPC na systemie Windows."""
        if not HAS_WINAPI:
            return False, None, "Brak wsparcia dla _winapi", 0.0

        handle = None
        try:
            handle = self._open_pipe_win()
            _winapi.WriteFile(handle, raw_req)
            raw_resp, err = _winapi.ReadFile(handle, 16384)
            t_elapsed_us = (time.perf_counter() - t_start) * 1_000_000

            if not raw_resp:
                return False, None, f"Pusta odpowiedź z potoku IPC (kod błędu: {err})", t_elapsed_us

            line = raw_resp.decode("utf-8", errors="replace").strip().split("\n")[0]
            resp_obj = json.loads(line)

            if resp_obj.get("status") == "ok":
                return True, resp_obj.get("data"), None, t_elapsed_us
            else:
                return False, None, resp_obj.get("error", "Błąd wewnętrzny daemona"), t_elapsed_us

        except FileNotFoundError:
            t_elapsed_us = (time.perf_counter() - t_start) * 1_000_000
            return (
                False,
                None,
                f"Potok {self.ipc_path} nie istnieje (daemon Błyskawicy nie jest uruchomiony)",
                t_elapsed_us,
            )
        except Exception as e:
            t_elapsed_us = (time.perf_counter() - t_start) * 1_000_000
            return False, None, f"Błąd komunikacji IPC Windows: {str(e)}", t_elapsed_us
        finally:
            if handle is not None:
                try:
                    _winapi.CloseHandle(handle)
                except Exception:
                    pass

    def _send_unix(self, raw_req: bytes, t_start: float) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], float]:
        """Obsługa transmisji IPC na systemie Linux / Docker / Kubernetes (UNIX Domain Socket)."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout_ms / 1000.0)
                s.connect(self.ipc_path)
                s.sendall(raw_req)

                raw_resp = s.recv(16384)
                t_elapsed_us = (time.perf_counter() - t_start) * 1_000_000

                if not raw_resp:
                    return False, None, "Pusta odpowiedź z gniazda UNIX", t_elapsed_us

                line = raw_resp.decode("utf-8", errors="replace").strip().split("\n")[0]
                resp_obj = json.loads(line)

                if resp_obj.get("status") == "ok":
                    return True, resp_obj.get("data"), None, t_elapsed_us
                else:
                    return False, None, resp_obj.get("error", "Błąd wewnętrzny daemona"), t_elapsed_us

        except FileNotFoundError:
            t_elapsed_us = (time.perf_counter() - t_start) * 1_000_000
            return (
                False,
                None,
                f"Gniazdo UNIX {self.ipc_path} nie istnieje (daemon Błyskawicy nie jest uruchomiony w kontenerze)",
                t_elapsed_us,
            )
        except Exception as e:
            t_elapsed_us = (time.perf_counter() - t_start) * 1_000_000
            return False, None, f"Błąd komunikacji UNIX Socket: {str(e)}", t_elapsed_us
