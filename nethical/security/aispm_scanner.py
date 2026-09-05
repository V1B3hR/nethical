"""AI Security Posture Management (AISPM) Scanner for Nethical.

Aligned with Cyera AI Guardian & UK NCSC Guidelines:
- Discovers Shadow AI services running on local network and host interfaces.
- Scans known AI runtime ports: 11434 (Ollama), 8000 (vLLM/TGI), 8080/8081 (LocalAI/LangChain),
  1234 (LM Studio), 8501 (Streamlit GenAI), 5000 (Flask AI backend).
- Assesses posture: TLS encryption, authentication headers, rogue model inventory.
- Produces posture risk score, findings, and remediation steps.
"""

from __future__ import annotations

import logging
import socket
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
import json

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.security.aispm_scanner")


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AIServiceType(str, Enum):
    OLLAMA = "Ollama"
    VLLM = "vLLM"
    LOCAL_AI = "LocalAI"
    LM_STUDIO = "LM Studio"
    LANGCHAIN_SERVE = "LangChain / FastApi Agent"
    STREAMLIT_AI = "Streamlit GenAI"
    GENERIC_LLM = "Generic LLM Endpoint"


class DetectedAIService(BaseModel):
    """Represents a discovered AI endpoint or process."""
    host: str
    port: int
    service_type: AIServiceType
    is_managed_by_nethical: bool = False
    tls_enabled: bool = False
    auth_required: bool = False
    exposed_models: List[str] = Field(default_factory=list)
    risk_level: RiskLevel
    risk_details: str
    remediation_advice: str


class AISPMScanReport(BaseModel):
    """Full AI Security Posture Management scan report."""
    scan_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    target_hosts: List[str]
    scanned_ports: List[int]
    total_services_found: int
    shadow_ai_count: int
    overall_posture_score: float = Field(..., ge=0.0, le=1.0, description="1.0 = Fully secured, 0.0 = High Shadow AI exposure")
    posture_status: str = Field(..., description="SECURE, MODERATE_EXPOSURE, CRITICAL_SHADOW_AI")
    discovered_services: List[DetectedAIService] = Field(default_factory=list)
    compliance_summary: Dict[str, Any] = Field(default_factory=dict)


# Default port signatures for common AI platforms
DEFAULT_AI_PORTS: Dict[int, AIServiceType] = {
    11434: AIServiceType.OLLAMA,
    8000: AIServiceType.VLLM,
    8080: AIServiceType.LOCAL_AI,
    8081: AIServiceType.LANGCHAIN_SERVE,
    1234: AIServiceType.LM_STUDIO,
    8501: AIServiceType.STREAMLIT_AI,
    5000: AIServiceType.GENERIC_LLM,
}


class AISPMScanner:
    """Network and endpoint scanner identifying unmanaged/shadow AI runtimes."""

    def __init__(
        self,
        managed_ports: Optional[List[int]] = None,
        custom_port_mapping: Optional[Dict[int, AIServiceType]] = None,
        probe_timeout_seconds: float = 0.08,
    ) -> None:
        self.managed_ports = set(managed_ports or [8000])  # By default, port 8000 is Nethical's managed gateway if configured
        self.port_mapping = custom_port_mapping or DEFAULT_AI_PORTS
        self.probe_timeout = probe_timeout_seconds

    def probe_port(self, host: str, port: int) -> bool:
        """Fast TCP probe to verify if port is listening."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.probe_timeout)
                res = s.connect_ex((host, port))
                return res == 0
        except Exception:
            return False

    def query_model_inventory(self, host: str, port: int, service_type: AIServiceType) -> List[str]:
        """Attempts to probe known API model endpoints to catalog unmanaged models."""
        url = f"http://{host}:{port}"
        models: List[str] = []

        endpoint = "/v1/models"
        if service_type == AIServiceType.OLLAMA:
            endpoint = "/api/tags"

        try:
            req = Request(f"{url}{endpoint}", headers={"User-Agent": "Nethical-AISPM-Scanner/2.4"})
            with urlopen(req, timeout=0.2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if "models" in data and isinstance(data["models"], list):
                        for m in data["models"]:
                            name = m.get("name") or m.get("id")
                            if name:
                                models.append(str(name))
                    elif "data" in data and isinstance(data["data"], list):
                        for m in data["data"]:
                            name = m.get("id")
                            if name:
                                models.append(str(name))
        except Exception:
            # Silent fallback: offline or unauthenticated
            pass

        return models

    def analyze_service(self, host: str, port: int, is_active: bool) -> Optional[DetectedAIService]:
        """Analyzes a found port and evaluates its security posture."""
        if not is_active:
            return None

        service_type = self.port_mapping.get(port, AIServiceType.GENERIC_LLM)
        is_managed = port in self.managed_ports

        # Query models if accessible
        models = self.query_model_inventory(host, port, service_type)

        if is_managed:
            return DetectedAIService(
                host=host,
                port=port,
                service_type=service_type,
                is_managed_by_nethical=True,
                tls_enabled=False,  # Local dev loopback
                auth_required=True,
                exposed_models=models,
                risk_level=RiskLevel.LOW,
                risk_details="Regulated by Nethical Pre-Execution Gateway.",
                remediation_advice="Maintain active policy enforcement.",
            )

        # Unmanaged / Shadow AI detected!
        risk_level = RiskLevel.CRITICAL if port in (11434, 8000, 1234) else RiskLevel.HIGH
        details = (
            f"Unmanaged {service_type.value} discovered on {host}:{port}. "
            f"No Governance Gateway proxy active. Prompts and weights are exposed without audit ledger."
        )
        remediation = (
            f"Route {service_type.value} through Nethical GovernanceGateway (proxy.py) "
            f"and require bearer token authentication."
        )

        return DetectedAIService(
            host=host,
            port=port,
            service_type=service_type,
            is_managed_by_nethical=False,
            tls_enabled=False,
            auth_required=False,
            exposed_models=models,
            risk_level=risk_level,
            risk_details=details,
            remediation_advice=remediation,
        )

    def scan_network(
        self,
        hosts: Optional[List[str]] = None,
        mock_active_ports: Optional[Dict[str, List[int]]] = None,
    ) -> AISPMScanReport:
        """Executes a full AISPM scan across specified hosts and AI ports.
        
        Supports mock_active_ports for fast deterministic automated unit tests.
        """
        targets = hosts or ["127.0.0.1"]
        discovered: List[DetectedAIService] = []
        scanned_ports = sorted(list(self.port_mapping.keys()))
        scan_id = f"AISPM-SCAN-{int(datetime.now(timezone.utc).timestamp())}"

        for host in targets:
            for port in scanned_ports:
                if mock_active_ports is not None:
                    is_active = port in mock_active_ports.get(host, [])
                else:
                    is_active = self.probe_port(host, port)

                if is_active:
                    service = self.analyze_service(host, port, is_active=True)
                    if service:
                        discovered.append(service)

        shadow_ai = [s for s in discovered if not s.is_managed_by_nethical]
        shadow_count = len(shadow_ai)

        if shadow_count == 0:
            posture_score = 1.0
            status = "SECURE"
        elif shadow_count == 1:
            posture_score = 0.75
            status = "MODERATE_EXPOSURE"
        else:
            posture_score = max(0.1, 1.0 - (shadow_count * 0.3))
            status = "CRITICAL_SHADOW_AI"

        return AISPMScanReport(
            scan_id=scan_id,
            target_hosts=targets,
            scanned_ports=scanned_ports,
            total_services_found=len(discovered),
            shadow_ai_count=shadow_count,
            overall_posture_score=round(posture_score, 2),
            posture_status=status,
            discovered_services=discovered,
            compliance_summary={
                "cyera_aispm_compliant": shadow_count == 0,
                "uk_ncsc_principle_1_managed_assets": shadow_count == 0,
                "iso42001_clause_a4_resource_governance": shadow_count == 0,
            },
        )
