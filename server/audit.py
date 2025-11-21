"""
Audit Logging for MCP Server

Provides audit logging functionality that writes JSON files to audit/ directory.
Each tool invocation creates a unique audit file.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from .models import Finding


class AuditLogger:
    """Handles audit logging for MCP tool invocations."""
    
    def __init__(self, audit_dir: str = "audit"):
        """
        Initialize the audit logger.
        
        Args:
            audit_dir: Directory to store audit files (default: "audit")
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True)
    
    def log_tool_invocation(
        self,
        tool: str,
        arguments: Dict[str, Any],
        findings: List[Finding],
        status: str,
        summary: str
    ) -> str:
        """
        Log a tool invocation to an audit file.
        
        Args:
            tool: Name of the tool invoked
            arguments: Arguments passed to the tool
            findings: List of findings from the tool
            status: Status of the invocation (ALLOW or BLOCK)
            summary: Summary of the results
            
        Returns:
            audit_id: Unique identifier for this audit entry
        """
        audit_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        audit_entry = {
            "audit_id": audit_id,
            "timestamp": timestamp,
            "tool": tool,
            "arguments": arguments,
            "status": status,
            "findings": [f.model_dump() for f in findings],
            "findings_count": len(findings),
            "summary": summary
        }
        
        # Write to audit file
        audit_file = self.audit_dir / f"{audit_id}.json"
        with open(audit_file, "w") as f:
            json.dump(audit_entry, f, indent=2)
        
        return audit_id
