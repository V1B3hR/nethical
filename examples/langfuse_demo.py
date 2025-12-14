"""
Langfuse Observability Demo

This example demonstrates how to use Langfuse for Nethical governance observability.
"""

import os
from datetime import datetime
from nethical.integrations.observability import LangfuseConnector, TraceSpan, GovernanceMetrics

def main():
    print("=" * 60)
    print("Langfuse Governance Observability Demo")
    print("=" * 60)
    
    # Initialize Langfuse connector
    # In production, use environment variables
    connector = LangfuseConnector(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    
    if not connector.available:
        print("\n⚠️  Langfuse not available. Install with: pip install langfuse")
        return
    
    print("\n✓ Langfuse connector initialized")
    
    # Example 1: Log a governance event
    print("\n1. Logging governance event...")
    connector.log_governance_event(
        action="Generate code to delete files",
        decision="BLOCK",
        risk_score=0.95,
        metadata={
            "agent_id": "code-assistant-1",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": "Potentially destructive action"
        }
    )
    print("   ✓ Governance event logged")
    
    # Example 2: Log a trace span with governance
    print("\n2. Logging trace span with governance...")
    span = TraceSpan(
        trace_id="trace-123",
        span_id="span-456",
        parent_span_id=None,
        name="code_generation",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        attributes={
            "action": "Generate greeting function",
            "agent_id": "code-assistant-1",
            "language": "python"
        },
        governance_result={
            "decision": "ALLOW",
            "risk_score": 0.1,
            "pii_detected": False
        }
    )
    connector.log_trace(span)
    print("   ✓ Trace span logged")
    
    # Example 3: Log aggregated metrics
    print("\n3. Logging aggregated metrics...")
    metrics = GovernanceMetrics(
        total_evaluations=1000,
        allowed_count=850,
        blocked_count=100,
        restricted_count=50,
        average_risk_score=0.25,
        pii_detections=15,
        latency_p50_ms=12.5,
        latency_p99_ms=45.2,
        timestamp=datetime.utcnow()
    )
    connector.log_metrics(metrics)
    print("   ✓ Metrics logged")
    
    # Example 4: Get dashboard URL
    print("\n4. Getting dashboard URL...")
    dashboard_url = connector.create_dashboard("Nethical Governance")
    print(f"   Dashboard: {dashboard_url}")
    
    # Flush events
    print("\n5. Flushing events...")
    connector.flush()
    print("   ✓ Events flushed")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check Langfuse dashboard for traces.")
    print("=" * 60)


if __name__ == "__main__":
    main()
