"""
Full Stack Observability & Cloud ML Demo

This example demonstrates the complete ecosystem integration:
- Multi-provider observability
- Cloud ML platforms with governance
- End-to-end governance tracking
"""

import os
from datetime import datetime
from nethical.integrations.observability import create_observability_stack, GovernanceMetrics
from nethical.integrations.cloud import VertexAIConnector, DatabricksConnector

def main():
    print("=" * 70)
    print("Full Stack Governance Integration Demo")
    print("Observability + Cloud ML + Governance")
    print("=" * 70)
    
    # Part 1: Initialize Observability Stack
    print("\n" + "=" * 70)
    print("PART 1: Observability Stack Setup")
    print("=" * 70)
    
    obs_manager = create_observability_stack(
        langfuse_config={
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-..."),
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-..."),
        } if os.getenv("LANGFUSE_PUBLIC_KEY") else None,
        
        langsmith_config={
            "api_key": os.getenv("LANGSMITH_API_KEY", "ls-..."),
            "project_name": "nethical-full-stack-demo"
        } if os.getenv("LANGSMITH_API_KEY") else None,
        
        arize_config={
            "api_key": os.getenv("ARIZE_API_KEY", "api-..."),
            "space_key": os.getenv("ARIZE_SPACE_KEY", "space-..."),
            "model_id": "nethical-governance"
        } if os.getenv("ARIZE_API_KEY") else None,
    )
    
    providers = obs_manager.list_providers()
    print(f"\n✓ Observability stack initialized with {len(providers)} providers:")
    for provider in providers:
        print(f"   - {provider}")
    
    # Part 2: Initialize Cloud ML Platforms
    print("\n" + "=" * 70)
    print("PART 2: Cloud ML Platform Setup")
    print("=" * 70)
    
    # Vertex AI
    vertex_ai = None
    if os.getenv("GCP_PROJECT"):
        vertex_ai = VertexAIConnector(
            project=os.getenv("GCP_PROJECT"),
            location=os.getenv("GCP_LOCATION", "us-central1"),
            enable_governance=True
        )
        if vertex_ai.available:
            print("\n✓ Vertex AI connector initialized")
            print(f"   Project: {vertex_ai.project}")
        else:
            vertex_ai = None
    
    # Databricks
    databricks = None
    if os.getenv("DATABRICKS_HOST"):
        databricks = DatabricksConnector(
            workspace_url=os.getenv("DATABRICKS_HOST"),
            token=os.getenv("DATABRICKS_TOKEN"),
            enable_governance=True
        )
        if databricks.available:
            print("\n✓ Databricks connector initialized")
            print(f"   Workspace: {databricks.workspace_url}")
        else:
            databricks = None
    
    # Part 3: Run Governed ML Experiment
    print("\n" + "=" * 70)
    print("PART 3: Governed ML Experiment")
    print("=" * 70)
    
    if vertex_ai and vertex_ai.available:
        print("\nRunning Vertex AI experiment with governance...")
        
        # Start experiment run
        run_id = vertex_ai.start_run(
            experiment_name="full-stack-demo",
            run_name=f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        )
        print(f"   ✓ Experiment started: {run_id}")
        
        # Log governance event to all observability providers
        print("\n   Logging governance events...")
        obs_manager.log_governance_event_all(
            action="Train ML model on sensitive data",
            decision="RESTRICT",
            risk_score=0.65,
            metadata={
                "experiment_id": run_id,
                "platform": "vertex_ai",
                "data_classification": "confidential",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        print("   ✓ Governance events logged to all providers")
        
        # Log experiment parameters
        vertex_ai.log_parameters(run_id, {
            "model_type": "transformer",
            "learning_rate": 0.001,
            "batch_size": 32,
            "governance_enabled": True
        })
        
        # Simulate training metrics
        vertex_ai.log_metrics(run_id, {
            "train_loss": 0.15,
            "val_loss": 0.18,
            "governance_score": 0.92,
            "pii_detections": 0
        })
        
        # End experiment
        vertex_ai.end_run(run_id, status="completed")
        print("   ✓ Experiment completed")
    
    # Part 4: Log Aggregated Metrics
    print("\n" + "=" * 70)
    print("PART 4: Aggregated Metrics")
    print("=" * 70)
    
    print("\nLogging aggregated governance metrics...")
    metrics = GovernanceMetrics(
        total_evaluations=5000,
        allowed_count=4200,
        blocked_count=600,
        restricted_count=200,
        average_risk_score=0.28,
        pii_detections=75,
        latency_p50_ms=15.3,
        latency_p99_ms=52.7,
        timestamp=datetime.utcnow()
    )
    
    obs_manager.log_metrics_all(metrics)
    print("✓ Metrics logged to all observability providers:")
    print(f"   - Total evaluations: {metrics.total_evaluations}")
    print(f"   - Allowed: {metrics.allowed_count} ({metrics.allowed_count/metrics.total_evaluations*100:.1f}%)")
    print(f"   - Blocked: {metrics.blocked_count} ({metrics.blocked_count/metrics.total_evaluations*100:.1f}%)")
    print(f"   - Restricted: {metrics.restricted_count} ({metrics.restricted_count/metrics.total_evaluations*100:.1f}%)")
    print(f"   - Avg risk score: {metrics.average_risk_score:.2f}")
    print(f"   - PII detections: {metrics.pii_detections}")
    print(f"   - Latency P50: {metrics.latency_p50_ms:.1f}ms")
    print(f"   - Latency P99: {metrics.latency_p99_ms:.1f}ms")
    
    # Part 5: Cleanup
    print("\n" + "=" * 70)
    print("PART 5: Cleanup")
    print("=" * 70)
    
    print("\nFlushing all observability providers...")
    obs_manager.flush_all()
    print("✓ All providers flushed")
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE - Summary")
    print("=" * 70)
    print(f"\n✓ Observability providers: {len(providers)}")
    print(f"✓ Cloud ML platforms: {sum([1 for p in [vertex_ai, databricks] if p and p.available])}")
    print(f"✓ Governance events logged: Multiple")
    print(f"✓ Experiments tracked: {'Yes' if vertex_ai and vertex_ai.available else 'Simulated'}")
    print("\nCheck your observability dashboards:")
    if "langfuse" in providers:
        print("   - Langfuse: https://cloud.langfuse.com")
    if "langsmith" in providers:
        print("   - LangSmith: https://smith.langchain.com")
    if "arize" in providers:
        print("   - Arize: https://app.arize.com")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
