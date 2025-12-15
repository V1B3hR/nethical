"""
Databricks Integration Demo

This example demonstrates how to use Databricks with Nethical governance.
"""

import os
from nethical.integrations.cloud import DatabricksConnector

def main():
    print("=" * 60)
    print("Databricks Governance Integration Demo")
    print("=" * 60)
    
    # Initialize Databricks connector
    connector = DatabricksConnector(
        workspace_url=os.getenv("DATABRICKS_HOST", "https://...databricks.com"),
        token=os.getenv("DATABRICKS_TOKEN"),
        enable_governance=True
    )
    
    if not connector.available:
        print("\n⚠️  Databricks SDK not available. Install with: pip install databricks-sdk mlflow")
        return
    
    print("\n✓ Databricks connector initialized")
    print(f"  Workspace: {connector.workspace_url}")
    print(f"  Governance: {'Enabled' if connector.enable_governance else 'Disabled'}")
    
    # Example 1: Start an MLflow run
    print("\n1. Starting MLflow run...")
    run_id = connector.start_run(
        experiment_name="/Users/your-user/nethical-demo",
        run_name="governance-demo-1"
    )
    print(f"   ✓ Run started: {run_id}")
    
    # Example 2: Log parameters
    print("\n2. Logging parameters...")
    connector.log_parameters(run_id, {
        "governance_enabled": True,
        "risk_threshold": 0.7,
        "model_version": "1.0",
        "dataset": "governance-test"
    })
    print("   ✓ Parameters logged")
    
    # Example 3: Log metrics
    print("\n3. Logging metrics...")
    connector.log_metrics(run_id, {
        "governance_score": 0.92,
        "blocked_actions": 15,
        "allowed_actions": 985,
        "pii_detections": 3,
        "avg_latency_ms": 25.5
    })
    print("   ✓ Metrics logged")
    
    # Example 4: Query serving endpoint with governance (simulated)
    print("\n4. Querying serving endpoint with governance...")
    print("   (This requires a deployed endpoint - showing structure)")
    
    # Uncomment when you have a serving endpoint:
    # result = connector.query_with_governance(
    #     endpoint_name="your-endpoint",
    #     query="What are best practices for AI safety?"
    # )
    # print(f"   Result: {result}")
    
    print("   ✓ Query flow demonstrated")
    
    # Example 5: Model registration with governance (simulated)
    print("\n5. Registering model with governance...")
    print("   (This requires a trained model - showing structure)")
    
    # Uncomment when you have a model to register:
    # result = connector.register_model_with_governance(
    #     model_name="governance-model",
    #     model_uri="runs:/run-id/model"
    # )
    # print(f"   Registration: {result}")
    
    print("   ✓ Registration flow demonstrated")
    
    # Example 6: End the run
    print("\n6. Ending MLflow run...")
    connector.end_run(run_id, status="completed")
    print("   ✓ Run completed")
    
    # Get run metadata
    run_info = connector.get_run(run_id)
    if run_info:
        print(f"\n   Run Info:")
        print(f"   - Experiment: {run_info.experiment_name}")
        print(f"   - Parameters: {len(run_info.parameters)} logged")
        print(f"   - Metrics: {len(run_info.metrics)} logged")
        print(f"   - Status: {run_info.status.value}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Check Databricks workspace for experiments.")
    print("=" * 60)


if __name__ == "__main__":
    main()
