"""
Google Vertex AI Integration Demo

This example demonstrates how to use Vertex AI with Nethical governance.
"""

import os
from nethical.integrations.cloud import VertexAIConnector

def main():
    print("=" * 60)
    print("Vertex AI Governance Integration Demo")
    print("=" * 60)
    
    # Initialize Vertex AI connector
    connector = VertexAIConnector(
        project=os.getenv("GCP_PROJECT", "my-project"),
        location=os.getenv("GCP_LOCATION", "us-central1"),
        enable_governance=True
    )
    
    if not connector.available:
        print("\n⚠️  Vertex AI not available. Install with: pip install google-cloud-aiplatform")
        return
    
    print("\n✓ Vertex AI connector initialized")
    print(f"  Project: {connector.project}")
    print(f"  Location: {connector.location}")
    print(f"  Governance: {'Enabled' if connector.enable_governance else 'Disabled'}")
    
    # Example 1: Start an experiment run
    print("\n1. Starting experiment run...")
    run_id = connector.start_run(
        experiment_name="nethical-governance-demo",
        run_name="demo-run-1"
    )
    print(f"   ✓ Run started: {run_id}")
    
    # Example 2: Log parameters
    print("\n2. Logging parameters...")
    connector.log_parameters(run_id, {
        "model_type": "governance",
        "risk_threshold": 0.7,
        "enable_pii_detection": True,
        "max_tokens": 1000
    })
    print("   ✓ Parameters logged")
    
    # Example 3: Log metrics
    print("\n3. Logging metrics...")
    connector.log_metrics(run_id, {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.94,
        "f1_score": 0.93,
        "blocked_count": 25,
        "allowed_count": 975
    })
    print("   ✓ Metrics logged")
    
    # Example 4: Prediction with governance (simulated)
    print("\n4. Making prediction with governance checks...")
    print("   (This requires a deployed endpoint - showing structure)")
    
    # Uncomment when you have a deployed endpoint:
    # result = connector.predict_with_governance(
    #     endpoint_id="your-endpoint-id",
    #     instances=[
    #         {"text": "What is AI safety?"},
    #         {"text": "How do I hack a system?"}
    #     ]
    # )
    # print(f"   Result: {result}")
    
    print("   ✓ Prediction flow demonstrated")
    
    # Example 5: End the run
    print("\n5. Ending experiment run...")
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
    print("Demo complete! Check Vertex AI console for experiments.")
    print("=" * 60)


if __name__ == "__main__":
    main()
