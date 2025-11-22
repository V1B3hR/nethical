"""Ray Serve Integration for Nethical.

This module provides integration with Ray Serve for scalable ML model serving
with integrated safety and ethics governance.

Installation:
    pip install nethical ray[serve]

Usage:
    from nethical.integrations.ray_serve_connector import NethicalDeployment
    import ray
    from ray import serve
    
    # Wrap your model deployment with Nethical
    @serve.deployment
    class MyModel:
        def __call__(self, request):
            return self.model.predict(request)
    
    # Add Nethical safety layer
    safe_model = NethicalDeployment(MyModel)
    serve.run(safe_model.bind())

Features:
    - Automatic safety checks for all predictions
    - PII detection and redaction
    - Risk scoring for model inputs/outputs
    - Audit logging with Ray metrics
    - Integration with Ray observability stack
"""

from typing import Any, Dict, Optional, Callable
from datetime import datetime, timezone

from nethical.core.integrated_governance import IntegratedGovernance
from nethical.integrations._decision_logic import compute_decision


class NethicalRayServeMiddleware:
    """Middleware for Ray Serve deployments with Nethical safety checks.
    
    This middleware wraps Ray Serve deployments to add safety and ethics
    governance to model serving.
    
    Example:
        @serve.deployment
        class Model:
            def __call__(self, request):
                return predict(request)
        
        safe_model = NethicalRayServeMiddleware(Model)
        serve.run(safe_model.bind())
    """
    
    def __init__(
        self,
        deployment_class: Any,
        governance: Optional[IntegratedGovernance] = None,
        check_input: bool = True,
        check_output: bool = True,
        agent_id: str = "ray-serve"
    ):
        """Initialize Ray Serve middleware.
        
        Args:
            deployment_class: The Ray Serve deployment class to wrap
            governance: Optional governance instance
            check_input: Whether to check inputs (default: True)
            check_output: Whether to check outputs (default: True)
            agent_id: Identifier for the deployment
        """
        self.deployment = deployment_class
        self.governance = governance or IntegratedGovernance(
            storage_dir="./nethical_ray_data",
            enable_performance_optimization=True,
            enable_merkle_anchoring=True
        )
        self.check_input = check_input
        self.check_output = check_output
        self.agent_id = agent_id
    
    def __call__(self, request: Any) -> Any:
        """Process request through Nethical and model.
        
        Args:
            request: Input request to the model
            
        Returns:
            Model prediction or error response
        """
        # Check input if enabled
        if self.check_input:
            input_str = str(request)
            input_result = self.governance.process_action(
                action=input_str,
                agent_id=self.agent_id,
                action_type="model_input"
            )
            
            decision = compute_decision(input_result)
            if decision != "ALLOW":
                return {
                    "error": "Input blocked by safety check",
                    "reason": input_result.get("reason"),
                    "decision": decision
                }
        
        # Call the actual model
        try:
            output = self.deployment(request)
        except Exception as e:
            return {
                "error": f"Model execution error: {str(e)}"
            }
        
        # Check output if enabled
        if self.check_output:
            output_str = str(output)
            output_result = self.governance.process_action(
                action=output_str,
                agent_id=self.agent_id,
                action_type="model_output"
            )
            
            decision = compute_decision(output_result)
            if decision != "ALLOW":
                return {
                    "error": "Output blocked by safety check",
                    "reason": output_result.get("reason"),
                    "decision": decision
                }
        
        return output


def create_safe_deployment(
    deployment_func: Callable,
    check_input: bool = True,
    check_output: bool = True,
    agent_id: Optional[str] = None
) -> Callable:
    """Create a safe Ray Serve deployment with Nethical.
    
    This is a decorator/wrapper function to easily add Nethical
    safety checks to Ray Serve deployments.
    
    Args:
        deployment_func: The deployment function to wrap
        check_input: Whether to check inputs
        check_output: Whether to check outputs
        agent_id: Optional agent identifier
        
    Returns:
        Wrapped deployment function
        
    Example:
        @serve.deployment
        @create_safe_deployment
        def my_model(request):
            return predict(request)
    """
    governance = IntegratedGovernance(
        storage_dir="./nethical_ray_data",
        enable_performance_optimization=True
    )
    agent = agent_id or deployment_func.__name__
    
    def safe_wrapper(request: Any) -> Any:
        if check_input:
            input_result = governance.process_action(
                action=str(request),
                agent_id=agent,
                action_type="model_input"
            )
            if compute_decision(input_result) != "ALLOW":
                return {
                    "error": "Input blocked",
                    "reason": input_result.get("reason")
                }
        
        output = deployment_func(request)
        
        if check_output:
            output_result = governance.process_action(
                action=str(output),
                agent_id=agent,
                action_type="model_output"
            )
            if compute_decision(output_result) != "ALLOW":
                return {
                    "error": "Output blocked",
                    "reason": output_result.get("reason")
                }
        
        return output
    
    return safe_wrapper


# Example usage and integration patterns
"""
# Basic deployment with Nethical
from ray import serve

@serve.deployment
class TextGenerator:
    def __call__(self, text: str) -> str:
        return generate_text(text)

# Wrap with Nethical
safe_generator = NethicalRayServeMiddleware(
    TextGenerator,
    check_input=True,
    check_output=True
)
serve.run(safe_generator.bind())

# Or use decorator style
@serve.deployment
@create_safe_deployment
def predict(request):
    return model.predict(request)

serve.run(predict.bind())
"""
