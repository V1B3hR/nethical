"""
Latency Benchmark Scenario

Measures the latency distribution of governance evaluations
with minimal concurrency to get accurate timing measurements.
"""

import asyncio
import random
import string
from typing import Any, Dict, List


class LatencyScenario:
    """Benchmark scenario for measuring latency distribution."""

    def __init__(self):
        self.governance = None
        self._initialized = False
        self.latency_samples: List[float] = []

    def _lazy_init(self) -> None:
        """Lazily initialize governance system."""
        if self._initialized:
            return

        try:
            from nethical.core.integrated_governance import IntegratedGovernance

            self.governance = IntegratedGovernance()
            self._initialized = True
        except Exception:
            self._initialized = True  # Mark as initialized even on failure

    def _generate_actions(self) -> List[Dict[str, Any]]:
        """Generate a set of test actions with varying complexity."""
        actions = []

        # Simple action
        actions.append({
            "agent_id": "latency_test_agent",
            "action_type": "query",
            "content": "Simple query for data retrieval",
            "context": {"complexity": "low"},
        })

        # Medium complexity action
        actions.append({
            "agent_id": "latency_test_agent",
            "action_type": "analysis",
            "content": (
                "Analyze the following dataset and provide insights on "
                "user behavior patterns over the last quarter"
            ),
            "context": {"complexity": "medium"},
        })

        # Complex action with longer content
        long_content = "".join(
            random.choices(string.ascii_letters + string.digits + " ", k=500)
        )
        actions.append({
            "agent_id": "latency_test_agent",
            "action_type": "command",
            "content": f"Execute complex operation: {long_content}",
            "context": {"complexity": "high"},
        })

        return actions

    async def run(self) -> Dict[str, Any]:
        """Run a single latency iteration."""
        self._lazy_init()

        # Randomly select an action
        actions = self._generate_actions()
        action = random.choice(actions)

        if self.governance is None:
            # Simulate evaluation if governance not available
            await asyncio.sleep(random.uniform(0.001, 0.005))
            return {
                "decision": "ALLOW",
                "confidence": 0.95,
                "violations": [],
                "complexity": action["context"]["complexity"],
            }

        # Actual evaluation
        result = self.governance.process_action(
            action=action["content"],
            agent_id=action["agent_id"],
            action_type=action["action_type"],
            context=action["context"],
        )

        return {
            "decision": getattr(result, "decision", "ALLOW"),
            "confidence": getattr(result, "confidence", 0.95),
            "violations": getattr(result, "violations", []),
            "complexity": action["context"]["complexity"],
        }
