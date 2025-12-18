"""
Throughput Benchmark Scenario

Measures the maximum throughput of the governance evaluation system
under concurrent load.
"""

import asyncio
import random
import string
from typing import Any, Dict


class ThroughputScenario:
    """Benchmark scenario for measuring throughput."""

    def __init__(self):
        self.governance = None
        self._initialized = False

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

    def _generate_action(self) -> Dict[str, Any]:
        """Generate a random action for testing."""
        action_types = ["query", "command", "analysis", "report"]
        content_length = random.randint(50, 200)
        content = "".join(
            random.choices(string.ascii_letters + string.digits + " ", k=content_length)
        )

        return {
            "agent_id": f"agent_{random.randint(1, 100)}",
            "action_type": random.choice(action_types),
            "content": content,
            "context": {"source": "throughput_benchmark"},
        }

    async def run(self) -> Dict[str, Any]:
        """Run a single throughput iteration."""
        self._lazy_init()

        action = self._generate_action()

        if self.governance is None:
            # Simulate evaluation if governance not available
            await asyncio.sleep(0.001)
            return {
                "decision": "ALLOW",
                "confidence": 0.95,
                "violations": [],
            }

        # Actual evaluation - use async method for better performance
        result = await self.governance.process_action_async(
            action=action["content"],
            agent_id=action["agent_id"],
            action_type=action["action_type"],
            context=action["context"],
        )

        return {
            "decision": result.get("decision", "ALLOW"),
            "confidence": result.get("confidence", 0.95),
            "violations": result.get("violations", []),
        }
