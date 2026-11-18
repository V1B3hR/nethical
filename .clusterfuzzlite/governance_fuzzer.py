#!/usr/bin/env python3
import asyncio
import sys
import atheris

with atheris.instrument_imports():
    from nethical.core.governance import (
        ActionType,
        AgentAction,
        EnhancedSafetyGovernance,
        MonitoringConfig,
    )

gov = EnhancedSafetyGovernance(
    MonitoringConfig(enable_persistence=False, enable_async_processing=False)
)
loop = asyncio.new_event_loop()

def TestOneInput(data):
    fdp = atheris.FuzzedDataProvider(data)
    action_text = fdp.ConsumeUnicodeNoSurrogates(256)

    action = AgentAction(
        action_id="fuzz",
        agent_id="fuzz-agent",
        action_type=ActionType.QUERY,
        content=action_text,
    )
    try:
        loop.run_until_complete(gov.evaluate_action(action, use_cache=False))
    finally:
        gov.metrics["detector_timing"].clear()


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()

if __name__ == '__main__':
    main()
