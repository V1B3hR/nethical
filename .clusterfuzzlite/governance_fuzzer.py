#!/usr/bin/env python3
import sys
import atheris

with atheris.instrument_imports():
    from nethical.core.governance import EnhancedSafetyGovernance

gov = None

def TestOneInput(data):
    global gov
    if gov is None:
        try:
            gov = EnhancedSafetyGovernance(storage_dir='/tmp/fuzz_gov')
        except Exception:
            return

    fdp = atheris.FuzzedDataProvider(data)
    action_text = fdp.ConsumeUnicodeNoSurrogates(256)
    agent_id = fdp.ConsumeUnicodeNoSurrogates(64)

    try:
        gov.process_action(agent_id=agent_id, action=action_text, cohort='fuzz')
    except (ValueError, TypeError, KeyError):
        pass

def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()

if __name__ == '__main__':
    main()
