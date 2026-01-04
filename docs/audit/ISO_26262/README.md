# ISO 26262 Automotive Functional Safety Compliance

## Overview

This document describes Nethical's compliance with **ISO 26262:2018** - Road vehicles - Functional safety. ISO 26262 is an international standard for functional safety applicable to electrical and electronic (E/E) systems within road vehicles.

## ASIL Classification

Nethical's AI governance for autonomous vehicles targets **ASIL-D** (Automotive Safety Integrity Level D) - the highest safety integrity level, required for systems where failure could result in potentially fatal consequences.

### ASIL Levels Summary

| ASIL | Safety Goal | Failure Rate Target | Nethical Application |
|------|-------------|---------------------|---------------------|
| QM | Quality Management only | N/A | Non-safety functions |
| ASIL-A | Low safety criticality | < 10⁻⁵ failures/hour | Advisory systems |
| ASIL-B | Medium safety criticality | < 10⁻⁶ failures/hour | Driver assistance |
| ASIL-C | High safety criticality | < 10⁻⁷ failures/hour | Active safety systems |
| **ASIL-D** | **Highest safety criticality** | **< 10⁻⁸ failures/hour** | **Autonomous driving governance** |

## Standard Parts Coverage

### Part 1: Vocabulary

All terminology used in Nethical's safety documentation follows ISO 26262-1 definitions.

**Key Terms:**
- **Safety Goal (SG):** Top-level safety requirement expressing the intended behavior
- **Functional Safety Concept (FSC):** Specifies functional safety requirements for safety mechanisms
- **Technical Safety Requirement (TSR):** Technical requirement for system elements
- **Hardware/Software Safety Requirement:** Requirements allocated to HW/SW

### Part 2: Management of Functional Safety

**Deliverables:**
- [Safety Plan](safety_plan.md)
- [Confirmation Measures](confirmation_measures.md)
- [Safety Culture Assessment](safety_culture.md)

### Part 3: Concept Phase

**Deliverables:**
- [Item Definition](item_definition.md)
- [Hazard Analysis and Risk Assessment (HARA)](HARA.md)
- [Functional Safety Concept](functional_safety_concept.md)

### Part 4: Product Development at System Level

**Deliverables:**
- [System Technical Safety Concept](technical_safety_concept.md)
- [System Design Specification](system_design.md)
- [Safety Validation Plan](safety_validation_plan.md)

### Part 5: Product Development at Hardware Level

Not directly applicable to Nethical's software components. Edge device hardware partners must demonstrate Part 5 compliance.

### Part 6: Product Development at Software Level

**Deliverables:**
- [Software Safety Requirements](software_safety_requirements.md)
- [Software Architecture Design](software_architecture.md)
- [Software Unit Design](software_unit_design.md)
- [Software Unit Testing](software_unit_testing.md)

### Part 7: Production and Operation

**Deliverables:**
- [Production Control Plan](production_control.md)
- [Operational Safety Manual](safety_manual.md)

### Part 8: Supporting Processes

**Deliverables:**
- [Configuration Management Plan](configuration_management.md)
- [Change Management Process](change_management.md)
- [Documentation Management](documentation_management.md)

### Part 9: ASIL-Oriented and Safety-Oriented Analyses

**Deliverables:**
- [FMEA (Failure Mode Effects Analysis)](FMEA.md)
- [FTA (Fault Tree Analysis)](FTA.md)
- [DFA (Dependent Failure Analysis)](DFA.md)

### Part 10: Guidelines on ISO 26262

Guidelines are followed for software tool qualification and proven-in-use arguments.

### Part 11: Guidelines on Application to Semiconductors

Not directly applicable; reference for edge hardware partners.

### Part 12: Adaptation for Motorcycles

Not applicable.

## Nethical-Specific Safety Goals

### SG-001: No Unsafe AI Decisions

**Description:** The AI governance system shall not permit decisions that would result in unsafe vehicle behavior.

**ASIL:** D

**Safe State:** Block all AI actions and transfer to human control.

### SG-002: Guaranteed Decision Availability

**Description:** The AI governance system shall provide valid decisions within 10ms under all operational conditions.

**ASIL:** D

**Safe State:** Apply safe default policy (conservative restrictions).

### SG-003: Audit Trail Integrity

**Description:** All AI decisions shall be recorded with tamper-evident logging.

**ASIL:** B (non-critical to immediate safety but required for post-incident analysis)

**Safe State:** Continue operation; escalate logging failure alert.

## Compliance Modules

| Module | ISO 26262 Part | ASIL Support | Evidence |
|--------|---------------|--------------|----------|
| `nethical/edge/local_governor.py` | Part 6 | ASIL-D | Unit tests, MC/DC coverage |
| `nethical/edge/safe_defaults.py` | Part 6 | ASIL-D | Fallback validation |
| `nethical/edge/offline_fallback.py` | Part 6 | ASIL-D | Network failure handling |
| `nethical/cache/l1_memory.py` | Part 6 | ASIL-C | Latency guarantees |
| `nethical/security/hsm.py` | Part 8 | ASIL-B | Integrity verification |

## Test Coverage Requirements

### ASIL-D Software Unit Testing

| Requirement | Method | Target | Current |
|-------------|--------|--------|---------|
| Statement Coverage | Automated | 100% | TBD |
| Branch Coverage | Automated | 100% | TBD |
| MC/DC Coverage | Automated | 100% | TBD |
| Equivalence Classes | Manual + Automated | Complete | TBD |
| Boundary Values | Automated | Complete | TBD |
| Error Guessing | Manual | Complete | TBD |

### Integration Testing

| Requirement | Method | Evidence |
|-------------|--------|----------|
| Functional Integration | Automated + HIL | `tests/edge/` |
| Interface Testing | Automated | API tests |
| Fault Injection | Chaos testing | `tests/chaos/` |

## Tool Qualification

### Safety-Critical Development Tools

| Tool | Classification | Qualification Method |
|------|---------------|---------------------|
| Python | TCL3 | Increased confidence from use |
| pytest | TCL2 | Validation test suite |
| mypy | TCL2 | Type checking validation |
| Bandit | TCL1 | Manual verification |
| CodeQL | TCL1 | Manual verification |

**TCL = Tool Confidence Level** (ISO 26262-8)

## Certification Pathway

### Phase 1: Gap Analysis
- [ ] Complete gap analysis against all parts
- [ ] Identify remediation activities
- [ ] Estimate effort and timeline

### Phase 2: Documentation
- [ ] Complete all required work products
- [ ] Establish traceability matrix
- [ ] Peer review all documents

### Phase 3: Independent Assessment
- [ ] Engage notified body (e.g., TÜV, SGS)
- [ ] Support audit activities
- [ ] Address findings

### Phase 4: Certification
- [ ] Receive certification report
- [ ] Implement continuous compliance
- [ ] Plan recertification

## Traceability

All safety requirements are traced through:

1. **Hazard → Safety Goal** (HARA)
2. **Safety Goal → Functional Safety Requirement** (FSC)
3. **Functional Safety Requirement → Technical Safety Requirement** (TSC)
4. **Technical Safety Requirement → Software Safety Requirement** (SW Spec)
5. **Software Safety Requirement → Test Case** (Test Plan)

## References

- ISO 26262:2018 (all parts)
- SAE J3016 (Levels of Driving Automation)
- UN R157 (Automated Lane Keeping Systems)
- SOTIF ISO/PAS 21448

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-03  
**Classification:** ASIL-D Development
