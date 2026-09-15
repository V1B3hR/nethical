# Business Continuity & Disaster Recovery Plan (BCP / DRP)

**Document ID:** BCP-DRP-2026  
**Version:** 2.0  
**Effective Date:** 2026-09-15  
**Standards:** ISO/IEC 27001:2022 Annex A.5.29–A.5.30, SOC 2 Availability (A1.2), EU DORA (Regulation 2022/2554) Article 11  

---

## 1. Objectives & Metrics

This Business Continuity and Disaster Recovery Plan ensures that the Nethical AI Governance Platform can recover from catastrophic infrastructure disruptions, cyber incidents, or datacenter outages with minimal loss of service.

### Recovery Metrics Targets
- **Recovery Point Objective (RPO):** < 15 minutes (maximum permissible audit ledger data loss)
- **Recovery Time Objective (RTO):** < 1 hour (maximum permissible downtime for primary governance engine)
- **Degraded Fallback Mode:** Immediate (< 50ms local air-gapped node failover)

---

## 2. Incident Command & Roles

| Role | Responsibility | Alternate | Contact |
|---|---|---|---|
| **Incident Commander (IC)** | Directs overall disaster response and resource allocation | Deputy IC | `ops-ic@nethical.ai` |
| **Lead Infrastructure Engineer** | Executes server provisioning, DNS redirection, and container deployment | Senior SRE | PagerDuty Tier 1 |
| **Data Integrity Officer** | Verifies Merkle-DAG ledger continuity and cryptographic signatures | Crypto Lead | PagerDuty Tier 2 |
| **Communications Lead** | Manages external customer advisories and status page updates | Legal Lead | `status@nethical.ai` |

---

## 3. Disaster Scenarios & Recovery Procedures

### Scenario A: Primary Cloud Region / Datacenter Outage
1. **Detection:** Automated synthetic health checks trigger alert after 3 consecutive failures.
2. **Failover Execution:**
   - Automated DNS failover switches traffic to secondary active-standby region.
   - Secondary region mounts replicated SQLite WAL / PostgreSQL cluster.
3. **Verification:**
   - Health endpoint `/health` returns HTTP 200 OK.
   - Run cryptographic ledger sanity check (`nethical verify-ledger`).
4. **Resumption:** Route live API traffic; notify operators.

### Scenario B: Severe Network Partition (Air-Gapped Operation)
- If central governance nodes are severed from internet connectivity:
  - Local edge nodes switch automatically to **Autonomous Sovereign Mode**.
  - Decisions are evaluated against cached policy bundles and logged to local encrypted storage.
  - Upon network restoration, local Merkle trees are merged into the upstream DAG.

### Scenario C: Ransomware / Hostile Compromise
1. Isolate compromised cluster nodes immediately.
2. Restore compute images from verified SLSA-attested container images.
3. Restore database state from immutable WORM (Write Once, Read Many) S3 snapshots.
4. Verify root hashes against published out-of-band anchors.

---

## 4. Testing & Drill Frequency

- **Tabletop Exercise:** Semi-annually with security and leadership teams.
- **Failover Simulation:** Annual unannounced disaster recovery failover drill in staging/canary environment.
- **Backup Restoration Test:** Monthly automated restoration of ledger snapshots to an isolated sandbox.
