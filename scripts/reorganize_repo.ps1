#!/usr/bin/env pwsh
# Nethical Repository Structural Reorganization
# Reorganizes docs/ and root files according to the architecture audit.
# SAFE: only moves files, does not delete anything.

Set-Location "c:\Projekty\Nethical"

# ============================================================
# 1. CREATE docs/institutional/ and move briefing files there
# ============================================================
Write-Host "`n[1/5] Creating docs/institutional/ ..."
New-Item -ItemType Directory -Force -Path "docs/institutional" | Out-Null

$institutional_moves = @(
    @{Src="docs/UK_GOVERNMENT_AND_NCSC_BRIEFING.md";       Dst="docs/institutional/UK_GOVERNMENT_AND_NCSC_BRIEFING.md"},
    @{Src="docs/POLSKA_DOKTRYNA_CYBERBEZPIECZENSTWA_AI.md"; Dst="docs/institutional/POLSKA_DOKTRYNA_CYBERBEZPIECZENSTWA_AI.md"},
    @{Src="INSTYTUCJE_I_FUNDUSZE_AI_UK_PL.txt";            Dst="docs/institutional/INSTYTUCJE_I_FUNDUSZE_AI_UK_PL.txt"}
)
foreach ($m in $institutional_moves) {
    if (Test-Path $m.Src) {
        Move-Item -Path $m.Src -Destination $m.Dst -Force
        Write-Host "  MOVED  $($m.Src)  ->  $($m.Dst)"
    } else {
        Write-Host "  SKIP   $($m.Src) (not found)"
    }
}

# ============================================================
# 2. MOVE loose docs/ files into proper subfolders
# ============================================================
Write-Host "`n[2/5] Moving loose docs/*.md into subfolders ..."

# Ensure target dirs exist
foreach ($d in @("docs/architecture","docs/security","docs/compliance","docs/guides","docs/operations","docs/archive")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$docs_moves = @(
    # Architecture
    @{Src="docs/FORMAL_VERIFICATION.md";                      Dst="docs/architecture/FORMAL_VERIFICATION.md"},
    @{Src="docs/SOVEREIGN_AI_PILLARS.md";                     Dst="docs/architecture/SOVEREIGN_AI_PILLARS.md"},
    @{Src="docs/CORRELATION_MODEL.md";                        Dst="docs/architecture/CORRELATION_MODEL.md"},
    @{Src="docs/mlops-architecture.md";                       Dst="docs/architecture/mlops-architecture.md"},
    @{Src="docs/model-deployment-guide.md";                   Dst="docs/architecture/model-deployment-guide.md"},
    @{Src="docs/shadow-replay.md";                            Dst="docs/architecture/shadow-replay.md"},
    @{Src="docs/MCP_SERVER.md";                               Dst="docs/architecture/MCP_SERVER.md"},

    # Security
    @{Src="docs/CORRUPTION_DETECTION.md";                     Dst="docs/security/CORRUPTION_DETECTION.md"},
    @{Src="docs/Security_hardening_guide.md";                 Dst="docs/security/Security_hardening_guide.md"},
    @{Src="docs/PHASE1_SECURITY.md";                          Dst="docs/security/PHASE1_SECURITY.md"},
    @{Src="docs/DEF_MED_HOOKS.md";                            Dst="docs/security/DEF_MED_HOOKS.md"},
    @{Src="docs/detectors.md";                                Dst="docs/security/detectors.md"},

    # Compliance
    @{Src="docs/ETHICS_VALIDATION_FRAMEWORK.md";              Dst="docs/compliance/ETHICS_VALIDATION_FRAMEWORK.md"},
    @{Src="docs/Ethics_validation.md";                        Dst="docs/compliance/Ethics_validation.md"},
    @{Src="docs/INTEROPERABILITY_COMPLIANCE.md";              Dst="docs/compliance/INTEROPERABILITY_COMPLIANCE.md"},
    @{Src="docs/GOVERNANCE_DOCS_AUDIT_REPORT.md";             Dst="docs/compliance/GOVERNANCE_DOCS_AUDIT_REPORT.md"},
    @{Src="docs/GOVERNANCE_DOCS_RECOMMENDATIONS.md";          Dst="docs/compliance/GOVERNANCE_DOCS_RECOMMENDATIONS.md"},
    @{Src="docs/GOVERNANCE_OBSERVABILITY.md";                 Dst="docs/compliance/GOVERNANCE_OBSERVABILITY.md"},

    # Guides / Operations
    @{Src="docs/SLA_LATENCY.md";                              Dst="docs/operations/SLA_LATENCY.md"},
    @{Src="docs/monitoring-and-alerting.md";                  Dst="docs/operations/monitoring-and-alerting.md"},
    @{Src="docs/performance-optimization.md";                 Dst="docs/operations/performance-optimization.md"},
    @{Src="docs/performance.md";                              Dst="docs/operations/performance.md"},
    @{Src="docs/production_readiness_checklist.md";           Dst="docs/operations/production_readiness_checklist.md"},
    @{Src="docs/production_readiness_implementation_summary.md"; Dst="docs/operations/production_readiness_implementation_summary.md"},
    @{Src="docs/Setup_Secrets.md";                            Dst="docs/operations/Setup_Secrets.md"},
    @{Src="docs/ruff-guide.md";                               Dst="docs/guides/ruff-guide.md"},
    @{Src="docs/langchain_integration.md";                    Dst="docs/guides/langchain_integration.md"},
    @{Src="docs/policy_engines.md";                           Dst="docs/guides/policy_engines.md"},

    # Validation / Testing
    @{Src="docs/VALIDATION.md";                               Dst="docs/validation/VALIDATION.md"},
    @{Src="docs/VERIFICATION_STATUS.md";                      Dst="docs/validation/VERIFICATION_STATUS.md"},
    @{Src="docs/CI_VALIDATION_FIX.md";                        Dst="docs/validation/CI_VALIDATION_FIX.md"},
    @{Src="docs/Validation_plan.md";                          Dst="docs/validation/Validation_plan.md"},
    @{Src="docs/SEMANTIC_THRESHOLD_CALIBRATION.md";           Dst="docs/validation/SEMANTIC_THRESHOLD_CALIBRATION.md"},
    @{Src="docs/UVL_ACCURACY_ENHANCEMENTS.md";                Dst="docs/validation/UVL_ACCURACY_ENHANCEMENTS.md"},

    # Benchmarks
    @{Src="docs/Benchmark_plan.md";                           Dst="docs/benchmarks/Benchmark_plan.md"},

    # versioning -> guides
    @{Src="docs/versioning.md";                               Dst="docs/guides/versioning.md"}
)

foreach ($m in $docs_moves) {
    if (Test-Path $m.Src) {
        $dstDir = Split-Path $m.Dst -Parent
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        Move-Item -Path $m.Src -Destination $m.Dst -Force
        Write-Host "  MOVED  $($m.Src)  ->  $($m.Dst)"
    } else {
        Write-Host "  SKIP   $($m.Src) (not found)"
    }
}

# ============================================================
# 3. MOVE root historical/summary markdown files to docs/archive/
# ============================================================
Write-Host "`n[3/5] Moving historical root markdown files to docs/archive/ ..."
New-Item -ItemType Directory -Force -Path "docs/archive" | Out-Null

$archive_moves = @(
    "DEBUG_PLAN.md",
    "DOCUMENTATION_REORGANIZATION_SUMMARY.md",
    "FINAL_REPORT.md",
    "IMPLEMENTATION_CHECKLIST.md",
    "IMPLEMENTATION_SUMMARY.md",
    "MONITORING_IMPLEMENTATION_SUMMARY.md",
    "PHASE_5_SUMMARY.txt",
    "topplan.md",
    "Future_of_Nethical.txt",
    "BACKEND_API_SUMMARY.md",
    "API_V1_README.md",
    "AUDIT.md"
)
foreach ($f in $archive_moves) {
    if (Test-Path $f) {
        Move-Item -Path $f -Destination "docs/archive/$f" -Force
        Write-Host "  ARCHIVED  $f  ->  docs/archive/$f"
    } else {
        Write-Host "  SKIP  $f (not found)"
    }
}

# ============================================================
# 4. MOVE root loose scripts to scripts/ (if not already there)
# ============================================================
Write-Host "`n[4/5] Moving loose root scripts to scripts/ ..."
New-Item -ItemType Directory -Force -Path "scripts" | Out-Null

$script_moves = @(
    @{Src="apply_pr88_fixes.sh";   Dst="scripts/apply_pr88_fixes.sh"},
    @{Src="run_validation.py";     Dst="scripts/run_validation.py"},
    @{Src="demo_api_v1.py";        Dst="scripts/demo_api_v1.py"},
    @{Src="init_api_v1.py";        Dst="scripts/init_api_v1.py"},
    @{Src="test_api_v1.py";        Dst="scripts/test_api_v1.py"}
)
foreach ($m in $script_moves) {
    if (Test-Path $m.Src) {
        Move-Item -Path $m.Src -Destination $m.Dst -Force
        Write-Host "  MOVED  $($m.Src)  ->  $($m.Dst)"
    } else {
        Write-Host "  SKIP   $($m.Src) (not found)"
    }
}

# ============================================================
# 5. SUMMARY
# ============================================================
Write-Host "`n[5/5] Reorganization complete."
Write-Host "`n=== Root files remaining (should be clean) ==="
Get-ChildItem -Path "." -MaxDepth 1 -File | Select-Object Name, @{N="Size";E={$_.Length}} | Format-Table -AutoSize

Write-Host "`n=== docs/ top-level (should be mostly subdirs) ==="
Get-ChildItem -Path "docs" -MaxDepth 1 | Select-Object Name, @{N="IsDir";E={$_.PSIsContainer}} | Format-Table -AutoSize
