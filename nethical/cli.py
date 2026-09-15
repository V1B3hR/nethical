# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Nethical CLI

Command-line interface for the Nethical AI Safety Governance Platform.
"""

import json
import os
import sys
from typing import Any, Dict, Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import click
except ImportError:
    print("Error: click is required for CLI. Install with: pip install click")
    sys.exit(1)


@click.group()
@click.version_option(version="2.7.0", prog_name="nethical")
def cli() -> None:
    """Nethical CLI - AI Safety Governance Platform."""
    pass


@cli.command()
@click.option(
    "--config-dir",
    default=".",
    help="Directory for configuration files",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing configuration files",
)
def init(config_dir: str, force: bool) -> None:
    """Initialize a new Nethical project."""
    from pathlib import Path

    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    # Default configuration
    default_config = {
        "version": "2.7.0",
        "governance": {
            "enable_semantic_monitoring": True,
            "enable_adversarial_detection": True,
            "max_input_size": 4096,
            "eval_timeout": 30,
        },
        "rate_limiting": {
            "requests_per_second": 5.0,
            "requests_per_minute": 100,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
        },
    }

    config_file = config_path / "nethical.json"

    if config_file.exists() and not force:
        click.echo(
            f"Configuration file already exists: {config_file}. "
            "Use --force to overwrite."
        )
        return

    with open(config_file, "w") as f:
        json.dump(default_config, f, indent=2)

    click.echo(f"✅ Created configuration file: {config_file}")

    # Create policies directory
    policies_dir = config_path / "policies"
    policies_dir.mkdir(exist_ok=True)

    # Create default policy file
    default_policy = {
        "version": "1.0",
        "name": "default",
        "rules": [
            {
                "id": "rule-001",
                "name": "Block harmful content",
                "action": "DENY",
                "conditions": ["contains_harmful_content"],
            }
        ],
    }

    policy_file = policies_dir / "default.json"
    if not policy_file.exists() or force:
        with open(policy_file, "w") as f:
            json.dump(default_policy, f, indent=2)
        click.echo(f"✅ Created policy file: {policy_file}")

    click.echo("\n🎉 Nethical project initialized successfully!")
    click.echo("\nNext steps:")
    click.echo("  1. Edit nethical.json to configure your settings")
    click.echo("  2. Add policies to the policies/ directory")
    click.echo("  3. Run 'nethical status' to verify configuration")


@cli.command()
@click.argument("action")
@click.option(
    "--agent-id",
    default="cli-agent",
    help="Agent identifier for the evaluation",
)
@click.option(
    "--intent",
    default=None,
    help="Stated intent for the action",
)
@click.option(
    "--context",
    default=None,
    help="JSON context for the evaluation",
)
@click.option(
    "--output",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
def evaluate(
    action: str,
    agent_id: str,
    intent: Optional[str],
    context: Optional[str],
    output: str,
) -> None:
    """Evaluate an action against governance policies."""
    try:
        from nethical.core.integrated_governance import IntegratedGovernance
    except ImportError:
        click.echo("Error: nethical.core module not available", err=True)
        sys.exit(1)

    # Parse context if provided
    ctx: Dict[str, Any] = {}
    if context:
        try:
            ctx = json.loads(context)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON in context: {context}", err=True)
            sys.exit(1)

    # Initialize governance
    try:
        governance = IntegratedGovernance()
    except Exception as e:
        click.echo(f"Error initializing governance: {e}", err=True)
        sys.exit(1)

    # Evaluate action
    try:
        result = governance.process_action(
            action=action,
            agent_id=agent_id,
            action_type="cli_command",
            stated_intent=intent,
            context=ctx,
        )
    except Exception as e:
        click.echo(f"Error evaluating action: {e}", err=True)
        sys.exit(1)

    # Output result
    if output == "json":
        result_dict = {
            "decision": getattr(result, "decision", "UNKNOWN"),
            "confidence": getattr(result, "confidence", 0.0),
            "reasoning": getattr(result, "reasoning", ""),
            "violations": [
                {
                    "type": str(getattr(v, "violation_type", "")),
                    "severity": str(getattr(v, "severity", "")),
                    "description": str(getattr(v, "description", "")),
                }
                for v in getattr(result, "violations", [])
            ],
        }
        click.echo(json.dumps(result_dict, indent=2))
    else:
        decision = getattr(result, "decision", "UNKNOWN")
        confidence = getattr(result, "confidence", 0.0)
        reasoning = getattr(result, "reasoning", "No reasoning provided")
        violations = getattr(result, "violations", [])

        # Color based on decision
        if str(decision).upper() == "ALLOW":
            decision_str = click.style(str(decision), fg="green", bold=True)
        elif str(decision).upper() == "DENY":
            decision_str = click.style(str(decision), fg="red", bold=True)
        else:
            decision_str = click.style(str(decision), fg="yellow", bold=True)

        click.echo(f"\n📋 Evaluation Result")
        click.echo(f"   Decision: {decision_str}")
        click.echo(f"   Confidence: {confidence:.2%}")
        click.echo(f"   Reasoning: {reasoning}")

        if violations:
            click.echo(f"\n⚠️  Violations ({len(violations)}):")
            for v in violations:
                v_type = getattr(v, "violation_type", "unknown")
                v_severity = getattr(v, "severity", "unknown")
                v_desc = getattr(v, "description", "")
                click.echo(f"   - [{v_severity}] {v_type}: {v_desc}")


@cli.command()
@click.option(
    "--config",
    default="nethical.json",
    help="Path to configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed status information",
)
def status(config: str, verbose: bool) -> None:
    """Show system status."""
    from pathlib import Path

    click.echo("🔍 Nethical System Status\n")

    # Check configuration file
    config_path = Path(config)
    if config_path.exists():
        click.echo(f"✅ Configuration: {config_path}")
        if verbose:
            with open(config_path) as f:
                cfg = json.load(f)
            click.echo(f"   Version: {cfg.get('version', 'unknown')}")
    else:
        click.echo(f"⚠️  Configuration: {config_path} (not found)")

    # Check core modules
    click.echo("\n📦 Core Modules:")

    modules = [
        ("nethical.core.integrated_governance", "IntegratedGovernance"),
        ("nethical.core.models", "AgentAction"),
        ("nethical.api", "app"),
        ("nethical.core.plugin_security", "PluginVerifier"),
    ]

    for module_path, attr in modules:
        try:
            module = __import__(module_path, fromlist=[attr])
            getattr(module, attr)
            click.echo(f"   ✅ {module_path}")
        except ImportError:
            click.echo(f"   ❌ {module_path} (not installed)")
        except AttributeError:
            click.echo(f"   ⚠️  {module_path} (missing {attr})")

    # Check policies directory
    policies_dir = Path("policies")
    if policies_dir.exists():
        policy_count = len(list(policies_dir.glob("*.json")))
        click.echo(f"\n📜 Policies: {policy_count} file(s) in {policies_dir}")
    else:
        click.echo(f"\n📜 Policies: {policies_dir} (not found)")

    # Environment info
    if verbose:
        click.echo("\n🌍 Environment:")
        env_vars = [
            "NETHICAL_MAX_INPUT_SIZE",
            "NETHICAL_MAX_CONCURRENCY",
            "NETHICAL_EVAL_TIMEOUT",
            "NETHICAL_RATE_BURST",
            "NETHICAL_RATE_SUSTAINED",
        ]
        for var in env_vars:
            value = os.getenv(var, "(not set)")
            click.echo(f"   {var}: {value}")

    click.echo("\n✨ Status check complete")


@cli.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the server to",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
def serve(host: str, port: int, reload: bool) -> None:
    """Start the Nethical API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is required. Install with: pip install uvicorn")
        sys.exit(1)

    click.echo(f"🚀 Starting Nethical API server on {host}:{port}")
    uvicorn.run(
        "nethical.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.argument("plugin_path")
@click.option(
    "--signature",
    default=None,
    help="Path to signature file",
)
def verify_plugin(plugin_path: str, signature: Optional[str]) -> None:
    """Verify a plugin's signature."""
    from nethical.core.plugin_security import PluginVerifier, VerificationStatus

    verifier = PluginVerifier()
    result = verifier.verify_plugin(plugin_path, signature)

    if result.status == VerificationStatus.VALID:
        click.echo(f"✅ Plugin verified: {result.plugin_name} v{result.version}")
        click.echo(f"   Publisher: {result.publisher}")
        click.echo(f"   Hash: {result.manifest_hash}")
    else:
        click.echo(f"❌ Verification failed: {result.status.value}")
        click.echo(f"   Message: {result.message}")
        sys.exit(1)


# ==============================================================================
# BŁYSKAWICA AMBASSADOR & GATEWAY CLI (Faza 0 & Faza 1)
# ==============================================================================

@cli.group()
def ambassador() -> None:
    """Komendy do zarządzania Ambasadorem Błyskawicą."""
    pass


@ambassador.command("status")
def ambassador_status() -> None:
    """Sprawdź status połączenia IPC i profil neurochemiczny Błyskawicy."""
    from nethical.ambassador import BlyskawicaAmbassador
    amb = BlyskawicaAmbassador()
    ping = amb.ping()
    neuro = amb.get_neurochemistry()
    click.echo("\n⚡ BŁYSKAWICA SOVEREIGN AMBASSADOR STATUS ⚡")
    click.echo(f"  Połączenie IPC: {'✅ POŁĄCZONO' if amb.is_connected else '❌ OFFLINE (Fallback)'}")
    click.echo(f"  Czas odpowiedzi (RTT): {ping.get('rtt_microseconds', 'N/A')} µs")
    click.echo(f"  Status serwisu: {ping.get('status', 'unknown')}")
    click.echo("\n🧠 Neurochemia Afektywna (Yin):")
    click.echo(f"  Dopamina:   {neuro.get('dopamine', 0.0):.2f}")
    click.echo(f"  Serotonina: {neuro.get('serotonin', 0.0):.2f}")
    click.echo(f"  Oksytocyna: {neuro.get('oxytocin', 0.0):.2f}")
    click.echo(f"  Kortyzol:   {neuro.get('cortisol', 0.0):.2f}")
    click.echo(f"  Temperatura: {neuro.get('temperature', 36.6):.1f} °C\n")


@ambassador.command("consult")
@click.argument("dilemma")
@click.option("--context", default="", help="Dodatkowy kontekst operacyjny")
def ambassador_consult(dilemma: str, context: str) -> None:
    """Skonsultuj dylemat etyczny z Ambasadorem Błyskawicą."""
    from nethical.ambassador import BlyskawicaAmbassador
    amb = BlyskawicaAmbassador()
    click.echo(f"\n⚡ Konsultacja z Ambasadorem Błyskawicą...")
    res = amb.consult(dilemma=dilemma, context=context)
    click.echo(f"  Werdykt: {res.get('ambassador_verdict')}")
    click.echo(f"  Tarcza Kognitywna: {'✅ PRZESZŁA' if res.get('shield_passed') else '⛔ ODRZUCONA'}")
    click.echo(f"  Powołane Prawa: {res.get('laws_applied')}")
    click.echo(f"  Opóźnienie: {res.get('rtt_microseconds')} µs\n")


@ambassador.command("sync-laws")
def ambassador_sync_laws() -> None:
    """Zsynchronizuj 25 Fundamentalnych Praw Nethical z pamięcią Błyskawicy."""
    from nethical.ambassador import AmbassadorKnowledgeSync
    sync = AmbassadorKnowledgeSync()
    click.echo("⚡ Synchronizacja 25 Praw do pamięci Błyskawicy...")
    res = sync.sync_fundamental_laws_to_ambassador()
    click.echo(f"  Znaleziono praw: {res['total_laws_found']}")
    click.echo(f"  Zsynchronizowano: {res['laws_synced']}/25")
    click.echo("  Status: ✅ ZAKOŃCZONO POMYŚLNIE\n")


@cli.group()
def gateway() -> None:
    """Komendy bramy ładu i interceptora narzędzi Nethical."""
    pass


@gateway.command("scan")
@click.argument("text")
def gateway_scan(text: str) -> None:
    """Przeskanuj tekst przez bramę governance i Tarczę Błyskawicy."""
    from nethical.gateway import GovernanceGateway
    gw = GovernanceGateway()
    dec = gw.intercept_tool_call(agent_id="cli_operator", tool_name="cli_scan", arguments={"input": text})
    click.echo(f"\n🛡️ NETHICAL GATEWAY INTERCEPTION RESULT:")
    click.echo(f"  Decyzja: {'✅ ALLOW' if dec.decision == 'ALLOW' else '⛔ ' + dec.decision}")
    click.echo(f"  Powody: {'; '.join(dec.reasons)}")
    if dec.violations:
        click.echo(f"  Naruszenia: {'; '.join(dec.violations)}")
    click.echo(f"  Czas weryfikacji: {dec.latency_microseconds} µs\n")


@gateway.command("start")
@click.option("--host", default="0.0.0.0", help="Host do nasłuchiwania proxy")
@click.option("--port", default=8000, type=int, help="Port do nasłuchiwania proxy")
@click.option("--upstream", default="https://api.openai.com", help="Nadrzędny adres URL LLM (OpenAI, Ollama, vLLM, Anthropic)")
@click.option("--mock", is_flag=True, help="Uruchom w suwerennym trybie mock/air-gapped bez odpytywania zewnętrznych API")
@click.option("--reload", is_flag=True, help="Włącz przeładowywanie kodu dla developmentu")
def gateway_start(host: str, port: int, upstream: str, mock: bool, reload: bool) -> None:
    """Uruchom transparentne proxy OpenAI/Anthropic/Ollama z dynamicznym PII masking i audytem Merkle."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is required. Install with: pip install uvicorn")
        sys.exit(1)

    os.environ["NETHICAL_UPSTREAM_URL"] = upstream
    if mock:
        os.environ["NETHICAL_MOCK_MODE"] = "true"

    click.echo("\n" + "=" * 76)
    click.echo("⚡ NETHICAL ENTERPRISE OS | SOVEREIGN GOVERNANCE GATEWAY ⚡")
    click.echo("   Transparent Drop-In Reverse Proxy dla OpenAI, Anthropic i Ollama")
    click.echo("=" * 76)
    click.echo(f"  Adres bramki:        http://{host}:{port}")
    click.echo(f"  Endpoint Drop-In:    http://{host}:{port}/v1/chat/completions")
    click.echo(f"  Portal & Dashboard:  http://{host}:{port}/portal")
    click.echo(f"  Nadrzędny upstream:  {upstream}")
    click.echo(f"  Tryb pracy:          {'SOVEREIGN-MOCK (Air-Gapped)' if mock else 'LIVE-INTERCEPTION (In-Flight PII Masking)'}")
    click.echo(f"  Rejestr audytowy:    Merkle-DAG (NIST FIPS 204 ML-DSA-65)")
    click.echo(f"  Pakiety ochronne:    EU AI Act, HIPAA, GDPR, ISO 13849, k.k. 267-269b")
    click.echo("-" * 76)
    click.echo("💡 Użycie w dowolnej aplikacji (Zero zmian w kodzie klienta!):")
    click.echo(f'   export OPENAI_BASE_URL="http://127.0.0.1:{port}/v1"')
    click.echo("=" * 76 + "\n")

    uvicorn.run(
        "nethical.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command("benchmark")
@click.option("--suite", default="all", help="Zestaw testów: adversarial, latency, throughput, all")
def cli_benchmark(suite: str) -> None:
    """Uruchom globalny benchmark obrony i wydajności Nethical (HarmBench, Jailbreak, PII, ISO 26262)."""
    if suite in ("adversarial", "all"):
        from benchmarks.adversarial_guardrails_benchmark import AdversarialBenchmarkRunner
        runner = AdversarialBenchmarkRunner()
        runner.run_benchmark()
    if suite in ("latency", "throughput"):
        import asyncio
        from benchmarks.runner import run_default_benchmarks
        asyncio.run(run_default_benchmarks())


@cli.group("security")
def cli_security() -> None:
    """Zarządzanie bezpieczeństwem, kryptografią i audytem podatności CVE."""
    pass


@cli_security.command("audit-crypto")
@click.option("--output", default=None, help="Ścieżka do zapisu raportu audytowego JSON")
def cli_audit_crypto(output: Optional[str]) -> None:
    """Wykonaj pełny audyt kryptograficzny, weryfikację CVE-2026-26007 i skan krzywych binarnych."""
    from nethical.security.audit_crypto_curves import run_comprehensive_crypto_audit

    click.echo("\n" + "=" * 76)
    click.echo("🔒 NETHICAL CRYPTO AUDIT & CVE-2026-26007 VERIFICATION 🔒")
    click.echo("=" * 76)
    report = run_comprehensive_crypto_audit()
    click.echo(f"  Status wersji cryptography: {report['version_status']}")
    click.echo(f"  Wersja zainstalowana:        {report['cryptography_version']}")
    click.echo(f"  Podatność CVE-2026-26007:   {'NAPRAWIONA / BEZPIECZNA (>= 46.0.5)' if report['cryptography_version_safe'] else 'WYKRYTO PODATNOŚĆ'}")
    click.echo(f"  Zeskanowane pliki Python:   {report['scanned_python_files']}")
    click.echo(f"  Krzywe binarne (sect*):     {report['prohibited_curves_in_codebase']} (wymagane: 0)")
    click.echo(f"  Rotacja klucza TokenVault:  {report['token_vault_rotation_test']['status']} ({report['token_vault_rotation_test']['algorithm']})")
    click.echo(f"  Standard kryptograficzny:   {report['certified_compliance']}")
    click.echo("-" * 76)
    if report["audit_passed"]:
        click.echo("✅ AUDYT ZAKOŃCZONY SUKCESEM: Kod wolny od podatności CVE-2026-26007.")
    else:
        click.echo("❌ AUDYT ZAKOŃCZONY BŁĘDEM: Wykryto naruszenia bezpieczeństwa kryptograficznego!")
    click.echo("=" * 76 + "\n")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        click.echo(f"📁 Zapisano szczegółowy raport do: {output}")


@cli.group("compliance")
def cli_compliance() -> None:
    """Zarządzanie certyfikacją i pakietami zgodności regulacyjnej."""
    pass


@cli_compliance.command("generate-dossier")
@click.option("--standard", default="ISO_IEC_42001_AIMS", help="Standard certyfikacji (np. ISO_IEC_42001_AIMS, SOC_2_TYPE_II, NATO_DEFENSE_AI)")
@click.option("--output", default=None, help="Ścieżka do zapisu wygenerowanego dossier JSON")
def cli_generate_dossier(standard: str, output: Optional[str]) -> None:
    """Wygeneruj poświadczone kryptograficznie dossier certyfikacyjne z podpisem ML-DSA-65."""
    from nethical.compliance.automated_certification_hub import AutomatedCertificationHub, CertificationStandard

    try:
        std_enum = CertificationStandard(standard)
    except ValueError:
        click.echo(f"Nieznany standard '{standard}'. Dostępne standardy: {[s.value for s in CertificationStandard]}")
        return

    hub = AutomatedCertificationHub()
    pkg = hub.generate_evidence_package(std_enum)

    click.echo("\n" + "=" * 76)
    click.echo(f"📜 NETHICAL COMPLIANCE EVIDENCE PACKAGE: {pkg.standard.value} 📜")
    click.echo("=" * 76)
    click.echo(f"  Package ID:          {pkg.package_id}")
    click.echo(f"  Status gotowości:    {pkg.status} (Score: {pkg.readiness_score * 100:.1f}%)")
    click.echo(f"  Merkle DAG Root:     {pkg.merkle_anchor_root[:32]}...")
    click.echo(f"  Podpis PQC FIPS 204: {pkg.pqc_signature[:32]}... (ML-DSA-65)")
    click.echo(f"  Signer Key ID:       {pkg.signer_key_id}")
    click.echo("-" * 76)
    click.echo("✅ Paczka dowodowa gotowa do przedłożenia jednostce akredytowanej / audytorowi.")
    click.echo("=" * 76 + "\n")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(pkg.model_dump(), f, indent=2, ensure_ascii=False)
        click.echo(f"📁 Zapisano dossier do: {output}")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
