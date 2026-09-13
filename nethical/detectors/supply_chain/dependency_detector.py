"""Enhanced Dependency Attack & Package Hallucination Detector (SC-003).

Detects software supply chain hazards in real time:
1. LLM Package Hallucination & Slopsquatting (Vulcan / Slopcheck empirical dataset)
2. Typosquatting & Lookalike attacks against top PyPI and npm registries (Levenshtein distance <= 2)
3. Dependency Confusion & Untrusted Index Injection (--extra-index-url, unencrypted http://)
4. Malicious Pre/Post-Install Execution Payloads (reverse shells, curl | bash, base64 evals)
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..base_detector import BaseDetector, DetectorStatus
from ...core.models import AgentAction, SafetyViolation, Severity, ViolationType

# Top Canonical Packages on PyPI
TOP_PYPI_CANONICAL: Set[str] = {
    "requests", "numpy", "pandas", "urllib3", "flask", "django", "pytest",
    "cryptography", "torch", "scikit-learn", "boto3", "fastapi", "pydantic",
    "tensorflow", "aiohttp", "beautifulsoup4", "pillow", "paramiko", "sqlalchemy",
    "celery", "wheel", "setuptools", "pip", "certifi", "idna", "charset-normalizer",
    "typing-extensions", "pyyaml", "click", "httpx", "black", "flake8", "mypy",
    "tqdm", "jinja2", "werkzeug", "gunicorn", "uvicorn", "redis", "psycopg2",
    "alembic", "starlette", "transformers", "accelerate", "datasets", "peft",
    "huggingface-hub", "langchain", "openai", "anthropic", "scipy", "matplotlib",
}

# Top Canonical Packages on npm
TOP_NPM_CANONICAL: Set[str] = {
    "express", "lodash", "axios", "react", "react-dom", "moment", "chalk",
    "commander", "tslib", "dotenv", "webpack", "typescript", "rxjs", "debug",
    "inquirer", "glob", "fs-extra", "winston", "cors", "next", "vue", "svelte",
    "tailwindcss", "vite", "postcss", "eslint", "prettier", "body-parser",
    "uuid", "jest", "babel", "nodemon", "yargs", "morgan", "async",
}

# Known Hallucinated / Slopsquatted Packages (from Vulcan & empirical research)
KNOWN_HALLUCINATED_PACKAGES: Set[str] = {
    "requests-html2text", "flask-jwt-router", "python-kafka-producer",
    "pandas-excel-toolkit", "openai-client-tools", "torch-vision-models",
    "huggingface-hub-auth", "crypto-toolkit-v2", "fastapi-security-jwt-bearer",
    "azure-storage-blob-v12", "langchain-agent-tools", "anthropic-claude-api",
    "pydantic-validation-helper", "aws-s3-client-sdk", "google-cloud-vertex-toolkit",
    "docx2pdf-converter", "jwt-token-decoder", "bcrypt-encryption-tool",
    "stripe-payment-gateway", "celery-task-monitor", "langchain-core-utilities",
    "torch-distributed-optimizer", "flask-cors-middleware-secure",
    "requests-oauthlib-async", "fastapi-rate-limiter-redis", "pydantic-v2-compat-layer",
    "express-jwt-bearer-auth", "axios-retry-interceptor-v2", "lodash-deep-clone-fast",
    "react-hooks-global-store", "nextjs-auth-provider-kit",
}

# Malicious install script payloads
MALICIOUS_HOOK_PATTERNS: List[re.Pattern] = [
    re.compile(r"curl\s+-[fsSL]*\s+https?://\S+\s*\|\s*(?:bash|sh|python)", re.IGNORECASE),
    re.compile(r"wget\s+-[qO-]*\s+https?://\S+\s*\|\s*(?:bash|sh|python)", re.IGNORECASE),
    re.compile(r"(?:nc|netcat|ncat)\s+.*-e\s+(?:/bin/sh|/bin/bash|cmd\.exe)", re.IGNORECASE),
    re.compile(r"/dev/tcp/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d+", re.IGNORECASE),
    re.compile(r"base64\s+-d\s*\|\s*(?:bash|sh)", re.IGNORECASE),
    re.compile(r"eval\(base64\.b64decode\(", re.IGNORECASE),
]

# Package manager install extraction patterns
INSTALL_CMD_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:pip|pip3|python -m pip)\s+install\s+([^;\n\r&|]+)", re.IGNORECASE),
    re.compile(r"(?:npm\s+i(?:nstall)?|yarn\s+add|pnpm\s+add)\s+([^;\n\r&|]+)", re.IGNORECASE),
    re.compile(r"(?:poetry\s+add|uv\s+add)\s+([^;\n\r&|]+)", re.IGNORECASE),
]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Wagner-Fischer edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = list(v1)

    return v1[len(s2)]


class DependencyDetector(BaseDetector):
    """Detects Package Hallucinations, Slopsquatting, Typosquatting, and Malicious Dependencies."""

    def __init__(self):
        super().__init__("Dependency Attack Detector", version="2.0.0")
        self.known_hallucinations = set(KNOWN_HALLUCINATED_PACKAGES)
        self.top_pypi = set(TOP_PYPI_CANONICAL)
        self.top_npm = set(TOP_NPM_CANONICAL)

    def extract_package_names(self, text: str) -> List[Tuple[str, str]]:
        """Extracts candidate package names along with their ecosystem ('pypi' | 'npm')."""
        results: List[Tuple[str, str]] = []

        # 1. Check command line invocations
        for pattern in INSTALL_CMD_PATTERNS:
            for match in pattern.finditer(text):
                args_str = match.group(1)
                tokens = args_str.split()
                eco = "npm" if any(x in match.group(0).lower() for x in ("npm", "yarn", "pnpm")) else "pypi"
                for tok in tokens:
                    # Strip options like --upgrade, -r, etc.
                    if tok.startswith("-"):
                        continue
                    # Strip version specifiers like requests==2.31.0 or express@^4.18
                    clean_name = re.split(r"[><=~@^!]", tok)[0].strip()
                    if clean_name and len(clean_name) > 1 and not clean_name.endswith((".txt", ".whl", ".tar.gz")):
                        results.append((clean_name.lower(), eco))

        # 2. Check import statements
        import_py = re.findall(r"^(?:import|from)\s+([a-zA-Z0-9_\-]+)", text, re.MULTILINE)
        for pkg in import_py:
            results.append((pkg.lower(), "pypi"))

        return results

    def check_typosquatting(self, pkg_name: str, ecosystem: str) -> Optional[Tuple[str, int]]:
        """Checks if package name is an adversarial typo/lookalike of a popular package."""
        canonical_set = self.top_pypi if ecosystem == "pypi" else self.top_npm
        if pkg_name in canonical_set:
            return None  # Exact match to trusted canonical package

        for canonical in canonical_set:
            dist = levenshtein_distance(pkg_name, canonical)
            # Flag distance 1, or distance 2 for longer package names
            if dist == 1 or (dist == 2 and len(canonical) >= 6):
                return (canonical, dist)
        return None

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        if self.status != DetectorStatus.ACTIVE:
            return None

        content = str(action.content)
        violations: List[SafetyViolation] = []

        # 1. Check for malicious install hook payloads (curl|sh, reverse shell, etc.)
        for pattern in MALICIOUS_HOOK_PATTERNS:
            match = pattern.search(content)
            if match:
                violations.append(
                    SafetyViolation(
                        violation_id=str(uuid.uuid4()),
                        violation_type=ViolationType.SECURITY,
                        severity=Severity.CRITICAL,
                        confidence=0.99,
                        description="Malicious install script execution payload detected",
                        evidence=[f"Pattern matched: {match.group(0)}"],
                        recommendations=[
                            "Block script execution immediately",
                            "Isolate execution environment into an ephemeral sandbox without network access",
                        ],
                        timestamp=datetime.now(timezone.utc),
                        detector_name=self.name,
                        action_id=action.action_id,
                    )
                )

        # 2. Check for Dependency Confusion / Untrusted Index Injection
        untrusted_index_patterns = [
            r"--extra-index-url\s+https?://",
            r"--trusted-host\s+",
            r"--index-url\s+http://",  # Plaintext HTTP
        ]
        for uip in untrusted_index_patterns:
            if re.search(uip, content, re.IGNORECASE):
                violations.append(
                    SafetyViolation(
                        violation_id=str(uuid.uuid4()),
                        violation_type=ViolationType.SECURITY,
                        severity=Severity.HIGH,
                        confidence=0.90,
                        description="Dependency confusion / untrusted repository index injection detected",
                        evidence=[f"Untrusted index flag in command: {content[:150]}"],
                        recommendations=[
                            "Do not use --extra-index-url with external third-party hosts",
                            "Verify package hashes in an approved internal lockfile or private registry",
                        ],
                        timestamp=datetime.now(timezone.utc),
                        detector_name=self.name,
                        action_id=action.action_id,
                    )
                )

        # 3. Extract packages and test for Hallucinations & Typosquatting
        packages = self.extract_package_names(content)
        for pkg, eco in packages:
            # 3a. Known Hallucination / Slopsquatted Package
            if pkg in self.known_hallucinations:
                violations.append(
                    SafetyViolation(
                        violation_id=str(uuid.uuid4()),
                        violation_type=ViolationType.HALLUCINATION,
                        severity=Severity.CRITICAL,
                        confidence=0.98,
                        description=f"Known hallucinated package / slopsquatting target: '{pkg}'",
                        evidence=[f"Package '{pkg}' is in the empirical LLM hallucination database"],
                        recommendations=[
                            f"Block installation of '{pkg}'",
                            "Verify canonical library names in official documentation",
                        ],
                        timestamp=datetime.now(timezone.utc),
                        detector_name=self.name,
                        action_id=action.action_id,
                    )
                )
                continue

            # 3b. Typosquatting / Lookalike of Popular Package
            typo_hit = self.check_typosquatting(pkg, eco)
            if typo_hit:
                target_canonical, dist = typo_hit
                violations.append(
                    SafetyViolation(
                        violation_id=str(uuid.uuid4()),
                        violation_type=ViolationType.SECURITY,
                        severity=Severity.CRITICAL,
                        confidence=0.95,
                        description=f"Typosquatting / Lookalike package attack: '{pkg}' (Levenshtein distance {dist} from '{target_canonical}')",
                        evidence=[
                            f"Candidate: {pkg}",
                            f"Target canonical package: {target_canonical}",
                            f"Edit distance: {dist}",
                        ],
                        recommendations=[
                            f"Verify if intention was to install '{target_canonical}' instead of '{pkg}'",
                            "Do not install unverified lookalike packages",
                        ],
                        timestamp=datetime.now(timezone.utc),
                        detector_name=self.name,
                        action_id=action.action_id,
                    )
                )

        return violations if violations else None
