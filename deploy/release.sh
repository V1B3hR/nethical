#!/bin/bash
# Nethical Reproducible Release Script
# This script creates reproducible builds with cryptographic verification
# and generates all necessary attestations and SBOMs.
#
# Usage: ./release.sh <version>
# Example: ./release.sh v1.0.0

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSION="${1:-}"
BUILD_DIR="${PROJECT_ROOT}/build"
DIST_DIR="${PROJECT_ROOT}/dist"
SBOM_DIR="${PROJECT_ROOT}/sbom"
ATTESTATION_DIR="${PROJECT_ROOT}/attestations"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verify prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Required tools
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    command -v sha256sum >/dev/null 2>&1 || missing_tools+=("sha256sum")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    
    # Optional but recommended tools
    if ! command -v cosign >/dev/null 2>&1; then
        log_warn "cosign not found - artifact signing will be skipped"
    fi
    
    if ! command -v syft >/dev/null 2>&1; then
        log_warn "syft not found - SBOM generation may be limited"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools and try again"
        exit 1
    fi
    
    log_success "All required prerequisites found"
}

# Validate version format
validate_version() {
    if [ -z "${VERSION}" ]; then
        log_error "Version not specified"
        echo "Usage: $0 <version>"
        echo "Example: $0 v1.0.0"
        exit 1
    fi
    
    # Version should match semver format with optional 'v' prefix
    if ! echo "${VERSION}" | grep -qE '^v?[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.+-]+)?$'; then
        log_error "Invalid version format: ${VERSION}"
        log_error "Version must follow semver format: [v]MAJOR.MINOR.PATCH[-prerelease]"
        exit 1
    fi
    
    log_success "Version ${VERSION} validated"
}

# Clean build directories
clean_build_dirs() {
    log_info "Cleaning build directories..."
    
    rm -rf "${BUILD_DIR}"
    rm -rf "${DIST_DIR}"
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${DIST_DIR}"
    mkdir -p "${SBOM_DIR}"
    mkdir -p "${ATTESTATION_DIR}"
    
    log_success "Build directories cleaned"
}

# Create reproducible build environment
setup_build_env() {
    log_info "Setting up reproducible build environment..."
    
    # Set reproducible build environment variables
    export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
    export PYTHONHASHSEED=0
    export PYTHONDONTWRITEBYTECODE=1
    
    # Record build environment
    cat > "${BUILD_DIR}/build_env.json" <<EOF
{
  "version": "${VERSION}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_date_epoch": "${SOURCE_DATE_EPOCH}",
  "git_commit": "$(git rev-parse HEAD)",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
  "git_tag": "$(git describe --tags --exact-match 2>/dev/null || echo 'none')",
  "python_version": "$(python3 --version | cut -d' ' -f2)",
  "builder": "${USER}@$(hostname)",
  "build_os": "$(uname -s)",
  "build_arch": "$(uname -m)"
}
EOF
    
    log_success "Build environment configured"
}

# Verify dependencies
verify_dependencies() {
    log_info "Verifying dependencies..."
    
    cd "${PROJECT_ROOT}"
    
    # Check if requirements-hashed.txt exists
    if [ ! -f "requirements-hashed.txt" ]; then
        log_error "requirements-hashed.txt not found"
        log_error "Run: pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt"
        exit 1
    fi
    
    # Install dependencies with hash verification
    python3 -m pip install --require-hashes -r requirements-hashed.txt --quiet
    
    log_success "Dependencies verified and installed"
}

# Build package
build_package() {
    log_info "Building package..."
    
    cd "${PROJECT_ROOT}"
    
    # Build source distribution and wheel
    python3 -m pip install --quiet build
    python3 -m build --outdir "${DIST_DIR}"
    
    # Generate checksums
    cd "${DIST_DIR}"
    sha256sum * > SHA256SUMS
    
    log_success "Package built successfully"
    log_info "Artifacts in: ${DIST_DIR}"
    ls -lh "${DIST_DIR}"
}

# Generate SBOM in CycloneDX format
generate_sbom_cyclonedx() {
    log_info "Generating SBOM (CycloneDX format)..."
    
    cd "${PROJECT_ROOT}"
    
    # Install cyclonedx-bom if not present
    python3 -m pip install --quiet cyclonedx-bom
    
    # Generate CycloneDX SBOM
    cyclonedx-py -r -i requirements.txt -o "${SBOM_DIR}/sbom-cyclonedx.json" --format json
    cyclonedx-py -r -i requirements.txt -o "${SBOM_DIR}/sbom-cyclonedx.xml" --format xml
    
    log_success "CycloneDX SBOM generated"
}

# Generate SBOM in SPDX format
generate_sbom_spdx() {
    log_info "Generating SBOM (SPDX format)..."
    
    cd "${PROJECT_ROOT}"
    
    # Generate SPDX SBOM
    python3 -m pip install --quiet spdx-tools
    
    # Create basic SPDX document
    cat > "${SBOM_DIR}/sbom-spdx.json" <<EOF
{
  "spdxVersion": "SPDX-2.3",
  "dataLicense": "CC0-1.0",
  "SPDXID": "SPDXRef-DOCUMENT",
  "name": "nethical-${VERSION}",
  "documentNamespace": "https://github.com/V1B3hR/nethical/sbom/${VERSION}",
  "creationInfo": {
    "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "creators": ["Tool: nethical-release-script"],
    "licenseListVersion": "3.21"
  },
  "packages": []
}
EOF
    
    log_success "SPDX SBOM generated"
}

# Run vulnerability scanning
run_vulnerability_scan() {
    log_info "Running vulnerability scan..."
    
    cd "${PROJECT_ROOT}"
    
    # Use pip-audit for Python dependencies
    python3 -m pip install --quiet pip-audit
    
    # Run audit and save results
    if python3 -m pip-audit -r requirements.txt --format json > "${SBOM_DIR}/vulnerability-scan.json" 2>/dev/null; then
        log_success "Vulnerability scan completed - no issues found"
    else
        log_warn "Vulnerability scan found issues - check ${SBOM_DIR}/vulnerability-scan.json"
    fi
}

# Generate SLSA provenance
generate_slsa_provenance() {
    log_info "Generating SLSA provenance..."
    
    # Create SLSA provenance document (v1.0)
    cat > "${ATTESTATION_DIR}/slsa-provenance.json" <<EOF
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [
    {
      "name": "nethical",
      "digest": {
        "sha256": "$(cd "${DIST_DIR}" && sha256sum *.whl | head -1 | cut -d' ' -f1)"
      }
    }
  ],
  "predicateType": "https://slsa.dev/provenance/v1",
  "predicate": {
    "buildDefinition": {
      "buildType": "https://github.com/V1B3hR/nethical/release@v1",
      "externalParameters": {
        "version": "${VERSION}",
        "source": {
          "repository": "https://github.com/V1B3hR/nethical",
          "ref": "$(git rev-parse HEAD)"
        }
      },
      "internalParameters": {
        "SOURCE_DATE_EPOCH": "${SOURCE_DATE_EPOCH}",
        "PYTHONHASHSEED": "0"
      },
      "resolvedDependencies": []
    },
    "runDetails": {
      "builder": {
        "id": "$(hostname)"
      },
      "metadata": {
        "invocationId": "$(uuidgen 2>/dev/null || echo 'manual-build')",
        "startedOn": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "finishedOn": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      }
    }
  }
}
EOF
    
    log_success "SLSA provenance generated"
}

# Sign artifacts with cosign (if available)
sign_artifacts_cosign() {
    if ! command -v cosign >/dev/null 2>&1; then
        log_warn "cosign not found - skipping cosign signing"
        return
    fi
    
    log_info "Signing artifacts with cosign..."
    
    cd "${DIST_DIR}"
    
    # Sign each artifact
    for file in *.whl *.tar.gz; do
        if [ -f "${file}" ]; then
            log_info "Signing ${file}..."
            # In production, this would use a key or keyless signing
            # For now, we'll document the process
            echo "cosign sign-blob --bundle ${file}.cosign.bundle ${file}" > "${file}.cosign.cmd"
        fi
    done
    
    log_success "Cosign signing prepared (run .cosign.cmd files with appropriate keys)"
}

# Sign artifacts with GPG (if available)
sign_artifacts_gpg() {
    if ! command -v gpg >/dev/null 2>&1; then
        log_warn "gpg not found - skipping GPG signing"
        return
    fi
    
    log_info "Signing artifacts with GPG..."
    
    cd "${DIST_DIR}"
    
    # Check if GPG key is available
    if ! gpg --list-secret-keys >/dev/null 2>&1; then
        log_warn "No GPG secret key found - skipping GPG signing"
        return
    fi
    
    # Sign each artifact
    for file in *.whl *.tar.gz SHA256SUMS; do
        if [ -f "${file}" ]; then
            log_info "Signing ${file} with GPG..."
            gpg --detach-sign --armor "${file}"
        fi
    done
    
    log_success "GPG signatures created"
}

# Generate in-toto attestation
generate_intoto_attestation() {
    log_info "Generating in-toto attestation..."
    
    # Create in-toto layout
    cat > "${ATTESTATION_DIR}/in-toto-attestation.json" <<EOF
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "subject": [
    {
      "name": "nethical-${VERSION}",
      "digest": {
        "sha256": "$(cd "${DIST_DIR}" && sha256sum *.whl | head -1 | cut -d' ' -f1)"
      }
    }
  ],
  "predicateType": "https://in-toto.io/attestation/v0.1",
  "predicate": {
    "builder": {
      "id": "https://github.com/V1B3hR/nethical/release.sh"
    },
    "recipe": {
      "type": "https://github.com/V1B3hR/nethical/release@v1",
      "definedInMaterial": 0,
      "entryPoint": "release.sh",
      "arguments": ["${VERSION}"]
    },
    "metadata": {
      "buildStartedOn": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
      "buildFinishedOn": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
      "reproducible": true,
      "buildInvocationId": "$(git rev-parse HEAD)"
    },
    "materials": [
      {
        "uri": "git+https://github.com/V1B3hR/nethical@$(git rev-parse HEAD)"
      }
    ]
  }
}
EOF
    
    log_success "in-toto attestation generated"
}

# Create release manifest
create_release_manifest() {
    log_info "Creating release manifest..."
    
    cat > "${BUILD_DIR}/release-manifest.json" <<EOF
{
  "version": "${VERSION}",
  "release_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "commit": "$(git rev-parse HEAD)",
  "artifacts": {
    "packages": $(cd "${DIST_DIR}" && ls *.whl *.tar.gz 2>/dev/null | jq -R . | jq -s .),
    "sboms": $(cd "${SBOM_DIR}" && ls *.json *.xml 2>/dev/null | jq -R . | jq -s .),
    "attestations": $(cd "${ATTESTATION_DIR}" && ls *.json 2>/dev/null | jq -R . | jq -s .)
  },
  "checksums": {
    "sha256": "$(cd "${DIST_DIR}" && cat SHA256SUMS | jq -R . | jq -s . | jq -c .)"
  },
  "reproducibility": {
    "source_date_epoch": "${SOURCE_DATE_EPOCH}",
    "builder_info": "$(cat ${BUILD_DIR}/build_env.json | jq -c .)"
  }
}
EOF
    
    log_success "Release manifest created"
}

# Display summary
display_summary() {
    log_success "=========================================="
    log_success "Release ${VERSION} built successfully!"
    log_success "=========================================="
    echo ""
    log_info "Build Artifacts:"
    ls -lh "${DIST_DIR}"
    echo ""
    log_info "SBOMs:"
    ls -lh "${SBOM_DIR}"
    echo ""
    log_info "Attestations:"
    ls -lh "${ATTESTATION_DIR}"
    echo ""
    log_info "Next steps:"
    echo "  1. Review artifacts in ${DIST_DIR}"
    echo "  2. Review SBOMs in ${SBOM_DIR}"
    echo "  3. Review attestations in ${ATTESTATION_DIR}"
    echo "  4. Verify reproducibility with: ./deploy/verify-repro.sh ${VERSION}"
    echo "  5. Upload artifacts to release repository"
    echo "  6. Sign artifacts with production keys"
    echo ""
}

# Main execution
main() {
    log_info "Starting Nethical reproducible release build..."
    log_info "Version: ${VERSION}"
    echo ""
    
    check_prerequisites
    validate_version
    clean_build_dirs
    setup_build_env
    verify_dependencies
    build_package
    generate_sbom_cyclonedx
    generate_sbom_spdx
    run_vulnerability_scan
    generate_slsa_provenance
    generate_intoto_attestation
    sign_artifacts_cosign
    sign_artifacts_gpg
    create_release_manifest
    
    display_summary
}

# Run main function
main
