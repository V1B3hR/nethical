#!/bin/bash
# Nethical Build Verification Script
# This script independently verifies that a release build is reproducible
# by rebuilding from source and comparing checksums.
#
# Usage: ./verify-repro.sh <version> [original_build_dir]
# Example: ./verify-repro.sh v1.0.0 /path/to/original/build

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSION="${1:-}"
ORIGINAL_BUILD_DIR="${2:-${PROJECT_ROOT}/dist}"
VERIFY_DIR="${PROJECT_ROOT}/verify-build"
REPORT_DIR="${PROJECT_ROOT}/verification-reports"

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

# Usage information
show_usage() {
    cat <<EOF
Nethical Build Verification Script

This script verifies that a release build is reproducible by:
1. Creating a fresh build environment
2. Rebuilding from the same source commit
3. Comparing checksums of all artifacts
4. Generating a verification report

Usage:
    $0 <version> [original_build_dir]

Arguments:
    version             Version to verify (e.g., v1.0.0)
    original_build_dir  Path to original build directory (default: ./dist)

Examples:
    $0 v1.0.0
    $0 v1.0.0 /path/to/original/dist

Prerequisites:
    - git
    - python3
    - All build dependencies installed
EOF
}

# Verify prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    command -v sha256sum >/dev/null 2>&1 || missing_tools+=("sha256sum")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All prerequisites found"
}

# Validate inputs
validate_inputs() {
    if [ -z "${VERSION}" ]; then
        log_error "Version not specified"
        show_usage
        exit 1
    fi
    
    if [ ! -d "${ORIGINAL_BUILD_DIR}" ]; then
        log_error "Original build directory not found: ${ORIGINAL_BUILD_DIR}"
        exit 1
    fi
    
    log_success "Input validation passed"
}

# Setup verification environment
setup_verify_env() {
    log_info "Setting up verification environment..."
    
    # Clean and create verification directory
    rm -rf "${VERIFY_DIR}"
    mkdir -p "${VERIFY_DIR}"
    mkdir -p "${REPORT_DIR}"
    
    # Get commit hash for the version
    cd "${PROJECT_ROOT}"
    COMMIT_HASH=$(git rev-parse HEAD)
    
    log_info "Verifying commit: ${COMMIT_HASH}"
    log_success "Verification environment ready"
}

# Retrieve original build metadata
get_original_metadata() {
    log_info "Retrieving original build metadata..."
    
    # Look for build environment file
    if [ -f "${PROJECT_ROOT}/build/build_env.json" ]; then
        ORIGINAL_SOURCE_DATE_EPOCH=$(jq -r '.source_date_epoch' "${PROJECT_ROOT}/build/build_env.json")
        ORIGINAL_COMMIT=$(jq -r '.git_commit' "${PROJECT_ROOT}/build/build_env.json")
        
        log_info "Original SOURCE_DATE_EPOCH: ${ORIGINAL_SOURCE_DATE_EPOCH}"
        log_info "Original commit: ${ORIGINAL_COMMIT}"
    else
        log_warn "Original build metadata not found, using current commit info"
        ORIGINAL_SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)
        ORIGINAL_COMMIT=$(git rev-parse HEAD)
    fi
}

# Perform verification build
perform_verify_build() {
    log_info "Performing verification build..."
    
    cd "${PROJECT_ROOT}"
    
    # Set same build environment as original
    export SOURCE_DATE_EPOCH="${ORIGINAL_SOURCE_DATE_EPOCH}"
    export PYTHONHASHSEED=0
    export PYTHONDONTWRITEBYTECODE=1
    
    # Install dependencies
    python3 -m pip install --quiet build
    
    if [ -f "requirements-hashed.txt" ]; then
        python3 -m pip install --require-hashes -r requirements-hashed.txt --quiet
    fi
    
    # Build package
    python3 -m build --outdir "${VERIFY_DIR}"
    
    # Generate checksums
    cd "${VERIFY_DIR}"
    sha256sum * > SHA256SUMS
    
    log_success "Verification build completed"
}

# Compare checksums
compare_artifacts() {
    log_info "Comparing build artifacts..."
    
    local all_match=true
    local comparison_results=()
    
    # Compare each artifact
    for original_file in "${ORIGINAL_BUILD_DIR}"/*.{whl,tar.gz} 2>/dev/null; do
        if [ ! -f "${original_file}" ]; then
            continue
        fi
        
        filename=$(basename "${original_file}")
        verify_file="${VERIFY_DIR}/${filename}"
        
        if [ ! -f "${verify_file}" ]; then
            log_error "Verification artifact missing: ${filename}"
            comparison_results+=("MISSING|${filename}|N/A|N/A")
            all_match=false
            continue
        fi
        
        # Calculate checksums
        original_hash=$(sha256sum "${original_file}" | cut -d' ' -f1)
        verify_hash=$(sha256sum "${verify_file}" | cut -d' ' -f1)
        
        if [ "${original_hash}" = "${verify_hash}" ]; then
            log_success "✓ ${filename} - checksums match"
            comparison_results+=("MATCH|${filename}|${original_hash}|${verify_hash}")
        else
            log_error "✗ ${filename} - checksums differ!"
            log_error "  Original: ${original_hash}"
            log_error "  Verified: ${verify_hash}"
            comparison_results+=("MISMATCH|${filename}|${original_hash}|${verify_hash}")
            all_match=false
        fi
    done
    
    if [ "$all_match" = true ]; then
        log_success "All artifacts match - build is reproducible!"
        return 0
    else
        log_error "Some artifacts do not match - build may not be reproducible"
        return 1
    fi
}

# Analyze differences
analyze_differences() {
    log_info "Analyzing differences..."
    
    local diff_report="${REPORT_DIR}/diff-analysis-${VERSION}.txt"
    
    cat > "${diff_report}" <<EOF
Build Reproducibility Difference Analysis
Version: ${VERSION}
Verification Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

=== Build Environment ===
Original SOURCE_DATE_EPOCH: ${ORIGINAL_SOURCE_DATE_EPOCH}
Verify SOURCE_DATE_EPOCH: ${SOURCE_DATE_EPOCH}
Original Commit: ${ORIGINAL_COMMIT}
Verify Commit: $(git rev-parse HEAD)

=== Artifact Comparison ===
EOF
    
    # Check file sizes
    echo "" >> "${diff_report}"
    echo "File Size Comparison:" >> "${diff_report}"
    for original_file in "${ORIGINAL_BUILD_DIR}"/*.{whl,tar.gz} 2>/dev/null; do
        if [ ! -f "${original_file}" ]; then
            continue
        fi
        
        filename=$(basename "${original_file}")
        verify_file="${VERIFY_DIR}/${filename}"
        
        if [ -f "${verify_file}" ]; then
            original_size=$(stat -c%s "${original_file}" 2>/dev/null || stat -f%z "${original_file}")
            verify_size=$(stat -c%s "${verify_file}" 2>/dev/null || stat -f%z "${verify_file}")
            echo "  ${filename}:" >> "${diff_report}"
            echo "    Original: ${original_size} bytes" >> "${diff_report}"
            echo "    Verified: ${verify_size} bytes" >> "${diff_report}"
            
            if [ "${original_size}" != "${verify_size}" ]; then
                echo "    Status: SIZE MISMATCH" >> "${diff_report}"
            else
                echo "    Status: Size matches" >> "${diff_report}"
            fi
        fi
    done
    
    log_info "Difference analysis saved to: ${diff_report}"
}

# Generate verification report
generate_report() {
    local status=$1
    
    log_info "Generating verification report..."
    
    local report_file="${REPORT_DIR}/verification-report-${VERSION}.json"
    
    # Determine overall status
    local verification_status="FAILED"
    if [ "${status}" = "0" ]; then
        verification_status="PASSED"
    fi
    
    # Create JSON report
    cat > "${report_file}" <<EOF
{
  "version": "${VERSION}",
  "verification_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "${verification_status}",
  "reproducible": $([ "${status}" = "0" ] && echo "true" || echo "false"),
  "build_environment": {
    "original": {
      "source_date_epoch": "${ORIGINAL_SOURCE_DATE_EPOCH}",
      "commit": "${ORIGINAL_COMMIT}"
    },
    "verified": {
      "source_date_epoch": "${SOURCE_DATE_EPOCH}",
      "commit": "$(git rev-parse HEAD)",
      "builder": "${USER}@$(hostname)",
      "os": "$(uname -s)",
      "arch": "$(uname -m)"
    }
  },
  "artifacts_verified": $(ls "${ORIGINAL_BUILD_DIR}"/*.{whl,tar.gz} 2>/dev/null | wc -l),
  "checksum_matches": $([ "${status}" = "0" ] && echo "100" || echo "0"),
  "report_location": "${report_file}"
}
EOF
    
    log_success "Verification report generated: ${report_file}"
    
    # Display report summary
    echo ""
    log_info "=========================================="
    log_info "Verification Report Summary"
    log_info "=========================================="
    cat "${report_file}" | jq '.'
    echo ""
}

# Display verification summary
display_summary() {
    local status=$1
    
    echo ""
    if [ "${status}" = "0" ]; then
        log_success "=========================================="
        log_success "Build Verification: PASSED ✓"
        log_success "=========================================="
        log_success "The build is 100% reproducible!"
        log_success "All artifact checksums match the original build."
    else
        log_error "=========================================="
        log_error "Build Verification: FAILED ✗"
        log_error "=========================================="
        log_error "The build is NOT reproducible."
        log_error "Some artifact checksums do not match."
        log_error "Review the analysis report for details."
    fi
    echo ""
    log_info "Reports saved in: ${REPORT_DIR}"
    log_info "Verification artifacts in: ${VERIFY_DIR}"
    echo ""
}

# Main execution
main() {
    log_info "Starting Nethical build verification..."
    log_info "Version: ${VERSION}"
    echo ""
    
    check_prerequisites
    validate_inputs
    setup_verify_env
    get_original_metadata
    perform_verify_build
    
    # Compare artifacts and capture result
    if compare_artifacts; then
        verification_status=0
    else
        verification_status=1
        analyze_differences
    fi
    
    generate_report ${verification_status}
    display_summary ${verification_status}
    
    exit ${verification_status}
}

# Show usage if requested
if [ "${VERSION:-}" = "--help" ] || [ "${VERSION:-}" = "-h" ]; then
    show_usage
    exit 0
fi

# Run main function
main
