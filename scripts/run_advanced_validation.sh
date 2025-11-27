#!/bin/bash
#
# Advanced Validation Testing Runner
#
# This script provides convenient ways to run the advanced validation test suite.
#
# Usage:
#   ./scripts/run_advanced_validation.sh                 # Run standard tests
#   ./scripts/run_advanced_validation.sh --quick         # Run quick tests only
#   ./scripts/run_advanced_validation.sh --extended      # Run with extended tests
#   ./scripts/run_advanced_validation.sh --soak          # Run with soak tests
#   ./scripts/run_advanced_validation.sh --stress        # Run stress tests
#   ./scripts/run_advanced_validation.sh --all           # Run all tests
#
# Environment Variables:
#   NETHICAL_ADVANCED_ITERATIONS  - Override iteration count (default: 10000)
#   NETHICAL_MAX_WORKERS          - Override max workers (default: 150)
#   NETHICAL_SOAK_DURATION        - Override soak duration in seconds (default: 300)
#   OUTPUT_DIR                    - Override output directory for reports
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/tests/validation/results/advanced}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

show_help() {
    cat << EOF
Advanced Validation Testing Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --quick       Run quick tests only (fewer iterations, faster)
    --standard    Run standard tests (default, 10k iterations)
    --extended    Run extended tests (100k iterations, 70+ workers)
    --soak        Run soak tests (2-hour sustained load)
    --stress      Run stress tests (100+ workers with failures)
    --all         Run all tests including extended and soak
    --help, -h    Show this help message

EXAMPLES:
    # Run standard validation tests
    $0

    # Run quick validation for CI/CD
    $0 --quick

    # Run extended tests for pre-production validation
    $0 --extended

    # Run overnight soak test
    $0 --soak

    # Run comprehensive validation (all tests)
    $0 --all

ENVIRONMENT VARIABLES:
    NETHICAL_ADVANCED_ITERATIONS    Override default iteration count
    NETHICAL_MAX_WORKERS            Override maximum worker count
    NETHICAL_SOAK_DURATION          Override soak test duration (seconds)
    OUTPUT_DIR                       Override output directory for reports

OUTPUT:
    Test reports are saved to: $OUTPUT_DIR
    - JSON reports with detailed metrics
    - Markdown reports with human-readable summaries

EOF
}

run_quick_tests() {
    log_info "Running quick validation tests..."
    export NETHICAL_ADVANCED_ITERATIONS=1000
    export NETHICAL_MAX_WORKERS=50
    
    cd "$PROJECT_ROOT"
    python -m pytest tests/validation/test_advanced_validation.py \
        -k "test_chaos_failure_injection or test_generate_comprehensive_report" \
        -v -s --tb=short
}

run_standard_tests() {
    log_info "Running standard validation tests..."
    export NETHICAL_ADVANCED_ITERATIONS=10000
    export NETHICAL_MAX_WORKERS=100
    
    cd "$PROJECT_ROOT"
    python -m pytest tests/validation/test_advanced_validation.py \
        -v -s --tb=short \
        -m "not slow"
}

run_extended_tests() {
    log_info "Running extended validation tests..."
    export NETHICAL_ADVANCED_ITERATIONS=100000
    export NETHICAL_MAX_WORKERS=150
    
    cd "$PROJECT_ROOT"
    python -m pytest tests/validation/test_advanced_validation.py \
        -v -s --tb=short \
        --run-extended
}

run_soak_tests() {
    log_info "Running soak tests (this will take ~2 hours)..."
    export NETHICAL_SOAK_DURATION=7200
    
    cd "$PROJECT_ROOT"
    python -m pytest tests/validation/test_advanced_validation.py \
        -v -s --tb=short \
        --run-soak
}

run_stress_tests() {
    log_info "Running stress tests..."
    export NETHICAL_MAX_WORKERS=150
    
    cd "$PROJECT_ROOT"
    python -m pytest tests/validation/test_advanced_validation.py \
        -k "stress" \
        -v -s --tb=short
}

run_all_tests() {
    log_info "Running all validation tests..."
    
    cd "$PROJECT_ROOT"
    python -m pytest tests/validation/test_advanced_validation.py \
        -v -s --tb=short \
        --run-extended --run-soak
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Parse command line arguments
if [ -z "${1:-}" ]; then
    # No argument provided, run standard tests
    run_standard_tests
else
    case "$1" in
        --quick)
            run_quick_tests
            ;;
        --standard)
            run_standard_tests
            ;;
        --extended)
            run_extended_tests
            ;;
        --soak)
            run_soak_tests
            ;;
        --stress)
            run_stress_tests
            ;;
        --all)
            run_all_tests
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
fi

# Check result
if [ $? -eq 0 ]; then
    log_success "All validation tests passed!"
    log_info "Reports saved to: $OUTPUT_DIR"
else
    log_error "Some tests failed. Check the output above."
    exit 1
fi
