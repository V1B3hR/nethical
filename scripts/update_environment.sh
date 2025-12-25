#!/bin/bash
# Environment Update Script for Nethical
# This script updates system packages, Python, and dependencies
# 
# Usage: ./scripts/update_environment.sh [--security-only] [--skip-system] [--dry-run]
#
# Options:
#   --security-only: Only install security updates
#   --skip-system:   Skip system package updates
#   --dry-run:       Show what would be updated without making changes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
SECURITY_ONLY=false
SKIP_SYSTEM=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --security-only)
            SECURITY_ONLY=true
            shift
            ;;
        --skip-system)
            SKIP_SYSTEM=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--security-only] [--skip-system] [--dry-run]"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_warning "Running as root is not recommended for Python package updates"
fi

log_info "Starting Nethical environment update..."
log_info "Dry run: $DRY_RUN"
log_info "Security only: $SECURITY_ONLY"
log_info "Skip system: $SKIP_SYSTEM"

# System package updates
if [ "$SKIP_SYSTEM" = false ]; then
    log_info "Updating system packages..."
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        log_error "Cannot detect OS"
        exit 1
    fi
    
    case $OS in
        ubuntu|debian)
            if [ "$DRY_RUN" = true ]; then
                log_info "Would update apt packages:"
                apt list --upgradable 2>/dev/null || true
            else
                log_info "Updating apt package lists..."
                sudo apt-get update
                
                if [ "$SECURITY_ONLY" = true ]; then
                    log_info "Installing security updates only..."
                    sudo apt-get upgrade -y -o "Dir::Etc::SourceList=/etc/apt/sources.list.d/security.list"
                else
                    log_info "Installing all updates..."
                    sudo apt-get upgrade -y
                fi
                
                log_info "Removing unused packages..."
                sudo apt-get autoremove -y
            fi
            ;;
        centos|rhel|fedora)
            if [ "$DRY_RUN" = true ]; then
                log_info "Would update yum/dnf packages:"
                yum check-update 2>/dev/null || dnf check-update 2>/dev/null || true
            else
                if [ "$SECURITY_ONLY" = true ]; then
                    log_info "Installing security updates only..."
                    sudo yum update -y --security 2>/dev/null || sudo dnf upgrade -y --security 2>/dev/null
                else
                    log_info "Installing all updates..."
                    sudo yum update -y 2>/dev/null || sudo dnf upgrade -y 2>/dev/null
                fi
            fi
            ;;
        *)
            log_warning "Unsupported OS: $OS. Skipping system updates."
            ;;
    esac
else
    log_info "Skipping system package updates (--skip-system)"
fi

# Python and pip updates
log_info "Checking Python and pip versions..."

PYTHON_CMD=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
if [ -z "$PYTHON_CMD" ]; then
    log_error "Python not found!"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

if [ "$DRY_RUN" = true ]; then
    log_info "Would upgrade pip..."
    $PYTHON_CMD -m pip list --outdated
else
    log_info "Upgrading pip..."
    $PYTHON_CMD -m pip install --upgrade pip
    
    log_info "Upgrading pip tools..."
    $PYTHON_CMD -m pip install --upgrade setuptools wheel pip-tools
fi

# Nethical dependencies update
if [ -f "requirements.txt" ]; then
    log_info "Checking Nethical dependencies..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Packages that would be updated:"
        $PYTHON_CMD -m pip list --outdated --format=columns | grep -E "$(cat requirements.txt | grep -v '^#' | grep -v '^$' | cut -d'=' -f1 | paste -sd'|')" || true
    else
        if [ "$SECURITY_ONLY" = true ]; then
            log_info "Checking for security vulnerabilities..."
            
            # Install scanning tools if not present
            $PYTHON_CMD -m pip install --upgrade pip-audit safety 2>/dev/null || true
            
            log_info "Running pip-audit..."
            if $PYTHON_CMD -m pip_audit -r requirements.txt --format json > /tmp/pip-audit.json 2>/dev/null; then
                log_info "✓ No vulnerabilities found by pip-audit"
            else
                log_warning "Vulnerabilities detected! Review /tmp/pip-audit.json"
                cat /tmp/pip-audit.json
                
                log_info "Attempting to fix vulnerabilities..."
                $PYTHON_CMD -m pip_audit --fix -r requirements.txt || log_warning "Some vulnerabilities could not be auto-fixed"
            fi
            
            log_info "Running safety check..."
            if $PYTHON_CMD -m safety check --json > /tmp/safety-check.json 2>/dev/null; then
                log_info "✓ No vulnerabilities found by safety"
            else
                log_warning "Vulnerabilities detected! Review /tmp/safety-check.json"
                cat /tmp/safety-check.json
            fi
        else
            log_info "Updating all dependencies..."
            
            # Backup current requirements
            cp requirements.txt requirements.txt.backup
            log_info "Backed up requirements.txt to requirements.txt.backup"
            
            # Update dependencies
            if [ -f "requirements.in" ]; then
                log_info "Using pip-compile to update dependencies..."
                $PYTHON_CMD -m piptools compile --upgrade requirements.in -o requirements.txt
            else
                log_info "Updating packages from requirements.txt..."
                $PYTHON_CMD -m pip install --upgrade -r requirements.txt
            fi
            
            log_info "Running vulnerability scan after update..."
            $PYTHON_CMD -m pip install --upgrade pip-audit 2>/dev/null || true
            $PYTHON_CMD -m pip_audit || log_warning "Vulnerabilities still present after update"
        fi
    fi
else
    log_warning "requirements.txt not found in current directory"
fi

# Update hashed requirements if present
if [ -f "requirements-hashed.txt" ] && [ "$DRY_RUN" = false ] && [ "$SECURITY_ONLY" = false ]; then
    log_info "Regenerating hashed requirements..."
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install pip-tools 2>/dev/null || true
        $PYTHON_CMD -m piptools compile --generate-hashes --output-file requirements-hashed.txt requirements.txt || log_warning "Failed to regenerate hashed requirements"
        log_info "✓ Updated requirements-hashed.txt"
    fi
fi

# Docker image updates (if in Docker environment)
if [ -f "Dockerfile" ] && command -v docker &> /dev/null; then
    log_info "Checking Docker image base..."
    BASE_IMAGE=$(grep -E "^FROM " Dockerfile | head -1 | awk '{print $2}')
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Would check for updates to: $BASE_IMAGE"
    else
        log_info "Pulling latest base image: $BASE_IMAGE"
        docker pull "$BASE_IMAGE" || log_warning "Failed to pull latest base image"
    fi
fi

# Summary
log_info ""
log_info "═══════════════════════════════════════"
log_info "Update Summary"
log_info "═══════════════════════════════════════"

if [ "$DRY_RUN" = true ]; then
    log_info "DRY RUN - No changes were made"
else
    log_info "✓ Updates completed successfully"
    log_info ""
    log_info "Next steps:"
    echo "  1. Review changes: git diff requirements.txt"
    echo "  2. Run tests: pytest tests/"
    echo "  3. Rebuild Docker image: docker build -t nethical:latest ."
    echo "  4. Commit changes: git add requirements*.txt && git commit -m 'chore: update dependencies'"
fi

log_info "═══════════════════════════════════════"

# Cleanup
rm -f /tmp/pip-audit.json /tmp/safety-check.json 2>/dev/null || true

log_info "Update process completed!"
exit 0
