#!/bin/bash
# Nethical Edge - Cross-Compilation Build Script
#
# Usage: ./build.sh [--target <arch>] [--output <dir>] [--cuda]
#
# Supported targets:
#   x86_64  - Standard x86-64 Linux
#   arm64   - ARM 64-bit (Raspberry Pi 4/5, Jetson)
#   arm32   - ARM 32-bit (BeagleBone, older Pi)
#   riscv64 - RISC-V 64-bit
#
# Examples:
#   ./build.sh --target arm64
#   ./build.sh --target arm64 --cuda --output dist/

set -e

# Default values
TARGET="x86_64"
OUTPUT_DIR="dist"
CUDA_ENABLED=false
OPTIMIZE_LEVEL=2
STRIP_BINARY=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cuda)
            CUDA_ENABLED=true
            shift
            ;;
        --optimize)
            OPTIMIZE_LEVEL="$2"
            shift 2
            ;;
        --no-strip)
            STRIP_BINARY=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--target <arch>] [--output <dir>] [--cuda] [--optimize <level>] [--no-strip]"
            echo ""
            echo "Targets:"
            echo "  x86_64  - Standard x86-64 Linux"
            echo "  arm64   - ARM 64-bit (Raspberry Pi 4/5, Jetson)"
            echo "  arm32   - ARM 32-bit (BeagleBone, older Pi)"
            echo "  riscv64 - RISC-V 64-bit"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate target
case $TARGET in
    x86_64|arm64|arm32|riscv64)
        log_info "Building for target: $TARGET"
        ;;
    *)
        log_error "Unsupported target: $TARGET"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set cross-compilation environment
case $TARGET in
    x86_64)
        export CC=gcc
        export CXX=g++
        export ARCH=x86_64
        ;;
    arm64)
        export CC=aarch64-linux-gnu-gcc
        export CXX=aarch64-linux-gnu-g++
        export ARCH=aarch64
        if ! command -v $CC &> /dev/null; then
            log_warn "Cross-compiler not found. Installing..."
            sudo apt-get update
            sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
        fi
        ;;
    arm32)
        export CC=arm-linux-gnueabihf-gcc
        export CXX=arm-linux-gnueabihf-g++
        export ARCH=armhf
        if ! command -v $CC &> /dev/null; then
            log_warn "Cross-compiler not found. Installing..."
            sudo apt-get update
            sudo apt-get install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
        fi
        ;;
    riscv64)
        export CC=riscv64-linux-gnu-gcc
        export CXX=riscv64-linux-gnu-g++
        export ARCH=riscv64
        if ! command -v $CC &> /dev/null; then
            log_warn "Cross-compiler not found. Installing..."
            sudo apt-get update
            sudo apt-get install -y gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
        fi
        ;;
esac

log_info "Using compiler: $CC"

# Build Python wheel
log_info "Building Python wheel..."
cd "$(dirname "$0")/.."

# Create virtual environment for build
python3 -m venv .build-venv
source .build-venv/bin/activate

# Install build dependencies
pip install --upgrade pip setuptools wheel cython numpy

# Build with optimization
if [ "$CUDA_ENABLED" = true ] && [ "$TARGET" = "arm64" ]; then
    log_info "Building with CUDA support for Jetson..."
    export CUDA_HOME=/usr/local/cuda
    export USE_CUDA=1
fi

# Compile Cython extensions
log_info "Compiling optimized extensions..."
python setup.py build_ext --inplace

# Create distribution package
log_info "Creating distribution package..."
python -m build --wheel

# Copy to output directory
cp dist/*.whl "$OUTPUT_DIR/"

# Create standalone archive
log_info "Creating standalone archive..."
ARCHIVE_NAME="nethical-edge-${TARGET}"
if [ "$CUDA_ENABLED" = true ]; then
    ARCHIVE_NAME="${ARCHIVE_NAME}-cuda"
fi

mkdir -p "$OUTPUT_DIR/$ARCHIVE_NAME"

# Copy essential files
cp -r src "$OUTPUT_DIR/$ARCHIVE_NAME/"
cp -r config "$OUTPUT_DIR/$ARCHIVE_NAME/"
cp -r scripts "$OUTPUT_DIR/$ARCHIVE_NAME/"
cp README.md "$OUTPUT_DIR/$ARCHIVE_NAME/"
cp pyproject.toml "$OUTPUT_DIR/$ARCHIVE_NAME/" 2>/dev/null || true

# Copy nethical core modules
cp -r ../nethical/edge "$OUTPUT_DIR/$ARCHIVE_NAME/nethical_edge/"
cp -r ../nethical/sync "$OUTPUT_DIR/$ARCHIVE_NAME/nethical_sync/"
cp -r ../nethical/core "$OUTPUT_DIR/$ARCHIVE_NAME/nethical_core/"

# Create startup script
cat > "$OUTPUT_DIR/$ARCHIVE_NAME/run.sh" << 'EOF'
#!/bin/bash
# Nethical Edge Startup Script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Default configuration
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config/edge.yaml}"
DEVICE_ID="${DEVICE_ID:-$(hostname)}"

python3 -c "
from src import create_governor
governor = create_governor('$DEVICE_ID', config_path='$CONFIG_FILE')
governor.warmup()
print('Nethical Edge started for device: $DEVICE_ID')
print('Governance engine ready')
" "$@"
EOF
chmod +x "$OUTPUT_DIR/$ARCHIVE_NAME/run.sh"

# Create tarball
cd "$OUTPUT_DIR"
tar -czf "${ARCHIVE_NAME}.tar.gz" "$ARCHIVE_NAME"
rm -rf "$ARCHIVE_NAME"

# Cleanup
cd -
deactivate
rm -rf .build-venv

log_info "Build complete!"
log_info "Output: $OUTPUT_DIR/${ARCHIVE_NAME}.tar.gz"
log_info "Wheel: $OUTPUT_DIR/*.whl"

# Print size information
echo ""
echo "Package sizes:"
ls -lh "$OUTPUT_DIR/${ARCHIVE_NAME}.tar.gz"
ls -lh "$OUTPUT_DIR/"*.whl 2>/dev/null || true
