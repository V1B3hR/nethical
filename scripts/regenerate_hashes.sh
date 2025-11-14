#!/bin/bash
# Regenerate requirements-hashed.txt with pip-compile
# This script ensures supply chain security by generating SHA256 hashes for all dependencies

set -e

echo "==================================="
echo "Regenerating hashed requirements"
echo "==================================="
echo

# Check if pip-tools is installed
if ! command -v pip-compile &> /dev/null; then
    echo "pip-tools not found. Installing..."
    python -m pip install --upgrade pip pip-tools
fi

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found in current directory"
    exit 1
fi

echo "Generating hashed lock file..."
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt

echo
echo "==================================="
echo "Hash generation complete!"
echo "==================================="
echo
echo "Verifying hash integrity..."

# Count hashes
hash_count=$(grep -c "\-\-hash=" requirements-hashed.txt || true)
package_count=$(grep -c "^[a-zA-Z]" requirements-hashed.txt || true)

echo "  - Packages: $package_count"
echo "  - Hashes: $hash_count"
echo

# Test install (dry run)
echo "Testing installation with --require-hashes (dry run)..."
if pip install --require-hashes --dry-run -r requirements-hashed.txt > /dev/null 2>&1; then
    echo "  ✓ Hash verification passed!"
else
    echo "  ✗ Hash verification failed!"
    exit 1
fi

echo
echo "==================================="
echo "Success! requirements-hashed.txt is ready."
echo "==================================="
echo
echo "Next steps:"
echo "  1. Review the changes in requirements-hashed.txt"
echo "  2. Commit both requirements.txt and requirements-hashed.txt"
echo "  3. Test with: pip install --require-hashes -r requirements-hashed.txt"
