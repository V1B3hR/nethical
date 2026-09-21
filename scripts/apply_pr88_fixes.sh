#!/bin/bash
# Script to apply fixes to PR #88 (copilot/enhance-threat-modeling-tools branch)
#
# This script should be run on the PR #88 branch to:
# 1. Restore the correct test file
# 2. Fix the security alerts

set -e

echo "======================================"
echo "Applying fixes to PR #88"
echo "======================================"
echo ""

# Check if we're on the right branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "copilot/enhance-threat-modeling-tools" ]; then
    echo "⚠️  Warning: You are not on the 'copilot/enhance-threat-modeling-tools' branch"
    echo "   Current branch: $CURRENT_BRANCH"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Step 1: Backing up current test file..."
if [ -f tests/unit/test_sso.py ]; then
    cp tests/unit/test_sso.py tests/unit/test_sso.py.backup
    echo "✓ Backup created: tests/unit/test_sso.py.backup"
else
    echo "⚠️  tests/unit/test_sso.py not found"
fi
echo ""

echo "Step 2: Restoring correct test file with security fixes..."
if [ -f test_sso_CORRECTED.py ]; then
    cp test_sso_CORRECTED.py tests/unit/test_sso.py
    echo "✓ Test file restored from test_sso_CORRECTED.py"
else
    # Create secure temporary file
    TEMP_FILE=$(mktemp) || { echo "❌ Error: Failed to create temporary file"; exit 1; }
    trap "rm -f $TEMP_FILE" EXIT
    
    if git show 889d794:tests/unit/test_sso.py > "$TEMP_FILE" 2>/dev/null; then
        # Apply the security fix using pattern-based replacement
        sed -i 's/assert "oauth\.example\.com" in auth_url/assert urlparse(auth_url).hostname == "oauth.example.com"/' "$TEMP_FILE"
        cp "$TEMP_FILE" tests/unit/test_sso.py
        echo "✓ Test file restored from commit 889d794 with security fix applied"
    else
        echo "❌ Error: Could not find test_sso_CORRECTED.py or access commit 889d794"
        echo "   Please manually restore tests/unit/test_sso.py"
        exit 1
    fi
fi
echo ""

echo "Step 3: Verifying the fix..."
if grep -q "urlparse(login_url).hostname == \"idp.example.com\"" tests/unit/test_sso.py && \
   grep -q "urlparse(auth_url).hostname == \"oauth.example.com\"" tests/unit/test_sso.py; then
    echo "✓ Security fixes verified in test file"
else
    echo "❌ Error: Security fixes not found in test file"
    echo "   Please manually verify the changes"
    exit 1
fi
echo ""

echo "Step 4: Running Python syntax check..."
if python3 -m py_compile tests/unit/test_sso.py; then
    echo "✓ Python syntax is valid"
else
    echo "❌ Error: Python syntax errors detected"
    exit 1
fi
echo ""

echo "======================================"
echo "✅ Fixes applied successfully!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Review the changes: git diff tests/unit/test_sso.py"
echo "2. Run tests: python -m pytest tests/unit/test_sso.py -v"
echo "3. Commit the fix: git add tests/unit/test_sso.py"
echo "4. Commit message: 'Fix test file corruption and security alerts'"
echo "5. Push: git push origin copilot/enhance-threat-modeling-tools"
echo ""
echo "The backup file is at: tests/unit/test_sso.py.backup"
