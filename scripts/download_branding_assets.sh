#!/bin/bash
# Script to download official Nethical branding assets from provided URLs
# Usage: ./download_branding_assets.sh <banner_url> <logo_url>

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASSETS_DIR="$SCRIPT_DIR/../assets"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "Nethical Branding Assets Download Script"
echo "=================================================="
echo ""

# Check if URLs are provided
if [ $# -ne 2 ]; then
    echo -e "${YELLOW}Usage: $0 <banner_url> <logo_url>${NC}"
    echo ""
    echo "Example:"
    echo "  $0 https://example.com/banner.png https://example.com/logo.png"
    echo ""
    echo "Or manually place files as:"
    echo "  assets/nethical_banner_original.png"
    echo "  assets/nethical_logo_original.png"
    echo ""
    echo "Then run: python3 scripts/create_branding_assets.py"
    exit 1
fi

BANNER_URL="$1"
LOGO_URL="$2"

# Create assets directory if it doesn't exist
mkdir -p "$ASSETS_DIR"

# Download banner
echo "Downloading banner from: $BANNER_URL"
if curl -L -f -o "$ASSETS_DIR/nethical_banner_original.png" "$BANNER_URL"; then
    echo -e "${GREEN}✓ Banner downloaded successfully${NC}"
else
    echo -e "${RED}✗ Failed to download banner${NC}"
    exit 1
fi

# Download logo
echo "Downloading logo from: $LOGO_URL"
if curl -L -f -o "$ASSETS_DIR/nethical_logo_original.png" "$LOGO_URL"; then
    echo -e "${GREEN}✓ Logo downloaded successfully${NC}"
else
    echo -e "${RED}✗ Failed to download logo${NC}"
    exit 1
fi

echo ""
echo "=================================================="
echo -e "${GREEN}Downloads complete!${NC}"
echo "=================================================="
echo ""
echo "Original files saved to:"
echo "  - $ASSETS_DIR/nethical_banner_original.png"
echo "  - $ASSETS_DIR/nethical_logo_original.png"
echo ""
echo "Now run the resize script:"
echo "  python3 scripts/create_branding_assets.py"
echo ""
