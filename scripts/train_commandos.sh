#!/bin/bash
#
# Nethical Commando Training Script
#
# This script serves as a single entry point to launch the full Nethical training
# pipeline with all security and robustness features enabled.
#
# Usage:
#   ./scripts/train_commandos.sh                          # Run with default settings
#   ./scripts/train_commandos.sh --model-type anomaly     # Override model type
#   ./scripts/train_commandos.sh --epochs 20              # Add extra arguments
#
# Default "Commando" features enabled:
#   --include-adversarial   : Use adversarial generator for hard negatives
#   --enable-governance     : Activate safety checks
#   --enable-drift-tracking : Monitor model stability
#   --enable-audit          : Full accountability via Merkle audit logging
#   --model-type heuristic  : Default model type (can be overridden)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Mission Briefing
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}       NETHICAL COMMANDO TRAINING       ${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo -e "${YELLOW}MISSION BRIEFING:${NC}"
echo -e "  Starting commando training with all security and robustness features:"
echo -e "  ${GREEN}✓${NC} Adversarial data injection for robust threat detection"
echo -e "  ${GREEN}✓${NC} Governance validation for safety compliance"
echo -e "  ${GREEN}✓${NC} Drift tracking for model stability monitoring"
echo -e "  ${GREEN}✓${NC} Full audit logging with Merkle anchors"
echo
echo -e "${YELLOW}Default model type:${NC} heuristic (override with --model-type <type>)"
echo
echo -e "${BLUE}========================================${NC}"
echo

# Change to project root
cd "$PROJECT_ROOT"

# Default commando flags
COMMANDO_FLAGS=(
    "--include-adversarial"
    "--enable-governance"
    "--enable-drift-tracking"
    "--enable-audit"
)

# Check if --model-type was provided in arguments (handles both --model-type value and --model-type=value)
MODEL_TYPE_PROVIDED=false
for arg in "$@"; do
    if [[ "$arg" == "--model-type" ]] || [[ "$arg" == --model-type=* ]]; then
        MODEL_TYPE_PROVIDED=true
        break
    fi
done

# Add default model type if not provided
if [ "$MODEL_TYPE_PROVIDED" = false ]; then
    COMMANDO_FLAGS+=("--model-type" "heuristic")
fi

echo -e "${YELLOW}Launching training pipeline...${NC}"
echo

# Execute training with commando flags and any additional arguments
python training/train_any_model.py "${COMMANDO_FLAGS[@]}" "$@"

EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   MISSION COMPLETE - Training Finished ${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}   MISSION FAILED - Training Error      ${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE
