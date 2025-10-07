#!/bin/bash
# Demo script for Phase 4 Integration in Model Training

echo "=================================================="
echo "Phase 4 Integration Demo for Model Training"
echo "=================================================="
echo ""

echo "This demo shows how Phase 4 Integrated Governance"
echo "tracks model training events, detects violations,"
echo "and generates comprehensive governance reports."
echo ""

# Example 1: Training with Phase 4
echo "Example 1: Training with Phase 4 Integrated Governance"
echo "--------------------------------------------------------"
echo ""
echo "Command:"
echo "python training/train_any_model.py \\"
echo "    --model-type heuristic \\"
echo "    --epochs 5 \\"
echo "    --num-samples 200 \\"
echo "    --enable-phase4 \\"
echo "    --seed 42"
echo ""

python training/train_any_model.py \
    --model-type heuristic \
    --epochs 5 \
    --num-samples 200 \
    --enable-phase4 \
    --seed 42

echo ""
echo "Phase 4 report generated! View it at:"
ls -1 training_phase4_data/*.md 2>/dev/null | tail -1
echo ""

# Example 2: Show the audit log structure
echo "Example 2: Audit Log Structure"
echo "--------------------------------------------------------"
echo ""
echo "Audit logs are stored in JSON format with Merkle roots:"
ls -1 training_phase4_data/audit_logs/*.json 2>/dev/null | tail -1 | xargs -I {} sh -c 'echo "{}:" && cat {} | python -m json.tool | head -20'
echo "..."
echo ""

# Example 3: Show the comprehensive report
echo "Example 3: Comprehensive Phase 4 Report"
echo "--------------------------------------------------------"
echo ""
ls -1 training_phase4_data/*.md 2>/dev/null | tail -1 | xargs cat
echo ""

echo "=================================================="
echo "Demo Complete!"
echo "=================================================="
echo ""
echo "Key Features Demonstrated:"
echo "  ✅ Merkle-anchored audit trail"
echo "  ✅ Violation detection (promotion gate failures)"
echo "  ✅ SLA performance monitoring"
echo "  ✅ Ethical taxonomy coverage tracking"
echo "  ✅ Comprehensive governance reports"
echo ""
echo "For more information, see:"
echo "  - training/PHASE4_INTEGRATION.md"
echo "  - examples/phase4_demo.py"
echo ""
