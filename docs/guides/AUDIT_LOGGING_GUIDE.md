# Training with Merkle Audit Logging

This guide demonstrates how to use the Merkle audit logging feature in `train_any_model.py` to create immutable, cryptographically-verifiable training audit trails.

## Overview

The audit logging feature uses Merkle trees to create an immutable record of training sessions. Each training event is logged and incorporated into a Merkle tree, with the final Merkle root providing cryptographic proof of the training session's integrity.

## Key Features

- **Immutable Audit Trail**: Events are hashed and stored in a Merkle tree structure
- **Cryptographic Verification**: Merkle root provides verifiable proof of all events
- **Structured Event Logging**: Captures key training milestones with rich metadata
- **Compliance-Ready**: Supports regulatory requirements for ML model training audits
- **Tamper-Evident**: Any modification to event logs invalidates the Merkle root

## Usage

### Basic Usage

```bash
# Train with audit logging enabled
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-audit
```

### Custom Audit Path

```bash
# Specify custom audit log location
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 5000 \
    --enable-audit \
    --audit-path my_training_audit
```

### Without Audit Logging (Default)

```bash
# Train without audit logging
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000
```

## Logged Events

The following events are captured during training:

1. **training_start**: Initial configuration and parameters
   - Model type, epochs, batch size, sample count, seed
   
2. **data_loaded**: Data loading completion
   - Number of samples loaded
   
3. **data_split**: Train/validation split
   - Train sample count, validation sample count
   
4. **training_completed**: Training completion
   - Model type, epochs, training duration in seconds
   
5. **validation_metrics**: Model performance metrics
   - Precision, recall, accuracy, F1, ECE
   
6. **model_saved**: Model persistence
   - Model path, metrics path, promotion status

## Output Files

When audit logging is enabled, the following files are generated in the audit path:

### 1. Chunk File (`chunk_<timestamp>.json`)

Contains all training events with the computed Merkle root:

```json
{
  "chunk_id": "chunk_20251007_081034_182569",
  "merkle_root": "afa2f9326d7a4a99d66b141c02b9aa1a2eb5af02700373e9e56bcb86922dbbac",
  "created_at": "2025-10-07T08:10:34.182576",
  "finalized_at": "2025-10-07T08:10:34.343736",
  "event_count": 6,
  "events": [
    {
      "event_type": "training_start",
      "model_type": "heuristic",
      "epochs": 2,
      "batch_size": 32,
      "num_samples": 100,
      "seed": 42,
      "timestamp": "2025-10-07T08:10:34.182719"
    },
    // ... more events
  ]
}
```

### 2. Summary File (`training_summary.json`)

High-level summary with Merkle root and key metrics:

```json
{
  "merkle_root": "afa2f9326d7a4a99d66b141c02b9aa1a2eb5af02700373e9e56bcb86922dbbac",
  "model_type": "heuristic",
  "promoted": false,
  "metrics": {
    "precision": 0.7500,
    "recall": 1.0000,
    "accuracy": 0.8500,
    "f1": 0.8571,
    "ece": 0.1000
  },
  "timestamp": "2025-10-07T08:10:34.344004"
}
```

## Verification

### Verify Merkle Root

The Merkle root in the summary file should match the root in the chunk file:

```bash
# Extract and compare Merkle roots
summary_root=$(jq -r '.merkle_root' training_audit_logs/training_summary.json)
chunk_root=$(jq -r '.merkle_root' training_audit_logs/chunk_*.json)

if [ "$summary_root" = "$chunk_root" ]; then
    echo "✓ Merkle roots match"
else
    echo "✗ Merkle root mismatch - possible tampering"
fi
```

### Verify Event Integrity

You can programmatically verify the integrity of events using the MerkleAnchor API:

```python
from nethical.core.audit_merkle import MerkleAnchor

# Load audit logs
anchor = MerkleAnchor(storage_path="training_audit_logs")

# Get chunk info
chunks = anchor.list_chunks()
for chunk in chunks:
    chunk_id = chunk['chunk_id']
    is_valid = anchor.verify_chunk(chunk_id)
    print(f"Chunk {chunk_id}: {'✓ Valid' if is_valid else '✗ Invalid'}")
```

## Use Cases

### 1. Compliance & Auditing
- Regulatory compliance for ML model training
- Audit trails for model lineage tracking
- Proof of training provenance

### 2. Reproducibility
- Verify training configuration
- Track data splits and preprocessing
- Document model promotion decisions

### 3. Security
- Detect tampering with training records
- Cryptographic proof of training events
- Immutable training history

### 4. Model Governance
- Track model versions and metrics
- Document promotion gate decisions
- Maintain training history across deployments

## Best Practices

1. **Always enable for production training**: Use `--enable-audit` for all production model training
2. **Archive audit logs**: Store audit logs alongside trained models
3. **Verify before deployment**: Check Merkle root integrity before deploying models
4. **Document Merkle roots**: Include Merkle roots in model cards and documentation
5. **Regular verification**: Periodically verify audit log integrity

## Integration with Phase 4

This feature is part of Phase 4 (Integrity & Ethics Operationalization) and integrates with:

- **Merkle Anchoring System** (`nethical.core.audit_merkle`)
- **Policy Diff Auditing** (for tracking policy changes)
- **SLA Monitoring** (for tracking training performance)
- **Ethical Taxonomy** (for tagging training decisions)

## Example Workflow

```bash
# 1. Train model with audit logging
python training/train_any_model.py \
    --model-type logistic \
    --epochs 30 \
    --num-samples 10000 \
    --enable-audit \
    --audit-path production_audit_logs

# 2. Check training results
cat production_audit_logs/training_summary.json

# 3. Verify integrity
python -c "
from nethical.core.audit_merkle import MerkleAnchor
anchor = MerkleAnchor(storage_path='production_audit_logs')
chunks = anchor.list_chunks()
print(f'Total chunks: {len(chunks)}')
for chunk in chunks:
    valid = anchor.verify_chunk(chunk['chunk_id'])
    print(f\"Chunk {chunk['chunk_id']}: {'✓' if valid else '✗'}\")
"

# 4. Archive with model
cp -r production_audit_logs models/current/audit_logs_$(date +%Y%m%d_%H%M%S)
```

## Troubleshooting

### Issue: "MerkleAnchor not available"

**Solution**: Ensure nethical package is properly installed:
```bash
pip install -e .
```

### Issue: Audit logs not generated

**Solution**: Check that `--enable-audit` flag is specified and no errors occurred during training.

### Issue: Merkle root mismatch

**Solution**: This indicates potential tampering. Do not use the audit logs or model. Re-run training with audit logging enabled.

## Related Documentation

- [PHASE4_GUIDE.md](../PHASE4_GUIDE.md) - Phase 4 implementation guide
- [TrainTestPipeline.md](../TrainTestPipeline.md) - Training pipeline overview
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - General training guide

## Technical Details

### Merkle Tree Construction

1. Each event is serialized to JSON and hashed (SHA-256)
2. Event hashes form the leaf nodes of the Merkle tree
3. Parent nodes are computed by hashing concatenated child hashes
4. Process continues until single root hash is computed
5. Root hash provides cryptographic proof of all events

### Security Properties

- **Collision Resistance**: SHA-256 makes it computationally infeasible to find two events with same hash
- **Tamper Evidence**: Any modification to an event changes its hash and invalidates the Merkle root
- **Efficient Verification**: Can verify individual events without accessing all events
- **Immutability**: Once finalized, chunk data cannot be modified without detection

## Future Enhancements

Planned improvements include:
- S3 integration for remote audit log storage
- External timestamp anchoring (e.g., blockchain)
- Merkle proof generation for individual events
- Real-time audit log streaming
- Integration with model registry systems
