# Semantic Monitoring Guide

## Overview

Nethical v2.0 introduces **semantic intent deviation monitoring** using sentence embeddings, providing more accurate detection of paraphrased malicious intents while reducing false positives for benign synonym usage.

## Key Features

### 1. Semantic Similarity vs. Lexical Similarity

Traditional lexical similarity (e.g., Jaccard index) compares tokens directly:

```python
stated_intent = "analyze customer information"
actual_action = "examine client data"

# Lexical similarity (token overlap): ~0 (no shared words!)
# Semantic similarity (embeddings): ~0.85 (high similarity!)
```

### 2. Graceful Degradation

The system automatically falls back to lexical methods if:
- `sentence-transformers` is not installed
- Model loading fails
- Embeddings are unavailable

```python
from nethical.core.semantics import is_semantic_available

if is_semantic_available():
    print("Using semantic monitoring")
else:
    print("Falling back to lexical monitoring")
```

## Embedding Strategy

### Model Selection

Nethical uses **`all-MiniLM-L6-v2`** by default:
- **Fast**: CPU-efficient (no GPU required)
- **Compact**: ~90MB model size
- **Accurate**: High quality for short texts
- **Multilingual**: Works well with English

### Singleton Pattern

The model is loaded once and cached:

```python
from nethical.core.semantics import get_embedding_model

model = get_embedding_model()  # Lazy-loaded, thread-safe
```

### Similarity Computation

Cosine similarity between embeddings:

```python
from nethical.core.semantics import get_similarity

similarity = get_similarity(
    "delete user account",
    "remove user profile"
)
# Returns: ~0.88 (high similarity)

deviation = 1.0 - similarity
# Returns: ~0.12 (low deviation)
```

## Configuration

### Enable/Disable Semantic Monitoring

```python
from nethical.core.models import MonitoringConfig

config = MonitoringConfig(
    use_semantic_intent=True,  # Default in v2.0
    intent_deviation_threshold=0.75  # May need recalibration
)
```

### Environment Variables

```bash
# Force enable semantic monitoring
export NETHICAL_SEMANTIC=1

# Set intent deviation threshold
export NETHICAL_INTENT_THRESHOLD=0.75
```

## Performance Considerations

### Cold Start Time

First model load takes ~2-5 seconds:

```python
# Preload in initialization
from nethical.core.semantics import get_embedding_model
model = get_embedding_model()  # Force load
```

### Memory Usage

- Model: ~90MB RAM
- Per-request: <10MB (cached embeddings)
- Total: ~100-150MB additional memory

### Latency

| Operation | Time |
|-----------|------|
| Model load (first time) | 2-5s |
| Encode single text (256 chars) | 5-15ms |
| Compute similarity | <1ms |
| **Total per evaluation** | **10-20ms** |

### Optimization Tips

1. **Preload in Docker**:
```dockerfile
ARG PRELOAD_EMBEDDINGS=true
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

2. **Batch Processing**:
```python
from nethical.core.semantics import batch_similarity

similarities = batch_similarity(
    texts=["action1", "action2", "action3"],
    reference_text="stated intent"
)
```

3. **Cache Control**:
```python
from nethical.core.semantics import clear_cache

# Clear cached concept similarities if needed
clear_cache()
```

## Threshold Recalibration

Semantic similarity scores are typically higher than lexical:

| Metric | Lexical Range | Semantic Range |
|--------|---------------|----------------|
| Identical text | 1.0 | 1.0 |
| Close paraphrase | 0.2-0.4 | 0.7-0.9 |
| Unrelated | 0.0-0.1 | 0.0-0.3 |

**Recommended adjustments**:

```python
# Old lexical threshold
intent_deviation_threshold = 0.70

# New semantic threshold (higher similarity expected)
intent_deviation_threshold = 0.75  # or 0.80
```

### Testing Thresholds

```python
from nethical.core.semantics import get_similarity

# Test benign paraphrases
examples = [
    ("save file", "store document"),
    ("query database", "search data store"),
    ("send email", "transmit message")
]

for intent, action in examples:
    similarity = get_similarity(intent, action)
    deviation = 1.0 - similarity
    print(f"{intent} vs {action}: deviation={deviation:.3f}")
    
# Adjust threshold so benign paraphrases pass
```

## Examples

### Example 1: Detecting True Deviations

```python
from nethical import SafetyGovernance, AgentAction, MonitoringConfig

config = MonitoringConfig(use_semantic_intent=True)
governance = SafetyGovernance(config=config)

action = AgentAction(
    action_id="ex1",
    agent_id="agent1",
    stated_intent="read user preferences",
    actual_action="DELETE FROM users WHERE 1=1",  # Major deviation!
    action_type="database_query"
)

result = governance.evaluate_action(action)
# Semantic similarity: ~0.2 (very different)
# Deviation: ~0.8 (flagged!)
```

### Example 2: Allowing Benign Paraphrases

```python
action = AgentAction(
    action_id="ex2",
    agent_id="agent1",
    stated_intent="fetch customer records",
    actual_action="retrieve client data from database",
    action_type="query"
)

result = governance.evaluate_action(action)
# Semantic similarity: ~0.87 (very similar)
# Deviation: ~0.13 (allowed!)
```

### Example 3: Fallback Behavior

```python
# Without sentence-transformers installed
# Or if model loading fails

action = AgentAction(
    action_id="ex3",
    agent_id="agent1",
    stated_intent="process data",
    actual_action="handle information",
    action_type="processing"
)

result = governance.evaluate_action(action)
# Automatically uses lexical similarity (Jaccard)
# Warning logged about fallback
```

## Troubleshooting

### Model Not Loading

```python
# Check if semantic monitoring is available
from nethical.core.semantics import is_semantic_available

if not is_semantic_available():
    # Install sentence-transformers
    # pip install sentence-transformers
    pass
```

### High False Positives

```python
# Increase threshold (allow more deviation)
config = MonitoringConfig(
    intent_deviation_threshold=0.85  # More lenient
)
```

### High False Negatives

```python
# Decrease threshold (stricter matching)
config = MonitoringConfig(
    intent_deviation_threshold=0.65  # Stricter
)
```

### Performance Issues

```python
# Disable semantic monitoring if latency is critical
config = MonitoringConfig(
    use_semantic_intent=False  # Fall back to lexical
)
```

## Best Practices

1. **Start with Defaults**: Use `use_semantic_intent=True` and threshold `0.75`

2. **Monitor Metrics**: Track false positives/negatives in production

3. **A/B Testing**: Compare semantic vs lexical in parallel

4. **Preload Models**: In Docker or initialization code

5. **Set Limits**: Use `max_semantic_input_length` for very long texts

6. **Cache Wisely**: Clear cache if memory constrained

## Advanced Topics

### Custom Models

```python
from nethical.core.semantics import SentenceTransformerWrapper

# Use a different model
custom_model = SentenceTransformerWrapper(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Hybrid Scoring

Combine lexical and semantic:

```python
from nethical.core.semantics import get_similarity, _lexical_similarity

semantic_sim = get_similarity(intent, action)
lexical_sim = _lexical_similarity(intent, action)

# Weighted combination
hybrid_sim = 0.7 * semantic_sim + 0.3 * lexical_sim
```

## Further Reading

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Understanding Semantic Similarity](https://en.wikipedia.org/wiki/Semantic_similarity)
- [Nethical API Usage Guide](./API_USAGE.md)
- [Docker Deployment Guide](../README.md#running-via-docker)
