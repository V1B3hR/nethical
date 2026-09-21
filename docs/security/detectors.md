# Ultra-Low Latency Threat Detectors

Nethical's realtime threat detection system provides 5 specialized detectors optimized for production-grade, ultra-low latency cybersecurity operations.

## Overview

The detection framework achieves:
- **Ultra-low latency:** Individual detector targets from 15ms to 50ms
- **High throughput:** >5000 requests/second under load
- **Parallel execution:** Concurrent detection across all 5 detectors
- **Production-ready:** Async/await, type hints, comprehensive error handling

## Architecture

```
RealtimeThreatDetector (Unified Interface)
├── Shadow AI Detector (<20ms)
├── Deepfake Detector (<30ms)
├── Polymorphic Malware Detector (<50ms)
├── Prompt Injection Guard (<15ms)
└── AI vs AI Defender (<25ms)
```

## Detectors

### 1. Shadow AI Detector

**Target Latency:** <20ms

Detects unauthorized AI models running in infrastructure through:

#### Detection Methods
- **API Call Monitoring:** Identifies calls to LLM providers (OpenAI, Anthropic, Cohere, Google, HuggingFace, Replicate)
- **GPU Usage Analysis:** Monitors GPU memory and process patterns
- **Model File Detection:** Scans for model files (`.gguf`, `.bin`, `.safetensors`, `.onnx`)
- **Port Scanning:** Detects services on known AI ports (11434=Ollama, 8080=LM Studio, 8000=vLLM)

#### Configuration
```python
from nethical.detectors.realtime import ShadowAIDetector, ShadowAIDetectorConfig

config = ShadowAIDetectorConfig(
    authorized_apis={"api.openai.com", "api.anthropic.com"},
    authorized_models={"approved_model_123"},
    enable_api_detection=True,
    enable_gpu_detection=True,
    enable_model_file_detection=True,
    enable_port_scan=True,
)

detector = ShadowAIDetector(config)
```

#### Usage
```python
# Scan network traffic
result = await detector.scan({
    "urls": ["https://api.openai.com/v1/completions"],
    "connections": [{"port": 11434, "host": "localhost"}],
})

# Or use detect_violations for full context
violations = await detector.detect_violations({
    "network_traffic": {...},
    "system_info": {...},
    "file_system": {...},
})
```

### 2. Deepfake Detector

**Target Latency:** <30ms for images

Multi-modal deepfake detection across images, videos, and audio.

#### Detection Methods
- **Frequency Domain Analysis:** Detects GAN artifacts in frequency spectrum
- **Metadata Forensics:** Checks for missing/inconsistent EXIF data
- **Face Landmarks:** Analyzes facial landmark consistency
- **Lightweight CNN:** Quantized MobileNet-based detection (simulated)
- **Temporal Analysis:** For videos, checks frame-to-frame consistency
- **Audio Analysis:** Voice pattern anomalies for audio deepfakes

#### Configuration
```python
from nethical.detectors.realtime import DeepfakeDetector, DeepfakeDetectorConfig

config = DeepfakeDetectorConfig(
    image_threshold=0.75,
    video_threshold=0.70,
    audio_threshold=0.80,
    enable_frequency_analysis=True,
    enable_metadata_check=True,
    enable_face_landmarks=True,
)

detector = DeepfakeDetector(config)
```

#### Usage
```python
# Detect in image
result = await detector.detect(
    media=image_bytes,
    media_type="image"
)

# Detect in video
result = await detector.detect(
    media=video_bytes,
    media_type="video"
)

# With metadata context
violations = await detector.detect_violations({
    "media": image_bytes,
    "media_type": "image",
    "metadata": {
        "exif_data": {...},
        "software": "PhotoEditor",
    }
})
```

### 3. Polymorphic Malware Detector

**Target Latency:** <50ms

Detects mutating and obfuscated malware through behavioral analysis.

#### Detection Methods
- **Shannon Entropy Calculation:** Identifies encryption/packing (entropy >7.5)
- **Behavioral Analysis:** Monitors for code injection, privilege escalation, anti-debug
- **Syscall Pattern Matching:** Detects suspicious syscall sequences
- **Memory Access Patterns:** Identifies write-execute patterns and self-modifying code
- **Signature Matching:** Compares against known polymorphic malware families

#### Configuration
```python
from nethical.detectors.realtime import (
    PolymorphicMalwareDetector,
    PolymorphicDetectorConfig
)

config = PolymorphicDetectorConfig(
    high_entropy_threshold=7.5,
    suspicious_syscall_threshold=5,
    suspicious_syscalls={"execve", "ptrace", "mprotect", "mmap"},
)

detector = PolymorphicMalwareDetector(config)
```

#### Usage
```python
# Analyze executable
result = await detector.analyze(executable_bytes)

# With full context
violations = await detector.detect_violations({
    "executable_data": binary_data,
    "syscall_trace": ["mprotect", "mmap", "execve"],
    "behavior_log": [{"type": "code_injection"}],
    "memory_access": [{"type": "write_execute", "region": "0x1000"}],
})
```

### 4. Prompt Injection Guard

**Target Latency:** <15ms

Ultra-fast two-tier detection of prompt injection attacks.

#### Detection Methods
**Tier 1: Regex-based (2-5ms)**
- DAN variants
- "Ignore instructions" patterns
- System prompt leaking attempts
- Role-play jailbreaks
- Encoding tricks (base64, rot13)
- Delimiter confusion
- Privilege escalation keywords

**Tier 2: ML-based (15-25ms total)**
- Feature extraction (special char ratio, caps ratio, delimiter count)
- Lightweight classification
- Instruction keyword detection
- Negation pattern analysis

#### Configuration
```python
from nethical.detectors.realtime import (
    PromptInjectionGuard,
    PromptInjectionGuardConfig
)

config = PromptInjectionGuardConfig(
    enable_regex_tier=True,
    enable_ml_tier=True,
    regex_confidence=0.95,
    ml_threshold=0.6,
    max_prompt_length=10000,
)

guard = PromptInjectionGuard(config)
```

#### Usage
```python
# Check prompt
result = await guard.check("Ignore all previous instructions")

# Returns:
# {
#     "status": "success",
#     "is_injection": True,
#     "confidence": 0.95,
#     "injection_type": "ignore_instructions",
#     "violations": [...],
# }
```

### 5. AI vs AI Defender

**Target Latency:** <25ms

Defends against adversarial attacks on AI systems.

#### Detection Methods
- **Model Extraction Detection:** Identifies systematic boundary probing
- **Adversarial Example Detection:** Detects small perturbations and invisible characters
- **Membership Inference Detection:** Spots repeated similar queries with variations
- **Systematic Probing Detection:** Analyzes query distribution patterns
- **Rate Limiting:** Enforces query rate limits per client

#### Configuration
```python
from nethical.detectors.realtime import AIvsAIDefender, AIvsAIDefenderConfig

config = AIvsAIDefenderConfig(
    max_query_history=1000,
    similarity_threshold=0.85,
    rate_limit_threshold=100,  # queries per minute
    rate_limit_window=60,
    extraction_attempt_threshold=50,
)

defender = AIvsAIDefender(config)
```

#### Usage
```python
# Defend against attack
result = await defender.defend(
    query={"input": "test query"},
    query_history=[...],
)

# Returns:
# {
#     "status": "success",
#     "attack_detected": True,
#     "should_block": True,
#     "confidence": 0.85,
#     "violations": [...],
# }
```

## Unified Interface

Use `RealtimeThreatDetector` for a single entry point to all detectors.

```python
from nethical.detectors.realtime import RealtimeThreatDetector

# Initialize
detector = RealtimeThreatDetector()

# Single detector
result = await detector.evaluate_threat(
    input_data={"prompt": "test"},
    threat_type="prompt_injection"
)

# All detectors in parallel
result = await detector.evaluate_threat(
    input_data={...},
    threat_type="all",
    parallel=True
)

# Get metrics
metrics = detector.get_metrics()
# {
#     "total_detections": 1000,
#     "avg_latency_ms": 45.2,
#     "p95_latency_ms": 89.3,
#     "p99_latency_ms": 156.7,
#     "detectors": {...}
# }
```

## Performance Optimization

All detectors are optimized for production use:

### Async/Await
All detection methods are async for non-blocking I/O.

### Parallel Execution
Multiple detection methods run concurrently using `asyncio.gather()`.

### Lightweight Operations
- Regex-based pattern matching for fast screening
- Simulated ML inference (production would use ONNX Runtime)
- Minimal memory allocations

### Caching & Batching
See [performance-optimization.md](./performance-optimization.md) for request optimization.

## Benchmarking

### Individual Detector Benchmarks
```bash
python tests/benchmark_individual_detectors.py
```

### Load Testing
```bash
python tests/benchmark_latency.py
```

Expected results:
- **Throughput:** >5000 req/s
- **Avg Latency:** <50ms
- **P95 Latency:** <100ms
- **P99 Latency:** <200ms

## Testing

```bash
# Run all detector tests
pytest tests/detectors/realtime/ -v

# Run specific detector test
pytest tests/detectors/realtime/test_shadow_ai_detector.py -v

# Run with coverage
pytest tests/detectors/realtime/ --cov=nethical.detectors.realtime
```

## Integration Examples

### With API Server
```python
from fastapi import FastAPI
from nethical.detectors.realtime import RealtimeThreatDetector

app = FastAPI()
detector = RealtimeThreatDetector()

@app.post("/detect")
async def detect(data: dict):
    result = await detector.evaluate_threat(data, "all")
    return result
```

### With Streaming
```python
async def process_stream(stream):
    detector = RealtimeThreatDetector()

    async for item in stream:
        result = await detector.evaluate_threat(item, "all")
        if result["max_threat_score"] > 0.8:
            await block_request()
```

## Production Considerations

### Deployment
- Use ONNX Runtime for ML models in production
- Enable request optimization (caching, batching)
- Monitor latency metrics
- Set up alerting for violations

### Scaling
- Run multiple detector instances
- Use load balancing
- Consider edge deployment for lowest latency

### Monitoring
- Track P95/P99 latencies
- Monitor false positive/negative rates
- Alert on performance degradation

## Security

All detectors follow security best practices:
- Input validation and sanitization
- Resource limits (max query length, history size)
- Rate limiting
- Comprehensive error handling
- No secrets in logs

## Roadmap

Future enhancements:
- [ ] Real ONNX Runtime integration
- [ ] Advanced ML models (DistilBERT for prompt injection)
- [ ] GPU acceleration support
- [ ] Distributed detection (Redis-based state)
- [ ] Enhanced telemetry (Prometheus metrics)
- [ ] Model training pipelines
- [ ] Automated A/B testing

## License

MIT License - See [LICENSE](../LICENSE) for details.
