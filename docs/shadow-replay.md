# Shadow Traffic Replay Tool

## Overview

The Shadow Traffic Replay Tool allows you to safely mirror production traffic patterns into an isolated staging environment for realistic end-to-end validation, reliability testing, and security rehearsals without impacting external users.

## Purpose

- **Production Validation**: Test how staging handles real-world traffic patterns before deploying to production
- **Reliability Testing**: Validate system behavior under realistic load conditions
- **Security Rehearsals**: Practice incident response scenarios with production-like traffic
- **Performance Benchmarking**: Compare staging performance against production baselines
- **Regression Testing**: Ensure new deployments handle existing traffic patterns correctly

## Safety Considerations

### Built-in Safety Features

1. **Dry-Run Mode (Default)**: Requests are processed and logged but never sent
2. **Method Filtering**: State-changing methods (POST, PUT, PATCH, DELETE) are blocked by default
3. **Production Detection**: Automatically detects and blocks production-like target URLs
4. **Environment Checks**: Validates environment variables to prevent accidental production replay
5. **Explicit Opt-in**: Dangerous operations require explicit flags (`--send`, `--allow-modifying`, `--force`)
6. **Shadow Headers**: All replayed requests include `X-Nethical-Shadow: true` header for identification

### Safety Checklist

Before running shadow replay with `--send` flag, ensure:

- [ ] **Isolated Staging Environment**
  - Staging uses a separate database (not production data)
  - Staging connects to test/sandbox external services (payment, email, etc.)
  - No production users can see or be affected by staging traffic
  
- [ ] **Configuration Validation**
  - Staging base URL is correct and points to staging (not production)
  - Authentication uses staging-only credentials (never production keys)
  - Feature flags are set appropriately for testing environment
  
- [ ] **Team Coordination**
  - Team members are aware of the replay test
  - Monitoring is in place for staging environment
  - On-call engineer is available if issues arise
  
- [ ] **Risk Assessment**
  - Understand what state changes may occur (if using `--allow-modifying`)
  - Have a rollback plan if issues are discovered
  - Start with small traffic samples before full replay
  
- [ ] **Test Progression**
  - Always run `--dry-run` mode first to validate setup
  - Test with read-only traffic before allowing modifications
  - Gradually increase traffic volume and complexity

## Installation

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Install dependencies
pip install -r requirements-dev.txt
```

The tool requires:
- `requests>=2.31.0` - HTTP client library
- `requests-mock>=1.11.0` - For testing (dev only)

### Verify Installation

```bash
# Check that the tool is accessible
python scripts/shadow_replay.py --help
```

## Usage

### Basic Command Structure

```bash
python scripts/shadow_replay.py \
  --input <traffic_file> \
  --staging-url <staging_base_url> \
  [options]
```

### Common Usage Patterns

#### 1. Safe Dry-Run (Recommended First Step)

Process traffic file and validate configuration without sending any requests:

```bash
python scripts/shadow_replay.py \
  --input examples/shadow/sample_traffic.har \
  --staging-url https://staging.nethical.example.com \
  --dry-run
```

Expected output:
```
Shadow Replay Tool initialized:
  Target: https://staging.nethical.example.com
  Dry-run: True
  Skip methods: ['POST', 'PUT', 'PATCH', 'DELETE']
  ...
[DRY-RUN] Would send: GET https://staging.nethical.example.com/api/users
...
Replay complete!
  Total requests: 10
  Processed: 6
  Skipped: 4
  Sent: 0
```

#### 2. Read-Only Traffic Replay

Send only safe read-only requests (GET, HEAD, OPTIONS):

```bash
python scripts/shadow_replay.py \
  --input production_traffic_sample.har \
  --staging-url https://staging.nethical.example.com \
  --send \
  --rps 10 \
  --auth "staging-api-key-here" \
  --report staging_replay_report.json
```

#### 3. Full Replay with State Changes (Caution!)

Allow state-changing methods (use with extreme caution):

```bash
python scripts/shadow_replay.py \
  --input production_traffic.har \
  --staging-url https://staging.nethical.example.com \
  --send \
  --allow-modifying \
  --rps 5 \
  --auth "staging-api-key" \
  --retries 3 \
  --report full_replay_report.json
```

⚠️ **Warning**: This can modify staging data. Ensure staging is truly isolated!

#### 4. Debug Mode

Enable detailed logging for troubleshooting:

```bash
python scripts/shadow_replay.py \
  --input traffic.json \
  --staging-url https://staging.nethical.example.com \
  --dry-run \
  --debug
```

#### 5. Custom Method Filtering

Skip only specific methods:

```bash
python scripts/shadow_replay.py \
  --input traffic.har \
  --staging-url https://staging.nethical.example.com \
  --send \
  --skip-methods DELETE PATCH \
  --rps 10
```

## Command-Line Options

### Required Arguments

| Option | Description | Example |
|--------|-------------|---------|
| `--input`, `-i` | Path to traffic file (HAR or JSON) | `--input traffic.har` |
| `--staging-url` | Base URL of staging environment | `--staging-url https://staging.example.com` |

### Operation Mode

| Option | Description | Default |
|--------|-------------|---------|
| `--send` | Actually send requests to staging | Not set (dry-run) |
| `--dry-run` | Process but don't send requests | True |

### Safety Options

| Option | Description | Default |
|--------|-------------|---------|
| `--skip-methods` | HTTP methods to skip | POST PUT PATCH DELETE |
| `--allow-modifying` | Explicitly allow state-changing methods | False |
| `--force` | Override production safety checks | False (not recommended) |

### Performance & Reliability

| Option | Description | Default |
|--------|-------------|---------|
| `--rps` | Requests per second rate limit | None (no limit) |
| `--retries` | Number of retry attempts for failures | 3 |

### Authentication

| Option | Description | Default |
|--------|-------------|---------|
| `--auth` | Authorization token/key | None |

### Output & Logging

| Option | Description | Default |
|--------|-------------|---------|
| `--report` | Path to save replay report JSON | `replay_report.json` |
| `--debug` | Enable debug logging | False |

## Input Formats

### HAR (HTTP Archive) Format

Standard HAR format exported from browser DevTools or proxy tools:

```json
{
  "log": {
    "version": "1.2",
    "entries": [
      {
        "request": {
          "method": "GET",
          "url": "https://api.example.com/users",
          "headers": [
            {"name": "Accept", "value": "application/json"}
          ],
          "queryString": [
            {"name": "page", "value": "1"}
          ]
        }
      }
    ]
  }
}
```

### Custom JSON Format

Simplified JSON format for easier manual creation:

```json
[
  {
    "method": "GET",
    "url": "/api/users",
    "headers": {
      "Accept": "application/json"
    },
    "body": null,
    "query_params": {
      "page": "1"
    }
  },
  {
    "method": "POST",
    "url": "/api/users",
    "headers": {
      "Content-Type": "application/json"
    },
    "body": "{\"name\": \"Test User\"}",
    "query_params": {}
  }
]
```

## Output Report

The tool generates a comprehensive JSON report with replay statistics:

```json
{
  "total_requests": 100,
  "processed": 75,
  "skipped": 25,
  "sent": 75,
  "successful": 70,
  "failed": 5,
  "errors": 3,
  "skipped_reasons": {
    "Method POST in skip list": 20,
    "Method DELETE in skip list": 5
  },
  "response_samples": [
    {
      "request": {
        "method": "GET",
        "url": "https://staging.example.com/api/users"
      },
      "response": {
        "status_code": 200,
        "latency_ms": 45.2
      }
    }
  ],
  "error_samples": [
    {
      "request": {
        "method": "GET",
        "url": "https://staging.example.com/api/broken"
      },
      "error": "Connection timeout"
    }
  ],
  "total_duration_seconds": 120.5
}
```

### Report Fields

- **total_requests**: Total number of requests in input file
- **processed**: Requests that passed filtering and were processed
- **skipped**: Requests skipped due to method filtering or other rules
- **sent**: Requests actually sent to staging (0 in dry-run mode)
- **successful**: Requests with 2xx status codes
- **failed**: Requests with non-2xx status codes or errors
- **errors**: Requests that encountered exceptions
- **skipped_reasons**: Breakdown of why requests were skipped
- **response_samples**: Sample of successful responses (up to 10)
- **error_samples**: Sample of failed requests (up to 10)
- **total_duration_seconds**: Total time taken for replay

## Step-by-Step Deployment Guide

### Step 1: Prepare Staging Environment

1. **Provision isolated staging infrastructure**
   ```bash
   # Example: Deploy staging with Kubernetes
   kubectl apply -f deploy/staging/
   ```

2. **Configure isolated backing services**
   - Separate database instance (not production)
   - Test/sandbox external API credentials
   - Isolated message queues, caches, etc.

3. **Set appropriate feature flags**
   ```bash
   # Example: Disable production-only features
   export FEATURE_PRODUCTION_ALERTS=false
   export FEATURE_EXTERNAL_NOTIFICATIONS=false
   ```

### Step 2: Export Production Traffic

Using browser DevTools:
1. Open DevTools (F12)
2. Go to Network tab
3. Perform actions you want to replay
4. Right-click → "Save all as HAR"

Using Charles Proxy or similar:
1. Record session
2. Export as HAR format

Using nginx/Apache logs:
1. Parse logs into custom JSON format
2. Extract method, path, headers, body

### Step 3: Validate Setup (Dry-Run)

```bash
python scripts/shadow_replay.py \
  --input exported_traffic.har \
  --staging-url https://staging.nethical.example.com \
  --dry-run \
  --debug \
  --report dry_run_report.json
```

Review the output:
- Verify staging URL is correct
- Check that appropriate methods are skipped
- Confirm request rewriting looks correct

### Step 4: Test with Read-Only Traffic

```bash
python scripts/shadow_replay.py \
  --input exported_traffic.har \
  --staging-url https://staging.nethical.example.com \
  --send \
  --rps 10 \
  --auth "$STAGING_API_KEY" \
  --report readonly_replay_report.json
```

Monitor staging:
- Watch error logs for unexpected failures
- Check metrics for performance issues
- Validate responses are as expected

### Step 5: Review Results

```bash
# View report
cat readonly_replay_report.json | jq .

# Check for errors
jq '.error_samples' readonly_replay_report.json

# Calculate success rate
jq '(.successful / .sent * 100)' readonly_replay_report.json
```

### Step 6: Full Replay (If Needed)

Only if you need to test state-changing operations:

```bash
python scripts/shadow_replay.py \
  --input exported_traffic.har \
  --staging-url https://staging.nethical.example.com \
  --send \
  --allow-modifying \
  --rps 5 \
  --auth "$STAGING_API_KEY" \
  --report full_replay_report.json
```

⚠️ **Monitor closely**: State changes can affect staging data!

### Step 7: Analyze and Act

Review the full replay report:
1. **Check error rate**: Are there more errors than in production?
2. **Compare latency**: Is staging slower than production?
3. **Validate functionality**: Do responses match expected behavior?
4. **Security review**: Any unexpected authentication or authorization issues?

If issues found:
- Fix bugs in staging deployment
- Update configuration
- Re-run replay to validate fixes
- Repeat until staging performs acceptably

If all tests pass:
- Promote changes to production
- Monitor production deployment
- Keep replay reports for comparison

## Integration with CI/CD

### GitHub Actions Example

Create `.github/workflows/shadow-replay-validation.yml`:

```yaml
name: Shadow Replay Validation

on:
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      traffic_file:
        description: 'Traffic file to replay'
        required: false
        default: 'examples/shadow/sample_traffic.har'

jobs:
  shadow-replay:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Deploy to staging
        run: |
          # Your staging deployment commands
          ./deploy/deploy-staging.sh
      
      - name: Wait for staging ready
        run: |
          timeout 300 bash -c 'until curl -f ${{ secrets.STAGING_URL }}/health; do sleep 5; done'
      
      - name: Run shadow replay
        run: |
          python scripts/shadow_replay.py \
            --input ${{ github.event.inputs.traffic_file || 'examples/shadow/sample_traffic.har' }} \
            --staging-url ${{ secrets.STAGING_URL }} \
            --send \
            --rps 10 \
            --auth ${{ secrets.STAGING_API_KEY }} \
            --report replay_report.json
      
      - name: Check replay results
        run: |
          # Fail if error rate > 5%
          ERROR_RATE=$(jq '(.errors / .sent * 100)' replay_report.json)
          if (( $(echo "$ERROR_RATE > 5" | bc -l) )); then
            echo "Error rate too high: $ERROR_RATE%"
            exit 1
          fi
      
      - name: Upload replay report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: shadow-replay-report
          path: replay_report.json
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    environment {
        STAGING_URL = credentials('staging-url')
        STAGING_API_KEY = credentials('staging-api-key')
    }
    
    stages {
        stage('Deploy Staging') {
            steps {
                sh './deploy/deploy-staging.sh'
            }
        }
        
        stage('Shadow Replay') {
            steps {
                sh '''
                    python scripts/shadow_replay.py \
                        --input traffic_export.har \
                        --staging-url ${STAGING_URL} \
                        --send \
                        --rps 10 \
                        --auth ${STAGING_API_KEY} \
                        --report replay_report.json
                '''
            }
        }
        
        stage('Validate Results') {
            steps {
                script {
                    def report = readJSON file: 'replay_report.json'
                    def errorRate = (report.errors / report.sent) * 100
                    
                    if (errorRate > 5) {
                        error("Shadow replay error rate too high: ${errorRate}%")
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'replay_report.json'
        }
    }
}
```

## Troubleshooting

### Common Issues

#### 1. "SAFETY ERROR: Target URL contains production pattern"

**Cause**: Tool detected production-like URL to prevent accidents

**Solution**: 
- Verify you meant to use staging URL
- If staging URL legitimately contains "prod" in name, use `--force` flag (carefully!)

```bash
python scripts/shadow_replay.py ... --force
```

#### 2. "Error: requests library is required"

**Cause**: Missing dependency

**Solution**:
```bash
pip install requests
# or
pip install -r requirements-dev.txt
```

#### 3. High error rate in replay report

**Possible causes**:
- Staging environment not fully deployed
- Database schema mismatch
- External service credentials incorrect
- Authentication token expired

**Debug**:
```bash
# Run with debug logging
python scripts/shadow_replay.py ... --debug

# Check staging logs
kubectl logs -f deployment/staging-api

# Test staging health endpoint
curl https://staging.example.com/health
```

#### 4. Rate limiting too aggressive

**Cause**: Staging environment has stricter rate limits

**Solution**: Reduce `--rps` value
```bash
python scripts/shadow_replay.py ... --rps 5
```

#### 5. Authentication failures

**Cause**: Wrong API key or token format

**Solution**: 
- Verify staging API key is correct
- Check if token needs different format (e.g., "Bearer" vs "Token")
- Test auth manually:
  ```bash
  curl -H "Authorization: Bearer your-token" https://staging.example.com/api/test
  ```

## Advanced Topics

### Rate Limiting Strategy

Match production traffic patterns:

```bash
# Calculate production RPS from logs
awk '{print $4}' access.log | cut -d: -f1-2 | uniq -c

# Use slightly lower RPS for safety
python scripts/shadow_replay.py ... --rps 8
```

### Traffic Sampling

For large production traffic:

```bash
# Sample 10% of requests
jq '[.log.entries[] | select((now | floor) % 10 == 0)]' full_traffic.har > sampled.json

python scripts/shadow_replay.py --input sampled.json ...
```

### Custom Traffic Generation

Create synthetic traffic for specific scenarios:

```python
# generate_traffic.py
import json

traffic = []
for i in range(100):
    traffic.append({
        "method": "GET",
        "url": f"/api/users/{i}",
        "headers": {"Accept": "application/json"},
        "body": None,
        "query_params": {}
    })

with open("custom_traffic.json", "w") as f:
    json.dump(traffic, f)
```

### Parallel Replays

Test multiple scenarios concurrently:

```bash
# Terminal 1: Read-only traffic
python scripts/shadow_replay.py --input readonly.har ... &

# Terminal 2: API v2 traffic  
python scripts/shadow_replay.py --input apiv2.har ... &

# Wait for both
wait
```

## Security Considerations

### Never Replay to Production

The tool includes safety checks, but always:
- Double-check target URL
- Review command before pressing Enter
- Use staging credentials only
- Monitor first few requests

### Sanitize Traffic Data

Before exporting production traffic:
- Remove sensitive headers (auth tokens, cookies)
- Redact PII in request bodies
- Strip out internal-only data

Example sanitization script:
```python
import json

with open("production.har", "r") as f:
    har = json.load(f)

for entry in har["log"]["entries"]:
    # Remove sensitive headers
    entry["request"]["headers"] = [
        h for h in entry["request"]["headers"]
        if h["name"] not in ["Authorization", "Cookie", "X-API-Key"]
    ]

with open("sanitized.har", "w") as f:
    json.dump(har, f)
```

### Audit Replay Activity

Log all replay runs:
```bash
python scripts/shadow_replay.py ... 2>&1 | tee replay_$(date +%Y%m%d_%H%M%S).log
```

## Best Practices

1. **Always start with dry-run**: Never skip the dry-run validation step
2. **Progressive testing**: Read-only → Small sample → Full replay
3. **Monitor staging**: Watch logs and metrics during replay
4. **Isolate staging**: Ensure complete isolation from production
5. **Document runs**: Keep logs and reports for future reference
6. **Automate validation**: Integrate with CI/CD for consistent testing
7. **Team communication**: Notify team before running large replays
8. **Regular testing**: Run shadow replays regularly, not just before releases

## Support and Contributing

For issues, questions, or contributions:
- GitHub Issues: [V1B3hR/nethical/issues](https://github.com/V1B3hR/nethical/issues)
- Documentation: [docs/](https://github.com/V1B3hR/nethical/tree/main/docs)
- Security concerns: See [SECURITY.md](../SECURITY.md)

## License

See [LICENSE](../LICENSE) file for details.
