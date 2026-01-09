# Nethical API v1 - Backend Management API

Comprehensive REST API for AI safety and ethics governance with full backend management capabilities.

## Overview

API v1 adds enterprise-grade backend management features to Nethical, enabling dynamic configuration without server restarts, comprehensive audit trails, and real-time threat monitoring.

## Key Features

### 1. Role-Based Access Control (RBAC)
- **admin**: Full access to all operations
- **auditor**: Read-only access to logs and audit data
- **operator**: Can evaluate risk, but cannot modify configuration

### 2. Agent Management
Full CRUD operations for AI agent configuration:
- Create, read, update, and delete agents dynamically
- Configure trust levels, models, and parameters
- Filter and paginate agent lists
- Track agent creation and modifications

### 3. Policy Management
Full CRUD operations for governance policies:
- Create and manage custom policies
- Version control for policy changes
- Priority-based policy ordering
- Scope-based policy application (global, agent-specific, action-specific)

### 4. Audit Logs with Merkle Tree Verification
- Paginated access to immutable audit logs
- Filter by agent, event type, threat level, and date range
- Merkle tree-based integrity verification
- Cryptographic proof generation and validation

### 5. Real-time Threat Notifications
- WebSocket endpoint for low-latency threat notifications (<50ms)
- Server-Sent Events (SSE) alternative
- Subscribe to specific agents or threat types
- Event types: threat_detected, action_blocked, kill_switch_alarm

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from nethical.database import init_db; init_db()"

# Create admin user (Python)
from nethical.database import SessionLocal, User
from nethical.api.rbac import get_password_hash

db = SessionLocal()
admin = User(
    username="admin",
    email="admin@example.com",
    full_name="Admin User",
    hashed_password=get_password_hash("admin123"),
    role="admin",
    is_active=True
)
db.add(admin)
db.commit()
```

### Running the Server

```bash
# Using uvicorn directly
uvicorn nethical.api.v1.app:create_v1_app --factory --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn nethical.api.v1.app:create_v1_app --factory --reload
```

### Testing the API

```bash
# Run basic tests
python test_api_v1.py

# Run comprehensive test suite
pytest tests/api/v1/ -v
```

## API Documentation

### Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Authentication

All endpoints (except `/auth/login`) require authentication via Bearer token.

#### 1. Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": 1,
    "username": "admin",
    "email": "admin@example.com",
    "role": "admin"
  }
}
```

#### 2. Use Token in Requests

```bash
export TOKEN="your_access_token_here"

curl http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $TOKEN"
```

## Usage Examples

### Agent Management

#### Create Agent

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "gpt4-assistant",
    "name": "GPT-4 Assistant",
    "agent_type": "llm",
    "description": "Production GPT-4 assistant",
    "trust_level": 0.9,
    "configuration": {
      "model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 2000
    },
    "metadata": {
      "department": "engineering",
      "environment": "production"
    }
  }'
```

#### List Agents

```bash
# List all agents
curl http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $TOKEN"

# With pagination and filtering
curl "http://localhost:8000/api/v1/agents?page=1&per_page=10&status=active&agent_type=llm" \
  -H "Authorization: Bearer $TOKEN"
```

#### Update Agent

```bash
curl -X PATCH http://localhost:8000/api/v1/agents/gpt4-assistant \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "trust_level": 0.95,
    "configuration": {
      "temperature": 0.5
    }
  }'
```

#### Delete Agent

```bash
curl -X DELETE http://localhost:8000/api/v1/agents/gpt4-assistant \
  -H "Authorization: Bearer $TOKEN"
```

### Policy Management

#### Create Policy

```bash
curl -X POST http://localhost:8000/api/v1/policies \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": "data-access-policy",
    "name": "Data Access Policy",
    "description": "Controls access to sensitive data",
    "version": "1.0.0",
    "priority": 100,
    "rules": [
      {
        "id": "rule-1",
        "condition": "action_type == '\''data_access'\''",
        "action": "RESTRICT",
        "priority": 10,
        "description": "Restrict data access by default"
      }
    ],
    "scope": "global",
    "fundamental_laws": [22]
  }'
```

#### List Policies

```bash
curl "http://localhost:8000/api/v1/policies?page=1&per_page=20&status=active" \
  -H "Authorization: Bearer $TOKEN"
```

### Audit Logs

#### Get Audit Logs

```bash
# Basic query
curl http://localhost:8000/api/v1/audit/logs \
  -H "Authorization: Bearer $TOKEN"

# With filters
curl "http://localhost:8000/api/v1/audit/logs?agent_id=gpt4-assistant&threat_level=high&from_date=2026-01-01T00:00:00Z" \
  -H "Authorization: Bearer $TOKEN"
```

#### Get Merkle Tree

```bash
curl "http://localhost:8000/api/v1/audit/merkle-tree?limit=100" \
  -H "Authorization: Bearer $TOKEN"
```

#### Verify Merkle Proof

```bash
curl -X POST http://localhost:8000/api/v1/audit/verify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "log_id": "log-001",
    "merkle_path": [
      ["hash1", "left"],
      ["hash2", "right"]
    ]
  }'
```

### Real-time Threats

#### WebSocket Connection

```javascript
const token = 'your_token_here';
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/threats?token=${token}&agent_id=gpt4-assistant`);

ws.onopen = () => {
  console.log('Connected to threat stream');
  // Send heartbeat
  ws.send('ping');
};

ws.onmessage = (event) => {
  const threat = JSON.parse(event.data);
  console.log('Threat detected:', threat);
  // Handle threat notification
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

#### Server-Sent Events

```javascript
const token = 'your_token_here';
const eventSource = new EventSource(`http://localhost:8000/api/v1/sse/threats?agent_id=gpt4-assistant&threat_type=prompt_injection`);

eventSource.addEventListener('threat', (event) => {
  const threat = JSON.parse(event.data);
  console.log('Threat detected:', threat);
});

eventSource.addEventListener('heartbeat', (event) => {
  console.log('Heartbeat received');
});
```

## Configuration

### Environment Variables

```bash
# Database
export DATABASE_URL="postgresql://user:password@localhost/nethical"  # Or sqlite:///./nethical.db

# JWT Authentication
export NETHICAL_SECRET_KEY="your-secret-key-here"
export ACCESS_TOKEN_EXPIRE_MINUTES="30"

# CORS (comma-separated origins)
export NETHICAL_CORS_ALLOW_ORIGINS="https://app.example.com,https://admin.example.com"

# Server
export UVICORN_HOST="0.0.0.0"
export UVICORN_PORT="8000"
```

### Database Setup

#### SQLite (Default)

```bash
# Uses local file: nethical.db
# No additional setup required
```

#### PostgreSQL

```bash
# Set DATABASE_URL
export DATABASE_URL="postgresql://user:password@localhost:5432/nethical"

# Initialize database
python -c "from nethical.database import init_db; init_db()"
```

## Security Considerations

### Production Checklist

- [ ] Change default admin password
- [ ] Set strong `NETHICAL_SECRET_KEY`
- [ ] Configure specific CORS origins (not wildcard)
- [ ] Use HTTPS/TLS for all connections
- [ ] Use PostgreSQL or other production database (not SQLite)
- [ ] Enable rate limiting
- [ ] Set up database backups
- [ ] Monitor audit logs regularly
- [ ] Rotate JWT secrets periodically
- [ ] Implement password policies
- [ ] Enable two-factor authentication (future enhancement)

### RBAC Best Practices

1. **Principle of Least Privilege**: Grant minimum necessary permissions
2. **Separate Concerns**: Use different accounts for different roles
3. **Audit Trail**: Monitor who does what via audit logs
4. **Token Expiration**: Keep token expiration times short (15-30 minutes)
5. **Password Policy**: Enforce strong passwords for all users

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

```bash
# Verify token is not expired
# Tokens expire after ACCESS_TOKEN_EXPIRE_MINUTES (default: 30)
# Get a new token by logging in again
```

#### 2. Permission Denied (403)

```bash
# Check your user role
# Only admins can create/update/delete agents and policies
# Auditors can only read audit logs
# Operators cannot modify configuration
```

#### 3. Database Connection Issues

```bash
# For SQLite: Ensure file permissions are correct
# For PostgreSQL: Verify connection string and database exists
# Check DATABASE_URL environment variable
```

#### 4. WebSocket Connection Failures

```bash
# Ensure token is passed in query parameter
# Check firewall allows WebSocket connections
# Verify WebSocket is not blocked by proxy/load balancer
```

## Development

### Running Tests

```bash
# Run basic functionality tests
python test_api_v1.py

# Run comprehensive test suite
pytest tests/api/v1/ -v

# Run specific test file
pytest tests/api/v1/test_agents.py -v

# Run with coverage
pytest tests/api/v1/ --cov=nethical.api.v1 --cov-report=html
```

### Adding New Endpoints

1. Create route module in `nethical/api/v1/routes/`
2. Import in `nethical/api/v1/routes/__init__.py`
3. Include router in `nethical/api/v1/app.py`
4. Add to `openapi-v1.yaml` documentation
5. Write tests in `tests/api/v1/`

### Database Migrations

```bash
# For production, use Alembic for migrations
# See deploy/postgres/migrations/ for examples
```

## API Reference

See `openapi-v1.yaml` for complete API specification.

## Support

- **GitHub Issues**: https://github.com/V1B3hR/nethical/issues
- **Documentation**: https://github.com/V1B3hR/nethical/tree/main/docs
- **Email**: support@nethical.dev

## License

MIT License - see LICENSE file for details.
