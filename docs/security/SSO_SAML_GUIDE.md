# SSO/SAML Integration Guide

## Overview

Nethical now supports Single Sign-On (SSO) authentication through multiple protocols:
- **SAML 2.0**: Enterprise identity provider integration
- **OAuth 2.0**: Social login and third-party authentication
- **OpenID Connect (OIDC)**: Modern identity layer on top of OAuth 2.0

## Table of Contents

1. [SAML 2.0 Configuration](#saml-20-configuration)
2. [OAuth 2.0 / OIDC Configuration](#oauth-20--oidc-configuration)
3. [User Provisioning](#user-provisioning)
4. [Attribute Mapping](#attribute-mapping)
5. [Production Deployment](#production-deployment)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## SAML 2.0 Configuration

### Prerequisites

Install the SAML library (optional, but recommended for production):

```bash
pip install python3-saml
```

### Basic Setup

```python
from nethical.security.sso import SSOManager, get_sso_manager

# Initialize SSO manager
sso = SSOManager(base_url="https://your-app.company.com")

# Configure SAML
config = sso.configure_saml(
    config_name="corporate_idp",
    sp_entity_id="https://your-app.company.com",
    idp_entity_id="https://idp.company.com",
    idp_sso_url="https://idp.company.com/sso",
    idp_x509_cert="""-----BEGIN CERTIFICATE-----
MIIDXTCCAkWgAwIBAgIJAKL...
-----END CERTIFICATE-----""",
)
```

### Advanced SAML Configuration

```python
from nethical.security.sso import SAMLConfig

# Custom attribute mapping
config = sso.configure_saml(
    config_name="okta",
    sp_entity_id="https://nethical.mycompany.com",
    idp_entity_id="http://www.okta.com/exk...",
    idp_sso_url="https://mycompany.okta.com/app/exk.../sso/saml",
    idp_x509_cert=open("okta_cert.pem").read(),
    attribute_mapping={
        'email': 'email',
        'firstName': 'first_name',
        'lastName': 'last_name',
        'department': 'department',
        'groups': 'groups',
    }
)

# Configure security settings
config.saml_config.want_assertions_signed = True
config.saml_config.want_messages_signed = True
config.saml_config.want_name_id_encrypted = False
```

### SAML Login Flow

```python
# 1. Initiate login (returns redirect URL)
login_url = sso.initiate_saml_login("corporate_idp")
# Redirect user to login_url

# 2. Handle callback (in your ACS endpoint)
from flask import request

@app.route('/auth/saml/acs', methods=['POST'])
def saml_acs():
    saml_response = request.form['SAMLResponse']
    
    try:
        user_data = sso.handle_saml_response(
            saml_response=saml_response,
            config_name="corporate_idp"
        )
        
        # user_data contains:
        # {
        #     'name_id': 'user@company.com',
        #     'email': 'user@company.com',
        #     'user_id': 'emp12345',
        #     'first_name': 'John',
        #     'last_name': 'Doe',
        #     'groups': ['engineers', 'admins']
        # }
        
        # Create session or issue JWT token
        create_user_session(user_data)
        return redirect('/dashboard')
        
    except SSOError as e:
        return f"Authentication failed: {e}", 401
```

## OAuth 2.0 / OIDC Configuration

### OAuth 2.0 (Generic)

```python
# Configure OAuth provider (e.g., GitHub)
config = sso.configure_oauth(
    config_name="github",
    client_id="your_github_client_id",
    client_secret="your_github_client_secret",
    authorization_url="https://github.com/login/oauth/authorize",
    token_url="https://github.com/login/oauth/access_token",
)
```

### OpenID Connect (OIDC)

```python
# Configure OIDC provider (e.g., Google)
config = sso.configure_oauth(
    config_name="google",
    client_id="your_google_client_id.apps.googleusercontent.com",
    client_secret="your_google_client_secret",
    authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
)
```

### OAuth Login Flow

```python
# 1. Initiate login
@app.route('/auth/oauth/login')
def oauth_login():
    auth_url = sso.initiate_oauth_login("google")
    return redirect(auth_url)

# 2. Handle callback
@app.route('/auth/oauth/callback')
def oauth_callback():
    authorization_response = request.url
    
    try:
        user_info = sso.handle_oauth_callback(
            authorization_response=authorization_response,
            config_name="google"
        )
        
        # user_info contains:
        # {
        #     'sub': '1234567890',
        #     'email': 'user@gmail.com',
        #     'name': 'John Doe',
        #     'picture': 'https://...'
        # }
        
        create_user_session(user_info)
        return redirect('/dashboard')
        
    except SSOError as e:
        return f"Authentication failed: {e}", 401
```

## User Provisioning

### Automatic User Creation

```python
# Enable auto-provisioning
config.auto_create_users = True
config.default_role = "viewer"

# In your callback handler
def handle_sso_callback(user_data):
    user_id = user_data.get('user_id') or user_data.get('email')
    
    # Check if user exists
    user = get_user(user_id)
    
    if not user and config.auto_create_users:
        # Create new user
        user = create_user(
            user_id=user_id,
            email=user_data.get('email'),
            name=user_data.get('name'),
            role=config.default_role,
        )
    
    return user
```

### Role Mapping from Groups

```python
# Map SAML/OIDC groups to internal roles
GROUP_ROLE_MAPPING = {
    'admins': 'admin',
    'operators': 'operator',
    'auditors': 'auditor',
    'engineers': 'viewer',
}

def map_groups_to_role(groups):
    """Map user groups to highest privilege role"""
    for group in groups:
        if group in GROUP_ROLE_MAPPING:
            role = GROUP_ROLE_MAPPING[group]
            if role == 'admin':
                return role
    return 'viewer'  # Default

# In callback
def handle_sso_callback(user_data):
    groups = user_data.get('groups', [])
    role = map_groups_to_role(groups)
    
    user = create_or_update_user(
        user_id=user_data['user_id'],
        email=user_data['email'],
        role=role,
    )
    return user
```

## Attribute Mapping

### SAML Attributes

Different IdPs use different attribute names. Configure mapping accordingly:

```python
# Okta
okta_mapping = {
    'email': 'email',
    'firstName': 'first_name',
    'lastName': 'last_name',
}

# Azure AD
azure_mapping = {
    'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress': 'email',
    'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname': 'first_name',
    'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname': 'last_name',
}

# OneLogin
onelogin_mapping = {
    'User.email': 'email',
    'User.FirstName': 'first_name',
    'User.LastName': 'last_name',
    'memberOf': 'groups',
}
```

### Custom Attribute Processing

```python
def process_user_attributes(raw_attributes, mapping):
    """Process and validate user attributes"""
    processed = {}
    
    for saml_attr, internal_field in mapping.items():
        if saml_attr in raw_attributes:
            value = raw_attributes[saml_attr]
            
            # Handle multi-value attributes
            if isinstance(value, list):
                value = value[0] if value else None
            
            # Validate email
            if internal_field == 'email' and value:
                if '@' not in value:
                    raise ValueError(f"Invalid email: {value}")
            
            processed[internal_field] = value
    
    return processed
```

## Production Deployment

### Security Checklist

- [ ] **Use HTTPS**: SSO requires secure connections
- [ ] **Validate Signatures**: Enable `want_assertions_signed` for SAML
- [ ] **Rotate Secrets**: Regularly update OAuth client secrets
- [ ] **Validate Redirect URIs**: Whitelist callback URLs
- [ ] **Log Authentication Events**: Monitor SSO login attempts
- [ ] **Implement Session Management**: Handle token expiration
- [ ] **Use State Parameter**: Prevent CSRF in OAuth flows

### Environment Configuration

```bash
# .env file
SSO_ENABLED=true
SSO_BASE_URL=https://nethical.company.com

# SAML
SAML_SP_ENTITY_ID=https://nethical.company.com
SAML_IDP_ENTITY_ID=https://idp.company.com
SAML_IDP_SSO_URL=https://idp.company.com/sso
SAML_IDP_CERT_PATH=/etc/nethical/certs/idp_cert.pem

# OAuth
OAUTH_CLIENT_ID=your_client_id
OAUTH_CLIENT_SECRET=your_client_secret
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install SAML dependencies
RUN apt-get update && apt-get install -y \
    xmlsec1 \
    libxmlsec1-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install python3-saml requests-oauthlib

# ... rest of Dockerfile
```

### Nginx Configuration

```nginx
# SSL termination for SAML
server {
    listen 443 ssl http2;
    server_name nethical.company.com;

    ssl_certificate /etc/ssl/certs/nethical.crt;
    ssl_certificate_key /etc/ssl/private/nethical.key;

    location /auth/saml/ {
        proxy_pass http://nethical-app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Examples

### Complete Flask Integration

```python
from flask import Flask, request, redirect, session
from nethical.security.sso import SSOManager, SSOError
from nethical.security.auth import AuthManager
import os

app = Flask(__name__)
app.secret_key = os.environ['SECRET_KEY']

# Initialize managers
sso = SSOManager(base_url=os.environ['BASE_URL'])
auth = AuthManager()

# Configure SAML
sso.configure_saml(
    config_name="corporate",
    sp_entity_id=os.environ['SAML_SP_ENTITY_ID'],
    idp_entity_id=os.environ['SAML_IDP_ENTITY_ID'],
    idp_sso_url=os.environ['SAML_IDP_SSO_URL'],
    idp_x509_cert=open(os.environ['SAML_IDP_CERT_PATH']).read(),
)

@app.route('/login')
def login():
    """Initiate SSO login"""
    login_url = sso.initiate_saml_login("corporate")
    return redirect(login_url)

@app.route('/auth/saml/acs', methods=['POST'])
def saml_acs():
    """Handle SAML assertion"""
    try:
        user_data = sso.handle_saml_response(
            saml_response=request.form['SAMLResponse'],
            config_name="corporate"
        )
        
        # Create JWT token
        access_token, _ = auth.create_access_token(user_data['user_id'])
        
        # Store in session
        session['access_token'] = access_token
        session['user_id'] = user_data['user_id']
        
        return redirect('/dashboard')
        
    except SSOError as e:
        app.logger.error(f"SAML authentication failed: {e}")
        return redirect('/login?error=authentication_failed')

@app.route('/dashboard')
def dashboard():
    """Protected route"""
    if 'access_token' not in session:
        return redirect('/login')
    
    # Verify token
    try:
        payload = auth.verify_token(session['access_token'])
        return f"Welcome, {payload.user_id}!"
    except Exception:
        return redirect('/login')
```

### FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from nethical.security.sso import SSOManager, SSOError

app = FastAPI()
sso = SSOManager(base_url="https://api.company.com")

# Configure OIDC (Google)
sso.configure_oauth(
    config_name="google",
    client_id=os.environ['GOOGLE_CLIENT_ID'],
    client_secret=os.environ['GOOGLE_CLIENT_SECRET'],
    authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
)

@app.get("/auth/login")
async def login():
    """Initiate OAuth login"""
    auth_url = sso.initiate_oauth_login("google")
    return RedirectResponse(auth_url)

@app.get("/auth/callback")
async def callback(request: Request):
    """Handle OAuth callback"""
    try:
        user_info = sso.handle_oauth_callback(
            authorization_response=str(request.url),
            config_name="google"
        )
        
        # Create session token
        # ... implementation
        
        return RedirectResponse("/dashboard")
        
    except SSOError as e:
        raise HTTPException(status_code=401, detail=str(e))
```

## Troubleshooting

### SAML Issues

**Problem**: "Invalid signature" error

```python
# Solution: Check certificate format
# Ensure certificate is PEM format without headers/footers in one line
# OR with proper newlines
```

**Problem**: "Assertion expired"

```python
# Solution: Check time synchronization
# Ensure server clocks are synchronized (use NTP)
# Adjust clock skew tolerance if needed
```

**Problem**: "Invalid audience"

```python
# Solution: Verify entity IDs match
# SP entity ID must match what's configured in IdP
assert config.saml_config.sp_entity_id == "https://your-exact-url.com"
```

### OAuth Issues

**Problem**: "redirect_uri_mismatch" error

```python
# Solution: Whitelist exact callback URL in OAuth provider
# URL must match exactly including protocol, domain, and path
redirect_uri = "https://nethical.company.com/auth/oauth/callback"
```

**Problem**: "invalid_client" error

```python
# Solution: Verify client credentials
# Check CLIENT_ID and CLIENT_SECRET are correct
# Ensure credentials are not expired
```

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('nethical.security.sso').setLevel(logging.DEBUG)
```

## Additional Resources

- [SAML 2.0 Specification](https://docs.oasis-open.org/security/saml/Post2.0/sstc-saml-tech-overview-2.0.html)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [OpenID Connect Specification](https://openid.net/specs/openid-connect-core-1_0.html)
- [python3-saml Documentation](https://github.com/onelogin/python3-saml)
- [requests-oauthlib Documentation](https://requests-oauthlib.readthedocs.io/)

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing documentation in `docs/security/`
- Review test examples in `tests/unit/test_sso.py`
