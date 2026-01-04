# Multi-Factor Authentication (MFA) Guide

## Overview

Nethical supports Multi-Factor Authentication (MFA) to provide enhanced security for user accounts, with mandatory enforcement for administrative operations.

**Supported Methods:**
- **TOTP (Time-based One-Time Password)**: Compatible with Google Authenticator, Authy, Microsoft Authenticator
- **Backup Codes**: Recovery codes for account access when primary MFA is unavailable
- **SMS** (stub): Framework for SMS-based verification (requires external service integration)

## Table of Contents

1. [Quick Start](#quick-start)
2. [TOTP Setup](#totp-setup)
3. [Backup Codes](#backup-codes)
4. [Admin MFA Enforcement](#admin-mfa-enforcement)
5. [Integration Examples](#integration-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

For TOTP support with QR code generation:

```bash
pip install pyotp qrcode[pil]
```

*Note: MFA works without these dependencies using fallback implementations, but they are recommended for production.*

### Basic Usage

```python
from nethical.security.mfa import MFAManager, get_mfa_manager

# Initialize MFA manager
mfa = MFAManager()

# Setup TOTP for a user
secret, provisioning_uri, backup_codes = mfa.setup_totp("user123", issuer="MyApp")

# Display QR code to user
qr_code_uri = mfa.get_qr_code_data_uri(provisioning_uri)

# Enable MFA
mfa.enable_mfa("user123")

# Verify code
is_valid = mfa.verify_totp("user123", "123456")
```

## TOTP Setup

### Step-by-Step User Enrollment

```python
from nethical.security.mfa import MFAManager

def enroll_user_mfa(user_id: str):
    """Enroll a user in MFA"""
    mfa = MFAManager()
    
    # 1. Generate TOTP secret and provisioning URI
    secret, provisioning_uri, backup_codes = mfa.setup_totp(
        user_id=user_id,
        issuer="Nethical AI"  # Shows in authenticator app
    )
    
    # 2. Generate QR code for easy scanning
    qr_data_uri = mfa.get_qr_code_data_uri(provisioning_uri)
    
    # 3. Return setup information to display to user
    return {
        'qr_code': qr_data_uri,
        'secret': secret,  # For manual entry
        'backup_codes': backup_codes,  # User must save these!
        'provisioning_uri': provisioning_uri,
    }
```

### Display QR Code to User

#### Flask/HTML Example

```python
from flask import Flask, render_template, session

@app.route('/mfa/setup')
def mfa_setup():
    user_id = session['user_id']
    setup_data = enroll_user_mfa(user_id)
    
    # Store secret temporarily until verified
    session['mfa_temp_secret'] = setup_data['secret']
    
    return render_template('mfa_setup.html', 
                         qr_code=setup_data['qr_code'],
                         secret=setup_data['secret'],
                         backup_codes=setup_data['backup_codes'])
```

```html
<!-- mfa_setup.html -->
<div class="mfa-setup">
    <h2>Set Up Two-Factor Authentication</h2>
    
    <div class="step">
        <h3>Step 1: Scan QR Code</h3>
        <img src="{{ qr_code }}" alt="QR Code">
        <p>Or enter this code manually: <code>{{ secret }}</code></p>
    </div>
    
    <div class="step">
        <h3>Step 2: Save Backup Codes</h3>
        <div class="backup-codes">
            {% for code in backup_codes %}
                <code>{{ code }}</code>
            {% endfor %}
        </div>
        <p><strong>Important:</strong> Save these codes in a secure place!</p>
    </div>
    
    <div class="step">
        <h3>Step 3: Verify Code</h3>
        <form action="/mfa/verify-setup" method="POST">
            <input type="text" name="code" placeholder="Enter 6-digit code" required>
            <button type="submit">Verify & Enable</button>
        </form>
    </div>
</div>
```

### Verify and Enable MFA

```python
@app.route('/mfa/verify-setup', methods=['POST'])
def verify_mfa_setup():
    user_id = session['user_id']
    code = request.form['code']
    
    mfa = get_mfa_manager()
    
    # Verify the code before enabling
    if mfa.verify_totp(user_id, code):
        # Enable MFA for this user
        mfa.enable_mfa(user_id)
        
        # Clear temporary secret
        session.pop('mfa_temp_secret', None)
        
        flash('Two-factor authentication enabled successfully!', 'success')
        return redirect('/dashboard')
    else:
        flash('Invalid code. Please try again.', 'error')
        return redirect('/mfa/setup')
```

## Backup Codes

### Generation

Backup codes are automatically generated during TOTP setup:

```python
secret, uri, backup_codes = mfa.setup_totp("user123")

# backup_codes contains 10 codes in format: XXXX-XXXX
# Example: ['AB3D-FGH9', 'KM5P-QR8T', ...]
```

### Regeneration

Users should be able to regenerate backup codes (invalidates old ones):

```python
@app.route('/mfa/regenerate-codes', methods=['POST'])
def regenerate_backup_codes():
    user_id = session['user_id']
    
    # Require current MFA verification
    code = request.form['current_code']
    mfa = get_mfa_manager()
    
    if not mfa.verify_totp(user_id, code):
        return jsonify({'error': 'Invalid code'}), 401
    
    # Regenerate codes
    new_codes = mfa.regenerate_backup_codes(user_id)
    
    return jsonify({'backup_codes': new_codes})
```

### Verification

```python
def verify_backup_code(user_id: str, code: str) -> bool:
    """Verify a backup code (one-time use)"""
    mfa = get_mfa_manager()
    
    # Backup codes are consumed after use
    is_valid = mfa.verify_backup_code(user_id, code)
    
    if is_valid:
        # Warn user they've used a backup code
        send_notification(user_id, 
            "A backup code was used to access your account. "
            "Please ensure this was you.")
    
    return is_valid
```

## Admin MFA Enforcement

### Enable Admin MFA Requirement

```python
from nethical.security.mfa import get_mfa_manager
from nethical.core.rbac import get_rbac_manager, Role

mfa = get_mfa_manager()
rbac = get_rbac_manager()

# Enforce MFA for all admin operations (enabled by default)
mfa.require_mfa_for_admin(True)

# Check if admin needs MFA
def check_admin_access(user_id: str) -> bool:
    """Check if admin user has MFA enabled"""
    user_role = rbac.get_user_role(user_id)
    
    if user_role == Role.ADMIN:
        if mfa.check_admin_mfa_required(user_id, "admin"):
            # Admin needs to set up MFA
            return False
    
    return True
```

### Decorator for Admin Operations

```python
from functools import wraps
from flask import session, redirect, flash

def require_admin_mfa(f):
    """Decorator to enforce MFA for admin operations"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        
        if not user_id:
            return redirect('/login')
        
        mfa = get_mfa_manager()
        rbac = get_rbac_manager()
        
        user_role = rbac.get_user_role(user_id)
        
        if user_role == Role.ADMIN:
            # Check if MFA is enabled
            if not mfa.is_mfa_enabled(user_id):
                flash('Admin accounts require MFA. Please set up MFA.', 'warning')
                return redirect('/mfa/setup')
            
            # Check if MFA was verified in this session
            if not session.get('mfa_verified'):
                return redirect('/mfa/verify')
        
        return f(*args, **kwargs)
    return decorated_function

# Usage
@app.route('/admin/users')
@require_admin_mfa
def manage_users():
    """Admin operation requiring MFA"""
    return render_template('admin/users.html')
```

### MFA Challenge for Admin Operations

```python
@app.route('/mfa/verify')
def mfa_challenge():
    """Challenge user for MFA code"""
    return render_template('mfa_verify.html')

@app.route('/mfa/verify', methods=['POST'])
def mfa_verify():
    """Verify MFA code"""
    user_id = session['user_id']
    code = request.form['code']
    
    mfa = get_mfa_manager()
    
    # Try TOTP or backup code
    if mfa.verify_mfa(user_id, code):
        # Mark session as MFA verified
        session['mfa_verified'] = True
        session['mfa_verified_at'] = datetime.now(timezone.utc)
        
        # Redirect to original destination
        next_url = session.get('next_url', '/dashboard')
        return redirect(next_url)
    else:
        flash('Invalid code. Please try again.', 'error')
        return redirect('/mfa/verify')
```

## Integration Examples

### Complete Login Flow with MFA

```python
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # 1. Verify credentials
        user = authenticate(username, password)
        if not user:
            flash('Invalid credentials', 'error')
            return redirect('/login')
        
        # 2. Check if MFA is enabled
        mfa = get_mfa_manager()
        if mfa.is_mfa_enabled(user.id):
            # Store user ID and redirect to MFA
            session['pending_user_id'] = user.id
            return redirect('/mfa/verify')
        else:
            # No MFA, create session
            session['user_id'] = user.id
            return redirect('/dashboard')
    
    return render_template('login.html')

@app.route('/mfa/verify', methods=['POST'])
def verify_mfa_login():
    """Verify MFA during login"""
    user_id = session.get('pending_user_id')
    if not user_id:
        return redirect('/login')
    
    code = request.form['code']
    mfa = get_mfa_manager()
    
    if mfa.verify_mfa(user_id, code):
        # MFA verified, create session
        session.pop('pending_user_id')
        session['user_id'] = user_id
        session['mfa_verified'] = True
        
        return redirect('/dashboard')
    else:
        flash('Invalid code', 'error')
        return redirect('/mfa/verify')
```

### API Authentication with MFA

```python
from flask import jsonify, request
from nethical.security.auth import get_auth_manager
from nethical.security.mfa import get_mfa_manager

@app.route('/api/login', methods=['POST'])
def api_login():
    """API login endpoint with MFA support"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    mfa_code = data.get('mfa_code')
    
    # Verify credentials
    user = authenticate(username, password)
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Check MFA
    mfa = get_mfa_manager()
    if mfa.is_mfa_enabled(user.id):
        if not mfa_code:
            return jsonify({
                'error': 'MFA required',
                'mfa_required': True
            }), 401
        
        if not mfa.verify_mfa(user.id, mfa_code):
            return jsonify({'error': 'Invalid MFA code'}), 401
    
    # Create JWT token
    auth = get_auth_manager()
    access_token, _ = auth.create_access_token(user.id)
    refresh_token, _ = auth.create_refresh_token(user.id)
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'token_type': 'Bearer'
    })
```

### FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from nethical.security.mfa import get_mfa_manager, MFAManager

app = FastAPI()

class MFASetupRequest(BaseModel):
    user_id: str

class MFAVerifyRequest(BaseModel):
    user_id: str
    code: str

@app.post("/api/mfa/setup")
async def setup_mfa(request: MFASetupRequest, mfa: MFAManager = Depends(get_mfa_manager)):
    """Setup MFA for a user"""
    secret, uri, backup_codes = mfa.setup_totp(request.user_id, issuer="Nethical API")
    qr_code = mfa.get_qr_code_data_uri(uri)
    
    return {
        "qr_code": qr_code,
        "secret": secret,
        "backup_codes": backup_codes
    }

@app.post("/api/mfa/verify")
async def verify_mfa(request: MFAVerifyRequest, mfa: MFAManager = Depends(get_mfa_manager)):
    """Verify MFA code"""
    is_valid = mfa.verify_mfa(request.user_id, request.code)
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code"
        )
    
    return {"status": "verified"}
```

## Best Practices

### Security

1. **Always use HTTPS**: MFA codes should never be transmitted over unencrypted connections
2. **Rate limiting**: Implement rate limiting on MFA verification endpoints
3. **Session timeout**: Expire MFA verification after a reasonable time (e.g., 15 minutes)
4. **Secure secret storage**: Never log TOTP secrets or backup codes
5. **Notify on MFA changes**: Alert users when MFA is enabled/disabled

### User Experience

1. **Clear instructions**: Provide step-by-step setup instructions with screenshots
2. **Multiple options**: Offer both QR code and manual entry
3. **Backup codes**: Emphasize importance of saving backup codes
4. **Recovery process**: Provide clear account recovery procedure
5. **Remember device**: Consider "trust this device" option for frequent users

### Implementation

```python
# Rate limiting example
from functools import wraps
from time import time

mfa_attempts = {}  # user_id -> (count, timestamp)

def rate_limit_mfa(max_attempts=5, window=300):
    """Rate limit MFA verification attempts"""
    def decorator(f):
        @wraps(f)
        def wrapper(user_id, code):
            now = time()
            
            if user_id in mfa_attempts:
                count, timestamp = mfa_attempts[user_id]
                
                # Reset if outside window
                if now - timestamp > window:
                    mfa_attempts[user_id] = (1, now)
                elif count >= max_attempts:
                    raise Exception("Too many MFA attempts. Try again later.")
                else:
                    mfa_attempts[user_id] = (count + 1, timestamp)
            else:
                mfa_attempts[user_id] = (1, now)
            
            return f(user_id, code)
        return wrapper
    return decorator

@rate_limit_mfa(max_attempts=5, window=300)
def verify_with_rate_limit(user_id, code):
    mfa = get_mfa_manager()
    return mfa.verify_mfa(user_id, code)
```

## Troubleshooting

### Common Issues

**Problem**: "Time-based codes not working"

```
Solution: Ensure server time is synchronized
- Use NTP for time synchronization
- TOTP has 30-second window, time drift affects validation
```

**Problem**: "QR code not generating"

```python
# Install qrcode library
pip install qrcode[pil]

# Or use provisioning URI directly
print(f"Add this to your authenticator: {provisioning_uri}")
```

**Problem**: "Backup codes not working"

```python
# Codes are one-time use - check if already used
# Regenerate if all codes are consumed
new_codes = mfa.regenerate_backup_codes(user_id)
```

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('nethical.security.mfa').setLevel(logging.DEBUG)

# Test TOTP verification
mfa = MFAManager()
secret, uri, codes = mfa.setup_totp("test_user")
print(f"Secret: {secret}")
print(f"URI: {uri}")
print(f"Backup codes: {codes}")
```

### Account Recovery

If user loses access to MFA device:

```python
def admin_reset_mfa(admin_user_id: str, target_user_id: str):
    """Admin function to reset user's MFA"""
    rbac = get_rbac_manager()
    
    # Verify admin permissions
    if rbac.get_user_role(admin_user_id) != Role.ADMIN:
        raise PermissionError("Admin access required")
    
    mfa = get_mfa_manager()
    
    # Disable MFA (user will need to re-enroll)
    mfa.disable_mfa(target_user_id)
    
    # Log the action
    log_security_event(
        event="mfa_reset",
        admin=admin_user_id,
        target=target_user_id,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Notify user
    send_email(target_user_id, 
        subject="MFA Reset",
        body="Your MFA has been reset by an administrator. "
             "Please set up MFA again on your next login.")
```

## Additional Resources

- [RFC 6238 - TOTP Specification](https://tools.ietf.org/html/rfc6238)
- [NIST Special Publication 800-63B](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [Google Authenticator](https://support.google.com/accounts/answer/1066447)

## Support

For issues or questions:
- Open an issue on GitHub
- Check test examples in `tests/unit/test_mfa.py`
- Review phase1 implementation docs in `docs/security/phase1_implementation.md`
