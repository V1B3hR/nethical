# FDA 21 CFR Part 11 Electronic Signature Implementation

## Document Information

| Field | Value |
|-------|-------|
| Document ID | ES-FDA-001 |
| Version | 1.0 |
| Date | 2025-12-03 |
| Author | Nethical Development Team |
| Status | Active |

## 1. Overview

This document describes the implementation of electronic signatures in the Nethical AI Governance System, ensuring compliance with FDA 21 CFR Part 11 Subpart C requirements.

## 2. Electronic Signature Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ELECTRONIC SIGNATURE SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Signature     │ │   Multi-Factor  │ │   Signing       │ │   Verification  │
│   Request       │ │   Authentication│ │   Certificate   │ │   Service       │
│   Service       │ │   Service       │ │   Service       │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
         │                 │                 │                 │
         └─────────────────┼─────────────────┼─────────────────┘
                           │                 │
                           ▼                 ▼
                    ┌─────────────────┐ ┌─────────────────┐
                    │   Audit Trail   │ │   Signature     │
                    │   Service       │ │   Storage       │
                    └─────────────────┘ └─────────────────┘
```

### 2.2 Data Model

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

class SignatureMeaning(Enum):
    """Signature meaning per Part 11 § 11.50"""
    AUTHORED = "AUTHORED"           # Created the record
    REVIEWED = "REVIEWED"           # Reviewed the record
    APPROVED = "APPROVED"           # Approved the record
    VERIFIED = "VERIFIED"           # Verified accuracy
    REJECTED = "REJECTED"           # Rejected the record
    WITNESSED = "WITNESSED"         # Witnessed the action

@dataclass
class ElectronicSignature:
    """Electronic signature compliant with 21 CFR Part 11"""
    
    # Unique identifier
    signature_id: uuid.UUID
    
    # Record being signed (§ 11.70 linking)
    record_id: uuid.UUID
    record_type: str
    record_hash: str  # SHA-256 hash of record content
    
    # Signer identity (§ 11.50, § 11.100)
    signer_id: str
    signer_name: str  # Full name as registered
    signer_title: Optional[str] = None
    
    # Signature details (§ 11.50)
    meaning: SignatureMeaning
    timestamp: datetime  # UTC
    comment: Optional[str] = None
    
    # Authentication evidence (§ 11.200)
    auth_method: str  # "password+totp", "password+biometric"
    auth_session_id: str
    
    # Cryptographic proof
    signature_value: str  # Base64 encoded signature
    algorithm: str  # e.g., "RSA-SHA256"
    certificate_thumbprint: str
    
    # Chain integrity
    previous_signature_hash: Optional[str] = None
    signature_hash: str

@dataclass
class SignatureRequest:
    """Request to create an electronic signature"""
    
    request_id: uuid.UUID
    record_id: uuid.UUID
    record_type: str
    meaning: SignatureMeaning
    requestor_id: str
    requested_at: datetime
    expires_at: datetime
    status: str  # "pending", "completed", "expired", "cancelled"
    
    # MFA challenge
    mfa_method: str
    mfa_challenge_id: str
```

## 3. Signature Workflow

### 3.1 Signature Request Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Request    │───►│   Validate   │───►│    MFA       │───►│   Create     │
│   Signature  │    │   Authority  │    │   Challenge  │    │   Signature  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  User submits        Check role &        TOTP/Push/         Generate
  signature           signing             Biometric          cryptographic
  request             privileges          verification       signature
```

### 3.2 Implementation Steps

#### Step 1: Request Initiation

```python
async def request_signature(
    record_id: str,
    record_type: str,
    meaning: SignatureMeaning,
    comment: Optional[str] = None
) -> SignatureRequest:
    """
    Initiate an electronic signature request.
    
    Args:
        record_id: ID of record to sign
        record_type: Type of record (policy, decision, etc.)
        meaning: Meaning of signature per Part 11
        comment: Optional comment
        
    Returns:
        SignatureRequest with MFA challenge
    """
    # Get current user
    user = get_current_user()
    
    # Verify signing authority
    if not has_signature_authority(user, record_type, meaning):
        raise PermissionDenied("User lacks signature authority")
    
    # Verify record exists and is signable
    record = await get_record(record_id, record_type)
    if not can_sign(record, meaning):
        raise InvalidState("Record cannot be signed in current state")
    
    # Create signature request
    request = SignatureRequest(
        request_id=uuid.uuid4(),
        record_id=record_id,
        record_type=record_type,
        meaning=meaning,
        requestor_id=user.id,
        requested_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        status="pending",
        mfa_method=user.preferred_mfa,
        mfa_challenge_id=await create_mfa_challenge(user)
    )
    
    # Store request
    await store_signature_request(request)
    
    # Audit log
    await audit_log(
        action="SIGNATURE_REQUESTED",
        record_id=record_id,
        user_id=user.id,
        details={"meaning": meaning.value}
    )
    
    return request
```

#### Step 2: MFA Verification

```python
async def verify_mfa_for_signature(
    request_id: str,
    mfa_code: str
) -> bool:
    """
    Verify MFA code for signature request.
    
    Args:
        request_id: Signature request ID
        mfa_code: User-provided MFA code
        
    Returns:
        True if MFA verified successfully
    """
    request = await get_signature_request(request_id)
    
    # Verify request is still valid
    if request.status != "pending":
        raise InvalidState("Signature request not pending")
    if datetime.utcnow() > request.expires_at:
        await update_request_status(request_id, "expired")
        raise InvalidState("Signature request expired")
    
    # Verify MFA
    user = await get_user(request.requestor_id)
    if not await verify_mfa(user, request.mfa_challenge_id, mfa_code):
        await audit_log(
            action="SIGNATURE_MFA_FAILED",
            record_id=request.record_id,
            user_id=user.id
        )
        raise AuthenticationFailed("MFA verification failed")
    
    return True
```

#### Step 3: Signature Creation

```python
async def create_electronic_signature(
    request_id: str,
    mfa_code: str
) -> ElectronicSignature:
    """
    Create an electronic signature after MFA verification.
    
    Args:
        request_id: Signature request ID
        mfa_code: Verified MFA code
        
    Returns:
        Completed ElectronicSignature
    """
    # Verify MFA
    if not await verify_mfa_for_signature(request_id, mfa_code):
        raise AuthenticationFailed()
    
    request = await get_signature_request(request_id)
    user = await get_user(request.requestor_id)
    record = await get_record(request.record_id, request.record_type)
    
    # Calculate record hash
    record_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    
    # Get signing certificate
    cert = await get_user_signing_certificate(user.id)
    
    # Create signature data
    signature_data = {
        "record_id": str(request.record_id),
        "record_hash": record_hash,
        "signer_id": user.id,
        "meaning": request.meaning.value,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Sign with private key (HSM-backed)
    signature_value = await hsm_sign(
        data=json.dumps(signature_data).encode(),
        key_id=cert.key_id
    )
    
    # Get previous signature for chain
    prev_sig = await get_latest_signature(request.record_id)
    prev_hash = prev_sig.signature_hash if prev_sig else None
    
    # Create signature object
    signature = ElectronicSignature(
        signature_id=uuid.uuid4(),
        record_id=request.record_id,
        record_type=request.record_type,
        record_hash=record_hash,
        signer_id=user.id,
        signer_name=user.full_name,
        signer_title=user.title,
        meaning=request.meaning,
        timestamp=datetime.utcnow(),
        comment=request.comment,
        auth_method=request.mfa_method,
        auth_session_id=request.mfa_challenge_id,
        signature_value=base64.b64encode(signature_value).decode(),
        algorithm="RSA-SHA256",
        certificate_thumbprint=cert.thumbprint,
        previous_signature_hash=prev_hash,
        signature_hash=calculate_signature_hash(signature_data, signature_value)
    )
    
    # Store signature
    await store_signature(signature)
    
    # Update record status if needed
    await update_record_after_signature(
        request.record_id,
        request.record_type,
        request.meaning
    )
    
    # Update request status
    await update_request_status(request_id, "completed")
    
    # Audit log
    await audit_log(
        action="SIGNATURE_CREATED",
        record_id=request.record_id,
        user_id=user.id,
        details={
            "signature_id": str(signature.signature_id),
            "meaning": request.meaning.value,
            "algorithm": signature.algorithm
        }
    )
    
    return signature
```

## 4. Signature Verification

### 4.1 Verification Process

```python
async def verify_electronic_signature(
    signature_id: str
) -> dict:
    """
    Verify an electronic signature is valid.
    
    Args:
        signature_id: ID of signature to verify
        
    Returns:
        Verification result with details
    """
    signature = await get_signature(signature_id)
    record = await get_record(signature.record_id, signature.record_type)
    
    result = {
        "signature_id": str(signature.signature_id),
        "valid": True,
        "checks": []
    }
    
    # Check 1: Record integrity
    current_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    if current_hash != signature.record_hash:
        result["valid"] = False
        result["checks"].append({
            "check": "record_integrity",
            "passed": False,
            "message": "Record has been modified since signing"
        })
    else:
        result["checks"].append({
            "check": "record_integrity",
            "passed": True
        })
    
    # Check 2: Cryptographic validity
    cert = await get_certificate(signature.certificate_thumbprint)
    signature_data = {
        "record_id": str(signature.record_id),
        "record_hash": signature.record_hash,
        "signer_id": signature.signer_id,
        "meaning": signature.meaning.value,
        "timestamp": signature.timestamp.isoformat()
    }
    
    if not verify_signature(
        data=json.dumps(signature_data).encode(),
        signature=base64.b64decode(signature.signature_value),
        certificate=cert
    ):
        result["valid"] = False
        result["checks"].append({
            "check": "cryptographic_validity",
            "passed": False,
            "message": "Cryptographic signature verification failed"
        })
    else:
        result["checks"].append({
            "check": "cryptographic_validity",
            "passed": True
        })
    
    # Check 3: Certificate validity at signing time
    if not cert.was_valid_at(signature.timestamp):
        result["valid"] = False
        result["checks"].append({
            "check": "certificate_validity",
            "passed": False,
            "message": "Certificate was not valid at signing time"
        })
    else:
        result["checks"].append({
            "check": "certificate_validity",
            "passed": True
        })
    
    # Check 4: Chain integrity
    if signature.previous_signature_hash:
        prev_sig = await get_signature_by_hash(signature.previous_signature_hash)
        if not prev_sig:
            result["checks"].append({
                "check": "chain_integrity",
                "passed": False,
                "message": "Previous signature in chain not found"
            })
        else:
            result["checks"].append({
                "check": "chain_integrity",
                "passed": True
            })
    
    # Check 5: Signer authority
    signer = await get_user(signature.signer_id)
    if not signer:
        result["checks"].append({
            "check": "signer_exists",
            "passed": False,
            "message": "Signer no longer exists in system"
        })
    else:
        result["checks"].append({
            "check": "signer_exists",
            "passed": True,
            "signer_name": signer.full_name
        })
    
    return result
```

## 5. Signature Display

Per § 11.50, signatures must display:

```html
<!-- Signature Display Template -->
<div class="electronic-signature">
    <div class="signature-header">
        <span class="signature-icon">✓</span>
        <span class="signature-title">Electronic Signature</span>
    </div>
    
    <div class="signature-details">
        <div class="detail-row">
            <label>Signed By:</label>
            <span class="signer-name">{{ signature.signer_name }}</span>
        </div>
        
        <div class="detail-row">
            <label>Title:</label>
            <span class="signer-title">{{ signature.signer_title }}</span>
        </div>
        
        <div class="detail-row">
            <label>Date & Time:</label>
            <span class="timestamp">{{ signature.timestamp | datetime }}</span>
        </div>
        
        <div class="detail-row">
            <label>Meaning:</label>
            <span class="meaning">{{ signature.meaning }}</span>
        </div>
        
        {% if signature.comment %}
        <div class="detail-row">
            <label>Comment:</label>
            <span class="comment">{{ signature.comment }}</span>
        </div>
        {% endif %}
    </div>
    
    <div class="signature-verification">
        <span class="verification-status verified">✓ Signature Verified</span>
        <small>Signature ID: {{ signature.signature_id }}</small>
    </div>
</div>
```

## 6. API Reference

### Request Signature

```http
POST /v2/signatures/request
Authorization: Bearer <token>
Content-Type: application/json

{
    "record_id": "uuid",
    "record_type": "policy",
    "meaning": "APPROVED",
    "comment": "Approved after compliance review"
}

Response:
HTTP 202 Accepted
{
    "request_id": "uuid",
    "expires_at": "2025-12-03T14:35:00Z",
    "mfa_required": true,
    "mfa_method": "totp"
}
```

### Complete Signature

```http
POST /v2/signatures/complete
Authorization: Bearer <token>
Content-Type: application/json

{
    "request_id": "uuid",
    "mfa_code": "123456"
}

Response:
HTTP 200 OK
{
    "signature_id": "uuid",
    "signer_name": "Dr. Jane Smith",
    "signer_title": "Quality Manager",
    "timestamp": "2025-12-03T14:30:00Z",
    "meaning": "APPROVED",
    "valid": true
}
```

### Verify Signature

```http
GET /v2/signatures/{signature_id}/verify
Authorization: Bearer <token>

Response:
HTTP 200 OK
{
    "signature_id": "uuid",
    "valid": true,
    "checks": [
        {"check": "record_integrity", "passed": true},
        {"check": "cryptographic_validity", "passed": true},
        {"check": "certificate_validity", "passed": true},
        {"check": "chain_integrity", "passed": true},
        {"check": "signer_exists", "passed": true}
    ]
}
```

## 7. Security Considerations

### 7.1 Key Management

- Signing keys stored in HSM
- Per-user signing certificates
- Certificate rotation annually
- Revocation support

### 7.2 Audit Requirements

All signature operations logged:
- Request creation
- MFA attempts (success/failure)
- Signature creation
- Signature verification
- Certificate operations

### 7.3 Compliance Controls

| Control | Implementation |
|---------|---------------|
| Signature uniqueness | UUID per signature |
| Non-transferability | Cryptographic binding |
| Re-authentication | MFA for each signature |
| Timestamp accuracy | NTP-synchronized |
| Display requirements | Mandatory fields shown |

## 8. References

- FDA 21 CFR Part 11
- NIST SP 800-57 (Key Management)
- PKCS #7 (Cryptographic Message Syntax)

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-03
