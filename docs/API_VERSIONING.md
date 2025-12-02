# API Versioning Strategy

## Overview

The Nethical Governance API uses URL-based versioning to ensure backward compatibility
while enabling evolution of the API. This document outlines our versioning strategy
and commitment to API stability.

## Versioning Scheme

### URL-Based Versioning

All API endpoints are prefixed with a version identifier:

```
/v1/evaluate     # Legacy version (deprecated)
/v2/evaluate     # Current stable version
/v3/evaluate     # Future version (when available)
```

### Version Lifecycle

| Phase | Duration | Description |
|-------|----------|-------------|
| **Preview** | 3-6 months | New features for testing, may change |
| **Stable** | 2+ years | Production-ready, backward compatible |
| **Deprecated** | 12 months | Still functional, migration recommended |
| **Sunset** | N/A | No longer available |

## Current Versions

| Version | Status | Release Date | Sunset Date |
|---------|--------|--------------|-------------|
| v1 | Deprecated | 2024-01-01 | 2025-12-31 |
| **v2** | **Stable** | 2025-12-02 | TBD |

## Backward Compatibility Promise

For **stable** API versions, we guarantee:

1. **Additive Changes Only**: New fields may be added to responses
2. **No Breaking Removals**: Existing fields will not be removed
3. **Type Stability**: Field types will not change
4. **Deprecation Notices**: Deprecated features will be documented

### What Constitutes a Breaking Change

The following are considered breaking changes and require a new major version:

- Removing an endpoint
- Removing a response field
- Changing a field's data type
- Changing authentication requirements
- Changing error response structure
- Renaming endpoints or fields

### Non-Breaking Changes

The following are safe to make without a version bump:

- Adding new endpoints
- Adding optional request fields
- Adding new response fields
- Adding new enum values (when client handles unknowns gracefully)
- Improving error messages
- Adding new HTTP headers

## Version Headers

All API responses include version information in headers:

```http
X-API-Version: 2.0.0
X-API-Deprecation-Notice: This version is deprecated. Migrate to v2.
X-API-Sunset-Date: 2025-12-31
```

## Migration Guide

### v1 to v2 Migration

Key changes in v2:

1. **Enhanced Evaluate Response**
   - Added `decision_id` field for tracking
   - Added `latency_ms` for performance monitoring
   - Added `fundamental_laws_checked` array

2. **New Endpoints**
   - `POST /v2/batch-evaluate` for batch processing
   - `GET /v2/fairness` for fairness metrics
   - `POST /v2/appeals` for appeal submission

3. **Structured Error Responses**
   - Consistent error format with `request_id`
   - Detailed validation errors

### Migration Steps

1. Update base URL from `/v1/` to `/v2/`
2. Handle new fields in responses (additive)
3. Update error handling for new error format
4. Test thoroughly in staging environment

## SDK Version Compatibility

| SDK Version | API v1 | API v2 |
|-------------|--------|--------|
| Python 1.x | ✅ | ❌ |
| **Python 2.x** | ✅ | ✅ |
| JavaScript 1.x | ❌ | ✅ |
| Go 1.x | ❌ | ✅ |
| Rust 1.x | ❌ | ✅ |

## Rate Limiting by Version

Rate limits may differ by API version:

| Version | Requests/min | Burst |
|---------|-------------|-------|
| v1 | 60 | 10 |
| v2 | 100 | 20 |

## Deprecation Communication

When an API version is deprecated:

1. Documentation is updated with deprecation notice
2. API responses include `X-API-Deprecation-Notice` header
3. Email notification sent to registered API users
4. Blog post announcing deprecation timeline
5. SDK releases include deprecation warnings

## Fundamental Laws Alignment

All API versions adhere to the 25 Fundamental Laws of AI Ethics:

- **Law 8: Constraint Transparency** - Version changes are clearly documented
- **Law 10: Reasoning Transparency** - Breaking changes are explained
- **Law 15: Audit Compliance** - Version history is maintained
- **Law 25: Evolutionary Preparation** - Versioning supports evolution

## Contact

For questions about API versioning or migration assistance:

- Documentation: [docs.nethical.io](https://docs.nethical.io)
- Email: api@nethical.io
- GitHub Issues: [github.com/v1b3hr/nethical/issues](https://github.com/v1b3hr/nethical/issues)
