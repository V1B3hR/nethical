"""
Authentication stub for Nethical API.

Provides simple API key authentication with environment-based configuration.
Designed for single-instance deployments with minimal overhead.
"""

import os
import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Simple API key authentication manager.
    
    Supports environment-based API key configuration. If no keys configured,
    runs in permissive mode (no authentication required).
    """
    
    def __init__(self):
        """Initialize auth manager from environment variables."""
        self._api_keys: Optional[Set[str]] = None
        self._permissive_mode = True
        
        # Load API keys from environment
        keys_env = os.getenv("NETHICAL_API_KEYS", "").strip()
        if keys_env:
            # Parse comma-separated keys
            self._api_keys = set(k.strip() for k in keys_env.split(",") if k.strip())
            self._permissive_mode = False
            logger.info(f"Authentication enabled with {len(self._api_keys)} configured API keys")
        else:
            logger.warning(
                "No API keys configured (NETHICAL_API_KEYS not set). "
                "Running in PERMISSIVE MODE - all requests authenticated by IP-based rate limiting only."
            )
    
    def is_permissive(self) -> bool:
        """Check if running in permissive mode (no auth required)."""
        return self._permissive_mode
    
    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if key is valid or in permissive mode, False otherwise
        """
        if self._permissive_mode:
            return True
        
        if not api_key:
            return False
        
        return api_key in self._api_keys
    
    def extract_identity(self, api_key: Optional[str], client_ip: str) -> str:
        """
        Extract identity for rate limiting.
        
        Prefers API key if provided and valid, otherwise falls back to IP.
        
        Args:
            api_key: Optional API key from request
            client_ip: Client IP address
            
        Returns:
            Identity string for rate limiting
        """
        if api_key and (self._permissive_mode or api_key in self._api_keys):
            # Use hash of API key for privacy in logs
            import hashlib
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"key:{key_hash}"
        
        return f"ip:{client_ip}"
    
    def get_stats(self) -> dict:
        """Get authentication statistics."""
        return {
            "permissive_mode": self._permissive_mode,
            "configured_keys": len(self._api_keys) if self._api_keys else 0
        }
