# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Military-Grade Server-Side Request Forgery (SSRF) Protection Engine.

Provides deep defense-in-depth URL and IP validation across outbound gateways,
webhooks, agentic tools, and proxies:
- Strict protocol enforcement (HTTP/HTTPS only; blocks file://, gopher://, dict://)
- Anti-DNS-Rebinding validation resolving all A / AAAA resource records
- Complete IPv4/IPv6 address filtration:
  * RFC 1918 Private ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
  * Loopback addresses (127.0.0.0/8, ::1)
  * Link-Local & Cloud Metadata endpoints (169.254.169.254, fe80::/10, fd00:ec2::254)
  * Carrier-Grade NAT (100.64.0.0/10)
  * Multicast and reserved test blocks (224.0.0.0/4, 240.0.0.0/4, 0.0.0.0/8)
- Optional domain whitelist enforcement
- Configurable environment bypass for testbeds (NETHICAL_ALLOW_PRIVATE_NETWORKS)
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from typing import Iterable, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("nethical.security.ssrf_protection")

# Additional blocked IPv4 networks not always flagged by basic is_private
_BLOCKED_IPV4_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),         # Broadcast / "this" network
    ipaddress.ip_network("10.0.0.0/8"),        # RFC 1918
    ipaddress.ip_network("100.64.0.0/10"),     # Carrier-Grade NAT (RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local / Cloud Metadata
    ipaddress.ip_network("172.16.0.0/12"),     # RFC 1918
    ipaddress.ip_network("192.0.0.0/24"),      # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),      # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),    # RFC 1918
    ipaddress.ip_network("198.18.0.0/15"),     # Network benchmark tests
    ipaddress.ip_network("198.51.100.0/24"),   # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),    # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),       # Multicast
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved / Class E
    ipaddress.ip_network("255.255.255.255/32"),# Limited broadcast
)

_BLOCKED_IPV6_NETWORKS = (
    ipaddress.ip_network("::/128"),            # Unspecified
    ipaddress.ip_network("::1/128"),           # Loopback
    ipaddress.ip_network("fc00::/7"),          # Unique Local Address (ULA)
    ipaddress.ip_network("fe80::/10"),         # Link-local unicast
    ipaddress.ip_network("ff00::/8"),          # Multicast
    ipaddress.ip_network("2001:db8::/32"),     # Documentation
)

# Explicit cloud metadata IPs
_CLOUD_METADATA_IPS = {
    "169.254.169.254",                         # AWS, Azure, GCP, OpenStack
    "169.254.169.253",                         # AWS DNS
    "100.100.100.200",                         # Alibaba Cloud metadata
    "fd00:ec2::254",                           # AWS IPv6 metadata
}

# Regex to detect dangerous embedded userinfo or octal/hex IP representations
_DANGEROUS_HOST_PATTERNS = re.compile(r"(^0x[0-9a-f]+$)|(^0[0-7]+$)|(^\d+$)", re.IGNORECASE)


class SSRFValidationError(ValueError):
    """Raised when an outbound URL violates SSRF security boundaries."""
    pass


def is_ip_private_or_restricted(
    ip_str: str,
    allow_private: bool = False,
) -> Tuple[bool, str]:
    """
    Check if an IP address falls into loopback, private, link-local, or metadata ranges.

    Returns:
        (is_restricted, reason)
    """
    try:
        ip = ipaddress.ip_address(ip_str.strip())
    except ValueError:
        return True, f"Invalid IP address format: {ip_str}"

    if ip_str.strip() in _CLOUD_METADATA_IPS:
        return True, f"Direct access to cloud instance metadata service blocked ({ip_str})"

    if ip.is_loopback:
        return True, f"Loopback address access blocked ({ip_str})"

    if ip.is_link_local:
        return True, f"Link-local address access blocked ({ip_str})"

    if ip.is_multicast:
        return True, f"Multicast address access blocked ({ip_str})"

    if ip.is_unspecified:
        return True, f"Unspecified address access blocked ({ip_str})"

    if ip.version == 4:
        for net in _BLOCKED_IPV4_NETWORKS:
            if ip in net:
                if allow_private and net.prefixlen != 32 and str(net.network_address).startswith(("10.", "172.", "192.168.")):
                    # Legitimate private intranet allowed if explicitly requested
                    continue
                return True, f"IPv4 address {ip_str} belongs to restricted range {net}"
    elif ip.version == 6:
        for net in _BLOCKED_IPV6_NETWORKS:
            if ip in net:
                if allow_private and str(net).startswith("fc00"):
                    continue
                return True, f"IPv6 address {ip_str} belongs to restricted range {net}"

    return False, ""


def resolve_all_ips(hostname: str) -> Set[str]:
    """
    Resolve hostname to all IPv4 and IPv6 addresses to defend against DNS rebinding.
    """
    results: Set[str] = set()
    try:
        addrinfo = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for entry in addrinfo:
            sockaddr = entry[4]
            ip_candidate = sockaddr[0]
            results.add(ip_candidate)
    except socket.gaierror as e:
        logger.warning("SSRF DNS resolution failed for host '%s': %s", hostname, e)
        raise SSRFValidationError(f"Could not resolve host '{hostname}': {e}") from e
    return results


def validate_safe_url(
    url: str,
    allow_private: bool = False,
    allowed_domains: Optional[Iterable[str]] = None,
    require_resolvable: bool = True,
) -> Tuple[bool, str]:
    """
    Validate a URL to ensure it cannot be abused for SSRF attacks.

    Args:
        url: The candidate URL string
        allow_private: If True, allows RFC 1918 private IPs (does NOT allow loopback or metadata)
        allowed_domains: Optional list/set of allowed hostnames or domains
        require_resolvable: If True, requires DNS resolution to succeed. If False, syntax/IP validation
                           passes even if hostname is offline/mocked (validated again before dispatch).

    Returns:
        Tuple of (is_safe, error_reason)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    url_clean = url.strip()

    # Check for illegal control characters
    if any(c in url_clean for c in ("\r", "\n", "\t", "\x00")):
        return False, "URL contains illegal control characters"

    try:
        parsed = urlparse(url_clean)
    except Exception as e:
        return False, f"Malformed URL syntax: {e}"

    # Enforce strictly http or https
    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Scheme '{parsed.scheme}' disallowed. Only http and https are permitted"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no valid hostname"

    hostname = hostname.strip().lower()

    # Reject localhost or common aliases immediately
    if hostname in ("localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"):
        return False, "Targeting localhost is prohibited"

    # Check for numerical / octal tricks
    if _DANGEROUS_HOST_PATTERNS.match(hostname):
        return False, f"Dangerous numeric hostname pattern detected: {hostname}"

    # Enforce domain whitelist if specified
    if allowed_domains:
        normalized_allowed = {d.strip().lower().lstrip(".") for d in allowed_domains if d}
        matched = False
        for allowed in normalized_allowed:
            if hostname == allowed or hostname.endswith("." + allowed):
                matched = True
                break
        if not matched:
            return False, f"Host '{hostname}' is not in allowed domains whitelist"

    # Check environment override
    env_allow_private = os.getenv("NETHICAL_ALLOW_PRIVATE_NETWORKS", "").lower() in ("1", "true", "yes")
    effective_allow_private = allow_private or env_allow_private

    # Direct IP or Hostname resolution
    # Check if hostname is direct IP literal
    try:
        ipaddress.ip_address(hostname)
        is_direct_ip = True
        resolved_ips = {hostname}
    except ValueError:
        is_direct_ip = False

    if not is_direct_ip:
        try:
            resolved_ips = resolve_all_ips(hostname)
        except SSRFValidationError as exc:
            if not require_resolvable:
                # Syntax and scheme are safe, defer DNS IP check until active dispatch
                return True, ""
            return False, str(exc)

    if not resolved_ips:
        if not require_resolvable:
            return True, ""
        return False, f"Host '{hostname}' did not resolve to any IP addresses"

    for ip_addr in resolved_ips:
        is_restricted, reason = is_ip_private_or_restricted(
            ip_addr,
            allow_private=effective_allow_private,
        )
        if is_restricted:
            return False, f"SSRF Protection Blocked: Host '{hostname}' resolved to {ip_addr} ({reason})"

    return True, ""


def assert_safe_url(
    url: str,
    allow_private: bool = False,
    allowed_domains: Optional[Iterable[str]] = None,
    require_resolvable: bool = True,
) -> str:
    """
    Validate URL and raise SSRFValidationError if unsafe. Returns sanitized URL string.
    """
    is_safe, reason = validate_safe_url(
        url=url,
        allow_private=allow_private,
        allowed_domains=allowed_domains,
        require_resolvable=require_resolvable,
    )
    if not is_safe:
        logger.security_warning = getattr(logger, "security_warning", logger.warning)
        logger.warning("SSRF ATTEMPT BLOCKED: %s [URL: %s]", reason, url)
        raise SSRFValidationError(reason)
    return url.strip()
