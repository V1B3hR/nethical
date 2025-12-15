#!/usr/bin/env python3
"""
Shadow Traffic Replay Tool for Nethical

A safe, repeatable traffic replay tool for mirroring production traffic
into isolated staging environments without affecting external users.

Features:
- HAR (HTTP Archive) and custom JSON traffic format support
- Dry-run mode (default) for safe testing
- Header rewriting for staging environment targeting
- Configurable method filtering for safety
- Rate limiting support
- Authentication injection
- Retry logic with configurable attempts
- Comprehensive replay reports
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    print("Error: requests library is required. Install with: pip install requests")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("shadow_replay")


@dataclass
class ReplayRequest:
    """Represents a single HTTP request to replay."""
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    query_params: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReplayResponse:
    """Represents the response from a replayed request."""
    status_code: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ReplayReport:
    """Comprehensive report of a replay session."""
    total_requests: int = 0
    processed: int = 0
    skipped: int = 0
    sent: int = 0
    errors: int = 0
    successful: int = 0
    failed: int = 0
    skipped_reasons: Dict[str, int] = field(default_factory=dict)
    response_samples: List[Dict[str, Any]] = field(default_factory=list)
    error_samples: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "total_requests": self.total_requests,
            "processed": self.processed,
            "skipped": self.skipped,
            "sent": self.sent,
            "errors": self.errors,
            "successful": self.successful,
            "failed": self.failed,
            "skipped_reasons": self.skipped_reasons,
            "response_samples": self.response_samples,
            "error_samples": self.error_samples,
            "total_duration_seconds": self.total_duration_seconds,
        }


class ShadowReplayTool:
    """Main shadow traffic replay tool."""
    
    # Production domain patterns for safety checks
    PRODUCTION_DOMAINS = [
        "prod", "production", "api.nethical.com", "nethical.com",
        "live", "www"
    ]
    
    def __init__(
        self,
        staging_base_url: str,
        dry_run: bool = True,
        skip_methods: Optional[List[str]] = None,
        allow_modifying: bool = False,
        rps: Optional[float] = None,
        auth: Optional[str] = None,
        retries: int = 3,
        force: bool = False,
    ):
        """
        Initialize the shadow replay tool.
        
        Args:
            staging_base_url: Base URL of the staging environment
            dry_run: If True, process but don't send requests (default: True)
            skip_methods: HTTP methods to skip (default: POST, PUT, PATCH, DELETE)
            allow_modifying: Explicitly allow state-changing requests
            rps: Requests per second rate limit (optional)
            auth: Authorization token/key (optional)
            retries: Number of retry attempts for failed requests
            force: Force execution even if target looks like production
        """
        self.staging_base_url = staging_base_url.rstrip("/")
        self.dry_run = dry_run
        self.allow_modifying = allow_modifying
        self.rps = rps
        self.auth = auth
        self.retries = retries
        self.force = force
        
        # Default to skipping state-changing methods unless explicitly allowed
        if skip_methods is None:
            self.skip_methods = [] if allow_modifying else ["POST", "PUT", "PATCH", "DELETE"]
        else:
            self.skip_methods = [m.upper() for m in skip_methods]
        
        self.report = ReplayReport()
        self.session: Optional[requests.Session] = None
        
        # Safety check for production environment
        self._check_production_safety()
        
        logger.info(f"Shadow Replay Tool initialized:")
        logger.info(f"  Target: {self.staging_base_url}")
        logger.info(f"  Dry-run: {self.dry_run}")
        logger.info(f"  Skip methods: {self.skip_methods}")
        logger.info(f"  Rate limit: {self.rps if self.rps else 'None'} req/sec")
        logger.info(f"  Retries: {self.retries}")
    
    def _check_production_safety(self) -> None:
        """Check if target URL looks like production and abort if unsafe."""
        target_lower = self.staging_base_url.lower()
        
        # Check environment variables for production indicators
        env_checks = [
            os.environ.get("ENV", "").lower(),
            os.environ.get("ENVIRONMENT", "").lower(),
            os.environ.get("TARGET_ENV", "").lower(),
        ]
        
        for env_val in env_checks:
            if "prod" in env_val or "production" in env_val:
                if not self.force:
                    logger.error(
                        "SAFETY ERROR: Environment appears to be production! "
                        f"ENV variable contains: {env_val}"
                    )
                    logger.error("Use --force flag to override (NOT RECOMMENDED)")
                    sys.exit(1)
                else:
                    logger.warning(
                        "WARNING: Overriding production safety check with --force flag!"
                    )
        
        # Check if target URL contains production patterns
        for pattern in self.PRODUCTION_DOMAINS:
            if pattern in target_lower:
                if not self.force:
                    logger.error(
                        f"SAFETY ERROR: Target URL contains production pattern '{pattern}': "
                        f"{self.staging_base_url}"
                    )
                    logger.error("Use --force flag to override (NOT RECOMMENDED)")
                    sys.exit(1)
                else:
                    logger.warning(
                        f"WARNING: Target URL contains production pattern '{pattern}' "
                        "but overridden with --force flag!"
                    )
    
    def _init_session(self) -> requests.Session:
        """Initialize requests session with retry logic."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST", "PATCH"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def parse_har(self, har_path: Path) -> List[ReplayRequest]:
        """
        Parse HAR (HTTP Archive) file and extract requests.
        
        Args:
            har_path: Path to HAR file
            
        Returns:
            List of ReplayRequest objects
        """
        logger.info(f"Parsing HAR file: {har_path}")
        
        with open(har_path, "r") as f:
            har_data = json.load(f)
        
        requests_list = []
        
        # HAR format: log.entries[].request
        entries = har_data.get("log", {}).get("entries", [])
        
        for entry in entries:
            request_data = entry.get("request", {})
            
            # Extract method and URL
            method = request_data.get("method", "GET")
            url = request_data.get("url", "")
            
            if not url:
                continue
            
            # Extract headers
            headers = {}
            for header in request_data.get("headers", []):
                headers[header.get("name", "")] = header.get("value", "")
            
            # Extract query parameters
            query_params = {}
            for param in request_data.get("queryString", []):
                query_params[param.get("name", "")] = param.get("value", "")
            
            # Extract body
            body = None
            post_data = request_data.get("postData", {})
            if post_data:
                body = post_data.get("text")
            
            replay_req = ReplayRequest(
                method=method,
                url=url,
                headers=headers,
                body=body,
                query_params=query_params,
            )
            
            requests_list.append(replay_req)
        
        logger.info(f"Parsed {len(requests_list)} requests from HAR file")
        return requests_list
    
    def parse_json(self, json_path: Path) -> List[ReplayRequest]:
        """
        Parse custom JSON format with list of request dictionaries.
        
        Expected format:
        [
            {
                "method": "GET",
                "url": "/api/users",
                "headers": {"Content-Type": "application/json"},
                "body": null,
                "query_params": {}
            },
            ...
        ]
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            List of ReplayRequest objects
        """
        logger.info(f"Parsing JSON file: {json_path}")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of request objects")
        
        requests_list = []
        
        for req_dict in data:
            replay_req = ReplayRequest(
                method=req_dict.get("method", "GET"),
                url=req_dict.get("url", ""),
                headers=req_dict.get("headers", {}),
                body=req_dict.get("body"),
                query_params=req_dict.get("query_params", {}),
            )
            requests_list.append(replay_req)
        
        logger.info(f"Parsed {len(requests_list)} requests from JSON file")
        return requests_list
    
    def _rewrite_request(self, request: ReplayRequest) -> ReplayRequest:
        """
        Rewrite request to target staging environment.
        
        - Changes URL to point to staging base URL
        - Adds X-Nethical-Shadow header
        - Injects auth header if provided
        - Preserves query parameters
        
        Args:
            request: Original request
            
        Returns:
            Modified request for staging
        """
        # Parse original URL to extract path and query
        parsed = urlparse(request.url)
        path = parsed.path
        
        # Build staging URL
        staging_url = urljoin(self.staging_base_url, path)
        
        # Copy headers and add shadow marker
        new_headers = request.headers.copy()
        new_headers["X-Nethical-Shadow"] = "true"
        
        # Add authentication if provided
        if self.auth:
            new_headers["Authorization"] = f"Bearer {self.auth}"
        
        # Remove headers that shouldn't be forwarded
        headers_to_remove = ["Host", "Content-Length"]
        for header in headers_to_remove:
            new_headers.pop(header, None)
        
        return ReplayRequest(
            method=request.method,
            url=staging_url,
            headers=new_headers,
            body=request.body,
            query_params=request.query_params,
        )
    
    def _should_skip_request(self, request: ReplayRequest) -> Tuple[bool, Optional[str]]:
        """
        Determine if request should be skipped.
        
        Args:
            request: Request to check
            
        Returns:
            Tuple of (should_skip, reason)
        """
        # Check if method is in skip list
        if request.method.upper() in self.skip_methods:
            return True, f"Method {request.method} in skip list"
        
        # Add more safety checks here if needed
        # For example, checking for destructive endpoints
        
        return False, None
    
    def _send_request(
        self, request: ReplayRequest
    ) -> ReplayResponse:
        """
        Send a single request to staging environment.
        
        Args:
            request: Request to send
            
        Returns:
            Response object
        """
        if self.session is None:
            self.session = self._init_session()
        
        response = ReplayResponse()
        start_time = time.time()
        
        try:
            logger.debug(f"Sending {request.method} {request.url}")
            
            # Send request
            resp = self.session.request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                params=request.query_params,
                data=request.body,
                timeout=30,
            )
            
            # Record response
            response.status_code = resp.status_code
            response.headers = dict(resp.headers)
            response.body = resp.text[:1000]  # Truncate for report
            response.latency_ms = (time.time() - start_time) * 1000
            
            logger.debug(f"Response: {resp.status_code} in {response.latency_ms:.2f}ms")
            
        except requests.exceptions.RequestException as e:
            response.error = str(e)
            response.latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"Request failed: {e}")
        
        return response
    
    def _apply_rate_limit(self, request_index: int, start_time: float) -> None:
        """
        Apply rate limiting between requests.
        
        Args:
            request_index: Index of current request
            start_time: Start time of replay session
        """
        if self.rps is None or self.rps <= 0:
            return
        
        # Calculate expected time for this request
        expected_time = start_time + (request_index / self.rps)
        current_time = time.time()
        
        # Sleep if we're ahead of schedule
        if current_time < expected_time:
            sleep_time = expected_time - current_time
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
    
    def replay_traffic(self, requests_list: List[ReplayRequest]) -> ReplayReport:
        """
        Replay a list of requests to the staging environment.
        
        Args:
            requests_list: List of requests to replay
            
        Returns:
            Replay report
        """
        self.report = ReplayReport()
        self.report.total_requests = len(requests_list)
        
        start_time = time.time()
        
        logger.info(f"Starting replay of {len(requests_list)} requests...")
        
        for idx, request in enumerate(requests_list):
            logger.debug(f"Processing request {idx + 1}/{len(requests_list)}")
            
            # Check if request should be skipped
            should_skip, skip_reason = self._should_skip_request(request)
            
            if should_skip:
                self.report.skipped += 1
                self.report.skipped_reasons[skip_reason] = (
                    self.report.skipped_reasons.get(skip_reason, 0) + 1
                )
                logger.info(f"Skipped: {request.method} {request.url} - {skip_reason}")
                continue
            
            # Rewrite request for staging
            staging_request = self._rewrite_request(request)
            self.report.processed += 1
            
            # Dry-run mode: log but don't send
            if self.dry_run:
                logger.info(
                    f"[DRY-RUN] Would send: {staging_request.method} {staging_request.url}"
                )
                continue
            
            # Apply rate limiting
            self._apply_rate_limit(idx, start_time)
            
            # Send request
            response = self._send_request(staging_request)
            self.report.sent += 1
            
            # Update report
            if response.error:
                self.report.errors += 1
                self.report.failed += 1
                
                # Sample errors
                if len(self.report.error_samples) < 10:
                    self.report.error_samples.append({
                        "request": {
                            "method": staging_request.method,
                            "url": staging_request.url,
                        },
                        "error": response.error,
                    })
            else:
                if response.status_code and 200 <= response.status_code < 300:
                    self.report.successful += 1
                else:
                    self.report.failed += 1
                
                # Sample responses
                if len(self.report.response_samples) < 10:
                    self.report.response_samples.append({
                        "request": {
                            "method": staging_request.method,
                            "url": staging_request.url,
                        },
                        "response": {
                            "status_code": response.status_code,
                            "latency_ms": response.latency_ms,
                        },
                    })
        
        self.report.total_duration_seconds = time.time() - start_time
        
        logger.info("Replay complete!")
        logger.info(f"  Total requests: {self.report.total_requests}")
        logger.info(f"  Processed: {self.report.processed}")
        logger.info(f"  Skipped: {self.report.skipped}")
        logger.info(f"  Sent: {self.report.sent}")
        logger.info(f"  Successful: {self.report.successful}")
        logger.info(f"  Failed: {self.report.failed}")
        logger.info(f"  Errors: {self.report.errors}")
        logger.info(f"  Duration: {self.report.total_duration_seconds:.2f}s")
        
        return self.report
    
    def save_report(self, output_path: Path) -> None:
        """
        Save replay report to JSON file.
        
        Args:
            output_path: Path to save report
        """
        with open(output_path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2)
        
        logger.info(f"Report saved to: {output_path}")


def main():
    """Main entry point for shadow replay tool."""
    parser = argparse.ArgumentParser(
        description="Shadow Traffic Replay Tool - Mirror production traffic to staging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run replay of HAR file (safe, no actual requests sent)
  python shadow_replay.py --input traffic.har --staging-url https://staging.example.com

  # Actual replay with rate limiting and auth
  python shadow_replay.py --input traffic.json --staging-url https://staging.example.com \\
                          --send --rps 10 --auth "your-token-here"

  # Allow state-changing methods (use with caution!)
  python shadow_replay.py --input traffic.har --staging-url https://staging.example.com \\
                          --send --allow-modifying

Safety Notes:
  - Default mode is dry-run (--dry-run flag, no --send flag needed)
  - State-changing methods (POST, PUT, PATCH, DELETE) are skipped by default
  - Use --allow-modifying to explicitly enable state-changing requests
  - Tool will abort if target URL looks like production (override with --force)
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to input traffic file (HAR or JSON format)"
    )
    
    parser.add_argument(
        "--staging-url",
        required=True,
        help="Base URL of staging environment (e.g., https://staging.example.com)"
    )
    
    # Operation mode
    parser.add_argument(
        "--send",
        action="store_true",
        help="Actually send requests (default is dry-run mode)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run mode: process requests but don't send (default)"
    )
    
    # Safety options
    parser.add_argument(
        "--skip-methods",
        nargs="+",
        help="HTTP methods to skip (default: POST PUT PATCH DELETE unless --allow-modifying)"
    )
    
    parser.add_argument(
        "--allow-modifying",
        action="store_true",
        help="Explicitly allow state-changing methods (POST, PUT, PATCH, DELETE)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if target looks like production (NOT RECOMMENDED)"
    )
    
    # Rate limiting and performance
    parser.add_argument(
        "--rps",
        type=float,
        help="Requests per second rate limit (optional)"
    )
    
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts for failed requests (default: 3)"
    )
    
    # Authentication
    parser.add_argument(
        "--auth",
        help="Authorization token/key to use for requests"
    )
    
    # Output
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("replay_report.json"),
        help="Path to save replay report JSON (default: replay_report.json)"
    )
    
    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Override dry_run if --send is specified
    dry_run = not args.send
    
    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Determine input format
    input_suffix = args.input.suffix.lower()
    
    try:
        # Initialize tool
        tool = ShadowReplayTool(
            staging_base_url=args.staging_url,
            dry_run=dry_run,
            skip_methods=args.skip_methods,
            allow_modifying=args.allow_modifying,
            rps=args.rps,
            auth=args.auth,
            retries=args.retries,
            force=args.force,
        )
        
        # Parse input
        if input_suffix == ".har":
            requests_list = tool.parse_har(args.input)
        elif input_suffix == ".json":
            requests_list = tool.parse_json(args.input)
        else:
            logger.error(f"Unsupported file format: {input_suffix}")
            logger.error("Supported formats: .har, .json")
            sys.exit(1)
        
        if not requests_list:
            logger.warning("No requests found in input file")
            sys.exit(0)
        
        # Replay traffic
        report = tool.replay_traffic(requests_list)
        
        # Save report
        tool.save_report(args.report)
        
        # Exit with error code if there were failures
        if report.errors > 0:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.warning("\nReplay interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
