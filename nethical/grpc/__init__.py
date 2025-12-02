"""gRPC module for Nethical.

Provides gRPC server and client for low-latency
inter-service communication.

Features:
- GovernanceService implementation
- Streaming support for batch operations
- Client with retry and timeout handling

All implementations adhere to the 25 Fundamental Laws.
"""

from .server import GovernanceServicer, create_grpc_server
from .client import NethicalGRPCClient

__all__ = [
    "GovernanceServicer",
    "create_grpc_server",
    "NethicalGRPCClient",
]
