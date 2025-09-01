"""
Model Context Protocol (MCP) Package
===================================

This package implements MCP client and servers for the MAS system,
providing inter-service communication with full trace propagation.
"""

from .client_stdio import StdioMCPClient as MCPClient

class MCPError(Exception):
    """MCP-related errors"""
    pass

__version__ = "1.0.0"
__all__ = [
    "MCPClient",
    "MCPError"
]
