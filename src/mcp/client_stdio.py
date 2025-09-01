#!/usr/bin/env python3
"""
MCP Client - Stdio-Based Implementation for FastMCP
===================================================

Stdio-based MCP client for communicating with FastMCP servers.
Uses the standard MCP protocol over stdin/stdout.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

logger = logging.getLogger(__name__)

class StdioMCPClient:
    """Stdio-based MCP client for FastMCP servers with singleton pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, trace_id: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, trace_id: str):
        if not self._initialized:
            self.trace_id = trace_id
            self.server_processes: Dict[str, subprocess.Popen] = {}
            self.request_id_counter = 0
            self._initialized = True
        else:
            # Update trace_id for existing instance
            self.trace_id = trace_id
        
        # MCP server configurations (FastMCP stdio-based)
        self.server_configs = {
            'kg': {
                'script': 'src/mcp/servers/kg_server.py',
                'name': 'KG Server'
            },
            'dense': {
                'script': 'src/mcp/servers/dense_server.py',
                'name': 'Dense Server'
            },
            'memory': {
                'script': 'src/mcp/servers/memory_server.py',
                'name': 'Memory Server'
            },
            'validator': {
                'script': 'src/mcp/servers/validator_server.py',
                'name': 'Validator Server'
            },
            'trace': {
                'script': 'src/mcp/servers/trace_server.py',
                'name': 'Trace Server'
            },
            'explain': {
                'script': 'src/mcp/servers/explain_server.py',
                'name': 'Explain Server'
            }
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Only start servers if not already running
        if not self.server_processes:
            await self.start_servers()
        return self
        
    async def start_servers(self):
        """Start all MCP servers with stdio communication"""
        project_root = Path(__file__).parent.parent.parent
        
        # Set up environment with proper PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root) + ':' + env.get('PYTHONPATH', '')
        
        for server_name, config in self.server_configs.items():
            try:
                logger.info(f"Starting {server_name} server ({config['name']})...")
                
                # Start server process with stdio and proper environment
                process = subprocess.Popen(
                    ['python3', config['script']],
                    cwd=project_root,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0,  # Unbuffered for real-time communication
                    env=env  # Pass environment with PYTHONPATH
                )
                
                self.server_processes[server_name] = process
                
                # Give server time to initialize
                await asyncio.sleep(1)
                
                # Check if server is running
                if process.poll() is None:
                    logger.info(f"✅ {server_name} server started successfully")
                    
                    # Initialize MCP connection
                    await self._initialize_server(server_name)
                else:
                    logger.error(f"❌ {server_name} server failed to start")
                    stderr_output = process.stderr.read() if process.stderr else "No error output"
                    logger.error(f"Server error: {stderr_output}")
                    
            except Exception as e:
                logger.error(f"Error starting {server_name} server: {e}")
    
    async def _initialize_server(self, server_name: str):
        """Initialize MCP connection with a server"""
        try:
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "HelixGRAG MCP Client",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self._send_request(server_name, init_request)
            
            if response and "result" in response:
                logger.info(f"✅ {server_name} server initialized successfully")
                
                # Send initialized notification
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                
                await self._send_notification(server_name, initialized_notification)
            else:
                logger.error(f"❌ Failed to initialize {server_name} server: {response}")
                
        except Exception as e:
            logger.error(f"Error initializing {server_name} server: {e}")
    
    def _get_request_id(self) -> str:
        """Generate unique request ID"""
        self.request_id_counter += 1
        return f"req_{self.request_id_counter}_{int(time.time())}"
    
    async def _send_request(self, server_name: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC request to server and get response"""
        try:
            process = self.server_processes.get(server_name)
            if not process or process.poll() is not None:
                return {"error": f"Server {server_name} is not running"}
            
            # Send request
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                asyncio.create_task(self._read_line(process)),
                timeout=10.0
            )
            
            if response_line:
                return json.loads(response_line.strip())
            else:
                return {"error": "No response from server"}
                
        except asyncio.TimeoutError:
            return {"error": f"Timeout waiting for response from {server_name}"}
        except Exception as e:
            return {"error": f"Communication error with {server_name}: {str(e)}"}
    
    async def _send_notification(self, server_name: str, notification: Dict[str, Any]):
        """Send JSON-RPC notification to server (no response expected)"""
        try:
            process = self.server_processes.get(server_name)
            if not process or process.poll() is not None:
                return
            
            # Send notification
            notification_json = json.dumps(notification) + "\n"
            process.stdin.write(notification_json)
            process.stdin.flush()
            
        except Exception as e:
            logger.error(f"Error sending notification to {server_name}: {e}")
    
    async def _read_line(self, process: subprocess.Popen) -> str:
        """Read a line from process stdout asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process.stdout.readline)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - don't stop servers for singleton pattern"""
        # Don't stop servers automatically - they will be reused
        # Only stop on explicit shutdown or process termination
        pass
    
    async def stop_servers(self):
        """Stop all MCP servers"""
        for server_name, process in self.server_processes.items():
            try:
                if process.poll() is None:
                    logger.info(f"Stopping {server_name} server...")
                    
                    # Send shutdown notification
                    shutdown_notification = {
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled"
                    }
                    await self._send_notification(server_name, shutdown_notification)
                    
                    # Terminate process
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        
            except Exception as e:
                logger.error(f"Error stopping {server_name} server: {e}")
    
    async def call_tool(self, 
                       server: str, 
                       tool_name: str, 
                       arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on a specific MCP server
        
        Args:
            server: Server name (kg, dense, memory, etc.)
            tool_name: Tool name to call
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        try:
            if server not in self.server_processes:
                return {"error": f"Unknown server: {server}"}
            
            # Add trace_id to arguments if not present
            if 'trace_id' not in arguments:
                arguments['trace_id'] = self.trace_id
            
            # Create tool call request
            request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = await self._send_request(server, request)
            
            if response and "result" in response:
                return response["result"]
            elif response and "error" in response:
                return {"error": response["error"]}
            else:
                return {"error": f"Invalid response from {server}"}
                
        except Exception as e:
            logger.error(f"Error calling {tool_name} on {server}: {e}")
            return {
                "error": str(e),
                "server": server,
                "tool": tool_name
            }
    
    async def list_tools(self, server: str) -> List[str]:
        """List available tools for a server"""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._get_request_id(),
                "method": "tools/list"
            }
            
            response = await self._send_request(server, request)
            
            if response and "result" in response and "tools" in response["result"]:
                return [tool["name"] for tool in response["result"]["tools"]]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error listing tools for {server}: {e}")
            return []
    
    async def health_check(self, server: str) -> Dict[str, Any]:
        """Check if a server is healthy"""
        try:
            process = self.server_processes.get(server)
            if not process:
                return {"healthy": False, "server": server, "error": "Server not found"}
            
            if process.poll() is not None:
                return {"healthy": False, "server": server, "error": "Server process terminated"}
            
            # Try to list tools as a health check
            tools = await self.list_tools(server)
            return {
                "healthy": True, 
                "server": server, 
                "tool_count": len(tools),
                "tools": tools
            }
            
        except Exception as e:
            return {"healthy": False, "server": server, "error": str(e)}

# Convenience functions for MAS integration
async def with_stdio_mcp_client(trace_id: str, func):
    """Context manager wrapper for stdio MCP client"""
    async with StdioMCPClient(trace_id) as client:
        return await func(client)
