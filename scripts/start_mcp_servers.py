#!/usr/bin/env python3
"""
MCP Servers Startup Script
==========================

This script starts all MCP servers required for the MAS system:
- Trace Server (port 8005) - Critical for observability
- KG Server (port 8001) - Knowledge graph operations
- Dense Server (port 8002) - Dense retrieval operations
- Memory Server (port 8003) - Memory management operations
- Validator Server (port 8004) - Quality assurance
- Explain Server (port 8006) - Persona-aware explanations
"""

import asyncio
import subprocess
import time
import sys
import os
import signal
import logging
from pathlib import Path
from typing import List, Dict
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server configurations
SERVERS = {
    "trace": {
        "port": 8005,
        "module": "src.mcp.servers.trace_server_mcp",
        "priority": 1,  # Start first (others depend on it)
        "health_endpoint": "/health"
    },
    "kg": {
        "port": 8001,
        "module": "src.mcp.servers.kg_server_mcp",
        "priority": 2,
        "health_endpoint": "/health"
    },
    "dense": {
        "port": 8002,
        "module": "src.mcp.servers.dense_server_mcp", 
        "priority": 2,
        "health_endpoint": "/health"
    },
    "memory": {
        "port": 8003,
        "module": "src.mcp.servers.memory_server_mcp",
        "priority": 2,
        "health_endpoint": "/health"
    },
    "validator": {
        "port": 8004,
        "module": "src.mcp.servers.validator_server_mcp",
        "priority": 3,
        "health_endpoint": "/health"
    },
    "explain": {
        "port": 8006,
        "module": "src.mcp.servers.explain_server_mcp",
        "priority": 3,
        "health_endpoint": "/health"
    }
}

class MCPServerManager:
    """Manages lifecycle of all MCP servers"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        
    async def start_all_servers(self):
        """Start all servers in priority order"""
        logger.info("🚀 Starting MAS MCP Servers...")
        
        # Sort servers by priority
        sorted_servers = sorted(SERVERS.items(), key=lambda x: x[1]["priority"])
        
        for server_name, config in sorted_servers:
            await self.start_server(server_name, config)
            
            # Wait for server to be ready
            if await self.wait_for_health(server_name, config, timeout=30):
                logger.info(f"✅ {server_name} server ready on port {config['port']}")
            else:
                logger.error(f"❌ {server_name} server failed to start")
                await self.stop_all_servers()
                return False
        
        self.running = True
        logger.info("🎉 All MCP servers started successfully!")
        return True
    
    async def start_server(self, server_name: str, config: Dict):
        """Start individual server"""
        logger.info(f"Starting {server_name} server on port {config['port']}...")
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Start server process
        cmd = [
            sys.executable, "-m", config["module"]
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd(),
                env={**dict(os.environ), "PYTHONPATH": str(Path.cwd())}
            )
            
            self.processes[server_name] = process
            logger.info(f"Started {server_name} server (PID: {process.pid})")
            
            # Give server a moment to initialize
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Failed to start {server_name} server: {e}")
            raise
    
    async def wait_for_health(self, server_name: str, config: Dict, timeout: int = 30) -> bool:
        """Wait for server to become healthy"""
        url = f"http://localhost:{config['port']}{config['health_endpoint']}"
        
        for attempt in range(timeout):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={}) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            if health_data.get("status") == "healthy":
                                return True
                            
            except Exception as e:
                if attempt == 0:
                    logger.info(f"Waiting for {server_name} server to be ready...")
                
            await asyncio.sleep(1)
        
        return False
    
    async def stop_all_servers(self):
        """Stop all running servers"""
        logger.info("🛑 Stopping all MCP servers...")
        
        for server_name, process in self.processes.items():
            try:
                logger.info(f"Stopping {server_name} server (PID: {process.pid})")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {server_name} server")
                    process.kill()
                    process.wait()
                    
                logger.info(f"✅ {server_name} server stopped")
                
            except Exception as e:
                logger.error(f"Error stopping {server_name} server: {e}")
        
        self.processes.clear()
        self.running = False
        logger.info("All servers stopped")
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all servers"""
        results = {}
        
        for server_name, config in SERVERS.items():
            url = f"http://localhost:{config['port']}{config['health_endpoint']}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={}, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            results[server_name] = health_data.get("status") == "healthy"
                        else:
                            results[server_name] = False
                            
            except Exception as e:
                logger.warning(f"Health check failed for {server_name}: {e}")
                results[server_name] = False
        
        return results
    
    async def monitor_servers(self):
        """Monitor server health and restart if needed"""
        logger.info("🔍 Starting server monitoring...")
        
        while self.running:
            try:
                health_results = await self.health_check_all()
                
                unhealthy_servers = [name for name, healthy in health_results.items() if not healthy]
                
                if unhealthy_servers:
                    logger.warning(f"Unhealthy servers detected: {unhealthy_servers}")
                    # In production, we might restart unhealthy servers here
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error during server monitoring: {e}")
                await asyncio.sleep(10)

async def main():
    """Main startup function"""
    import os
    
    manager = MCPServerManager()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(manager.stop_all_servers())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all servers
        success = await manager.start_all_servers()
        
        if not success:
            logger.error("Failed to start all servers")
            return 1
        
        # Print status
        print("\n" + "="*60)
        print("🎯 MAS MCP Servers Status")
        print("="*60)
        
        health_results = await manager.health_check_all()
        for server_name, config in SERVERS.items():
            status = "✅ HEALTHY" if health_results.get(server_name, False) else "❌ UNHEALTHY"
            print(f"{server_name.upper():>10} Server: {status} (port {config['port']})")
        
        print("="*60)
        print("🚀 MAS System Ready for Operations!")
        print("📊 Trace logs: ./logs/")
        print("🔍 Health checks: http://localhost:<port>/health")
        print("⚠️  Press Ctrl+C to stop all servers")
        print("="*60)
        
        # Start monitoring
        await manager.monitor_servers()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        return 1
    finally:
        await manager.stop_all_servers()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
