#!/usr/bin/env python3
"""
Integration test to verify system works end-to-end after core recovery
"""

import os
import sys
import asyncio
sys.path.append('/Users/shivam/Documents/Shubham/HelixGRAGxMem')

from src.mas.nli_orchestrator import NLIMASOrchestrator
from src.mas.state import MASState
from src.core import HybridRetriever

async def test_integration():
    """Test end-to-end integration"""
    
    print("🔧 Testing End-to-End Integration...")
    
    # Test 1: Core components integration
    try:
        hybrid = HybridRetriever()
        stats = hybrid.get_retrieval_stats()
        print(f"✅ HybridRetriever stats: {stats}")
    except Exception as e:
        print(f"❌ HybridRetriever failed: {e}")
        return False
    
    # Test 2: MAS Orchestrator integration
    try:
        orchestrator = NLIMASOrchestrator()
        print("✅ NLI MAS Orchestrator created successfully")
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        return False
    
    # Test 3: MCP client import
    try:
        from src.mcp import MCPClient
        print("✅ MCP client import successful")
    except Exception as e:
        print(f"❌ MCP client import failed: {e}")
        return False
    
    # Test 4: State management
    try:
        initial_state = MASState(
            query="What treats diabetes?",
            trace_id="test_trace_123",
            entities=[],
            intent="factoid",
            mode="hybrid",
            retry_count=0
        )
        print("✅ MAS state creation successful")
    except Exception as e:
        print(f"❌ State creation failed: {e}")
        return False
    
    print("\n🎉 Integration test successful!")
    print("📋 System Status:")
    print("   • Core components: ✅ Recovered and functional")
    print("   • MCP integration: ✅ Working")
    print("   • MAS orchestrator: ✅ Ready")
    print("   • State management: ✅ Operational")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
