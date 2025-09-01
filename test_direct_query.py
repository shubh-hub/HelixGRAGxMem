#!/usr/bin/env python3
"""
Direct Query Test - Test the system without FastAPI overhead
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mas.nli_orchestrator import NLIMASOrchestrator

async def test_direct_query():
    """Test query processing directly"""
    print("🧪 Testing Direct Query Processing...")
    
    try:
        orchestrator = NLIMASOrchestrator()
        
        result = await orchestrator.process_query(
            query="What drugs treat hypertension?",
            trace_id="direct_test_001"
        )
        
        print(f"✅ Success: {result['success']}")
        if result['success']:
            print(f"📝 Answer: {result.get('answer', 'No answer')}")
            print(f"🎯 Confidence: {result.get('confidence', 0.0)}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Direct test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_query())
