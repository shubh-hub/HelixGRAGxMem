#!/usr/bin/env python3
"""
Simple NLI Workflow Test
=======================

Simplified test to verify the 12-step NLI workflow with fixed MCP client.
Tests one query end-to-end to ensure LLM-driven intelligence works.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mas.nli_orchestrator import NLISession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_single_query():
    """Test a single query through the complete NLI workflow"""
    
    test_query = "What does metformin treat?"
    
    logger.info(f"🚀 Testing NLI workflow with query: '{test_query}'")
    
    try:
        async with NLISession() as orchestrator:
            logger.info("✅ NLI Session initialized successfully")
            
            start_time = time.time()
            
            # Execute the complete 12-step workflow
            result = await orchestrator.execute_nli_workflow(
                query=test_query,
                persona="doctor"
            )
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Workflow completed in {execution_time:.2f} seconds")
            
            # Print key results
            print("\n" + "="*60)
            print("🎯 NLI WORKFLOW RESULTS")
            print("="*60)
            print(f"Query: {test_query}")
            print(f"Intent: {result.get('intent', 'Unknown')}")
            print(f"Mode: {result.get('mode', 'Unknown')}")
            print(f"Entities: {result.get('entities_surface', [])}")
            print(f"Final Answer: {result.get('final_answer', 'No answer generated')}")
            print(f"Execution Time: {execution_time:.2f}s")
            
            # Print workflow steps executed
            if 'spans' in result:
                print(f"\nWorkflow Steps Executed: {len(result['spans'])}")
                for span in result['spans']:
                    print(f"  - {span.get('node', 'Unknown')}: {span.get('duration_ms', 0):.0f}ms")
            
            print("="*60)
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    logger.info("🧪 Starting Simple NLI Workflow Test")
    
    result = await test_single_query()
    
    if result:
        logger.info("✅ Test completed successfully!")
        return 0
    else:
        logger.error("❌ Test failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
