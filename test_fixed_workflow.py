#!/usr/bin/env python3
"""
Test script for the fixed LangGraph workflow
"""

import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mas.nli_orchestrator import NLIMASOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_workflow():
    """Test the fixed LangGraph workflow with a biomedical query"""
    
    # Test query
    query = "What are the side effects of metformin for diabetes treatment?"
    
    logger.info(f"🧪 Testing LangGraph workflow with query: {query}")
    
    try:
        # Initialize orchestrator
        orchestrator = NLIMASOrchestrator()
        await orchestrator.initialize()
        
        # Process query
        result = await orchestrator.process_query(query)
        
        # Print results
        logger.info("=" * 60)
        logger.info("WORKFLOW TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Success: {result.get('success', False)}")
        logger.info(f"Intent: {result.get('intent', 'unknown')}")
        logger.info(f"Mode: {result.get('mode', 'unknown')}")
        logger.info(f"Entities: {result.get('entities', [])}")
        logger.info(f"Evidence KG: {len(result.get('evidence', {}).get('kg_paths', []))}")
        logger.info(f"Evidence Dense: {len(result.get('evidence', {}).get('dense_hits', []))}")
        logger.info(f"Answer: {result.get('answer', 'No answer')[:200]}...")
        
        if not result.get('success'):
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        await orchestrator.cleanup()
        
        return result.get('success', False)
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow())
    if success:
        logger.info("✅ Workflow test completed successfully!")
    else:
        logger.error("❌ Workflow test failed!")
        sys.exit(1)
