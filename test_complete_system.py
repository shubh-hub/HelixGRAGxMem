#!/usr/bin/env python3
"""
Comprehensive test of the complete HelixGRAGxMem system with evidence retrieval
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

async def test_complete_system():
    """Test the complete system with multiple biomedical queries"""
    
    test_queries = [
        "What are the side effects of metformin for diabetes treatment?",
        "How does aspirin work for cardiovascular disease?",
        "What are the interactions between warfarin and other drugs?"
    ]
    
    logger.info("🧪 Testing complete HelixGRAGxMem system with evidence retrieval")
    
    try:
        # Initialize orchestrator
        orchestrator = NLIMASOrchestrator()
        await orchestrator.initialize()
        
        logger.info(f"✅ System initialized with {len(orchestrator.tools)} MCP tools")
        logger.info(f"Available tools: {[tool.name for tool in orchestrator.tools]}")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST {i}: {query}")
            logger.info(f"{'='*60}")
            
            # Process query
            result = await orchestrator.process_query(query)
            
            # Print results
            logger.info(f"Success: {result.get('success', False)}")
            logger.info(f"Intent: {result.get('intent', 'unknown')}")
            logger.info(f"Mode: {result.get('mode', 'unknown')}")
            logger.info(f"Entities: {result.get('entities', [])}")
            
            evidence = result.get('evidence', {})
            kg_evidence = evidence.get('kg_paths', [])
            dense_evidence = evidence.get('dense_hits', [])
            
            logger.info(f"Evidence - KG: {len(kg_evidence)}, Dense: {len(dense_evidence)}")
            
            if dense_evidence:
                logger.info("Sample dense evidence:")
                for j, item in enumerate(dense_evidence[:3], 1):
                    logger.info(f"  {j}. {item.get('text', 'N/A')[:100]}...")
            
            if kg_evidence:
                logger.info("Sample KG evidence:")
                for j, item in enumerate(kg_evidence[:3], 1):
                    logger.info(f"  {j}. {item.get('text', 'N/A')[:100]}...")
            
            answer = result.get('answer', 'No answer')
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Answer preview: {answer[:200]}...")
            
            if not result.get('success'):
                logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        await orchestrator.cleanup()
        
        logger.info("\n✅ Complete system test finished successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_system())
    if success:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("💥 Tests failed!")
        sys.exit(1)
