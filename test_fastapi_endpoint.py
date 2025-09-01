#!/usr/bin/env python3
"""
Test script for the FastAPI endpoint with fixed LangGraph workflow
"""

import asyncio
import logging
import os
import sys
import requests
import time
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_fastapi_endpoint():
    """Test the FastAPI endpoint with the fixed workflow"""
    
    # Start the FastAPI server
    logger.info("🚀 Starting FastAPI server...")
    server_process = subprocess.Popen([
        "python3", "-m", "uvicorn", "src.api.v1.main:app", 
        "--host", "0.0.0.0", "--port", "8000"
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        # Test query
        query = "What are the side effects of metformin for diabetes treatment?"
        
        logger.info(f"🧪 Testing FastAPI endpoint with query: {query}")
        
        # Make request to FastAPI endpoint
        response = requests.post(
            "http://localhost:8000/api/v1/query",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Print results
            logger.info("=" * 60)
            logger.info("FASTAPI ENDPOINT TEST RESULTS")
            logger.info("=" * 60)
            logger.info(f"Success: {result.get('success', False)}")
            logger.info(f"Intent: {result.get('intent', 'unknown')}")
            logger.info(f"Mode: {result.get('mode', 'unknown')}")
            logger.info(f"Entities: {result.get('entities', [])}")
            logger.info(f"Evidence KG: {len(result.get('evidence', {}).get('kg_paths', []))}")
            logger.info(f"Evidence Dense: {len(result.get('evidence', {}).get('dense_hits', []))}")
            logger.info(f"Answer: {result.get('answer', 'No answer')[:200]}...")
            
            return result.get('success', False)
        else:
            logger.error(f"❌ FastAPI request failed with status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Stop the server
        logger.info("🛑 Stopping FastAPI server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    success = asyncio.run(test_fastapi_endpoint())
    if success:
        logger.info("✅ FastAPI endpoint test completed successfully!")
    else:
        logger.error("❌ FastAPI endpoint test failed!")
        sys.exit(1)
