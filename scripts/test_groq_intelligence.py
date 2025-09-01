#!/usr/bin/env python3
"""
Groq LLM Intelligence Test
=========================

Test to verify that the Groq Llama model is providing intelligent responses
in the NLI workflow instead of mock responses.
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

from src.mcp.client_stdio import StdioMCPClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_groq_llm_responses():
    """Test Groq LLM responses directly through validator server"""
    
    test_prompts = [
        {
            "name": "Intent Detection",
            "prompt": """Analyze this biomedical query and determine the intent:
Query: What does metformin treat?

Determine:
- primary_intent: factoid, enumeration, causal, comparative
- reasoning_type: simple_lookup, multi_hop, temporal, quantitative
- complexity_level: low, medium, high

Format as JSON with keys: primary_intent, reasoning_type, complexity_level"""
        },
        {
            "name": "Entity Extraction", 
            "prompt": """Extract biomedical entities from this query:
Query: What are the side effects of aspirin in diabetic patients?

Extract:
1. Diseases/conditions
2. Drugs/compounds  
3. Patient populations
4. Key phrases for search

Format as JSON with keys: diseases, drugs, populations, phrases"""
        },
        {
            "name": "Query Transformation",
            "prompt": """Transform this biomedical query for optimal retrieval:
Query: How does insulin resistance lead to type 2 diabetes?

Provide:
1. Logical subquestions (skyline approach)
2. Simplified version
3. Rewritten query with streamlined context
4. Alternative phrasings

Format as JSON with keys: subquestions, simplified, rewritten, alternatives"""
        }
    ]
    
    logger.info("🧪 Testing Groq LLM Intelligence in Validator Server")
    
    try:
        async with StdioMCPClient("test-groq-intelligence") as client:
            logger.info("✅ MCP Client initialized")
            
            for i, test in enumerate(test_prompts, 1):
                logger.info(f"🔍 Test {i}: {test['name']}")
                
                start_time = time.time()
                
                # Call validator with LLM prompt
                result = await client.call_tool("validator", "validate_answer", {
                    "query": test["prompt"],
                    "answer": "",  # Empty answer triggers LLM mode
                    "evidence": []
                })
                
                execution_time = time.time() - start_time
                
                print(f"\n{'='*60}")
                print(f"🎯 TEST {i}: {test['name']}")
                print(f"{'='*60}")
                print(f"Prompt: {test['prompt'][:100]}...")
                print(f"Execution Time: {execution_time:.2f}s")
                print(f"Response Type: {type(result)}")
                
                if "llm_analysis" in result:
                    print(f"✅ Real LLM Response Detected!")
                    print(f"LLM Analysis: {result['llm_analysis'][:200]}...")
                else:
                    print(f"⚠️ Mock Response (no llm_analysis field)")
                
                print(f"Structured Response: {json.dumps(result, indent=2)[:300]}...")
                print(f"{'='*60}\n")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing Groq LLM: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Groq LLM Intelligence Test")
    
    success = await test_groq_llm_responses()
    
    if success:
        logger.info("✅ Groq LLM Intelligence Test completed!")
        return 0
    else:
        logger.error("❌ Groq LLM Intelligence Test failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
