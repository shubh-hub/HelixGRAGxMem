#!/usr/bin/env python3
"""
Evidence Pipeline Debugger
==========================
Systematically debug what each node produces vs expectations
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mas.nli_orchestrator import NLIMASOrchestrator
from src.mas.state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_evidence_pipeline():
    """Debug each step of the evidence retrieval pipeline"""
    
    print("🔍 DEBUGGING EVIDENCE PIPELINE")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = NLIMASOrchestrator()
    await orchestrator.initialize_mcp_client()
    
    # Test query
    test_query = "What drugs treat hypertension?"
    
    print(f"📝 Test Query: {test_query}")
    print()
    
    # Skip to evidence retrieval debugging
    initial_state = create_initial_state(query=test_query)
    analysis_state = await orchestrator._query_analysis_node(initial_state)
    entity_state = await orchestrator._entity_extraction_node(analysis_state)
    
    # Step 3: Evidence Retrieval - Debug Each Tool
    print("🔍 STEP 3: Evidence Retrieval")
    print("-" * 30)
    
    # Debug KG Tool
    print("📊 KG Search Tool Debug:")
    kg_tool = next((tool for tool in orchestrator.tools if tool.name == "search_kg"), None)
    if kg_tool:
        print(f"  Tool found: {kg_tool.name}")
        
        kg_result = await kg_tool.ainvoke({
            "query": test_query,
            "intent": entity_state.get('intent', 'factoid'),
            "max_results": 5
        })
        
        print(f"  KG Result type: {type(kg_result)}")
        print(f"  KG Full result: {kg_result}")
        
    else:
        print("  ❌ KG tool not found!")
    
    print()
    
    # Debug Dense Tool
    print("📚 Dense Search Tool Debug:")
    dense_tool = next((tool for tool in orchestrator.tools if tool.name == "search_passages"), None)
    if dense_tool:
        print(f"  Tool found: {dense_tool.name}")
        
        dense_result = await dense_tool.ainvoke({
            "query": test_query,
            "max_results": 5
        })
        
        print(f"  Dense Result type: {type(dense_result)}")
        print(f"  Dense Full result: {dense_result}")
        
    else:
        print("  ❌ Dense tool not found!")
    
    print()
    
    # Step 4: Full Evidence Retrieval Node
    print("🔗 STEP 4: Full Evidence Retrieval Node")
    print("-" * 30)
    evidence_state = await orchestrator._evidence_retrieval_node(entity_state)
    
    evidence = evidence_state.get('evidence', {})
    kg_paths = evidence.get('kg_paths', [])
    dense_hits = evidence.get('dense_hits', [])
    
    print(f"KG paths count: {len(kg_paths)}")
    print(f"Dense hits count: {len(dense_hits)}")
    
    if kg_paths:
        print(f"Sample KG path: {json.dumps(kg_paths[0], indent=2)[:300]}...")
    
    if dense_hits:
        print(f"Sample Dense hit: {json.dumps(dense_hits[0], indent=2)[:300]}...")
    
    print()
    
    # Step 5: Reasoning Generation
    print("🧠 STEP 5: Reasoning Generation")
    print("-" * 30)
    reasoning_state = await orchestrator._reasoning_generation_node(evidence_state)
    
    print(f"Final answer: {reasoning_state.get('answer', 'None')}")
    print(f"Confidence: {reasoning_state.get('confidence', 'None')}")
    print(f"Reasoning metadata: {reasoning_state.get('metadata', {}).get('reasoning', 'None')}")
    
    print()
    print("🎯 ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(debug_evidence_pipeline())
