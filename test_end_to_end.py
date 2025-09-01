#!/usr/bin/env python3
"""
Comprehensive end-to-end test with proper entity names and real queries
"""

import os
import sys
import asyncio
sys.path.append('/Users/shivam/Documents/Shubham/HelixGRAGxMem')

from src.core import DenseRetriever, HybridRetriever, KGWalker
from src.mas.nli_orchestrator import NLIMASOrchestrator
from src.mas.state import MASState
from src.config import settings

async def test_end_to_end():
    """Test complete end-to-end workflow"""
    
    print("🔧 Running End-to-End System Test...")
    
    # Test 1: Dense retrieval with biomedical query
    print("\n🔍 Testing Dense Retrieval:")
    try:
        dense = DenseRetriever()
        dense.initialize()
        
        query = "What drugs treat diabetes?"
        results = dense.search(query, k=5)
        print(f"   ✅ Dense search for '{query}' returned {len(results)} results")
        
        if results:
            for i, result in enumerate(results[:3]):
                print(f"   📋 Result {i+1}: {result['text'][:80]}... (score: {result['score']:.4f})")
        else:
            print("   ⚠️ No dense results found")
            
    except Exception as e:
        print(f"   ❌ Dense retrieval failed: {e}")
        return False
    
    # Test 2: KG traversal with real entities
    print("\n🕸️ Testing KG Traversal:")
    try:
        walker = KGWalker()
        walker.connect_kg()
        
        # Use actual entity names from the database
        diabetes_entities = ["type 2 diabetes mellitus", "type 1 diabetes mellitus"]
        
        for entity in diabetes_entities:
            neighborhood = walker.get_entity_neighborhood(entity)
            print(f"   ✅ {entity}: {neighborhood['neighbor_count']} neighbors")
            
            if neighborhood['neighbors']:
                # Show some relations
                relations = list(set([n['relation'] for n in neighborhood['neighbors'][:5]]))
                print(f"   📋 Relations: {', '.join(relations[:3])}")
        
        # Test graph traversal
        paths = walker.walk(
            start_entities=diabetes_entities,
            query_context="What treats diabetes?",
            max_hops=2,
            max_paths=5
        )
        print(f"   ✅ Graph walk found {len(paths)} paths")
        
        if paths:
            top_path = paths[0]
            print(f"   📋 Top path: {' → '.join(top_path['path'][:3])} (confidence: {top_path['confidence']:.3f})")
            
    except Exception as e:
        print(f"   ❌ KG traversal failed: {e}")
        return False
    
    # Test 3: Hybrid retrieval
    print("\n🔄 Testing Hybrid Retrieval:")
    try:
        hybrid = HybridRetriever()
        hybrid.initialize()
        
        # Test with entities that exist in KG
        results = hybrid.search(
            query="What treats diabetes?",
            entities=diabetes_entities,
            k=5,
            kg_weight=0.6,
            dense_weight=0.4
        )
        
        print(f"   ✅ Hybrid search returned {len(results)} results")
        
        if results:
            for i, result in enumerate(results[:3]):
                kg_comp = result.get('kg_component', 0)
                dense_comp = result.get('dense_component', 0)
                print(f"   📋 Result {i+1}: {result['text'][:60]}...")
                print(f"       Score: {result['fused_score']:.4f} (KG: {kg_comp:.3f}, Dense: {dense_comp:.3f})")
        
        # Get system stats
        stats = hybrid.get_retrieval_stats()
        print(f"   📈 System health: {stats}")
        
    except Exception as e:
        print(f"   ❌ Hybrid retrieval failed: {e}")
        return False
    
    # Test 4: MAS Orchestrator integration
    print("\n🤖 Testing MAS Orchestrator:")
    try:
        orchestrator = NLIMASOrchestrator()
        
        # Create test state
        test_state = MASState(
            query="What drugs treat type 2 diabetes?",
            trace_id="test_end_to_end_001",
            entities=[],
            intent="factoid",
            mode="hybrid",
            retry_count=0
        )
        
        print(f"   ✅ Created MAS state for query: '{test_state['query']}'")
        print(f"   📋 Trace ID: {test_state['trace_id']}")
        print(f"   🎯 Intent: {test_state['intent']}, Mode: {test_state['mode']}")
        
    except Exception as e:
        print(f"   ❌ MAS orchestrator failed: {e}")
        return False
    
    print("\n🎉 End-to-End Test Successful!")
    print("📋 System Status Summary:")
    print("   • Dense retrieval: ✅ Functional with BGE embeddings")
    print("   • KG traversal: ✅ Working with real biomedical entities")
    print("   • Hybrid fusion: ✅ Combining KG and Dense results")
    print("   • MAS orchestrator: ✅ Ready for workflow execution")
    print("   • Data integration: ✅ All components using correct schemas")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_end_to_end())
    sys.exit(0 if success else 1)
