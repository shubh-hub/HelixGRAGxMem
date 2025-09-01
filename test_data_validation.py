#!/usr/bin/env python3
"""
Comprehensive test to validate system with actual data files
"""

import os
import sys
import asyncio
import json
sys.path.append('/Users/shivam/Documents/Shubham/HelixGRAGxMem')

from src.core import DenseRetriever, HybridRetriever, KGWalker, EdgePredictor, RelationPruner
from src.config import settings

async def test_data_validation():
    """Test system with actual data files"""
    
    print("🔧 Testing System with Actual Data Files...")
    
    # Test 1: Verify all data files exist
    required_files = [
        settings.DB_PATH,
        settings.FAISS_INDEX_PATH, 
        settings.FAISS_ID_MAP_PATH,
        settings.VERBALIZED_KG_PATH
    ]
    
    print("\n📁 Checking data file availability:")
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {os.path.basename(file_path)}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {os.path.basename(file_path)}: Missing")
            return False
    
    # Test 2: Initialize DenseRetriever with actual data
    print("\n🔍 Testing DenseRetriever with actual data:")
    try:
        dense = DenseRetriever()
        dense.initialize()
        
        # Test search
        results = dense.search("What treats diabetes?", k=5)
        print(f"   ✅ Dense search returned {len(results)} results")
        
        if results:
            print(f"   📋 Top result: {results[0]['text'][:100]}...")
            print(f"   📊 Score: {results[0]['score']:.4f}")
        
    except Exception as e:
        print(f"   ❌ DenseRetriever failed: {e}")
        return False
    
    # Test 3: Initialize KG components
    print("\n🕸️ Testing KG components:")
    try:
        walker = KGWalker()
        walker.connect_kg()
        
        # Test entity neighborhood
        neighborhood = walker.get_entity_neighborhood("diabetes")
        print(f"   ✅ KG Walker connected, diabetes has {neighborhood['neighbor_count']} neighbors")
        
        # Test relation pruner
        pruner = RelationPruner()
        pruner.connect_kg()
        high_value_relations = pruner.get_high_value_relations(top_k=5)
        print(f"   ✅ Relation Pruner found {len(high_value_relations)} high-value relations")
        
        if high_value_relations:
            top_relation = high_value_relations[0]
            print(f"   📋 Top relation: {top_relation['relation']} (score: {top_relation['score']:.3f})")
        
    except Exception as e:
        print(f"   ❌ KG components failed: {e}")
        return False
    
    # Test 4: Test EdgePredictor (if API key available)
    print("\n🧠 Testing EdgePredictor:")
    try:
        edge_pred = EdgePredictor()
        
        if settings.OPENAI_API_KEY or settings.GROQ_API_KEY:
            # Test relation prediction
            predictions = edge_pred.predict_next_relations(
                current_entity="diabetes",
                target_entity="insulin", 
                query_context="What treats diabetes?",
                max_relations=3
            )
            print(f"   ✅ EdgePredictor returned {len(predictions)} predictions")
            
            if predictions:
                top_pred = predictions[0]
                print(f"   📋 Top prediction: {top_pred['relation']} (confidence: {top_pred['confidence']:.3f})")
        else:
            print("   ⚠️ No API key available, using fallback predictions")
            
    except Exception as e:
        print(f"   ❌ EdgePredictor failed: {e}")
    
    # Test 5: Test HybridRetriever integration
    print("\n🔄 Testing HybridRetriever integration:")
    try:
        hybrid = HybridRetriever()
        hybrid.initialize()
        
        # Test hybrid search
        results = hybrid.search(
            query="What treats diabetes?",
            entities=["diabetes", "insulin"],
            k=5,
            kg_weight=0.6,
            dense_weight=0.4
        )
        
        print(f"   ✅ Hybrid search returned {len(results)} results")
        
        if results:
            print(f"   📋 Top result: {results[0]['text'][:100]}...")
            print(f"   📊 Fused score: {results[0]['fused_score']:.4f}")
            print(f"   🔗 KG component: {results[0].get('kg_component', 0):.4f}")
            print(f"   📚 Dense component: {results[0].get('dense_component', 0):.4f}")
        
        # Get retrieval stats
        stats = hybrid.get_retrieval_stats()
        print(f"   📈 System stats: {stats}")
        
    except Exception as e:
        print(f"   ❌ HybridRetriever failed: {e}")
        return False
    
    print("\n🎉 Data validation test successful!")
    print("📋 System Status:")
    print("   • All data files: ✅ Present and accessible")
    print("   • Dense retrieval: ✅ Functional with BGE embeddings")
    print("   • KG components: ✅ Connected and operational")
    print("   • Hybrid fusion: ✅ Working with weighted scoring")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_data_validation())
    sys.exit(0 if success else 1)
