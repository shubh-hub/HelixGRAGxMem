#!/usr/bin/env python3
"""
Test script to verify recovered core components
"""

import os
import sys
sys.path.append('/Users/shivam/Documents/Shubham/HelixGRAGxMem')

from src.core import DenseRetriever, EdgePredictor, RelationPruner, KGWalker, HybridRetriever

def test_core_components():
    """Test that all core components can be imported and initialized"""
    
    print("🔧 Testing Core Component Recovery...")
    
    # Test 1: Import all components
    try:
        print("✅ All core components imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Initialize components (without requiring data files)
    try:
        dense = DenseRetriever()
        edge_pred = EdgePredictor()
        pruner = RelationPruner()
        walker = KGWalker()
        hybrid = HybridRetriever()
        print("✅ All components can be instantiated")
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False
    
    # Test 3: Check component methods exist
    try:
        # DenseRetriever methods
        assert hasattr(dense, 'search')
        assert hasattr(dense, 'initialize')
        assert hasattr(dense, 'get_embedding')
        
        # EdgePredictor methods
        assert hasattr(edge_pred, 'predict_next_relations')
        assert hasattr(edge_pred, 'connect_kg')
        
        # RelationPruner methods
        assert hasattr(pruner, 'prune_relations')
        assert hasattr(pruner, 'get_high_value_relations')
        
        # KGWalker methods
        assert hasattr(walker, 'walk')
        assert hasattr(walker, 'find_shortest_paths')
        assert hasattr(walker, 'get_entity_neighborhood')
        
        # HybridRetriever methods
        assert hasattr(hybrid, 'search')
        assert hasattr(hybrid, 'search_kg_only')
        assert hasattr(hybrid, 'search_dense_only')
        
        print("✅ All required methods are present")
    except AssertionError as e:
        print(f"❌ Missing methods: {e}")
        return False
    
    print("\n🎉 Core component recovery successful!")
    print("📋 Recovered components:")
    print("   • DenseRetriever - BGE embeddings + FAISS search")
    print("   • EdgePredictor - LLM-based relation prediction")
    print("   • RelationPruner - Multi-signal relation filtering")
    print("   • KGWalker - Entropy-based graph traversal")
    print("   • HybridRetriever - Orchestration with fusion scoring")
    
    return True

if __name__ == "__main__":
    success = test_core_components()
    sys.exit(0 if success else 1)
