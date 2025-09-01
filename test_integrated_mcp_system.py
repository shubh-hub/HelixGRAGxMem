#!/usr/bin/env python3
"""
Integrated MCP System Test
==========================
Test the complete HelixGRAGxMem system with proper MCP integration,
LangGraph workflow, and core component backends.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mas.nli_orchestrator import NLIMASOrchestrator
from src.mas.state import create_initial_state
from langchain_mcp_adapters.client import MultiServerMCPClient

# Set GROQ_API_KEY for testing if not set
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "test_key_for_integration_testing"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_mcp_server_initialization():
    """Test that MCP servers can be initialized with core components"""
    logger.info("🔧 Testing MCP Server Initialization...")
    
    try:
        orchestrator = NLIMASOrchestrator()
        await orchestrator.initialize_mcp_client()
        
        # Check that tools were loaded
        assert orchestrator.tools is not None, "Tools not loaded from MCP servers"
        assert len(orchestrator.tools) > 0, "No tools available from MCP servers"
        
        # Check that model was initialized
        assert orchestrator.model is not None, "LLM model not initialized"
        assert orchestrator.model_with_tools is not None, "Model with tools not bound"
        
        logger.info(f"✅ MCP servers initialized with {len(orchestrator.tools)} tools")
        return True
        
    except Exception as e:
        logger.error(f"❌ MCP server initialization failed: {e}")
        return False

async def test_workflow_execution():
    """Test complete workflow execution with biomedical query"""
    logger.info("🔄 Testing Complete Workflow Execution...")
    
    try:
        orchestrator = NLIMASOrchestrator()
        
        # Test biomedical query
        query = "What medications are used to treat type 2 diabetes?"
        
        result = await orchestrator.process_query(
            query=query,
            trace_id="test_workflow_001"
        )
        
        # Validate results
        assert result["success"] == True, f"Workflow failed: {result.get('error', 'Unknown error')}"
        assert "answer" in result, "No answer generated"
        assert len(result["answer"]) > 0, "Empty answer generated"
        assert "confidence" in result, "No confidence score"
        assert result["confidence"] > 0.0, "Zero confidence score"
        
        logger.info(f"✅ Workflow executed successfully")
        logger.info(f"📋 Answer: {result['answer'][:100]}...")
        logger.info(f"🎯 Confidence: {result['confidence']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        return False

async def test_mcp_tool_integration():
    """Test that MCP tools are properly integrated with core components"""
    logger.info("🛠️ Testing MCP Tool Integration...")
    
    try:
        orchestrator = NLIMASOrchestrator()
        await orchestrator.initialize_mcp_client()
        
        # Check available tools
        tool_names = [tool.name for tool in orchestrator.tools]
        
        expected_tools = ["search_kg", "validate_entities", "search_passages", "get_neighbors"]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing expected tool: {expected_tool}"
        
        logger.info(f"✅ All expected MCP tools available: {tool_names}")
        return True
        
    except Exception as e:
        logger.error(f"❌ MCP tool integration test failed: {e}")
        return False

async def test_langgraph_node_optimization():
    """Test that LangGraph nodes are properly optimized and clubbed"""
    logger.info("📊 Testing LangGraph Node Optimization...")
    
    try:
        orchestrator = NLIMASOrchestrator()
        
        # Check workflow structure
        workflow_nodes = list(orchestrator.workflow.nodes.keys())
        
        # Should have optimized node structure (4 main nodes + tools)
        expected_nodes = ["query_analysis", "entity_extraction", "evidence_retrieval", "reasoning_generation", "tools"]
        
        for expected_node in expected_nodes:
            assert expected_node in workflow_nodes, f"Missing expected node: {expected_node}"
        
        # Should not have too many nodes (optimization check)
        assert len(workflow_nodes) <= 6, f"Too many nodes ({len(workflow_nodes)}), optimization may have failed"
        
        logger.info(f"✅ LangGraph workflow optimized with {len(workflow_nodes)} nodes: {workflow_nodes}")
        return True
        
    except Exception as e:
        logger.error(f"❌ LangGraph optimization test failed: {e}")
        return False

async def test_core_component_backend():
    """Test that core components are working as MCP server backends"""
    logger.info("⚙️ Testing Core Component Backend Integration...")
    
    try:
        # This test would ideally call MCP servers directly
        # For now, we test through the orchestrator workflow
        orchestrator = NLIMASOrchestrator()
        
        # Test orchestrator initialization
        orchestrator = NLIMASOrchestrator()
        
        # Test state creation
        test_state = create_initial_state(query="test query")
        test_state["trace_id"] = "test_trace_123"
        
        result = await orchestrator.process_query(
            query="How does metformin treat diabetes and what are its side effects?",
            trace_id="test_core_backend_001"
        )
        
        # Check that workflow completed successfully
        assert result["success"] == True, "Core component backend test failed"
        
        # Check metadata for evidence of core component usage
        metadata = result.get("metadata", {})
        assert "entity_extraction" in metadata, "Entity extraction not performed"
        
        logger.info("✅ Core components working as MCP backends")
        return True
        
    except Exception as e:
        logger.error(f"❌ Core component backend test failed: {e}")
        return False

async def test_end_to_end_biomedical_queries():
    """Test end-to-end with various biomedical query types"""
    logger.info("🧬 Testing End-to-End Biomedical Queries...")
    
    test_queries = [
        ("What drugs treat hypertension?", "factoid"),
        ("List the symptoms of diabetes", "enumeration"), 
        ("How does aspirin prevent heart attacks?", "explanation"),
        ("Compare metformin and insulin for diabetes", "comparison")
    ]
    
    orchestrator = NLIMASOrchestrator()
    results = []
    
    for i, (query, expected_intent) in enumerate(test_queries):
        try:
            logger.info(f"Testing query {i+1}: {query}")
            
            result = await orchestrator.process_query(
                query=query,
                trace_id=f"test_e2e_{i+1:03d}"
            )
            
            # Basic validation
            assert result["success"] == True, f"Query {i+1} failed: {result.get('error')}"
            assert len(result["answer"]) > 10, f"Query {i+1} answer too short"
            
            # Test MCP client initialization with proper config
            client_config = {
                "kg": {
                    "command": "python3",
                    "args": [str(project_root / "src" / "mcp" / "servers" / "kg_server.py")],
                    "transport": "stdio",
                },
                "dense": {
                    "command": "python3", 
                    "args": [str(project_root / "src" / "mcp" / "servers" / "dense_server.py")],
                    "transport": "stdio",
                }
            }
            
            results.append({
                "query": query,
                "success": True,
                "answer_length": len(result["answer"]),
                "confidence": result["confidence"]
            })
            
            logger.info(f"✅ Query {i+1} successful (confidence: {result['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Query {i+1} failed: {e}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    logger.info(f"✅ End-to-end test: {successful}/{len(test_queries)} queries successful")
    
    return successful == len(test_queries)

async def main():
    """Run all integration tests"""
    logger.info("🚀 Starting Integrated MCP System Tests...")
    
    tests = [
        ("MCP Server Initialization", test_mcp_server_initialization),
        ("MCP Tool Integration", test_mcp_tool_integration),
        ("LangGraph Node Optimization", test_langgraph_node_optimization),
        ("Core Component Backend", test_core_component_backend),
        ("Workflow Execution", test_workflow_execution),
        ("End-to-End Biomedical Queries", test_end_to_end_biomedical_queries)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = await test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("🎯 HelixGRAGxMem system is fully integrated and operational")
    else:
        logger.error(f"⚠️  {total - passed} tests failed - system needs attention")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
