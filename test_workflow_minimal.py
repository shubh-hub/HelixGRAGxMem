#!/usr/bin/env python3
"""
Minimal workflow test to isolate the issue
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mas.state import create_initial_state, MASState
from langgraph.graph import StateGraph, END
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_node(state: MASState) -> MASState:
    """Simple test node"""
    logger.info("🔥 TEST NODE EXECUTED!")
    new_state = state.copy()
    new_state["final_answer"] = "Test answer from minimal workflow"
    new_state["confidence"] = 0.9
    return new_state

async def test_minimal_workflow():
    """Test minimal workflow execution"""
    print("🧪 Testing Minimal Workflow...")
    
    try:
        # Create workflow
        workflow = StateGraph(MASState)
        workflow.add_node("test", test_node)
        workflow.set_entry_point("test")
        workflow.add_edge("test", END)
        
        # Compile workflow
        compiled = workflow.compile()
        logger.info("Workflow compiled successfully")
        
        # Create initial state
        initial_state = create_initial_state(query="test query")
        logger.info(f"Initial state: {list(initial_state.keys())}")
        
        # Execute workflow
        logger.info("Executing workflow...")
        result = await compiled.ainvoke(initial_state)
        
        logger.info(f"Result: {result.get('final_answer', 'No answer')}")
        print(f"✅ Success: {result.get('final_answer', 'No answer')}")
        
    except Exception as e:
        logger.error(f"❌ Minimal test failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_minimal_workflow())
