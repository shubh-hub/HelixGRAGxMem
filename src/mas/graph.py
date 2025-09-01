"""
LangGraph MAS Orchestration
===========================

Implements the Multi-Agent System using LangGraph with conditional routing
and typed state management for biomedical hybrid retrieval.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from .state import MASState, copy_state
# Import intelligent LLM-driven nodes
from .nodes.intelligent_planner import intelligent_planner_node
from .nodes.intelligent_retriever import intelligent_retriever_node
from .nodes.intelligent_generator import intelligent_generator_node
from .nodes.intelligent_reviewer import intelligent_reviewer_node
from .nodes.router import router_node
from .nodes.fuse import fuse_node
from .nodes.explain import explain_node
from .nodes.memory_store import memory_store_node
from ..mcp.client_stdio import StdioMCPClient

logger = logging.getLogger(__name__)

def create_mas_graph() -> StateGraph:
    """Create the Multi-Agent System graph using LangGraph"""
    
    # Create the StateGraph with our MASState schema
    workflow = StateGraph(MASState)
    
    # Add intelligent LLM-driven nodes
    workflow.add_node("planner", intelligent_planner_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", intelligent_retriever_node)
    workflow.add_node("generate", intelligent_generator_node)  # New generation step
    workflow.add_node("review", intelligent_reviewer_node)     # New review step
    workflow.add_node("fuse", fuse_node)
    workflow.add_node("explain", explain_node)
    workflow.add_node("memory_store", memory_store_node)
    
    # Add edges for the intelligent MAS flow
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "fuse")
    workflow.add_edge("fuse", "generate")     # Generate answer from evidence
    workflow.add_edge("generate", "review")   # Review generated answer
    
    # Add conditional edges from review node
    workflow.add_conditional_edges(
        "review",
        route_after_review,
        {
            "approve": "explain",      # Answer approved, proceed to explanation
            "revise": "generate",      # Need revision, go back to generation
            "expand": "retrieve",      # Need more evidence, go back to retrieval
            "replan": "planner"        # Need replanning, start over
        }
    )
    
    workflow.add_edge("explain", "memory_store")
    workflow.add_edge("memory_store", END)
    
    return workflow

def route_after_review(state: MASState) -> Literal["approve", "revise", "expand", "replan"]:
    """Route based on intelligent review results"""
    
    # Get review results from intelligent reviewer
    recommended_action = state.get("recommended_action", "approve")
    next_action = state.get("next_action", "explain")
    quality_score = state.get("quality_score", 0.7)
    
    # Map review actions to routing decisions
    if recommended_action == "approve" or next_action == "explain":
        return "approve"
    elif recommended_action == "request_revision" and quality_score > 0.4:
        return "revise"  # Try to improve the answer
    elif recommended_action == "request_revision" and quality_score <= 0.4:
        return "expand"  # Need more evidence
    elif recommended_action == "reject":
        return "replan"  # Start over with new strategy
    
    # Default to approve if unclear
    return "approve"

class MASOrchestrator:
    """Multi-Agent System orchestrator using LangGraph"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Create the graph
        self.workflow = create_mas_graph()
        
        # Add memory for checkpointing
        self.memory = MemorySaver()
        
        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def execute(self, query: str, persona: str = "doctor", **kwargs) -> Dict[str, Any]:
        """Execute the MAS workflow using LangGraph"""
        
        trace_id = self._generate_trace_id()
        start_time = datetime.utcnow()
        
        try:
            # Create MCP client with trace_id
            async with MCPClient(trace_id) as mcp_client:
                
                # Create initial state
                initial_state = MASState(
                    trace_id=trace_id,
                    query=query,
                    persona=persona,
                    mcp_client=mcp_client,
                    timestamp=start_time.isoformat(),
                    status="initialized",
                    **kwargs
                )
                
                # Execute the workflow
                config = RunnableConfig(
                    configurable={"thread_id": trace_id},
                    metadata={"persona": persona, "start_time": start_time.isoformat()}
                )
                
                final_state = await self.app.ainvoke(initial_state, config=config)
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Return final result
                return {
                    "trace_id": trace_id,
                    "query": query,
                    "persona": persona,
                    "final_answer": final_state.get("final_answer", ""),
                    "explanation": final_state.get("explanation", ""),
                    "confidence_score": final_state.get("confidence_score", 0.0),
                    "mode": final_state.get("mode", "unknown"),
                    "kg_paths": len(final_state.get("kg_paths", [])),
                    "dense_hits": len(final_state.get("dense_hits", [])),
                    "citations": final_state.get("citations", []),
                    "safety_verified": final_state.get("safety_verified", False),
                    "status": final_state.get("status", "completed"),
                    "execution_time": execution_time,
                    "error": None
                }
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"MAS execution failed for trace {trace_id}: {e}")
            
            return {
                "trace_id": trace_id,
                "query": query,
                "persona": persona,
                "final_answer": "Error occurred during processing",
                "error": str(e),
                "status": "failed",
                "execution_time": execution_time
            }
    
    async def get_execution_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get execution trace for debugging and analysis"""
        try:
            # Get checkpoint data
            config = RunnableConfig(configurable={"thread_id": trace_id})
            checkpoint = await self.app.aget_state(config)
            
            if checkpoint:
                return {
                    "trace_id": trace_id,
                    "state": checkpoint.values,
                    "next_nodes": checkpoint.next,
                    "metadata": checkpoint.metadata
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving trace {trace_id}: {e}")
            return None

# Factory function for easy instantiation
def create_mas_orchestrator(config: Optional[Dict[str, Any]] = None) -> MASOrchestrator:
    """Create and configure MAS orchestrator"""
    return MASOrchestrator()

# Async context manager for resource cleanup
class MASSession:
    """Context manager for MAS execution sessions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.orchestrator = None
    
    async def __aenter__(self):
        self.orchestrator = create_mas_orchestrator(self.config)
        return self.orchestrator
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources if needed
        if exc_type:
            logger.error(f"MAS session ended with error: {exc_val}")
        else:
            logger.info("MAS session completed successfully")
