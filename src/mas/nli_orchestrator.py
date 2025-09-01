"""
Natural Language Interface Multi-Agent System Orchestrator
Handles biomedical query processing through LangGraph workflow with MCP integration
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
import copy

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import MASState, create_initial_state
from ..config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_state_mutation(state: Dict[str, Any], node: str, action: str, details: Dict[str, Any]):
    """Log state mutations for debugging"""
    logger.info(f"[{node}] {action}: {details}")

def copy_state(state: MASState) -> MASState:
    """Create a deep copy of state"""
    return copy.deepcopy(state)

class NLIMASOrchestrator:
    """
    Natural Language Interface Multi-Agent System Orchestrator
    
    Orchestrates biomedical query processing through a LangGraph workflow
    with MCP tool integration for knowledge graph and dense retrieval.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = settings
        self.trace_id = str(uuid.uuid4())
        self.model = None
        self.tools = []
        self.compiled_workflow = None
        
        logger.info(f"🚀 Initializing NLI MAS Orchestrator with trace_id: {self.trace_id}")
    
    async def initialize(self):
        """Initialize the orchestrator with LLM and MCP tools"""
        try:
            # Initialize LLM
            groq_api_key = os.getenv("GROQ_API_KEY") or self.config.GROQ_API_KEY
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment or config")
            
            self.model = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=groq_api_key,
                temperature=0.1
            )
            logger.info("✅ LLM initialized successfully")
            
            # Initialize MCP tools
            await self._initialize_mcp_tools()
            
            # Create and compile LangGraph workflow
            await self._create_workflow()
            
            logger.info("🎯 Orchestrator initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize orchestrator: {e}")
            raise
    
    async def _initialize_mcp_tools(self):
        """Initialize MCP client with KG and Dense servers"""
        try:
            # Get project root path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            # MCP server configurations
            mcp_servers = {
                "kg_server": {
                    "command": "python3",
                    "args": [os.path.join(project_root, "scripts", "kg_server.py")],
                    "env": None
                },
                "dense_server": {
                    "command": "python3", 
                    "args": [os.path.join(project_root, "scripts", "dense_server.py")],
                    "env": None
                }
            }
            
            # Create MCP client
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            await self.mcp_client.connect()
            self.tools = await self.mcp_client.get_available_tools()
            
            logger.info(f"✅ MCP tools initialized: {[tool.name for tool in self.tools]}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP tools: {e}")
            # Continue without MCP tools for now
            self.tools = []
            self.mcp_client = None
    
    async def _create_workflow(self):
        """Create and compile the LangGraph workflow"""
        try:
            # Create StateGraph
            workflow = StateGraph(MASState)
            
            # Add nodes
            workflow.add_node("query_analysis", self._query_analysis_node)
            workflow.add_node("entity_extraction", self._entity_extraction_node)
            workflow.add_node("evidence_retrieval", self._evidence_retrieval_node)
            workflow.add_node("reasoning_generation", self._reasoning_generation_node)
            
            # Define edges
            workflow.set_entry_point("query_analysis")
            workflow.add_edge("query_analysis", "entity_extraction")
            workflow.add_edge("entity_extraction", "evidence_retrieval")
            workflow.add_edge("evidence_retrieval", "reasoning_generation")
            workflow.set_finish_point("reasoning_generation")
            
            # Compile workflow with checkpointer
            checkpointer = MemorySaver()
            self.compiled_workflow = workflow.compile(checkpointer=checkpointer)
            
            logger.info("✅ LangGraph workflow compiled successfully")
            logger.info(f"Workflow nodes: {list(self.compiled_workflow.get_graph().nodes.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create workflow: {e}")
            raise
    
    async def _query_analysis_node(self, state: MASState) -> dict:
        """
        Combined Steps 1-4: Query Understanding, Intent Detection, 
        Query Transformation, Resource Strategy Selection
        """
        logger.info("🧠 Starting query analysis node")
        
        query = state['query']
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze this biomedical query and provide a JSON response:

Query: "{query}"

Provide analysis in this exact JSON format:
{{
    "intent": "factoid|list|comparison|causal|procedural",
    "resource_strategy": {{
        "mode": "hybrid|kg_only|dense_only",
        "reasoning": "explanation for mode choice"
    }}
}}

Focus on biomedical accuracy and choose the best retrieval strategy."""
        
        try:
            response = await self.model.ainvoke([{"role": "user", "content": analysis_prompt}])
            analysis_result = json.loads(response.content)
            
            return {
                "intent": analysis_result.get("intent", "factoid"),
                "mode": analysis_result.get("resource_strategy", {}).get("mode", "hybrid")
            }
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse query analysis JSON, using defaults")
            return {
                "intent": "factoid",
                "mode": "hybrid"
            }
    
    async def _entity_extraction_node(self, state: MASState) -> dict:
        """
        Combined Steps 5-7: Entity Extraction, Entity Canonicalization, 
        Seed Strategy Selection
        """
        logger.info("🏷️ Starting entity extraction node")
        
        # Step 5: Entity Extraction using simplified approach
        extraction_result = await self._extract_entities_simple(state['query'])
        
        # Store extracted entities
        entities_to_validate = []
        if extraction_result and isinstance(extraction_result, list):
            entities_to_validate = [entity.get("text", "") for entity in extraction_result if entity.get("text")]
        
        # Return state updates
        return {
            "entities_surface": entities_to_validate,
            "entities_canonical": entities_to_validate,  # Simplified - no canonicalization
            "metadata": {
                "entity_extraction": extraction_result or []
            }
        }
    
    async def _extract_entities_simple(self, query: str) -> List[Dict[str, Any]]:
        """Simple entity extraction using basic patterns"""
        entities = []
        
        # Extract potential medical terms (capitalized words, drug names, etc.)
        import re
        
        # Common medical entity patterns
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b\w+ine\b',  # Drug suffixes like -ine
            r'\b\w+ol\b',   # Drug suffixes like -ol
            r'\bhypertension\b',  # Specific conditions
            r'\bdiabetes\b',
            r'\bdrug[s]?\b',
            r'\bmedication[s]?\b'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity_text = match.group().strip()
                if len(entity_text) > 2:  # Filter out very short matches
                    entities.append({
                        "text": entity_text,
                        "type": "medical_term",
                        "confidence": 0.8,
                        "start": match.start(),
                        "end": match.end()
                    })
        
        return entities
    
    async def _evidence_retrieval_node(self, state: MASState) -> dict:
        """
        Step 8: Evidence Retrieval from KG and Dense sources
        """
        logger.info("🔍 Starting evidence retrieval node")
        
        # Initialize evidence containers
        kg_evidence = []
        dense_evidence = []
        
        try:
            # Use MCP client to call tools if available
            if self.mcp_client:
                # Find and invoke KG search tool if in KG or hybrid mode
                if state["mode"] in ["kg_only", "hybrid"]:
                    kg_result = await self.mcp_client.call_tool("search_kg", {
                        "query": state['query'],
                        "intent": state.get('intent', 'factoid'),
                        "max_results": 5
                    })
                    
                    # Parse KG result - it's a JSON string
                    if isinstance(kg_result, str):
                        kg_data = json.loads(kg_result)
                        kg_evidence = kg_data.get("results", [])
                    else:
                        kg_evidence = kg_result.get("results", [])
                
                # Find and invoke Dense search tool if in dense or hybrid mode
                if state["mode"] in ["dense_only", "hybrid"]:
                    dense_result = await self.mcp_client.call_tool("search_passages", {
                        "query": state['query'],
                        "max_results": 5
                    })
                    
                    # Parse Dense result - it's a JSON string
                    if isinstance(dense_result, str):
                        dense_data = json.loads(dense_result)
                        dense_evidence = dense_data.get("results", [])
                    else:
                        dense_evidence = dense_result.get("results", [])
                        
        except Exception as e:
            logger.warning(f"Evidence retrieval failed: {e}")
            # Continue with empty evidence
        
        # Debug logging
        logger.info(f"Evidence retrieved - KG: {len(kg_evidence)}, Dense: {len(dense_evidence)}")
        if dense_evidence:
            logger.info(f"Sample dense evidence: {dense_evidence[0] if dense_evidence else 'None'}")
        
        # Return state updates
        return {
            "evidence": {
                "kg_paths": kg_evidence,
                "dense_hits": dense_evidence
            }
        }
    
    async def _reasoning_generation_node(self, state: MASState) -> dict:
        """
        Combined Steps 9-12: Evidence Aggregation, LLM Reasoning, 
        Answer Generation, and Logging
        """
        logger.info("🤔 Starting reasoning generation node")
        
        # Step 9: Evidence Aggregation
        evidence = state.get("evidence", {})
        kg_evidence = evidence.get("kg_paths", [])
        dense_evidence = evidence.get("dense_hits", [])
        
        # Debug logging for reasoning node
        logger.info(f"Reasoning node - KG evidence: {len(kg_evidence)}, Dense evidence: {len(dense_evidence)}")
        
        # Step 10-11: LLM Reasoning and Answer Generation
        # Format evidence for LLM
        kg_text = ""
        if kg_evidence:
            kg_text = "\n".join([f"- {item.get('text', str(item))}" for item in kg_evidence[:5]])
        
        dense_text = ""
        if dense_evidence:
            dense_text = "\n".join([f"- {item.get('text', str(item))}" for item in dense_evidence[:5]])
            logger.info(f"Dense text for LLM: {dense_text[:200]}...")
        
        reasoning_prompt = f"""You are a biomedical expert. Answer this query using the provided evidence.

Query: {state['query']}

Knowledge Graph Evidence:
{kg_text if kg_text else "No KG evidence available"}

Dense Retrieval Evidence:
{dense_text if dense_text else "No dense evidence available"}

Provide a comprehensive, accurate answer based on the evidence. If evidence is limited, acknowledge this limitation."""
        
        try:
            response = await self.model.ainvoke([{"role": "user", "content": reasoning_prompt}])
            final_answer = response.content
            
            logger.info(f"Generated answer: {final_answer[:100]}...")
            
            return {
                "final_answer": final_answer,
                "answer": final_answer  # Backup field
            }
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return {
                "final_answer": "I apologize, but I encountered an error while processing your query.",
                "answer": "Error in reasoning generation"
            }
    
    async def process_query(self, query: str, trace_id: str = None, config: Optional[RunnableConfig] = None):
        """
        Process a biomedical query through the LangGraph workflow
        
        Args:
            query: The biomedical query to process
            trace_id: Optional trace ID for tracking
            config: Optional LangGraph configuration
            
        Returns:
            Dict containing success status, answer, and metadata
        """
        if trace_id:
            self.trace_id = trace_id
        
        logger.info(f"🔄 Processing query: {query}")
        
        if not self.compiled_workflow:
            logger.error("Workflow not compiled!")
            return {"success": False, "error": "Workflow not compiled", "trace_id": self.trace_id}
        
        # Create initial state
        initial_state = create_initial_state(query=query)
        initial_state["trace_id"] = self.trace_id
        logger.info(f"Initial state created: {list(initial_state.keys())}")
        
        try:
            # Execute workflow using proper LangGraph compilation
            logger.info(f"Starting LangGraph workflow execution for query: {query}")
            
            # Execute with proper config for checkpointer
            result = await self.compiled_workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": self.trace_id}}
            )
            
            logger.info(f"Workflow completed. Result keys: {list(result.keys())}")
            logger.info(f"Evidence in result: {result.get('evidence', {})}")
            logger.info(f"Final answer: {result.get('final_answer', 'None')}")
            logger.info(f"Answer field: {result.get('answer', 'None')}")
            logger.info(f"Intent: {result.get('intent', 'None')}")
            logger.info(f"Mode: {result.get('mode', 'None')}")
            
            return {
                "success": True,
                "answer": result.get("final_answer") or result.get("answer", "No answer generated"),
                "evidence": result.get("evidence", {}),
                "intent": result.get("intent", "unknown"),
                "mode": result.get("mode", "unknown"),
                "entities": result.get("entities_surface", []),
                "trace_id": self.trace_id,
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "trace_id": self.trace_id
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close MCP client connections if any
            if hasattr(self, 'mcp_client') and self.mcp_client:
                await self.mcp_client.disconnect()
            logger.info("🧹 Orchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
