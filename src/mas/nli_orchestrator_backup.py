#!/usr/bin/env python3
"""
NLI-Aligned MAS Orchestrator
============================

Implements the 12-step NLI workflow as specified by the user:
1. Query Understanding
2. Intent Detection  
3. Query Transformation (subquestions, simplification, rewriting)
4. Evidence Resource Identification
5. Entity/Phrase Extraction
6. Canonical Entity Resolution
7. Seed Node & Constraint Identification
8. Retrieval Engine (confidence-guided KG + dense search)
9. Evidence Aggregation & Pruning
10. LLM-driven Reasoning
11. Answer Generation, Review & Explanation
12. Summarized Logging

All intelligence is LLM-driven with MCP integration via langchain-mcp-adapters.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_mcp import MCPToolkit
from langchain_mcp.adapters import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from .state import MASState, create_initial_state, copy_state, log_state_mutation
from ..config import GROQ_API_KEY
from .nodes.planner import _extract_entities_llm

logger = logging.getLogger(__name__)

class NLIMASOrchestrator:
    """
    NLI-aligned Multi-Agent System orchestrator implementing the 12-step workflow
    with LLM-driven intelligence and MCP tool integration.
    """
    
    def __init__(self):
        """Initialize the NLI MAS Orchestrator"""
        logger.info("🚀 Initializing NLI MAS Orchestrator")
        self.trace_id = None
        self.mcp_client = None
        self.tools = []
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        self.model = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.1
        )
        
        # Create workflow and memory
        logger.info("Creating workflow...")
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        self.compiled_workflow = None  # Will be compiled after MCP initialization
        logger.info("✅ NLI MAS Orchestrator initialized")
    
    async def initialize_mcp_client(self):
        """Initialize MCP client and tools"""
        if not self.mcp_client:
            # Get absolute paths to MCP servers
            project_root = Path(__file__).parent.parent.parent
            kg_server_path = project_root / "src" / "mcp" / "servers" / "kg_server.py"
            dense_server_path = project_root / "src" / "mcp" / "servers" / "dense_server.py"
            
            self.mcp_client = MultiServerMCPClient({
                "kg": {
                    "command": "python3",
                    "args": [str(kg_server_path)],
                    "transport": "stdio",
                },
                "dense": {
                    "command": "python3", 
                    "args": [str(dense_server_path)],
                    "transport": "stdio",
                }
            })
            
            # Get tools from MCP servers
            self.tools = await self.mcp_client.get_tools()
            
            # Initialize model with tools and API key
            self.model = init_chat_model(
                model=settings.LLM_MODEL,
                model_provider=settings.LLM_PROVIDER,
                api_key=settings.GROQ_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            # Don't bind tools to avoid unwanted tool calls - we handle tools manually
            
            # Compile workflow after MCP initialization
            if not self.compiled_workflow:
                self.compiled_workflow = self.workflow.compile(checkpointer=self.memory)
            
            logger.info(f"✅ MCP client initialized with {len(self.tools)} tools")
    
    def _create_workflow(self):
        """Create the optimized 4-node LangGraph workflow"""
        workflow = StateGraph(MASState)
        
        # Add nodes with debug logging
        logger.info("Creating workflow with nodes...")
        workflow.add_node("query_analysis", self._query_analysis_node)
        workflow.add_node("entity_extraction", self._entity_extraction_node)
        workflow.add_node("evidence_retrieval", self._evidence_retrieval_node)
        workflow.add_node("reasoning_generation", self._reasoning_generation_node)
        
        # Set entry point
        workflow.set_entry_point("query_analysis")
        
        # Add edges - simplified linear flow
        workflow.add_edge("query_analysis", "entity_extraction")
        workflow.add_edge("entity_extraction", "evidence_retrieval")
        workflow.add_edge("evidence_retrieval", "reasoning_generation")
        workflow.add_edge("reasoning_generation", END)
        
        logger.info("Workflow graph created successfully")
        return workflow
    
    async def _query_analysis_node(self, state: MASState) -> dict:
        """
        Combined Steps 1-4: Query Understanding, Intent Detection, 
        Query Transformation, Resource Identification
        """
        logger.info("🔍 Starting query analysis node")
        await self.initialize_mcp_client()
        
        analysis_prompt = f"""
        Analyze this biomedical query comprehensively:
        Query: {state['query']}
        
        Provide analysis in JSON format:
        {{
            "understanding": "detailed query understanding",
            "intent": "factoid|enumeration|comparison|explanation",
            "complexity": "simple|moderate|complex",
            "domain_focus": "specific biomedical domain",
            "transformations": {{
                "subquestions": ["list of subquestions"],
                "simplified": "simplified version",
                "expanded": "expanded version with synonyms"
            }},
            "resource_strategy": {{
                "mode": "kg_only|dense_only|hybrid",
                "reasoning": "why this strategy",
                "priority": "kg|dense"
            }}
        }}
        """
        
        try:
            response = await self.model.ainvoke([{"role": "user", "content": analysis_prompt}])
            analysis_result = json.loads(response.content)
            
            # Return state updates
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
        # Find and invoke KG search tool if in KG or hybrid mode
        if state["mode"] in ["kg_only", "hybrid"]:
            kg_tool = next((tool for tool in self.tools if tool.name == "search_kg"), None)
            if kg_tool:
                kg_result = await kg_tool.ainvoke({
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
            dense_tool = next((tool for tool in self.tools if tool.name == "search_passages"), None)
            if dense_tool:
                dense_result = await dense_tool.ainvoke({
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
    
async def _reasoning_generation_node(self, state: MASState) -> MASState:
    """
    Combined Steps 9-12: Evidence Aggregation, LLM Reasoning, 
    Answer Generation, and Logging
    """
    new_state = copy_state(state)
    
    # Step 9: Evidence Aggregation
    evidence = new_state.get("evidence", {})
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
        new_state = copy_state(state)
        
        # Step 9: Evidence Aggregation
        evidence = new_state.get("evidence", {})
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
Intent: {state['intent']}

Knowledge Graph Evidence:
{kg_text if kg_text else "No KG evidence available"}

Dense Search Evidence:
{dense_text if dense_text else "No dense evidence available"}

Based on the above evidence, provide a direct answer to the biomedical query. Use the specific information from the evidence to support your answer.

Return your response in this exact JSON format:
{{
    "answer": "Your direct answer here",
    "explanation": "Brief explanation of your reasoning",
    "confidence": 0.8
}}"""
        
        response = await self.model.ainvoke([{"role": "user", "content": reasoning_prompt}])
        
        try:
            # Try to extract JSON from response
            response_text = response.content.strip()
            
            # Look for JSON block in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                reasoning_result = json.loads(json_text)
                
                # Update state with final results
                new_state["answer"] = reasoning_result.get("answer", "Unable to provide answer")
                new_state["explanation"] = reasoning_result.get("explanation", "")
                new_state["confidence"] = reasoning_result.get("confidence", 0.5)
                new_state["metadata"]["reasoning"] = reasoning_result
            else:
                # Fallback: use raw response as answer
                new_state["answer"] = response_text[:500] if response_text else "No response generated"
                new_state["explanation"] = "Direct LLM response (JSON parsing failed)"
                new_state["confidence"] = 0.6
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse reasoning JSON: {e}")
            # Fallback: use raw response as answer
            response_text = getattr(response, 'content', str(response))
            new_state["answer"] = response_text[:500] if response_text else "Error in reasoning process"
            new_state["explanation"] = "Fallback response due to parsing error"
            new_state["confidence"] = 0.3
        
        # Step 12: Finalize and log
        new_state["status"] = "completed"
        new_state["metadata"]["completion_time"] = datetime.utcnow().isoformat()
        
        log_state_mutation(new_state, "reasoning_generation", "workflow_complete", {
            "answer_length": len(new_state["answer"]),
            "confidence": new_state["confidence"],
            "evidence_sources": len(kg_evidence) + len(dense_evidence)
        })
        
        return new_state
    
    
    async def process_query(self, query: str, trace_id: str = None, config: Optional[RunnableConfig] = None):
        """
        Process a query through the complete NLI workflow
        
        Args:
            query: User query
            trace_id: Optional trace ID for logging
            config: Optional LangGraph configuration
            
        Returns:
            Processing results
        """
        self.trace_id = trace_id or f"nli_{int(time.time())}"
        
        # Ensure MCP client and workflow are initialized
        await self.initialize_mcp_client()
        
        # Check if workflow is compiled
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
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", 0.0),
                "trace_id": self.trace_id,
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "trace_id": self.trace_id
            }
    
    async def get_workflow_state(self, trace_id: str) -> Optional[MASState]:
        """Get current workflow state for a trace ID"""
        try:
            config = {"configurable": {"thread_id": trace_id}}
            state = await self.compiled_workflow.aget_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Failed to get workflow state: {e}")
            return None
