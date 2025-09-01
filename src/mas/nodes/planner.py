"""
Intelligent Planner Node - LLM-Driven Query Analysis
==================================================

LangGraph node function using LLM for intelligent intent detection, 
entity extraction, query analysis, and strategic planning.
"""

import logging
import json
import re
from typing import Dict, Any, List

from ..state import MASState, Intent, Mode
from ...config import settings

logger = logging.getLogger(__name__)

# LLM Planning Prompts
PLANNING_SYSTEM_PROMPT = """
You are an intelligent biomedical query planning agent. Your role is to analyze user queries and create strategic retrieval plans.

Your tasks:
1. Detect query intent (factoid, enumeration, causal, comparative, procedural)
2. Extract and canonicalize biomedical entities (drugs, diseases, symptoms, anatomy)
3. Generate strategic sub-questions for comprehensive coverage
4. Determine optimal retrieval policy (KG vs Dense vs Hybrid)
5. Assess query complexity and difficulty

Respond with a structured JSON containing your analysis and plan.
"""

PLANNING_USER_PROMPT = """
Analyze this biomedical query and create a strategic retrieval plan:

Query: "{query}"
Persona: {persona}

Provide a JSON response with:
{{
  "intent": "factoid|enumeration|causal|comparative|procedural",
  "entities_surface": ["raw entities found"],
  "entities_canonical": ["standardized biomedical terms"],
  "subquestions": ["strategic sub-questions for comprehensive retrieval"],
  "complexity": "simple|moderate|complex|expert",
  "recommended_mode": "dense|kg|hybrid",
  "policy": {{
    "max_hops": 2,
    "max_neighbors": 10,
    "time_budget": 30,
    "fusion_strategy": "weighted|ranked|consensus"
  }},
  "reasoning": "explanation of your planning decisions"
}}
"""

async def planner_node(state: MASState) -> MASState:
    """Intelligent planner node using LLM for query analysis"""
    
    # Get MCP client from state
    mcp_client = state.get("mcp_client")
    if not mcp_client:
        raise ValueError("MCP client not found in state")
    
    trace_id = state["trace_id"]
    query = state["query"]
    persona = state["persona"]
    
    logger.info(f"[{trace_id[:8]}] Executing intelligent planner node")
    
    # Log planning start
    try:
        await mcp_client.call(
            server="trace",
            method="log_event",
            payload={
                "trace_id": trace_id,
                "event_type": "planner_start",
                "data": {"query": query, "persona": persona},
                "span": "planning"
            },
            span="planner_start",
            node="planner"
        )
    except Exception as e:
        logger.warning(f"Failed to log to trace server: {e}")
    
    # Use LLM for intelligent planning
    try:
        planning_result = await _llm_plan_query(query, persona, mcp_client, trace_id)
        
        # Extract planning results
        intent_str = planning_result.get("intent", "factoid")
        entities_surface = planning_result.get("entities_surface", [])
        entities_canonical = planning_result.get("entities_canonical", [])
        subquestions = planning_result.get("subquestions", [])
        policy = planning_result.get("policy", {})
        recommended_mode = planning_result.get("recommended_mode", "hybrid")
        
        # Convert intent string to enum
        intent = Intent.FACTOID  # Default
        if intent_str == "enumeration":
            intent = Intent.ENUMERATION
        elif intent_str == "causal":
            intent = Intent.CAUSAL
        elif intent_str == "comparative":
            intent = Intent.COMPARATIVE
        elif intent_str == "procedural":
            intent = Intent.PROCEDURAL
        
        # Convert mode string to enum
        mode = Mode.HYBRID  # Default
        if recommended_mode == "dense":
            mode = Mode.DENSE
        elif recommended_mode == "kg":
            mode = Mode.KG
        
    except Exception as e:
        logger.error(f"LLM planning failed: {e}")
        # Fallback to simple analysis
        intent = Intent.FACTOID
        entities_surface = []
        entities_canonical = []
        subquestions = [query]
        policy = {"max_hops": 2, "max_neighbors": 10, "time_budget": 30}
        mode = Mode.HYBRID
    
    # Log planning completion
    try:
        await mcp_client.call(
            server="trace",
            method="log_event",
            payload={
                "trace_id": trace_id,
                "event_type": "planner_complete",
                "data": {
                    "intent": intent,
                    "entities_count": len(entities),
                    "subquestions_count": len(subquestions)
                },
                "span": "planning"
            },
            span="planner_complete",
            node="planner"
        )
    except Exception as e:
        logger.warning(f"Failed to log completion: {e}")
    
    logger.info(f"[{trace_id[:8]}] Planning complete: intent={intent}, entities={len(entities_surface)}")
    
    # Return state updates (LangGraph will merge with existing state)
    return {
        "intent": intent,
        "subqs": subquestions,
        "entities_surface": entities_surface,
        "entities_canonical": entities_canonical,
        "ambiguity_flags": [],
        "policy": policy,
        "mode": mode
    }


async def _llm_plan_query(query: str, persona: str, mcp_client, trace_id: str) -> Dict[str, Any]:
    """Use LLM to intelligently plan query retrieval strategy"""
    
    # Create LLM messages
    messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {"role": "user", "content": PLANNING_USER_PROMPT.format(query=query, persona=persona)}
    ]
    
    try:
        # Use Groq LLM for planning
        from langchain_groq import ChatGroq
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL or "llama3-8b-8192",
            temperature=0.1,  # Low temperature for consistent planning
            max_tokens=1000
        )
        
        # Convert to LangChain messages
        lc_messages = [
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=PLANNING_USER_PROMPT.format(query=query, persona=persona))
        ]
        
        # Get LLM response
        response = await llm.ainvoke(lc_messages)
        response_text = response.content
        
        # Parse JSON response
        try:
            planning_result = json.loads(response_text)
            logger.info(f"[{trace_id[:8]}] LLM planning successful: {planning_result.get('intent', 'unknown')}")
            return planning_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM planning response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    planning_result = json.loads(json_match.group())
                    return planning_result
                except:
                    pass
            
            # Return fallback
            return {
                "intent": "factoid",
                "entities_surface": [],
                "entities_canonical": [],
                "subquestions": [query],
                "complexity": "moderate",
                "recommended_mode": "hybrid",
                "policy": {"max_hops": 2, "max_neighbors": 10, "time_budget": 30},
                "reasoning": "LLM parsing failed, using fallback"
            }
            
    except Exception as e:
        logger.error(f"LLM planning call failed: {e}")
        # Return fallback planning
        return {
            "intent": "factoid", 
            "entities_surface": [],
            "entities_canonical": [],
            "subquestions": [query],
            "complexity": "moderate",
            "recommended_mode": "hybrid", 
            "policy": {"max_hops": 2, "max_neighbors": 10, "time_budget": 30},
            "reasoning": f"LLM call failed: {str(e)}"
        }

def _detect_intent(query: str) -> str:
    """Detect query intent using pattern matching and heuristics"""
    
    query_lower = query.lower()
    intent_scores = {"factoid": 0, "enumeration": 0, "causal": 0}
    
    # Pattern-based scoring
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                intent_scores[intent] += 1
    
    # Heuristic scoring
    if query_lower.startswith("what"):
        if "are" in query_lower or "list" in query_lower:
            intent_scores["enumeration"] += 2
        else:
            intent_scores["factoid"] += 2
    
    if query_lower.startswith("why"):
        intent_scores["causal"] += 3
    
    if "how many" in query_lower or "list" in query_lower:
        intent_scores["enumeration"] += 2
    
    # Determine intent with highest score
    detected_intent = max(intent_scores, key=intent_scores.get)
    
    # If no patterns matched, default to factoid
    if intent_scores[detected_intent] == 0:
        detected_intent = "factoid"
    
    return detected_intent

async def _extract_entities_llm(query: str, mcp_client, trace_id: str) -> List[Dict[str, Any]]:
    """Extract entities from query using LLM intelligence via MCP validator"""
    
    try:
        # Use LLM for intelligent entity extraction
        extraction_result = await mcp_client.call(
            server="validator",
            method="validate_step",
            payload={
                "step": "entity_extraction",
                "query": query,
                "prompt": f"""Extract biomedical entities from this query:
Query: "{query}"

Identify and extract:
- Drugs/medications (e.g., metformin, aspirin)
- Diseases/conditions (e.g., diabetes, hypertension)
- Symptoms (e.g., headache, nausea)
- Anatomy (e.g., heart, liver)
- Procedures (e.g., surgery, MRI)

Respond with JSON:
{{
  "entities": [
    {{
      "text": "entity text",
      "type": "drug|disease|symptom|anatomy|procedure|other",
      "canonical_form": "standardized term",
      "confidence": 0.95,
      "start": 0,
      "end": 10
    }}
  ]
}}"""
            },
            span="entity_extraction",
            node="planner"
        )
        
        # Parse LLM response
        if extraction_result and "entities" in extraction_result:
            return extraction_result["entities"]
        else:
            logger.warning("LLM entity extraction failed, using fallback")
            return _extract_entities_fallback(query)
            
    except Exception as e:
        logger.error(f"LLM entity extraction error: {e}")
        return _extract_entities_fallback(query)

def _extract_entities_fallback(query: str) -> List[Dict[str, Any]]:
    """Fallback entity extraction using simple heuristics"""
    entities = []
    
    # Simple capitalized terms extraction as fallback
    capitalized_terms = re.findall(r'\b[A-Z][a-z]+\b', query)
    for i, term in enumerate(capitalized_terms):
        entities.append({
            "text": term,
            "type": "unknown",
            "canonical_form": term.lower(),
            "confidence": 0.6,
            "source": "fallback_heuristic",
            "start": query.find(term),
            "end": query.find(term) + len(term)
        })
    
    return entities

def _canonicalize_query(query: str, entities: List[Dict[str, Any]]) -> str:
    """Canonicalize query for better retrieval"""
    
    canonical = query.strip()
    
    # Remove common stop words that don't affect meaning
    stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
    words = canonical.split()
    content_words = [w for w in words if w.lower() not in stop_words or len(w) <= 2]
    canonical = " ".join(content_words)
    
    # Normalize punctuation
    canonical = re.sub(r'[^\w\s]', ' ', canonical)
    canonical = re.sub(r'\s+', ' ', canonical).strip()
    
    # Add entity canonical forms if not already present
    for entity in entities:
        if entity["canonical_form"] not in canonical.lower():
            canonical += f" {entity['canonical_form']}"
    
    return canonical

def _generate_subquestions(query: str, intent: str) -> List[str]:
    """Generate subquestions based on intent and query"""
    
    subquestions = []
    
    if intent == "factoid":
        subquestions = [
            f"What is {query}?",
            f"How does {query} work?",
            f"What are the effects of {query}?"
        ]
    elif intent == "enumeration":
        subquestions = [
            f"List all types of {query}",
            f"What are the categories of {query}?",
            f"Enumerate {query} examples"
        ]
    elif intent == "causal":
        subquestions = [
            f"What causes {query}?",
            f"Why does {query} happen?",
            f"What leads to {query}?"
        ]
    
    # Filter and limit subquestions
    subquestions = [sq for sq in subquestions if sq.lower() != query.lower()]
    subquestions = subquestions[:3]  # Limit to 3 subquestions
    
    return subquestions

def _select_policy(intent: str) -> Dict[str, Any]:
    """Select policy based on intent"""
    
    policies = {
        "factoid": {
            "max_kg_hops": 3,
            "max_dense_results": 5,
            "fusion_strategy": "weighted",
            "confidence_threshold": 0.7
        },
        "enumeration": {
            "max_kg_hops": 2,
            "max_dense_results": 10,
            "fusion_strategy": "union",
            "confidence_threshold": 0.6
        },
        "causal": {
            "max_kg_hops": 4,
            "max_dense_results": 8,
            "fusion_strategy": "causal_chain",
            "confidence_threshold": 0.8
        }
    }
    
    return policies.get(intent, policies["factoid"])
