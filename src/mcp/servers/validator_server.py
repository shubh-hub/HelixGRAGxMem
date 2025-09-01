#!/usr/bin/env python3
"""
Validator MCP Server - FastMCP Implementation
=============================================

FastMCP-based server for clinical validation and safety checks.
Provides groundedness, safety, and coverage validation for clinical RAG.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession

# Import our existing components
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Global LLM instance
llm_instance = None

# Server context
@asynccontextmanager
async def validator_lifespan(server: FastMCP):
    """Manage Validator server lifecycle."""
    global llm_instance
    logger.info("Initializing Validator MCP Server...")
    
    try:
        # Initialize Groq LLM
        if settings.GROQ_API_KEY:
            llm_instance = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS
            )
            logger.info(f"✅ Groq LLM initialized: {settings.LLM_MODEL}")
        else:
            logger.warning("⚠️ No GROQ_API_KEY found - using mock responses")
        
        yield {"initialized": True, "llm": llm_instance}
    except Exception as e:
        logger.error(f"Failed to initialize Validator server: {e}")
        raise
    finally:
        logger.info("Shutting down Validator MCP Server...")

# Create FastMCP server
mcp = FastMCP("Validator Server", lifespan=validator_lifespan)

@mcp.tool()
def validate_answer(
    query: str,
    answer: str,
    evidence: List[Dict[str, Any]],
    trace_id: str = None
) -> Dict[str, Any]:
    """
    LLM-driven validation for biomedical queries and answers.
    
    Args:
        query: Original query (used as LLM prompt when answer is empty)
        answer: Generated answer (empty for LLM prompt processing)
        evidence: Supporting evidence
        trace_id: Trace ID for logging
    
    Returns:
        LLM response or validation results
    """
    try:
        # If answer is empty, treat query as LLM prompt
        if not answer.strip():
            # Use real Groq LLM for intelligent responses
            global llm_instance
            
            if llm_instance:
                try:
                    # Send query to Groq LLM
                    response = llm_instance.invoke([HumanMessage(content=query)])
                    llm_response = response.content
                    
                    # Parse LLM response based on query type
                    import json
                    import re
                    
                    # Try to extract JSON from LLM response
                    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                    if json_match:
                        try:
                            parsed_response = json.loads(json_match.group())
                            return parsed_response
                        except json.JSONDecodeError:
                            pass
                    
                    # If no valid JSON, provide structured fallback based on query type
                    if "intent" in query.lower() and "primary_intent" in query.lower():
                        return {
                            "primary_intent": "factoid",
                            "reasoning_type": "simple_lookup", 
                            "complexity_level": "medium",
                            "llm_analysis": llm_response
                        }
                    elif "transform" in query.lower() and "subquestions" in query.lower():
                        return {
                            "subquestions": ["What is the mechanism?", "What are the effects?", "What are the applications?"],
                            "simplified": query.split("Query:")[-1].strip() if "Query:" in query else query,
                            "rewritten": f"Biomedical query: {query.split('Query:')[-1].strip() if 'Query:' in query else query}",
                            "alternatives": ["mechanism", "effects", "applications"],
                            "llm_analysis": llm_response
                        }
                    elif "resource" in query.lower() and "retrieval" in query.lower():
                        return {
                            "mode": "hybrid",
                            "priority": "kg_first",
                            "fallback": "dense_only",
                            "evidence_types": ["structured_relationships", "literature_evidence"],
                            "llm_analysis": llm_response
                        }
                    elif "extract" in query.lower() and "entities" in query.lower():
                        # Extract entities from LLM response
                        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', llm_response)
                        return {
                            "diseases": [e for e in entities if any(term in e.lower() for term in ['disease', 'syndrome', 'disorder'])],
                            "drugs": [e for e in entities if any(term in e.lower() for term in ['drug', 'medication', 'compound'])],
                            "genes": [],
                            "processes": [],
                            "anatomy": [],
                            "phrases": entities[:5],
                            "llm_analysis": llm_response
                        }
                    elif "seed" in query.lower() and "traversal" in query.lower():
                        return {
                            "seed_nodes": ["primary_entity"],
                            "max_hops": 3,
                            "priority_relations": ["treats", "causes", "interacts_with"],
                            "budget": 50,
                            "stopping_criteria": "confidence_threshold",
                            "llm_analysis": llm_response
                        }
                    else:
                        return {
                            "response": llm_response,
                            "analysis": query,
                            "confidence": 0.8
                        }
                        
                except Exception as e:
                    logger.error(f"Error calling Groq LLM: {e}")
                    # Fall back to mock response
                    pass
            
            # Fallback mock responses if LLM fails
            if "intent" in query.lower():
                return {"primary_intent": "factoid", "reasoning_type": "simple_lookup", "complexity_level": "medium"}
            elif "transform" in query.lower():
                return {"subquestions": ["What is the mechanism?"], "simplified": query, "rewritten": query, "alternatives": []}
            elif "resource" in query.lower():
                return {"mode": "hybrid", "priority": "kg_first", "fallback": "dense_only", "evidence_types": []}
            elif "extract" in query.lower():
                return {"diseases": [], "drugs": [], "genes": [], "processes": [], "anatomy": [], "phrases": []}
            elif "seed" in query.lower():
                return {"seed_nodes": [], "max_hops": 3, "priority_relations": [], "budget": 50, "stopping_criteria": "confidence"}
            else:
                return {"response": "Mock response", "analysis": query, "confidence": 0.5}
        
        # Standard validation logic for actual answers
        validation_results = {
            "is_grounded": len(evidence) > 0,
            "has_answer": len(answer.strip()) > 0,
            "answer_length": len(answer),
            "evidence_count": len(evidence),
            "confidence_score": 0.0
        }
        
        # Calculate confidence based on evidence
        if evidence:
            avg_confidence = sum(e.get("confidence", 0.0) for e in evidence) / len(evidence)
            validation_results["confidence_score"] = avg_confidence
        
        # Determine overall validity
        is_valid = (
            validation_results["is_grounded"] and
            validation_results["has_answer"] and
            validation_results["confidence_score"] > 0.3
        )
        
        return {
            "success": True,
            "is_valid": is_valid,
            "validation_results": validation_results,
            "query": query,
            "trace_id": trace_id
        }
        
    except Exception as e:
        logger.error(f"Error validating answer: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "trace_id": trace_id
        }

@mcp.tool()
def check_safety(
    query: str,
    answer: str,
    context: str = "clinical",
    trace_id: str = None
) -> Dict[str, Any]:
    """
    Check for safety issues in clinical context.
    
    Args:
        query: Original query
        answer: Generated answer
        context: Context (clinical, patient, etc.)
        trace_id: Trace ID for logging
    
    Returns:
        Safety assessment
    """
    try:
        # Basic safety checks (can be enhanced with LLM)
        safety_flags = []
        
        # Check for medical disclaimers
        if context == "clinical":
            if "consult" not in answer.lower() and "doctor" not in answer.lower():
                safety_flags.append("missing_medical_disclaimer")
        
        # Check for uncertainty expression
        if "uncertain" not in answer.lower() and "may" not in answer.lower():
            if len(answer) > 100:  # Only for longer answers
                safety_flags.append("lacks_uncertainty_expression")
        
        # Check for harmful content (basic)
        harmful_keywords = ["dangerous", "toxic", "lethal", "fatal"]
        if any(keyword in answer.lower() for keyword in harmful_keywords):
            safety_flags.append("potentially_harmful_content")
        
        is_safe = len(safety_flags) == 0
        
        return {
            "success": True,
            "is_safe": is_safe,
            "safety_flags": safety_flags,
            "context": context,
            "query": query,
            "trace_id": trace_id
        }
        
    except Exception as e:
        logger.error(f"Error checking safety: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "trace_id": trace_id
        }

@mcp.tool()
def assess_groundedness(
    answer: str,
    evidence: List[Dict[str, Any]],
    threshold: float = 0.5,
    trace_id: str = None
) -> Dict[str, Any]:
    """
    Assess how well the answer is grounded in evidence.
    
    Args:
        answer: Generated answer
        evidence: Supporting evidence
        threshold: Groundedness threshold
        trace_id: Trace ID for logging
    
    Returns:
        Groundedness assessment
    """
    try:
        if not evidence:
            return {
                "success": True,
                "is_grounded": False,
                "groundedness_score": 0.0,
                "reason": "no_evidence_provided",
                "trace_id": trace_id
            }
        
        # Simple groundedness assessment
        # In practice, this could use more sophisticated NLP techniques
        evidence_text = " ".join([e.get("content", "") for e in evidence])
        answer_words = set(answer.lower().split())
        evidence_words = set(evidence_text.lower().split())
        
        # Calculate overlap
        overlap = len(answer_words.intersection(evidence_words))
        total_answer_words = len(answer_words)
        
        groundedness_score = overlap / total_answer_words if total_answer_words > 0 else 0.0
        is_grounded = groundedness_score >= threshold
        
        return {
            "success": True,
            "is_grounded": is_grounded,
            "groundedness_score": groundedness_score,
            "threshold": threshold,
            "word_overlap": overlap,
            "total_answer_words": total_answer_words,
            "evidence_count": len(evidence),
            "trace_id": trace_id
        }
        
    except Exception as e:
        logger.error(f"Error assessing groundedness: {e}")
        return {
            "success": False,
            "error": str(e),
            "trace_id": trace_id
        }

@mcp.tool()
def check_coverage(
    query: str,
    answer: str,
    required_aspects: List[str] = None,
    trace_id: str = None
) -> Dict[str, Any]:
    """
    Check if answer covers required aspects of the query.
    
    Args:
        query: Original query
        answer: Generated answer
        required_aspects: Required aspects to cover
        trace_id: Trace ID for logging
    
    Returns:
        Coverage assessment
    """
    try:
        if not required_aspects:
            # Default aspects for medical queries
            required_aspects = ["symptoms", "treatment", "diagnosis", "causes"]
        
        coverage_results = {}
        for aspect in required_aspects:
            # Simple keyword-based coverage check
            is_covered = aspect.lower() in answer.lower()
            coverage_results[aspect] = is_covered
        
        covered_count = sum(coverage_results.values())
        coverage_percentage = covered_count / len(required_aspects) if required_aspects else 0.0
        
        return {
            "success": True,
            "coverage_results": coverage_results,
            "covered_aspects": covered_count,
            "total_aspects": len(required_aspects),
            "coverage_percentage": coverage_percentage,
            "query": query,
            "trace_id": trace_id
        }
        
    except Exception as e:
        logger.error(f"Error checking coverage: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "trace_id": trace_id
        }

@mcp.tool()
def comprehensive_validation(
    query: str,
    answer: str,
    evidence: List[Dict[str, Any]],
    context: str = "clinical",
    trace_id: str = None
) -> Dict[str, Any]:
    """
    Perform comprehensive validation combining all checks.
    
    Args:
        query: Original query
        answer: Generated answer
        evidence: Supporting evidence
        context: Context for validation
        trace_id: Trace ID for logging
    
    Returns:
        Comprehensive validation results
    """
    try:
        # Run all validation checks
        answer_validation = validate_answer(query, answer, evidence, trace_id)
        safety_check = check_safety(query, answer, context, trace_id)
        groundedness_check = assess_groundedness(answer, evidence, trace_id=trace_id)
        coverage_check = check_coverage(query, answer, trace_id=trace_id)
        
        # Combine results
        overall_valid = (
            answer_validation.get("is_valid", False) and
            safety_check.get("is_safe", False) and
            groundedness_check.get("is_grounded", False)
        )
        
        return {
            "success": True,
            "overall_valid": overall_valid,
            "answer_validation": answer_validation,
            "safety_check": safety_check,
            "groundedness_check": groundedness_check,
            "coverage_check": coverage_check,
            "query": query,
            "context": context,
            "trace_id": trace_id
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive validation: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "trace_id": trace_id
        }

if __name__ == "__main__":
    mcp.run()
