#!/usr/bin/env python3
"""
Explain MCP Server - FastMCP Implementation
===========================================

FastMCP-based server for persona-aware explanations.
Generates clinical explanations tailored for doctors, patients, and researchers.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession

# Import our existing components
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

logger = logging.getLogger(__name__)

# Persona-specific configurations
PERSONA_CONFIGS = {
    "doctor": {
        "style": "clinical",
        "terminology": "medical",
        "detail_level": "high",
        "confidence_expression": "clinical_certainty",
        "max_length": 500,
        "focus": ["diagnosis", "treatment", "contraindications", "evidence_quality"]
    },
    "patient": {
        "style": "accessible",
        "terminology": "simplified",
        "detail_level": "moderate", 
        "confidence_expression": "plain_language",
        "max_length": 300,
        "focus": ["condition_explanation", "treatment_options", "next_steps", "reassurance"]
    },
    "researcher": {
        "style": "analytical",
        "terminology": "technical",
        "detail_level": "comprehensive",
        "confidence_expression": "statistical",
        "max_length": 600,
        "focus": ["methodology", "evidence_strength", "limitations", "future_research"]
    }
}

# Server context
@asynccontextmanager
async def explain_lifespan(server: FastMCP):
    """Manage Explain server lifecycle."""
    logger.info("Initializing Explain MCP Server...")
    
    try:
        yield {"persona_configs": PERSONA_CONFIGS}
    except Exception as e:
        logger.error(f"Failed to initialize Explain server: {e}")
        raise
    finally:
        logger.info("Shutting down Explain MCP Server...")

# Create FastMCP server
mcp = FastMCP("Explain Server", lifespan=explain_lifespan)

@mcp.tool()
def generate_explanation(
    persona: str,
    query: str,
    answer: str,
    evidence: List[Dict[str, Any]] = None,
    confidence: float = None,
    reasoning_path: List[str] = None,
    trace_id: str = None
) -> Dict[str, Any]:
    """
    Generate persona-aware explanation.
    
    Args:
        persona: Target persona (doctor, patient, researcher)
        query: Original query
        answer: System answer to explain
        evidence: Supporting evidence
        confidence: Answer confidence
        reasoning_path: Reasoning steps
        trace_id: Trace ID for logging
    
    Returns:
        Persona-aware explanation
    """
    try:
        config = PERSONA_CONFIGS.get(persona, PERSONA_CONFIGS["doctor"])
        evidence = evidence or []
        reasoning_path = reasoning_path or []
        
        # Build explanation components
        explanation_parts = []
        
        # 1. Answer summary (persona-appropriate)
        if persona == "doctor":
            explanation_parts.append(f"**Clinical Assessment:** {answer}")
        elif persona == "patient":
            explanation_parts.append(f"**In Simple Terms:** {answer}")
        else:  # researcher
            explanation_parts.append(f"**Research Finding:** {answer}")
        
        # 2. Confidence explanation
        if confidence is not None:
            conf_explanation = explain_confidence(persona, confidence)
            if conf_explanation.get("success"):
                explanation_parts.append(conf_explanation["explanation"])
        
        # 3. Reasoning path (if available)
        if reasoning_path:
            if persona == "doctor":
                explanation_parts.append("**Clinical Reasoning:**")
            elif persona == "patient":
                explanation_parts.append("**How We Got This Answer:**")
            else:
                explanation_parts.append("**Analytical Process:**")
            
            for i, step in enumerate(reasoning_path[:3], 1):  # Limit steps by persona
                explanation_parts.append(f"{i}. {step}")
        
        # 4. Evidence summary
        if evidence:
            citations = format_citations(persona, evidence)
            if citations.get("success"):
                explanation_parts.append(citations["formatted_citations"])
        
        # 5. Persona-specific additions
        if persona == "doctor" and evidence:
            explanation_parts.append("**Clinical Considerations:** Review contraindications and patient-specific factors before implementation.")
        elif persona == "patient":
            explanation_parts.append("**Next Steps:** Discuss these findings with your healthcare provider for personalized guidance.")
        elif persona == "researcher":
            explanation_parts.append("**Limitations:** Consider study design, sample size, and generalizability when interpreting these results.")
        
        # Combine and truncate if needed
        full_explanation = "\n\n".join(explanation_parts)
        
        if len(full_explanation) > config["max_length"]:
            truncated = full_explanation[:config["max_length"]-50] + "... [truncated]"
            full_explanation = truncated
        
        return {
            "success": True,
            "explanation": full_explanation,
            "persona": persona,
            "style": config["style"],
            "length": len(full_explanation),
            "components": {
                "answer_summary": True,
                "confidence_explanation": confidence is not None,
                "reasoning_path": len(reasoning_path) > 0,
                "evidence_citations": len(evidence) > 0
            },
            "trace_id": trace_id,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return {
            "success": False,
            "error": str(e),
            "persona": persona,
            "trace_id": trace_id
        }

@mcp.tool()
def format_citations(
    persona: str,
    evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format evidence citations for persona.
    
    Args:
        persona: Target persona
        evidence: Evidence to cite
    
    Returns:
        Formatted citations
    """
    try:
        if not evidence:
            return {
                "success": True,
                "formatted_citations": "",
                "citation_count": 0
            }
        
        citations = []
        
        if persona == "doctor":
            citations.append("**Evidence Base:**")
            for i, item in enumerate(evidence[:5], 1):  # Limit for doctors
                source = item.get("source", "Unknown")
                confidence = item.get("confidence", 0.0)
                citations.append(f"{i}. {source} (confidence: {confidence:.2f})")
        
        elif persona == "patient":
            citations.append("**Sources:**")
            for i, item in enumerate(evidence[:3], 1):  # Fewer for patients
                source = item.get("source", "Medical literature")
                citations.append(f"{i}. {source}")
        
        else:  # researcher
            citations.append("**References:**")
            for i, item in enumerate(evidence[:10], 1):  # More for researchers
                source = item.get("source", "Unknown")
                confidence = item.get("confidence", 0.0)
                metadata = item.get("metadata", {})
                citation = f"{i}. {source}"
                if confidence:
                    citation += f" (conf: {confidence:.3f})"
                if metadata.get("type"):
                    citation += f" [{metadata['type']}]"
                citations.append(citation)
        
        formatted = "\n".join(citations)
        
        return {
            "success": True,
            "formatted_citations": formatted,
            "citation_count": len(evidence),
            "persona": persona,
            "truncated": len(evidence) > (5 if persona == "doctor" else 3 if persona == "patient" else 10)
        }
        
    except Exception as e:
        logger.error(f"Error formatting citations: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def explain_confidence(
    persona: str,
    confidence: float,
    uncertainty_sources: List[str] = None
) -> Dict[str, Any]:
    """
    Explain confidence levels in persona-appropriate way.
    
    Args:
        persona: Target persona
        confidence: Confidence score
        uncertainty_sources: Sources of uncertainty
    
    Returns:
        Confidence explanation
    """
    try:
        uncertainty_sources = uncertainty_sources or []
        
        if persona == "doctor":
            if confidence >= 0.9:
                explanation = f"**High Clinical Confidence** ({confidence:.1%}): Strong evidence supports this assessment."
            elif confidence >= 0.7:
                explanation = f"**Moderate Clinical Confidence** ({confidence:.1%}): Good evidence with some limitations."
            elif confidence >= 0.5:
                explanation = f"**Limited Clinical Confidence** ({confidence:.1%}): Insufficient evidence for definitive assessment."
            else:
                explanation = f"**Low Clinical Confidence** ({confidence:.1%}): Significant uncertainty - consider additional evaluation."
        
        elif persona == "patient":
            if confidence >= 0.9:
                explanation = "**Very Confident**: We have strong evidence for this information."
            elif confidence >= 0.7:
                explanation = "**Fairly Confident**: Good evidence supports this, with minor uncertainties."
            elif confidence >= 0.5:
                explanation = "**Somewhat Uncertain**: Limited evidence - discuss with your doctor."
            else:
                explanation = "**Highly Uncertain**: Not enough evidence - definitely consult your healthcare provider."
        
        else:  # researcher
            if confidence >= 0.9:
                explanation = f"**High Statistical Confidence** (p={confidence:.3f}): Robust evidence with minimal uncertainty."
            elif confidence >= 0.7:
                explanation = f"**Moderate Statistical Confidence** (p={confidence:.3f}): Acceptable evidence quality."
            elif confidence >= 0.5:
                explanation = f"**Low Statistical Confidence** (p={confidence:.3f}): Significant methodological limitations."
            else:
                explanation = f"**Very Low Statistical Confidence** (p={confidence:.3f}): Results should be interpreted with extreme caution."
        
        # Add uncertainty sources if provided
        if uncertainty_sources:
            if persona == "doctor":
                explanation += f"\n**Uncertainty Factors:** {', '.join(uncertainty_sources)}"
            elif persona == "patient":
                explanation += f"\n**Why we're uncertain:** {', '.join(uncertainty_sources[:2])}"  # Limit for patients
            else:
                explanation += f"\n**Uncertainty Sources:** {'; '.join(uncertainty_sources)}"
        
        return {
            "success": True,
            "explanation": explanation,
            "confidence_level": confidence,
            "persona": persona,
            "uncertainty_sources": uncertainty_sources
        }
        
    except Exception as e:
        logger.error(f"Error explaining confidence: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_persona_config(
    persona: str
) -> Dict[str, Any]:
    """
    Get configuration for specific persona.
    
    Args:
        persona: Persona to get config for
    
    Returns:
        Persona configuration
    """
    try:
        config = PERSONA_CONFIGS.get(persona)
        if not config:
            return {
                "success": False,
                "error": f"Unknown persona: {persona}",
                "available_personas": list(PERSONA_CONFIGS.keys())
            }
        
        return {
            "success": True,
            "persona": persona,
            "config": config,
            "available_personas": list(PERSONA_CONFIGS.keys())
        }
        
    except Exception as e:
        logger.error(f"Error getting persona config: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    mcp.run()
