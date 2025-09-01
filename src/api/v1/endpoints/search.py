from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging

from src.config import settings
from src.mas.nli_orchestrator import NLIMASOrchestrator

router = APIRouter()
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    trace_id: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    confidence: Optional[float] = None
    trace_id: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None

@router.post("/query", response_model=QueryResponse)
async def process_biomedical_query(request: QueryRequest):
    """
    Process a biomedical query using the NLI-MAS orchestrator with MCP integration.
    
    This endpoint provides the complete HelixGRAGxMem functionality:
    - LangGraph workflow orchestration
    - MCP server communication for KG and dense retrieval
    - Core component backends (KGWalker, DenseRetriever, etc.)
    - Hybrid retrieval with fusion scoring
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Initialize NLI orchestrator
        orchestrator = NLIMASOrchestrator()
        
        # Process query through complete workflow
        result = await orchestrator.process_query(
            query=request.query,
            trace_id=request.trace_id
        )
        
        return QueryResponse(
            success=result["success"],
            answer=result.get("answer"),
            confidence=result.get("confidence", 0.0),
            trace_id=result.get("trace_id"),
            metadata=result.get("metadata", {}),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

@router.get("/health")
def health_check():
    """
    Health check endpoint for the search API.
    """
    return {
        "status": "healthy",
        "message": "HelixGRAGxMem search API is operational",
        "version": "2.0.0"
    }

@router.get("/status")
def system_status():
    """
    System status endpoint showing configuration.
    """
    return {
        "system": "HelixGRAGxMem",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "architecture": "NLI-MAS with MCP integration"
    }
