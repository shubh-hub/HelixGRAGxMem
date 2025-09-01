"""
FastAPI main application for HelixGRAGxMem biomedical RAG system
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import asyncio
from typing import Dict, Any

from ...mas.nli_orchestrator import NLIMASOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HelixGRAGxMem API",
    description="Biomedical Hybrid Retrieval-Augmented Generation System",
    version="1.0.0"
)

# Global orchestrator instance
orchestrator = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    success: bool
    answer: str
    intent: str = "unknown"
    mode: str = "unknown"
    entities: list = []
    evidence: Dict[str, Any] = {}
    trace_id: str = ""
    error: str = ""

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    global orchestrator
    try:
        logger.info("🚀 Initializing HelixGRAGxMem orchestrator...")
        orchestrator = NLIMASOrchestrator()
        await orchestrator.initialize()
        logger.info("✅ Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup orchestrator on shutdown"""
    global orchestrator
    if orchestrator:
        try:
            await orchestrator.cleanup()
            logger.info("🧹 Orchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "HelixGRAGxMem Biomedical RAG API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "orchestrator_ready": orchestrator is not None}

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a biomedical query through the hybrid RAG system
    
    Args:
        request: QueryRequest containing the biomedical query
        
    Returns:
        QueryResponse with answer and metadata
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        logger.info(f"🔍 Processing query: {request.query}")
        
        # Process query through orchestrator
        result = await orchestrator.process_query(request.query)
        
        if result.get("success"):
            return QueryResponse(
                success=True,
                answer=result.get("answer", "No answer generated"),
                intent=result.get("intent", "unknown"),
                mode=result.get("mode", "unknown"),
                entities=result.get("entities", []),
                evidence=result.get("evidence", {}),
                trace_id=result.get("trace_id", "")
            )
        else:
            return QueryResponse(
                success=False,
                answer="",
                error=result.get("error", "Unknown error"),
                trace_id=result.get("trace_id", "")
            )
            
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
