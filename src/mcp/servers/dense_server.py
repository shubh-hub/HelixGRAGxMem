#!/usr/bin/env python3
"""
Dense Retrieval MCP Server
==========================
Provides dense semantic search tools via MCP protocol,
powered by core HelixGRAGxMem components.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core import DenseRetriever
from src.config import settings

logger = logging.getLogger(__name__)

# Global components - initialized once per server
dense_retriever = None

@asynccontextmanager
async def dense_lifespan(server: FastMCP):
    """Initialize core components on server startup"""
    global dense_retriever
    
    logger.info("🚀 Starting Dense MCP Server with core components...")
    
    try:
        # Initialize core components
        dense_retriever = DenseRetriever()
        dense_retriever.load_index(
            index_path=settings.FAISS_INDEX_PATH,
            id_map_path=settings.FAISS_ID_MAP_PATH,
            passages_path=settings.VERBALIZED_KG_PATH
        )
        
        logger.info("✅ Dense MCP Server initialized with core components")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize Dense MCP Server: {e}")
        raise

# Create FastMCP server
mcp = FastMCP("Dense Server", lifespan=dense_lifespan)

@mcp.tool()
async def search_passages(
    query: str,
    max_results: int = 10,
    score_threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Perform sophisticated dense search using core components
    
    Args:
        query: Search query
        max_results: Maximum number of results
        score_threshold: Minimum similarity threshold
        
    Returns:
        Relevant passages with similarity scores
    """
    global dense_retriever
    
    try:
        # Use DenseRetriever for sophisticated semantic search
        results = dense_retriever.search(
            query=query,
            k=max_results
        )
        
        # Apply score threshold filtering if specified
        if score_threshold > 0.0:
            results = [r for r in results if r.get("score", 0.0) >= score_threshold]
        
        # Format results for MCP response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "passage_id": result.get("id", result.get("passage_id", "")),
                "text": result["text"],
                "score": result["score"],
                "metadata": result.get("metadata", {})
            })
        
        return {
            "results": formatted_results,
            "total_found": len(formatted_results),
            "method": "core_dense_retriever"
        }
        
    except Exception as e:
        logger.error(f"Dense search failed: {e}")
        return {"error": str(e), "results": []}

@mcp.tool()
async def get_passage_details(
    passage_ids: List[str]
) -> Dict[str, Any]:
    """
    Get detailed information for specific passages
    
    Args:
        passage_ids: List of passage IDs
        
    Returns:
        Detailed passage information
    """
    global dense_retriever
    
    try:
        details = []
        for passage_id in passage_ids:
            # Get passage from dense retriever
            passage_info = dense_retriever.get_passage_by_id(passage_id)
            if passage_info:
                details.append(passage_info)
        
        return {
            "passage_details": details,
            "total_found": len(details)
        }
        
    except Exception as e:
        logger.error(f"Get passage details failed: {e}")
        return {"error": str(e), "passage_details": []}

@mcp.tool()
async def get_dense_stats() -> Dict[str, Any]:
    """Get dense retrieval statistics"""
    global dense_retriever
    
    try:
        stats = dense_retriever.get_stats()
        
        return {
            "statistics": stats,
            "index_info": {
                "total_passages": stats.get("total_passages", 0),
                "embedding_dimension": stats.get("embedding_dim", 0),
                "model_name": stats.get("model_name", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Get dense stats failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
