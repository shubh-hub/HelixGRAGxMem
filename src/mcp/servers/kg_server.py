#!/usr/bin/env python3
"""
Knowledge Graph MCP Server
==========================
Provides KG search, entity validation, and graph traversal tools
via MCP protocol, powered by core HelixGRAGxMem components.
"""

import asyncio
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import duckdb
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core import KGWalker, RelationPruner, EdgePredictor
from core.retrieval_engine import HybridRetriever
from config import settings

logger = logging.getLogger(__name__)

# Global components - initialized once per server
kg_walker = None
hybrid_retriever = None
kg_db = None

@asynccontextmanager
async def kg_lifespan(server: FastMCP):
    """Initialize enhanced KG components on server startup"""
    global kg_walker, hybrid_retriever, kg_db
    
    logger.info("🚀 Starting Enhanced KG MCP Server with scalable components...")
    
    try:
        # Initialize database connection
        kg_db = duckdb.connect(settings.DB_PATH)
        
        # Initialize enhanced components
        kg_walker = KGWalker()
        kg_walker.connect_kg(settings.DB_PATH)
        
        hybrid_retriever = HybridRetriever()
        hybrid_retriever.initialize()
        
        edge_predictor = EdgePredictor()
        edge_predictor.connect_kg(settings.DB_PATH)
        
        logger.info("✅ Enhanced KG Server initialized with scalable components")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced KG Server: {e}")
        raise
    finally:
        if kg_db:
            kg_db.close()

# Create FastMCP server
mcp = FastMCP("KG Server", lifespan=kg_lifespan)

@mcp.tool()
async def search_kg(
    query: str, 
    entities: List[str] = None, 
    intent: str = "factoid", 
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Search knowledge graph using enhanced entropy-based traversal
    
    Args:
        query: Natural language query
        intent: Query intent (factoid, causal, enumeration, therapeutic)
        max_results: Maximum results to return
    
    Returns:
        Dict with results and metadata
    """
    global kg_walker, kg_db
    
    try:
        # Initialize components if not already done
        if kg_db is None:
            kg_db = duckdb.connect(settings.DB_PATH)
            logger.info("✅ KG database connected")
        
        if kg_walker is None:
            kg_walker = KGWalker()
            kg_walker.connect_kg(settings.DB_PATH)
            logger.info("✅ KG Walker initialized")
        
        # Use provided entities or extract from query
        if entities is None:
            entities = []
        
        if not entities:
            entities = []
            if kg_db:
                # Extract potential entities from query using multiple approaches
                query_lower = query.lower()
                
                # Look for entities that appear in the query
                words = query_lower.split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        result = kg_db.execute("""
                            SELECT DISTINCT subject FROM triples 
                            WHERE LOWER(subject) LIKE ?
                            LIMIT 5
                        """, (f"%{word}%",)).fetchall()
                        
                        for row in result:
                            entity = row[0]
                            # Check if the full entity name appears in the query
                            if entity.lower() in query_lower:
                                entities.append(entity)
                
                # If no entities found, try word-by-word matching
                if not entities:
                    words = query.split()
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            result = kg_db.execute("""
                                SELECT DISTINCT subject FROM triples 
                                WHERE LOWER(subject) LIKE ?
                                LIMIT 3
                            """, (f"%{word.lower()}%",)).fetchall()
                            entities.extend([row[0] for row in result if row[0]])
                            if entities:
                                break
        
        if not entities:
            return {"results": [], "total_found": 0, "method": "core_kg_walker"}
        
        # Use priority queue traversal for methodology alignment
        results = kg_walker.walk_priority_queue(
            query=query,
            seed_entities=entities,
            intent=intent,
            max_expansions=1000
        )
        
        # Format results for MCP response
        formatted_results = []
        for path_info in results:
            formatted_results.append({
                "entity": path_info.get("entity", ""),
                "path": path_info.get("path", []),
                "relations": path_info.get("relation", ""),  # Single relation from priority queue
                "confidence": path_info.get("confidence", 0.0),
                "depth": path_info.get("depth", 0),
                "reasoning": path_info.get("reasoning", "")
            })
        
        return {
            "results": formatted_results,
            "total_found": len(formatted_results),
            "entities_resolved": len(entities),
            "cache_enabled": False,
            "method": "priority_queue_traversal"
        }
        
    except Exception as e:
        logger.error(f"Enhanced KG search failed: {e}")
        return {"error": str(e), "results": []}

@mcp.tool()
async def validate_entities(entities: List[str]) -> Dict[str, Any]:
    """
    Validate entities exist in KG
    
    Args:
        entities: List of entities to validate
        
    Returns:
        Validation results
    """
    global kg_db
    
    try:
        valid_entities = []
        entity_info = {}
        
        for entity in entities:
            result = kg_db.execute("""
                SELECT COUNT(*) as count,
                       GROUP_CONCAT(DISTINCT subject_type) as s_types,
                       GROUP_CONCAT(DISTINCT object_type) as o_types
                FROM triples 
                WHERE subject = ? OR object = ?
            """, (entity, entity)).fetchone()
            
            if result[0] > 0:
                valid_entities.append(entity)
                entity_info[entity] = {
                    "connection_count": result[0],
                    "types": list(set(filter(None, (result[1] or "").split(",") + (result[2] or "").split(","))))
                }
        
        return {
            "valid_entities": valid_entities,
            "total_validated": len(valid_entities),
            "entity_info": entity_info
        }
        
    except Exception as e:
        logger.error(f"Entity validation failed: {e}")
        return {"error": str(e), "valid_entities": []}

@mcp.tool()
async def get_neighbors(
    entity: str,
    max_neighbors: int = 20,
    relation_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get neighbors of an entity in the KG
    
    Args:
        entity: Entity name
        max_neighbors: Maximum number of neighbors
        relation_types: Filter by relation types
        
    Returns:
        Neighbor entities and relations
    """
    global kg_db
    
    try:
        # Build query with optional relation type filter
        where_clause = "WHERE (subject = ? OR object = ?)"
        params = [entity, entity]
        
        if relation_types:
            placeholders = ",".join("?" * len(relation_types))
            where_clause += f" AND predicate IN ({placeholders})"
            params.extend(relation_types)
        
        query = f"""
            SELECT subject, predicate, object, subject_type, object_type
            FROM triples 
            {where_clause}
            LIMIT ?
        """
        params.append(max_neighbors)
        
        result = kg_db.execute(query, params).fetchall()
        
        neighbors = []
        for row in result:
            subject, predicate, obj, s_type, o_type = row
            
            # Determine which is the neighbor
            if subject == entity:
                neighbor = {"entity": obj, "type": o_type, "relation": predicate, "direction": "outgoing"}
            else:
                neighbor = {"entity": subject, "type": s_type, "relation": predicate, "direction": "incoming"}
            
            neighbors.append(neighbor)
        
        return {
            "entity": entity,
            "neighbors": neighbors,
            "total_found": len(neighbors)
        }
        
    except Exception as e:
        logger.error(f"Get neighbors failed: {e}")
        return {"error": str(e), "neighbors": []}

@mcp.tool()
async def get_schema_info() -> Dict[str, Any]:
    """Get KG schema information"""
    global kg_db
    
    try:
        # Get relation types
        relations = kg_db.execute("""
            SELECT predicate, COUNT(*) as count
            FROM triples 
            GROUP BY predicate
            ORDER BY count DESC
            LIMIT 20
        """).fetchall()
        
        # Get entity types
        entity_types = kg_db.execute("""
            SELECT subject_type, COUNT(DISTINCT subject) as count
            FROM triples 
            WHERE subject_type IS NOT NULL
            GROUP BY subject_type
            ORDER BY count DESC
            LIMIT 20
        """).fetchall()
        
        # Get total stats
        stats = kg_db.execute("""
            SELECT 
                COUNT(*) as total_triples,
                COUNT(DISTINCT subject) as total_subjects,
                COUNT(DISTINCT object) as total_objects,
                COUNT(DISTINCT predicate) as total_relations
            FROM triples
        """).fetchone()
        
        return {
            "schema": {
                "relations": [{"relation": r[0], "count": r[1]} for r in relations],
                "entity_types": [{"type": t[0], "count": t[1]} for t in entity_types]
            },
            "components_initialized": {
                "kg_walker": kg_walker is not None,
                "hybrid_retriever": hybrid_retriever is not None,
                "kg_database": kg_db is not None
            },
            "statistics": {
                "total_triples": stats[0],
                "total_subjects": stats[1],
                "total_objects": stats[2],
                "total_relations": stats[3]
            }
        }
        
    except Exception as e:
        logger.error(f"Schema info failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="stdio")
