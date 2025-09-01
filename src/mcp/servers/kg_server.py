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
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
import duckdb

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core import KGWalker, RelationPruner, EdgePredictor
from src.config import settings

logger = logging.getLogger(__name__)

# Global components - initialized once per server
kg_walker = None
relation_pruner = None
edge_predictor = None
kg_db = None

@asynccontextmanager
async def kg_lifespan(server: FastMCP):
    """Initialize core components on server startup"""
    global kg_walker, relation_pruner, edge_predictor, kg_db
    
    logger.info("🚀 Starting KG MCP Server with core components...")
    
    try:
        # Initialize database connection
        kg_db = duckdb.connect(settings.DB_PATH)
        
        # Initialize core components
        kg_walker = KGWalker()
        kg_walker.connect_kg(settings.DB_PATH)
        
        relation_pruner = RelationPruner()
        relation_pruner.connect_kg(settings.DB_PATH)
        
        edge_predictor = EdgePredictor()
        edge_predictor.connect_kg(settings.DB_PATH)
        
        logger.info("✅ KG MCP Server initialized with core components")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize KG MCP Server: {e}")
        raise
    finally:
        if kg_db:
            kg_db.close()

# Create FastMCP server
mcp = FastMCP("KG Server", lifespan=kg_lifespan)

@mcp.tool()
async def search_kg(
    query: str,
    intent: str = "factoid",
    max_results: int = 10,
    max_hops: int = 3,
    entities: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform sophisticated KG search using core components
    
    Args:
        query: Search query
        intent: Query intent (factoid, enumeration, etc.)
        max_results: Maximum number of results
        max_hops: Maximum traversal hops
        entities: Starting entities (optional)
        
    Returns:
        Search results with confidence-scored paths
    """
    global kg_walker, kg_db
    
    try:
        # Extract entities if not provided
        if not entities:
            result = kg_db.execute("""
                SELECT DISTINCT subject FROM triples 
                WHERE subject ILIKE ? OR object ILIKE ?
                LIMIT 5
            """, (f"%{query.split()[0]}%", f"%{query.split()[0]}%")).fetchall()
            entities = [row[0] for row in result if row[0]]
        
        if not entities:
            return {"results": [], "total_found": 0, "method": "core_kg_walker"}
        
        # Use KGWalker for sophisticated graph traversal
        paths = kg_walker.walk(
            start_entities=entities,
            query_context=query,
            max_hops=max_hops,
            max_paths=max_results,
            confidence_threshold=0.3
        )
        
        # Format results for MCP response
        results = []
        for path_info in paths:
            results.append({
                "path": path_info["path"],
                "relations": path_info["relations"],
                "confidence": path_info["confidence"],
                "hops": path_info["hops"],
                "entities": path_info.get("entities_found", [])
            })
        
        return {
            "results": results,
            "total_found": len(results),
            "method": "core_kg_walker"
        }
        
    except Exception as e:
        logger.error(f"KG search failed: {e}")
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
