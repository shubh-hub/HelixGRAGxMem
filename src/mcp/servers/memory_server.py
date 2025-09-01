#!/usr/bin/env python3
"""
Memory MCP Server - FastMCP Implementation
==========================================

FastMCP-based server implementing MIRIX-inspired memory management.
Provides 6 memory components: Core, Episodic, Semantic, Procedural, Resource, Vault.
"""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession

# Import our existing components
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

logger = logging.getLogger(__name__)

# Server context for managing memory database
@asynccontextmanager
async def memory_lifespan(server: FastMCP):
    """Manage Memory server lifecycle and database."""
    logger.info("Initializing Memory MCP Server...")
    
    try:
        # Initialize SQLite database
        db_path = project_root / "data" / "processed" / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        db_connection = sqlite3.connect(str(db_path))
        db_connection.row_factory = sqlite3.Row
        
        # Create memory tables
        await create_memory_tables(db_connection)
        
        yield {"db_connection": db_connection}
    except Exception as e:
        logger.error(f"Failed to initialize Memory server: {e}")
        raise
    finally:
        if 'db_connection' in locals():
            db_connection.close()
        logger.info("Shutting down Memory MCP Server...")

async def create_memory_tables(db_connection):
    """Create MIRIX-inspired memory tables"""
    cursor = db_connection.cursor()
    
    # Core Memory: Persistent persona/user facts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS core_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(persona, fact_type, content)
        )
    """)
    
    # Episodic Memory: Time-stamped events and interactions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT,
            intent TEXT,
            confidence REAL,
            evidence_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT
        )
    """)
    
    # Create indexes separately
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_trace_id ON episodic_memory(trace_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_created_at ON episodic_memory(created_at)")
    
    # Knowledge Vault: External knowledge integration
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_vault (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            entity TEXT NOT NULL,
            knowledge_type TEXT NOT NULL,
            content TEXT NOT NULL,
            relevance_score REAL DEFAULT 0.0,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            UNIQUE(source, entity, knowledge_type)
        )
    """)
    
    db_connection.commit()

# Create FastMCP server
mcp = FastMCP("Memory Server", lifespan=memory_lifespan)

@mcp.tool()
def store_episodic(
    trace_id: str,
    query: str,
    answer: str = None,
    intent: str = None,
    confidence: float = None,
    evidence_summary: str = None
) -> Dict[str, Any]:
    """
    Store query-answer interactions in episodic memory.
    
    Args:
        trace_id: Trace ID
        query: User query
        answer: System answer
        intent: Query intent
        confidence: Answer confidence
        evidence_summary: Evidence summary
    
    Returns:
        Storage result
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        cursor.execute("""
            INSERT INTO episodic_memory 
            (trace_id, query, answer, intent, confidence, evidence_summary)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (trace_id, query, answer, intent, confidence, evidence_summary))
        
        db_connection.commit()
        
        return {
            "success": True,
            "memory_id": cursor.lastrowid,
            "trace_id": trace_id,
            "stored_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error storing episodic memory: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def retrieve_episodic(
    query: str,
    limit: int = 5,
    time_window_hours: int = 24
) -> Dict[str, Any]:
    """
    Retrieve similar past interactions from episodic memory.
    
    Args:
        query: Query to find similar interactions
        limit: Max results
        time_window_hours: Time window in hours
    
    Returns:
        Similar past interactions
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        time_cutoff = datetime.now() - timedelta(hours=time_window_hours)
        
        cursor.execute("""
            SELECT trace_id, query, answer, intent, confidence, evidence_summary, created_at
            FROM episodic_memory
            WHERE created_at > ? 
            AND (query LIKE ? OR answer LIKE ?)
            ORDER BY created_at DESC
            LIMIT ?
        """, (time_cutoff.isoformat(), f"%{query}%", f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "trace_id": row["trace_id"],
                "query": row["query"],
                "answer": row["answer"],
                "intent": row["intent"],
                "confidence": row["confidence"],
                "evidence_summary": row["evidence_summary"],
                "created_at": row["created_at"]
            })
        
        return {
            "success": True,
            "memories": results,
            "num_found": len(results),
            "query": query,
            "time_window_hours": time_window_hours
        }
        
    except Exception as e:
        logger.error(f"Error retrieving episodic memory: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def store_core(
    persona: str,
    fact_type: str,
    content: str,
    confidence: float = 1.0
) -> Dict[str, Any]:
    """
    Store persistent user/system facts in core memory.
    
    Args:
        persona: Persona (doctor, patient, etc.)
        fact_type: Type of fact
        content: Fact content
        confidence: Confidence level
    
    Returns:
        Storage result
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO core_memory 
            (persona, fact_type, content, confidence, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (persona, fact_type, content, confidence))
        
        db_connection.commit()
        
        return {
            "success": True,
            "memory_id": cursor.lastrowid,
            "persona": persona,
            "fact_type": fact_type,
            "stored_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error storing core memory: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def retrieve_core(
    persona: str,
    fact_type: str = None
) -> Dict[str, Any]:
    """
    Get relevant core memories for persona.
    
    Args:
        persona: Persona to retrieve for
        fact_type: Optional fact type filter
    
    Returns:
        Core memories
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        if fact_type:
            cursor.execute("""
                SELECT fact_type, content, confidence, created_at, updated_at
                FROM core_memory
                WHERE persona = ? AND fact_type = ?
                ORDER BY updated_at DESC
            """, (persona, fact_type))
        else:
            cursor.execute("""
                SELECT fact_type, content, confidence, created_at, updated_at
                FROM core_memory
                WHERE persona = ?
                ORDER BY updated_at DESC
            """, (persona,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "fact_type": row["fact_type"],
                "content": row["content"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            })
        
        return {
            "success": True,
            "core_memories": results,
            "num_found": len(results),
            "persona": persona,
            "fact_type_filter": fact_type
        }
        
    except Exception as e:
        logger.error(f"Error retrieving core memory: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def update_vault(
    source: str,
    entity: str,
    knowledge_type: str,
    content: str,
    relevance_score: float = 0.0
) -> Dict[str, Any]:
    """
    Store full documents/evidence in knowledge vault.
    
    Args:
        source: Knowledge source
        entity: Related entity
        knowledge_type: Type of knowledge
        content: Knowledge content
        relevance_score: Relevance score
    
    Returns:
        Storage result
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge_vault 
            (source, entity, knowledge_type, content, relevance_score, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 
                    COALESCE((SELECT access_count FROM knowledge_vault 
                             WHERE source = ? AND entity = ? AND knowledge_type = ?), 0) + 1)
        """, (source, entity, knowledge_type, content, relevance_score, source, entity, knowledge_type))
        
        db_connection.commit()
        
        return {
            "success": True,
            "vault_id": cursor.lastrowid,
            "source": source,
            "entity": entity,
            "knowledge_type": knowledge_type,
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating knowledge vault: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_memory_stats() -> Dict[str, Any]:
    """
    Get memory usage and performance statistics.
    
    Returns:
        Memory statistics
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        stats = {}
        
        # Get counts for each memory type
        for table in ["core_memory", "episodic_memory", "knowledge_vault"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]
        
        # Get recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM episodic_memory 
            WHERE created_at > datetime('now', '-24 hours')
        """)
        stats["recent_episodic_24h"] = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        stats["database_size_bytes"] = cursor.fetchone()[0]
        
        stats.update({
            "memory_architecture": "MIRIX-inspired",
            "components": ["core", "episodic", "vault"],
            "stats_generated_at": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    mcp.run()
