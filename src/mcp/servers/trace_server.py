#!/usr/bin/env python3
"""
Trace MCP Server - FastMCP Implementation
=========================================

FastMCP-based server for observability and audit logging.
Critical for clinical RAG compliance and system debugging.
"""

import asyncio
import json
import logging
import sqlite3
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

# Server context for managing trace database
@asynccontextmanager
async def trace_lifespan(server: FastMCP):
    """Manage Trace server lifecycle and database."""
    logger.info("Initializing Trace MCP Server...")
    
    try:
        # Initialize SQLite database
        db_path = project_root / "data" / "processed" / "traces.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        db_connection = sqlite3.connect(str(db_path))
        db_connection.row_factory = sqlite3.Row
        
        # Create trace tables
        await create_trace_tables(db_connection)
        
        yield {"db_connection": db_connection}
    except Exception as e:
        logger.error(f"Failed to initialize Trace server: {e}")
        raise
    finally:
        if 'db_connection' in locals():
            db_connection.close()
        logger.info("Shutting down Trace MCP Server...")

async def create_trace_tables(db_connection):
    """Create trace and audit tables"""
    cursor = db_connection.cursor()
    
    # Traces table for span tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            parent_span_id TEXT,
            operation_name TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_ms REAL,
            status TEXT DEFAULT 'running',
            tags TEXT,
            logs TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes separately
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON traces(trace_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_span_id ON traces(span_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_start_time ON traces(start_time)")
    
    # Events table for discrete events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            level TEXT DEFAULT 'info'
        )
    """)
    
    # Create indexes separately
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_trace_id ON events(trace_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
    
    db_connection.commit()

# Create FastMCP server
mcp = FastMCP("Trace Server", lifespan=trace_lifespan)

@mcp.tool()
def start_span(
    trace_id: str,
    span_id: str,
    operation_name: str,
    parent_span_id: str = None,
    tags: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Begin operation tracing with span.
    
    Args:
        trace_id: Trace ID
        span_id: Span ID
        operation_name: Operation name
        parent_span_id: Parent span ID
        tags: Span tags
    
    Returns:
        Span start result
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        cursor.execute("""
            INSERT INTO traces 
            (trace_id, span_id, parent_span_id, operation_name, start_time, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (trace_id, span_id, parent_span_id, operation_name, 
              datetime.now().isoformat(), json.dumps(tags or {})))
        
        db_connection.commit()
        
        return {
            "success": True,
            "trace_id": trace_id,
            "span_id": span_id,
            "operation_name": operation_name,
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting span: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def end_span(
    trace_id: str,
    span_id: str,
    status: str = "success",
    logs: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete span with results.
    
    Args:
        trace_id: Trace ID
        span_id: Span ID
        status: Span status
        logs: Span logs
    
    Returns:
        Span end result
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        # Get start time to calculate duration
        cursor.execute("""
            SELECT start_time FROM traces 
            WHERE trace_id = ? AND span_id = ?
        """, (trace_id, span_id))
        
        row = cursor.fetchone()
        if not row:
            return {"success": False, "error": f"Span {span_id} not found"}
        
        start_time = datetime.fromisoformat(row["start_time"])
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Update span
        cursor.execute("""
            UPDATE traces 
            SET end_time = ?, duration_ms = ?, status = ?, logs = ?
            WHERE trace_id = ? AND span_id = ?
        """, (end_time.isoformat(), duration_ms, status, 
              json.dumps(logs or []), trace_id, span_id))
        
        db_connection.commit()
        
        return {
            "success": True,
            "trace_id": trace_id,
            "span_id": span_id,
            "duration_ms": duration_ms,
            "status": status,
            "ended_at": end_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error ending span: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def log_event(
    trace_id: str,
    event_type: str,
    event_data: Dict[str, Any],
    source: str = None,
    level: str = "info"
) -> Dict[str, Any]:
    """
    Log discrete events.
    
    Args:
        trace_id: Trace ID
        event_type: Event type
        event_data: Event data
        source: Event source
        level: Log level
    
    Returns:
        Event log result
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        cursor.execute("""
            INSERT INTO events 
            (trace_id, event_type, event_data, source, level)
            VALUES (?, ?, ?, ?, ?)
        """, (trace_id, event_type, json.dumps(event_data), source, level))
        
        db_connection.commit()
        
        return {
            "success": True,
            "event_id": cursor.lastrowid,
            "trace_id": trace_id,
            "event_type": event_type,
            "logged_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error logging event: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_trace_summary(
    trace_id: str
) -> Dict[str, Any]:
    """
    Retrieve full trace summary.
    
    Args:
        trace_id: Trace ID to retrieve
    
    Returns:
        Complete trace summary
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        # Get all spans for trace
        cursor.execute("""
            SELECT span_id, parent_span_id, operation_name, start_time, 
                   end_time, duration_ms, status, tags, logs
            FROM traces 
            WHERE trace_id = ?
            ORDER BY start_time
        """, (trace_id,))
        
        spans = []
        for row in cursor.fetchall():
            spans.append({
                "span_id": row["span_id"],
                "parent_span_id": row["parent_span_id"],
                "operation_name": row["operation_name"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "duration_ms": row["duration_ms"],
                "status": row["status"],
                "tags": json.loads(row["tags"] or "{}"),
                "logs": json.loads(row["logs"] or "[]")
            })
        
        # Get all events for trace
        cursor.execute("""
            SELECT event_type, event_data, timestamp, source, level
            FROM events 
            WHERE trace_id = ?
            ORDER BY timestamp
        """, (trace_id,))
        
        events = []
        for row in cursor.fetchall():
            events.append({
                "event_type": row["event_type"],
                "event_data": json.loads(row["event_data"]),
                "timestamp": row["timestamp"],
                "source": row["source"],
                "level": row["level"]
            })
        
        return {
            "success": True,
            "trace_id": trace_id,
            "spans": spans,
            "events": events,
            "total_spans": len(spans),
            "total_events": len(events),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trace summary: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
def get_trace_stats() -> Dict[str, Any]:
    """
    Get trace system statistics.
    
    Returns:
        Trace statistics
    """
    try:
        db_connection = mcp.dependencies["db_connection"]
        cursor = db_connection.cursor()
        
        stats = {}
        
        # Get trace counts
        cursor.execute("SELECT COUNT(DISTINCT trace_id) FROM traces")
        stats["unique_traces"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM traces")
        stats["total_spans"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM events")
        stats["total_events"] = cursor.fetchone()[0]
        
        # Get recent activity
        cursor.execute("""
            SELECT COUNT(DISTINCT trace_id) FROM traces 
            WHERE created_at > datetime('now', '-24 hours')
        """)
        stats["recent_traces_24h"] = cursor.fetchone()[0]
        
        stats.update({
            "stats_generated_at": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting trace stats: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    mcp.run()
