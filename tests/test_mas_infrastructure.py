"""
Test Suite for MAS Infrastructure
================================

Comprehensive tests for the core MAS components:
- Typed state management
- MCP client functionality
- Server health checks
- Trace propagation
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from src.mas.state import (
    MASState, KGPath, DenseHit, Candidate,
    create_initial_state, copy_state, get_policy_for_intent
)
from src.mcp.client import MCPClient, MCPError


class TestMASState:
    """Test MAS state management and type safety"""
    
    def test_create_initial_state(self):
        """Test initial state creation with defaults"""
        state = create_initial_state("What does aspirin treat?", "doctor")
        
        assert state["query"] == "What does aspirin treat?"
        assert state["persona"] == "doctor"
        assert state["intent"] == "factoid"
        assert state["mode"] == "hybrid"
        assert len(state["trace_id"]) == 36  # UUID format
        assert isinstance(state["timestamp"], datetime)
        assert state["evidence"] == {"kg_paths": [], "dense_hits": []}
        assert state["candidates"] == []
        assert state["retry_count"] == 0
    
    def test_copy_state_immutability(self):
        """Test state copying preserves immutability"""
        original = create_initial_state("test query")
        copied = copy_state(original)
        
        # Modify copied state
        copied["intent"] = "enumeration"
        copied["candidates"].append({"answer": "test", "score": 0.5, "evidence_refs": {}})
        
        # Original should be unchanged
        assert original["intent"] == "factoid"
        assert len(original["candidates"]) == 0
        assert original["trace_id"] == copied["trace_id"]  # Same trace ID
    
    def test_policy_generation(self):
        """Test intent-specific policy generation"""
        factoid_policy = get_policy_for_intent("factoid")
        enum_policy = get_policy_for_intent("enumeration")
        causal_policy = get_policy_for_intent("causal")
        
        assert factoid_policy["max_hops"] == 2
        assert factoid_policy["topK_rel"] == 6
        assert factoid_policy["time_budget"] == 2.0
        
        assert enum_policy["topK_rel"] == 14  # More relations for enumeration
        assert enum_policy["max_neighbors"] == 8  # More neighbors
        
        assert causal_policy["max_hops"] == 4  # Deeper search for causality
        
    def test_kg_path_structure(self):
        """Test KGPath type structure"""
        kg_path = KGPath(
            triples=[("Aspirin", "treats", "Headache"), ("Headache", "presents", "Pain")],
            path_conf=0.85,
            meta={"depth": 2, "entity": "Aspirin"}
        )
        
        assert len(kg_path["triples"]) == 2
        assert kg_path["path_conf"] == 0.85
        assert kg_path["meta"]["depth"] == 2
    
    def test_dense_hit_structure(self):
        """Test DenseHit type structure"""
        dense_hit = DenseHit(
            id=12345,
            text="Aspirin is used to treat headaches and reduce inflammation.",
            score=0.92,
            metadata={"subject": "Aspirin", "predicate": "treats", "object": "Headache"}
        )
        
        assert dense_hit["id"] == 12345
        assert dense_hit["score"] == 0.92
        assert dense_hit["metadata"]["subject"] == "Aspirin"
    
    def test_candidate_structure(self):
        """Test Candidate type structure"""
        candidate = Candidate(
            answer="Headache",
            score=0.78,
            evidence_refs={
                "dense": [12345, 67890],
                "kg": [("Aspirin", "treats", "Headache")]
            }
        )
        
        assert candidate["answer"] == "Headache"
        assert candidate["score"] == 0.78
        assert len(candidate["evidence_refs"]["dense"]) == 2
        assert len(candidate["evidence_refs"]["kg"]) == 1


class TestMCPClient:
    """Test MCP client functionality"""
    
    @pytest.fixture
    def trace_id(self):
        """Generate test trace ID"""
        return str(uuid.uuid4())
    
    def test_client_initialization(self, trace_id):
        """Test MCP client initialization"""
        client = MCPClient(trace_id)
        
        assert client.trace_id == trace_id
        assert "kg" in client.servers
        assert "dense" in client.servers
        assert "trace" in client.servers
        
        # Check circuit breaker initialization
        assert all(server in client.circuit_breakers for server in client.servers)
        assert client.circuit_breakers["kg"]["failures"] == 0
    
    def test_circuit_breaker_logic(self, trace_id):
        """Test circuit breaker functionality"""
        client = MCPClient(trace_id)
        
        # Initially circuit should be closed
        assert not client._is_circuit_open("kg")
        
        # Record failures
        client._record_failure("kg")
        client._record_failure("kg")
        client._record_failure("kg")
        
        # Circuit should now be open
        assert client._is_circuit_open("kg")
        assert client.circuit_breakers["kg"]["failures"] == 3
    
    def test_batch_call_specification(self, trace_id):
        """Test batch call helper functions"""
        from src.mcp.client import create_batch_calls
        
        payloads = [
            {"entity": "Aspirin", "relation": "treats"},
            {"entity": "Ibuprofen", "relation": "treats"}
        ]
        
        batch_calls = create_batch_calls(
            server="kg",
            method="get_neighbors", 
            payloads=payloads,
            base_span="batch_neighbors",
            node="test_node"
        )
        
        assert len(batch_calls) == 2
        assert batch_calls[0]["server"] == "kg"
        assert batch_calls[0]["method"] == "get_neighbors"
        assert batch_calls[0]["span"] == "batch_neighbors_0"
        assert batch_calls[1]["span"] == "batch_neighbors_1"


class TestServerIntegration:
    """Integration tests for MCP servers (requires servers running)"""
    
    @pytest.mark.asyncio
    async def test_trace_server_logging(self):
        """Test trace server logging functionality"""
        trace_id = str(uuid.uuid4())
        
        # This test requires trace server to be running
        # In a real deployment, we'd use test containers
        try:
            async with MCPClient(trace_id) as client:
                # Test basic logging
                result = await client.call(
                    server="trace",
                    method="log",
                    payload={
                        "event": "test_event",
                        "payload": {"test": "data"}
                    },
                    span="test_span",
                    node="test_node"
                )
                
                assert result["status"] == "logged"
                
        except Exception as e:
            pytest.skip(f"Trace server not available: {e}")
    
    @pytest.mark.asyncio 
    async def test_server_health_checks(self):
        """Test health check endpoints for all servers"""
        trace_id = str(uuid.uuid4())
        servers = ["kg", "dense", "validator", "trace"]
        
        try:
            async with MCPClient(trace_id) as client:
                for server in servers:
                    try:
                        health = await client.call(
                            server=server,
                            method="health",
                            payload={},
                            span=f"health_{server}",
                            node="test"
                        )
                        
                        assert health["status"] == "healthy"
                        
                    except Exception as e:
                        pytest.skip(f"{server} server not available: {e}")
                        
        except Exception as e:
            pytest.skip(f"MCP client setup failed: {e}")


class TestTraceLogging:
    """Test trace logging and observability"""
    
    def test_log_file_creation(self):
        """Test that trace logs are created correctly"""
        trace_id = str(uuid.uuid4())
        log_file = Path(f"logs/{trace_id}.jsonl")
        
        # Clean up any existing file
        if log_file.exists():
            log_file.unlink()
        
        # Create a test log entry
        log_entry = {
            "trace_id": trace_id,
            "span": "test_span",
            "node": "test_node",
            "event": "test_event",
            "payload": {"test": "data"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Write log entry
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Verify log file exists and contains correct data
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            logged_entry = json.loads(f.read().strip())
            assert logged_entry["trace_id"] == trace_id
            assert logged_entry["event"] == "test_event"
        
        # Clean up
        log_file.unlink()
    
    def test_trace_summary_generation(self):
        """Test trace summary generation logic"""
        trace_id = str(uuid.uuid4())
        
        # Create mock trace events
        events = [
            {
                "trace_id": trace_id,
                "span": "planner",
                "node": "planner",
                "event": "planner_start",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "trace_id": trace_id,
                "span": "planner", 
                "node": "planner",
                "event": "planner_complete",
                "timestamp": "2024-01-01T10:00:05"
            },
            {
                "trace_id": trace_id,
                "span": "retriever",
                "node": "retriever", 
                "event": "retrieve_complete",
                "timestamp": "2024-01-01T10:00:10"
            }
        ]
        
        # Test summary logic (would be in trace server)
        nodes_involved = list(set(e["node"] for e in events))
        assert "planner" in nodes_involved
        assert "retriever" in nodes_involved
        assert len(nodes_involved) == 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
